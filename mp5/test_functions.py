import copy
import random
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from gradescope_utils.autograder_utils.decorators import weight, visibility, number

import flow_unet
from flow import FlowMatching
from flow_unet import TimeEmbedding, ResBlock, FlowUNet, ImageFlowMatching, update_ema
from flow_editing import add_noise, cfg_velocity, sdedit, anagram_velocity, rotate_180

SEED = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_error(obj, ref, pred):
    msg = f'{obj} does not match the expected values!\n'
    ae = torch.abs(ref - pred).view(-1)
    msg += f'Total absolute error: {ae.sum().item():.4f}\n'
    msg += f'Mean absolute error: {ae.mean().item():.4f}\n'
    msg += f'Max absolute error: {ae.max().item():.4f}\n'
    return msg


# Q2 fixtures

# Expected outputs of the Q2.1-Q2.3 tests, computed with the reference solution
REFERENCE = 'test_data/reference.pt'


def load_reference(name):
    return torch.load(REFERENCE, weights_only=True)[name]


def standin_resblock(block, x, emb):
    """Stand-in for ResBlock.forward in the Q2.3 test (NOT the block of Q2.2), so that the test only checks the wiring."""
    scale = block.emb_proj(emb)[:, :block.conv1.out_channels, None, None]
    return block.skip(x) + torch.tanh(block.conv1(x)) * (1 + scale)


def standin_get_velocity(fm, x, t, labels, scale=1.0):
    """Stand-in for ImageFlowMatching.get_velocity (Q2.6) without guidance, so that the Q2.4 test does not need Q2.6."""
    return fm.velocity_net(x, t, labels)


def standin_embed(net, t, labels):
    """Stand-in for FlowUNet.embed in the Q2.3 test (NOT the embedding of Q2.2)."""
    return net.class_emb(labels) + t[:, None]


class ToyUNet(nn.Module):
    """Stand-in for a FlowUNet that records the arguments of every call: u = (label + 1) * x + t * PATTERN."""
    null_label = 3

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, labels):
        self.calls.append((x.clone(), t.clone() if torch.is_tensor(t) else t, labels))
        t = torch.as_tensor(t, dtype=x.dtype).reshape(-1, 1, 1, 1)
        labels = torch.as_tensor(labels, dtype=x.dtype).reshape(-1, 1, 1, 1)
        pattern = torch.arange(x[0].numel(), dtype=x.dtype).reshape(x.shape[1:]) / x[0].numel()
        return (labels + 1) * x + t * pattern


# Q3 fixtures

SHAPE = (2, 3, 8, 8)
# Times are deliberately not evenly spaced, so that the step size has to be computed from ts
TS = [0.1, 0.4, 0.75, 0.95, 1.0]
# Fixed spatial pattern that is not symmetric under a 180 degree rotation
PATTERN = torch.arange(SHAPE[-2] * SHAPE[-1], dtype=torch.float32).reshape(1, 1, *SHAPE[-2:]) / 64


def toy_model(x, t, emb):
    """Toy text-conditioned velocity model whose output is not rotation-equivariant. `emb` is a float here."""
    return emb * x + t * PATTERN + emb


class RecordingModel:
    """Wraps a model or a velocity field and records the arguments of every call."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, x, t, *args):
        self.calls.append((x.clone(), float(t), *args))
        return self.fn(x, t, *args)


ANAGRAM_ARGS = (0.6, 2.0, -1.5, 0.5, 3.0)  # t, emb_1, emb_2, null_emb, scale


def toy_field(x, t):
    """Toy velocity field for the SDEdit test."""
    return -x + t * PATTERN


class TestClass(unittest.TestCase):

    ########################################
    # Q1: Flow Matching                    #
    ########################################

    @weight(2.0)
    @number('1.1')
    @visibility('visible')
    def test_flow_interpolate(self):
        flow = FlowMatching(nn.Identity())
        seed_all(SEED)
        n_sample = 10000
        z = torch.randn(n_sample, 1000)
        eps, t, x_t = flow.interpolate(z)

        # Shapes
        self.assertTrue(
            eps.shape == z.shape,
            f'Shape of eps is {eps.shape}, which does not match expected shape {z.shape}!'
        )
        self.assertTrue(
            tuple(t.shape) == (n_sample, 1),
            f'Shape of t is {t.shape}, which does not match expected shape ({n_sample}, 1)!'
        )
        self.assertTrue(
            x_t.shape == z.shape,
            f'Shape of x_t is {x_t.shape}, which does not match expected shape {z.shape}!'
        )

        # Noise eps ~ N(0, I), independent of the data
        eps_mu, eps_std = eps.mean(), eps.std()
        self.assertTrue(
            torch.allclose(eps_mu, torch.FloatTensor([0.]), atol=1e-2),
            f'Mean of eps should be 0, but got {eps_mu.item():.4f}!'
        )
        self.assertTrue(
            torch.allclose(eps_std, torch.FloatTensor([1.]), atol=1e-2),
            f'Standard deviation of eps should be 1, but got {eps_std.item():.4f}!'
        )
        corr = (eps * z).mean()
        self.assertTrue(
            torch.allclose(corr, torch.FloatTensor([0.]), atol=1e-2),
            f'eps should be independent of z, but E[eps * z] = {corr.item():.4f}!'
        )

        # t ~ U[0, 1]
        self.assertTrue(
            t.min() >= 0 and t.max() <= 1,
            f't should lie in [0, 1], but got range [{t.min().item():.4f}, {t.max().item():.4f}]!'
        )
        t_mu, t_var = t.mean(), t.var()
        self.assertTrue(
            torch.allclose(t_mu, torch.FloatTensor([0.5]), atol=1e-2),
            f'Mean of t should be 0.5, but got {t_mu.item():.4f}!'
        )
        self.assertTrue(
            torch.allclose(t_var, torch.FloatTensor([1 / 12]), atol=1e-2),
            f'Variance of t should be {1 / 12:.4f} (uniform on [0, 1]), but got {t_var.item():.4f}!'
        )

        # x_t lies on the straight line from eps (t = 0) to z (t = 1)
        x_t_gt = t * z + (1 - t) * eps
        self.assertTrue(
            torch.allclose(x_t, x_t_gt, atol=1e-5),
            log_error('Interpolated points x_t', x_t_gt, x_t)
        )

        # Q2 reuses interpolate for images, with its own time distribution
        class ConstantTimes(FlowMatching):
            def sample_times(self, n, device=None, dtype=None):
                return torch.full((n,), 0.3, device=device, dtype=dtype)

        z = torch.randn(8, 3, 4, 4)
        eps, t, x_t = ConstantTimes(nn.Identity()).interpolate(z)
        self.assertTrue(
            tuple(t.shape) == (8, 1, 1, 1),
            f'For data of shape (8, 3, 4, 4), t should have shape (8, 1, 1, 1), but got {tuple(t.shape)}!'
        )
        self.assertTrue(
            torch.allclose(t, torch.full_like(t, 0.3)),
            'Draw the times with self.sample_times, so that subclasses (Q2) can change their distribution!'
        )
        self.assertTrue(
            eps.shape == z.shape and torch.allclose(x_t, 0.3 * z + 0.7 * eps, atol=1e-5),
            'interpolate should work for data of any shape (N, ...)!'
        )

    @weight(2.0)
    @number('1.2')
    @visibility('visible')
    def test_flow_loss(self):
        # velocity net u_t(x) = t * x
        flow = FlowMatching(lambda xt: xt[:, -1:] * xt[:, :-1])
        seed_all(SEED)
        z = torch.zeros(10000, 1000)
        loss = flow.get_loss(z)
        self.assertTrue(
            loss.dim() == 0,
            f'loss should be a scalar, but got shape {loss.shape}!'
        )
        # x_t = (1 - t) * eps, target = z - eps = -eps
        # loss = E[(t (1 - t) + 1)^2] * E[eps^2] = 41 / 30
        gt = torch.FloatTensor([41 / 30])
        self.assertTrue(
            torch.isclose(loss, gt, rtol=1e-2).item(),
            f'loss should be {gt.item():.4f}, but got {loss.item():.4f}!'
        )

        # Q2 reuses get_loss with extra arguments for get_velocity
        class Conditional(FlowMatching):
            def get_velocity(self, x, t, scale):
                return scale * torch.ones_like(x)

        loss = Conditional(None).get_loss(torch.zeros(4, 3, 2, 2), 2.0)
        self.assertTrue(
            torch.isfinite(loss) and loss.item() > 0,
            'get_loss should pass its extra arguments *cond on to self.get_velocity(x_t, t, *cond)!'
        )

    @weight(2.0)
    @number('1.3')
    @visibility('visible')
    def test_flow_sample(self):
        n_sample = 10000
        dim = 256
        n_steps = 4

        # velocity net u_t(x) = t, independent of x
        flow = FlowMatching(lambda xt: xt[:, -1:].expand(-1, xt.size(1) - 1))
        seed_all(SEED)
        x_init = torch.randn(n_sample, dim)
        traj = flow.sample(x_init, n_steps=n_steps)
        self.assertTrue(
            tuple(traj.shape) == (n_steps + 1, n_sample, dim),
            f'Shape of trajectory is {traj.shape}, '
            f'which does not match expected shape ({n_steps + 1}, {n_sample}, {dim})!'
        )
        self.assertTrue(torch.equal(traj[0], x_init), 'The trajectory should start at the initial noise x!')
        # Euler from t = 0 to t = 1 with u evaluated at t_i = i / n_steps:
        # x_k - x_init = sum_{i<k} h * t_i = sum_{i<k} (i / n_steps) / n_steps
        disp = (traj - x_init).mean(dim=(1, 2))
        disp_gt = torch.FloatTensor([sum((i / n_steps) / n_steps for i in range(k)) for k in range(n_steps + 1)])
        self.assertTrue(
            torch.allclose(disp, disp_gt, atol=1e-5),
            f'Displacement along the trajectory should be {disp_gt.tolist()}, but got {disp.tolist()}!\n'
            f'Make sure you integrate from t = 0 to t = 1 and evaluate the velocity at the current time t.'
        )

        # velocity net u_t(x) = -x: each step multiplies x by (1 - 1 / n_steps)
        flow = FlowMatching(lambda xt: -xt[:, :-1])
        traj = flow.sample(x_init, n_steps=n_steps)
        x_gt = (1 - 1 / n_steps) ** n_steps * x_init
        self.assertTrue(
            torch.allclose(traj[-1], x_gt, atol=1e-4),
            log_error('Final samples', x_gt, traj[-1])
        )

        # Q2 reuses sample for images, with extra arguments for get_velocity
        class Conditional(FlowMatching):
            def get_velocity(self, x, t, scale):
                return scale * torch.ones_like(x)

        traj = Conditional(None).sample(torch.zeros(2, 3, 4, 4), 2.0, n_steps=n_steps)
        self.assertTrue(
            tuple(traj.shape) == (n_steps + 1, 2, 3, 4, 4) and torch.allclose(traj[-1], torch.full_like(traj[-1], 2.)),
            'sample should work for data of any shape and pass *cond on to self.get_velocity(x, t, *cond)!'
        )

    ########################################
    # Q2: Flow Matching on Images          #
    ########################################

    @weight(1.0)
    @number('2.1')
    @visibility('visible')
    def test_time_embedding(self):
        ref = load_reference('sinusoidal')
        dim, t = ref['dim'], ref['t']
        emb = TimeEmbedding(dim, 32)
        feats = emb.sinusoidal(t)
        self.assertTrue(
            feats.shape == (len(t), dim),
            f'Shape of the features is {tuple(feats.shape)}, which does not match expected shape {(len(t), dim)}!'
        )
        self.assertTrue(feats.dtype == torch.float32, f'Features should be float32, but are {feats.dtype}!')
        feats_gt = ref['feats']
        self.assertTrue(
            torch.allclose(feats, feats_gt, atol=1e-3),
            log_error('Sinusoidal features', feats_gt, feats) +
            'Expected [sin(1000 t f_0), ..., sin(1000 t f_{h-1}), cos(1000 t f_0), ..., cos(1000 t f_{h-1})].'
        )
        # Works for any batch size and for times in a different dtype
        feats = emb.sinusoidal(torch.full((3,), 0.3, dtype=torch.float64))
        self.assertTrue(
            feats.shape == (3, dim) and torch.allclose(feats, ref['feats_03'], atol=1e-3),
            'Sinusoidal features are wrong for float64 times!'
        )

    @weight(2.0)
    @number('2.2')
    @visibility('visible')
    def test_resblock_film(self):
        for case in load_reference('resblock'):
            in_ch, out_ch, x, emb = case['in_ch'], case['out_ch'], case['x'], case['emb']
            block = ResBlock(in_ch, out_ch, emb_dim=8).eval()
            block.load_state_dict(case['state'])
            with torch.no_grad():
                out = block(x, emb)
            self.assertTrue(
                out.shape == (3, out_ch, 6, 6),
                f'Shape of output is {tuple(out.shape)}, which does not match expected shape {(3, out_ch, 6, 6)}!'
            )
            out_gt = case['out']
            self.assertTrue(
                torch.allclose(out, out_gt, atol=1e-4),
                log_error(f'ResBlock output ({in_ch} -> {out_ch} channels)', out_gt, out) +
                'Check the order of the operations, and that the scale comes before the shift in emb_proj\'s output.'
            )

            # With zero scale and shift, FiLM does nothing
            nn.init.zeros_(block.emb_proj.weight)
            nn.init.zeros_(block.emb_proj.bias)
            with torch.no_grad():
                out = block(x, emb)
            out_gt = case['out_zero']
            self.assertTrue(
                torch.allclose(out, out_gt, atol=1e-4),
                log_error('ResBlock output with zero scale and shift', out_gt, out) +
                'With scale = shift = 0, the block should not change the normalized features (multiply by 1 + scale).'
            )

        # FlowUNet.embed: the sum of the time and class embeddings, including the null label
        net = FlowUNet(num_classes=3, channels=(16, 32)).eval()
        t, labels = torch.tensor([0.1, 0.5, 0.9, 0.3]), torch.tensor([0, 1, 2, net.null_label])
        emb = net.embed(t, labels)
        emb_gt = net.time_emb(t) + net.class_emb(labels)
        self.assertTrue(
            emb.shape == emb_gt.shape and torch.allclose(emb, emb_gt, atol=1e-5),
            'FlowUNet.embed should return time_emb(t) + class_emb(labels)!'
        )

    @weight(2.0)
    @number('2.3')
    @visibility('visible')
    def test_unet_forward(self):
        ref = load_reference('unet')
        # Stand-ins replace your Q2.1 and Q2.2 code, so that this test only checks how the blocks are connected
        # (patch the classes that flow_unet has now: the notebooks' autoreload may have replaced them since the import)
        with mock.patch.object(flow_unet.ResBlock, 'forward', standin_resblock), \
                mock.patch.object(flow_unet.FlowUNet, 'embed', standin_embed):
            net = flow_unet.FlowUNet(num_classes=3, channels=ref['channels']).eval()
            net.load_state_dict(ref['state'])
            x, t, labels = ref['x'], ref['t'], ref['labels']
            with torch.no_grad():
                out = net(x, t, labels)
                self.assertTrue(
                    out.shape == x.shape,
                    f'Shape of output is {tuple(out.shape)}, which does not match expected shape {tuple(x.shape)}!'
                )
                out_gt = ref['out']
                self.assertTrue(
                    torch.allclose(out, out_gt, atol=1e-4),
                    log_error('FlowUNet output', out_gt, out) +
                    'Check that you compute the embedding with self.embed, the order of the blocks, where you '
                    'downsample and upsample, which encoder output you concatenate at each level, and the order of '
                    'the concatenation ([decoder features, skip]).'
                )
                # A float time and an int label are used for the whole batch
                out = net(x, 0.3, 2)
                self.assertTrue(torch.allclose(out, ref['out_scalar'], atol=1e-4),
                                'FlowUNet output is wrong for a float t and an int label!')

    @weight(2.0)
    @number('2.4')
    @visibility('visible')
    # A stand-in replaces your Q2.6 get_velocity, so that this test only checks get_loss
    @mock.patch.object(ImageFlowMatching, 'get_velocity', standin_get_velocity)
    def test_image_flow_loss(self):
        seed_all(SEED)
        batch_size, p_uncond = 4096, 0.25
        z = torch.randn(batch_size, 3, 4, 4, dtype=torch.float64)
        labels = torch.randint(0, 3, (batch_size,))

        net = ToyUNet()
        loss = ImageFlowMatching(net, p_uncond).get_loss(z, labels)
        self.assertTrue(loss.dim() == 0, f'loss should be a scalar, but got shape {loss.shape}!')
        self.assertTrue(
            len(net.calls) == 1,
            f'The network should be called once per loss evaluation, but was called {len(net.calls)} times!'
        )
        x_t, t, labels_used = net.calls[0]

        # Your Q1 interpolate draws one logit-normal time per image with self.sample_times
        self.assertTrue(
            torch.is_tensor(t) and t.numel() == batch_size,
            f'The network should receive one time per image, but got {t}! Check your Q1 interpolate and get_loss.'
        )
        t = t.reshape(-1, 1, 1, 1).double()
        logit_t = torch.log(t / (1 - t))
        self.assertTrue(
            abs(logit_t.mean().item()) < 0.1 and abs(logit_t.std().item() - 1) < 0.1,
            f'logit(t) should have mean 0 and standard deviation 1, but got {logit_t.mean().item():.3f} and '
            f'{logit_t.std().item():.3f}! Your Q1 interpolate should draw the times with self.sample_times.'
        )

        # x_t = t * z + (1 - t) * eps for some eps ~ N(0, I) that is independent of z
        eps = (x_t.double() - t * z) / (1 - t)
        self.assertTrue(
            abs(eps.mean().item()) < 1e-2 and abs(eps.std().item() - 1) < 1e-2,
            f'The noise in x_t should have mean 0 and standard deviation 1, but got {eps.mean().item():.4f} and '
            f'{eps.std().item():.4f}! Make sure you reuse your Q1 get_loss.'
        )

        # The loss is your Q1 loss, with the target z - eps and the labels after dropout
        net.calls.clear()
        u = net(x_t.double(), t, labels_used)
        loss_gt = ((u - (z - eps)) ** 2).mean()
        self.assertTrue(
            torch.isclose(loss.double(), loss_gt, rtol=1e-3).item(),
            f'loss should be {loss_gt.item():.4f}, but got {loss.item():.4f}!'
        )

        # Each label is replaced by the null label independently with probability p_uncond
        labels_used = torch.as_tensor(labels_used).reshape(-1)
        dropped = labels_used == ToyUNet.null_label
        self.assertTrue(
            labels_used.shape == labels.shape and torch.equal(labels_used[~dropped], labels[~dropped]),
            'The network should receive the label of each image, or the null label for dropped images!'
        )
        for p in [0.0, p_uncond, 1.0]:
            net = ToyUNet()
            ImageFlowMatching(net, p).get_loss(z, labels)
            frac = (torch.as_tensor(net.calls[0][2]).reshape(-1) == ToyUNet.null_label).double().mean().item()
            self.assertTrue(
                abs(frac - p) < 0.03,
                f'With p_uncond = {p}, a fraction {frac:.3f} of the labels was replaced by the null label!'
            )

    @weight(1.0)
    @number('2.5')
    @visibility('visible')
    def test_update_ema(self):
        seed_all(SEED)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        ema = copy.deepcopy(model).requires_grad_(False)
        for p in ema.parameters():
            p.data.normal_()
        before_ema = [p.detach().clone() for p in ema.parameters()]
        before_model = [p.detach().clone() for p in model.parameters()]
        ema_tensors = [p for p in ema.parameters()]

        update_ema(ema, model, 0.9)
        for p_ema, p0, p in zip(ema.parameters(), before_ema, before_model):
            p_gt = 0.9 * p0 + 0.1 * p
            self.assertTrue(
                torch.allclose(p_ema, p_gt, atol=1e-6),
                log_error('EMA parameters after one update with decay 0.9', p_gt, p_ema) +
                'Expected p_ema = decay * p_ema + (1 - decay) * p.'
            )
        self.assertTrue(
            all(a is b for a, b in zip(ema.parameters(), ema_tensors)),
            'update_ema should update the parameters of ema_model in place, not replace them!'
        )
        self.assertTrue(
            all(torch.equal(p, p0) for p, p0 in zip(model.parameters(), before_model)),
            'update_ema should not modify the parameters of the trained model!'
        )

        # decay = 0 copies the model
        update_ema(ema, model, 0.0)
        self.assertTrue(
            all(torch.allclose(p_ema, p) for p_ema, p in zip(ema.parameters(), model.parameters())),
            'With decay = 0, the EMA should be equal to the model!'
        )


    @weight(1.0)
    @number('2.6')
    @visibility('visible')
    def test_image_cfg_velocity(self):
        seed_all(SEED)
        x, t, labels = torch.randn(3, 3, 4, 4), 0.7, torch.tensor([0, 1, 2])
        null = ToyUNet.null_label
        for scale in [0.0, 1.0, 3.5]:
            net = ToyUNet()
            u = ImageFlowMatching(net).get_velocity(x, t, labels, scale)
            u_null = ToyUNet()(x, t, null)
            u_gt = u_null + scale * (ToyUNet()(x, t, labels) - u_null)
            self.assertTrue(
                torch.allclose(u, u_gt, atol=1e-5),
                log_error(f'Guided velocity with scale {scale}', u_gt, u)
            )
            calls = [call[2] for call in net.calls]
            self.assertTrue(
                all(torch.equal(call[0], x) and call[1] == t for call in net.calls),
                'All network calls should use the x and t that were passed to get_velocity!'
            )
            if scale == 1:
                self.assertTrue(len(calls) == 1, 'With scale = 1, call the network only once (with the labels)!')
            else:
                self.assertTrue(
                    len(calls) == 2 and any(torch.is_tensor(c) and torch.equal(c, labels) for c in calls)
                    and any(bool((torch.as_tensor(c) == null).all()) for c in calls),
                    'With guidance, call the network once with the labels and once with the null label!'
                )

    ########################################
    # Q3: Editing with PixNerd             #
    ########################################

    @weight(1.0)
    @number('3.1')
    @visibility('visible')
    def test_sdedit(self):
        ref = load_reference('q3')
        z, eps = ref['z'], ref['eps']

        # The first call should see z noised to ts[0] with the given noise
        field = RecordingModel(toy_field)
        out = sdedit(field, z, eps, TS)
        x_start, t_start = field.calls[0][0], field.calls[0][1]
        x_start_gt = add_noise(z, eps, TS[0])
        self.assertTrue(
            np.isclose(t_start, TS[0]),
            f'The first velocity should be evaluated at t = {TS[0]}, but was evaluated at t = {t_start}!'
        )
        self.assertTrue(
            torch.allclose(x_start, x_start_gt, atol=1e-5),
            log_error(f'Noisy input at t = {TS[0]}', x_start_gt, x_start)
        )
        out_gt = ref['sdedit']
        self.assertTrue(
            torch.allclose(out, out_gt, atol=1e-5),
            log_error('Edited images', out_gt, out)
        )

        # A field that points straight at z returns z unchanged
        out = sdedit(lambda x, t: (z - x) / (1 - t), z, eps, TS)
        self.assertTrue(
            torch.allclose(out, z, atol=1e-5),
            log_error('Edited images for a field that points at z', z, out)
        )

    @weight(2.0)
    @number('3.3')
    @visibility('visible')
    def test_anagram_velocity(self):
        ref = load_reference('q3')
        x = ref['x']
        # emb_1 + emb_2 != 2 * null_emb, otherwise guidance cancels out in the average for this toy model
        t, emb_1, emb_2, null_emb, scale = ANAGRAM_ARGS

        u = anagram_velocity(toy_model, x, t, emb_1, emb_2, null_emb, scale)
        u_gt = ref['anagram']
        self.assertTrue(
            u.shape == x.shape,
            f'Shape of output is {u.shape}, which does not match expected shape {x.shape}!'
        )
        self.assertTrue(
            torch.allclose(u, u_gt, atol=1e-5),
            log_error('Anagram velocity', u_gt, u) +
            'Make sure you rotate x before the second model call, and rotate its velocity back afterwards.'
        )

        # Swapping the prompts and rotating x should give the rotated velocity
        u_swap = anagram_velocity(toy_model, rotate_180(x), t, emb_2, emb_1, null_emb, scale)
        self.assertTrue(
            torch.allclose(u_swap, rotate_180(u), atol=1e-5),
            log_error('Anagram velocity with swapped prompts on the rotated image', rotate_180(u), u_swap)
        )


if __name__ == '__main__':
    unittest.main()
