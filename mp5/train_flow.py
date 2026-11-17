"""
Q2 training script (provided, you do not need to modify it): trains a FlowUNet on AFHQ with your code from Q1 and
Q2.1-Q2.6, and saves checkpoints, sample grids and its log (log.txt) to --run_dir. Every run starts a new model. If
--run_dir already has a run, the new run goes to the first free folder of <run_dir>_2, <run_dir>_3, ... (it prints
which), so no run is ever overwritten. To continue a run that stopped early, or to train it for longer (with a larger
--max_steps), add --resume: it continues the run in --run_dir with the hyperparameters it was started with.

    python train_flow.py --run_dir runs/lr3e-4 --lr 3e-4     # locally, or from the Q2 notebook
    python train_flow.py --run_dir runs/lr3e-4 --resume --max_steps 20000   # train the same run for longer
    sbatch --export=ALL,RUN_DIR=runs/lr3e-4 train_flow.sbatch --lr 3e-4  # on Delta AI

The defaults train, but are not tuned: Q2.7 compares runs with different hyperparameters (see the notebook).
"""
import argparse
import copy
import math
import os
import time

import torch
from torchvision.utils import save_image

from flow_unet import FlowUNet, ImageFlowMatching, update_ema
from q2_helpers import DATA_PATH, RUN_DIR, get_device, load_afhq, randn, to_float, val_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=DATA_PATH)
parser.add_argument('--run_dir', default=RUN_DIR)
parser.add_argument('--max_minutes', type=float, default=40, help='stop (and save) after this many minutes in this run')
parser.add_argument('--max_steps', type=int, default=None,
                    help='stop after this many steps in total (default: 10000, or with --resume, that of the run)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4, help='peak learning rate of Adam')
parser.add_argument('--lr_schedule', default='constant', choices=['constant', 'cosine'],
                    help='constant, or cosine decay from lr to 0 at max_steps')
parser.add_argument('--warmup_steps', type=int, default=0, help='increase the learning rate linearly from 0 to lr')
parser.add_argument('--grad_clip', type=float, default=0, help='clip the gradient norm to this value (0: no clipping)')
parser.add_argument('--ema_decay', type=float, default=0.99)
parser.add_argument('--channels', default='32,64,128,256', help='FlowUNet channels at each resolution')
parser.add_argument('--p_uncond', type=float, default=0.1, help='probability of dropping the class label')
parser.add_argument('--sample_every', type=int, default=1000,
                    help='evaluate, save samples and save a checkpoint every this many steps')
parser.add_argument('--resume', action='store_true',
                    help='continue the run in --run_dir (with its own hyperparameters) instead of starting a new one')
args = parser.parse_args()

# The hyperparameters that define a run: --resume always continues a run with the values it was started with
RUN_KEYS = ['batch_size', 'lr', 'lr_schedule', 'warmup_steps', 'grad_clip', 'ema_decay', 'channels', 'p_uncond']


def has_run(run_dir):
    return any(os.path.exists(os.path.join(run_dir, f)) for f in ['ckpt.pt', 'log.txt'])


ckpt = None
if args.resume:
    ckpt_path = os.path.join(args.run_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise SystemExit(f'--resume: {ckpt_path} does not exist (a run saves its first checkpoint after '
                         f'--sample_every steps)')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for k in RUN_KEYS:
        setattr(args, k, ckpt['args'][k])
    args.max_steps = args.max_steps or ckpt['args']['max_steps']
else:
    args.max_steps = args.max_steps or 10000
    run_dir, k = args.run_dir, 1
    while has_run(run_dir):
        k += 1
        run_dir = f'{args.run_dir}_{k}'
    note = f'{args.run_dir} already has a run, so this new run uses {run_dir}' if run_dir != args.run_dir else None
    args.run_dir = run_dir
ckpt_path = os.path.join(args.run_dir, 'ckpt.pt')
os.makedirs(os.path.join(args.run_dir, 'samples'), exist_ok=True)
log_file = open(os.path.join(args.run_dir, 'log.txt'), 'a')


def log(message):
    # print, and append to <run_dir>/log.txt
    print(message, flush=True)
    print(message, file=log_file, flush=True)


if args.resume:
    log(f'Continuing {args.run_dir} with its own hyperparameters: '
        + ', '.join(f'{k} {getattr(args, k)}' for k in RUN_KEYS))
elif note:
    log(note)
DEVICE = get_device()  # 'cuda' (NVIDIA GPU), 'mps' (Apple Silicon GPU), or 'cpu' if no GPU is found
log(f'Using device: {DEVICE}')

# The whole dataset fits in GPU memory as uint8 (~200 MB), so we index batches directly on the GPU
data = load_afhq(args.data_path)
train_images, train_labels = data['train_images'].to(DEVICE), data['train_labels'].to(DEVICE)
val_images, val_labels = to_float(data['val_images']).to(DEVICE), data['val_labels'].to(DEVICE)
classes = data['classes']
log(f'{len(train_images)} training images: '
      + ', '.join(f'{(train_labels == c).sum().item()} {name}' for c, name in enumerate(classes))
      + f'; {len(val_images)} validation images')

model_config = dict(num_classes=len(classes), channels=tuple(int(c) for c in args.channels.split(',')))
net = FlowUNet(**model_config).to(DEVICE)
ema = copy.deepcopy(net).eval().requires_grad_(False)  # the EMA copy is never trained directly
flow = ImageFlowMatching(net, args.p_uncond)            # for training: your Q1 loss with label dropout
ema_flow = ImageFlowMatching(ema)                       # for sampling with the EMA weights
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
step, minutes_before, losses, val_losses = 0, 0.0, [], []
if ckpt is not None:
    net.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    optimizer.load_state_dict(ckpt['optimizer'])
    step, minutes_before, losses = ckpt['step'], ckpt['minutes'], ckpt['losses']
    val_losses = ckpt['val_losses'] if 'val_losses' in ckpt else ckpt['heldout']  # 'heldout' before 2026-11-16
    log(f'Continuing from the checkpoint at step {step} ({minutes_before:.0f} minutes of training)')
n_params = sum(p.numel() for p in net.parameters())
log(f'FlowUNet with {n_params / 1e6:.1f}M parameters')

# bfloat16 autocast on NVIDIA GPUs: about 2x faster; the weights and the loss stay in float32
autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == 'cuda')
sample_noise = randn(8, seed=1234, device=DEVICE)  # the same noise for every sample grid


def sample_grid(f, scale=2.0, n_steps=32):
    # samples of the flow model f: one row of 8 images per class, from sample_noise, with guidance
    with autocast:
        rows = [f.sample(sample_noise, c, scale, n_steps=n_steps)[-1].float() for c in range(len(classes))]
    return torch.cat(rows).clamp(-1, 1)


def learning_rate(step):
    # linear warmup over warmup_steps, then constant or cosine decay to 0 at max_steps
    lr = args.lr * min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps > 0 else args.lr
    if args.lr_schedule == 'cosine':
        lr *= 0.5 * (1 + math.cos(math.pi * min(step / args.max_steps, 1.0)))
    return lr


def save_checkpoint(minutes):
    ckpt = {'model': net.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
            'minutes': minutes, 'losses': losses, 'val_losses': val_losses, 'model_config': model_config,
            'classes': classes, 'args': vars(args)}
    torch.save(ckpt, ckpt_path + '.tmp')
    os.replace(ckpt_path + '.tmp', ckpt_path)  # never leave a half-written checkpoint behind


net.train()
t_start, step_start, running, n_running, minutes = time.time(), step, 0.0, 0, minutes_before
while step < args.max_steps:
    idx = torch.randint(len(train_images), (args.batch_size,), device=DEVICE)
    z = to_float(train_images[idx])
    z = torch.where(torch.rand(len(z), 1, 1, 1, device=DEVICE) < 0.5, z.flip(-1), z)  # random horizontal flips
    with autocast:
        loss = flow.get_loss(z, train_labels[idx])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(step)
    optimizer.step()
    update_ema(ema, net, args.ema_decay)
    step += 1

    running, n_running = running + loss.item(), n_running + 1
    if step % 100 == 0:  # record the average loss of the last 100 steps
        losses.append([step, running / n_running])
        running, n_running = 0.0, 0
    minutes = minutes_before + (time.time() - t_start) / 60
    done = step >= args.max_steps or minutes - minutes_before >= args.max_minutes
    if step % args.sample_every == 0 or done:
        val_losses.append([step, val_loss(ema, val_images, val_labels)])
        log(f'step {step:6d} | loss {losses[-1][1] if losses else loss.item():.4f} | val loss {val_losses[-1][1]:.4f} | '
            f'{minutes:5.1f} min | {(step - step_start) / (time.time() - t_start):.1f} steps/s')
        save_image(sample_grid(ema_flow) / 2 + 0.5, os.path.join(args.run_dir, 'samples', f'step{step:06d}_ema.png'),
                   nrow=8)
        save_checkpoint(minutes)
    if done:
        break
log(f'Stopped after {step} steps ({minutes:.0f} minutes of training in total), saved in {args.run_dir}')
if step == step_start:
    log(f'This run already had --max_steps {args.max_steps} steps: raise --max_steps to train it for longer')
elif step < args.max_steps:
    log(f'To continue this run, train again with --run_dir {args.run_dir} --resume')
