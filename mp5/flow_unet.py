import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow import FlowMatching


def group_norm(channels):
    """GroupNorm with 32 groups (or 8 for layers with fewer than 64 channels), as in most image UNets (provided)."""
    return nn.GroupNorm(32 if channels >= 64 and channels % 32 == 0 else 8, channels)


####################
# Q2.1             #
####################

class TimeEmbedding(nn.Module):
    """Embedding of the time t in [0, 1]: sinusoidal features, followed by a small MLP."""
    
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def sinusoidal(self, t):
        """
        Sinusoidal features of the times t, as in the positional encoding of Transformers. With h = dim / 2
        frequencies f_i = 10000 ** (-i / h) for i = 0, ..., h - 1, the features of a time t are
            [sin(1000 t f_0), ..., sin(1000 t f_{h-1}), cos(1000 t f_0), ..., cos(1000 t f_{h-1})].
        (t is scaled by 1000 so that the fastest features go through many periods as t goes from 0 to 1.)
        :param t: times of shape (B,)
        :return: features of shape (B, dim), in float32
        """
        # TODO(student): Complete this function
        raise NotImplementedError

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


####################
# Q2.2             #
####################

class ResBlock(nn.Module):
    """Residual block whose features are modulated by the embedding of the time and the class (FiLM)."""

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)  # predicts a scale and a shift for each of the out_ch channels
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        """
        Each convolution is preceded by its normalization (norm1, norm2) and a SiLU. Between the two convolutions,
        modulate the normalized features with FiLM, as in the notebook: emb_proj maps the SiLU of the embedding to a
        scale (the first out_ch values) and a shift (the last out_ch values) for every channel. Finally, add the skip
        connection of the input.
        :param x: features of shape (B, in_ch, H, W)
        :param emb: embeddings of shape (B, emb_dim)
        :return: features of shape (B, out_ch, H, W)
        """
        # TODO(student): Complete this function
        raise NotImplementedError


class AttentionBlock(nn.Module):
    """Multi-head self-attention over the pixels of a feature map, with a residual connection (provided)."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)  # (B, heads, H * W, C / heads)
        return x + self.proj(h.transpose(-1, -2).reshape(B, C, H, W))


####################
# Q2.2, Q2.3       #
####################

class FlowUNet(nn.Module):
    """
    UNet that predicts the velocity v(x_t, t, label) for class-conditional flow matching on RGB images. The layers are
    provided; you implement the embedding (Q2.2) and how the layers are connected (Q2.3).
    Every level has two ResBlocks and halves the resolution of the previous level (64 -> 32 -> 16 -> 8 for the default
    four levels). The decoder mirrors the encoder and concatenates the features of the same level (skip connections).
    Self-attention is only used at the lowest resolution, in the middle of the network.
    The label `num_classes` is the *null* label, used for the unconditional velocity in classifier-free guidance.
    Call it as `net(x, t, labels)`; `ImageFlowMatching` below wraps it for training and sampling.
    """

    def __init__(self, num_classes=3, channels=(32, 64, 128, 256)):
        super().__init__()
        self.num_classes = num_classes
        self.null_label = num_classes
        emb_dim = 4 * channels[0]
        self.time_emb = TimeEmbedding(channels[0], emb_dim)
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)  # one extra embedding for the null label

        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        ch = channels[0]
        for out_ch in channels:
            self.down.append(nn.ModuleList([ResBlock(ch, out_ch, emb_dim), ResBlock(out_ch, out_ch, emb_dim)]))
            ch = out_ch
        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim), AttentionBlock(ch), ResBlock(ch, ch, emb_dim)])
        for out_ch in reversed(channels):  # the first block of a level also takes the skip connection (out_ch channels)
            self.up.append(nn.ModuleList([ResBlock(ch + out_ch, out_ch, emb_dim), ResBlock(out_ch, out_ch, emb_dim)]))
            ch = out_ch
        self.downsample = nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1) for c in channels[:-1]])
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)

    def embed(self, t, labels):
        """
        Embedding of the times and the class labels, combined into one vector per image as described in the notebook.
        :param t: times of shape (B,)
        :param labels: class labels of shape (B,), with values in {0, ..., num_classes} (num_classes is the null label)
        :return: embeddings of shape (B, emb_dim)
        """
        # TODO(student): Complete this function
        raise NotImplementedError

    def forward(self, x, t, labels):
        """
        Predict the velocity (Q2.3), using the layers created in __init__. The provided first lines turn t and labels
        into tensors of shape (B,).
        1. Compute the embedding once with `self.embed`. Every ResBlock takes it as its second input (the attention
           block only takes the features).
        2. Encoder: apply conv_in, then the levels of `self.down` in order. Before every level but the first, halve the
           resolution with the matching layer of `self.downsample`. After the two ResBlocks of a level, keep the
           features for the skip connection.
        3. Middle: apply the three blocks of `self.mid` in order.
        4. Decoder: apply the levels of `self.up` in order (from the lowest resolution back to the highest). Before
           every level but the first, double the resolution with F.interpolate(h, scale_factor=2, mode='nearest').
           Then concatenate the kept encoder features of the same resolution along the channels, in the order
           [h, skip] (the test expects this order), and apply the two ResBlocks of the level.
        5. Output: apply norm_out, a SiLU and conv_out.
        :param x: images of shape (B, 3, H, W)
        :param t: time, a float, or a tensor of shape (B,)
        :param labels: class label, an int, or a tensor of shape (B,); `null_label` for the unconditional velocity
        :return: predicted velocity of shape (B, 3, H, W)
        """
        B = x.shape[0]
        t = torch.as_tensor(t, dtype=torch.float32, device=x.device).reshape(-1).expand(B)
        labels = torch.as_tensor(labels, dtype=torch.long, device=x.device).reshape(-1).expand(B)
        # TODO(student): Complete this function
        raise NotImplementedError


####################
# Q2.4, Q2.6       #
####################

class ImageFlowMatching(FlowMatching):
    """
    Class-conditional flow matching on images with a `FlowUNet`. This reuses `interpolate`, `get_loss` and `sample`
    from your `FlowMatching` in flow.py (Q1) without changes, and only overrides what is different for images:
    the velocity (a UNet that also takes the class label, with classifier-free guidance), the distribution of the
    training times, and label dropout in the loss.
    """

    def __init__(self, velocity_net, p_uncond=0.1):
        """
        :param velocity_net: a `FlowUNet`, called as `velocity_net(x, t, labels)`, with a `null_label` attribute
        :param p_uncond: probability of replacing a label by the null label during training
        """
        super().__init__(velocity_net)
        self.p_uncond = p_uncond

    def sample_times(self, n, device=None, dtype=None):
        """
        Logit-normal training times t = sigmoid(n) with n ~ N(0, 1), as in Stable Diffusion 3 (provided).
        Your `interpolate` from Q1 draws its times with this method.
        :return: times of shape (n,)
        """
        return torch.sigmoid(torch.randn(n, device=device, dtype=dtype))

    def get_loss(self, z, labels):
        """
        Conditional flow matching loss with label dropout (Q2.4): replace each label by the null label with probability
        `self.p_uncond`, independently for each image, and compute your Q1 loss with the new labels.
        :param z: clean images of shape (B, C, H, W)
        :param labels: class labels of shape (B,)
        :return: conditional flow matching loss, a scalar tensor
        """
        # TODO(student): Complete this function
        raise NotImplementedError

    def get_velocity(self, x, t, labels, scale=1.0):
        """
        Velocity with classifier-free guidance (Q2.6), combining the guided and the unconditional (null label)
        predictions of `self.velocity_net` as in the CFG equation in the notebook. With scale = 1, this is just the
        guided velocity: call the network only once in that case (training always uses scale = 1).
        :param x: images of shape (B, C, H, W)
        :param t: time, a float, or a tensor with B elements
        :param labels: class label, an int, or a tensor of shape (B,)
        :param scale: guidance scale w
        :return: velocity of shape (B, C, H, W)
        """
        # TODO(student): Complete this function
        raise NotImplementedError


####################
# Q2.5             #
####################

@torch.no_grad()
def update_ema(ema_model, model, decay):
    """
    Update the exponential moving average (EMA) of the weights in place, as in the EMA equation in the notebook:
    move every parameter of ema_model slightly towards the corresponding parameter of model.
    :param ema_model: model holding the averaged weights, modified in place
    :param model: model being trained, must not be modified
    :param decay: EMA decay, close to 1 (e.g. 0.999)
    """
    # TODO(student): Complete this function
    raise NotImplementedError
