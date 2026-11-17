"""Provided helpers for Q2. You do not need to modify this file."""
import math
import os

import torch
from PIL import Image

DELTA_DATA_PATH = '/work/nvme/bfdu/cs444/fa26/mp5/data/afhq64.pt'  # already prepared for you on Delta AI
DATA_PATH = DELTA_DATA_PATH if os.path.exists(DELTA_DATA_PATH) else 'data/afhq64.pt'
RUN_DIR = 'runs/afhq'
OUT_DIR = 'outputs'


def get_device():
    """The best available device: an NVIDIA GPU ('cuda'), an Apple Silicon GPU ('mps'), or else the CPU ('cpu')."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


####################
# Images           #
####################

def to_float(images):
    """uint8 images in [0, 255] -> float images in [-1, 1]."""
    return images.float() / 127.5 - 1


def load_afhq(path=DATA_PATH):
    """
    AFHQ at 64 x 64: a dict with 'train_images' and 'val_images' (uint8 tensors of shape (N, 3, 64, 64)),
    'train_labels' and 'val_labels' (0 = cat, 1 = dog, 2 = wild), and 'classes'. The validation images are never used
    for training: they are held out to evaluate trained models and to choose hyperparameters (see `val_loss`).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run prepare_afhq() first (see the Training section of the notebook).')
    data = torch.load(path)
    if 'val_images' not in data:  # old format (before 2026-11-16), with test images instead of validation images
        data = split_val({'train_images': torch.cat([data['train_images'], data['test_images']]),
                          'train_labels': torch.cat([data['train_labels'], data['test_labels']]),
                          'classes': data['classes']})
        torch.save(data, path)
        print(f'Converted {path} to the new format: the test images are now training images, and 64 images per '
              f'class are validation images.')
    return data


def prepare_afhq(path=DATA_PATH, res=64, n_val=64, max_shards=None, seed=0):
    """
    Download AFHQv2 (~15.8k photos of cats, dogs and wild animals at 512 x 512, from huggan/AFHQv2 on Hugging Face),
    resize it to res x res, and save it to `path` as uint8 tensors (~200 MB at 64 x 64), in the format of `load_afhq`.
    The download (~7 GB) comes in 13 parts, and each part is deleted once it is processed, so this needs only ~1 GB of
    free disk space. It takes about 10 minutes. n_val random images per class are held out for validation, and all
    the others are training images.
    """
    import io
    import shutil
    import tempfile

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    repo, classes = 'huggan/AFHQv2', ['cat', 'dog', 'wild']  # the label names of huggan/AFHQv2, in order
    shards = sorted(f for f in list_repo_files(repo, repo_type='dataset') if f.endswith('.parquet'))
    if max_shards:  # evenly spaced parts (for quick tests), since the parts are sorted by class
        shards = [shards[i] for i in np.linspace(0, len(shards) - 1, max_shards).round().astype(int)]
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=os.path.dirname(path) or '.')
    images, labels = [], []
    try:
        for k, shard in enumerate(shards):
            shard_path = hf_hub_download(repo, shard, repo_type='dataset', local_dir=tmp_dir)
            for row in pq.read_table(shard_path, columns=['image', 'label']).to_pylist():
                im = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
                s = min(im.size)  # center crop to a square (AFHQ images are already square)
                im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
                images.append(np.asarray(im.resize((res, res), Image.LANCZOS)))
                labels.append(row['label'])
            os.remove(shard_path)
            print(f'part {k + 1}/{len(shards)}: {len(images)} images')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()  # (N, 3, res, res), uint8
    data = split_val({'train_images': images, 'train_labels': torch.tensor(labels), 'classes': classes}, n_val, seed)
    torch.save(data, path)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.0f} MB): {len(data['train_labels'])} training and "
          f"{len(data['val_labels'])} validation images")


def split_val(data, n_val=64, seed=1):
    """Move n_val random training images per class of `data` to 'val_images' and 'val_labels' (used by prepare_afhq)."""
    labels = data['train_labels']
    g = torch.Generator().manual_seed(seed)
    val = torch.cat([torch.nonzero(labels == c)[:, 0][torch.randperm(int((labels == c).sum()), generator=g)[:n_val]]
                     for c in range(len(data['classes']))])
    train = torch.ones(len(labels), dtype=torch.bool)
    train[val] = False
    return {'train_images': data['train_images'][train], 'train_labels': labels[train],
            'val_images': data['train_images'][val], 'val_labels': labels[val], 'classes': data['classes']}


def show_images(images, titles=None, ncols=None, fname=None, size=1.6, row_labels=None):
    """
    Show a grid of images with values in [-1, 1], and save it to outputs/<fname> if fname is given.
    :param images: a tensor of shape (N, 3, H, W), or a list of tensors of shape (3, H, W) or (1, 3, H, W)
    :param titles: titles of the first len(titles) images (e.g. only of the first row)
    :param row_labels: labels shown left of each row
    """
    import matplotlib.pyplot as plt  # here, not at the top: the autograder imports this file and has no matplotlib
    images =[im[0] if im.dim() == 4 else im for im in images]
    ncols = ncols or len(images)
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for k, (ax, im) in enumerate(zip(axes.flat, images)):
        ax.imshow((im.detach().float().cpu().permute(1, 2, 0).clamp(-1, 1) / 2 + 0.5).numpy())
        if titles and k < len(titles) and titles[k]:  # titles may cover only the first row
            ax.set_title(titles[k], fontsize=8)
    if row_labels:
        for r, label in enumerate(row_labels):
            axes[r, 0].set_ylabel(label, fontsize=9)
    plt.tight_layout()
    if fname:
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=100)
    plt.show()


def randn(n, seed, device, shape=(3, 64, 64)):
    """n noise images, drawn on the CPU so that a seed gives the same noise on every device."""
    return torch.randn(n, *shape, generator=torch.Generator().manual_seed(seed)).to(device)


####################
# Trained model    #
####################

def load_trained_model(run_dir=RUN_DIR, device='cpu'):
    """
    Load the EMA weights of the FlowUNet trained in Q2_flow_matching.ipynb from <run_dir>/ckpt.pt, in eval mode.
    :return: the network, and the training log (a dict with 'step', 'minutes', 'losses', 'val_losses',
             'model_config', 'classes', 'args'): 'losses' (training) and 'val_losses' are lists of [step, loss]
    """
    from flow_unet import FlowUNet
    ckpt = torch.load(os.path.join(run_dir, 'ckpt.pt'), map_location='cpu')
    net = FlowUNet(**ckpt['model_config'])
    net.load_state_dict(ckpt['ema'])
    log = {k: ckpt[k] for k in ['step', 'minutes', 'losses', 'model_config', 'classes']}
    log['val_losses'] = ckpt['val_losses'] if 'val_losses' in ckpt else ckpt['heldout']  # 'heldout' before 2026-11-16
    log['args'] = ckpt.get('args', {})  # the hyperparameters of train_flow.py
    return net.to(device).eval().requires_grad_(False), log


####################
# Evaluation       #
####################

VAL_TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)


@torch.no_grad()
def val_loss(net, images, labels, times=VAL_TIMES, batch_size=64):
    """
    Conditional flow matching loss of `net` on the validation images, with fixed noise and fixed times instead of
    random ones, so that the same weights always get the same value. Lower is better. As in training, the target of
    x_t = t z + (1 - t) eps is z - eps, and the loss is averaged over the images, the pixels and the times.
    :param net: a network called as net(x, t, labels), e.g. a FlowUNet with the EMA weights
    :param images: validation images z in [-1, 1], of shape (N, 3, H, W)
    :param labels: their class labels, of shape (N,)
    :return: the validation loss, a float
    """
    device = next(net.parameters()).device
    total = 0.0
    for k, t in enumerate(times):
        eps = torch.randn(images.shape, generator=torch.Generator().manual_seed(k))  # the same noise on every device
        for i in range(0, len(images), batch_size):
            z, e, y = images[i:i + batch_size].to(device), eps[i:i + batch_size].to(device), labels[i:i + batch_size]
            u = net(t * z + (1 - t) * e, t, y.to(device)).float()
            total += ((u - (z - e)) ** 2).sum().item()
    return total / (len(times) * images.numel())
