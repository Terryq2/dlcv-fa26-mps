"""
PixNerd-XXL-P16-T2I (Wang et al., 2025, https://github.com/MCG-NJU/PixNerd, MIT license, see LICENSE): a 1.2B parameter
text-to-image flow matching model that generates 512 x 512 images directly in pixel space (no VAE), with Qwen3-1.7B as
its text encoder. The other files in this folder are copied unchanged from the official repository (only the imports
were made relative); this wrapper is provided for Q4.

Convention (same as Q2 and Q3): t = 0 is noise and t = 1 is the image, x_t = t z + (1 - t) eps, and
the model predicts the velocity u_t(x_t | prompt) ~ z - eps. Images are float tensors in [-1, 1] of shape (B, 3, 512, 512).
"""
import os

# On Delta AI, the weights are already downloaded to a shared, read-only Hugging Face cache. This has to be set before
# huggingface_hub is imported.
DELTA_HF_HOME = '/work/nvme/bfdu/cs444/fa26/mp5/hf_cache'
if os.path.isdir(DELTA_HF_HOME) and 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = DELTA_HF_HOME
    os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from huggingface_hub import hf_hub_download

from .denoiser import PixNerDiT

MODEL_REPO = 'MCG-NJU/PixNerd-XXL-P16-T2I'
TEXT_ENCODER_REPO = 'Qwen/Qwen3-1.7B'
MAX_TOKENS = 128
RESOLUTION = 512


class PixNerd:
    """
    Usage:
        model = PixNerd('cuda')               # downloads ~8 GB the first time (already done on Delta AI)
        emb = model.encode('a photo of a cat')  # prompt embedding of shape (1, 128, 2048), cached on disk
        u = model(x, t, emb)                  # velocity u_t(x | prompt) of the same shape as x
        model.null_emb                        # embedding of the empty prompt, for classifier-free guidance
    """

    def __init__(self, device='cuda', cache_path='pixnerd_prompt_embeds.pt'):
        self.device = device
        self.net = PixNerDiT(in_channels=3, patch_size=16, num_groups=24, hidden_size=1536, txt_embed_dim=2048,
                             txt_max_length=MAX_TOKENS, num_text_blocks=4, decoder_hidden_size=64,
                             num_encoder_blocks=16, num_decoder_blocks=2)
        ckpt = torch.load(hf_hub_download(MODEL_REPO, 'model.ckpt'), map_location='cpu', weights_only=True)
        prefix = 'ema_denoiser.'
        self.net.load_state_dict({k[len(prefix):]: v for k, v in ckpt['state_dict'].items() if k.startswith(prefix)})
        del ckpt
        self.net = self.net.to(device, torch.bfloat16).eval().requires_grad_(False)
        self.cache_path = cache_path
        self.cache = torch.load(cache_path) if cache_path and os.path.exists(cache_path) else {}
        self.tokenizer = self.text_encoder = None
        self.null_emb = self.encode('')

    @torch.no_grad()
    def encode(self, prompt):
        """Embedding of a text prompt, of shape (1, 128, 2048). New prompts are encoded with Qwen3 and cached."""
        if prompt not in self.cache:
            if self.text_encoder is None:
                from transformers import AutoTokenizer, Qwen3Model
                self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_REPO, padding_side='right')
                self.text_encoder = Qwen3Model.from_pretrained(TEXT_ENCODER_REPO, torch_dtype=torch.bfloat16).eval()
            encoder = self.text_encoder.to(self.device)  # on the GPU only while encoding, to save memory
            tokens = self.tokenizer([prompt], truncation=True, max_length=MAX_TOKENS, padding='max_length',
                                    return_tensors='pt').to(self.device)
            self.cache[prompt] = encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)[0].cpu()
            encoder.to('cpu')
            if self.cache_path:
                torch.save(self.cache, self.cache_path)
        return self.cache[prompt].to(self.device)

    @torch.no_grad()
    def __call__(self, x, t, emb):
        """
        Predict the velocity u_t(x | prompt).
        :param x: images of shape (B, 3, 512, 512), float in [-1, 1] (and beyond, for noisy images)
        :param t: time, a float, or a tensor with B elements
        :param emb: prompt embeddings from `encode`, of shape (1, 128, 2048) or (B, 128, 2048)
        :return: velocity of shape (B, 3, 512, 512), in float32
        """
        B = x.shape[0]
        t = torch.as_tensor(t, dtype=torch.float32, device=x.device).reshape(-1).expand(B)
        emb = emb.to(x.device).expand(B, -1, -1)
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            u = self.net(x, t, emb)
        return u.float()
