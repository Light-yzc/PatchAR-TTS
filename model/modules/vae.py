import torch
from pathlib import Path
from diffusers.models import AutoencoderOobleck


def load_vae(vae_path: str, device="cuda", precision="bf16"):
    """Load pre-trained VAE model.
    
    Args:
        vae_path: path to VAE model directory (or HuggingFace model ID)
        device: device to load the model on
        precision: 'bf16', 'fp16', or 'fp32'
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(precision, torch.bfloat16)

    vae_path = Path(vae_path)
    if vae_path.is_dir():
        vae = AutoencoderOobleck.from_pretrained(str(vae_path))
    else:
        # Try loading as HuggingFace model ID string
        vae = AutoencoderOobleck.from_pretrained(str(vae_path))

    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def vae_encode(vae, waveform: torch.Tensor) -> torch.Tensor:
    """Encode waveform to latent.
    
    Args:
        waveform: (B, C, samples) or (B, samples) audio tensor at 48kHz
    Returns:
        latent: (B, T, D) latent representation
    """
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)  # (B, samples) → (B, 1, samples)
    # VAE expects 2-channel audio; duplicate mono → stereo
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)  # (B, 1, S) → (B, 2, S)
    # Match VAE dtype and device
    waveform = waveform.to(device=vae.device, dtype=vae.dtype)
    latent = vae.encode(waveform).latent_dist.sample()
    # AutoencoderOobleck returns (B, D, T), transpose to (B, T, D)
    if latent.dim() == 3 and latent.shape[1] != latent.shape[2]:
        latent = latent.transpose(1, 2)  # (B, D, T) → (B, T, D)
    return latent


@torch.no_grad()
def vae_decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to waveform.
    
    Args:
        latent: (B, T, D) latent representation
    Returns:
        waveform: (B, 2, samples) stereo audio tensor at 48kHz
    """
    latent = latent.to(device=vae.device, dtype=vae.dtype)
    # AutoencoderOobleck expects (B, D, T)
    if latent.dim() == 3:
        latent = latent.transpose(1, 2)  # (B, T, D) → (B, D, T)
    waveform = vae.decode(latent).sample  # (B, 2, samples)
    return waveform