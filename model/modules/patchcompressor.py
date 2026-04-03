from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCompressor(nn.Module):
    """
    Compress latent frame patches into one token per patch.

    Input:
    - latents: [B, T, D]
    - lengths or frame_mask to mark valid frames

    Steps:
    1. Right-pad frames so T is divisible by patch_size
    2. Reshape to [B, N, P, D]
    3. Project each frame to model_dim
    4. Prepend one learnable summary token per patch
    5. Run a small non-causal Transformer on each patch
    6. Take the summary token as the patch token
    """

    def __init__(
        self,
        latent_dim: int,
        model_dim: int,
        patch_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        infer_mask_from_zero_pad: bool = True,
        pad_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.infer_mask_from_zero_pad = infer_mask_from_zero_pad
        self.pad_epsilon = pad_epsilon

        self.input_proj = nn.Linear(latent_dim, model_dim)
        self.summary_token = nn.Parameter(torch.randn(1, 1, 1, model_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, patch_size + 1, model_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)

    def _build_frame_mask(
        self,
        latents: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        lengths: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return a bool mask with True = valid frame, False = padded frame.
        """
        if frame_mask is not None:
            if frame_mask.shape != (batch_size, seq_len):
                raise ValueError(f"frame_mask must have shape {(batch_size, seq_len)}")
            return frame_mask.to(device=device, dtype=torch.bool)

        if lengths is None:
            if self.infer_mask_from_zero_pad:
                # In this project the collator pads latent sequences with all-zero frames.
                # Treat a frame as padded only if every channel is near zero.
                return latents.abs().amax(dim=-1) > self.pad_epsilon
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        if lengths.shape != (batch_size,):
            raise ValueError(f"lengths must have shape {(batch_size,)}")

        time_index = torch.arange(seq_len, device=device).unsqueeze(0)
        return time_index < lengths.to(device=device).unsqueeze(1)

    def patchify(
        self,
        latents: torch.Tensor,
        lengths: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert [B, T, D] into:
        - patches: [B, N, P, D]
        - patch_frame_mask: [B, N, P] with True = valid frame
        """
        if latents.dim() != 3:
            raise ValueError("latents must have shape [B, T, D]")

        batch_size, seq_len, latent_dim = latents.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")

        valid_frame_mask = self._build_frame_mask(
            latents=latents,
            batch_size=batch_size,
            seq_len=seq_len,
            device=latents.device,
            lengths=lengths,
            frame_mask=frame_mask,
        )

        if seq_len == 0:
            empty_patches = latents.new_zeros(batch_size, 0, self.patch_size, self.latent_dim)
            empty_mask = torch.zeros(batch_size, 0, self.patch_size, device=latents.device, dtype=torch.bool)
            return empty_patches, empty_mask

        pad_frames = (-seq_len) % self.patch_size
        if pad_frames > 0:
            latents = F.pad(latents, (0, 0, 0, pad_frames))
            valid_frame_mask = F.pad(valid_frame_mask, (0, pad_frames), value=False)

        num_patches = latents.shape[1] // self.patch_size
        patches = latents.view(batch_size, num_patches, self.patch_size, self.latent_dim)
        patch_frame_mask = valid_frame_mask.view(batch_size, num_patches, self.patch_size)
        return patches, patch_frame_mask

    def forward(
        self,
        latents: torch.Tensor,
        lengths: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        return_patch_mask: bool = False,
    ):
        """
        Returns:
        - patch_tokens: [B, N, H]
        - patch_mask: [B, N] where True means this patch contains at least one real frame
        """
        patches, patch_frame_mask = self.patchify(
            latents=latents,
            lengths=lengths,
            frame_mask=frame_mask,
        )

        batch_size, num_patches, patch_size, _ = patches.shape
        if num_patches == 0:
            empty_tokens = latents.new_zeros(batch_size, 0, self.model_dim)
            empty_mask = torch.zeros(batch_size, 0, device=latents.device, dtype=torch.bool)
            return (empty_tokens, empty_mask) if return_patch_mask else empty_tokens

        patch_tokens = self.input_proj(patches)

        summary = self.summary_token.expand(batch_size, num_patches, 1, self.model_dim)
        patch_tokens = torch.cat([summary, patch_tokens], dim=2)
        patch_tokens = patch_tokens + self.pos_embed[:, : patch_size + 1].unsqueeze(1)

        summary_mask = torch.ones(batch_size, num_patches, 1, device=latents.device, dtype=torch.bool)
        token_valid_mask = torch.cat([summary_mask, patch_frame_mask], dim=2)
        patch_valid_mask = patch_frame_mask.any(dim=-1)

        flat_tokens = patch_tokens.view(batch_size * num_patches, patch_size + 1, self.model_dim)
        flat_valid_mask = token_valid_mask.view(batch_size * num_patches, patch_size + 1)

        encoded = self.encoder(
            flat_tokens,
            src_key_padding_mask=~flat_valid_mask,
        )
        summary_tokens = self.output_norm(encoded[:, 0])
        summary_tokens = summary_tokens.view(batch_size, num_patches, self.model_dim)

        # Zero-out fully padded patches so later modules do not accidentally use them.
        summary_tokens = summary_tokens * patch_valid_mask.unsqueeze(-1).to(summary_tokens.dtype)

        if return_patch_mask:
            return summary_tokens, patch_valid_mask
        return summary_tokens
