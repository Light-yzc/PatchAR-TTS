from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class DiTConfig:
    """Configuration for the patch-level DiT decoder."""

    latent_dim: int
    max_chunk_size: int
    cond_token_dim: int

    model_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    infer_mask_from_zero_pad: bool = True
    pad_epsilon: float = 1e-8


@dataclass
class FlowMatchingConfig:
    """Configuration for the continuous flow-matching objective and sampler."""

    sigma_min: float = 1e-5
    cond_dropout_prob: float = 0.1
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    log_norm_mu: float = -0.4
    log_norm_sigma: float = 1.0
    loss_type: str = "mse"
    huber_delta: float = 0.1


class SinusoidalEmbedding(nn.Module):
    """Classic sin/cos embedding used for diffusion time and short token positions."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Sinusoidal embedding dim must be even")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)

        x = x.float()
        half_dim = self.dim // 2
        device = x.device
        freq = torch.exp(
            -torch.arange(half_dim, device=device, dtype=x.dtype)
            * (torch.log(torch.tensor(10000.0, device=device, dtype=x.dtype)) / max(half_dim - 1, 1))
        )
        angles = x.unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-style affine modulation."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = model_dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    A chunk decoder block:
    1. self-attn on [history fine, current fine]
    2. cross-attn from current fine tokens to patch condition tokens
    3. FFN on the full fine-token sequence
    """

    def __init__(self, model_dim: int, num_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ffn = FeedForward(model_dim=model_dim, ff_mult=ff_mult, dropout=dropout)
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 9),
        )

    def forward(
        self,
        latent_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
        cond_tokens: torch.Tensor,
        cond_mask: torch.Tensor,
        adaln_cond: torch.Tensor,
        current_length: int,
    ) -> torch.Tensor:
        # One fused conditioning vector produces the affine parameters and
        # residual gates for self-attn, cross-attn, and FFN inside this block.
        (
            self_shift,
            self_scale,
            self_gate,
            cross_shift,
            cross_scale,
            cross_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = self.ada_proj(adaln_cond).chunk(9, dim=-1)

        x = modulate(self.self_norm(latent_tokens), self_shift, self_scale)
        self_attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=~latent_mask,
            need_weights=False,
        )
        latent_tokens = latent_tokens + self_gate.unsqueeze(1) * self_attn_out
        latent_tokens = latent_tokens * latent_mask.unsqueeze(-1).to(latent_tokens.dtype)

        if cond_tokens.shape[1] > 0:
            history_tokens = latent_tokens[:, :-current_length, :]
            current_tokens = latent_tokens[:, -current_length:, :]
            current_mask = latent_mask[:, -current_length:]

            # Only the current patch queries the slot condition tokens.
            # History stays on the latent side and influences the current patch
            # through self-attention instead.
            current_q = modulate(self.cross_norm(current_tokens), cross_shift, cross_scale)
            cross_attn_out, _ = self.cross_attn(
                query=current_q,
                key=cond_tokens,
                value=cond_tokens,
                key_padding_mask=~cond_mask,
                need_weights=False,
            )
            current_tokens = current_tokens + cross_gate.unsqueeze(1) * cross_attn_out
            current_tokens = current_tokens * current_mask.unsqueeze(-1).to(current_tokens.dtype)
            latent_tokens = torch.cat([history_tokens, current_tokens], dim=1)

        ffn_input = modulate(self.ffn_norm(latent_tokens), ffn_shift, ffn_scale)
        latent_tokens = latent_tokens + ffn_gate.unsqueeze(1) * self.ffn(ffn_input)
        latent_tokens = latent_tokens * latent_mask.unsqueeze(-1).to(latent_tokens.dtype)
        return latent_tokens


class PatchDiT(nn.Module):
    """
    Chunk-level DiT.

    Inputs:
    - noisy_fine_chunk: current noisy fine latent chunk
    - cond_tokens: patch condition tokens for this chunk
    - history_fine_latents: previous fine latent history
    """

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        if config.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if config.model_dim % config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.config = config
        self.latent_dim = config.latent_dim
        self.max_chunk_size = config.max_chunk_size
        self.cond_token_dim = config.cond_token_dim
        self.model_dim = config.model_dim
        self.infer_mask_from_zero_pad = config.infer_mask_from_zero_pad
        self.pad_epsilon = config.pad_epsilon
        self.gradient_checkpointing = False

        self.latent_proj = nn.Linear(config.latent_dim, config.model_dim)
        self.cond_proj = nn.Linear(config.cond_token_dim, config.model_dim)
        self.time_embed = SinusoidalEmbedding(config.model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.adaln_cond_fuse = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.latent_pos_embed = SinusoidalEmbedding(config.model_dim)
        self.cond_pos_embed = SinusoidalEmbedding(config.model_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    model_dim=config.model_dim,
                    num_heads=config.num_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.latent_dim)

        self._init_adaln_zero()

    def _init_adaln_zero(self) -> None:
        """AdaLN-Zero: zero-initialize gate/shift/scale projections and the
        final output projection so the network starts as the identity function.
        This is critical for stable training (Peebles & Xie, 2023)."""
        for block in self.blocks:
            # ada_proj is nn.Sequential(SiLU(), Linear) — zero the Linear
            nn.init.zeros_(block.ada_proj[-1].weight)
            nn.init.zeros_(block.ada_proj[-1].bias)
        # Zero-init the final output projection so initial velocity prediction is zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def enable_gradient_checkpointing(self) -> None:
        """Recompute DiT blocks on backward to lower activation memory."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _infer_mask_from_zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Treat all-zero rows as padding when an explicit mask is absent."""
        return x.abs().amax(dim=-1) > self.pad_epsilon

    def _prepare_history(
        self,
        history_fine_latents: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project optional fine-latent history into the DiT token space.

        History is kept as explicit latent tokens, not merged into cond_tokens,
        so local acoustic continuity flows through self-attention.
        """
        if history_fine_latents is None:
            history_tokens = torch.zeros(batch_size, 0, self.model_dim, device=device)
            history_mask_bool = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
            return history_tokens, history_mask_bool

        if history_fine_latents.dim() != 3:
            raise ValueError("history_fine_latents must have shape [B, T_hist, D]")
        if history_fine_latents.shape[0] != batch_size:
            raise ValueError("history_fine_latents batch size must match noisy_fine_chunk batch size")
        if history_fine_latents.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected history latent_dim={self.latent_dim}, got {history_fine_latents.shape[-1]}")

        history_tokens = self.latent_proj(history_fine_latents)
        if history_mask is None:
            if self.infer_mask_from_zero_pad:
                history_mask_bool = self._infer_mask_from_zero_pad(history_fine_latents)
            else:
                history_mask_bool = torch.ones(history_fine_latents.shape[:2], device=device, dtype=torch.bool)
        else:
            if history_mask.shape != history_fine_latents.shape[:2]:
                raise ValueError(f"history_mask must have shape {history_fine_latents.shape[:2]}")
            history_mask_bool = history_mask.to(device=device, dtype=torch.bool)

        history_tokens = history_tokens * history_mask_bool.unsqueeze(-1).to(history_tokens.dtype)
        return history_tokens, history_mask_bool

    def forward(
        self,
        noisy_fine_chunk: torch.Tensor,
        timesteps: torch.Tensor,
        cond_tokens: torch.Tensor,
        speaker_cond: torch.Tensor | None = None,
        history_fine_latents: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_fine_chunk.dim() != 3:
            raise ValueError("noisy_fine_chunk must have shape [B, T_chunk, D]")
        if cond_tokens.dim() == 2:
            cond_tokens = cond_tokens.unsqueeze(1)
        if cond_tokens.dim() != 3:
            raise ValueError("cond_tokens must have shape [B, T_cond, C] or [B, C]")

        batch_size, chunk_size, latent_dim = noisy_fine_chunk.shape
        if chunk_size > self.max_chunk_size:
            raise ValueError(f"Chunk length {chunk_size} exceeds max_chunk_size={self.max_chunk_size}")
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        if cond_tokens.shape[0] != batch_size:
            raise ValueError("cond_tokens batch size must match noisy_fine_chunk batch size")
        if cond_tokens.shape[-1] != self.cond_token_dim:
            raise ValueError(
                f"Expected cond_token_dim={self.cond_token_dim}, got {cond_tokens.shape[-1]}"
            )

        if chunk_mask is not None:
            if chunk_mask.shape != (batch_size, chunk_size):
                raise ValueError(f"chunk_mask must have shape {(batch_size, chunk_size)}")
            current_mask = chunk_mask.to(device=noisy_fine_chunk.device, dtype=torch.bool)
        elif self.infer_mask_from_zero_pad:
            # Training usually passes an explicit chunk_mask. This branch mainly
            # exists as a safe fallback when padded chunks are fed without one.
            current_mask = self._infer_mask_from_zero_pad(noisy_fine_chunk)
        else:
            current_mask = torch.ones(batch_size, chunk_size, device=noisy_fine_chunk.device, dtype=torch.bool)

        # cond_tokens are LM-derived slots, not autoregressive time steps. They
        # are a short cross-attention memory for the current patch.
        cond_mask = self._infer_mask_from_zero_pad(cond_tokens)
        cond_seq = self.cond_proj(cond_tokens)
        cond_positions = torch.arange(cond_seq.shape[1], device=noisy_fine_chunk.device, dtype=timesteps.dtype)
        cond_seq = cond_seq + self.cond_pos_embed(cond_positions).to(cond_seq.dtype).unsqueeze(0)
        cond_seq = cond_seq * cond_mask.unsqueeze(-1).to(cond_seq.dtype)

        history_tokens, history_mask_bool = self._prepare_history(
            history_fine_latents=history_fine_latents,
            history_mask=history_mask,
            batch_size=batch_size,
            device=noisy_fine_chunk.device,
        )

        current_tokens = self.latent_proj(noisy_fine_chunk)
        time_cond = self.time_mlp(self.time_embed(timesteps))
        if speaker_cond is None:
            speaker_cond = torch.zeros(batch_size, self.model_dim, device=noisy_fine_chunk.device, dtype=current_tokens.dtype)
        else:
            if speaker_cond.shape != (batch_size, self.model_dim):
                raise ValueError(f"speaker_cond must have shape {(batch_size, self.model_dim)}")
            speaker_cond = speaker_cond.to(device=noisy_fine_chunk.device, dtype=current_tokens.dtype)
        adaln_cond = self.adaln_cond_fuse(torch.cat([time_cond, speaker_cond], dim=-1))

        # The DiT sequence is [history fine tokens | current noisy fine tokens].
        latent_tokens = torch.cat([history_tokens, current_tokens], dim=1)
        latent_mask = torch.cat([history_mask_bool, current_mask], dim=1)
        latent_positions = torch.arange(latent_tokens.shape[1], device=noisy_fine_chunk.device, dtype=timesteps.dtype)
        latent_tokens = latent_tokens + self.latent_pos_embed(latent_positions).to(latent_tokens.dtype).unsqueeze(0)
        latent_tokens = latent_tokens * latent_mask.unsqueeze(-1).to(latent_tokens.dtype)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                latent_tokens = checkpoint(
                    block,
                    latent_tokens,
                    latent_mask,
                    cond_seq,
                    cond_mask,
                    adaln_cond,
                    chunk_size,
                    use_reentrant=False,
                )
            else:
                latent_tokens = block(
                    latent_tokens=latent_tokens,
                    latent_mask=latent_mask,
                    cond_tokens=cond_seq,
                    cond_mask=cond_mask,
                    adaln_cond=adaln_cond,
                    current_length=chunk_size,
                )

        latent_tokens = self.output_norm(latent_tokens)
        # Only the tail corresponds to the current patch being denoised.
        current_hidden = latent_tokens[:, -chunk_size:, :]
        velocity = self.output_proj(current_hidden)
        velocity = velocity * current_mask.unsqueeze(-1).to(velocity.dtype)
        return velocity


class ConditionalFlowMatching(nn.Module):
    """Flow-matching wrapper around PatchDiT with training loss and Euler sampling."""

    def __init__(self, estimator: PatchDiT, config: FlowMatchingConfig | None = None) -> None:
        super().__init__()
        self.estimator = estimator
        self.config = config or FlowMatchingConfig()

    def _per_frame_loss(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        loss_type = self.config.loss_type.lower()
        if loss_type == "mse":
            return F.mse_loss(pred_velocity, target_velocity, reduction="none").mean(dim=-1)
        if loss_type == "huber":
            return F.huber_loss(
                pred_velocity,
                target_velocity,
                reduction="none",
                delta=self.config.huber_delta,
            ).mean(dim=-1)
        raise ValueError(f"Unsupported flow loss_type={self.config.loss_type!r}")

    def _maybe_dropout_cond(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.cond_dropout_prob <= 0:
            return cond_tokens

        keep_mask = torch.rand(cond_tokens.shape[0], device=cond_tokens.device)
        keep_mask = keep_mask > self.config.cond_dropout_prob
        return cond_tokens * keep_mask.view(-1, 1, 1).to(cond_tokens.dtype)

    def compute_loss(
        self,
        target_chunk: torch.Tensor,
        cond_tokens: torch.Tensor,
        speaker_cond: torch.Tensor | None = None,
        history_fine_latents: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_chunk.dim() != 3:
            raise ValueError("target_chunk must have shape [B, T_chunk, D]")

        batch_size = target_chunk.shape[0]
        if self.config.t_scheduler == "log-norm":
            # Log-normal schedule (same as VoxCPM): concentrates on informative
            # mid-range timesteps instead of wasting capacity on extremes.
            s = torch.randn(batch_size, device=target_chunk.device, dtype=target_chunk.dtype)
            s = s * self.config.log_norm_sigma + self.config.log_norm_mu
            timesteps = torch.sigmoid(s)
        else:
            timesteps = torch.rand(batch_size, device=target_chunk.device, dtype=target_chunk.dtype)
        noise = torch.randn_like(target_chunk)

        # Straight interpolation path between clean target_chunk and Gaussian noise.
        x_t = (1.0 - timesteps.view(-1, 1, 1)) * target_chunk + timesteps.view(-1, 1, 1) * noise
        target_velocity = noise - target_chunk

        pred_velocity = self.estimator(
            noisy_fine_chunk=x_t,
            timesteps=timesteps,
            cond_tokens=self._maybe_dropout_cond(cond_tokens),
            speaker_cond=speaker_cond,
            history_fine_latents=history_fine_latents,
            history_mask=history_mask,
            chunk_mask=chunk_mask,
        )

        per_frame_loss = self._per_frame_loss(pred_velocity, target_velocity)
        if chunk_mask is not None:
            chunk_weight = chunk_mask.to(device=target_chunk.device, dtype=per_frame_loss.dtype)
            return (per_frame_loss * chunk_weight).sum() / chunk_weight.sum().clamp_min(1.0)
        return per_frame_loss.mean()

    def _predict_velocity_with_cfg(
        self,
        noisy_fine_chunk: torch.Tensor,
        timesteps: torch.Tensor,
        cond_tokens: torch.Tensor,
        speaker_cond: torch.Tensor | None,
        history_fine_latents: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        chunk_mask: torch.Tensor | None,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale == 1.0:
            return self.estimator(
                noisy_fine_chunk=noisy_fine_chunk,
                timesteps=timesteps,
                cond_tokens=cond_tokens,
                speaker_cond=speaker_cond,
                history_fine_latents=history_fine_latents,
                history_mask=history_mask,
                chunk_mask=chunk_mask,
            )

        # CFG is implemented by pairing a conditioned branch with a zeroed-cond branch.
        zeros_cond = torch.zeros_like(cond_tokens)
        noisy_pair = torch.cat([noisy_fine_chunk, noisy_fine_chunk], dim=0)
        timestep_pair = torch.cat([timesteps, timesteps], dim=0)
        cond_pair = torch.cat([cond_tokens, zeros_cond], dim=0)
        speaker_pair = None
        if speaker_cond is not None:
            speaker_pair = torch.cat([speaker_cond, speaker_cond], dim=0)

        history_pair = None
        if history_fine_latents is not None:
            history_pair = torch.cat([history_fine_latents, history_fine_latents], dim=0)

        history_mask_pair = None
        if history_mask is not None:
            history_mask_pair = torch.cat([history_mask, history_mask], dim=0)

        chunk_mask_pair = None
        if chunk_mask is not None:
            chunk_mask_pair = torch.cat([chunk_mask, chunk_mask], dim=0)

        pred = self.estimator(
            noisy_fine_chunk=noisy_pair,
            timesteps=timestep_pair,
            cond_tokens=cond_pair,
            speaker_cond=speaker_pair,
            history_fine_latents=history_pair,
            history_mask=history_mask_pair,
            chunk_mask=chunk_mask_pair,
        )
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

    @torch.no_grad()
    def sample(
        self,
        cond_tokens: torch.Tensor,
        speaker_cond: torch.Tensor | None = None,
        history_fine_latents: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
        num_steps: int = 16,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        if self.config.solver != "euler":
            raise ValueError(f"Unsupported solver: {self.config.solver}")

        batch_size = cond_tokens.shape[0]
        device = cond_tokens.device
        dtype = cond_tokens.dtype
        chunk_size = chunk_mask.shape[1] if chunk_mask is not None else self.estimator.max_chunk_size

        sample = torch.randn(
            batch_size,
            chunk_size,
            self.estimator.latent_dim,
            device=device,
            dtype=dtype,
        ) * temperature

        t_span = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        for step in range(num_steps):
            t_now = t_span[step].expand(batch_size)
            dt = t_span[step] - t_span[step + 1]

            # Euler integration over the learned velocity field.
            velocity = self._predict_velocity_with_cfg(
                noisy_fine_chunk=sample,
                timesteps=t_now,
                cond_tokens=cond_tokens,
                speaker_cond=speaker_cond,
                history_fine_latents=history_fine_latents,
                history_mask=history_mask,
                chunk_mask=chunk_mask,
                cfg_scale=cfg_scale,
            )
            sample = sample - dt * velocity

            if chunk_mask is not None:
                sample = sample * chunk_mask.unsqueeze(-1).to(sample.dtype)

        return sample


DiT = PatchDiT
FlowMatchingModel = ConditionalFlowMatching


__all__ = [
    "DiTConfig",
    "FlowMatchingConfig",
    "PatchDiT",
    "ConditionalFlowMatching",
    "DiT",
    "FlowMatchingModel",
]
