from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base_lm import MiniMindConfig, MiniMindModel
from model.flow.dit import ConditionalFlowMatching, DiTConfig, FlowMatchingConfig, PatchDiT


@dataclass
class LMTTSLosses:
    """Main training losses returned by LMTTSModel.forward()."""

    loss: torch.Tensor
    lm_loss: torch.Tensor
    dit_loss: torch.Tensor
    patch_lm_loss: torch.Tensor
    stop_head_loss: torch.Tensor
    weighted_patch_lm_loss: torch.Tensor
    weighted_stop_loss: torch.Tensor
    weighted_moe_aux_loss: torch.Tensor
    diff_loss: torch.Tensor
    stop_loss: torch.Tensor
    moe_aux_loss: torch.Tensor


class PatchAudioEncoder(nn.Module):
    """
    Compress one fixed-length fine latent patch into one LM-step embedding.

    Input patch shape:
        [B, N_patch, patch_size, latent_dim]

    Output embedding shape:
        [B, N_patch, hidden_size]

    This is the "audio tokenizer" used by the LM, but it stays continuous:
    every 25-frame fine patch becomes one LM step embedding x_j.
    """

    def __init__(self, latent_dim: int, patch_size: int, hidden_size: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        num_heads = self._pick_num_heads(hidden_size)
        self.frame_proj = nn.Linear(latent_dim, hidden_size)
        self.summary_token = nn.Parameter(torch.randn(1, 1, 1, hidden_size) * 0.02)
        self.frame_pos = nn.Parameter(torch.zeros(1, 1, patch_size + 1, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _pick_num_heads(self, hidden_size: int) -> int:
        for num_heads in (8, 4, 2, 1):
            if hidden_size % num_heads == 0:
                return num_heads
        raise ValueError(f"hidden_size={hidden_size} must be divisible by at least one of [8, 4, 2, 1]")

    def forward(self, patches: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError("patches must have shape [B, N, patch_size, D]")
        if patches.shape[2] != self.patch_size:
            raise ValueError(f"Expected patch_size={self.patch_size}, got {patches.shape[2]}")
        if patches.shape[3] != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {patches.shape[3]}")

        batch_size, num_patches, _, _ = patches.shape
        frame_tokens = self.frame_proj(patches)
        # A learnable summary token lets the encoder pool the whole patch with
        # attention instead of relying on plain mean pooling.
        summary_tokens = self.summary_token.expand(batch_size, num_patches, -1, -1)
        token_seq = torch.cat([summary_tokens, frame_tokens], dim=2)
        token_seq = token_seq + self.frame_pos[:, :, : token_seq.shape[2], :]

        if patch_mask is not None:
            if patch_mask.shape != patches.shape[:3]:
                raise ValueError(f"patch_mask must have shape {patches.shape[:3]}")
            frame_mask = patch_mask.to(device=frame_tokens.device, dtype=torch.bool)
        else:
            frame_mask = torch.ones(patches.shape[:3], device=frame_tokens.device, dtype=torch.bool)

        summary_mask = torch.ones(batch_size, num_patches, 1, device=frame_tokens.device, dtype=torch.bool)
        token_mask = torch.cat([summary_mask, frame_mask], dim=2)

        flat_tokens = token_seq.view(batch_size * num_patches, self.patch_size + 1, self.hidden_size)
        flat_mask = token_mask.view(batch_size * num_patches, self.patch_size + 1)
        encoded = self.encoder(
            flat_tokens,
            src_key_padding_mask=~flat_mask,
        )
        # The first output position corresponds to the summary token we prepended,
        # so it becomes the patch-level embedding consumed by the LM.
        pooled = self.output_norm(encoded[:, 0]).view(batch_size, num_patches, self.hidden_size)
        pooled = pooled * frame_mask.any(dim=-1, keepdim=True).to(pooled.dtype)

        return self.out_proj(pooled)


CoarseAudioEncoder = PatchAudioEncoder


class LMTTSModel(nn.Module):
    """
    Patch-level autoregressive LM + slot-conditioned DiT decoder.

    High-level data flow:
    1. [patch_size, latent_dim] fine patch -> patch embedding x_j
    2. BaseLM autoregresses over text + prompt patch embeddings + previous patch embeddings
    3. The LM predictor state h_j is expanded into a few condition slots
    4. DiT cross-attends to those slots and decodes the current fine patch
    5. In inference, the predicted patch is re-encoded into x_hat_j and fed back to the LM
    """

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        latent_rate: int,
        patch_size: int,
        cond_tokens_per_patch: int,
        lm_config: MiniMindConfig,
        dit_config: DiTConfig,
        flow_config: FlowMatchingConfig | None = None,
        patch_lm_loss_weight: float = 1.0,
        stop_loss_weight: float = 1.0,
        moe_aux_loss_weight: float = 1.0,
        stop_class_weights: tuple[float, float] | list[float] = (1.0, 20.0),
    ) -> None:
        super().__init__()
        if latent_rate <= 0:
            raise ValueError("latent_rate must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if cond_tokens_per_patch <= 0:
            raise ValueError("cond_tokens_per_patch must be positive")

        if lm_config.vocab_size != vocab_size:
            lm_config.vocab_size = vocab_size

        self.latent_dim = latent_dim
        self.latent_rate = latent_rate
        self.patch_size = patch_size
        self.cond_tokens_per_patch = cond_tokens_per_patch
        self.max_chunk_size = patch_size
        self.patch_lm_loss_weight = patch_lm_loss_weight
        self.stop_loss_weight = stop_loss_weight
        self.moe_aux_loss_weight = moe_aux_loss_weight
        if len(stop_class_weights) != 2:
            raise ValueError("stop_class_weights must contain exactly two values: [continue_weight, stop_weight].")

        if dit_config.latent_dim != latent_dim:
            dit_config.latent_dim = latent_dim
        dit_config.max_chunk_size = patch_size
        dit_config.cond_token_dim = lm_config.hidden_size

        self.base_lm = MiniMindModel(lm_config)
        self.hidden_size = lm_config.hidden_size
        self.patch_encoder = PatchAudioEncoder(
            latent_dim=latent_dim,
            patch_size=patch_size,
            hidden_size=self.hidden_size,
        )

        # AUDIO_BOS marks the handoff from text steps to audio patch steps.
        self.audio_bos = nn.Parameter(torch.randn(1, self.hidden_size) * 0.02)
        # One LM patch hidden expands into a small set of slot tokens that act
        # as DiT cross-attention memory for the current patch.
        self.cond_slot_proj = nn.Linear(self.hidden_size, cond_tokens_per_patch * self.hidden_size)
        self.cond_slot_embed = nn.Parameter(torch.randn(1, cond_tokens_per_patch, self.hidden_size) * 0.02)
        self.cond_type_embed = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.stop_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_act = nn.SiLU()
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=False)
        # The stop signal is extremely rare (1 per sample vs ~29 continue labels),
        # so we up-weight class 1 to prevent the model from always predicting "continue".
        stop_class_weight = torch.tensor(stop_class_weights, dtype=torch.float32)
        self.stop_loss_fn = nn.CrossEntropyLoss(weight=stop_class_weight, reduction="none")

        self.dit = PatchDiT(dit_config)
        self.flow = ConditionalFlowMatching(self.dit, flow_config)

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing across both LM and DiT trunks."""
        self.base_lm.enable_gradient_checkpointing()
        self.dit.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.base_lm.disable_gradient_checkpointing()
        self.dit.disable_gradient_checkpointing()

    def _split_into_patches(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split a [T, D] fine latent sequence into fixed-length patches.

        The last patch is right-padded with zeros when T is not divisible by patch_size.
        """
        if frames.dim() != 2:
            raise ValueError("frames must have shape [T, D]")

        total_frames = frames.shape[0]
        if total_frames == 0:
            empty_patches = frames.new_zeros(0, self.patch_size, self.latent_dim)
            empty_mask = torch.zeros(0, self.patch_size, device=frames.device, dtype=torch.bool)
            return empty_patches, empty_mask

        num_patches = math.ceil(total_frames / self.patch_size)
        padded_frames = frames.new_zeros(num_patches * self.patch_size, self.latent_dim)
        padded_frames[:total_frames] = frames
        patches = padded_frames.view(num_patches, self.patch_size, self.latent_dim)

        patch_mask = torch.zeros(num_patches * self.patch_size, device=frames.device, dtype=torch.bool)
        patch_mask[:total_frames] = True
        patch_mask = patch_mask.view(num_patches, self.patch_size)
        return patches, patch_mask

    def _hidden_to_cond_tokens(self, patch_hidden: torch.Tensor) -> torch.Tensor:
        """
        Expand one LM predictor hidden into several ordered DiT condition slots.

        This keeps the LM sequence short (1 patch = 1 LM step) while still giving
        the DiT a small condition sequence to cross-attend to.
        """
        if patch_hidden.dim() == 2:
            patch_hidden = patch_hidden.unsqueeze(1)
        if patch_hidden.dim() != 3:
            raise ValueError("patch_hidden must have shape [B, N, H] or [B, H]")
        batch_size, num_steps, hidden_dim = patch_hidden.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {hidden_dim}")

        cond_tokens = self.cond_slot_proj(patch_hidden)
        cond_tokens = cond_tokens.view(batch_size, num_steps, self.cond_tokens_per_patch, self.hidden_size)
        cond_tokens = cond_tokens + self.cond_slot_embed.unsqueeze(1) + self.cond_type_embed.unsqueeze(1)
        return cond_tokens

    def _pad_embedding_list(self, tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a batch of variable-length [T, H] embedding sequences."""
        max_len = max((tensor.shape[0] for tensor in tensors), default=0)
        batch_size = len(tensors)
        padded = self.audio_bos.new_zeros(batch_size, max_len, self.hidden_size)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.audio_bos.device)

        for batch_idx, tensor in enumerate(tensors):
            seq_len = tensor.shape[0]
            if seq_len == 0:
                continue
            padded[batch_idx, :seq_len] = tensor.to(device=padded.device, dtype=padded.dtype)
            mask[batch_idx, :seq_len] = True
        return padded, mask

    def _prepare_audio_segments(
        self,
        latents: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert packed [prompt | target | pad] fine latents into patch embeddings.

        Prompt and target are patchified separately on purpose so we never mix
        prompt frames and target frames inside the same LM patch.
        """
        batch_size, _, latent_dim = latents.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")

        prompt_mask = prompt_mask.to(dtype=torch.bool, device=latents.device)
        target_mask = target_mask.to(dtype=torch.bool, device=latents.device)
        padding_mask = padding_mask.to(dtype=torch.bool, device=latents.device)

        sample_infos: list[dict] = []
        prompt_patch_list: list[torch.Tensor] = []
        target_patch_list: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            valid_len = int(padding_mask[batch_idx].sum().item())
            prompt_len = int(prompt_mask[batch_idx].sum().item())
            target_len = int(target_mask[batch_idx].sum().item())
            if valid_len < prompt_len + target_len:
                raise ValueError("padding_mask must cover prompt + target frames")
            if target_len <= 0:
                raise ValueError("Each sample must contain at least one target frame")

            prompt_frames = latents[batch_idx, :prompt_len]
            target_start = prompt_len
            target_end = prompt_len + target_len
            target_frames = latents[batch_idx, target_start:target_end]

            prompt_patches, prompt_patch_mask = self._split_into_patches(prompt_frames)
            target_patches, target_patch_mask = self._split_into_patches(target_frames)

            if prompt_patches.shape[0] > 0:
                prompt_patch_embeds = self.patch_encoder(
                    prompt_patches.unsqueeze(0),
                    prompt_patch_mask.unsqueeze(0),
                ).squeeze(0)
            else:
                prompt_patch_embeds = self.audio_bos.new_zeros(0, self.hidden_size)

            target_patch_embeds = self.patch_encoder(
                target_patches.unsqueeze(0),
                target_patch_mask.unsqueeze(0),
            ).squeeze(0)

            prompt_patch_list.append(prompt_patch_embeds)
            target_patch_list.append(target_patch_embeds)
            sample_infos.append(
                {
                    "prompt_frames": prompt_frames,
                    "target_frames": target_frames,
                    "prompt_patch_steps": prompt_patch_embeds.shape[0],
                    "target_patch_steps": target_patch_embeds.shape[0],
                }
            )

        prompt_patch_padded, prompt_patch_mask = self._pad_embedding_list(prompt_patch_list)
        target_patch_padded, target_patch_mask = self._pad_embedding_list(target_patch_list)
        return sample_infos, prompt_patch_padded, prompt_patch_mask, target_patch_padded, target_patch_mask

    def _build_lm_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_patches: torch.Tensor,
        prompt_patch_mask: torch.Tensor,
        target_patches: torch.Tensor,
        target_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the mixed LM sequence:
            [text tokens] + [AUDIO_BOS] + [prompt patch embeds] + [target patch embeds]

        target_positions stores where the target patch steps live inside that mixed
        sequence so we can gather the right predictor states later.
        """
        batch_size = input_ids.shape[0]
        attention_mask = attention_mask.to(device=input_ids.device)

        sequences = []
        seq_masks = []
        target_positions = []
        max_target_steps = target_patches.shape[1]

        for batch_idx in range(batch_size):
            text_len = int(attention_mask[batch_idx].sum().item())
            text_embed = self.base_lm.embed_tokens(input_ids[batch_idx, :text_len])
            bos = self.audio_bos

            prompt_steps = int(prompt_patch_mask[batch_idx].sum().item())
            target_steps = int(target_patch_mask[batch_idx].sum().item())
            prompt_embed = prompt_patches[batch_idx, :prompt_steps]
            target_embed = target_patches[batch_idx, :target_steps]

            sequence = torch.cat([text_embed, bos, prompt_embed, target_embed], dim=0)
            sequence_mask = torch.ones(sequence.shape[0], device=input_ids.device, dtype=torch.bool)
            sequences.append(sequence)
            seq_masks.append(sequence_mask)

            seq_target_positions = torch.zeros(max_target_steps, device=input_ids.device, dtype=torch.long)
            if target_steps > 0:
                # First target patch step comes strictly after text + BOS + prompt patches.
                start = text_len + 1 + prompt_steps
                seq_target_positions[:target_steps] = torch.arange(target_steps, device=input_ids.device) + start
            target_positions.append(seq_target_positions)

        max_seq_len = max(sequence.shape[0] for sequence in sequences)
        inputs_embeds = self.audio_bos.new_zeros(batch_size, max_seq_len, self.hidden_size)
        lm_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=input_ids.device)
        padded_target_positions = torch.stack(target_positions, dim=0)

        for batch_idx, (sequence, sequence_mask) in enumerate(zip(sequences, seq_masks)):
            seq_len = sequence.shape[0]
            inputs_embeds[batch_idx, :seq_len] = sequence
            lm_attention_mask[batch_idx, :seq_len] = sequence_mask

        return inputs_embeds, lm_attention_mask, padded_target_positions

    def _gather_target_hidden(
        self,
        shifted_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather the predictor state for each target patch step.

        shifted_hidden is the LM output shifted right by one position, so the
        hidden gathered for target patch j only depends on the prefix before it.
        """
        gather_index = target_positions.unsqueeze(-1).expand(-1, -1, shifted_hidden.shape[-1])
        gathered = shifted_hidden.gather(dim=1, index=gather_index)
        return gathered * target_mask.unsqueeze(-1).to(gathered.dtype)

    def _build_stop_labels(self, target_patch_mask: torch.Tensor) -> torch.Tensor:
        """Mark only the last valid target patch in each sample as stop=1."""
        stop_labels = torch.zeros_like(target_patch_mask, dtype=torch.long)
        for batch_idx in range(target_patch_mask.shape[0]):
            valid_steps = int(target_patch_mask[batch_idx].sum().item())
            if valid_steps > 0:
                stop_labels[batch_idx, valid_steps - 1] = 1
        return stop_labels

    def _build_chunk_batch(
        self,
        sample_infos: list[dict],
        target_patch_cond_tokens: torch.Tensor,
        target_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flatten all target patches across the batch into one DiT training batch.

        Each returned row corresponds to one current patch to decode, plus:
        - cond_token_chunks: LM-derived condition slots for that patch
        - history_chunks: a short fine-latent prefix right before the patch
        - patch_masks/history_masks: valid positions inside padded tensors
        """
        target_patches = []
        cond_token_chunks = []
        history_chunks = []
        history_masks = []
        patch_masks = []

        for batch_idx, sample in enumerate(sample_infos):
            target_frames = sample["target_frames"]
            target_patch_steps = int(target_patch_mask[batch_idx].sum().item())

            for patch_idx in range(target_patch_steps):
                patch_start = patch_idx * self.patch_size
                patch_end = min(patch_start + self.patch_size, target_frames.shape[0])
                patch = target_frames[patch_start:patch_end]
                patch_valid_len = patch.shape[0]

                padded_patch = target_frames.new_zeros(self.patch_size, self.latent_dim)
                padded_patch[:patch_valid_len] = patch
                patch_mask = torch.zeros(self.patch_size, dtype=torch.bool, device=target_frames.device)
                patch_mask[:patch_valid_len] = True

                cond_tokens = target_patch_cond_tokens[batch_idx, patch_idx]

                # Use the immediately preceding fine frames as the local acoustic
                # history that helps the current patch connect smoothly.
                if patch_idx == 0:
                    prompt_frames = sample["prompt_frames"]
                    history = prompt_frames[-self.patch_size :]
                else:
                    history_start = max(0, patch_start - self.patch_size)
                    history = target_frames[history_start:patch_start]
                history_valid_len = history.shape[0]

                padded_history = target_frames.new_zeros(self.patch_size, self.latent_dim)
                padded_history[:history_valid_len] = history
                history_mask = torch.zeros(self.patch_size, dtype=torch.bool, device=target_frames.device)
                history_mask[:history_valid_len] = True

                target_patches.append(padded_patch)
                cond_token_chunks.append(cond_tokens)
                history_chunks.append(padded_history)
                history_masks.append(history_mask)
                patch_masks.append(patch_mask)

        return (
            torch.stack(target_patches, dim=0),
            torch.stack(cond_token_chunks, dim=0),
            torch.stack(history_chunks, dim=0),
            torch.stack(history_masks, dim=0),
            torch.stack(patch_masks, dim=0),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        moe_aux_scale: float = 1.0,
    ) -> LMTTSLosses:
        sample_infos, prompt_patches, prompt_patch_mask, target_patches, target_patch_mask = self._prepare_audio_segments(
            latents=latents,
            prompt_mask=prompt_mask,
            target_mask=target_mask,
            padding_mask=padding_mask,
        )

        inputs_embeds, lm_attention_mask, target_positions = self._build_lm_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_patches=prompt_patches,
            prompt_patch_mask=prompt_patch_mask,
            target_patches=target_patches,
            target_patch_mask=target_patch_mask,
        )

        hidden_states, _, moe_aux_loss = self.base_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=lm_attention_mask,
        )

        # Standard autoregressive alignment: the predictor state for target patch j
        # must come from the position right before target patch j in the mixed sequence.
        shifted_hidden = torch.cat(
            [torch.zeros_like(hidden_states[:, :1, :]), hidden_states[:, :-1, :]],
            dim=1,
        )
        target_patch_hidden = self._gather_target_hidden(
            shifted_hidden=shifted_hidden,
            target_positions=target_positions,
            target_mask=target_patch_mask,
        )

        patch_weight = target_patch_mask.to(dtype=hidden_states.dtype)

        # patch_lm_loss removed — DiT gradients drive LM learning directly
        patch_lm_loss = torch.tensor(0.0, device=hidden_states.device)

        stop_logits = self.stop_head(self.stop_act(self.stop_proj(target_patch_hidden)))
        stop_labels = self._build_stop_labels(target_patch_mask)
        stop_loss_per_step = self.stop_loss_fn(
            stop_logits.view(-1, stop_logits.shape[-1]),
            stop_labels.view(-1),
        ).view_as(stop_labels)
        stop_loss = (stop_loss_per_step * patch_weight).sum() / patch_weight.sum().clamp_min(1.0)

        # The same LM predictor state serves two roles:
        # 1) stop prediction
        # 2) condition slots for the DiT patch decoder
        target_patch_cond_tokens = self._hidden_to_cond_tokens(target_patch_hidden)
        fine_chunks, cond_token_chunks, history_chunks, history_masks, chunk_masks = self._build_chunk_batch(
            sample_infos=sample_infos,
            target_patch_cond_tokens=target_patch_cond_tokens,
            target_patch_mask=target_patch_mask,
        )
        diff_loss = self.flow.compute_loss(
            target_chunk=fine_chunks,
            cond_tokens=cond_token_chunks,
            history_fine_latents=history_chunks,
            history_mask=history_masks,
            chunk_mask=chunk_masks,
        )

        weighted_patch_lm_loss = self.patch_lm_loss_weight * patch_lm_loss
        weighted_stop_loss = self.stop_loss_weight * stop_loss
        weighted_moe_aux_loss = moe_aux_scale * self.moe_aux_loss_weight * moe_aux_loss
        lm_loss = weighted_patch_lm_loss + weighted_stop_loss + weighted_moe_aux_loss

        total_loss = diff_loss + lm_loss

        return LMTTSLosses(
            loss=total_loss,
            lm_loss=lm_loss.detach(),
            dit_loss=diff_loss.detach(),
            patch_lm_loss=patch_lm_loss.detach(),
            stop_head_loss=stop_loss.detach(),
            weighted_patch_lm_loss=weighted_patch_lm_loss.detach(),
            weighted_stop_loss=weighted_stop_loss.detach(),
            weighted_moe_aux_loss=weighted_moe_aux_loss.detach(),
            diff_loss=diff_loss.detach(),
            stop_loss=stop_loss.detach(),
            moe_aux_loss=moe_aux_loss.detach(),
        )

    @torch.no_grad()
    def generate_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_latents: torch.Tensor,
        max_target_patches: int = 50,
        min_target_patches: int = 1,
        num_flow_steps: int = 16,
        cfg_scale: float = 1.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Non-streaming autoregressive decoding.

        We generate one fine patch at a time:
        LM predictor hidden -> slot tokens -> DiT patch -> patch encoder -> LM.
        """
        if input_ids.shape[0] != 1:
            raise NotImplementedError("generate_latents currently supports batch_size=1 only")
        if prompt_latents.dim() != 3:
            raise ValueError("prompt_latents must have shape [1, T_prompt, D]")

        prompt_frames = prompt_latents[0]
        prompt_patches, prompt_patch_mask = self._split_into_patches(prompt_frames)
        if prompt_patches.shape[0] > 0:
            prompt_patch_embeds = self.patch_encoder(
                prompt_patches.unsqueeze(0),
                prompt_patch_mask.unsqueeze(0),
            )
        else:
            prompt_patch_embeds = self.audio_bos.new_zeros(1, 0, self.hidden_size)

        text_len = int(attention_mask[0].sum().item())
        text_embed = self.base_lm.embed_tokens(input_ids[0, :text_len]).unsqueeze(0)
        bos = self.audio_bos.unsqueeze(0)
        prefix_embeds = torch.cat([text_embed, bos, prompt_patch_embeds], dim=1)
        prefix_mask = torch.ones(prefix_embeds.shape[:2], device=prefix_embeds.device, dtype=torch.long)

        hidden_states, past_key_values, _ = self.base_lm(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            use_cache=True,
        )
        # The last hidden after text + BOS + prompt is the predictor state for
        # the first generated patch.
        lm_hidden = hidden_states[:, -1, :]

        generated_patches = []
        generated_fine = prompt_frames.new_zeros(0, self.latent_dim)

        for patch_idx in range(max_target_patches):
            predictor_hidden = lm_hidden
            stop_flag = self.stop_head(self.stop_act(self.stop_proj(predictor_hidden))).argmax(dim=-1).item() == 1
            should_stop = stop_flag and (patch_idx + 1) >= min_target_patches

            cond_tokens = self._hidden_to_cond_tokens(predictor_hidden).squeeze(1)
            # In v1 inference we decode a full patch each step, so every frame in
            # the current patch is valid.
            chunk_mask = torch.ones(1, self.patch_size, device=prefix_embeds.device, dtype=torch.bool)

            if patch_idx == 0:
                history = prompt_frames[-self.patch_size :]
            else:
                history = generated_fine[-self.patch_size :]
            history_padded = prompt_frames.new_zeros(1, self.patch_size, self.latent_dim)
            history_mask = torch.zeros(1, self.patch_size, device=prefix_embeds.device, dtype=torch.bool)
            if history.shape[0] > 0:
                history_padded[0, : history.shape[0]] = history
                history_mask[0, : history.shape[0]] = True

            pred_patch = self.flow.sample(
                cond_tokens=cond_tokens,
                history_fine_latents=history_padded,
                history_mask=history_mask,
                chunk_mask=chunk_mask,
                num_steps=num_flow_steps,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )[0]
            generated_patches.append(pred_patch)
            generated_fine = torch.cat([generated_fine, pred_patch], dim=0)

            # Close the AR loop: the newly generated patch becomes the next LM step embedding.
            pred_patch_embed = self.patch_encoder(
                pred_patch.unsqueeze(0).unsqueeze(0),
                chunk_mask.unsqueeze(1),
            ).squeeze(1)
            next_hidden, past_key_values, _ = self.base_lm(
                inputs_embeds=pred_patch_embed.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True,
            )
            lm_hidden = next_hidden[:, -1, :]

            if should_stop:
                break

        if not generated_patches:
            return prompt_frames.new_zeros(1, 0, self.latent_dim)

        return torch.cat(generated_patches, dim=0).unsqueeze(0)
