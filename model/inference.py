from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from model.lm_tts import LMTTSModel
from model.modules.vae import load_vae, vae_decode


@dataclass
class InferenceExample:
    """One inference sample together with optional decoded waveforms."""

    sample_idx: int
    text: str
    prompt_latent: torch.Tensor
    pred_latent: torch.Tensor
    target_latent: torch.Tensor | None = None
    pred_waveform: torch.Tensor | None = None
    gt_waveform: torch.Tensor | None = None

    @property
    def prompt_frames(self) -> int:
        return int(self.prompt_latent.shape[1])

    @property
    def pred_frames(self) -> int:
        return int(self.pred_latent.shape[1])

    @property
    def target_frames(self) -> int:
        if self.target_latent is None:
            return 0
        return int(self.target_latent.shape[1])


def extract_masked_latents(latents_row: torch.Tensor, mask_row: torch.Tensor) -> torch.Tensor:
    """Remove padded frames from one packed latent row."""
    mask_row = mask_row.to(dtype=torch.bool, device=latents_row.device)
    return latents_row[mask_row]


def decode_text_tokens(tokenizer: Any, input_ids_row: torch.Tensor, attention_mask_row: torch.Tensor) -> str:
    """Recover the visible text string from one padded token row."""
    valid_len = int(attention_mask_row.sum().item())
    return tokenizer.decode(input_ids_row[:valid_len].tolist(), skip_special_tokens=True)


def waveform_to_wandb_array(waveform: torch.Tensor):
    """Convert a [C, T] tensor into the ndarray format wandb.Audio expects."""
    waveform = waveform.detach().float().cpu()
    if waveform.dim() != 2:
        raise ValueError("Expected waveform shape [C, T]")
    if waveform.shape[0] == 1:
        return waveform[0].numpy()
    return waveform.transpose(0, 1).numpy()


def _write_waveform_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a floating-point waveform tensor as a PCM16 wav file."""
    waveform = waveform.detach().float().cpu().clamp(-1.0, 1.0)
    if waveform.dim() != 2:
        raise ValueError("Expected waveform shape [C, T]")

    channels, num_samples = waveform.shape
    pcm = (waveform.transpose(0, 1) * 32767.0).round().to(torch.int16).numpy()

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


@torch.no_grad()
def run_autoregressive_inference(
    model: LMTTSModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_latents: torch.Tensor,
    inference_cfg: dict,
) -> torch.Tensor:
    """Thin wrapper around model.generate_latents using YAML-style inference config."""
    max_target_patches = int(
        inference_cfg.get("max_target_patches", inference_cfg.get("max_target_coarse_steps", 50))
    )
    min_target_patches = int(
        inference_cfg.get("min_target_patches", inference_cfg.get("min_target_coarse_steps", 1))
    )
    return model.generate_latents(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_latents=prompt_latents,
        max_target_patches=max_target_patches,
        min_target_patches=min_target_patches,
        num_flow_steps=int(inference_cfg.get("num_flow_steps", 16)),
        cfg_scale=float(inference_cfg.get("cfg_scale", 1.0)),
        temperature=float(inference_cfg.get("temperature", 1.0)),
    )


@torch.no_grad()
def build_inference_examples(
    model: LMTTSModel,
    batch: dict,
    tokenizer: Any,
    inference_cfg: dict,
    num_samples: int | None = None,
) -> list[InferenceExample]:
    """
    Build a few prompt/pred/target examples from a training batch.

    This is the common path used by both periodic training-time inference and
    future standalone inference entrypoints.
    """
    batch_size = int(batch["latent"].shape[0])
    if num_samples is None:
        num_samples = int(inference_cfg.get("num_samples", 1))
    num_samples = min(num_samples, batch_size)

    examples: list[InferenceExample] = []
    for sample_idx in range(num_samples):
        prompt_latent = extract_masked_latents(
            batch["latent"][sample_idx],
            batch["prompt_mask"][sample_idx],
        ).unsqueeze(0)
        target_latent = extract_masked_latents(
            batch["latent"][sample_idx],
            batch["target_mask"][sample_idx],
        ).unsqueeze(0)
        pred_latent = run_autoregressive_inference(
            model=model,
            input_ids=batch["input_ids"][sample_idx : sample_idx + 1],
            attention_mask=batch["attention_mask"][sample_idx : sample_idx + 1],
            prompt_latents=prompt_latent,
            inference_cfg=inference_cfg,
        )
        text = decode_text_tokens(
            tokenizer,
            batch["input_ids"][sample_idx],
            batch["attention_mask"][sample_idx],
        )
        examples.append(
            InferenceExample(
                sample_idx=sample_idx,
                text=text,
                prompt_latent=prompt_latent,
                pred_latent=pred_latent,
                target_latent=target_latent,
            )
        )
    return examples


@torch.no_grad()
def attach_decoded_waveforms(
    examples: list[InferenceExample],
    vae_path: str,
    device: torch.device,
    precision: str = "fp16",
) -> list[InferenceExample]:
    """Decode only the generated target segment into audio with the VAE."""
    if not examples:
        return examples

    vae = load_vae(vae_path, device=str(device), precision=precision)
    try:
        for example in examples:
            example.pred_waveform = vae_decode(vae, example.pred_latent)[0]
    finally:
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return examples


def save_inference_examples(
    examples: list[InferenceExample],
    output_dir: str | Path,
    step: int,
    sample_rate: int = 48000,
) -> Path:
    """Persist latent predictions, optional wavs, and metadata for one inference step."""
    step_dir = Path(output_dir) / "infer" / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for example in examples:
        sample_prefix = step_dir / f"sample_{example.sample_idx:02d}"
        torch.save(example.prompt_latent.cpu(), sample_prefix.with_name(sample_prefix.name + "_prompt_latent.pt"))
        torch.save(example.pred_latent.cpu(), sample_prefix.with_name(sample_prefix.name + "_pred_latent.pt"))
        if example.target_latent is not None:
            torch.save(example.target_latent.cpu(), sample_prefix.with_name(sample_prefix.name + "_target_latent.pt"))

        if example.pred_waveform is not None:
            _write_waveform_wav(
                sample_prefix.with_name(sample_prefix.name + "_pred.wav"),
                example.pred_waveform,
                sample_rate=sample_rate,
            )
        if example.gt_waveform is not None:
            _write_waveform_wav(
                sample_prefix.with_name(sample_prefix.name + "_gt.wav"),
                example.gt_waveform,
                sample_rate=sample_rate,
            )

        metadata.append(
            {
                "sample_idx": example.sample_idx,
                "text": example.text,
                "prompt_frames": example.prompt_frames,
                "pred_frames": example.pred_frames,
                "target_frames": example.target_frames,
                "has_pred_waveform": example.pred_waveform is not None,
                "has_gt_waveform": example.gt_waveform is not None,
            }
        )

    (step_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return step_dir


__all__ = [
    "InferenceExample",
    "attach_decoded_waveforms",
    "build_inference_examples",
    "decode_text_tokens",
    "extract_masked_latents",
    "run_autoregressive_inference",
    "save_inference_examples",
    "waveform_to_wandb_array",
]
