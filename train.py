from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import warnings
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.char_tokenizer import CharTokenizer
from data.dataset import TTSDataset, TTSDatasetLoRA, collate_fn
from model.backbones.base_lm import MiniMindConfig
from model.flow.dit import DiTConfig, FlowMatchingConfig
from model.inference import (
    attach_decoded_waveforms,
    build_inference_examples,
    save_inference_examples,
    waveform_to_wandb_array,
)
from model.lm_tts import LMTTSModel

try:
    import wandb
except ImportError:
    wandb = None

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


warnings.filterwarnings("ignore", module="phonemizer")
logging.getLogger("phonemizer").setLevel(logging.ERROR)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(path: str | Path, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(explicit_device: str | None = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def build_dataset(cfg: dict, data_root: str):
    """Create the training dataset variant selected by the config."""
    data_cfg = cfg.get("data", {})
    audio_cfg = cfg["audio"]
    dataset_type = data_cfg.get("dataset_type", "default")

    kwargs = dict(
        data_root=data_root,
        latent_rate=audio_cfg["latent_rate"],
        min_duration_sec=audio_cfg["min_duration_sec"],
        max_duration_sec=audio_cfg["max_duration_sec"],
        prompt_ratio_min=audio_cfg["prompt_ratio_min"],
        prompt_ratio_max=audio_cfg["prompt_ratio_max"],
    )

    if dataset_type == "lora":
        return TTSDatasetLoRA(
            language=data_cfg.get("language", "JA"),
            **kwargs,
        )
    return TTSDataset(**kwargs)


def resolve_vocab_path(cfg: dict, args: argparse.Namespace, data_root: str) -> Path:
    if args.vocab is not None:
        return Path(args.vocab)
    vocab_path = cfg.get("data", {}).get("vocab_path")
    if vocab_path:
        return Path(vocab_path)
    return Path(data_root) / "char_vocab.json"


def build_tokenizer(
    cfg: dict,
    dataset,
    data_root: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[CharTokenizer, Path]:
    """Load an existing char vocab or build one from dataset samples as a fallback."""
    vocab_path = resolve_vocab_path(cfg, args, data_root)

    if vocab_path.exists():
        tokenizer = CharTokenizer.load(vocab_path)
        print(f"Loaded char vocab from {vocab_path} ({tokenizer.vocab_size} tokens)")
    else:
        print(f"Vocab not found at {vocab_path}, building from dataset...")
        tokenizer = CharTokenizer.build_from_dataset_samples(dataset.samples)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(vocab_path)
        print(f"Built and saved char vocab to {vocab_path} ({tokenizer.vocab_size} tokens)")

    output_vocab_path = output_dir / "char_vocab.json"
    tokenizer.save(output_vocab_path)
    return tokenizer, output_vocab_path


def build_model(cfg: dict, latent_dim: int, vocab_size: int) -> LMTTSModel:
    """
    Build the patch-level LM-TTS model from YAML config.

    One LM step corresponds to one audio patch. The LM hidden for that patch is
    later expanded into DiT condition slots.
    """
    model_cfg = cfg["model"]
    audio_cfg = cfg["audio"]
    train_cfg = cfg["training"]
    if "patch_size" in model_cfg:
        patch_size = int(model_cfg["patch_size"])
    else:
        patch_size = int(model_cfg["coarse_span"]) * int(model_cfg["chunk_coarse_steps"])

    if "cond_tokens_per_patch" in model_cfg:
        cond_tokens_per_patch = int(model_cfg["cond_tokens_per_patch"])
    else:
        cond_tokens_per_patch = int(model_cfg.get("chunk_coarse_steps", 5))

    config_latent_dim = model_cfg.get("latent_dim")
    if config_latent_dim is not None and int(config_latent_dim) != latent_dim:
        raise ValueError(
            f"Config latent_dim={config_latent_dim} does not match dataset latent_dim={latent_dim}"
        )

    lm_config = MiniMindConfig(
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_layers"],
        vocab_size=vocab_size,
        num_attention_heads=model_cfg["num_heads"],
        num_key_value_heads=model_cfg["num_kv_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        dropout=model_cfg.get("dropout", 0.0),
        use_moe=model_cfg.get("use_moe", True),
        dense_prefix_layers=model_cfg.get("dense_prefix_layers", 4),
        dense_tail_layers=model_cfg.get("dense_tail_layers", 4),
        num_experts=model_cfg.get("num_experts", 8),
        num_experts_per_tok=model_cfg.get("num_experts_per_tok", 2),
        moe_intermediate_size=model_cfg.get("moe_intermediate_size"),
        router_aux_loss_coef=model_cfg.get("router_aux_loss_coef", 5e-4),
    )

    dit_config = DiTConfig(
        latent_dim=latent_dim,
        max_chunk_size=patch_size,
        cond_token_dim=model_cfg["hidden_size"],
        model_dim=model_cfg["dit_model_dim"],
        num_layers=model_cfg["dit_num_layers"],
        num_heads=model_cfg["dit_num_heads"],
        ff_mult=model_cfg.get("dit_ff_mult", 4),
        dropout=model_cfg.get("dit_dropout", 0.0),
    )

    flow_config = FlowMatchingConfig(
        sigma_min=model_cfg.get("sigma_min", 1e-5),
        cond_dropout_prob=model_cfg.get("cond_dropout_prob", 0.1),
        solver=model_cfg.get("flow_solver", "euler"),
    )

    return LMTTSModel(
        latent_dim=latent_dim,
        vocab_size=vocab_size,
        latent_rate=audio_cfg["latent_rate"],
        patch_size=patch_size,
        cond_tokens_per_patch=cond_tokens_per_patch,
        lm_config=lm_config,
        dit_config=dit_config,
        flow_config=flow_config,
        stop_loss_weight=train_cfg.get("stop_loss_weight", 1.0),
        moe_aux_loss_weight=train_cfg.get("moe_aux_loss_weight", 1.0),
    )


def build_optimizer(model: LMTTSModel, cfg: dict, device: torch.device) -> torch.optim.Optimizer:
    """Optimizer for the full end-to-end model.

    AdamW8bit is preferred on CUDA because optimizer states dominate memory for
    this model size. Fallback to vanilla AdamW when bitsandbytes is unavailable.
    """
    train_cfg = cfg["training"]
    optimizer_name = str(train_cfg.get("optimizer", "adamw8bit")).lower()
    lr = train_cfg["learning_rate"]
    betas = (
        train_cfg.get("adam_beta1", 0.9),
        train_cfg.get("adam_beta2", 0.95),
    )
    weight_decay = train_cfg.get("weight_decay", 1e-2)

    if optimizer_name == "adamw8bit":
        if device.type != "cuda":
            print(f"Warning: AdamW8bit requested but device is {device.type}; falling back to torch.optim.AdamW.")
        elif bnb is None:
            print("Warning: bitsandbytes is not installed; falling back to torch.optim.AdamW.")
        else:
            return bnb.optim.AdamW8bit(
                model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine decay schedule used by the main trainer."""
    train_cfg = cfg["training"]
    max_steps = int(train_cfg["max_steps"])
    warmup_steps = int(train_cfg.get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def init_wandb(cfg: dict, output_dir: Path, resume_id: str | None = None):
    """Initialize wandb only when enabled in the config and installed locally."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        print("Warning: wandb is enabled in config but the package is not installed. Continuing without wandb.")
        return None

    init_kwargs = {
        "project": wandb_cfg.get("project", "myvox_lm_tts"),
        "config": cfg,
        "dir": str(output_dir),
        "mode": wandb_cfg.get("mode", "offline"),
    }
    if wandb_cfg.get("name"):
        init_kwargs["name"] = wandb_cfg["name"]
    if wandb_cfg.get("tags"):
        init_kwargs["tags"] = list(wandb_cfg["tags"])
    if resume_id is not None:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "allow"

    try:
        run = wandb.init(**init_kwargs)
        print(f"W&B initialized: project={init_kwargs['project']} mode={init_kwargs['mode']}")
        return run
    except Exception as exc:
        print(f"Warning: failed to initialize wandb: {exc}")
        return None


@torch.no_grad()
def run_periodic_inference(
    model: LMTTSModel,
    batch: dict,
    tokenizer: CharTokenizer,
    cfg: dict,
    device: torch.device,
    global_step: int,
    output_dir: Path,
    wandb_run=None,
) -> None:
    """
    Periodically run non-streaming AR inference on samples from the current batch.

    This is intentionally simple: prompt -> full autoregressive decode -> optional
    VAE decode -> save artifacts / log to wandb.
    """
    inference_cfg = cfg.get("inference", {})
    if not inference_cfg.get("enabled", False):
        return

    num_samples = min(int(inference_cfg.get("num_samples", 1)), int(batch["latent"].shape[0]))
    if num_samples <= 0:
        return

    model_was_training = model.training
    model.eval()

    logs: dict[str, object] = {}

    try:
        examples = build_inference_examples(
            model=model,
            batch=batch,
            tokenizer=tokenizer,
            inference_cfg=inference_cfg,
            num_samples=num_samples,
        )

        # Audio decoding is optional because it requires a VAE checkpoint and is
        # much heavier than latent-only inspection.
        if inference_cfg.get("log_audio", True) and inference_cfg.get("vae_path"):
            examples = attach_decoded_waveforms(
                examples=examples,
                vae_path=inference_cfg["vae_path"],
                device=device,
                precision=inference_cfg.get("vae_precision", "fp16"),
            )

        save_outputs = bool(inference_cfg.get("save_outputs", True))
        if save_outputs:
            infer_dir = save_inference_examples(
                examples=examples,
                output_dir=output_dir,
                step=global_step,
                sample_rate=int(inference_cfg.get("sample_rate", 48000)),
            )
            print(f"[Step {global_step}] saved inference outputs to {infer_dir}")

        for example in examples:
            sample_idx = example.sample_idx
            print(
                f"[Step {global_step}] infer sample={sample_idx} "
                f"prompt_frames={example.prompt_frames} "
                f"target_frames={example.target_frames} "
                f"pred_frames={example.pred_frames}"
            )

            logs[f"infer/prompt_frames_{sample_idx}"] = example.prompt_frames
            logs[f"infer/target_frames_{sample_idx}"] = example.target_frames
            logs[f"infer/pred_frames_{sample_idx}"] = example.pred_frames

        if wandb_run is not None and wandb is not None:
            table = wandb.Table(columns=["step", "sample_idx", "text", "prompt_frames", "target_frames", "pred_frames"])
            for example in examples:
                table.add_data(
                    global_step,
                    example.sample_idx,
                    example.text,
                    example.prompt_frames,
                    example.target_frames,
                    example.pred_frames,
                )
            logs["infer/examples"] = table

        if wandb_run is not None and wandb is not None:
            sample_rate = int(inference_cfg.get("sample_rate", 48000))
            for example in examples:
                caption_prefix = f"step={global_step} sample={example.sample_idx}"
                if example.pred_waveform is not None:
                    logs[f"infer/pred_audio_{example.sample_idx}"] = wandb.Audio(
                        waveform_to_wandb_array(example.pred_waveform),
                        sample_rate=sample_rate,
                        caption=f"{caption_prefix} pred | {example.text}",
                    )
                if example.gt_waveform is not None:
                    logs[f"infer/gt_audio_{example.sample_idx}"] = wandb.Audio(
                        waveform_to_wandb_array(example.gt_waveform),
                        sample_rate=sample_rate,
                        caption=f"{caption_prefix} gt | {example.text}",
                    )

        if wandb_run is not None and logs:
            wandb_run.log(logs, step=global_step)
    except Exception as exc:
        print(f"[Step {global_step}] Inference failed: {exc}")
    finally:
        if model_was_training:
            model.train()


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    model: LMTTSModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    cfg: dict,
    vocab_path: Path,
    keep_last_k: int,
    wandb_run_id: str | None = None,
) -> Path:
    """Save a full training checkpoint and trim old step directories."""
    ckpt_dir = output_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    torch.save(
        {
            "global_step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "vocab_path": str(vocab_path),
            "wandb_run_id": wandb_run_id,
        },
        ckpt_path,
    )

    if keep_last_k > 0:
        step_dirs = []
        for path in output_dir.glob("step_*"):
            if not path.is_dir():
                continue
            try:
                step_value = int(path.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            step_dirs.append((step_value, path))
        step_dirs.sort(key=lambda item: item[0])
        while len(step_dirs) > keep_last_k:
            _, old_path = step_dirs.pop(0)
            shutil.rmtree(old_path, ignore_errors=True)

    return ckpt_path


def load_checkpoint(
    resume_path: str | Path,
    model: LMTTSModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, int, str | None]:
    """Restore model/training state from a previously saved checkpoint."""
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Loaded checkpoint with compatibility mode: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as exc:
        print(f"Warning: could not restore optimizer state: {exc}")

    try:
        scheduler.load_state_dict(ckpt["scheduler"])
    except Exception as exc:
        print(f"Warning: could not restore scheduler state: {exc}")

    try:
        scaler.load_state_dict(ckpt["scaler"])
    except Exception as exc:
        print(f"Warning: could not restore scaler state: {exc}")

    global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
    start_epoch = int(ckpt.get("epoch", 0))
    wandb_run_id = ckpt.get("wandb_run_id")
    return global_step, start_epoch, wandb_run_id


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move every tensor in the collated batch to the selected device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=(device.type == "cuda"))
        else:
            moved[key] = value
    return moved


def train(args: argparse.Namespace) -> None:
    """Main training entrypoint used by the CLI below."""
    cfg = load_config(args.config)
    device = pick_device(args.device)
    set_seed(int(cfg["training"].get("seed", 42)))

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir or cfg["training"].get("output_dir", "outputs/lm_tts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg["data"]["data_root"] = args.data_root
    cfg["training"]["output_dir"] = str(output_dir)
    if args.vocab is not None:
        cfg["data"]["vocab_path"] = args.vocab

    save_config(output_dir / "config_resolved.yaml", cfg)
    (output_dir / "cli_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset = build_dataset(cfg, args.data_root)
    if len(dataset) == 0:
        raise ValueError(f"No samples found under {args.data_root}")

    tokenizer, saved_vocab_path = build_tokenizer(cfg, dataset, args.data_root, output_dir, args)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg["training"]

    collate_with_tokenizer = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_text_len=data_cfg.get("max_text_len", 512),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_with_tokenizer,
        pin_memory=(device.type == "cuda"),
        drop_last=train_cfg.get("drop_last", True),
    )
    if len(dataloader) == 0:
        raise ValueError(
            "Dataloader is empty. Either reduce training.batch_size or set training.drop_last=false."
        )

    example = dataset[0]
    latent_dim = int(example["target_latent"].shape[-1])
    model = build_model(cfg, latent_dim=latent_dim, vocab_size=tokenizer.vocab_size).to(device)
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = build_optimizer(model, cfg, device=device)
    scheduler = build_scheduler(optimizer, cfg)

    amp_enabled = bool(train_cfg.get("fp16", True)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    autocast_context = partial(autocast, device_type="cuda", enabled=amp_enabled)

    total_params, trainable_params = count_parameters(model)
    print(f"Device: {device}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")
    print(f"Model parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
    print(
        f"Training: batch_size={train_cfg['batch_size']}, max_steps={train_cfg['max_steps']}, "
        f"lr={train_cfg['learning_rate']}, warmup={train_cfg.get('warmup_steps', 0)}"
    )
    print(
        f"Optimizer: {str(train_cfg.get('optimizer', 'adamw8bit')).lower()} | "
        f"gradient_checkpointing={gradient_checkpointing}"
    )

    global_step = 0
    start_epoch = 0
    wandb_run_id = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        global_step, start_epoch, wandb_run_id = load_checkpoint(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(f"Resumed at step {global_step}, epoch {start_epoch}")

    wandb_run = init_wandb(cfg, output_dir=output_dir, resume_id=wandb_run_id)

    log_every_steps = int(train_cfg.get("log_every_steps", 10))
    save_every_steps = int(train_cfg.get("save_every_steps", 1000))
    max_steps = int(train_cfg["max_steps"])
    max_epochs = int(train_cfg.get("epochs", 1_000_000))
    gradient_clip = float(train_cfg.get("gradient_clip", 1.0))
    keep_last_k = int(train_cfg.get("keep_last_k", 2))

    model.train()
    progress_bar = tqdm(total=max_steps, initial=global_step, desc="Training")
    for epoch in range(start_epoch, max_epochs):
        for batch in dataloader:
            if global_step >= max_steps:
                break

            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context() if amp_enabled else nullcontext():
                losses = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    latents=batch["latent"],
                    prompt_mask=batch["prompt_mask"],
                    target_mask=batch["target_mask"],
                    padding_mask=batch["padding_mask"],
                )

            # Mixed precision and full precision follow the same optimizer path;
            # the only difference is whether GradScaler is active.
            if scaler.is_enabled():
                scaler.scale(losses.loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1
            progress_bar.update(1)

            if global_step % log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                metrics = {
                    "train/loss": float(losses.loss.item()),
                    "train/diff_loss": float(losses.diff_loss.item()),
                    "train/stop_loss": float(losses.stop_loss.item()),
                    "train/moe_aux_loss": float(losses.moe_aux_loss.item()),
                    "train/lr": float(lr),
                    "train/epoch": float(epoch + 1),
                }
                message = (
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={metrics['train/loss']:.4f} "
                    f"diff={metrics['train/diff_loss']:.4f} "
                    f"stop={metrics['train/stop_loss']:.4f} "
                    f"moe_aux={metrics['train/moe_aux_loss']:.4f} "
                    f"lr={lr:.2e}"
                )
                progress_bar.set_postfix_str(message)
                print(message)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)

            infer_every_steps = int(cfg.get("inference", {}).get("every_steps", 0))
            if infer_every_steps > 0 and global_step % infer_every_steps == 0:
                run_periodic_inference(
                    model=model,
                    batch=batch,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    device=device,
                    global_step=global_step,
                    output_dir=output_dir,
                    wandb_run=wandb_run,
                )

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                ckpt_path = save_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cfg=cfg,
                    vocab_path=saved_vocab_path,
                    keep_last_k=keep_last_k,
                    wandb_run_id=(wandb_run.id if wandb_run is not None else None),
                )
                print(f"Saved checkpoint to {ckpt_path}")

        if global_step >= max_steps:
            break

    final_ckpt = save_checkpoint(
        output_dir=output_dir,
        step=global_step,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cfg=cfg,
        vocab_path=saved_vocab_path,
        keep_last_k=max(keep_last_k, 1),
        wandb_run_id=(wandb_run.id if wandb_run is not None else None),
    )
    progress_bar.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Training complete. Final checkpoint: {final_ckpt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train patch-AR + slot-DiT LM-TTS")
    parser.add_argument("--config", type=str, default="configs/model_medium.yaml")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt")
    parser.add_argument("--vocab", type=str, default=None, help="Path to char_vocab.json")
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
