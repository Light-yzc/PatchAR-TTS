from __future__ import annotations

import argparse
import json
import logging
import math
import random
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
        router_use_input_norm=model_cfg.get("router_use_input_norm", True),
        router_logits_clip=model_cfg.get("router_logits_clip", 8.0),
        router_softmax_fp32=model_cfg.get("router_softmax_fp32", True),
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
        patch_lm_loss_weight=train_cfg.get("patch_lm_loss_weight", 1.0),
        stop_loss_weight=train_cfg.get("stop_loss_weight", 1.0),
        moe_aux_loss_weight=train_cfg.get("moe_aux_loss_weight", 1.0),
        stop_class_weights=train_cfg.get("stop_class_weights", [1.0, 20.0]),
    )


def _iter_trainable_params(*items) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, torch.nn.Parameter):
            if item.requires_grad and id(item) not in seen:
                params.append(item)
                seen.add(id(item))
            continue
        for param in item.parameters():
            if param.requires_grad and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def build_optimizer(model: LMTTSModel, cfg: dict, device: torch.device) -> torch.optim.Optimizer:
    """Optimizer for the full end-to-end model.

    AdamW8bit is preferred on CUDA because optimizer states dominate memory for
    this model size. Fallback to vanilla AdamW when bitsandbytes is unavailable.
    """
    train_cfg = cfg["training"]
    optimizer_name = str(train_cfg.get("optimizer", "adamw8bit")).lower()
    lr = float(train_cfg["learning_rate"])
    lm_lr = float(train_cfg.get("lm_learning_rate", lr))
    dit_lr = float(train_cfg.get("dit_learning_rate", lr))
    head_lr = float(train_cfg.get("head_learning_rate", lr))
    betas = (
        train_cfg.get("adam_beta1", 0.9),
        train_cfg.get("adam_beta2", 0.95),
    )
    weight_decay = train_cfg.get("weight_decay", 1e-2)

    seen: set[int] = set()
    param_groups: list[dict] = []

    def add_group(name: str, group_lr: float, params: list[torch.nn.Parameter]) -> None:
        unique_params = []
        for param in params:
            if id(param) in seen:
                continue
            seen.add(id(param))
            unique_params.append(param)
        if unique_params:
            param_groups.append(
                {
                    "name": name,
                    "params": unique_params,
                    "lr": group_lr,
                    "weight_decay": weight_decay,
                }
            )

    add_group("lm", lm_lr, _iter_trainable_params(model.base_lm))
    add_group("dit", dit_lr, _iter_trainable_params(model.dit))
    add_group(
        "heads",
        head_lr,
        _iter_trainable_params(
            model.patch_encoder,
            model.cond_slot_proj,
            model.patch_predictor,
            model.stop_proj,
            model.stop_head,
            model.audio_bos,
            model.cond_slot_embed,
            model.cond_type_embed,
        ),
    )
    remaining_params = [param for param in model.parameters() if param.requires_grad and id(param) not in seen]
    add_group("misc", head_lr, remaining_params)

    if optimizer_name == "adamw8bit":
        if device.type != "cuda":
            print(f"Warning: AdamW8bit requested but device is {device.type}; falling back to torch.optim.AdamW.")
        elif bnb is None:
            print("Warning: bitsandbytes is not installed; falling back to torch.optim.AdamW.")
        else:
            return bnb.optim.AdamW8bit(
                param_groups,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )

    return torch.optim.AdamW(
        param_groups,
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


def fast_forward_scheduler(scheduler: torch.optim.lr_scheduler.LambdaLR, step: int) -> None:
    """Align a freshly created scheduler with an already completed global step."""
    if step <= 0:
        return
    scheduler.last_epoch = step
    scheduler._step_count = step + 1
    closed_form_lrs = [
        base_lr * lr_lambda(step)
        for lr_lambda, base_lr in zip(scheduler.lr_lambdas, scheduler.base_lrs)
    ]
    for param_group, lr in zip(scheduler.optimizer.param_groups, closed_form_lrs):
        param_group["lr"] = lr
    scheduler._last_lr = list(closed_form_lrs)


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
    pred_audio_count = 0
    gt_audio_count = 0

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
        elif inference_cfg.get("log_audio", True):
            print(f"[Step {global_step}] log_audio is enabled but inference.vae_path is empty; skipping audio decode.")

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

            # logs[f"infer/prompt_frames_{sample_idx}"] = example.prompt_frames
            # logs[f"infer/target_frames_{sample_idx}"] = example.target_frames
            # logs[f"infer/pred_frames_{sample_idx}"] = example.pred_frames

        # if wandb_run is not None and wandb is not None:
        #     table = wandb.Table(columns=["step", "sample_idx", "text", "prompt_frames", "target_frames", "pred_frames"])
        #     for example in examples:
        #         table.add_data(
        #             global_step,
        #             example.sample_idx,
        #             example.text,
        #             example.prompt_frames,
        #             example.target_frames,
        #             example.pred_frames,
        #         )
        #     logs["infer/examples"] = table

        if wandb_run is not None and wandb is not None:
            sample_rate = int(inference_cfg.get("sample_rate", 48000))
            for example in examples:
                caption_prefix = f"step={global_step} sample={example.sample_idx}"
                if example.pred_waveform is not None:
                    pred_audio_count += 1
                    logs[f"infer/pred_audio_{example.sample_idx}"] = wandb.Audio(
                        waveform_to_wandb_array(example.pred_waveform),
                        sample_rate=sample_rate,
                        caption=f"{caption_prefix} pred | {example.text}",
                    )
                if example.gt_waveform is not None:
                    gt_audio_count += 1
                    logs[f"infer/gt_audio_{example.sample_idx}"] = wandb.Audio(
                        waveform_to_wandb_array(example.gt_waveform),
                        sample_rate=sample_rate,
                        caption=f"{caption_prefix} gt | {example.text}",
                    )
            # logs["infer/pred_audio_count"] = pred_audio_count
            # logs["infer/gt_audio_count"] = gt_audio_count
            # if inference_cfg.get("log_audio", True):
            #     print(
            #         f"[Step {global_step}] wandb audio prepared: "
            #         f"pred={pred_audio_count} gt={gt_audio_count}"
            #     )

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
    wandb_run_id: str | None = None,
) -> Path:
    """Save a full training checkpoint.

    Checkpoints are stored under `step_{global_step}/checkpoint.pt`.
    If the same step is saved again, the file is simply overwritten in place.
    """
    ckpt_dir = output_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    tmp_ckpt_path = ckpt_dir / "checkpoint.pt.tmp"
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
        tmp_ckpt_path,
    )
    tmp_ckpt_path.replace(ckpt_path)

    return ckpt_path


def load_checkpoint(
    resume_path: str | Path,
    model: LMTTSModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    resume_optimizer_state: bool = True,
) -> tuple[int, int, str | None, bool]:
    """Restore model/training state from a previously saved checkpoint."""
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Loaded checkpoint with compatibility mode: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    scheduler_state_restored = False
    if resume_optimizer_state:
        expected_group_count = len(optimizer.param_groups)

        try:
            optimizer_state = ckpt["optimizer"]
            checkpoint_group_count = len(optimizer_state.get("param_groups", []))
            if checkpoint_group_count != expected_group_count:
                print(
                    "Warning: skipping optimizer state restore because checkpoint has "
                    f"{checkpoint_group_count} parameter groups but current optimizer has "
                    f"{expected_group_count}."
                )
            else:
                optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            print(f"Warning: could not restore optimizer state: {exc}")

        try:
            scheduler_state = ckpt["scheduler"]
            checkpoint_scheduler_groups = len(scheduler_state.get("base_lrs", []))
            if checkpoint_scheduler_groups != expected_group_count:
                print(
                    "Warning: skipping scheduler state restore because checkpoint has "
                    f"{checkpoint_scheduler_groups} base LRs but current optimizer has "
                    f"{expected_group_count} parameter groups."
                )
            else:
                scheduler.load_state_dict(scheduler_state)
                scheduler_state_restored = True
        except Exception as exc:
            print(f"Warning: could not restore scheduler state: {exc}")

        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f"Warning: could not restore scaler state: {exc}")
    else:
        print("Checkpoint loaded without optimizer/scheduler/scaler state.")

    global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
    start_epoch = int(ckpt.get("epoch", 0))
    wandb_run_id = ckpt.get("wandb_run_id")
    return global_step, start_epoch, wandb_run_id, scheduler_state_restored


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move every tensor in the collated batch to the selected device."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=(device.type == "cuda"))
        else:
            moved[key] = value
    return moved


def _scalar_debug(name: str, value: torch.Tensor | float | int) -> str:
    """Format one scalar-ish debug value together with its finite flag."""
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    value = value.detach().float()
    if value.numel() == 0:
        return f"{name}=empty"
    scalar = float(value.reshape(-1)[0].cpu())
    finite = bool(torch.isfinite(value).all().item())
    return f"{name}={scalar:.6g} finite={finite}"


def _format_nonfinite_tensor(name: str, tensor: torch.Tensor) -> str:
    """Summarize one tensor that already contains non-finite values."""
    tensor = tensor.detach()
    finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    total_count = int(tensor.numel())
    finite_abs_max = float("nan")
    if finite_count > 0:
        finite_values = tensor[finite_mask].float()
        finite_abs_max = float(finite_values.abs().max().cpu())
    return (
        f"{name}(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"finite={finite_count}/{total_count}, finite_abs_max={finite_abs_max:.6g})"
    )


@torch.no_grad()
def _find_nonfinite_module_tensors(module: torch.nn.Module, max_items: int = 8) -> list[str]:
    """Return the first few model parameters / buffers that already contain NaN/Inf."""
    issues: list[str] = []
    for name, tensor in module.named_parameters():
        if not torch.isfinite(tensor).all():
            issues.append(_format_nonfinite_tensor(f"param:{name}", tensor))
            if len(issues) >= max_items:
                return issues
    for name, tensor in module.named_buffers():
        if not torch.isfinite(tensor).all():
            issues.append(_format_nonfinite_tensor(f"buffer:{name}", tensor))
            if len(issues) >= max_items:
                return issues
    return issues


@torch.no_grad()
def _find_nonfinite_optimizer_tensors(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    max_items: int = 8,
) -> list[str]:
    """Return the first few optimizer-state tensors that already contain NaN/Inf."""
    issues: list[str] = []
    param_names = {id(param): name for name, param in model.named_parameters()}
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(group["params"]):
            state = optimizer.state.get(param)
            if not state:
                continue
            param_name = param_names.get(id(param), f"<unnamed:{group_idx}:{param_idx}>")
            for state_name, state_value in state.items():
                if not torch.is_tensor(state_value):
                    continue
                if not torch.isfinite(state_value).all():
                    issues.append(
                        _format_nonfinite_tensor(
                            f"optim:{param_name}:{state_name}",
                            state_value,
                        )
                    )
                    if len(issues) >= max_items:
                        return issues
    return issues


def resolve_training_precision(train_cfg: dict, device: torch.device) -> tuple[str, bool, torch.dtype | None, bool]:
    """Resolve training precision with backward compatibility for the legacy fp16 flag."""
    precision_value = train_cfg.get("precision")
    if precision_value is None:
        precision = "fp16" if bool(train_cfg.get("fp16", True)) and device.type == "cuda" else "fp32"
    else:
        precision = str(precision_value).lower().strip()

    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported training.precision={precision!r}. Expected one of fp32/fp16/bf16.")

    if precision == "fp32":
        return precision, False, None, False

    if device.type != "cuda":
        raise ValueError(f"training.precision={precision} requires CUDA, but current device is {device.type}.")

    if precision == "fp16":
        return precision, True, torch.float16, True

    if not torch.cuda.is_bf16_supported():
        raise ValueError("training.precision=bf16 was requested, but this CUDA device does not report bf16 support.")

    return precision, True, torch.bfloat16, False


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

    precision_name, amp_enabled, autocast_dtype, scaler_enabled = resolve_training_precision(train_cfg, device)
    scaler = GradScaler("cuda", enabled=scaler_enabled)
    autocast_context = partial(
        autocast,
        device_type="cuda",
        enabled=amp_enabled,
        dtype=autocast_dtype,
    )

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
        f"gradient_checkpointing={gradient_checkpointing} | precision={precision_name}"
    )
    optimizer_group_lrs = {group.get("name", f"group_{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
    print("Optimizer group lrs: " + ", ".join(f"{name}={value:.2e}" for name, value in optimizer_group_lrs.items()))

    global_step = 0
    start_epoch = 0
    wandb_run_id = None
    scheduler_state_restored = False
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        global_step, start_epoch, wandb_run_id, scheduler_state_restored = load_checkpoint(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            resume_optimizer_state=not args.no_resume_optimizer,
        )
        print(f"Resumed at step {global_step}, epoch {start_epoch}")
        if global_step > 0 and (args.no_resume_optimizer or not scheduler_state_restored):
            fast_forward_scheduler(scheduler, global_step)
            print(f"Scheduler fast-forwarded to step {global_step} using the current optimizer groups.")

    wandb_run = init_wandb(cfg, output_dir=output_dir, resume_id=None)

    log_every_steps = int(train_cfg.get("log_every_steps", 10))
    save_every_steps = int(train_cfg.get("save_every_steps", 1000))
    max_steps = int(train_cfg["max_steps"])
    max_epochs = int(train_cfg.get("epochs", 1_000_000))
    gradient_clip = float(train_cfg.get("gradient_clip", 1.0))
    moe_aux_warmup_steps = int(train_cfg.get("moe_aux_warmup_steps", 5000))
    max_consecutive_nonfinite_steps = max(1, int(train_cfg.get("max_consecutive_nonfinite_steps", 8)))
    consecutive_nonfinite_steps = 0

    model.train()
    progress_bar = tqdm(total=max_steps, initial=global_step, desc="Training")
    for epoch in range(start_epoch, max_epochs):
        for batch in dataloader:
            if global_step >= max_steps:
                break

            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            moe_aux_scale = 1.0
            if moe_aux_warmup_steps > 0:
                moe_aux_scale = min((global_step + 1) / float(moe_aux_warmup_steps), 1.0)

            with autocast_context() if amp_enabled else nullcontext():
                losses = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    latents=batch["latent"],
                    prompt_mask=batch["prompt_mask"],
                    target_mask=batch["target_mask"],
                    padding_mask=batch["padding_mask"],
                    moe_aux_scale=moe_aux_scale,
                )

            if not torch.isfinite(losses.loss):
                latent_tensor = batch["latent"].detach().float()
                latent_abs_max = float(latent_tensor.abs().max().cpu())
                latent_finite = bool(torch.isfinite(latent_tensor).all().item())
                prompt_frames = batch["prompt_mask"].sum(dim=1).detach().cpu().tolist()
                target_frames = batch["target_mask"].sum(dim=1).detach().cpu().tolist()
                scaler_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                print(
                    f"[Step {global_step}] non-finite loss detected: "
                    f"loss={float(losses.loss.detach().float().cpu())}"
                )
                print(
                    "[Step {step}] components: {diff}, {lm}, {patch}, {stop}, {moe}, "
                    "weighted_patch={weighted_patch}, weighted_stop={weighted_stop}, "
                    "weighted_moe={weighted_moe}".format(
                        step=global_step,
                        diff=_scalar_debug("diff_loss", losses.diff_loss),
                        lm=_scalar_debug("lm_loss", losses.lm_loss),
                        patch=_scalar_debug("patch_lm_loss", losses.patch_lm_loss),
                        stop=_scalar_debug("stop_loss", losses.stop_loss),
                        moe=_scalar_debug("moe_aux_loss", losses.moe_aux_loss),
                        weighted_patch=_scalar_debug("weighted_patch_lm_loss", losses.weighted_patch_lm_loss),
                        weighted_stop=_scalar_debug("weighted_stop_loss", losses.weighted_stop_loss),
                        weighted_moe=_scalar_debug("weighted_moe_aux_loss", losses.weighted_moe_aux_loss),
                    )
                )
                print(
                    f"[Step {global_step}] batch stats: "
                    f"latent_abs_max={latent_abs_max:.6g} latent_finite={latent_finite} "
                    f"prompt_frames={prompt_frames} target_frames={target_frames} "
                    f"moe_aux_scale={moe_aux_scale:.6g} grad_scaler_scale={float(scaler_scale):.6g}"
                )
                consecutive_nonfinite_steps += 1
                print(
                    f"[Step {global_step}] non-finite counter="
                    f"{consecutive_nonfinite_steps}/{max_consecutive_nonfinite_steps}"
                )
                bad_model_tensors = _find_nonfinite_module_tensors(model)
                if bad_model_tensors:
                    print(
                        f"[Step {global_step}] model tensors already corrupted:\n  - "
                        + "\n  - ".join(bad_model_tensors)
                    )
                bad_optimizer_tensors = _find_nonfinite_optimizer_tensors(optimizer, model)
                if bad_optimizer_tensors:
                    print(
                        f"[Step {global_step}] optimizer tensors already corrupted:\n  - "
                        + "\n  - ".join(bad_optimizer_tensors)
                    )
                if bad_model_tensors or bad_optimizer_tensors:
                    raise RuntimeError(
                        f"Non-finite loss at step {global_step}: model/optimizer state already contains NaN/Inf. "
                        "Stop training and resume from an earlier checkpoint."
                    )
                if consecutive_nonfinite_steps >= max_consecutive_nonfinite_steps:
                    raise RuntimeError(
                        f"Non-finite loss persisted for {consecutive_nonfinite_steps} consecutive batches at step "
                        f"{global_step}. Model tensors are still finite, so the instability is being recreated in "
                        "forward/backward every batch. Stopping instead of looping forever."
                    )
                continue
            consecutive_nonfinite_steps = 0

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
                lr_values = scheduler.get_last_lr()
                group_lrs = {
                    optimizer.param_groups[idx].get("name", f"group_{idx}"): float(value)
                    for idx, value in enumerate(lr_values)
                }
                lm_lr = group_lrs.get("lm", float(lr_values[0]))
                dit_lr = group_lrs.get("dit", float(lr_values[0]))
                head_lr = group_lrs.get("heads", group_lrs.get("misc", float(lr_values[0])))
                metrics = {
                    "train/loss": float(losses.loss.item()),
                    "train/lm_loss": float(losses.lm_loss.item()),
                    "train/dit_loss": float(losses.dit_loss.item()),
                    "train/patch_lm_loss": float(losses.patch_lm_loss.item()),
                    "train/stop_head_loss": float(losses.stop_head_loss.item()),
                    "train/weighted_patch_lm_loss": float(losses.weighted_patch_lm_loss.item()),
                    "train/weighted_stop_loss": float(losses.weighted_stop_loss.item()),
                    "train/weighted_moe_aux_loss": float(losses.weighted_moe_aux_loss.item()),
                    # "train/lm_moe_aux_loss": float(losses.moe_aux_loss.item()),
                    "train/diff_loss": float(losses.diff_loss.item()),
                    "train/stop_loss": float(losses.stop_loss.item()),
                    "train/moe_aux_loss": float(losses.moe_aux_loss.item()),
                    "train/moe_aux_scale": float(moe_aux_scale),
                    "train/lr": lm_lr,
                    "train/lr_lm": lm_lr,
                    "train/lr_dit": dit_lr,
                    "train/lr_head": head_lr,
                    "train/epoch": float(epoch + 1),
                }
                message = (
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={metrics['train/loss']:.4f} "
                    f"lm={metrics['train/lm_loss']:.4f} "
                    f"patch={metrics['train/patch_lm_loss']:.4f} "
                    f"dit={metrics['train/dit_loss']:.4f} "
                    f"stop={metrics['train/stop_head_loss']:.4f} "
                    f"moe_aux={metrics['train/moe_aux_loss']:.4f} "
                    f"lm_lr={metrics['train/lr_lm']:.2e} "
                    f"dit_lr={metrics['train/lr_dit']:.2e} "
                    f"head_lr={metrics['train/lr_head']:.2e}"
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
    parser.add_argument(
        "--no_resume_optimizer",
        action="store_true",
        help="When used with --resume, load only model weights and skip optimizer/scheduler/scaler state.",
    )
    parser.add_argument("--vocab", type=str, default=None, help="Path to char_vocab.json")
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
