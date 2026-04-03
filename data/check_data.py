"""
Quick diagnostic script to check VAE latent file quality.

Usage:
    python check_data.py --data_root /path/to/processed/train/
"""

import os
import argparse
import torch
from pathlib import Path


def check_latents(data_root: str, max_files: int = 100):
    folder = Path(data_root)
    wav_dir = folder / "wav"
    
    if not wav_dir.exists():
        print(f"ERROR: {wav_dir} does not exist!")
        return
    
    pt_files = list(wav_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {wav_dir}")
    
    if len(pt_files) == 0:
        print("No .pt files found!")
        return
    
    nan_count = 0
    inf_count = 0
    stats = {"min": float("inf"), "max": float("-inf"), "means": [], "stds": []}
    
    check_count = min(max_files, len(pt_files))
    for i, pt_file in enumerate(pt_files[:check_count]):
        latent = torch.load(pt_file, map_location="cpu", weights_only=True)
        
        has_nan = torch.isnan(latent).any().item()
        has_inf = torch.isinf(latent).any().item()
        
        if has_nan:
            nan_count += 1
            print(f"  ⚠️  NaN found in: {pt_file.name} (shape={latent.shape})")
        if has_inf:
            inf_count += 1
            print(f"  ⚠️  Inf found in: {pt_file.name} (shape={latent.shape})")
        
        if not has_nan and not has_inf:
            stats["min"] = min(stats["min"], latent.min().item())
            stats["max"] = max(stats["max"], latent.max().item())
            stats["means"].append(latent.mean().item())
            stats["stds"].append(latent.std().item())
    
    print(f"\n=== Checked {check_count}/{len(pt_files)} files ===")
    print(f"  NaN files: {nan_count}")
    print(f"  Inf files: {inf_count}")
    
    if stats["means"]:
        import statistics
        print(f"\n=== Latent value statistics ===")
        print(f"  Global min:  {stats['min']:.4f}")
        print(f"  Global max:  {stats['max']:.4f}")
        print(f"  Global abs max: {max(abs(stats['min']), abs(stats['max'])):.4f}")
        print(f"  Avg mean:    {statistics.mean(stats['means']):.4f}")
        print(f"  Avg std:     {statistics.mean(stats['stds']):.4f}")
        
        abs_max = max(abs(stats['min']), abs(stats['max']))
        if abs_max > 65504:
            print(f"\n  🔴 CRITICAL: abs max ({abs_max:.1f}) exceeds fp16 range (65504)!")
            print(f"     This WILL cause NaN in fp16 training.")
            print(f"     Solution: scale down latents or use bf16/fp32.")
        elif abs_max > 100:
            print(f"\n  🟡 WARNING: abs max ({abs_max:.1f}) is large.")
            print(f"     This may cause instability in fp16 training.")
            print(f"     Consider normalizing latents or using bf16.")
        elif abs_max > 10:
            print(f"\n  🟢 Latent range looks reasonable but on the larger side.")
        else:
            print(f"\n  🟢 Latent range looks good.")
    
    # Check first file shape
    sample = torch.load(pt_files[0], map_location="cpu", weights_only=True)
    print(f"\n=== Sample file: {pt_files[0].name} ===")
    print(f"  Shape: {sample.shape}")
    print(f"  Dtype: {sample.dtype}")
    print(f"  Mean:  {sample.mean():.4f}")
    print(f"  Std:   {sample.std():.4f}")
    print(f"  Min:   {sample.min():.4f}")
    print(f"  Max:   {sample.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=100)
    args = parser.parse_args()
    check_latents(args.data_root, args.max_files)
