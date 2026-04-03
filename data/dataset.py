"""
Dataset and data loading for VAE-DiT TTS.

Expects pre-processed data where audio has been encoded to VAE latents offline.
Each sample in the dataset directory should contain:
  - latent.pt: (T, D) VAE latent tensor
  - text.txt: full transcription text

During training, each sample is randomly split into prompt + target.
"""

import os
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import tqdm
from pathlib import Path

# Import the IPA G2P module.
# Prefer the in-repo implementation, but keep a fallback import path so
# older experiments do not break immediately.
from model.modules.g2p_ipa import text_to_phonemes_ipa as text_to_phonemes

class TTSDataset(Dataset):
    """
    TTS dataset loading pre-computed VAE latents and text.

    Data directory structure:
      data_root/
        sample_000/
          latent.pt   — (T, D) tensor
          text.txt    — transcription
        sample_001/
          ...

    Each __getitem__ returns:
      - prompt_latent: (T_prompt, D)
      - target_latent: (T_gen, D)
      - text: str (full text, to be tokenized by collator)
    """

    def __init__(
        self,
        data_root: str,
        latent_rate: int = 25,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 30.0,
        prompt_ratio_min: float = 0.2,
        prompt_ratio_max: float = 0.5,
    ):
        super().__init__()
        self.data_root = data_root
        self.latent_rate = latent_rate
        self.min_frames = int(min_duration_sec * latent_rate)
        self.max_frames = int(max_duration_sec * latent_rate)
        self.prompt_ratio_min = prompt_ratio_min
        self.prompt_ratio_max = prompt_ratio_max

        # Discover samples
        # content.txt format: "speaker_utteranceId_text"
        # e.g. "SSB0001_SSB00010001_今天天气真好"
        self.samples = []
        self.speaker_to_indices = {}  # speaker_id → [sample indices]
        folder = Path(data_root)
        with open(os.path.join(folder, 'content.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                speaker, utt_id, text = line.split('_', 2)  # maxsplit=2
                sample_idx = len(self.samples)
                latent_name = utt_id
                if speaker == "none":
                    latent_name = utt_id
                    language = "JA"
                elif speaker.startswith("jvs"):
                    latent_name = f"{utt_id.split('.')[0]}.pt"
                    language = "JA"
                elif speaker.startswith("SSB"):
                    latent_name = f"{speaker}_{utt_id.split('.')[0]}.pt"
                    language = "ZH"
                elif utt_id.startswith("char"):
                    latent_name = utt_id
                    language = "JA"
                else:
                    latent_name = f"{utt_id.split('.')[0]}.pt"
                    language = "EN"
                self.samples.append({
                    "latent_path": str(folder / 'wav' / latent_name),
                    "text": text,
                    "speaker": speaker,
                    "language": language,
                })
                self.speaker_to_indices.setdefault(speaker, []).append(sample_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_latent(self, path: str) -> torch.Tensor:
        """Load latent and clamp to valid length range."""
        latent = torch.load(path, map_location="cpu", weights_only=True)
        # Handle stereo latents: (C, T, D) → (T, D) via channel average
        if latent.dim() == 3:
            latent = latent.mean(dim=0)  # average over channels
        if latent.shape[0] > self.max_frames:
            start = random.randint(0, latent.shape[0] - self.max_frames)
            latent = latent[start : start + self.max_frames]
        if latent.shape[0] < self.min_frames:
            pad = self.min_frames - latent.shape[0]
            latent = F.pad(latent, (0, 0, 0, pad))
        return latent

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        speaker = sample["speaker"]

        # Target: current sample
        target_latent = self._load_latent(sample["latent_path"])

        # Prompt: pick a different utterance from the same speaker
        same_speaker_indices = self.speaker_to_indices[speaker]
        if speaker == "none":
            # No reliable reference exists for these samples.
            # Keep the prompt side empty so the model learns target-text-only conditioning.
            prompt_latent = target_latent[:0]
            prompt_text = ""
            target_text = sample["text"]
        elif len(same_speaker_indices) > 1:
            prompt_idx = idx
            while prompt_idx == idx:
                prompt_idx = random.choice(same_speaker_indices)
            prompt_latent = self._load_latent(self.samples[prompt_idx]["latent_path"])
            prompt_text = self.samples[prompt_idx]["text"]
            target_text = sample["text"]
        else:
            # Fallback: split the same utterance into prompt + target, matching LoRA behavior
            ratio = random.uniform(self.prompt_ratio_min, self.prompt_ratio_max)
            split = max(1, min(int(target_latent.shape[0] * ratio), target_latent.shape[0] - 1))
            prompt_latent = target_latent[:split]
            target_latent = target_latent[split:]
            split_char = max(1, int(len(sample["text"]) * ratio))
            prompt_text = sample["text"][:split_char]
            target_text = sample["text"][split_char:]

        # Apply G2P and language tags individually to prompt and target
        lang = sample["language"]
        mapped_prompt = text_to_phonemes(prompt_text, lang) if prompt_text else ""
        mapped_target = text_to_phonemes(target_text, lang)
        if speaker == "none":
            mapped_prompt = f'[EROGE] {mapped_prompt}'
        # Combined text for Text Encoder: "[LANG] prompt_phonemes [SEP] [LANG] target_phonemes"
        full_text = f"{mapped_prompt} [SEP] {mapped_target}"

        return {
            "prompt_latent": prompt_latent,
            "target_latent": target_latent,
            "full_text": full_text,
            "prompt_text_raw": prompt_text,
            "target_text_raw": target_text,
            "prompt_text_mapped": mapped_prompt,
            "target_text_mapped": mapped_target,
            "language": lang,
            "total_frames": prompt_latent.shape[0] + target_latent.shape[0],
            "target_frames": target_latent.shape[0],
        }


def collate_fn(batch: list[dict], tokenizer=None, max_text_len: int = 512) -> dict:
    """
    Collate function that packs prompt + target contiguously, padding at end.

    Instead of padding prompt and target separately (which creates mid-sequence
    padding), we concatenate valid frames first, then pad only at the end.

    Output layout per sample:
        [valid_prompt_frames | valid_target_frames | padding...]

    Args:
        batch: list of dataset items
        tokenizer: T5 tokenizer for text encoding
        max_text_len: maximum text token length

    Returns:
        Collated batch dict:
            latent:       (B, T_max, D)  packed prompt+target, padded at end
            prompt_mask:  (B, T_max)     1=prompt frame, 0=other
            target_mask:  (B, T_max)     1=valid target frame, 0=other
            padding_mask: (B, T_max)     1=valid (prompt or target), 0=pad
            target_frames: (B,)          GT target frame count
    """
    B = len(batch)
    D = batch[0]["prompt_latent"].shape[-1]

    # Compute per-sample lengths and max combined length
    prompt_lens = [item["prompt_latent"].shape[0] for item in batch]
    target_lens = [item["target_latent"].shape[0] for item in batch]
    combined_lens = [p + t for p, t in zip(prompt_lens, target_lens)]
    T_max = max(combined_lens)

    # Allocate tensors
    latents = torch.zeros(B, T_max, D)
    prompt_masks = torch.zeros(B, T_max)
    target_masks = torch.zeros(B, T_max)
    padding_masks = torch.zeros(B, T_max)
    target_frames = torch.zeros(B)

    for i, item in enumerate(batch):
        t_p = prompt_lens[i]
        t_g = target_lens[i]

        # Pack: [prompt | target]
        latents[i, :t_p] = item["prompt_latent"]
        latents[i, t_p:t_p + t_g] = item["target_latent"]

        # Masks
        prompt_masks[i, :t_p] = 1.0
        target_masks[i, t_p:t_p + t_g] = 1.0
        padding_masks[i, :t_p + t_g] = 1.0
        target_frames[i] = item["target_frames"]

    result = {
        "latent": latents,
        "prompt_mask": prompt_masks,
        "target_mask": target_masks,
        "padding_mask": padding_masks,
        "target_frames": target_frames,
        "prompt_text_raw": [item["prompt_text_raw"] for item in batch],
        "target_text_raw": [item["target_text_raw"] for item in batch],
    }

    # Tokenize text
    if tokenizer is not None:
        texts = [item["full_text"] for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        result["input_ids"] = encoded["input_ids"]
        result["attention_mask"] = encoded["attention_mask"]

        # Calculate target_text_mask for duration predictor
        target_text_masks = torch.zeros_like(encoded["attention_mask"])
        for i, item in enumerate(batch):
            prefix_text = f'{item["prompt_text_mapped"]} [SEP] '
            start_idx = tokenizer.encoded_length(prefix_text) if hasattr(tokenizer, "encoded_length") else len(tokenizer.encode(prefix_text))
            target_text_masks[i, start_idx:] = encoded["attention_mask"][i, start_idx:]
                    
        result["target_text_mask"] = target_text_masks

        # # Encode CTC targets: target_text_mapped → char_vocab token IDs
        # ctc_targets_list = []
        # ctc_target_lengths = []
        # for item in batch:
        #     target_text = item["target_text_mapped"]
        #     ids = tokenizer.encode(target_text)  # list[int]
        #     ctc_targets_list.append(torch.tensor(ids, dtype=torch.long))
        #     ctc_target_lengths.append(len(ids))

        # # Flatten for CTC (1D target tensor, as required by torch.nn.CTCLoss)
        # result["ctc_targets"] = torch.cat(ctc_targets_list)
        # result["ctc_target_lengths"] = torch.tensor(ctc_target_lengths, dtype=torch.long)
    else:
        result["texts"] = [item["full_text"] for item in batch]

    return result


class TTSDatasetLoRA(Dataset):
    """
    Simplified dataset for LoRA fine-tuning.
    No speaker grouping — splits each utterance into prompt + target
    by random ratio. Text is split at the same character ratio.

    Data directory structure:
      data_root/
        wav/
          sample1.pt
        content.txt   — lines like "filename.pt_text"
    """

    def __init__(
        self,
        data_root: str,
        language: str = "JA",
        latent_rate: int = 25,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 30.0,
        prompt_ratio_min: float = 0.2,
        prompt_ratio_max: float = 0.5,
    ):
        super().__init__()
        self.data_root = data_root
        self.language = language
        self.latent_rate = latent_rate
        self.min_frames = int(min_duration_sec * latent_rate)
        self.max_frames = int(max_duration_sec * latent_rate)
        self.prompt_ratio_min = prompt_ratio_min
        self.prompt_ratio_max = prompt_ratio_max

        self.samples = []
        content_path = os.path.join(data_root, "content.txt")
        with open(content_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split only on first underscore — text may contain underscores
                parts = line.split("_", 1)
                if len(parts) < 2:
                    continue
                filename, text = parts
                latent_path = os.path.join(data_root, filename)
                if os.path.exists(latent_path):
                    self.samples.append({
                        "latent_path": latent_path,
                        "text": text,
                    })

        print(f"TTSDatasetLoRA: {len(self.samples)} samples, language={language}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_latent(self, path: str) -> torch.Tensor:
        latent = torch.load(path, map_location="cpu", weights_only=True)
        if latent.dim() == 3:
            latent = latent.mean(dim=0)
        return latent

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        latent = self._load_latent(sample["latent_path"])
        text = sample["text"]

        # If audio exceeds max_frames, truncate from the beginning
        if latent.shape[0] > self.max_frames:
            text = text[:int(len(text) * self.max_frames / latent.shape[0])]
            latent = latent[:self.max_frames]

        if latent.shape[0] < self.min_frames:
            latent = F.pad(latent, (0, 0, 0, self.min_frames - latent.shape[0]))

        # Random split ratio
        ratio = random.uniform(self.prompt_ratio_min, self.prompt_ratio_max)
        split_frame = max(1, min(int(latent.shape[0] * ratio), latent.shape[0] - 1))
        prompt_latent = latent[:split_frame]
        target_latent = latent[split_frame:]  # no overlap with prompt

        # Split text by same ratio (no overlap)
        split_char = max(1, int(len(text) * ratio))
        prompt_text = text[:split_char]
        target_text = text[split_char:]

        lang = self.language
        mapped_prompt = text_to_phonemes(prompt_text, lang)
        mapped_target = text_to_phonemes(target_text, lang)
        full_text = f"{mapped_prompt} [SEP] {mapped_target}"

        return {
            "prompt_latent": prompt_latent,
            "target_latent": target_latent,
            "full_text": full_text,
            "prompt_text_mapped": mapped_prompt,
            "target_text_mapped": mapped_target,
            "language": lang,
            "total_frames": prompt_latent.shape[0] + target_latent.shape[0],
            "target_frames": target_latent.shape[0],
        }
