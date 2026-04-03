from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from model.modules.g2p_ipa import g2p_ipa_batch, text_to_phonemes_ipa


class CharTokenizer:
    """
    Character-level tokenizer with optional multi-character special token support.

    If the vocab contains entries such as "<PAD>", "<UNK>", or "[EROGE]",
    encode() will greedily match the longest special token first, and fall back
    to single-character lookup for the rest.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        self.pad_id = 0
        self.unk_id = 1
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        self.special_tokens = sorted(
            [token for token in self.vocab if len(token) > 1],
            key=len,
            reverse=True,
        )

        # Compatibility aliases for code that expects HuggingFace-like names.
        self.pad_token_id = self.pad_id
        self.unk_token_id = self.unk_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def build_vocab(self, texts: list[str]):
        """Build vocab from a list of texts. Call once before training."""
        for text in texts:
            for ch in text:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        self.special_tokens = sorted(
            [token for token in self.vocab if len(token) > 1],
            key=len,
            reverse=True,
        )

    def encode(self, text: str) -> list[int]:
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for token in self.special_tokens:
                if text.startswith(token, i):
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue
            ids.append(self.vocab.get(text[i], self.unk_id))
            i += 1
        return ids

    def encoded_length(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        chars = []
        for idx in ids:
            token = self.id_to_char.get(idx, "<UNK>")
            if skip_special_tokens and token in {"<PAD>", "<UNK>"}:
                continue
            chars.append(token)
        return "".join(chars)

    def batch_encode(
        self,
        texts: list[str],
        max_len: int | None = None,
        return_tensors: bool = True,
    ) -> dict:
        """
        Encode a batch of texts and pad to max length.

        Returns:
            input_ids:      (B, L)
            attention_mask: (B, L) 1=valid, 0=pad
        """
        encoded = [self.encode(text) for text in texts]
        if max_len is None:
            max_len = max(len(e) for e in encoded) if encoded else 0
        else:
            encoded = [e[:max_len] for e in encoded]

        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            attention_mask.append([1.0] * len(ids) + [0.0] * pad_len)

        if return_tensors:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __call__(
        self,
        texts,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
        **kwargs,
    ):
        """HuggingFace-compatible interface for collate_fn."""
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        max_len = max_length if truncation else None
        return self.batch_encode(
            texts,
            max_len=max_len if padding or truncation else None,
            return_tensors=(return_tensors == "pt"),
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        vocab = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab)

    @classmethod
    def build_from_dataset_samples(
        cls,
        samples: list[dict],
        base_vocab: dict[str, int] | None = None,
    ) -> "CharTokenizer":
        """
        Build a vocab from dataset metadata in the same mapped text space used by
        data/dataset.py. This is only a fallback when no existing vocab file is
        provided.
        """
        grouped_texts: dict[str, list[str]] = defaultdict(list)
        grouped_indices: dict[str, list[int]] = defaultdict(list)

        for idx, sample in enumerate(samples):
            language = sample.get("language", "JA").upper()
            grouped_texts[language].append(sample["text"])
            grouped_indices[language].append(idx)

        mapped_cache: dict[int, str] = {}
        for language, texts in grouped_texts.items():
            try:
                batch_ipa = g2p_ipa_batch(texts, language)
                mapped_texts = [f"[{language}] {ipa}" for ipa in batch_ipa]
            except Exception:
                mapped_texts = [text_to_phonemes_ipa(text, language) for text in texts]

            for sample_idx, mapped in zip(grouped_indices[language], mapped_texts):
                mapped_cache[sample_idx] = mapped

        all_texts: list[str] = []
        for sample_idx, sample in enumerate(samples):
            mapped = mapped_cache[sample_idx]
            if sample.get("speaker", "") == "none":
                mapped = f"[EROGE] {mapped}"
            all_texts.append(mapped)

        tokenizer = cls(base_vocab)
        tokenizer.build_vocab(all_texts)
        return tokenizer
