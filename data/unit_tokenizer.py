from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from model.modules.g2p_ipa import g2p_ipa_batch, text_to_phonemes_ipa


DEFAULT_UNIT_SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "[SEP]",
    "[PROMPT_AUDIO_START]",
    "[TARGET_AUDIO_START]",
    "[EN]",
    "[ZH]",
    "[JA]",
    "[EROGE]",
    "|",
    "，",
    "。",
    "！",
    "？",
    "、",
    "；",
    "：",
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    "…",
    "—",
    "（",
    "）",
    "(",
    ")",
    "“",
    "”",
    "\"",
    "「",
    "」",
    "『",
    "』",
]


def default_unit_base_vocab(base_vocab: dict[str, int] | None = None) -> dict[str, int]:
    """Return a vocab that always contains the required unit-level special tokens."""
    if base_vocab is None:
        return {token: idx for idx, token in enumerate(DEFAULT_UNIT_SPECIAL_TOKENS)}

    merged = dict(sorted(base_vocab.items(), key=lambda item: item[1]))
    next_id = max(merged.values(), default=-1) + 1
    for token in DEFAULT_UNIT_SPECIAL_TOKENS:
        if token not in merged:
            merged[token] = next_id
            next_id += 1
    return merged


class UnitTokenizer:
    """Whitespace-delimited tokenizer for phoneme-unit text."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = default_unit_base_vocab(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
        self.pad_token_id = self.pad_id
        self.unk_token_id = self.unk_id
        self.special_tokens = set(DEFAULT_UNIT_SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        return text.split()

    def build_vocab(self, texts: list[str]) -> None:
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(token, self.unk_id) for token in self.tokenize(text)]

    def encoded_length(self, text: str) -> int:
        return len(self.tokenize(text))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, "<UNK>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def batch_encode(
        self,
        texts: list[str],
        max_len: int | None = None,
        return_tensors: bool = True,
    ) -> dict:
        encoded = [self.encode(text) for text in texts]
        if max_len is None:
            max_len = max((len(item) for item in encoded), default=0)
        else:
            encoded = [item[:max_len] for item in encoded]

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
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        max_len = max_length if truncation else None
        return self.batch_encode(
            texts,
            max_len=max_len if padding or truncation else None,
            return_tensors=(return_tensors == "pt"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UnitTokenizer":
        path = Path(path)
        vocab = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab)

    @classmethod
    def build_from_dataset_samples(
        cls,
        samples: list[dict],
        base_vocab: dict[str, int] | None = None,
    ) -> "UnitTokenizer":
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
                mapped_texts = [f"[{language}] {ipa}" if ipa else f"[{language}]" for ipa in batch_ipa]
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
