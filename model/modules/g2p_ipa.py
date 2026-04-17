import os


import sys
import logging
# 直接指向系统 espeak-ng 库
if sys.platform == "darwin":
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.1.dylib"
else:
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"



"""
IPA-based G2P for multilingual TTS.

Uses a **persistent** EspeakBackend singleton per language to avoid the
massive memory leak from re-creating backends on every call.
Supports batch processing for vocab building.
"""

import re
from functools import lru_cache
from tqdm.auto import tqdm

try:
    from phonemizer.separator import Separator
    from phonemizer.backend.espeak.espeak import EspeakBackend

    # 猴子补丁：阻止 phonemizer 复制 .so 到 /tmp（Colab /tmp 是 noexec）
    from phonemizer.backend.espeak import api as espeak_api
    espeak_api._copy_library = lambda lib: lib

    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False


logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("phonemizer.phonemize").setLevel(logging.ERROR)
logging.getLogger("phonemizer.backend").setLevel(logging.ERROR)

try:
    import pykakasi
    HAS_PYKAKASI = True
except ImportError:
    HAS_PYKAKASI = False


@lru_cache(maxsize=1)
def _get_kakasi():
    kks = pykakasi.kakasi()
    return kks


def _ja_kanji_to_kana(text: str) -> str:
    """Convert Japanese kanji to katakana readings."""
    if not HAS_PYKAKASI:
        import warnings
        warnings.warn("pykakasi not installed; Japanese kanji won't be converted")
        return text
    kks = _get_kakasi()
    result = kks.convert(text)
    return "".join([item['kana'] for item in result])


# ─── Language code mapping ───────────────────────────────────────────
LANG_MAP = {
    "ZH": "cmn",        # Mandarin Chinese
    "JA": "ja",         # Japanese
    "EN": "en-us",      # American English
}

# Emit explicit phone units so downstream tokenization can keep affricates /
# diphthongs intact instead of splitting the IPA string character by character.
IPA_SEP = Separator(phone=" ", word=" | ", syllable="") if HAS_PHONEMIZER else None

# Keep punctuation as standalone raw tokens in the text stream instead of
# letting phonemizer glue them to neighboring phoneme units.
PRESERVED_PUNCTUATION = {
    "，", "。", "！", "？", "、", "；", "：",
    ",", ".", "!", "?", ";", ":",
    "…", "—",
    "（", "）", "(", ")",
    "“", "”", "\"",
    "「", "」", "『", "』",
}
LANG_SWITCH_RE = re.compile(r"\((?:en(?:-us)?|ja|zh|cmn)\)", re.IGNORECASE)

# ─── Persistent backend singletons ──────────────────────────────────
# Key optimization: reuse backend instances instead of creating new ones.
# Each EspeakBackend loads libespeak-ng.so and allocates memory.
# Creating thousands of instances causes OOM.
_BACKENDS: dict[str, "EspeakBackend"] = {}


@lru_cache(maxsize=1)
def _get_silent_phonemizer_logger() -> logging.Logger:
    """Return a dedicated phonemizer logger that never emits training-time warnings."""
    logger = logging.getLogger("phonemizer.silent")
    logger.handlers = [logging.NullHandler()]
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.disabled = True
    return logger


def _get_backend(lang_code: str) -> "EspeakBackend":
    """Get or create a persistent EspeakBackend for the given language."""
    if lang_code not in _BACKENDS:
        _BACKENDS[lang_code] = EspeakBackend(
            language=lang_code,
            preserve_punctuation=False,
            with_stress=False,
            words_mismatch='ignore',
            logger=_get_silent_phonemizer_logger(),
        )
    return _BACKENDS[lang_code]


def _normalize_phoneme_units(text: str) -> str:
    """Collapse repeated whitespace while keeping '|' as a standalone word token."""
    text = LANG_SWITCH_RE.sub(" ", text)
    return " ".join(text.split())


def _split_text_for_g2p(text: str) -> list[tuple[str, str]]:
    """Split raw text into pronounceable spans and standalone punctuation tokens."""
    parts: list[tuple[str, str]] = []
    current_chars: list[str] = []

    def flush_text() -> None:
        if current_chars:
            parts.append(("text", "".join(current_chars)))
            current_chars.clear()

    for char in text:
        if char.isspace():
            flush_text()
            continue
        if char in PRESERVED_PUNCTUATION:
            flush_text()
            parts.append(("punct", char))
            continue
        current_chars.append(char)

    flush_text()
    return parts


def _prepare_segment_texts(texts: list[str], language: str) -> tuple[list[list[tuple[str, str]]], list[str]]:
    """Split texts and collect pronounceable spans for one-language batch G2P."""
    segmented_texts = [_split_text_for_g2p(text) for text in texts]
    text_segments = [value for segments in segmented_texts for kind, value in segments if kind == "text"]
    if language.upper() == "JA":
        text_segments = [_ja_kanji_to_kana(segment) for segment in text_segments]
    return segmented_texts, text_segments


def _stitch_phoneme_units(
    segmented_texts: list[list[tuple[str, str]]],
    phoneme_segments: list[str],
) -> list[str]:
    """Reinsert raw punctuation tokens between phoneme-unit spans."""
    stitched: list[str] = []
    phoneme_idx = 0
    for segments in segmented_texts:
        units: list[str] = []
        for kind, value in segments:
            if kind == "punct":
                units.append(value)
                continue
            phoneme_text = phoneme_segments[phoneme_idx]
            phoneme_idx += 1
            if phoneme_text:
                units.extend(phoneme_text.split())
        stitched.append(" ".join(units))
    return stitched


# ─── Core functions ─────────────────────────────────────────────────

def g2p_ipa(text: str, language: str) -> str:
    """
    Convert text to IPA using a persistent espeak-ng backend.

    Args:
        text:     input text in any supported language
        language: "ZH", "JA", or "EN"

    Returns:
        Space-delimited phone units, e.g. "tɕ in tʰ jɛn | tʰ jɛn tɕʰ i"
    """
    if not HAS_PHONEMIZER:
        raise ImportError(
            "Please install phonemizer: pip install phonemizer\n"
            "And espeak-ng: apt install espeak-ng (Linux)"
        )

    lang_code = LANG_MAP.get(language.upper())
    if lang_code is None:
        raise ValueError(f"Unsupported language: {language}. Supported: {list(LANG_MAP.keys())}")

    result = g2p_ipa_batch([text], language, chunk_size=1)
    return result[0] if result else ""


def g2p_ipa_batch(
    texts: list[str],
    language: str,
    chunk_size: int = 256,
    show_progress: bool = False,
) -> list[str]:
    """
    Batch convert texts to IPA. Much more efficient than calling g2p_ipa()
    in a loop because espeak-ng processes the entire batch in one subprocess call.

    Args:
        texts:    list of input texts
        language: "ZH", "JA", or "EN"

    Returns:
        list of IPA strings
    """
    if not texts:
        return []
    if not HAS_PHONEMIZER:
        raise ImportError("Please install phonemizer + espeak-ng")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    lang_code = LANG_MAP.get(language.upper())
    if lang_code is None:
        raise ValueError(f"Unsupported language: {language}")

    segmented_texts, text_segments = _prepare_segment_texts(texts, language)
    if not text_segments:
        return _stitch_phoneme_units(segmented_texts, [])

    backend = _get_backend(lang_code)
    phoneme_segments: list[str] = []
    chunk_starts = range(0, len(text_segments), chunk_size)
    for start in tqdm(
        chunk_starts,
        total=(len(text_segments) + chunk_size - 1) // chunk_size,
        desc=f"G2P {language.upper()}",
        unit="chunk",
        disable=not show_progress,
    ):
        chunk = text_segments[start : start + chunk_size]
        phoneme_segments.extend(
            _normalize_phoneme_units(item)
            for item in backend.phonemize(chunk, separator=IPA_SEP, strip=True)
        )
    return _stitch_phoneme_units(segmented_texts, phoneme_segments)


def text_to_phonemes_ipa(text: str, language: str) -> str:
    """
    Drop-in replacement for text_to_phonemes() in g2p.py.
    Converts text to IPA phone units and prepends a language tag.

    Example:
        text_to_phonemes_ipa("今天天气真好", "ZH")
        → "[ZH] tɕ in tʰ jɛn | tʰ jɛn tɕʰ i"
    """
    language = language.upper()
    ipa = g2p_ipa(text, language)
    return f"[{language}] {ipa}" if ipa else f"[{language}]"


# ─── Demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("IPA G2P Demo — Unified phoneme space across languages")
    print("=" * 60)

    test_cases = [
        ("今天天气真好，我们去玩吧！", "ZH"),
        ("我的名字叫Jack。", "ZH"),
        ("サポートシステム、40%カット……運動性、問題ありません！", "JA"),
        ("Hello world, this is a test.", "EN"),
        ("I'm an apple's! How are you?", "EN"),
    ]

    print("\n--- IPA Output ---")
    all_chars = set()
    for text, lang in test_cases:
        ipa = text_to_phonemes_ipa(text, lang)
        print(f"  {lang}: {ipa}")
        all_chars.update(ipa)

    print(f"\n--- Vocab Stats ---")
    print(f"  Total unique IPA characters: {len(all_chars)}")
    print(f"  Characters: {''.join(sorted(all_chars))}")
