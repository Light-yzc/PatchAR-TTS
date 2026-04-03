import os


import sys
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

try:
    from phonemizer.separator import Separator
    from phonemizer.backend.espeak.espeak import EspeakBackend

    # 猴子补丁：阻止 phonemizer 复制 .so 到 /tmp（Colab /tmp 是 noexec）
    from phonemizer.backend.espeak import api as espeak_api
    espeak_api._copy_library = lambda lib: lib

    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False

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
    "ZH": "cmn",       # Mandarin Chinese
    "JA": "ja",         # Japanese
    "EN": "en-us",      # American English
}

# Compact IPA separator: space between words, nothing between phones
IPA_SEP = Separator(phone="", word=" ", syllable="") if HAS_PHONEMIZER else None

# ─── Persistent backend singletons ──────────────────────────────────
# Key optimization: reuse backend instances instead of creating new ones.
# Each EspeakBackend loads libespeak-ng.so and allocates memory.
# Creating thousands of instances causes OOM.
_BACKENDS: dict[str, "EspeakBackend"] = {}


def _get_backend(lang_code: str) -> "EspeakBackend":
    """Get or create a persistent EspeakBackend for the given language."""
    if lang_code not in _BACKENDS:
        _BACKENDS[lang_code] = EspeakBackend(
            language=lang_code,
            preserve_punctuation=True,
            with_stress=False,
            words_mismatch='ignore'  
        )
    return _BACKENDS[lang_code]


# ─── Core functions ─────────────────────────────────────────────────

def g2p_ipa(text: str, language: str) -> str:
    """
    Convert text to IPA using a persistent espeak-ng backend.

    Args:
        text:     input text in any supported language
        language: "ZH", "JA", or "EN"

    Returns:
        IPA string, e.g. "tɕintʰjɛn tʰjɛntɕʰi tʂənxau"
    """
    if not HAS_PHONEMIZER:
        raise ImportError(
            "Please install phonemizer: pip install phonemizer\n"
            "And espeak-ng: apt install espeak-ng (Linux)"
        )

    lang_code = LANG_MAP.get(language.upper())
    if lang_code is None:
        raise ValueError(f"Unsupported language: {language}. Supported: {list(LANG_MAP.keys())}")

    if lang_code == "ja":
        text = _ja_kanji_to_kana(text)

    backend = _get_backend(lang_code)
    # Use the backend's phonemize method directly (no new backend creation!)
    result = backend.phonemize([text], separator=IPA_SEP, strip=True)
    return result[0] if result else ""


def g2p_ipa_batch(texts: list[str], language: str) -> list[str]:
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

    lang_code = LANG_MAP.get(language.upper())
    if lang_code is None:
        raise ValueError(f"Unsupported language: {language}")

    if lang_code == "ja":
        texts = [_ja_kanji_to_kana(t) for t in texts]

    backend = _get_backend(lang_code)
    return backend.phonemize(texts, separator=IPA_SEP, strip=True)


def text_to_phonemes_ipa(text: str, language: str) -> str:
    """
    Drop-in replacement for text_to_phonemes() in g2p.py.
    Converts text to IPA and prepends a language tag.

    Example:
        text_to_phonemes_ipa("今天天气真好", "ZH")
        → "[ZH] tɕintʰjɛn tʰjɛntɕʰi tʂənxau"
    """
    language = language.upper()
    ipa = g2p_ipa(text, language)
    return f"[{language}] {ipa}"


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
