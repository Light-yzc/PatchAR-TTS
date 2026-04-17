from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import TTSDataset
from data.unit_tokenizer import UnitTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a phoneme-unit vocab from dataset metadata.")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing content.txt")
    parser.add_argument("--output", type=str, required=True, help="Where to save the vocab JSON")
    parser.add_argument("--base_vocab", type=str, default=None, help="Optional existing vocab JSON to extend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TTSDataset(data_root=args.data_root)

    base_vocab = None
    if args.base_vocab is not None:
        base_vocab = UnitTokenizer.load(args.base_vocab).vocab

    tokenizer = UnitTokenizer.build_from_dataset_samples(dataset.samples, base_vocab=base_vocab)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"Built phoneme-unit vocab with {tokenizer.vocab_size} tokens -> {output_path}")


if __name__ == "__main__":
    main()
