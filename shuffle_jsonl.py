#!/usr/bin/env python3
"""
Shuffle a JSONL file to reduce lock contention during parallel loading.
Related records clustered together cause threads to compete for the same entity locks.
"""
import argparse
import random
from pathlib import Path


def shuffle_jsonl(input_file: str, output_file: str, seed: int = 42) -> None:
    """Shuffle lines in a JSONL file."""
    print(f"📖 Reading {input_file}...")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    original_count = len(lines)
    print(f"📊 Loaded {original_count:,} records")

    # Shuffle
    random.seed(seed)
    random.shuffle(lines)
    print(f"🔀 Shuffled with seed {seed}")

    # Write
    print(f"💾 Writing to {output_file}...")
    with open(output_file, 'w') as f:
        f.writelines(lines)

    print(f"✅ Done! Shuffled {original_count:,} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shuffle JSONL file to reduce lock contention during parallel loading"
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output file (default: <input>_shuffled.jsonl)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_shuffled.jsonl")

    shuffle_jsonl(args.input, args.output, args.seed)
