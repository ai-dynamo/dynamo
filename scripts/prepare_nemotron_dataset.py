#!/usr/bin/env python3
"""
Prepare balanced reasoning dataset from NVIDIA Nemotron.

Strategy:
- Reasoning: Math split (direct extraction)
- Non-reasoning: Chat split (joined with lmsys-chat-1m)

Outputs:
- train.txt (90% of data, fastText format)
- valid.txt (10% of data, fastText format)
- stats.json (dataset statistics)

Usage:
  python prepare_nemotron_dataset.py --samples 10000 --output data/nemotron_reasoning
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    import pandas as pd
    from datasets import load_dataset
except ImportError:
    print("❌ Please install: pip install datasets pandas")
    sys.exit(1)


def extract_user_prompt(messages):
    """Extract first user message from messages list"""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content and content != "-":
                return content
    return None


def load_reasoning_samples(n_samples, max_length):
    """Load reasoning samples from Math split"""
    print("1️⃣  Loading reasoning samples from Math split...")
    samples = []

    ds = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v1", split="math", streaming=True
    )

    for i, row in enumerate(ds):
        if len(samples) >= n_samples:
            break

        prompt = extract_user_prompt(row.get("messages", []))
        if prompt:
            # Truncate long prompts
            if len(prompt) > max_length:
                prompt = prompt[:max_length]
            samples.append(prompt)

        if (i + 1) % 10000 == 0:
            print(f"      Processed {i+1:,}, collected {len(samples):,}...")

    print(f"   ✅ Collected {len(samples):,} reasoning (math) samples\n")
    return samples


def load_non_reasoning_samples(n_samples, max_length):
    """Load non-reasoning samples from Chat split (joined with lmsys-chat-1m)"""
    print("2️⃣  Loading non-reasoning samples from Chat split...")

    # Step 1: Load lmsys-chat-1m
    print("   Loading lmsys-chat-1m (first 500K)...")
    lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    lmsys_data = []
    for i, row in enumerate(lmsys):
        if i >= 500000:
            break
        if i % 100000 == 0 and i > 0:
            print(f"      {i:,}...")

        conv_id = row.get("conversation_id")
        conversation = row.get("conversation", [])
        user_msg = next(
            (
                msg.get("content", "")
                for msg in conversation
                if msg.get("role") == "user"
            ),
            "",
        )

        if conv_id and user_msg.strip():
            lmsys_data.append({"conversation_id": conv_id, "prompt": user_msg})

    df_lmsys = pd.DataFrame(lmsys_data)
    df_lmsys.set_index("conversation_id", inplace=True)
    print(f"      Indexed {len(df_lmsys):,} conversations\n")

    # Step 2: Load Nemotron chat and join
    print("   Loading Nemotron chat split and joining...")
    chat_ds = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v1", split="chat", streaming=True
    )

    nemotron_ids = []
    for i, row in enumerate(chat_ds):
        if i >= 100000:
            break
        if i % 20000 == 0 and i > 0:
            print(f"      Processed {i:,}...")

        metadata = row.get("metadata", {})
        if isinstance(metadata, dict):
            conv_id = metadata.get("conversation_id")
        elif isinstance(metadata, str):
            try:
                conv_id = json.loads(metadata).get("conversation_id")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                conv_id = None
        else:
            conv_id = None

        if conv_id:
            nemotron_ids.append({"conversation_id": conv_id})

    # Join
    df_nemotron = pd.DataFrame(nemotron_ids)
    df_joined = df_nemotron.merge(
        df_lmsys, left_on="conversation_id", right_index=True, how="inner"
    )

    # Extract and truncate
    samples = []
    for prompt in df_joined["prompt"].tolist():
        if len(samples) >= n_samples:
            break
        if len(prompt) > max_length:
            prompt = prompt[:max_length]
        samples.append(prompt)

    print(f"   ✅ Collected {len(samples):,} non-reasoning (chat) samples\n")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Nemotron reasoning dataset for fastText training"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples per class (default: 10000)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=500,
        help="Max prompt length in characters (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nemotron_reasoning",
        help="Output directory (default: data/nemotron_reasoning)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Train split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Preparing Nemotron reasoning dataset")
    print(f"   Target: {args.samples:,} samples per class")
    print(f"   Max length: {args.max_length} chars")
    print(f"   Output: {output_dir}\n")

    # Load samples
    reasoning_data = load_reasoning_samples(args.samples, args.max_length)
    non_reasoning_data = load_non_reasoning_samples(args.samples, args.max_length)

    if not reasoning_data or not non_reasoning_data:
        print("❌ Failed to collect enough data for both classes")
        sys.exit(1)

    # Balance classes
    n = min(len(reasoning_data), len(non_reasoning_data))
    reasoning_data = reasoning_data[:n]
    non_reasoning_data = non_reasoning_data[:n]

    print(f"✅ Balanced dataset: {n:,} samples per class (total: {n*2:,})\n")

    # Shuffle
    random.shuffle(reasoning_data)
    random.shuffle(non_reasoning_data)

    # Split train/valid
    n_train = int(n * args.train_split)

    train_reasoning = reasoning_data[:n_train]
    valid_reasoning = reasoning_data[n_train:]
    train_non_reasoning = non_reasoning_data[:n_train]
    valid_non_reasoning = non_reasoning_data[n_train:]

    print("📊 Dataset splits:")
    print(
        f"   Train: {len(train_reasoning):,} reasoning + {len(train_non_reasoning):,} non-reasoning = {len(train_reasoning) + len(train_non_reasoning):,}"
    )
    print(
        f"   Valid: {len(valid_reasoning):,} reasoning + {len(valid_non_reasoning):,} non-reasoning = {len(valid_reasoning) + len(valid_non_reasoning):,}\n"
    )

    # Write fastText format files
    train_path = output_dir / "train.txt"
    valid_path = output_dir / "valid.txt"

    print("💾 Writing dataset files...")
    with open(train_path, "w", encoding="utf-8") as train_f:
        for text in train_reasoning:
            train_f.write(f"__label__reasoning {text}\n")
        for text in train_non_reasoning:
            train_f.write(f"__label__non-reasoning {text}\n")

    with open(valid_path, "w", encoding="utf-8") as valid_f:
        for text in valid_reasoning:
            valid_f.write(f"__label__reasoning {text}\n")
        for text in valid_non_reasoning:
            valid_f.write(f"__label__non-reasoning {text}\n")

    print(f"   ✅ {train_path}")
    print(f"   ✅ {valid_path}\n")

    # Save statistics
    stats = {
        "total_samples": n * 2,
        "samples_per_class": n,
        "train_samples": len(train_reasoning) + len(train_non_reasoning),
        "valid_samples": len(valid_reasoning) + len(valid_non_reasoning),
        "max_length": args.max_length,
        "seed": args.seed,
        "reasoning_source": "math split",
        "non_reasoning_source": "chat split (via lmsys-chat-1m)",
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"📈 Statistics saved to {stats_path}")

    # Show sample data
    print("\n🧪 Sample data (first 3 from each class):\n")
    print("REASONING (Math):")
    for i, text in enumerate(train_reasoning[:3], 1):
        print(f"  {i}. {text[:80]}...")

    print("\nNON-REASONING (Chat):")
    for i, text in enumerate(train_non_reasoning[:3], 1):
        print(f"  {i}. {text[:80]}...")

    print("\n✅ Dataset preparation complete!")
    print("\nNext step:")
    print(f"  python scripts/train_fasttext_nemotron.py --input {output_dir}")


if __name__ == "__main__":
    main()
