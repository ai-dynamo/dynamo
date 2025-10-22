#!/usr/bin/env python3
"""
Train fastText classifier from prepared Nemotron dataset.

Expects dataset prepared by prepare_nemotron_dataset.py.

Usage:
  python train_fasttext_nemotron.py --input data/nemotron_reasoning --output fasttext-reasoning-classifier
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import fasttext
except ImportError:
    print("❌ Please install: pip install fasttext")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train fastText reasoning classifier from prepared dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing train.txt and valid.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fasttext-reasoning-classifier",
        help="Output directory for model (default: fasttext-reasoning-classifier)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=100,
        help="Embedding dimension (default: 100)",
    )
    parser.add_argument(
        "--word-ngrams",
        type=int,
        default=3,
        help="Max word n-gram length (default: 3)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=25,
        help="Number of epochs (default: 25)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.3,
        help="Learning rate (default: 0.3, range: 0.1-1.0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads (default: 8)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Min word frequency (default: 1, try 2-3 for large datasets)",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Use fastText autotune to find optimal hyperparameters (slower)",
    )
    args = parser.parse_args()

    # Paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train.txt"
    valid_path = input_dir / "valid.txt"
    stats_path = input_dir / "stats.json"

    # Validate input
    if not train_path.exists():
        print(f"❌ Training file not found: {train_path}")
        print(
            f"   Run: python scripts/prepare_nemotron_dataset.py --output {input_dir}"
        )
        sys.exit(1)

    if not valid_path.exists():
        print(f"❌ Validation file not found: {valid_path}")
        sys.exit(1)

    # Load stats
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print("📊 Dataset statistics:")
        print(f"   Total samples: {stats.get('total_samples', 'unknown'):,}")
        print(f"   Train: {stats.get('train_samples', 'unknown'):,}")
        print(f"   Valid: {stats.get('valid_samples', 'unknown'):,}")
        print(f"   Reasoning source: {stats.get('reasoning_source', 'unknown')}")
        print(
            f"   Non-reasoning source: {stats.get('non_reasoning_source', 'unknown')}\n"
        )

    print("🚀 Training fastText classifier...")
    print("   Parameters:")
    print(f"     dim={args.dim}")
    print(f"     wordNgrams={args.word_ngrams}")
    print(f"     epoch={args.epoch}")
    print(f"     lr={args.lr}")
    print(f"     minCount={args.min_count}")
    print(f"     threads={args.threads}")
    print(f"     autotune={args.autotune}\n")

    # Train
    if args.autotune:
        print("   Running autotune (this may take several minutes)...")
        model = fasttext.train_supervised(
            input=str(train_path),
            autotuneValidationFile=str(valid_path),
            autotuneDuration=300,  # 10 minutes
            verbose=2,
            thread=args.threads,
        )
    else:
        model = fasttext.train_supervised(
            input=str(train_path),
            dim=args.dim,
            wordNgrams=args.word_ngrams,
            epoch=args.epoch,
            lr=args.lr,
            verbose=2,  # Show progress
            minCount=args.min_count,
            thread=args.threads,
        )

    # Save
    model_path = output_dir / "reasoning.bin"
    model.save_model(str(model_path))
    print(f"\n✅ Model saved to {model_path}")

    # Validation
    print("\n📊 Validation metrics:")
    results = model.test(str(valid_path))
    precision = results[1]
    recall = results[2]
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"   Samples: {results[0]}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1_score:.4f}")

    # Example predictions
    print("\n🧪 Example predictions:")
    test_cases = [
        "Calculate the derivative of x^2 + 3x",
        "What is the capital of France?",
        "Prove by induction that the sum of first n integers is n(n+1)/2",
        "Write a poem about the ocean",
        "Solve for x: 2x + 5 = 15",
        "Who won the 2020 Olympics?",
        "If Sarah has 5 apples and gives 2 to Tom, how many does she have left?",
        "What is a simple and healthy snack?",
        "Explain how photosynthesis works",
        "Find the area of a circle with radius 5",
        "Prove that the sum of the first n odd numbers is n^2",
        "Prove that the sqrt of 2 is irrational",
    ]

    for text in test_cases:
        preds = model.predict(text, k=2)
        label = preds[0][0].replace("__label__", "")
        prob = preds[1][0]
        icon = "🧮" if label == "reasoning" else "📚"
        print(f"   {icon} '{text[:60]}...'")
        print(f"      → {label} ({prob:.3f})")

    print(f"\n💾 Model size: {model_path.stat().st_size / 1024:.1f} KB")
    print("\n✅ Training complete!")
    print("\nNext steps:")
    print(f"  1. Set: export SEMROUTER_MODEL_PATH={model_path.absolute()}")
    print(
        "  2. Build: maturin develop --uv --manifest-path lib/bindings/python/Cargo.toml --features fasttext-classifier"
    )
    print("  3. Launch Dynamo with SEMROUTER_ENABLED=true")


if __name__ == "__main__":
    main()
