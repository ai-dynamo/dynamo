#!/usr/bin/env python3
"""
Train a fastText classifier for reasoning vs non-reasoning prompt detection.
Uses balanced curated data:
- Reasoning: GSM8K (math word problems)
- Non-reasoning: Generated simple factoid questions
"""

import argparse
import random
import sys
from pathlib import Path


def generate_nonreasoning_questions(n=5000):
    """Generate simple factoid questions that don't require multi-step reasoning"""
    templates = [
        "What is {topic}?",
        "Who was {person}?",
        "When did {event} happen?",
        "Where is {place} located?",
        "Define {concept}",
        "Explain what {concept} means",
        "Tell me about {topic}",
        "Describe {thing}",
        "What does {term} mean?",
        "Give me information about {topic}",
        "Summarize {topic}",
        "List facts about {topic}",
        "Who invented {invention}?",
        "What year was {event}?",
        "What is the capital of {place}?",
    ]

    fillers = {
        "topic": [
            "photosynthesis",
            "the Renaissance",
            "democracy",
            "the internet",
            "climate change",
            "jazz music",
            "impressionism",
            "cryptocurrency",
            "the Big Bang",
        ],
        "person": [
            "Shakespeare",
            "Einstein",
            "Napoleon",
            "Cleopatra",
            "Gandhi",
            "Leonardo da Vinci",
            "Marie Curie",
            "Mozart",
        ],
        "event": [
            "World War II",
            "the moon landing",
            "the Industrial Revolution",
            "the French Revolution",
            "the Cold War",
        ],
        "place": [
            "Paris",
            "Mount Everest",
            "the Amazon rainforest",
            "the Great Wall of China",
            "the Colosseum",
            "Tokyo",
        ],
        "concept": [
            "quantum physics",
            "evolution",
            "capitalism",
            "Buddhism",
            "relativity",
        ],
        "thing": [
            "a black hole",
            "DNA",
            "the solar system",
            "a volcano",
            "an ecosystem",
        ],
        "term": ["photosynthesis", "gravity", "democracy", "entropy", "mitosis"],
        "invention": [
            "the telephone",
            "the light bulb",
            "the airplane",
            "the computer",
            "the printing press",
        ],
    }

    questions = []
    for _ in range(n):
        template = random.choice(templates)
        # Find placeholder
        for key in fillers:
            if f"{{{key}}}" in template:
                value = random.choice(fillers[key])
                questions.append(template.format(**{key: value}))
                break

    return questions


def main():
    parser = argparse.ArgumentParser(description="Train fastText reasoning classifier")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fasttext-reasoning-classifier",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of samples per class",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Checking dependencies...")
    try:
        import fasttext
        from datasets import load_dataset
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install fasttext datasets")
        sys.exit(1)

    print("📥 Loading GSM8K dataset...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)

    print(f"   Loaded {len(gsm8k)} GSM8K samples")

    # Generate non-reasoning questions
    print(f"   Generating {args.samples} factoid questions...")
    nonreasoning_questions = generate_nonreasoning_questions(args.samples)

    # Prepare balanced data
    print(f"\n🔄 Preparing {args.samples} samples per class (balanced)...")

    train_path = output_dir / "train.txt"
    valid_path = output_dir / "valid.txt"

    train_count = {"reasoning": 0, "non-reasoning": 0}
    valid_count = {"reasoning": 0, "non-reasoning": 0}

    with open(train_path, "w", encoding="utf-8") as train_f, open(
        valid_path, "w", encoding="utf-8"
    ) as valid_f:
        # GSM8K (reasoning)
        for i, row in enumerate(gsm8k):
            if train_count["reasoning"] + valid_count["reasoning"] >= args.samples:
                break

            question = row.get("question", "").strip()
            if not question:
                continue

            label = "__label__reasoning"
            text_clean = " ".join(question.split())
            line = f"{label} {text_clean}\n"

            # 90/10 split
            if i % 10 == 0:
                valid_f.write(line)
                valid_count["reasoning"] += 1
            else:
                train_f.write(line)
                train_count["reasoning"] += 1

        # Non-reasoning (generated)
        for i, question in enumerate(nonreasoning_questions):
            if (
                train_count["non-reasoning"] + valid_count["non-reasoning"]
                >= args.samples
            ):
                break

            label = "__label__non-reasoning"
            text_clean = " ".join(question.split())
            line = f"{label} {text_clean}\n"

            # 90/10 split
            if i % 10 == 0:
                valid_f.write(line)
                valid_count["non-reasoning"] += 1
            else:
                train_f.write(line)
                train_count["non-reasoning"] += 1

    print(
        f"   Train: {train_count['reasoning']} reasoning, {train_count['non-reasoning']} non-reasoning"
    )
    print(
        f"   Valid: {valid_count['reasoning']} reasoning, {valid_count['non-reasoning']} non-reasoning"
    )

    balance = min(train_count.values()) / max(train_count.values())
    print(f"   Balance: {balance:.2%}")

    # Train fastText
    print("\n🚀 Training fastText classifier...")
    print("   Parameters: dim=100, wordNgrams=3, epoch=25, lr=0.8")

    model_path = str(output_dir / "reasoning")

    try:
        model = fasttext.train_supervised(
            input=str(train_path),
            lr=0.8,
            dim=100,
            wordNgrams=3,
            epoch=25,
            loss="softmax",
            thread=4,
            verbose=0,
        )

        # Save model
        model.save_model(model_path + ".bin")
        print(f"✅ Model saved to {model_path}.bin")

        # Validate
        print("\n📊 Validation metrics:")
        n_samples, precision, recall = model.test(str(valid_path))
        print(f"   Samples: {n_samples}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        print(f"   F1-Score: {f1:.4f}")

        # Test on example prompts
        print("\n🧪 Example predictions:")
        test_cases = [
            ("Calculate the derivative of x^2 + 3x", "reasoning"),
            ("What is the capital of France?", "non-reasoning"),
            (
                "Prove by induction that the sum of first n integers is n(n+1)/2",
                "reasoning",
            ),
            ("Write a poem about the ocean", "non-reasoning"),
            ("Given a binary tree, find the maximum path sum", "reasoning"),
            ("Who won the 2020 Olympics?", "non-reasoning"),
            (
                "If Sarah has 5 apples and gives 2 to Tom, how many does she have left?",
                "reasoning",
            ),
            ("When was the Declaration of Independence signed?", "non-reasoning"),
            ("Explain step by step why compound interest works", "reasoning"),
            ("What does photosynthesis mean?", "non-reasoning"),
            ("Prove sqrt(2) is irrational", "reasoning"),
            ("My name is Ryan. I enjoy skiing.", "non-reasoning"),
        ]

        correct = 0
        for text, expected in test_cases:
            preds = model.predict(text, k=2)
            labels, probs = preds
            main_label = labels[0].replace("__label__", "")
            main_prob = probs[0]
            emoji = "🧮" if main_label == "reasoning" else "📚"
            check = "✅" if main_label == expected else "❌"
            if main_label == expected:
                correct += 1
            print(f"   {check} {emoji} '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"      → {main_label} ({main_prob:.3f})")

        print(
            f"\n   Accuracy on test cases: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)"
        )

        # Model size
        model_size_kb = (output_dir / "reasoning.bin").stat().st_size / 1024
        print(f"\n💾 Model size: {model_size_kb:.1f} KB")

        print("\n✅ Training complete!")
        print("\nNext steps:")
        print(f"  1. Model is ready at: {output_dir.absolute() / 'reasoning.bin'}")
        print("  2. Restart your frontend to use the new model")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
