#!/usr/bin/env python3
"""Export CodeIsAbstract/ReasoningTextClassifier to ONNX format.

This script downloads the HuggingFace model and exports it to ONNX format
for use with the Dynamo semantic router.

Requirements:
    pip install transformers optimum[onnxruntime]

Usage:
    python export_reasoning_classifier.py [output_dir]

Default output directory: ./reasoning-classifier-onnx
"""

import os
import sys

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


def export_model(output_dir="./reasoning-classifier-onnx"):
    """Export the ReasoningTextClassifier model to ONNX format.

    Args:
        output_dir: Directory to save the exported model and tokenizer

    Returns:
        Path to the output directory
    """
    model_id = "CodeIsAbstract/ReasoningTextClassifier"

    print(f"📥 Downloading and exporting {model_id} to ONNX...")
    print("   This may take a few minutes on first run...")

    try:
        # Load and export to ONNX
        model = ORTModelForSequenceClassification.from_pretrained(
            model_id,
            export=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✅ Model exported successfully to {output_dir}/")
        print("\nFiles created:")
        print("  📄 model.onnx          - ONNX model file")
        print("  📄 tokenizer.json      - Tokenizer configuration")
        print("  📄 config.json         - Model configuration")
        print("\nNext steps:")
        print("  1. Set environment variables:")
        print(f"     export SEMROUTER_MODEL_PATH={output_dir}/model.onnx")
        print(f"     export SEMROUTER_TOKENIZER_PATH={output_dir}/tokenizer.json")
        print("  2. Build with: maturin develop --uv --features onnx-classifier")
        print("  3. See SEMANTIC_ROUTER.md for complete setup instructions")

        return output_dir

    except Exception as e:
        print(f"\n❌ Error during export: {e}", file=sys.stderr)
        print("\nMake sure you have the required packages installed:")
        print("  pip install transformers optimum[onnxruntime]")
        sys.exit(1)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "./reasoning-classifier-onnx"
    export_model(output)
