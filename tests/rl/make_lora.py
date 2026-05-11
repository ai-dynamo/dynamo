# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny untrained LoRA adapter for Qwen3-0.6B.

Used by the RL smoke test to exercise load_lora_adapter / unload_lora_adapter
without needing to download a pretrained adapter from HuggingFace.

Usage:
    python make_lora.py <output_dir>
"""

import os
import sys

# Force offline so we don't hit HF Hub (cached weights only).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: make_lora.py <output_dir>", file=sys.stderr)
        sys.exit(2)

    output_dir = sys.argv[1]
    model_name = "Qwen/Qwen3-0.6B"

    print(f"[make_lora] Loading base model {model_name}")
    base = AutoModelForCausalLM.from_pretrained(model_name)

    # Small rank, untrained — only meant to verify the load/unload code path.
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, config)
    peft_model.save_pretrained(output_dir)
    print(f"[make_lora] Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
