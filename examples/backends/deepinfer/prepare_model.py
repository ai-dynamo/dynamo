# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare real V4.1 tokenizer assets and mock model metadata; never fetch weights."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

RECIPE_REVISION = "935817c66f5d6e64838f419692545f06787e485f"
TOKENIZER_SHA256 = "81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99"
TOKENIZER_URL = (
    "https://raw.githubusercontent.com/deepseek-ai/deepseek-recipe/"
    f"{RECIPE_REVISION}/static/tokenizers/v41/tokenizer.json"
)


def prepare(model_dir: Path, tokenizer_json: Path | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "config.json").exists() and not (
        model_dir / "TOKENIZER_SOURCE.txt"
    ).exists():
        raise ValueError(
            "Refusing to overwrite an existing model; use a dedicated mock model directory"
        )
    destination = model_dir / "tokenizer.json"
    if tokenizer_json is not None:
        if tokenizer_json.resolve() != destination.resolve():
            shutil.copyfile(tokenizer_json, destination)
    elif not destination.exists():
        with urllib.request.urlopen(TOKENIZER_URL, timeout=60) as source:
            destination.write_bytes(source.read())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError(f"Unexpected V4.1 tokenizer hash: {digest}")
    tokenizer = Tokenizer.from_file(str(destination))
    bos, eos = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "max_position_embeddings": 8192,
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_token_id": tokenizer.token_to_id(bos),
        "eos_token_id": tokenizer.token_to_id(eos),
    }
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 8192,
        "bos_token": bos,
        "eos_token": eos,
    }
    for name, data in (
        ("config.json", config),
        ("tokenizer_config.json", tokenizer_config),
    ):
        (model_dir / name).write_text(json.dumps(data, indent=2) + "\n")
    (model_dir / "TOKENIZER_SOURCE.txt").write_text(
        f"Source: {TOKENIZER_URL}\nSHA256: {digest}\n"
        f"Notices: https://github.com/deepseek-ai/deepseek-recipe/blob/{RECIPE_REVISION}/static/tokenizers/README.md\n"
        "This directory contains mock model metadata and no model weights.\n"
    )
    print(model_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    args = parser.parse_args()
    prepare(args.model_dir, args.tokenizer_json)
