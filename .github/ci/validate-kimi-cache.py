# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check pinned Kimi snapshots without loading weights or writing the shared cache."""

import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def validate_weights(snapshot: Path, *, require_index: bool) -> None:
    json.loads((snapshot / "config.json").read_text())
    index = snapshot / "model.safetensors.index.json"
    if require_index or index.is_file():
        shards = {
            snapshot / name
            for name in json.loads(index.read_text())["weight_map"].values()
        }
    else:
        shards = set(snapshot.glob("*.safetensors"))
        if len(shards) != 1:
            raise ValueError(f"Expected a single Eagle weight file in {snapshot}")
    if not shards:
        raise ValueError(f"No weight shards in {snapshot}")
    for shard in shards:
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"Missing or empty weight shard: {shard}")
    print(f"Validated {len(shards)} weight shards in {snapshot}", flush=True)


if __name__ == "__main__":
    model, eagle = map(Path, sys.argv[1:])
    validate_weights(model, require_index=True)
    validate_weights(eagle, require_index=False)
    AutoConfig.from_pretrained(
        str(model), trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), trust_remote_code=True, local_files_only=True
    )
    if not tokenizer.encode("Hello world"):
        raise ValueError(f"Tokenizer produced no tokens: {model}")
    print("Offline model config/tokenizer and Eagle weights validated", flush=True)
