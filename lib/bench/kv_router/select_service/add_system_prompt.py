# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepend one globally shared system prompt to every request of a mooncake-format trace.

The prompt's hash ids start above the trace's largest id, so they collide with nothing.
Usage: add_system_prompt.py IN OUT PROMPT_HASH_IDS [TOKENS_PER_HASH_ID]"""
import json
import sys


def main(src, dst, prompt_ids, tokens_per_id=64):
    prompt_ids, tokens_per_id = int(prompt_ids), int(tokens_per_id)
    rows = [json.loads(line) for line in open(src) if line.strip()]
    base = max(max(r["hash_ids"], default=0) for r in rows) + 1
    prompt = list(range(base, base + prompt_ids))
    with open(dst, "w") as out:
        for row in rows:
            row["hash_ids"] = prompt + row["hash_ids"]
            if "input_length" in row:
                row["input_length"] += prompt_ids * tokens_per_id
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"{len(rows)} rows, prompt ids {base}..{base + prompt_ids - 1}")


if __name__ == "__main__":
    main(*sys.argv[1:])
