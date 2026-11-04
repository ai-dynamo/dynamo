#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backport of vllm-project/vllm#57662 (fixes #57661) onto the installed vLLM.

NIXL KV-region registration deduplicated transfer regions by ``base_addr`` alone. On
DeepSeek-V4.1-Flash the compressor ``state_cache`` and the indexer ``k_cache`` are overlaid
on the ``swa_cache`` allocation with a different ``block_len``, so they were folded into the
SWA region and PD disaggregation transferred truncated, misaligned state: silent garbled
output with HTTP 200. The fix keys regions by ``(base_addr, block_len)``.

The change is in vLLM main since 2026-10-29 and is NOT in v0.31.0, v0.32.0 or v0.33.0. Remove
this script and its Dockerfile step once the base image's vLLM contains #57662.

The script is idempotent (a no-op when the fix is already present) and fails the image build
loudly if the installed file no longer matches the expected shape, so a base-image bump can
never ship a silently unpatched or half-patched connector.
"""
import importlib.util
import pathlib
import sys

REL = "distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py"

# (old, new): each `old` must occur exactly once in the unpatched file.
SUBSTITUTIONS = [
    (
        "        seen_base_addresses: list[int] = []\n",
        "        seen_base_addresses: list[int] = []\n"
        "        seen_region_keys: list[tuple[int, int]] = []\n",
    ),
    (
        "                if base_addr in seen_base_addresses and not route_packed_layers:\n"
        "                    region_index = seen_base_addresses.index(base_addr)\n",
        "                region_key = (base_addr, block_len)\n"
        "                if region_key in seen_region_keys and not route_packed_layers:\n"
        "                    region_index = seen_region_keys.index(region_key)\n",
    ),
    (
        "                    seen_base_addresses.append(base_addr)\n",
        "                    seen_base_addresses.append(base_addr)\n"
        "                    seen_region_keys.append(region_key)\n",
    ),
]


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        print("ERROR: vllm is not installed", file=sys.stderr)
        return 1
    path = pathlib.Path(list(spec.submodule_search_locations)[0]) / REL
    src = path.read_text()
    if "seen_region_keys" in src:
        print(f"vllm#57662 already present in {path}; nothing to do")
        return 0
    for old, new in SUBSTITUTIONS:
        count = src.count(old)
        if count != 1:
            print(
                f"ERROR: expected exactly one match for the vllm#57662 anchor, found {count}:\n{old}",
                file=sys.stderr,
            )
            return 1
        src = src.replace(old, new)
    compile(src, str(path), "exec")  # refuse to write a file that does not parse
    path.write_text(src)
    print(f"vllm#57662 backport applied to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
