#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate installed Nemotron Ultra vLLM runtime markers."""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path

import vllm

MARKER_GROUPS = {
    "semantic_kv_events": [
        "kv_cache_spec_kind",
        "kv_cache_spec_sliding_window",
        "get_kv_cache_spec_kind",
    ],
    "mamba_spec_decode_runtime": [
        "postprocess_mamba_fused_kernel",
        "MambaSpecDecodeGPUContext",
        "MambaBuffers",
        "postprocess_mamba_align_gpu",
    ],
    "pr42554_mamba_pd_runtime": [
        "Skip block alignment when setting up async receive",
        "Partial prefix cache hit for FA group",
    ],
    "hybrid_hash_block_events": [
        "_maybe_emit_sub_block_events",
        "hash_block_size",
        "BlockStored",
    ],
    "mtp_ds_copy": [
        "NEMOTRON_ULTRA_MTP_DS_COPY",
        "ds_conv_tail_copy",
    ],
    "ssm_nixl_tailfix": [
        "NEMOTRON_ULTRA_SSM_NIXL_TAILFIX",
        "SSM group has no local blocks to receive",
    ],
}


def find_marker(files: list[Path], root: Path, marker: str) -> str | None:
    for path in files:
        if marker in path.read_text(errors="ignore"):
            return str(path.relative_to(root))
    return None


def main() -> None:
    root = Path(vllm.__file__).resolve().parent
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    missing: dict[str, list[str]] = {}
    evidence: dict[str, dict[str, str | None]] = {}

    for group, markers in MARKER_GROUPS.items():
        group_evidence = {}
        for marker in markers:
            hit = find_marker(files, root, marker)
            group_evidence[marker] = hit
            if hit is None:
                missing.setdefault(group, []).append(marker)
        evidence[group] = group_evidence

    vllm_version = metadata.version("vllm")
    if vllm_version != "0.22.0":
        missing.setdefault("vllm_version", []).append(vllm_version)

    humming_spec = importlib.util.find_spec("humming")
    if humming_spec is None:
        missing.setdefault("humming", []).append("humming python package")

    payload = {
        "vllm_root": str(root),
        "vllm_version": vllm_version,
        "humming_origin": humming_spec.origin if humming_spec else None,
        "marker_evidence": evidence,
        "missing": missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(f"missing Nemotron Ultra runtime markers: {missing}")


if __name__ == "__main__":
    main()
