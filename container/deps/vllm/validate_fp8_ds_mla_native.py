#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate the native fp8_ds_mla UE8M0 writer fix.

The default mode is safe during an image build and verifies that the installed
native vLLM library contains the new runtime switch.  ``--gpu`` additionally
runs the CUDA writer and checks the four stored tile scales on an SM10x GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import vllm
import vllm._flashmla_C  # noqa: F401


def native_library() -> Path:
    package_dir = Path(vllm.__file__).resolve().parent
    matches = list(package_dir.glob("_C_stable_libtorch*.so"))
    assert len(matches) == 1, f"expected one stable native library, found {matches}"
    return matches[0]


def validate_static() -> Path:
    assert vllm.__version__ == "0.26.0", vllm.__version__
    assert torch.__version__ == "2.11.0+cu130", torch.__version__
    assert torch.version.cuda == "13.0", torch.version.cuda
    library = native_library()
    marker = b"VLLM_DS_MLA_UE8M0_SCALE"
    assert marker in library.read_bytes(), f"{marker!r} is absent from {library}"
    return library


def validate_gpu() -> dict[str, object]:
    assert torch.cuda.is_available(), "CUDA is not available"
    capability = torch.cuda.get_device_capability()
    assert capability[0] == 10, f"expected SM10x, got compute capability {capability}"

    from vllm import _custom_ops as ops

    maxima = [100.0, 80.0, 50.0, 20.0]
    kv_c = torch.cat(
        [torch.full((128,), value, dtype=torch.bfloat16) for value in maxima]
    ).reshape(1, 512).cuda()
    k_pe = torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros((1, 1, 656), dtype=torch.uint8, device="cuda")
    slot_mapping = torch.tensor([0], dtype=torch.int64, device="cuda")
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")

    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache,
        slot_mapping,
        kv_cache_dtype="fp8_ds_mla",
        scale=scale,
    )
    torch.cuda.synchronize()

    # Layout: 512 E4M3 bytes, four fp32 tile scales, then 64 BF16 RoPE values.
    actual = cache[0, 0, 512:528].view(torch.float32).cpu().tolist()
    raw = [value / 448.0 for value in maxima]
    rounded = [math.exp2(math.ceil(math.log2(value))) for value in raw]
    override = os.getenv("VLLM_DS_MLA_UE8M0_SCALE")
    enabled = override != "0"  # SM10x default is enabled.
    expected = rounded if enabled else raw

    for got, want in zip(actual, expected, strict=True):
        assert math.isclose(got, want, rel_tol=2e-5, abs_tol=1e-8), (
            f"stored scale {got} does not match expected {want}; "
            f"override={override!r}"
        )

    return {
        "compute_capability": list(capability),
        "override": override,
        "ue8m0_enabled": enabled,
        "raw_scales": raw,
        "expected_scales": expected,
        "actual_scales": actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    library = validate_static()
    result: dict[str, object] = {
        "native_library": str(library),
        "vllm_version": vllm.__version__,
        "native_marker_present": True,
    }
    if args.gpu:
        result["gpu_validation"] = validate_gpu()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
