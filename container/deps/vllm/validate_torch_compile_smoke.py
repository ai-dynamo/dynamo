# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build-time regression guard for the host C/C++ toolchain.

torch.inductor / Triton JIT-compile kernels at *runtime* by shelling out to a
host compiler (``cc``/``gcc``/``g++`` + ``make``). When that toolchain is absent
the failure only surfaces on the first compile inside a live worker -- exactly
how QA hit ``InductorError: Failed to find C compiler`` in the 1.3.0 release
image after an over-broad ``apt-get autoremove`` swept gcc out of the codec
purge. This probe reproduces that path during the image build so a missing
toolchain aborts the build instead of shipping.

It runs CPU-only (no GPU at build time): the inductor CPU backend codegens a
fused kernel and compiles it with g++, which is the same compiler-discovery path
the GPU runtime exercises. ``suppress_errors = False`` forces a compiler error to
raise rather than silently fall back to eager.
"""

from __future__ import annotations

import shutil
import sys

import torch
import torch._dynamo as dynamo

REQUIRED_TOOLS = ("cc", "gcc", "g++", "make")


def _check_tools() -> None:
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "missing host compiler toolchain required by torch.inductor/Triton "
            f"runtime JIT: {', '.join(missing)}. Something removed it after the "
            "vLLM base image was built (check the ffmpeg/codec purge step)."
        )


def _check_torch_compile() -> None:
    dynamo.config.suppress_errors = False

    @torch.compile(backend="inductor", fullgraph=True)
    def f(x: torch.Tensor) -> torch.Tensor:
        return (x * 2 + 1).relu()

    out = f(torch.randn(64))
    if out.shape != (64,):
        raise RuntimeError(
            f"torch.compile smoke test failed: expected shape (64,), got {out.shape}"
        )


def main() -> int:
    _check_tools()
    _check_torch_compile()
    print("torch.compile CPU inductor smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
