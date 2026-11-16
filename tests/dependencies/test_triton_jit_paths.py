# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard: Triton's NVIDIA backend works without the TRITON_*_PATH image ENV.

The upstream tensorrt-llm/release base ships NVIDIA's NGC PyTorch Triton build,
which has no triton/backends/nvidia/{include,bin} trees (no cuda.h, ptxas,
cuobjdump, nvdisasm) and relies on image ENV instead (TRITON_CUDACRT_PATH,
TRITON_CUDART_PATH, TRITON_PTXAS_PATH, ...). Multinode worker-pod ranks are
started by mpirun over ssh and never see image ENV, only the operator's -x
allowlist, so Triton's cuda_utils JIT there failed with
"fatal error: cuda.h: No such file or directory" (GH-14864 / NVBug 6772753).
trtllm_runtime.Dockerfile restores the wheel layout with symlinks; this test
keeps a base-image bump or Dockerfile refactor from silently reintroducing the
environment dependency.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = [
    # Let the 30s subprocess timeout report its failure before pytest interrupts.
    pytest.mark.timeout(60),
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None
        or importlib.util.find_spec("triton") is None,
        reason="TRT-LLM images only (other frameworks ship the PyPI Triton wheel)",
    ),
]

# Variables that would let gcc or Triton find cuda.h / the CUDA tools even when the
# package layout is broken. The subprocess runs without them, like a worker-pod rank.
_ENV_HINTS = ("CPATH", "C_INCLUDE_PATH", "CUDA_HOME")


def _env_without_triton_hints(extra: dict[str, str]) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("TRITON_") and k not in _ENV_HINTS
    }
    env.update(extra)
    return env


def _run(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_nvidia_backend_paths_resolve_without_env():
    code = (
        "import os; from triton import knobs; from triton.backends.nvidia import driver; "
        "hdr = os.path.join(driver.include_dirs[0], 'cuda.h'); "
        "assert os.path.isfile(hdr), f'missing {hdr}'; "
        "print(knobs.nvidia.ptxas.path, knobs.nvidia.cuobjdump.path, knobs.nvidia.nvdisasm.path)"
    )
    result = _run(code, _env_without_triton_hints({}))
    assert result.returncode == 0, (
        "Triton's NVIDIA backend cannot resolve cuda.h or its CUDA tools without "
        "TRITON_*_PATH env vars; multinode worker-pod ranks never receive them. "
        "See the wheel-layout symlink RUN in trtllm_runtime.Dockerfile.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_cuda_utils_jit_compiles_without_env(tmp_path: pathlib.Path):
    stub = pathlib.Path("/usr/local/cuda/lib64/stubs/libcuda.so")
    if not stub.is_file():
        pytest.skip(
            "CUDA toolkit libcuda stub missing; cannot link cuda_utils without a GPU"
        )
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "libcuda.so.1").symlink_to(stub)
    env = _env_without_triton_hints(
        {
            # keep Triton's cache out of $HOME and off any pre-warmed cache
            "TRITON_HOME": str(tmp_path / "home"),
            "TRITON_CACHE_DIR": str(tmp_path / "cache"),
            # no GPU needed: link and load against the toolkit's driver stub
            "TRITON_LIBCUDA_PATH": str(stubs),
            "LD_LIBRARY_PATH": f"{stubs}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        }
    )
    code = "from triton.backends.nvidia.driver import CudaUtils; CudaUtils(); print('cuda_utils ok')"
    result = _run(code, env)
    assert result.returncode == 0, (
        "Triton's cuda_utils JIT failed without TRITON_*_PATH env vars (the multinode "
        "worker-pod environment); a base-image bump or Dockerfile refactor likely dropped "
        "the wheel-layout symlinks in trtllm_runtime.Dockerfile.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
