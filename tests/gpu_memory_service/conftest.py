# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-scoped teardown: drop CUDA state held by the pytest process itself.

gpu_memory_service opens CUDA contexts via the driver API in the pytest
interpreter (we can see ~150 /dev/nvidiactl + /dev/nvidia-uvm fds in
/proc/<pytest>/fd after the tests). ~2.4 GiB of device memory stays pinned
to those contexts for the rest of the pytest session unless we explicitly
tell the driver to release them. PyTorch cache/IPC cleanup doesn't help
because the library doesn't use PyTorch's allocator.

Call ``cuDevicePrimaryCtxReset`` on the device after each GMS module to
drop the primary context and all its allocations. If GMS uses non-primary
contexts this is a no-op; kvbm CI will tell us whether memory.used drops.
"""

from __future__ import annotations

import gc
import logging

import pytest

logger = logging.getLogger(__name__)


def _reset_cuda_primary_context() -> None:
    """Best-effort: reset the primary CUDA context on device 0.

    Works across the cuda-python 11.x ``cuda.cuda`` API and 12.x
    ``cuda.bindings.driver`` API; both return either a CUresult enum or a
    (CUresult, ...) tuple depending on version, so we normalize.
    """
    driver = None
    try:
        from cuda.bindings import driver as driver
    except ImportError:
        try:
            from cuda import cuda as driver  # type: ignore[no-redef]
        except ImportError:
            logger.info("cuda-python not available; skipping primary ctx reset")
            return

    def _first(result):
        return result[0] if isinstance(result, tuple) else result

    logger.info("cuInit(0) -> %s", _first(driver.cuInit(0)))
    logger.info(
        "cuDevicePrimaryCtxReset(0) -> %s",
        _first(driver.cuDevicePrimaryCtxReset(0)),
    )


@pytest.fixture(scope="module", autouse=True)
def _release_gms_cuda_state():
    yield
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        _reset_cuda_primary_context()
    except Exception as exc:
        logger.warning("cuDevicePrimaryCtxReset failed: %s", exc)

    gc.collect()
