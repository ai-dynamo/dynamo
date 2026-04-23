# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module-scoped teardown: drop CUDA state held by the pytest process itself.

The GMS tests import ``gpu_memory_service`` into the pytest interpreter, which
initializes CUDA state that survives normal test teardown (the pytest process
keeps running). Subsequent tests (e.g. kvbm) then see the GPU partially
occupied with no owning process. Release PyTorch's CUDA cache + IPC handles
after each GMS test module so memory is returned before the next module runs.
"""

from __future__ import annotations

import gc

import pytest


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
        # Never let cleanup failures mask real test failures.
        pass
    gc.collect()
