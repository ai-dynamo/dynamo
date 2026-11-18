# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip("gpu_memory_service is required", allow_module_level=True)
if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)
pytest.importorskip("vllm")

import gpu_memory_service  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

_SETTINGS = (
    "DYN_GMS_SCRATCH_KV_ENABLED",
    "DYN_GMS_VLLM_LOG_LEVEL",
    "MX_ENABLED",
)

_IMPORT_PROBE = """
import json, logging, sys
import gpu_memory_service.integrations.vllm.worker
from gpu_memory_service.common import vmm
from gpu_memory_service.integrations.common import patches as common_patches
from gpu_memory_service.integrations.vllm import patches
print(json.dumps({
    "patched": [
        patches._request_memory_patched,
        patches._register_kv_caches_patched,
        patches._memory_snapshot_patched,
        common_patches._empty_cache_patched,
    ],
    "device_detected": vmm._vmm_device_type is not None,
    "modelexpress_imported": "modelexpress" in sys.modules,
    "gms_log_handlers": len(logging.getLogger("gpu_memory_service").handlers),
}))
"""


@pytest.mark.timeout(30)  # ~8 s: a fresh interpreter imports vLLM
def test_importing_the_worker_module_decides_nothing():
    """Importing the worker module leaves every decision to the worker that
    starts later: with scratch KV and MX on, the import applies no patch,
    imports no modelexpress, detects no device and configures no logging."""
    lib_dir = str(Path(gpu_memory_service.__file__).resolve().parents[1])
    env = {k: v for k, v in os.environ.items() if k not in _SETTINGS}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [lib_dir, env.get("PYTHONPATH")]))
    env.update(DYN_GMS_SCRATCH_KV_ENABLED="1", MX_ENABLED="1")

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "patched": [False, False, False, False],
        "device_detected": False,
        "modelexpress_imported": False,
        "gms_log_handlers": 0,
    }
