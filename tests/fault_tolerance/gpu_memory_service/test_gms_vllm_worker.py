# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.fault_tolerance,
]


def _run_worker_script(source: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"worker subprocess failed with rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_initialize_from_config_uses_kv_cache_gms_scope():
    result = _run_worker_script(
        """
        import json
        import os
        import sys
        from contextlib import contextmanager

        from gpu_memory_service.integrations.vllm.worker import GMSWorker
        import gpu_memory_service.integrations.vllm.worker as worker_module

        create_calls = []
        pool_calls = []
        kv_transfer_calls = []
        kv_init_calls = []

        @contextmanager
        def fake_use_mem_pool(scope, device):
            pool_calls.append([scope, str(device)])
            yield

        def fake_get_or_create(socket_path, device, mode, *, scope, tag, timeout_ms=None):
            create_calls.append(
                [socket_path, device, mode.value, scope, tag, timeout_ms]
            )
            return object()

        def fake_ensure_initialized(vllm_config, kv_cache_config):
            kv_transfer_calls.append(kv_cache_config)

        worker_module.gms_use_mem_pool = fake_use_mem_pool
        worker_module.get_or_create_gms_client_memory_manager = fake_get_or_create
        worker_module.get_socket_path = lambda device, scope: f"/tmp/{scope}-{device}.sock"

        import vllm.distributed.kv_transfer as kv_transfer
        kv_transfer.ensure_kv_transfer_initialized = fake_ensure_initialized

        worker = object.__new__(GMSWorker)
        worker.local_rank = 3
        worker.vllm_config = type(
            "Config",
            (),
            {"model_config": type("ModelConfig", (), {"enable_sleep_mode": True})()},
        )()
        worker.model_runner = type(
            "Runner",
            (),
            {"initialize_kv_cache": lambda self, cfg: kv_init_calls.append(cfg)},
        )()

        worker.initialize_from_config("kv-config")

        print(
            json.dumps(
                {
                    "create_calls": create_calls,
                    "pool_calls": pool_calls,
                    "kv_transfer_calls": kv_transfer_calls,
                    "kv_init_calls": kv_init_calls,
                }
            )
        )
        sys.stdout.flush()
        os._exit(0)
        """
    )

    assert result["create_calls"] == [
        [result["create_calls"][0][0], 3, "rw", "kv_cache", "kv_cache", None]
    ]
    assert result["pool_calls"] == [["kv_cache", "cuda:3"]]
    assert result["kv_transfer_calls"] == ["kv-config"]
    assert result["kv_init_calls"] == ["kv-config"]


def test_sleep_level_2_unmaps_weights_and_kv_cache():
    result = _run_worker_script(
        """
        import json
        import os
        import sys

        from gpu_memory_service.integrations.vllm.worker import GMSWorker
        import gpu_memory_service.integrations.vllm.worker as worker_module

        class FakeManager:
            def __init__(self):
                self.is_unmapped = False
                self.calls = []

            def unmap_all_vas(self):
                self.calls.append("unmap_all_vas")
                self.is_unmapped = True

            def disconnect(self):
                self.calls.append("disconnect")

        weights = FakeManager()
        kv_cache = FakeManager()
        free_bytes = iter([(2 << 30, 8 << 30), (5 << 30, 8 << 30)])

        worker_module.get_gms_client_memory_manager = (
            lambda scope: weights if scope == "weights" else kv_cache
        )
        worker_module.torch.cuda.mem_get_info = lambda: next(free_bytes)

        worker = object.__new__(GMSWorker)
        worker.sleep(level=2)

        print(json.dumps({"weights": weights.calls, "kv_cache": kv_cache.calls}))
        sys.stdout.flush()
        os._exit(0)
        """
    )

    assert result["weights"] == ["unmap_all_vas", "disconnect"]
    assert result["kv_cache"] == ["unmap_all_vas", "disconnect"]


def test_wake_up_remaps_weights_and_reallocates_kv_cache():
    result = _run_worker_script(
        """
        import json
        import os
        import sys

        from gpu_memory_service.common.types import RequestedLockType
        from gpu_memory_service.integrations.vllm.worker import GMSWorker
        import gpu_memory_service.integrations.vllm.worker as worker_module

        class FakeManager:
            def __init__(self):
                self.is_unmapped = True
                self.calls = []

            def connect(self, lock_type):
                self.calls.append(["connect", lock_type.value])

            def reallocate_all_handles(self, *, tag):
                self.calls.append(["reallocate_all_handles", tag])

            def remap_all_vas(self):
                self.calls.append("remap_all_vas")
                self.is_unmapped = False

        weights = FakeManager()
        kv_cache = FakeManager()
        fp8_calls = []

        worker_module.get_gms_client_memory_manager = (
            lambda scope: weights if scope == "weights" else kv_cache
        )

        worker = object.__new__(GMSWorker)
        worker.cache_config = type("CacheConfig", (), {"cache_dtype": "fp8_e4m3"})()
        worker.model_runner = type(
            "Runner",
            (),
            {"init_fp8_kv_scales": lambda self: fp8_calls.append("fp8")},
        )()
        worker.wake_up(["weights", "kv_cache"])

        print(
            json.dumps(
                {
                    "weights": weights.calls,
                    "kv_cache": kv_cache.calls,
                    "fp8_calls": fp8_calls,
                    "expected_ro": RequestedLockType.RO.value,
                    "expected_rw": RequestedLockType.RW.value,
                }
            )
        )
        sys.stdout.flush()
        os._exit(0)
        """
    )

    assert result["weights"] == [
        ["connect", result["expected_ro"]],
        "remap_all_vas",
    ]
    assert result["kv_cache"] == [
        ["connect", result["expected_rw"]],
        ["reallocate_all_handles", "kv_cache"],
        "remap_all_vas",
    ]
    assert result["fp8_calls"] == ["fp8"]
