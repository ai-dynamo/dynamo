# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.lora.manager.get_lora_manager singleton."""

import threading
import time

import pytest

from dynamo.common.lora import manager as manager_module
from dynamo.common.lora.manager import get_lora_manager
from dynamo.common.lora.once import OnceLock

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def fresh_singleton(monkeypatch):
    """Reset the module-level OnceLock so each test starts with no cached manager."""
    monkeypatch.setattr(manager_module, "_lora_manager", OnceLock())


class TestGetLoraManagerSingleton:
    def test_returns_none_when_disabled(self, fresh_singleton, monkeypatch):
        monkeypatch.delenv("DYN_LORA_ENABLED", raising=False)
        assert get_lora_manager() is None

    def test_returns_none_when_env_value_not_truthy(self, fresh_singleton, monkeypatch):
        monkeypatch.setenv("DYN_LORA_ENABLED", "false")
        assert get_lora_manager() is None

    def test_returns_same_instance_on_subsequent_calls(
        self, fresh_singleton, monkeypatch
    ):
        instance_count = 0

        class CountingLoRAManager:
            def __init__(self, cache_path=None):
                nonlocal instance_count
                instance_count += 1

        monkeypatch.setattr(manager_module, "LoRAManager", CountingLoRAManager)
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")

        first = get_lora_manager()
        second = get_lora_manager()
        assert first is not None
        assert first is second
        assert instance_count == 1

    def test_init_failure_propagates_and_allows_retry(
        self, fresh_singleton, monkeypatch
    ):
        """Init exceptions bubble up to the caller. OnceLock does not cache
        failures, so a subsequent call retries and can succeed."""
        attempts = 0

        class FlakyLoRAManager:
            def __init__(self, cache_path=None):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("first attempt fails")

        monkeypatch.setattr(manager_module, "LoRAManager", FlakyLoRAManager)
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")

        with pytest.raises(RuntimeError, match="first attempt fails"):
            get_lora_manager()
        assert attempts == 1

        second = get_lora_manager()
        assert second is not None
        assert isinstance(second, FlakyLoRAManager)
        assert attempts == 2


class TestGetLoraManagerConcurrency:
    def test_no_race_under_concurrent_init(self, fresh_singleton, monkeypatch):
        """Without the OnceLock adoption multiple worker threads would each
        observe the singleton as None and construct their own LoRAManager."""
        num_threads = 50
        instance_count = 0
        count_lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        class CountingLoRAManager:
            def __init__(self, cache_path=None):
                nonlocal instance_count
                with count_lock:
                    instance_count += 1
                time.sleep(0.01)  # widen race window

        monkeypatch.setattr(manager_module, "LoRAManager", CountingLoRAManager)
        monkeypatch.setenv("DYN_LORA_ENABLED", "true")

        results: list[object] = [None] * num_threads
        errors: list[Exception | None] = [None] * num_threads

        def worker(idx: int) -> None:
            try:
                barrier.wait()  # release all threads at once
                results[idx] = get_lora_manager()
            except Exception as exc:
                errors[idx] = exc

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for idx, exc in enumerate(errors):
            if exc is not None:
                raise AssertionError(f"worker {idx} raised") from exc

        assert instance_count == 1, (
            f"expected 1 LoRAManager instance, got {instance_count} — "
            "race condition not fixed"
        )
        assert all(r is results[0] for r in results)
        assert all(isinstance(r, CountingLoRAManager) for r in results)
