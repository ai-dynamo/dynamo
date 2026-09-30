# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.lora.once module."""

import threading
import time

import pytest

from dynamo.common.lora.once import OnceLock

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestOnceLockBasicAPI:
    def test_get_returns_none_before_init(self):
        once: OnceLock[int] = OnceLock()
        assert once.get() is None

    def test_get_returns_value_after_init(self):
        once: OnceLock[int] = OnceLock()
        once.get_or_init(lambda: 42)
        assert once.get() == 42

    def test_get_or_init_returns_cached_value_on_subsequent_calls(self):
        once: OnceLock[int] = OnceLock()
        calls = 0

        def factory() -> int:
            nonlocal calls
            calls += 1
            return 7

        assert once.get_or_init(factory) == 7
        assert once.get_or_init(factory) == 7
        assert once.get_or_init(lambda: 999) == 7
        assert calls == 1


class TestOnceLockConcurrency:
    def test_get_or_init_runs_factory_exactly_once_under_concurrent_threads(self):
        """Without double-checked locking the slow factory would be invoked
        by multiple threads and `calls` would exceed 1."""
        num_threads = 50
        once: OnceLock[int] = OnceLock()
        calls = 0
        calls_lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def factory() -> int:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.01)  # widen race window
            return 42

        results: list[int | None] = [None] * num_threads
        errors: list[Exception | None] = [None] * num_threads

        def worker(idx: int) -> None:
            try:
                barrier.wait()  # release all threads at once
                results[idx] = once.get_or_init(factory)
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

        assert calls == 1, f"factory called {calls} times under {num_threads} threads"
        assert all(r == 42 for r in results)


class TestOnceLockFactoryExceptions:
    def test_factory_exception_propagates_and_leaves_value_unset(self):
        once: OnceLock[int] = OnceLock()

        def boom() -> int:
            raise RuntimeError("init failed")

        with pytest.raises(RuntimeError, match="init failed"):
            once.get_or_init(boom)

        assert once.get() is None

    def test_retry_after_failed_init_succeeds(self):
        once: OnceLock[int] = OnceLock()
        attempts = 0

        def flaky() -> int:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("first attempt fails")
            return 99

        with pytest.raises(RuntimeError):
            once.get_or_init(flaky)
        assert once.get_or_init(flaky) == 99
        assert attempts == 2
