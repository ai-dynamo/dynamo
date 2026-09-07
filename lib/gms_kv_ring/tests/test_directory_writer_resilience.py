# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The async directory publish worker must not die on a single mutation failure.

A transient daemon error -- or this engine being fenced/demoted mid-drain --
previously set a fatal error, cleared the queue and stopped the worker thread,
silently ending ALL future directory publications for the process. That is a
permanent prefix-cache-publishing cliff. The worker must instead skip the failed
mutation (a safe cache miss) and keep serving the queue.
"""

from __future__ import annotations

import pytest
from gms_kv_ring.common.content_directory import ContentDirectory

pytestmark = pytest.mark.pre_merge


def test_publish_worker_survives_a_failing_mutation():
    directory = ContentDirectory(
        "/tmp/gms-directory-writer-resilience.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    try:
        calls: list[list[dict]] = []

        def flaky_publish(items):
            calls.append(items)
            if len(calls) == 1:
                raise RuntimeError("simulated transient publish failure")
            return len(items)

        # Drive the worker through the real enqueue path with a controlled publish.
        directory.publish = flaky_publish  # type: ignore[method-assign]
        directory._defer_mutation("publish", [{"a": 1}])
        directory._defer_mutation("publish", [{"b": 2}])

        # The worker must survive the first failure and process the second, and
        # flush must return True (not raise, not block) despite the skipped one.
        assert directory.flush_deferred(timeout=5.0) is True
        assert len(calls) == 2, "worker died instead of continuing past the failure"
        assert directory._mutation_failed == 1
        assert (
            directory._mutation_error is None
        ), "per-mutation failure must not be fatal"
        assert (
            directory._mutation_thread is not None
            and directory._mutation_thread.is_alive()
        )
    finally:
        directory.close()
