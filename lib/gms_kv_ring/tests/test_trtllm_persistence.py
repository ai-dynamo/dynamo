# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM warm-restart persistence parity test.

Validates that the TRT-LLM connector inherits the same
prefix-tree warm-restart property as the vLLM and SGLang
connectors:

  - With GMS_TRTLLM_PREFIX_INDEX_SNAPSHOT pointing at a writable
    file, `request_finished` records prefixes into a PrefixIndex
    that auto-snapshots to disk.
  - A fresh scheduler pointed at the same file reloads the
    snapshot and immediately serves cache hits — without first
    serving any requests on the new instance."""

from __future__ import annotations

import os
from types import SimpleNamespace

from gpu_memory_service.integrations.trtllm.gms_connector import (
    GMSTrtllmKvCacheScheduler,
)


def _make_llm_args(block_size=4):
    return SimpleNamespace(
        kv_cache_config=SimpleNamespace(tokens_per_block=block_size),
    )


class _FakeRequest:
    def __init__(self, req_id, tokens, cache_salt_id=None):
        self.request_id = req_id
        self._tokens = list(tokens)
        self.prompt_token_ids = list(tokens)
        self.cache_salt_id = cache_salt_id
        self.context_current_position = 0

    def get_tokens(self, _beam):
        return self._tokens


def _make_sched(monkeypatch, snap_path, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    monkeypatch.setenv(
        "GMS_TRTLLM_PREFIX_INDEX_SNAPSHOT",
        snap_path,
    )
    # Stable engine_id across restarts — required for cache-hit
    # lookup to find entries persisted by a previous instance.
    # In production this is set per-deployment (one engine_id per GPU).
    monkeypatch.setenv("GMS_TRTLLM_ENGINE_ID", "trtllm-persist-test")
    # Force snapshot to flush after a single record.
    monkeypatch.setenv("GMS_KVR_PREFIX_INDEX_SNAPSHOT_THRESHOLD", "1")
    monkeypatch.setenv("GMS_KVR_PREFIX_INDEX_SNAPSHOT_INTERVAL", "0")
    return GMSTrtllmKvCacheScheduler(_make_llm_args(block_size=4))


def test_warm_restart_serves_cache_hit_without_re_recording(
    monkeypatch,
    tmp_path,
):
    snap = str(tmp_path / "prefix.snap")
    # First instance: record a prefix; snapshot flushes immediately.
    sched1 = _make_sched(monkeypatch, snap)
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    sched1.request_finished(_FakeRequest("r1", tokens), [100, 101])
    # Force a write (defensive — record() with snapshot_threshold=1
    # and interval=0 should have flushed already).
    sched1._prefix_index.snapshot()
    sched1.shutdown()
    assert os.path.exists(snap)

    # Second instance: rehydrate from disk.
    sched2 = _make_sched(monkeypatch, snap)
    # PrefixIndex should have loaded the entry.
    assert len(sched2._prefix_index) == 2
    # A fresh request for the same prefix gets the cache hit.
    extra, _ = sched2.get_num_new_matched_tokens(
        _FakeRequest("r2", tokens),
        0,
    )
    assert extra == 8  # 2 blocks × 4 tokens
    sched2.shutdown()


def test_corrupt_snapshot_falls_back_to_cold_start(monkeypatch, tmp_path):
    snap = str(tmp_path / "prefix.snap")
    with open(snap, "wb") as f:
        f.write(b"garbage" * 100)
    sched = _make_sched(monkeypatch, snap)
    # Cold-started despite the corrupt file.
    assert len(sched._prefix_index) == 0
    sched.shutdown()


def test_persistence_disabled_by_default(monkeypatch, tmp_path):
    """Without the env, no snapshot file is created."""
    monkeypatch.delenv("GMS_TRTLLM_PREFIX_INDEX_SNAPSHOT", raising=False)
    monkeypatch.delenv("GMS_KVR_PREFIX_INDEX_SNAPSHOT", raising=False)
    monkeypatch.setenv("GMS_TRTLLM_DAEMON_SOCKET", "")
    sched = GMSTrtllmKvCacheScheduler(
        SimpleNamespace(
            kv_cache_config=SimpleNamespace(tokens_per_block=4),
        ),
    )
    sched.request_finished(
        _FakeRequest("r1", [1, 2, 3, 4, 5, 6, 7, 8]),
        [100, 101],
    )
    # Nothing in the tmp_path.
    assert not os.listdir(str(tmp_path))
    sched.shutdown()
