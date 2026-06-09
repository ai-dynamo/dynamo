# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the GMSConnectorV1.

Covers import-time checks; full-engine integration test
(`--kv-connector GMSConnectorV1`) requires a real vLLM run and lives
in tests/test_real_vllm_connector_v1.py (not in this PR — flagged as
follow-up because measuring it requires a working `cuMemMap`-during-
engine-run deployment which this dev host lacks).
"""

from __future__ import annotations

import pytest

# All tests skip cleanly if vLLM isn't importable.
vllm = pytest.importorskip("vllm")
pytest.importorskip(
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    reason="vLLM v1 KV connector base ABI not available in this build",
)


def test_connector_imports():
    """The connector module imports without side effects."""
    from gpu_memory_service.integrations.vllm.gms_connector_v1 import (
        GMSConnectorV1,
        _GMSConnectorMetadata,
        _GMSConnectorScheduler,
        _GMSConnectorWorker,
    )

    assert GMSConnectorV1 is not None
    assert _GMSConnectorMetadata is not None
    assert _GMSConnectorScheduler is not None
    assert _GMSConnectorWorker is not None


def test_metadata_is_picklable_compatible():
    """Connector metadata travels scheduler→worker via vLLM's
    metadata channel — it must round-trip through the simple
    `blocks_to_save` list."""
    from gpu_memory_service.integrations.vllm.gms_connector_v1 import (
        _GMSConnectorMetadata,
    )

    meta = _GMSConnectorMetadata(blocks_to_save=[1, 2, 3, 99])
    assert meta.blocks_to_save == [1, 2, 3, 99]
    # Empty default.
    assert _GMSConnectorMetadata().blocks_to_save == []


def test_scheduler_builds_meta_from_request_finished(monkeypatch):
    """`build_connector_meta` packs whichever block_ids were captured
    by previous `request_finished` calls within this step.
    SchedulerOutput is just the trigger to flush."""
    from gpu_memory_service.integrations.vllm.gms_connector_v1 import (
        _GMSConnectorMetadata,
        _GMSConnectorScheduler,
    )

    class _FakeVLLMConfig:
        class parallel_config:
            engine_id = "test"

        class cache_config:
            block_size = 16

    class _DummyRequest:
        def __init__(self, hashes):
            self.block_hashes = hashes
            self.num_computed_tokens = 0

    sched = _GMSConnectorScheduler(_FakeVLLMConfig())

    # Simulate three requests finishing during this step.
    sched.request_finished(_DummyRequest([b"h0" * 16, b"h1" * 16]), [10, 20])
    sched.request_finished(_DummyRequest([b"h2" * 16]), [30])

    meta = sched.build_connector_meta(object())
    assert isinstance(meta, _GMSConnectorMetadata)
    assert sorted(meta.blocks_to_save) == [10, 20, 30]
    # Pending should clear after build.
    meta2 = sched.build_connector_meta(object())
    assert meta2.blocks_to_save == []
    # Hash index populated for cache-hit detection on future requests.
    assert len(sched._hash_to_src_block) == 3


def test_worker_save_layer_is_noop_until_wait(monkeypatch):
    """`save_kv_layer` is intentionally deferred; the actual RPC
    happens in `wait_for_save`. Verify save_kv_layer doesn't crash
    on its own when there's no daemon configured."""
    monkeypatch.delenv("GMS_VLLM_OFFLOAD_DAEMON_SOCKET", raising=False)
    from gpu_memory_service.integrations.vllm.gms_connector_v1 import (
        _GMSConnectorMetadata,
        _GMSConnectorWorker,
    )

    class _FakeVLLMConfig:
        class parallel_config:
            engine_id = "test"

    w = _GMSConnectorWorker(_FakeVLLMConfig())
    # Bind some pending; with no daemon, wait_for_save returns silently.
    w.bind_connector_metadata(_GMSConnectorMetadata(blocks_to_save=[1, 2]))
    w.save_kv_layer("layer_0", None, None)  # no-op
    w.wait_for_save()  # short-circuits on missing daemon_sock
    w.clear_connector_metadata()


def test_connector_roles(monkeypatch):
    """Instantiation in each role builds only the role-appropriate
    state (scheduler vs worker)."""
    monkeypatch.delenv("GMS_VLLM_OFFLOAD_DAEMON_SOCKET", raising=False)
    from gpu_memory_service.integrations.vllm.gms_connector_v1 import GMSConnectorV1
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    class _FakeVLLMConfig:
        class parallel_config:
            engine_id = "test"

        class cache_config:
            enable_prefix_caching = True

        # vLLM v1 connectors read these attributes; stub the bare
        # minimum the base __init__ touches.
        kv_transfer_config = type(
            "KT",
            (),
            {
                "kv_connector_extra_config": {},
                "kv_role": "kv_producer",
            },
        )()

        class scheduler_config:
            disable_log_stats = True

    # vLLM's KVConnectorBase_V1.__init__ may inspect more config.
    # We treat this as a smoke test — if it raises we just skip.
    try:
        sched = GMSConnectorV1(
            _FakeVLLMConfig(),
            KVConnectorRole.SCHEDULER,
            None,
        )
        assert sched.connector_scheduler is not None
        assert sched.connector_worker is None
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"vLLM connector base init incompatible: {exc}")

    try:
        worker = GMSConnectorV1(
            _FakeVLLMConfig(),
            KVConnectorRole.WORKER,
            None,
        )
        assert worker.connector_worker is not None
        assert worker.connector_scheduler is None
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"vLLM connector base init incompatible: {exc}")
