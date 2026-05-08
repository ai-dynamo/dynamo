# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the ``dynamo._core.backend`` PyO3 bindings.

These tests verify the Rust → Python binding surface that
``dynamo.common.backend.Worker`` delegates to. They DO NOT exercise the
full lifecycle (which would require etcd, NATS, and a running event
loop) — that's covered by the Rust unit tests in
``lib/backend-common/src/worker.rs``. Here we just pin down the Python
constructor signatures and class identity so the shim in ``worker.py``
can't silently drift from the Rust types.

If the compiled extension hasn't been built (e.g. fresh checkout without
``maturin develop``), every test in the module skips with a clear hint.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


# Import-time skip: if the extension hasn't been built, all tests below
# are skipped rather than crashing the collection phase.
backend = pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)


def test_module_exposes_expected_classes():
    """The four binding classes must all be importable as top-level
    attributes of ``dynamo._core.backend``."""
    for name in ("Worker", "WorkerConfig", "EngineConfig", "RuntimeConfig"):
        assert hasattr(backend, name), f"missing {name} on dynamo._core.backend"


def test_runtime_config_accepts_optional_fields():
    """RuntimeConfig must construct with no args and with each field."""
    backend.RuntimeConfig()
    backend.RuntimeConfig(discovery_backend="etcd")
    backend.RuntimeConfig(request_plane="tcp")
    backend.RuntimeConfig(event_plane="zmq")
    backend.RuntimeConfig(
        discovery_backend="etcd",
        request_plane="tcp",
        event_plane="zmq",
    )


def test_engine_config_required_model_only():
    """EngineConfig only requires ``model``; the rest are optional."""
    cfg = backend.EngineConfig(model="m1")
    assert cfg.model == "m1"
    assert cfg.served_model_name is None
    assert cfg.context_length is None


def test_engine_config_full_kwargs_round_trip_through_getters():
    cfg = backend.EngineConfig(
        model="m2",
        served_model_name="m2-serving",
        context_length=2048,
        kv_cache_block_size=16,
        total_kv_blocks=1000,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
    )
    assert cfg.model == "m2"
    assert cfg.served_model_name == "m2-serving"
    assert cfg.context_length == 2048
    assert cfg.kv_cache_block_size == 16
    assert cfg.total_kv_blocks == 1000
    assert cfg.max_num_seqs == 64
    assert cfg.max_num_batched_tokens == 2048


def test_worker_config_minimum_args():
    """``namespace`` is the only required positional arg; the rest fall
    back to the same defaults the Rust ``WorkerConfig::default`` uses."""
    backend.WorkerConfig(namespace="dynamo")


def test_worker_config_accepts_metrics_labels_and_runtime():
    """metrics_labels takes a list of (key, value) tuples; runtime takes
    a RuntimeConfig (or None)."""
    rt = backend.RuntimeConfig(discovery_backend="mem", request_plane="tcp")
    backend.WorkerConfig(
        namespace="dynamo",
        metrics_labels=[("model", "m1"), ("zone", "us-east-1")],
        runtime=rt,
    )


def test_worker_config_accepts_parser_runtime_settings():
    """Parser and local-indexer settings from the Python shim must remain
    accepted by the Rust WorkerConfig binding."""
    backend.WorkerConfig(
        namespace="dynamo",
        tool_call_parser="kimi_k2",
        reasoning_parser="kimi_k25",
        exclude_tools_when_tool_choice_none=False,
        enable_local_indexer=False,
    )


def test_python_worker_config_from_runtime_config_copies_parser_settings():
    from dynamo.common.backend.worker import WorkerConfig

    runtime_cfg = MagicMock()
    runtime_cfg.namespace = "test"
    runtime_cfg.component = None
    runtime_cfg.endpoint = None
    runtime_cfg.endpoint_types = "chat,completions"
    runtime_cfg.discovery_backend = "etcd"
    runtime_cfg.request_plane = "tcp"
    runtime_cfg.event_plane = "nats"
    runtime_cfg.use_kv_events = False
    runtime_cfg.custom_jinja_template = None
    runtime_cfg.dyn_tool_call_parser = "kimi_k2"
    runtime_cfg.dyn_reasoning_parser = "kimi_k25"
    runtime_cfg.exclude_tools_when_tool_choice_none = False
    runtime_cfg.enable_local_indexer = False

    config = WorkerConfig.from_runtime_config(runtime_cfg, "nvidia/Kimi-K2.5-NVFP4")

    assert config.tool_call_parser == "kimi_k2"
    assert config.reasoning_parser == "kimi_k25"
    assert config.exclude_tools_when_tool_choice_none is False
    assert config.enable_local_indexer is False


def test_worker_constructor_requires_engine_config_loop():
    """Worker takes (engine, WorkerConfig, event_loop). Missing args
    must surface as TypeError, not a downstream runtime panic."""

    class _Stub:
        async def start(self):
            return None

        async def generate(self, request, context):
            yield {}

        async def cleanup(self):
            return None

    with pytest.raises(TypeError):
        backend.Worker()  # type: ignore[call-arg]

    cfg = backend.WorkerConfig(namespace="dynamo")
    with pytest.raises(TypeError):
        backend.Worker(_Stub(), cfg)  # type: ignore[call-arg]


def test_worker_config_accepts_default_model_input():
    """ModelInput.Tokens is the default — engines that don't pass it must
    still construct cleanly so the Python shim's defaults are usable."""
    backend.WorkerConfig(namespace="dynamo")
