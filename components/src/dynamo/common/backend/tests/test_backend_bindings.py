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

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


# Import-time skip: if the extension hasn't been built, all tests below
# are skipped rather than crashing the collection phase.
core = pytest.importorskip(
    "dynamo._core",
    reason="dynamo._core not built — run `maturin develop` first",
)
backend = pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo.common.backend.engine import LlmRegistration  # noqa: E402


def test_module_exposes_expected_classes():
    """The five binding classes must all be importable as top-level
    attributes of ``dynamo._core.backend``."""
    for name in (
        "Worker",
        "WorkerConfig",
        "EngineConfig",
        "RuntimeConfig",
        "EngineMetrics",
    ):
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
    assert cfg.model_aliases == []
    assert cfg.llm is None


@pytest.mark.unified
def test_engine_config_full_kwargs_round_trip_through_getters():
    cfg = backend.EngineConfig(
        model="m2",
        served_model_name="m2-serving",
        model_aliases=["m2-alias"],
        runtime_data={"sglang_worker_group_id": "group-a"},
        llm=backend.LlmRegistration(
            context_length=2048,
            kv_cache_block_size=16,
            total_kv_blocks=1000,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
            enable_eagle=True,
        ),
    )
    assert cfg.model == "m2"
    assert cfg.served_model_name == "m2-serving"
    assert cfg.model_aliases == ["m2-alias"]
    assert cfg.runtime_data == {"sglang_worker_group_id": "group-a"}
    llm = cfg.llm
    assert llm.context_length == 2048
    assert llm.kv_cache_block_size == 16
    assert llm.total_kv_blocks == 1000
    assert llm.max_num_seqs == 64
    assert llm.max_num_batched_tokens == 2048
    assert llm.enable_eagle is True


@pytest.mark.unified
def test_llm_registration_preserves_legacy_positional_arguments():
    llm = backend.LlmRegistration(2048, 16, 1000, 64, 2048, 2, 1, "host", 9000)
    assert llm.bootstrap_host == "host"
    assert llm.bootstrap_port == 9000
    assert llm.enable_eagle is False


@pytest.mark.parametrize("kwargs", [{}, {"enable_eagle": True}])
@pytest.mark.unified
def test_llm_registration_dataclass_matches_binding(kwargs):
    registration = LlmRegistration(**kwargs)
    cfg = backend.EngineConfig(
        model="eagle-model",
        llm=backend.LlmRegistration(**asdict(registration)),
    )
    assert cfg.llm.enable_eagle is kwargs.get("enable_eagle", False)


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
        default_thinking_mode="disabled",
        exclude_tools_when_tool_choice_none=False,
        enable_local_indexer=False,
    )


def test_worker_config_preserves_legacy_positional_argument_order():
    """New optional fields must be appended after every existing argument."""
    backend.WorkerConfig(
        "dynamo",  # namespace
        "backend",  # component
        "generate",  # endpoint
        "",  # model_name
        None,  # served_model_name
        core.ModelInput.Tokens,  # model_input
        "chat,completions",  # endpoint_types
        None,  # custom_jinja_template
        None,  # tool_call_parser
        None,  # reasoning_parser
        False,  # exclude_tools_when_tool_choice_none
        False,  # enable_local_indexer
    )


@pytest.mark.unified
def test_python_worker_config_preserves_legacy_positional_argument_order():
    from dynamo.common.backend.worker import WorkerConfig

    config = WorkerConfig(
        "dynamo",  # namespace
        "backend",  # component
        "generate",  # endpoint
        "",  # model_name
        None,  # served_model_name
        core.ModelInput.Tokens,  # model_input
        "chat,completions",  # endpoint_types
        "etcd",  # discovery_backend
        "tcp",  # request_plane
        None,  # event_plane
        False,  # use_kv_events
        None,  # custom_jinja_template
        None,  # tool_call_parser
        None,  # reasoning_parser
        False,  # exclude_tools_when_tool_choice_none
        False,  # enable_local_indexer
    )

    assert config.exclude_tools_when_tool_choice_none is False
    assert config.enable_local_indexer is False
    assert config.default_thinking_mode is None


def test_worker_config_accepts_media_configuration():
    """Unified registration can advertise frontend media decoding."""
    from dynamo.llm import MediaDecoder, MediaFetcher

    backend.WorkerConfig(
        namespace="dynamo",
        media_decoder=MediaDecoder(),
        media_fetcher=MediaFetcher(),
    )


def test_worker_config_accepts_disaggregation_mode():
    """The Rust binding must accept a DisaggregationMode kwarg so the
    Python shim can plumb the field through. Each variant must construct."""
    for mode in (
        backend.DisaggregationMode.Aggregated,
        backend.DisaggregationMode.Prefill,
        backend.DisaggregationMode.Decode,
    ):
        backend.WorkerConfig(namespace="dynamo", disaggregation_mode=mode)


@pytest.mark.unified
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
    runtime_cfg.dyn_default_thinking_mode = "disabled"
    runtime_cfg.exclude_tools_when_tool_choice_none = False
    runtime_cfg.enable_local_indexer = False
    runtime_cfg.dyn_enable_structural_tag = True
    runtime_cfg.dyn_structural_tag_scope = "always"
    runtime_cfg.dyn_structural_tag_schema = "strict"
    # MagicMock auto-attrs would be rejected as a foreign type by the
    # strict coercer; pin them to None.
    runtime_cfg.disaggregation_mode = None
    runtime_cfg.serving_mode = None

    config = WorkerConfig.from_runtime_config(runtime_cfg, "nvidia/Kimi-K2.5-NVFP4")

    assert config.tool_call_parser == "kimi_k2"
    assert config.reasoning_parser == "kimi_k25"
    assert config.default_thinking_mode == "disabled"
    assert config.exclude_tools_when_tool_choice_none is False
    assert config.enable_local_indexer is False
    assert config.structural_tag_mode == "on"
    assert config.structural_tag_scope == "always"
    assert config.structural_tag_schema == "strict"


@pytest.mark.unified
def test_python_worker_config_from_runtime_config_applies_defaults_when_fields_absent():
    from dynamo.common.backend.worker import WorkerConfig

    class _BareRuntime:
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = "nats"

    cfg = WorkerConfig.from_runtime_config(_BareRuntime(), model_name="m")

    assert cfg.component == "backend"
    assert cfg.endpoint == "generate"
    assert cfg.endpoint_types == "chat,completions"
    assert cfg.use_kv_events is False
    assert cfg.custom_jinja_template is None
    assert cfg.default_thinking_mode is None
    assert cfg.structural_tag_mode == "off"
    assert cfg.structural_tag_scope == "auto"
    assert cfg.structural_tag_schema == "auto"


@pytest.mark.unified
def test_python_worker_config_from_runtime_config_overrides_win():
    from dynamo.common.backend.worker import WorkerConfig

    class _WithComponent:
        namespace = "ns"
        component = "from-runtime"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = "nats"

    cfg = WorkerConfig.from_runtime_config(
        _WithComponent(), model_name="m", component="from-override"
    )

    assert cfg.component == "from-override"


@pytest.mark.unified
def test_python_worker_config_picks_up_disaggregation_mode_from_runtime_config():
    from dynamo.common.backend.worker import WorkerConfig
    from dynamo.common.constants import DisaggregationMode

    class _Prefill:
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = "nats"
        # vLLM/TRT-LLM use this name; the helper's primary lookup path.
        disaggregation_mode = DisaggregationMode.PREFILL

    cfg = WorkerConfig.from_runtime_config(_Prefill(), model_name="m")
    assert cfg.disaggregation_mode is DisaggregationMode.PREFILL


@pytest.mark.unified
def test_python_worker_config_falls_back_to_serving_mode_for_sglang():
    from dynamo.common.backend.worker import WorkerConfig
    from dynamo.common.constants import DisaggregationMode

    class _Sglang:
        # SGLang stores the resolved mode under `serving_mode` rather than
        # `disaggregation_mode`. The from_runtime_config helper must probe
        # both names so both backends round-trip without per-backend wiring.
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = "nats"
        serving_mode = DisaggregationMode.DECODE

    cfg = WorkerConfig.from_runtime_config(_Sglang(), model_name="m")
    assert cfg.disaggregation_mode is DisaggregationMode.DECODE


@pytest.mark.unified
def test_python_worker_config_defaults_to_aggregated_when_runtime_lacks_mode():
    from dynamo.common.backend.worker import WorkerConfig
    from dynamo.common.constants import DisaggregationMode

    class _NoMode:
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = "nats"

    cfg = WorkerConfig.from_runtime_config(_NoMode(), model_name="m")
    assert cfg.disaggregation_mode is DisaggregationMode.AGGREGATED


@pytest.mark.unified
def test_python_worker_config_coerces_foreign_disaggregation_mode_enum_by_name():
    """Foreign enum on `runtime_cfg` (e.g. TRT-LLM's local
    `DisaggregationMode`) is coerced by `.name` when a member with the
    same name exists on `dynamo.common.constants.DisaggregationMode`.
    An explicit `disaggregation_mode=` override still wins."""
    import enum

    from dynamo.common.backend.worker import WorkerConfig
    from dynamo.common.constants import DisaggregationMode

    class _ForeignMode(enum.Enum):
        AGGREGATED = "prefill_and_decode"
        PREFILL = "prefill"

    class _RuntimeWithForeignMode:
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = "nats"
        disaggregation_mode = _ForeignMode.PREFILL

    cfg = WorkerConfig.from_runtime_config(_RuntimeWithForeignMode(), model_name="m")
    assert cfg.disaggregation_mode is DisaggregationMode.PREFILL

    # Explicit override still wins over the runtime_cfg field.
    cfg = WorkerConfig.from_runtime_config(
        _RuntimeWithForeignMode(),
        model_name="m",
        disaggregation_mode=DisaggregationMode.AGGREGATED,
    )
    assert cfg.disaggregation_mode is DisaggregationMode.AGGREGATED


@pytest.mark.unified
def test_python_worker_config_rejects_unrecognized_disaggregation_mode_value():
    """A non-enum or unrecognized name on `runtime_cfg.disaggregation_mode`
    raises TypeError so a typo-string can't silently degrade to AGG."""
    from dynamo.common.backend.worker import WorkerConfig

    class _RuntimeWithStringMode:
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = "nats"
        disaggregation_mode = "prefill"  # str, not enum

    with pytest.raises(TypeError, match="DisaggregationMode"):
        WorkerConfig.from_runtime_config(_RuntimeWithStringMode(), model_name="m")


@pytest.mark.unified
def test_python_worker_config_translates_all_disagg_modes():
    """Every variant of dynamo.common.constants.DisaggregationMode must map
    to a Rust binding value -- including ENCODE, which gained unified-path
    support. Regression for the prior `NotImplementedError`
    behavior where ENCODE was rejected at translation time."""
    from dynamo.common.backend.worker import _to_rust_disaggregation_mode
    from dynamo.common.constants import DisaggregationMode

    rust_mode_for = {
        DisaggregationMode.AGGREGATED: backend.DisaggregationMode.Aggregated,
        DisaggregationMode.PREFILL: backend.DisaggregationMode.Prefill,
        DisaggregationMode.DECODE: backend.DisaggregationMode.Decode,
        DisaggregationMode.ENCODE: backend.DisaggregationMode.Encode,
    }
    for py_mode, expected_rust in rust_mode_for.items():
        assert _to_rust_disaggregation_mode(py_mode) == expected_rust


@pytest.mark.unified
def test_python_worker_config_round_trips_route_to_encoder():
    """route_to_encoder must flow from runtime_cfg -> WorkerConfig dataclass
    -> Rust pyclass without being silently dropped at any layer. vLLM is the
    only Python backend with the field today; SGLang/TRT-LLM get False via
    the getattr default until they add the field."""
    from dynamo.common.backend.worker import WorkerConfig

    # Simulated vLLM-style runtime config that exposes the field.
    class _RuntimeWithRoute:
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = None
        route_to_encoder = True

    cfg = WorkerConfig.from_runtime_config(_RuntimeWithRoute(), model_name="m")
    assert cfg.route_to_encoder is True

    # Simulated SGLang/TRT-LLM-style runtime config that doesn't expose the
    # field yet -- the getattr default keeps it False so legacy behavior is
    # preserved until the backend adds the field on its own runtime config.
    class _RuntimeWithoutRoute:
        namespace = "ns"
        discovery_backend = "etcd"
        request_plane = "tcp"
        event_plane = None

    cfg_default = WorkerConfig.from_runtime_config(
        _RuntimeWithoutRoute(), model_name="m"
    )
    assert cfg_default.route_to_encoder is False

    # Verify the Rust pyclass accepts the kwarg at the end of its signature
    # (appended for backward-compat with positional callers).
    rust_cfg = backend.WorkerConfig(namespace="ns", route_to_encoder=True)
    assert rust_cfg is not None


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


# ---------------------------------------------------------------------------
# ShutdownConfig
# ---------------------------------------------------------------------------


def test_shutdown_config_accepts_partial_overrides():
    """Every field is optional: a caller states only what it changes, and the
    rest falls back to the environment."""
    cfg = backend.ShutdownConfig(total_secs=45.0, cleanup_timeout_secs=10.0)
    assert cfg is not None


def test_worker_config_accepts_a_nested_shutdown_config():
    """The nested field is what keeps a new knob from needing an edit in the
    Rust struct, the PyO3 signature and the dataclass."""
    cfg = backend.WorkerConfig(
        namespace="ns",
        shutdown=backend.ShutdownConfig(router_grace_secs=1.0),
    )
    assert cfg is not None


def test_worker_config_shutdown_defaults_to_none():
    """Omitting it must not be an error — every existing caller does."""
    assert backend.WorkerConfig(namespace="ns") is not None


@pytest.mark.parametrize("value", ["wait", "skip", "WAIT", " Skip "])
def test_shutdown_config_accepts_both_fallback_policies(value):
    assert backend.ShutdownConfig(kv_transfer_fallback=value) is not None


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 1e30, -1.0])
def test_shutdown_config_rejects_unrepresentable_durations(bad):
    """Rejected at construction because Duration::from_secs_f64 panics on a
    value it cannot represent — and a panic during shutdown aborts the drain.
    is_finite() alone is not enough: 1e30 is finite and still panics."""
    with pytest.raises(ValueError, match="finite"):
        backend.ShutdownConfig(total_secs=bad)


def test_shutdown_config_rejects_an_unknown_fallback_policy():
    """Parsed at construction, so a typo fails here rather than silently
    falling back to waiting at shutdown time, when nobody is watching."""
    with pytest.raises(ValueError, match="wait.*skip"):
        backend.ShutdownConfig(kv_transfer_fallback="nonsense")


def test_python_worker_config_forwards_shutdown_to_the_rust_config(monkeypatch):
    """The shim must pass the nested config through to `_backend.WorkerConfig`.

    Asserted by intercepting that call, not by reading the dataclass back:
    deleting the whole ``shutdown=`` block in ``worker.py`` left the previous
    version of this test green, so it protected nothing.
    """
    import asyncio

    from dynamo.common.backend import worker as worker_mod

    captured = {}

    def fake_worker_config(**kwargs):
        captured.update(kwargs)
        raise _StopBeforeRuntime

    monkeypatch.setattr(worker_mod._backend, "WorkerConfig", fake_worker_config)
    monkeypatch.setattr(worker_mod._backend, "RuntimeConfig", lambda **kwargs: object())

    cfg = worker_mod.WorkerConfig(
        namespace="ns",
        shutdown=worker_mod.ShutdownConfig(
            total_secs=30.0, kv_transfer_fallback="skip"
        ),
    )
    shim = worker_mod.Worker(MagicMock(), cfg)

    with pytest.raises(_StopBeforeRuntime):
        asyncio.run(shim.run())

    forwarded = captured["shutdown"]
    assert forwarded is not None, "the nested shutdown config was dropped"
    # It is a Rust pyclass by this point, so assert on what we can observe:
    # that it was constructed from our values rather than defaulted.
    assert isinstance(forwarded, core.backend.ShutdownConfig)


class _StopBeforeRuntime(Exception):
    """Aborts `Worker.run` once the config has been built, so the test never
    needs a live runtime."""


def test_python_worker_config_shutdown_defaults_are_all_unset():
    """Default construction must not pin any value, or it would override the
    environment for callers that never asked."""
    from dynamo.common.backend.worker import WorkerConfig as PyWorkerConfig

    shutdown = PyWorkerConfig(namespace="ns").shutdown
    assert asdict(shutdown) == {
        "total_secs": None,
        "router_grace_secs": None,
        "inflight_timeout_secs": None,
        "kv_transfer_timeout_secs": None,
        "cleanup_timeout_secs": None,
        "kv_transfer_fallback": None,
    }
