# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import pickle
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from dynamo._core import ModelProtectionError
from dynamo.vllm import protection_bootstrap as bootstrap

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


class _Session:
    model_path = "/run/test-models/session"

    def __init__(self) -> None:
        self.cleaned = False
        self.materialized = False

    def cleanup(self) -> None:
        self.cleaned = True

    def materialize(self) -> None:
        self.materialized = True

    def cancellation(self):
        return SimpleNamespace(cancel=lambda: None)


def test_plain_model_does_not_enter_protection_pipeline(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_PLUGINS", "plain-plugin")
    monkeypatch.setattr(bootstrap, "is_protected_model", lambda _: False)
    monkeypatch.setattr(
        bootstrap,
        "prepare_protected_model",
        lambda *_: pytest.fail("plain model entered protected preparation"),
    )

    prepared = bootstrap.prepare_model_argv(["--model", "Qwen/Qwen3-0.6B"])

    assert prepared.argv == ["--model", "Qwen/Qwen3-0.6B"]
    assert not prepared.protected
    assert os.environ["VLLM_PLUGINS"] == "plain-plugin"


def test_protected_model_rewrites_path_and_preserves_served_identity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_PLUGINS", "untrusted-plugin")
    session = _Session()
    monkeypatch.setattr(bootstrap, "is_protected_model", lambda _: True)
    monkeypatch.setattr(bootstrap, "prepare_protected_model", lambda *_: session)

    prepared = bootstrap.prepare_model_argv(
        [
            "--model=/packages/model",
            "--namespace",
            "test",
            "--model-protection-config",
            "/runtime/config.json",
        ]
    )

    assert prepared.argv[0] == "--model=/run/test-models/session"
    assert prepared.argv[-2:] == ["--served-model-name", "/packages/model"]
    assert os.environ["VLLM_PLUGINS"] == ""
    asyncio.run(bootstrap.materialize_protected_weights(prepared, asyncio.Event()))
    assert session.materialized
    prepared.cleanup()
    assert session.cleaned
    assert os.environ["VLLM_PLUGINS"] == "untrusted-plugin"


def test_protected_model_rejects_unsupported_loader_before_preparation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(bootstrap, "is_protected_model", lambda _: True)
    monkeypatch.setattr(
        bootstrap,
        "prepare_protected_model",
        lambda *_: pytest.fail("unsupported mode released protected state"),
    )

    with pytest.raises(ModelProtectionError, match="MODEL_PROTECTION_MODE_UNSUPPORTED"):
        bootstrap.prepare_model_argv(
            ["--model", "/packages/model", "--trust-remote-code"]
        )


@pytest.mark.parametrize("option", ["--trust-remote-cod", "--conf"])
def test_protected_model_rejects_abbreviated_forbidden_option(
    monkeypatch, option: str
) -> None:
    monkeypatch.setattr(bootstrap, "is_protected_model", lambda _: True)
    monkeypatch.setattr(
        bootstrap,
        "prepare_protected_model",
        lambda *_: pytest.fail("abbreviated mode released protected state"),
    )

    with pytest.raises(ModelProtectionError, match="MODEL_PROTECTION_MODE_UNSUPPORTED"):
        bootstrap.prepare_model_argv(["--model", "/packages/model", option])


def test_protected_model_restores_plugin_policy_when_preparation_fails(
    monkeypatch,
) -> None:
    def fail(*_args) -> None:
        raise ModelProtectionError("SECURE_PACKAGE_INVALID")

    monkeypatch.setenv("VLLM_PLUGINS", "plain-plugin")
    monkeypatch.setattr(bootstrap, "is_protected_model", lambda _: True)
    monkeypatch.setattr(bootstrap, "prepare_protected_model", fail)

    with pytest.raises(ModelProtectionError, match="SECURE_PACKAGE_INVALID"):
        bootstrap.prepare_model_argv(["--model", "/packages/model"])

    assert os.environ["VLLM_PLUGINS"] == "plain-plugin"


def test_protected_model_restores_plugin_policy_when_cleanup_fails(monkeypatch) -> None:
    class FailingSession(_Session):
        def cleanup(self) -> None:
            raise ModelProtectionError("CLEANUP_FAILED")

    monkeypatch.setenv("VLLM_PLUGINS", "plain-plugin")
    prepared = bootstrap.ProtectionBootstrap(
        [], FailingSession(), "/run/test-models/session", "test", "plain-plugin"
    )
    os.environ["VLLM_PLUGINS"] = ""

    with pytest.raises(ModelProtectionError, match="CLEANUP_FAILED"):
        prepared.cleanup()

    assert os.environ["VLLM_PLUGINS"] == "plain-plugin"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("model", "/outside/model", "MODEL_PROTECTION_CONFIG_INVALID"),
        ("tokenizer", "/outside/tokenizer", "MODEL_PROTECTION_CONFIG_INVALID"),
        ("scheduler_cls", "external.Scheduler", "MODEL_PROTECTION_MODE_UNSUPPORTED"),
        ("kv_transfer_config", object(), "MODEL_PROTECTION_MODE_UNSUPPORTED"),
        ("ec_transfer_config", object(), "MODEL_PROTECTION_MODE_UNSUPPORTED"),
    ],
)
def test_effective_gate_rejects_external_or_mutable_inputs(
    field: str, value: object, code: str
) -> None:
    session = _Session()
    prepared = bootstrap.ProtectionBootstrap([], session, session.model_path, "test")
    engine = SimpleNamespace(
        model=session.model_path,
        trust_remote_code=False,
        enable_lora=False,
        enable_sleep_mode=False,
        speculative_config=None,
        model_loader_extra_config=None,
        model_weights="",
        hf_config_path=None,
        io_processor_plugin=None,
        model_class_overrides={},
        hf_overrides={},
        logits_processors=None,
        weight_transfer_config=None,
        worker_extension_cls="",
        scheduler_cls=None,
        kv_transfer_config=None,
        nnodes=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend=None,
        worker_cls="auto",
        load_format="safetensors",
        tokenizer=session.model_path,
        config=None,
    )
    setattr(engine, field, value)
    config = SimpleNamespace(
        namespace="test",
        model=session.model_path,
        headless=False,
        enable_rl=False,
        enable_multimodal=False,
        embedding_worker=False,
        classify_worker=False,
        realtime=False,
        custom_encoder_class=None,
        custom_jinja_template=None,
        disaggregation_mode="agg",
        engine_args=engine,
    )

    with pytest.raises(ModelProtectionError, match=code):
        bootstrap.validate_protected_engine_args(config, prepared, "0.29.0")


def test_effective_gate_accepts_pinned_vllm_engine_args(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VLLM_CONFIG_ROOT", str(tmp_path))
    vllm = pytest.importorskip("vllm")
    if not vllm.__version__.startswith(bootstrap.SUPPORTED_VLLM_SERIES):
        pytest.skip(f"requires vLLM {bootstrap.SUPPORTED_VLLM_SERIES}x")
    from vllm.engine import arg_utils

    AsyncEngineArgs = pytest.importorskip("vllm.engine.arg_utils").AsyncEngineArgs
    from vllm.platforms.cpu import CpuPlatform
    from vllm.usage.usage_lib import UsageContext

    session = _Session()
    session.model_path = str(
        Path(__file__).resolve().parents[2]
        / "replay/tests/e2e/configs/unified_cli/fixtures/tiny-model"
    )
    prepared = bootstrap.ProtectionBootstrap([], session, session.model_path, "test")
    monkeypatch.setattr(arg_utils, "current_platform", CpuPlatform())
    engine = AsyncEngineArgs(
        model=session.model_path,
        tokenizer=session.model_path,
        load_format="safetensors",
        skip_tokenizer_init=True,
        enforce_eager=True,
    )
    config = SimpleNamespace(
        namespace="test",
        model=session.model_path,
        headless=False,
        enable_rl=False,
        enable_multimodal=False,
        embedding_worker=False,
        classify_worker=False,
        realtime=False,
        custom_encoder_class=None,
        custom_jinja_template=None,
        disaggregation_mode="agg",
        engine_args=engine,
    )

    bootstrap.validate_protected_engine_args(config, prepared, "0.29.0")
    normalized = engine.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    bootstrap.validate_protected_vllm_config(normalized, prepared)


def test_effective_gate_accepts_multimodal_vllm_engine(tmp_path) -> None:
    vllm = pytest.importorskip("vllm")
    if not vllm.__version__.startswith(bootstrap.SUPPORTED_VLLM_SERIES):
        pytest.skip(f"requires vLLM {bootstrap.SUPPORTED_VLLM_SERIES}x")
    AsyncEngineArgs = pytest.importorskip("vllm.engine.arg_utils").AsyncEngineArgs

    session = _Session()
    session.model_path = str(tmp_path)
    prepared = bootstrap.ProtectionBootstrap([], session, session.model_path, "test")
    engine = AsyncEngineArgs(
        model=session.model_path,
        tokenizer=session.model_path,
        load_format="safetensors",
    )
    config = SimpleNamespace(
        namespace="test",
        model=session.model_path,
        headless=False,
        enable_rl=False,
        enable_multimodal=True,
        embedding_worker=False,
        classify_worker=False,
        realtime=False,
        custom_encoder_class=None,
        custom_jinja_template=None,
        disaggregation_mode="agg",
        engine_args=engine,
    )

    bootstrap.validate_protected_engine_args(config, prepared, "0.29.0")


def test_effective_gate_rejects_unprofiled_vllm_version() -> None:
    session = _Session()
    prepared = bootstrap.ProtectionBootstrap([], session, session.model_path, "test")

    with pytest.raises(ModelProtectionError, match="MODEL_PROTECTION_MODE_UNSUPPORTED"):
        bootstrap.validate_protected_engine_args(object(), prepared, "0.28.0")


def test_normalized_gate_allows_only_dynamo_multimodal_cache() -> None:
    session = _Session()
    prepared = bootstrap.ProtectionBootstrap([], session, session.model_path, "test")
    allowed = SimpleNamespace(
        engine_id="test.backend.backend.0",
        ec_role="ec_both",
        ec_connector="DynamoMultimodalEmbeddingCacheConnector",
        ec_connector_module_path=(
            "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector"
        ),
        ec_connector_extra_config={
            "multimodal_embedding_cache_capacity_gb": 1.0,
            "component": "backend",
            "model_name": "model",
        },
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=session.model_path,
            model_weights="",
            tokenizer=session.model_path,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        load_config=SimpleNamespace(load_format="safetensors"),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(scheduler_cls=None),
        ec_transfer_config=allowed,
    )

    bootstrap.validate_protected_vllm_config(config, prepared)
    allowed.ec_connector_module_path = "external.module"
    with pytest.raises(ModelProtectionError, match="MODEL_PROTECTION_MODE_UNSUPPORTED"):
        bootstrap.validate_protected_vllm_config(config, prepared)


def test_engine_core_policy_is_installed_and_runs_in_child(monkeypatch) -> None:
    calls: list[str] = []

    class EngineCoreProc:
        @staticmethod
        def run_engine_core(value: str) -> str:
            calls.append(value)
            return value

    module = ModuleType("vllm.v1.engine.core")
    module.EngineCoreProc = EngineCoreProc
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", module)
    import dynamo._core as core

    monkeypatch.setattr(
        core,
        "enforce_model_protection_process_policy",
        lambda: calls.append("policy"),
        raising=False,
    )

    bootstrap.install_protected_engine_core_policy()
    try:
        child_entry = pickle.loads(pickle.dumps(EngineCoreProc.run_engine_core))
        assert child_entry("target") == "target"
        assert calls == ["policy", "target"]
    finally:
        bootstrap.uninstall_protected_engine_core_policy()
    assert EngineCoreProc.run_engine_core("restored") == "restored"
    assert calls[-1] == "restored"


def test_shutdown_cancels_and_joins_materialization_before_return() -> None:
    started = threading.Event()
    stopped = threading.Event()

    class Cancellation:
        def cancel(self) -> None:
            stopped.set()

    class SlowSession(_Session):
        def cancellation(self):
            return Cancellation()

        def materialize(self) -> None:
            started.set()
            assert stopped.wait(1)

    async def run() -> None:
        shutdown = asyncio.Event()
        prepared = bootstrap.ProtectionBootstrap([], SlowSession(), "/model", "test")
        task = asyncio.create_task(
            bootstrap.materialize_protected_weights(prepared, shutdown)
        )
        await asyncio.to_thread(started.wait, 1)
        shutdown.set()
        await task
        assert stopped.is_set()

    asyncio.run(run())


def test_task_cancellation_stops_and_joins_materialization() -> None:
    started = threading.Event()
    stopped = threading.Event()
    finished = threading.Event()

    class Cancellation:
        def cancel(self) -> None:
            stopped.set()

    class SlowSession(_Session):
        def cancellation(self):
            return Cancellation()

        def materialize(self) -> None:
            started.set()
            assert stopped.wait(1)
            finished.set()

    async def run() -> None:
        prepared = bootstrap.ProtectionBootstrap([], SlowSession(), "/model", "test")
        task = asyncio.create_task(
            bootstrap.materialize_protected_weights(prepared, asyncio.Event())
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
        assert finished.is_set()

    asyncio.run(run())


def test_repeated_task_cancellation_still_joins_materialization() -> None:
    started = threading.Event()
    first_cancel = threading.Event()
    release = threading.Event()
    writer_finished = threading.Event()

    class Cancellation:
        calls = 0

        def cancel(self) -> None:
            self.calls += 1
            if self.calls == 1:
                first_cancel.set()
            else:
                release.set()

    cancellation = Cancellation()

    class SlowSession(_Session):
        def cancellation(self):
            return cancellation

        def materialize(self) -> None:
            started.set()
            assert release.wait(1)
            writer_finished.set()

    async def run() -> None:
        prepared = bootstrap.ProtectionBootstrap([], SlowSession(), "/model", "test")
        task = asyncio.create_task(
            bootstrap.materialize_protected_weights(prepared, asyncio.Event())
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.to_thread(first_cancel.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancellation.calls >= 2
        assert writer_finished.is_set()

    asyncio.run(run())


def test_cancellation_preserves_caller_error_after_materializer_failure() -> None:
    started = threading.Event()
    stopped = threading.Event()
    finished = threading.Event()

    class Cancellation:
        def cancel(self) -> None:
            stopped.set()

    class FailingSession(_Session):
        def cancellation(self):
            return Cancellation()

        def materialize(self) -> None:
            started.set()
            assert stopped.wait(1)
            finished.set()
            raise RuntimeError("writer failed while cancellation was pending")

    async def run() -> None:
        prepared = bootstrap.ProtectionBootstrap([], FailingSession(), "/model", "test")
        task = asyncio.create_task(
            bootstrap.materialize_protected_weights(prepared, asyncio.Event())
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(run())


def test_shutdown_and_task_cancellation_join_materialization() -> None:
    started = threading.Event()
    stopped = threading.Event()
    writer_finished = threading.Event()

    class Cancellation:
        def cancel(self) -> None:
            stopped.set()

    class SlowSession(_Session):
        def cancellation(self):
            return Cancellation()

        def materialize(self) -> None:
            started.set()
            assert stopped.wait(1)
            writer_finished.set()

    async def run() -> None:
        shutdown = asyncio.Event()
        prepared = bootstrap.ProtectionBootstrap([], SlowSession(), "/model", "test")
        task = asyncio.create_task(
            bootstrap.materialize_protected_weights(prepared, shutdown)
        )
        await asyncio.to_thread(started.wait, 1)
        shutdown.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert writer_finished.is_set()

    asyncio.run(run())
