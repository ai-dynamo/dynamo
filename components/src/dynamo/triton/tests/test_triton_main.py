# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the dynamo.triton worker entry point (main.py):
_register_and_serve wires the RequestHandler into a slugified endpoint and
registers the tensor model with the Dynamo runtime."""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
import tritonclient.grpc.model_config_pb2 as mc

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.triton import main

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def patched_worker(monkeypatch):
    """Patch the worker's registration collaborators with mocks."""
    register_model = AsyncMock(name="register_model")
    # Stub text_format but keep Parse() -> a real ModelConfig so downstream
    # SerializeToString() yields bytes that _build_handler can parse.
    text_format_mock = MagicMock(name="text_format")
    text_format_mock.Parse.return_value = mc.ModelConfig()
    monkeypatch.setattr(main, "text_format", text_format_mock)
    monkeypatch.setattr(main, "register_model", register_model)
    return types.SimpleNamespace(register_model=register_model)


def _make_config(task: str = "tensor") -> MagicMock:
    """A ``DynamoTritonConfig``-shaped mock with the endpoint-selection attrs
    set explicitly. MagicMock would otherwise auto-magic ``config.task`` to a
    non-string sentinel that silently fails the ``config.task == "classify"``
    check in ``_register_and_serve``."""
    config = MagicMock(name="config")
    config.namespace = "dynamo"
    config.server_id = "triton"
    config.task = task
    config.classify_input_name = None
    config.classify_output_name = None
    return config


def test_register_and_serve_registers_and_serves(patched_worker, tmp_path):
    """The registration path slugifies the endpoint, registers the tensor model,
    and serves RequestHandler.generate bound to the loaded model."""
    model_name = "identity"
    (tmp_path / model_name).mkdir()
    (tmp_path / model_name / "config.pbtxt").write_text('name: "identity"\n')

    endpoint = MagicMock(name="endpoint")
    endpoint.serve_endpoint = AsyncMock()
    runtime = MagicMock(name="runtime")
    runtime.endpoint.return_value = endpoint
    config = _make_config(task="tensor")

    loaded_model = MagicMock(name="model")
    loaded_model.config.return_value = {}
    server = MagicMock(name="server")
    server.model.return_value = loaded_model

    asyncio.run(
        main._register_and_serve(runtime, config, server, str(tmp_path), model_name)
    )

    expected_path = (
        f"{config.namespace}.{config.server_id}.{main.endpoint_slug(model_name)}"
    )
    runtime.endpoint.assert_called_once_with(expected_path)
    server.model.assert_called_once_with(model_name)

    patched_worker.register_model.assert_awaited_once()
    reg_args, reg_kwargs = patched_worker.register_model.call_args
    assert reg_args[0] == main.ModelInput.Tensor
    assert reg_args[1] == main.ModelType.TensorBased
    assert reg_args[2] is endpoint
    assert reg_args[3] == model_name
    assert reg_kwargs["worker_type"] == main.WorkerType.Aggregated

    # register_model receives the tensor protocol layout via tensor_model_config.
    sent_config = reg_kwargs["tensor_model_config"]
    assert sent_config["name"] == ""
    assert sent_config["inputs"] == []
    assert sent_config["outputs"] == []
    assert "triton_model_config" in sent_config

    endpoint.serve_endpoint.assert_awaited_once()
    served = endpoint.serve_endpoint.call_args.args[0]
    # The served callable is RequestHandler.generate bound to the loaded model.
    assert served.__name__ == "generate"
    assert served.__self__._model is loaded_model
    assert served.__self__._server is server

    # A Triton-specific health check payload is registered with the endpoint so
    # framework probes emit backend-specific labels and carry the canary marker.
    served_kwargs = endpoint.serve_endpoint.call_args.kwargs
    health_check_payload = served_kwargs["health_check_payload"]
    assert health_check_payload["model"] == model_name
    assert health_check_payload[HEALTH_CHECK_KEY] is True


def test_register_and_serve_classify_task_wires_classify_handler(monkeypatch, tmp_path):
    from dynamo.triton.pooling_handlers import ClassifyWorkerHandler

    register_model = AsyncMock(name="register_model")
    monkeypatch.setattr(main, "register_model", register_model)

    model_name = "clf"
    (tmp_path / model_name).mkdir()
    (tmp_path / model_name / "config.pbtxt").write_text(
        'name: "clf"\n'
        "max_batch_size: 4\n"
        'input [{ name: "TEXT" data_type: TYPE_STRING dims: [-1] }]\n'
        'output [{ name: "probs" data_type: TYPE_FP32 dims: [-1] }]\n'
    )

    endpoint = MagicMock(name="endpoint")
    endpoint.serve_endpoint = AsyncMock()
    runtime = MagicMock(name="runtime")
    runtime.endpoint.return_value = endpoint
    config = _make_config(task="classify")

    loaded_model = MagicMock(name="model")
    # Empty runtime config forces _read_model_config to the disk-pbtxt path.
    loaded_model.config.return_value = {}
    loaded_model.name = model_name
    server = MagicMock(name="server")
    server.model.return_value = loaded_model

    asyncio.run(
        main._register_and_serve(runtime, config, server, str(tmp_path), model_name)
    )

    register_model.assert_awaited_once()
    reg_args, reg_kwargs = register_model.call_args
    assert reg_args[0] == main.ModelInput.Text
    assert reg_args[1] == main.ModelType.Classify
    assert reg_args[3] == model_name
    assert reg_kwargs["worker_type"] == main.WorkerType.Aggregated
    # Classify takes the asset-skip fast path (no HF resolve) and does
    # not attach the Triton protocol layout. The handler parses it
    # directly from config.pbtxt.
    assert reg_kwargs["skip_model_assets"] is True
    assert "tensor_model_config" not in reg_kwargs

    endpoint.serve_endpoint.assert_awaited_once()
    served = endpoint.serve_endpoint.call_args.args[0]
    assert served.__name__ == "generate"
    assert isinstance(served.__self__, ClassifyWorkerHandler)
    assert served.__self__._input_name == "TEXT"
    assert served.__self__._output_name == "probs"


def test_register_and_serve_missing_model_error(patched_worker, tmp_path):
    """A missing config.pbtxt surfaces as FileNotFoundError before registration."""
    runtime = MagicMock(name="runtime")
    server = MagicMock(name="server")
    config = MagicMock(name="config")
    config.namespace = "dynamo"
    config.server_id = "triton"

    with pytest.raises(FileNotFoundError):
        asyncio.run(
            main._register_and_serve(
                runtime, config, server, str(tmp_path), "absent_model"
            )
        )

    patched_worker.register_model.assert_not_awaited()


def _write_classify_ensemble_repo(
    root,
    ensemble_name: str = "classifier",
    tokenizer_name: str = "tokenizer",
    numeric_name: str = "numeric",
) -> None:
    """Write a minimal on-disk Triton repo with one ensemble and two
    dependency models, enough for ``_collect_classify_dependency_models``
    to walk.

    Shape mirrors a typical Triton classify ensemble: the ensemble is the
    only user-facing model (STRING in / FP32 out); the tokenizer has no
    FP32 output and the numeric stage has no STRING input, so constructing
    a ``ClassifyWorkerHandler`` for either raises and would cancel the
    sibling task that wraps the valid ensemble.
    """
    (root / tokenizer_name).mkdir()
    (root / tokenizer_name / "config.pbtxt").write_text(
        f'name: "{tokenizer_name}"\n'
        'backend: "python"\n'
        "max_batch_size: 4\n"
        'input [{ name: "TEXT" data_type: TYPE_STRING dims: [-1] }]\n'
        'output [{ name: "input_ids" data_type: TYPE_INT32 dims: [-1] }]\n'
    )
    (root / numeric_name).mkdir()
    (root / numeric_name / "config.pbtxt").write_text(
        f'name: "{numeric_name}"\n'
        'backend: "tensorrt"\n'
        "max_batch_size: 4\n"
        'input [{ name: "input_ids" data_type: TYPE_INT32 dims: [-1] }]\n'
        'output [{ name: "probs" data_type: TYPE_FP32 dims: [-1] }]\n'
    )
    (root / ensemble_name).mkdir()
    (root / ensemble_name / "config.pbtxt").write_text(
        f'name: "{ensemble_name}"\n'
        'platform: "ensemble"\n'
        "max_batch_size: 4\n"
        'input [{ name: "TEXT" data_type: TYPE_STRING dims: [-1] }]\n'
        'output [{ name: "probs" data_type: TYPE_FP32 dims: [-1] }]\n'
        "ensemble_scheduling {\n"
        "  step [\n"
        "    {\n"
        f'      model_name: "{tokenizer_name}"\n'
        "      model_version: -1\n"
        '      input_map { key: "TEXT" value: "TEXT" }\n'
        '      output_map { key: "input_ids" value: "tok_ids" }\n'
        "    },\n"
        "    {\n"
        f'      model_name: "{numeric_name}"\n'
        "      model_version: -1\n"
        '      input_map { key: "input_ids" value: "tok_ids" }\n'
        '      output_map { key: "probs" value: "probs" }\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _write_standalone_classifier_repo(root, model_name: str = "clf") -> None:
    """Write a repo with a single, standalone STRING->FP32 classifier."""
    (root / model_name).mkdir()
    (root / model_name / "config.pbtxt").write_text(
        f'name: "{model_name}"\n'
        "max_batch_size: 4\n"
        'input [{ name: "TEXT" data_type: TYPE_STRING dims: [-1] }]\n'
        'output [{ name: "probs" data_type: TYPE_FP32 dims: [-1] }]\n'
    )


@pytest.fixture
def init_worker_env(monkeypatch):
    """Patch out ``init_worker``'s side-effecting collaborators so the test
    only exercises the model-filter and dispatch logic.

    ``_register_and_serve`` is replaced with an AsyncMock so each test can
    assert exactly which model names reached registration without running
    the handler, Dynamo endpoint, or TaskGroup fan-out logic.
    """
    register_and_serve = AsyncMock(name="_register_and_serve")
    monkeypatch.setattr(main, "_register_and_serve", register_and_serve)

    # Avoid actually starting Triton; init_worker only needs the server object
    # to answer models() / model() calls, both set per-test.
    server_cls = MagicMock(name="TritonServer")
    monkeypatch.setattr(main, "TritonServer", server_cls)

    # Metrics bridge and log callback would try to touch the real Triton
    # server; stub them out to keep the test environment hermetic.
    monkeypatch.setattr(
        main, "_register_triton_metrics_bridge", MagicMock(return_value=None)
    )
    monkeypatch.setattr(main, "_triton_supports_log_callback", lambda: False)

    return types.SimpleNamespace(
        register_and_serve=register_and_serve,
        server_cls=server_cls,
    )


def _make_server_with_models(server_cls: MagicMock, model_names: list[str]):
    """Wire the patched ``TritonServer`` to report ``model_names`` as ready.

    An empty runtime config on every looked-up model makes
    ``_collect_classify_dependency_models``'s primary (runtime) scan a
    no-op, so it falls back to the ``config.pbtxt`` files written under
    ``tmp_path``.
    """
    server = server_cls.return_value
    server.models.return_value = [(n, 1) for n in model_names]
    server.model.return_value.config.return_value = {}
    return server


def _make_init_worker_config(
    tmp_path, task: str = "tensor", metrics: bool = False
) -> MagicMock:
    """``DynamoTritonConfig``-shaped mock for ``init_worker``. Metrics is
    opt-in to keep the common path from touching the stubbed metrics bridge."""
    config = _make_config(task=task)
    config.model_repository = str(tmp_path)
    config.metrics = metrics
    config.to_server_options = MagicMock(return_value={})
    return config


def test_init_worker_classify_filters_ensemble_dependencies(init_worker_env, tmp_path):
    """Dependencies are filtered before the TaskGroup fans out so a
    failing-to-construct handler for a dep cannot cancel the valid
    ensemble sibling task."""
    _write_classify_ensemble_repo(tmp_path)
    _make_server_with_models(
        init_worker_env.server_cls, ["classifier", "numeric", "tokenizer"]
    )
    config = _make_init_worker_config(tmp_path, task="classify")

    asyncio.run(main.init_worker(MagicMock(name="runtime"), config))

    # Only the ensemble reaches _register_and_serve.
    registered = [
        call.args[4] for call in init_worker_env.register_and_serve.call_args_list
    ]
    assert registered == ["classifier"]


def test_init_worker_classify_registers_standalone_model(init_worker_env, tmp_path):
    """Pin the no-op path: a future filter change must not silently
    strip the only user-facing model when no ensembles are present."""
    _write_standalone_classifier_repo(tmp_path, model_name="clf")
    _make_server_with_models(init_worker_env.server_cls, ["clf"])
    config = _make_init_worker_config(tmp_path, task="classify")

    asyncio.run(main.init_worker(MagicMock(name="runtime"), config))

    registered = [
        call.args[4] for call in init_worker_env.register_and_serve.call_args_list
    ]
    assert registered == ["clf"]


def test_init_worker_classify_raises_when_only_dependencies_present(
    init_worker_env, tmp_path
):
    """When every ready model is an ensemble dependency, the worker
    must raise rather than start with zero endpoints and report healthy
    to the orchestrator."""
    _write_classify_ensemble_repo(tmp_path)
    # Ensemble loaded but not-ready, so only the two deps appear in
    # server.models() and the exposed set collapses to empty.
    _make_server_with_models(init_worker_env.server_cls, ["numeric", "tokenizer"])
    config = _make_init_worker_config(tmp_path, task="classify")

    with pytest.raises(RuntimeError, match="No user-facing classify"):
        asyncio.run(main.init_worker(MagicMock(name="runtime"), config))

    init_worker_env.register_and_serve.assert_not_awaited()


def test_init_worker_tensor_task_registers_every_model_in_ensemble_repo(
    init_worker_env, tmp_path, monkeypatch
):
    """``--task tensor`` must not invoke the classify-only filter. A
    dependency model may still be useful to call directly over KServe
    gRPC for debugging, so the tensor path registers every ready model
    unchanged."""
    _write_classify_ensemble_repo(tmp_path)
    _make_server_with_models(
        init_worker_env.server_cls, ["classifier", "numeric", "tokenizer"]
    )
    config = _make_init_worker_config(tmp_path, task="tensor")

    collect_spy = MagicMock(wraps=main._collect_classify_dependency_models)
    monkeypatch.setattr(main, "_collect_classify_dependency_models", collect_spy)

    asyncio.run(main.init_worker(MagicMock(name="runtime"), config))

    registered = sorted(
        call.args[4] for call in init_worker_env.register_and_serve.call_args_list
    )
    assert registered == ["classifier", "numeric", "tokenizer"]
    # The filter must not run on the tensor path.
    collect_spy.assert_not_called()
