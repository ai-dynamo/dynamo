# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler.scale_elastic_ep.

Two concerns: input validation (the TP-derived EP-size floor) and fail-fast
handling of a failed grow. vLLM does no rollback on a failed scale, so the
handler restarts the worker instead of recovering in process; the fail-fast
tests assert the restart is actually triggered (via ``_WorkerShutdown``), not
just that a value was returned.

``ray`` is stubbed via the ``stub_ray`` fixture (the scale path imports it
lazily and CI lacks it).
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.config")

from vllm.config import ParallelConfig, VllmConfig  # noqa: E402
from vllm.v1.engine.exceptions import EngineDeadError  # noqa: E402

from dynamo.vllm.dp_topology import get_dp_range_for_worker  # noqa: E402
from dynamo.vllm.engine_group import EngineGroupServing  # noqa: E402
from dynamo.vllm.handlers import BaseWorkerHandler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler(
    tensor_parallel_size: int,
    prefill_context_parallel_size: int = 1,
    data_parallel_size: int = 1,
) -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
                prefill_context_parallel_size=prefill_context_parallel_size,
                data_parallel_size=data_parallel_size,
                # Elastic EP enabled + Ray backend so the capability gate passes.
                enable_elastic_ep=True,
                data_parallel_backend="ray",
                data_parallel_external_lb=False,
                elastic_ep_auto_bootstrap=False,
            ),
        ),
        scale_elastic_ep=AsyncMock(),
    )
    handler._scale_ep_lock = asyncio.Lock()
    handler._scale_ep_in_progress = False
    handler.engine_group_serving = None
    return handler


async def _successful_serving_probe():
    yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=[1])])


@pytest.fixture
def native_elastic_ep_middleware():
    return pytest.importorskip("vllm.entrypoints.serve.elastic_ep.middleware")


@pytest.mark.parametrize(
    ("scaling", "retired", "can_serve"),
    [(False, False, True), (True, False, False), (False, True, False)],
)
def test_managed_worker_honors_native_admission(
    monkeypatch, native_elastic_ep_middleware, scaling, retired, can_serve
):
    handler = _make_handler(tensor_parallel_size=1)
    handler._pause_controller = SimpleNamespace(is_paused=False)
    handler.engine_client.is_running = True
    handler.engine_client.errored = False
    monkeypatch.setattr(native_elastic_ep_middleware, "_scaling_elastic_ep", scaling)
    monkeypatch.setattr(
        native_elastic_ep_middleware, "_elastic_ep_rank_retired", retired
    )

    assert handler.can_serve_engine_group() is can_serve


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [1, 2, 4])
async def test_managed_probe_covers_every_local_engine_without_controller_action(width):
    handler = _make_handler(tensor_parallel_size=1)
    handler._pause_lock = asyncio.Lock()
    handler.dp_range = (2, width)
    engine = handler.engine_client
    engine.get_external_elastic_ep_status = AsyncMock(
        return_value={
            "instance_id": "engine-2",
            "operation_id": None,
            "epoch": None,
            "phase": "idle",
            "requested_data_parallel_size": None,
            "effective_data_parallel_size": 3,
            "topology_known": True,
            "error": None,
        }
    )
    engine.generate = Mock(
        side_effect=lambda *args, **kwargs: _successful_serving_probe()
    )
    engine.check_health = AsyncMock()
    handler.can_serve_engine_group = Mock(return_value=True)
    admission = EngineGroupServing(
        engine,
        Mock(
            register_endpoint_instance=AsyncMock(),
            unregister_endpoint_instance=AsyncMock(),
            connection_id=Mock(return_value=7),
        ),
        "pod-2",
        2,
        Mock(instance_ids=Mock(return_value=[])),
        handler.can_serve_engine_group,
        handler.probe_engine_group,
        rank_count=width,
    )
    admission.listener_started()
    handler.engine_group_serving = admission

    result = await handler.probe_engine_group()

    assert result
    assert [
        call.kwargs["data_parallel_rank"] for call in engine.generate.call_args_list
    ] == list(range(width))
    assert not admission.status()["admitted"]
    admission.endpoint.register_endpoint_instance.assert_not_awaited()


def test_packed_worker_advertises_only_its_local_range_after_growth():
    parallel = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_rank=0,
        distributed_executor_backend="mp",
    )
    config = VllmConfig(parallel_config=parallel)
    # Exercise the runtime projection without initializing a GPU-only EPLB configuration.
    parallel.enable_elastic_ep = True
    parallel.elastic_ep_auto_bootstrap = True
    assert get_dp_range_for_worker(config) == (0, 2)
    parallel.data_parallel_size = 4
    assert get_dp_range_for_worker(config) == (0, 2)


@pytest.mark.asyncio
async def test_external_prepare_does_not_force_a_serving_withdrawal():
    handler = _make_handler(tensor_parallel_size=1)
    handler._pause_lock = asyncio.Lock()
    handler.engine_client.vllm_config.parallel_config.data_parallel_external_lb = True
    admission = SimpleNamespace(withdraw=AsyncMock())
    handler.engine_group_serving = admission

    handler.engine_client.scale_elastic_ep.side_effect = TimeoutError(
        "ambiguous resize"
    )
    with pytest.raises(TimeoutError, match="ambiguous resize"):
        await handler.scale_elastic_ep(
            {
                "new_data_parallel_size": 3,
                "operation_id": "grow",
                "expected_instance_id": "engine-0",
            }
        )

    admission.withdraw.assert_not_awaited()
    handler.engine_client.scale_elastic_ep.assert_awaited_once()


@pytest.fixture
def stub_ray(monkeypatch):
    """Stand in for ray, which the scale path imports lazily and CI lacks."""
    state = SimpleNamespace(list_nodes=lambda **kwargs: [])
    util = SimpleNamespace(state=state)
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(nodes=lambda: [], util=util)
    )
    monkeypatch.setitem(sys.modules, "ray.util", util)
    monkeypatch.setitem(sys.modules, "ray.util.state", state)
    return state


# --------------------------------------------------------------------------- #
# Input validation and the TP-derived EP-size floor.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_size_is_rejected():
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({})

    assert result["status"] == "error"
    assert "new_data_parallel_size" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("size", [0, -1])
@pytest.mark.asyncio
async def test_sizes_below_one_are_rejected(size):
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "error"
    assert "must be >= 1" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("size", [1.5, "1.5", "abc", [2], True])
@pytest.mark.asyncio
async def test_non_integer_sizes_are_rejected(size):
    """A bare int() would truncate 1.5 -> 1 and coerce True -> 1; reject instead."""
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "error"
    assert "must be an integer" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.parametrize("size", [2.0, "2"])
@pytest.mark.asyncio
async def test_integer_valued_sizes_are_accepted(size, stub_ray):
    """Integer-valued floats and decimal-free strings coerce to the exact int."""
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "ok"
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(2)


@pytest.mark.asyncio
async def test_single_rank_expert_group_is_rejected():
    """At TP=1 a target of dp=1 leaves one EP rank, which EPLB does not allow."""
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 1})

    assert result["status"] == "error"
    assert "must be > 1" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_dp_one_allowed_when_tensor_parallelism_widens_the_group(stub_ray):
    """One pod per DP rank: TP=4 still leaves four EP ranks at dp=1."""
    handler = _make_handler(tensor_parallel_size=4)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 1})

    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 1
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_dp_two_allowed_at_tp_one(stub_ray):
    handler = _make_handler(tensor_parallel_size=1)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": 2})

    assert result["status"] == "ok"
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(2)


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.asyncio
async def test_prefill_context_parallelism_is_rejected(size):
    """vLLM's elastic EP sizes the EP world as data_parallel_size * tensor_parallel_size
    (excluding PCP) and forbids PCP>1 with DP>1, so a PCP>1 engine cannot be scaled --
    reject instead of admitting a topology elastic EP does not model."""
    handler = _make_handler(tensor_parallel_size=1, prefill_context_parallel_size=2)

    result = await handler.scale_elastic_ep({"new_data_parallel_size": size})

    assert result["status"] == "error"
    assert "prefill_context_parallel_size" in result["message"]
    handler.engine_client.scale_elastic_ep.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Fail-fast handling of a failed grow.
# --------------------------------------------------------------------------- #


class _WorkerShutdown(BaseException):
    """Stand-in for the NoReturn _shutdown_worker / _shutdown_on_engine_dead.
    BaseException (not Exception) so the handler's broad ``except Exception``
    can't swallow it -- the test sees the restart as production does: control
    leaves scale_elastic_ep without reporting success.
    """


_RESTARTED = object()  # sentinel: handler restarted the worker instead of returning


class _FakeVllmEngine:
    """Stand-in engine client: scaling to a ``fail_sizes`` size raises
    ``RuntimeError`` and to a ``dead_sizes`` size raises ``EngineDeadError``.
    ``tensor_parallel_size`` is set so the TP floor check passes and the grow is
    actually attempted.
    """

    def __init__(
        self,
        prev_dp,
        fail_sizes=(),
        dead_sizes=(),
        tensor_parallel_size=1,
        enable_elastic_ep=True,
        data_parallel_backend="ray",
    ):
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=prev_dp,
                tensor_parallel_size=tensor_parallel_size,
                enable_elastic_ep=enable_elastic_ep,
                data_parallel_backend=data_parallel_backend,
                data_parallel_external_lb=False,
                elastic_ep_auto_bootstrap=False,
            )
        )
        self._fail_sizes = list(fail_sizes)
        self._dead_sizes = list(dead_sizes)
        self.calls: list[int] = []  # every requested size, in order

    async def scale_elastic_ep(self, size: int) -> None:
        self.calls.append(size)
        if size in self._dead_sizes:
            raise EngineDeadError()
        if size in self._fail_sizes:
            raise RuntimeError(f"scale to {size} failed")
        self.vllm_config.parallel_config.data_parallel_size = size


def _make_self(engine: _FakeVllmEngine, shutdown_log: list) -> SimpleNamespace:
    def _shutdown_worker():
        shutdown_log.append("worker")
        raise _WorkerShutdown()

    def _shutdown_on_engine_dead(err):
        shutdown_log.append("engine_dead")
        raise _WorkerShutdown()

    return SimpleNamespace(
        _scale_ep_lock=asyncio.Lock(),
        _scale_ep_in_progress=False,
        engine_client=engine,
        _shutdown_worker=_shutdown_worker,
        _shutdown_on_engine_dead=_shutdown_on_engine_dead,
    )


def _run(engine: _FakeVllmEngine, body: dict):
    """Drive scale_elastic_ep. Returns ``(result, shutdown_log)``; ``result`` is
    ``_RESTARTED`` when the handler restarted the worker instead of returning."""
    shutdown_log: list[str] = []

    async def _coro():
        fake_self = _make_self(engine, shutdown_log)
        return await BaseWorkerHandler.scale_elastic_ep(fake_self, body)

    try:
        result = asyncio.run(_coro())
    except _WorkerShutdown:
        result = _RESTARTED
    return result, shutdown_log


def test_scale_success(stub_ray):
    engine = _FakeVllmEngine(prev_dp=2)

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert shutdown == []  # no restart on success
    assert result["status"] == "ok"
    assert result["new_data_parallel_size"] == 3
    assert engine.calls == [3]


def test_validation_error_does_not_restart_the_worker(stub_ray):
    # Fail-fast is scoped to real scale failures: a request rejected up front
    # (dp=1 at TP=1 collapses the EP world) returns an error and must NOT restart
    # the worker or touch the engine.
    engine = _FakeVllmEngine(prev_dp=2, tensor_parallel_size=1)

    result, shutdown = _run(engine, {"new_data_parallel_size": 1})

    assert shutdown == []  # no restart on a rejected request
    assert result["status"] == "error"
    assert engine.calls == []  # never reached the engine


def test_unsupported_config_is_rejected_without_restart(stub_ray):
    # control/scale_elastic_ep is registered on every worker, but a worker
    # without elastic EP / the Ray DP backend must get a nonfatal error, not a
    # fail-fast restart -- vLLM would raise NotImplementedError before any scale
    # state is mutated.
    engine = _FakeVllmEngine(
        prev_dp=2, enable_elastic_ep=False, data_parallel_backend="mp"
    )

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert shutdown == []  # a healthy worker is NOT restarted
    assert result["status"] == "error"
    assert "not enabled" in result["message"]
    assert engine.calls == []  # never reached the engine


def test_failed_grow_restarts_the_worker(stub_ray):
    # vLLM does not roll back a failed scale, so a failed grow must fail fast:
    # restart the worker rather than report a recovery that no caller acts on.
    engine = _FakeVllmEngine(prev_dp=2, fail_sizes=[3])

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert result is _RESTARTED  # never returned a success/recovery dict
    assert shutdown == ["worker"]  # worker restart was triggered
    assert engine.calls == [3]  # grow attempted once; no rollback call


def test_engine_dead_restarts_the_worker(stub_ray):
    # A dead engine routes through the same _shutdown_on_engine_dead path the rest
    # of the handler uses for engine death.
    engine = _FakeVllmEngine(prev_dp=2, dead_sizes=[3])

    result, shutdown = _run(engine, {"new_data_parallel_size": 3})

    assert result is _RESTARTED
    assert shutdown == ["engine_dead"]
    assert engine.calls == [3]  # no rollback call


@pytest.fixture
def external_handler():
    handler = _make_handler(tensor_parallel_size=1, data_parallel_size=2)
    parallel = handler.engine_client.vllm_config.parallel_config
    parallel.data_parallel_backend = "mp"
    parallel.data_parallel_external_lb = True
    status = {
        "instance_id": "api-0",
        "operation_id": "grow-three",
        "epoch": "native-epoch",
        "phase": "completed",
        "requested_data_parallel_size": 3,
        "effective_data_parallel_size": 3,
        "topology_known": True,
        "error": None,
    }
    handler.engine_client.scale_elastic_ep.return_value = status
    handler.engine_client.get_external_elastic_ep_status = AsyncMock(
        return_value=status
    )
    handler._shutdown_worker = Mock()
    return handler


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_external_scale_bypasses_ray_and_legacy_resize_lock(external_handler):
    handler = external_handler
    async with handler._scale_ep_lock:
        result = await asyncio.wait_for(
            handler.scale_elastic_ep(
                {
                    "new_data_parallel_size": 3,
                    "operation_id": "grow-three",
                    "expected_instance_id": "api-0",
                }
            ),
            timeout=1,
        )
    assert result["phase"] == "completed"
    handler.engine_client.scale_elastic_ep.assert_awaited_once_with(
        3,
        drain_timeout=120,
        operation_id="grow-three",
        expected_instance_id="api-0",
    )
    assert not handler._scale_ep_in_progress
    handler._shutdown_worker.assert_not_called()


@pytest.mark.asyncio
async def test_external_timeout_does_not_take_ray_restart_path(external_handler):
    handler = external_handler
    handler.engine_client.scale_elastic_ep.side_effect = TimeoutError("response lost")
    with pytest.raises(TimeoutError, match="response lost"):
        await handler.scale_elastic_ep(
            {
                "new_data_parallel_size": 3,
                "operation_id": "grow-three",
                "expected_instance_id": "api-0",
            }
        )
    handler._shutdown_worker.assert_not_called()
    result = await handler.is_scaling_elastic_ep({})
    assert result["phase"] == "completed"


@pytest.mark.asyncio
async def test_ray_worker_rejects_external_status_without_touching_engine():
    handler = _make_handler(tensor_parallel_size=1)
    result = await handler.is_scaling_elastic_ep({})
    assert result["status"] == "error"
    handler.engine_client.scale_elastic_ep.assert_not_awaited()
