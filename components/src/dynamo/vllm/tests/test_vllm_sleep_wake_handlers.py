# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.config")
pytest.importorskip("vllm.v1.engine.exceptions")

from dynamo.vllm.handlers import (  # noqa: E402
    BaseWorkerHandler,
    VllmEnginePauseController,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler() -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        is_sleeping=AsyncMock(return_value=False),
        is_paused=AsyncMock(return_value=False),
        resume_generation=AsyncMock(),
    )
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler._pause_controller = VllmEnginePauseController(handler.engine_client)
    handler._pause_lock = asyncio.Lock()
    handler._weight_version = "initial"
    return handler


class _StatefulVllmEngine:
    """Match vLLM core sleep/wake semantics, including scheduler conflation."""

    def __init__(self):
        self.scheduler_paused = False
        self.memory_sleeping = False
        self.pause_failures: list[BaseException] = []
        self.pause_calls: list[tuple[str, bool]] = []
        self.collective_rpc = AsyncMock()
        self.reset_prefix_cache = AsyncMock(return_value=True)

    async def pause_generation(self, *, mode="abort", clear_cache=True):
        self.pause_calls.append((mode, clear_cache))
        self.scheduler_paused = True
        if self.pause_failures:
            raise self.pause_failures.pop(0)

    async def resume_generation(self):
        self.scheduler_paused = False

    async def sleep(self, level=1):
        self.scheduler_paused = True
        self.memory_sleeping = level >= 1

    async def wake_up(self, tags=None):
        if tags is None:
            self.memory_sleeping = False
        if not self.memory_sleeping:
            self.scheduler_paused = False

    async def is_sleeping(self):
        return self.scheduler_paused or self.memory_sleeping

    async def is_paused(self):
        return self.scheduler_paused


def _use_stateful_engine(handler: _TestWorkerHandler) -> _StatefulVllmEngine:
    engine = _StatefulVllmEngine()
    handler.engine_client = engine
    handler._pause_controller = VllmEnginePauseController(engine)
    return engine


@pytest.mark.asyncio
async def test_wake_up_before_sleep_is_noop():
    handler = _make_handler()

    result = await handler.wake_up({})

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_not_awaited()
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_sleep_and_wake_are_idempotent():
    handler = _make_handler()

    first_sleep = await handler.sleep({"level": 2})
    second_sleep = await handler.sleep({"level": 2})
    first_wake = await handler.wake_up({})
    second_wake = await handler.wake_up({})

    assert first_sleep["status"] == "ok"
    assert second_sleep["status"] == "ok"
    assert first_wake["status"] == "ok"
    assert second_wake["status"] == "ok"

    handler.engine_client.pause_generation.assert_awaited_once()
    handler.engine_client.sleep.assert_awaited_once_with(2)
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine_client.wake_up.assert_awaited_once_with()
    # vLLM's full wake resumes scheduling as part of the native RPC.
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_and_wake_manage_versioned_generate_alias():
    handler = _make_handler()
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)

    assert (await handler.sleep({"level": 1}))["status"] == "ok"
    assert (await handler.wake_up({}))["status"] == "ok"

    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    versioned_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    versioned_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_alias_unregister_failure_rolls_back_every_endpoint():
    handler = _make_handler()
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(
            side_effect=RuntimeError("alias discovery unavailable")
        ),
        register_endpoint_instance=AsyncMock(),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)

    result = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    handler.engine_client.pause_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    # Converging to the published state must reconcile every endpoint, including
    # the endpoint whose unregister call failed with an unknown remote outcome.
    versioned_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_sleep_rollback_publication_recovers_via_wake_while_awake():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(
            side_effect=[RuntimeError("alias register failed"), None]
        ),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)

    sleep_result = await handler.sleep({"level": 1})
    assert sleep_result["status"] == "error"
    assert handler._pause_controller.is_paused is False

    # The engine is already awake, but endpoint publication still needs repair.
    wake_result = await handler.wake_up({})

    assert wake_result["status"] == "ok"
    assert versioned_endpoint.register_endpoint_instance.await_count == 2
    handler.engine_client.wake_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_wake_alias_registration_failure_fails_closed_then_recovers():
    handler = _make_handler()
    versioned_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(
            side_effect=[RuntimeError("alias register failed"), None]
        ),
    )
    handler.additional_generate_endpoints = (versioned_endpoint,)
    assert (await handler.sleep({"level": 1}))["status"] == "ok"

    first_wake = await handler.wake_up({})

    assert first_wake["status"] == "error"
    # A partial publish is compensated by unpublishing the complete endpoint
    # set, leaving the awake engine safely unreachable until a retry converges.
    assert handler.generate_endpoint.unregister_endpoint_instance.await_count == 2
    assert versioned_endpoint.unregister_endpoint_instance.await_count == 2
    assert handler._pause_controller.is_paused is False

    second_wake = await handler.wake_up({})

    assert second_wake["status"] == "ok"
    assert handler.generate_endpoint.register_endpoint_instance.await_count == 2
    assert versioned_endpoint.register_endpoint_instance.await_count == 2


@pytest.mark.asyncio
async def test_pause_without_level_uses_vllm_default_sleep():
    engine_client = SimpleNamespace(
        pause_generation=AsyncMock(),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        resume_generation=AsyncMock(),
    )
    controller = VllmEnginePauseController(engine_client)

    changed = await controller.pause(None)

    assert changed is True
    engine_client.pause_generation.assert_awaited_once()
    engine_client.sleep.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_wake_up_passes_explicit_tags_from_request():
    handler = _make_handler()
    await handler._pause_controller.pause(1)

    result = await handler.wake_up({"tags": ["weights"]})

    assert result["status"] == "ok"
    handler.engine_client.wake_up.assert_awaited_once_with(["weights"])
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_wake_remains_unpublished_and_generation_paused():
    handler = _make_handler()
    await handler._pause_controller.pause(1)
    handler.engine_client.is_paused.return_value = True

    result = await handler.wake_up({"tags": ["weights"]})

    assert result["status"] == "error"
    assert "still sleeping" in result["message"]
    handler.engine_client.wake_up.assert_awaited_once_with(["weights"])
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()
    assert handler._pause_controller.is_paused is True
    assert handler._pause_controller.needs_resume_recovery is True


@pytest.mark.asyncio
async def test_native_wake_retries_when_scheduler_remains_paused():
    handler = _make_handler()
    await handler._pause_controller.pause(1)
    handler.engine_client.is_paused.side_effect = [True, False]

    first = await handler.wake_up({})
    second = await handler.wake_up({})

    assert first["status"] == "error"
    assert second["status"] == "ok"
    assert handler.engine_client.wake_up.await_count == 2
    assert handler.engine_client.is_paused.await_count == 2
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_wake_is_reconciled_before_retry_after_residency_query_failure():
    handler = _make_handler()
    await handler._pause_controller.pause(1)
    handler.engine_client.is_paused.side_effect = [
        RuntimeError("scheduler query failed"),
        False,
    ]

    first = await handler.wake_up({})
    second = await handler.wake_up({})

    assert first["status"] == "error"
    assert second["status"] == "ok"
    assert handler.engine_client.wake_up.await_count == 2
    assert handler.engine_client.is_paused.await_count == 2
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_sleep_rollback_wakes_if_scheduler_is_still_paused():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))
    handler.engine_client.is_paused.side_effect = [True, False]

    sleep_result = await handler.sleep({"level": 1})

    assert sleep_result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is False
    handler.engine_client.wake_up.assert_awaited_once_with()
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_reregisters_after_clean_pause_rollback():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep failed"))

    result = await handler.sleep({"level": 1})

    # Rollback resumed generation cleanly, so the engine is serving-safe again.
    assert result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is False
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    # The endpoint must rejoin the routing pool since wake_up will early-return.
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_returns_error_for_unregister_failure():
    handler = _make_handler()
    handler.generate_endpoint.unregister_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery backend down")
    )

    result = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    handler.engine_client.pause_generation.assert_not_awaited()
    handler.engine_client.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_wake_up_returns_error_for_register_failure():
    handler = _make_handler()
    await handler._pause_controller.pause(1)
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.wake_up({})

    assert result["status"] == "error"
    handler.engine_client.wake_up.assert_awaited_once_with()
    handler.engine_client.resume_generation.assert_not_awaited()
    # Native wake already completed; only publication remains pending.
    assert handler._pause_controller.is_paused is False

    handler.generate_endpoint.register_endpoint_instance = AsyncMock()
    retry = await handler.wake_up({})

    assert retry["status"] == "ok"
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    handler.engine_client.wake_up.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_rl_pause_sleep_wake_preserves_pause_owner_until_rl_resume():
    handler = _make_handler()

    paused = await handler.pause_generation({"mode": "wait"})
    slept = await handler.sleep({"level": 1})
    woke = await handler.wake_up({})

    assert paused["status"] == "ok"
    assert slept["status"] == "ok"
    assert woke["status"] == "ok"
    assert handler._pause_controller.is_rl_paused is True
    assert handler._pause_controller.is_generation_paused is True
    assert handler._pause_controller.is_paused is False
    # Waking memory must not steal the generation-pause ownership established
    # by the RL control plane or republish a worker that is still paused.
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()

    resumed = await handler.resume_generation({})

    assert resumed["status"] == "ok"
    handler.engine_client.resume_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_vllm_full_wake_reestablishes_rl_drained_pause_before_update():
    handler = _make_handler()
    engine = _use_stateful_engine(handler)

    assert (await handler.pause_generation({"mode": "wait"}))["status"] == "ok"
    assert (await handler.sleep({"level": 1}))["status"] == "ok"
    assert (await handler.wake_up({}))["status"] == "ok"

    # Real vLLM full wake resumes its scheduler. RL ownership therefore has to
    # issue and await a fresh drained pause; is_sleeping() cannot certify this
    # because it is true for either scheduler pause OR sleeping memory.
    assert engine.scheduler_paused is True
    assert engine.memory_sleeping is False
    assert engine.pause_calls == [("wait", False), ("wait", False)]
    assert handler._pause_controller.is_rl_drained is True

    update = await handler.update_weights_from_distributed(
        {"engine_rpc": "finish_weight_update"}
    )
    assert update["status"] == "ok"


@pytest.mark.asyncio
async def test_sleep_then_rl_pause_cannot_mutate_weights_until_memory_is_awake():
    handler = _make_handler()
    engine = _use_stateful_engine(handler)

    assert (await handler.sleep({"level": 1}))["status"] == "ok"
    assert (await handler.pause_generation({"mode": "wait"}))["status"] == "ok"

    assert engine.memory_sleeping is True
    assert handler._pause_controller.is_rl_paused is True
    assert handler._pause_controller.is_rl_drained is False
    blocked = await handler.update_weights_from_distributed(
        {"engine_rpc": "finish_weight_update"}
    )
    assert blocked["status"] == "error"
    assert "drained" in blocked["message"]
    handler.engine_client.collective_rpc.assert_not_awaited()

    assert (await handler.wake_up({}))["status"] == "ok"
    assert engine.memory_sleeping is False
    assert handler._pause_controller.is_rl_drained is True
    update = await handler.update_weights_from_distributed(
        {"engine_rpc": "finish_weight_update"}
    )
    assert update["status"] == "ok"


@pytest.mark.asyncio
async def test_lost_rl_pause_response_is_owned_but_not_drained_until_repause():
    handler = _make_handler()
    engine = _use_stateful_engine(handler)
    engine.pause_failures = [RuntimeError("pause response lost")]

    first = await handler.pause_generation({"mode": "wait"})
    assert first["status"] == "error"
    assert handler._pause_controller.is_rl_paused is True
    assert handler._pause_controller.is_rl_drained is False

    blocked = await handler.update_weights_from_distributed(
        {"engine_rpc": "finish_weight_update"}
    )
    assert blocked["status"] == "error"
    assert "drained" in blocked["message"]

    retry = await handler.pause_generation({"mode": "wait"})
    assert retry["status"] == "ok"
    assert engine.pause_calls == [("wait", False), ("wait", False)]
    assert handler._pause_controller.is_rl_drained is True


@pytest.mark.asyncio
async def test_cancelled_rl_pause_is_owned_and_ambiguous():
    handler = _make_handler()
    engine = _use_stateful_engine(handler)
    engine.pause_failures = [asyncio.CancelledError()]

    with pytest.raises(asyncio.CancelledError):
        await handler.pause_generation({"mode": "abort"})

    assert engine.scheduler_paused is True
    assert handler._pause_controller.is_rl_paused is True
    assert handler._pause_controller.is_rl_drained is False


@pytest.mark.asyncio
async def test_lost_initial_sleep_pause_response_never_republishes_ambiguous_worker():
    handler = _make_handler()
    engine = _use_stateful_engine(handler)
    engine.pause_failures = [RuntimeError("initial pause response lost")]

    sleep = await handler.sleep({"level": 1})

    assert sleep["status"] == "error"
    assert engine.scheduler_paused is True
    assert handler._pause_controller.has_sleep_state is True
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()

    wake = await handler.wake_up({})
    assert wake["status"] == "ok"
    assert engine.scheduler_paused is False
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_lost_response_reconciles_sleeping_state_before_publication():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(
        side_effect=RuntimeError("sleep response lost")
    )
    # The remote sleep took effect even though the RPC raised locally. The
    # authoritative scheduler query sees the pause, so rollback performs an
    # idempotent full wake and verifies that scheduling resumed.
    handler.engine_client.is_paused.side_effect = [True, False]

    result = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    assert handler._pause_controller.can_serve is True
    assert handler._pause_controller.is_generation_paused is False
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.engine_client.wake_up.assert_awaited_once_with()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_reconciliation_failure_remains_fail_closed():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=RuntimeError("sleep uncertain"))
    handler.engine_client.is_paused = AsyncMock(
        side_effect=RuntimeError("scheduler state unavailable")
    )

    result = await handler.sleep({"level": 1})
    retry = await handler.sleep({"level": 1})

    assert result["status"] == "error"
    assert retry["status"] == "error"
    assert "wake_up required" in retry["message"]
    assert handler._pause_controller.is_paused is True
    assert handler._pause_controller.is_generation_paused is True
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_sleep_clean_rollback_restores_publication_then_propagates():
    handler = _make_handler()
    handler.engine_client.sleep = AsyncMock(side_effect=asyncio.CancelledError())
    handler.engine_client.is_sleeping.return_value = False

    with pytest.raises(asyncio.CancelledError):
        await handler.sleep({"level": 1})

    assert handler._pause_controller.can_serve is True
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_rl_resume_revokes_drained_weight_update_fence():
    handler = _make_handler()
    handler.engine_client.collective_rpc = AsyncMock()
    handler.engine_client.reset_prefix_cache = AsyncMock(return_value=True)
    await handler.pause_generation({"mode": "wait"})
    handler.engine_client.resume_generation = AsyncMock(
        side_effect=RuntimeError("resume response lost")
    )

    resume = await handler.resume_generation({})
    update = await handler.update_weights_from_distributed(
        {"engine_rpc": "finish_weight_update"}
    )

    assert resume["status"] == "error"
    assert handler._pause_controller.is_rl_paused is True
    assert handler._pause_controller.is_rl_drained is False
    assert update["status"] == "error"
    assert "drained" in update["message"]
    handler.engine_client.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("update_method", "body"),
    [
        ("update_weights_from_disk", {"model_path": "/weights"}),
        (
            "update_weights_from_distributed",
            {"engine_rpc": "finish_weight_update"},
        ),
    ],
)
async def test_post_mutation_cache_reset_failure_latches_fail_closed(
    update_method, body
):
    handler = _make_handler()
    handler.engine_client.collective_rpc = AsyncMock()
    handler.engine_client.reset_prefix_cache = AsyncMock(return_value=False)
    await handler.pause_generation({"mode": "wait"})

    update = await getattr(handler, update_method)(body)
    retry = await getattr(handler, update_method)(body)
    resume = await handler.resume_generation({})

    assert update["status"] == "error"
    assert retry["status"] == "error"
    assert "restart" in retry["message"].lower()
    assert resume["status"] == "error"
    assert "restart" in resume["message"].lower()
    assert handler._pause_controller.fatal_error is not None
    handler.engine_client.resume_generation.assert_not_awaited()
    handler.engine_client.collective_rpc.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_mutation_cache_reset_cancellation_still_latches_fail_closed():
    handler = _make_handler()
    handler.engine_client.collective_rpc = AsyncMock()
    handler.engine_client.reset_prefix_cache = AsyncMock(
        side_effect=asyncio.CancelledError()
    )
    await handler.pause_generation({"mode": "wait"})

    with pytest.raises(asyncio.CancelledError):
        await handler.update_weights_from_distributed(
            {"engine_rpc": "finish_weight_update"}
        )

    assert handler._pause_controller.fatal_error is not None
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited()
    resume = await handler.resume_generation({})
    assert resume["status"] == "error"
    assert "restart" in resume["message"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("update_method", "body"),
    [
        ("update_weights_from_disk", {"model_path": "/weights"}),
        (
            "update_weights_from_distributed",
            {"engine_rpc": "finish_weight_update"},
        ),
    ],
)
@pytest.mark.parametrize(
    ("mutation_error", "is_cancelled"),
    [
        (RuntimeError("collective response lost"), False),
        (asyncio.CancelledError(), True),
    ],
)
async def test_collective_dispatch_error_latches_indeterminate_weight_state(
    update_method, body, mutation_error, is_cancelled
):
    handler = _make_handler()
    handler.engine_client.collective_rpc = AsyncMock(side_effect=mutation_error)
    handler.engine_client.reset_prefix_cache = AsyncMock(return_value=True)
    await handler.pause_generation({"mode": "wait"})

    if is_cancelled:
        with pytest.raises(asyncio.CancelledError):
            await getattr(handler, update_method)(body)
    else:
        result = await getattr(handler, update_method)(body)
        assert result["status"] == "error"

    assert handler._pause_controller.fatal_error is not None
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited()
    handler.engine_client.reset_prefix_cache.assert_not_awaited()
    retry = await getattr(handler, update_method)(body)
    assert retry["status"] == "error"
    assert "restart" in retry["message"].lower()
    handler.engine_client.collective_rpc.assert_awaited_once()
