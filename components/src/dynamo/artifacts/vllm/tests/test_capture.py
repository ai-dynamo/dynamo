# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("msgspec")
pytest.importorskip("zstandard")

from dynamo.artifacts.vllm import (  # noqa: E402 -- optional dependency guards
    capture as generation_artifact,
)
from dynamo.artifacts.vllm.capture import (  # noqa: E402 -- optional dependency guards
    CaptureBudget,
    CaptureCapacityError,
    VllmArtifactSession,
)
from dynamo.artifacts.vllm.format import (  # noqa: E402 -- optional dependency guards
    decode_generation_artifact,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def test_frontend_local_string_stop_is_rejected_before_admission():
    req = request()
    req["stop_conditions"] = {"stop": ["END"]}
    budget = CaptureBudget(1 << 20)
    with pytest.raises(ValueError, match="string stops"):
        session(req, budget)
    assert budget.used == 0


def request(contents=(), mode="inline_base64", destination=None):
    settings = {"kind": "generation_artifact", "contents": list(contents), "mode": mode}
    if destination is not None:
        settings["destination"] = destination
    return {"extra_args": {"nvext": {"metadata_upload": settings}}}


def session(req=None, budget=None, **kwargs):
    return VllmArtifactSession.from_request(
        req or request(),
        prompt_token_count=2,
        max_tokens=2,
        choice_count=1,
        route_capture_enabled=True,
        router_profile=((3, 8, 2), (9, 300, 2)),
        budget=budget or CaptureBudget(1 << 20),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_inline_routes_use_exact_sequence_and_geometry():
    capture = session(request(("moe_routes",)))
    routes = np.array(
        [[[1, 2], [1, 2]], [[2, 3], [2, 3]], [[3, 4], [3, 4]]], dtype=np.int64
    )
    capture.observe(
        SimpleNamespace(index=0, token_ids=[12, 13], routed_experts=routes), [10, 11]
    )
    result = await capture.finalize()
    choice = decode_generation_artifact(
        base64.b64decode(result["data_base64"])
    ).choices[0]
    assert choice.sequence_token_ids.as_array().tolist() == [10, 11, 12, 13]
    assert choice.components[0].routers[1].expert_ids.dtype == "u16"
    assert capture.reserved_bytes == 0


@pytest.mark.parametrize("contents", [("unknown",), ("moe_routes", "moe_routes")])
def test_invalid_selection_is_rejected_before_inference(contents):
    with pytest.raises(ValueError):
        session(request(contents))


def test_budget_is_aggregate_and_admitted_before_capture():
    budget = CaptureBudget(100_000)
    first = session(budget=budget)
    with pytest.raises(CaptureCapacityError, match="capacity"):
        session(budget=CaptureBudget(1))
    assert budget.used > 0
    first.close()
    first.close()
    assert budget.used == 0


def test_default_budget_admits_representative_concurrent_route_requests(monkeypatch):
    monkeypatch.delenv("DYNAMO_ARTIFACT_CAPTURE_BYTES", raising=False)
    generation_artifact.process_capture_budget.cache_clear()
    budget = generation_artifact.process_capture_budget()
    runtime = SimpleNamespace(validate=Mock())
    sessions = []
    try:
        for _ in range(20):
            sessions.append(
                VllmArtifactSession.from_request(
                    request(("moe_routes",), "object_store", {"kind": "managed_store"}),
                    prompt_token_count=2048,
                    max_tokens=8192,
                    route_capture_enabled=True,
                    router_profile=tuple((i, 256, 8) for i in range(48)),
                    budget=budget,
                    runtime=runtime,
                )
            )
        assert budget.used <= budget.max_bytes
    finally:
        for capture in sessions:
            capture.close()
        generation_artifact.process_capture_budget.cache_clear()
    assert budget.used == 0


@pytest.mark.asyncio
async def test_delivery_rejection_is_not_reported_as_invalid_capture():
    runtime = SimpleNamespace(
        validate=Mock(), submit=AsyncMock(side_effect=RuntimeError("delivery full"))
    )
    capture = session(
        request((), "object_store", {"kind": "managed_store"}), runtime=runtime
    )
    capture.observe(SimpleNamespace(index=0, token_ids=[12]), [10, 11])
    result = await capture.finalize()
    assert result["state"] == "failed"
    assert result["error_code"] == "delivery_rejected"
    assert "delivery full" not in str(result)
    assert capture.reserved_bytes == 0


@pytest.mark.asyncio
async def test_missing_requested_routes_fails_explicitly():
    capture = session(request(("moe_routes",)))
    capture.observe(
        SimpleNamespace(index=0, token_ids=[12, 13], routed_experts=None), [10, 11]
    )
    result = await capture.finalize()
    assert result["state"] == "failed"
    assert result["error_code"] == "capture_invalid"
    assert capture.reserved_bytes == 0


def test_object_request_requires_configured_authority_before_inference():
    with pytest.raises(ValueError, match="configured"):
        session(
            request(
                (),
                "object_store",
                {"kind": "managed_store", "profile": "p", "object_key": "k"},
            )
        )


def test_legacy_request_does_not_activate_artifact_capture():
    assert (
        VllmArtifactSession.from_request(
            {"nvext": {"metadata_upload": {"url": "s3://bucket/prefix"}}}
        )
        is None
    )


@pytest.mark.parametrize(
    "profile",
    [
        ((-1, 8, 2),),
        ((1 << 32, 8, 2),),
        ((True, 8, 2),),
        ((1, 0, 1),),
        ((1, 65537, 1),),
        ((1, True, 1),),
        ((1, 8, 0),),
        ((1, 2, 3),),
        ((1, 8, True),),
        ((1, 8),),
        ((1, 8, 2), (1, 8, 2)),
        ((1, 8, 1), (2, 8, 2)),
    ],
)
def test_invalid_router_geometry_fails_before_admission(profile):
    budget = CaptureBudget(1 << 20)
    with pytest.raises(ValueError, match="router"):
        VllmArtifactSession.from_request(
            request(("moe_routes",)),
            prompt_token_count=2,
            max_tokens=2,
            route_capture_enabled=True,
            router_profile=profile,
            budget=budget,
        )
    assert budget.used == 0


@pytest.mark.parametrize("start", [-1, True, 1.5, 3])
def test_invalid_route_start_is_not_silently_clamped(start):
    budget = CaptureBudget(1 << 20)
    with pytest.raises(ValueError, match="token_start"):
        session(request(("moe_routes",)), token_start=start, budget=budget)
    assert budget.used == 0


@pytest.mark.parametrize("width", [0, True, 1025])
def test_invalid_logprob_width_fails_before_capture(width):
    with pytest.raises(ValueError, match="logprob width"):
        session(logprob_width=width)


@pytest.mark.asyncio
async def test_destination_and_profile_are_owned_after_admission():
    destination = {
        "kind": "managed_store",
        "profile": "safe",
        "object_key": "original",
        "nested": {"header": "original"},
    }
    runtime = SimpleNamespace(
        validate=Mock(), submit=AsyncMock(return_value={"state": "pending"})
    )
    capture = session(request((), "object_store", destination), runtime=runtime)
    destination["object_key"] = "changed"
    destination["nested"]["header"] = "changed"
    capture.observe(SimpleNamespace(index=0, token_ids=[12]), [10, 11])
    await capture.finalize()
    submitted = runtime.submit.call_args.args[2]
    assert submitted["object_key"] == "original"
    assert submitted["nested"]["header"] == "original"


def test_router_profile_is_immutable_after_admission():
    profile = [[3, 8, 2]]
    capture = VllmArtifactSession.from_request(
        request(("moe_routes",)),
        prompt_token_count=2,
        max_tokens=2,
        route_capture_enabled=True,
        router_profile=profile,
    )
    profile[0][1] = 1000
    profile.append([4, 1000, 2])
    assert capture.profile == ((3, 8, 2),)
    capture.close()


def test_router_profile_limit_names_the_tensor_budget():
    profile = tuple((index, 8, 2) for index in range(4093))
    with pytest.raises(ValueError, match="at most 4092 routers.*4096-tensor"):
        generation_artifact._validated_router_profile(profile)


def test_closed_capture_never_retains_new_observations():
    capture = session()
    capture.close()
    capture.observe(SimpleNamespace(index=0, token_ids=[12]), [10, 11])
    assert capture.prompt is None
    assert capture.tokens == []


def test_observed_prompt_cannot_silently_shift_route_origin():
    capture = session(request(("moe_routes",)), token_start=2)
    capture.observe(SimpleNamespace(index=0, token_ids=[12]), [10])
    assert capture.error
    assert capture.prompt is None
    capture.close()


@pytest.mark.asyncio
async def test_selected_and_topk_capture_keep_pairing_and_bounds():
    capture = session(request(("selected_logprobs", "topk_logprobs")), logprob_width=2)
    values = {
        12: SimpleNamespace(logprob=-0.2),
        99: SimpleNamespace(logprob=-0.1),
        88: SimpleNamespace(logprob=-2.0),
    }
    capture.observe(
        SimpleNamespace(index=0, token_ids=[12], logprobs=[values]), [10, 11]
    )
    result = await capture.finalize()
    choice = decode_generation_artifact(
        base64.b64decode(result["data_base64"])
    ).choices[0]
    selected, topk = choice.components
    assert selected.logprobs.as_array().tolist() == pytest.approx([-0.2])
    assert topk.candidate_token_ids.as_array().tolist() == [[99, 12]]
    assert topk.logprobs.as_array()[0].tolist() == pytest.approx([-0.1, -0.2])


@pytest.mark.parametrize(
    "values",
    [
        None,
        {},
        {13: SimpleNamespace(logprob=-1.0)},
        {12: SimpleNamespace(logprob=float("nan"))},
        {12: SimpleNamespace(logprob=float("inf"))},
        {12: SimpleNamespace(logprob=10**1000)},
        {12: SimpleNamespace(logprob="-1.0")},
        {12: SimpleNamespace(logprob=-1.0), -1: SimpleNamespace(logprob=-2.0)},
        {
            12: SimpleNamespace(logprob=-1.0),
            13: SimpleNamespace(logprob=-2.0),
            14: SimpleNamespace(logprob=-3.0),
        },
    ],
)
def test_invalid_logprob_rows_are_discarded_and_cannot_resume(values):
    capture = session(request(("selected_logprobs",)))
    capture.observe(
        SimpleNamespace(
            index=0, token_ids=[12], logprobs=None if values is None else [values]
        ),
        [10, 11],
    )
    assert capture.error
    assert capture.tokens == []
    capture.observe(
        SimpleNamespace(
            index=0, token_ids=[13], logprobs=[{13: SimpleNamespace(logprob=-1.0)}]
        ),
        [10, 11],
    )
    assert capture.tokens == []
    capture.close()


@pytest.mark.asyncio
async def test_prefill_only_selected_and_topk_have_zero_rows():
    capture = VllmArtifactSession.from_request(
        request(("selected_logprobs", "topk_logprobs")),
        prompt_token_count=2,
        max_tokens=0,
        logprob_width=2,
    )
    capture.observe(SimpleNamespace(index=0, token_ids=[], logprobs=None), [10, 11])
    result = await capture.finalize()
    components = (
        decode_generation_artifact(base64.b64decode(result["data_base64"]))
        .choices[0]
        .components
    )
    assert components[0].logprobs.shape == (0,)
    assert components[1].logprobs.shape == (0, 2)


def test_route_extent_is_bounded_before_compaction_and_failure_releases_views():
    capture = session(request(("moe_routes",)), token_start=1)
    native = np.zeros((20, 2, 2), dtype=np.int32)
    capture.observe(
        SimpleNamespace(index=0, token_ids=[12], routed_experts=native), [10, 11]
    )
    assert capture.error
    assert capture.routes is None
    assert capture.tokens == []
    capture.close()


@pytest.mark.asyncio
async def test_valid_nonzero_start_keeps_requested_origin():
    capture = session(request(("moe_routes",)), token_start=1)
    native = np.asarray([[[0, 1], [2, 3]], [[1, 2], [3, 4]]], dtype=np.int64)
    capture.observe(
        SimpleNamespace(index=0, token_ids=[12, 13], routed_experts=native), [10, 11]
    )
    result = await capture.finalize()
    routes = (
        decode_generation_artifact(base64.b64decode(result["data_base64"]))
        .choices[0]
        .components[0]
    )
    assert routes.token_start == 1


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_route_compaction_runs_off_loop_and_finishes_before_cancellation(
    monkeypatch,
):
    original = generation_artifact.compact_routes
    started = threading.Event()
    finish = threading.Event()
    thread_ids = []

    def delayed(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        started.set()
        if not finish.wait(timeout=5):
            raise RuntimeError("test route compaction was not released")
        return original(*args, **kwargs)

    monkeypatch.setattr(generation_artifact, "compact_routes", delayed)
    budget = CaptureBudget(1 << 20)
    capture = session(request(("moe_routes",)), budget=budget)
    routes = np.asarray([[[1, 2], [3, 4]], [[2, 3], [4, 5]]], dtype=np.int64)
    output = SimpleNamespace(index=0, token_ids=[12], routed_experts=routes)
    loop_thread = threading.get_ident()

    task = asyncio.create_task(capture.observe_async(output, [10, 11]))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert len(thread_ids) == 1
        assert thread_ids[0] != loop_thread
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert budget.used > 0
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    capture.close()
    assert budget.used == 0


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_cancelled_encoder_holds_reservation_until_thread_finishes(monkeypatch):
    original = generation_artifact.encode_generation_artifact
    started = threading.Event()
    finish = threading.Event()

    def delayed(view):
        started.set()
        if not finish.wait(timeout=5):
            raise RuntimeError("test encoder was not released")
        return original(view)

    monkeypatch.setattr(generation_artifact, "encode_generation_artifact", delayed)
    budget = CaptureBudget(1 << 20)
    capture = session(budget=budget)
    capture.observe(SimpleNamespace(index=0, token_ids=[12]), [10, 11])
    task = asyncio.create_task(capture.finalize())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        assert budget.used > 0
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert budget.used == 0
