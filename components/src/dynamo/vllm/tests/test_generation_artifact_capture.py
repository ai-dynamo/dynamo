# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import struct
import threading
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from dynamo.vllm.generation_artifact import (
    ArtifactCaptureError,
    VllmGenerationArtifactSession,
    _resolve_router_layout,
)
from dynamo.vllm.generation_artifact_format import decode_generation_artifact

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture(autouse=True)
def _enable_managed_test_storage(monkeypatch):
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", "true")
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"test":{"url":"s3://generation-artifacts/root","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    stored = {}
    session = SimpleNamespace(close=AsyncMock())

    async def pipe_file(path, data, mode, *, chunksize):
        assert mode == "create"
        assert chunksize > 0
        stored[path] = data

    filesystem = SimpleNamespace(
        protocol="s3",
        async_impl=True,
        set_session=AsyncMock(return_value=session),
        _pipe_file=pipe_file,
    )
    with patch(
        "dynamo.common.generation_artifact_storage.url_to_fs",
        return_value=(filesystem, "generation-artifacts/root"),
    ):
        yield stored


def _request(*contents: str) -> dict:
    return {
        "nvext": {
            "generation_artifact": {
                "format": "generation_artifact_v1",
                "contents": list(contents),
                "delivery": {
                    "mode": "object_store",
                    "target": {
                        "kind": "managed_fsspec",
                        "profile": "test",
                        "object_key": "request-1/output.dynexp",
                    },
                },
            }
        }
    }


def _model_config(**overrides):
    values = {
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 1,
        **overrides,
    }
    return SimpleNamespace(hf_config=SimpleNamespace(**values))


def _session(request: dict, **overrides) -> VllmGenerationArtifactSession:
    values = {
        "model_config": _model_config(),
        "enable_rl": True,
        "route_capture_enabled": True,
        "choice_count": 1,
        **overrides,
    }
    session = VllmGenerationArtifactSession.from_backend_request(request, **values)
    assert session is not None
    return session


@pytest.mark.asyncio
async def test_capture_delivers_decodable_artifact(
    monkeypatch, _enable_managed_test_storage
) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"test":{"url":"s3://generation-artifacts/root","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    session = _session(_request("moe_routes", "selected_logprobs"))
    await session.admit(prompt_token_count=2, max_tokens=2)
    routes = np.array([[[0, 2]], [[1, 3]], [[2, 0]]], dtype=np.int64)

    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101, 102],
        completion_token_ids=[201],
        selected_logprobs=[-0.25],
        routed_experts=None,
    )
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[],
        completion_token_ids=[202],
        selected_logprobs=[-0.5],
        routed_experts=routes,
    )
    receipt = await session.finalize_choice(choice_index=0, token_start=0)

    decoded = decode_generation_artifact(
        _enable_managed_test_storage[
            "generation-artifacts/root/request-1/output.dynexp"
        ]
    )
    choice = decoded.choices[0]
    np.testing.assert_array_equal(choice.sequence_token_ids, [101, 102, 201, 202])
    np.testing.assert_array_equal(choice.routed_experts, routes.astype(np.uint8))
    np.testing.assert_allclose(choice.selected_logprobs, [-0.25, -0.5])
    assert receipt["state"] == "ready"
    assert receipt["object_id"] == "test:request-1/output.dynexp"
    assert receipt["contents"] == ["moe_routes", "selected_logprobs"]


@pytest.mark.asyncio
async def test_capture_accepts_documented_zstd_codec(
    monkeypatch, _enable_managed_test_storage
) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"test":{"url":"s3://generation-artifacts/root","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    request = _request()
    request["nvext"]["generation_artifact"]["codec"] = "zstd"
    session = _session(request)
    await session.admit(prompt_token_count=1, max_tokens=1)
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=None,
    )

    await session.finalize_choice(choice_index=0, token_start=0)

    artifact = _enable_managed_test_storage[
        "generation-artifacts/root/request-1/output.dynexp"
    ]
    assert struct.unpack_from("<H", artifact, 12)[0] == 1


def test_capture_rejects_unknown_codec() -> None:
    request = _request()
    request["nvext"]["generation_artifact"]["codec"] = "snappy"
    with pytest.raises(ArtifactCaptureError, match="codec"):
        _session(request)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_rl": False}, "--enable-rl"),
        ({"route_capture_enabled": False}, "enable-return-routed-experts"),
        ({"choice_count": 2}, "n=1"),
    ],
)
def test_capture_rejects_unsupported_backend_modes(overrides, message: str) -> None:
    with pytest.raises(ArtifactCaptureError, match=message):
        _session(_request("moe_routes"), **overrides)


@pytest.mark.asyncio
async def test_capture_preserves_nonzero_route_start(
    monkeypatch, _enable_managed_test_storage
) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"test":{"url":"s3://generation-artifacts/root","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    session = _session(_request("moe_routes"))
    await session.admit(prompt_token_count=3, max_tokens=1)
    routes = np.array([[[1, 3]]], dtype=np.int64)
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101, 102, 103],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=routes,
    )

    await session.finalize_choice(choice_index=0, token_start=2)

    decoded = decode_generation_artifact(
        _enable_managed_test_storage[
            "generation-artifacts/root/request-1/output.dynexp"
        ]
    )
    choice = decoded.choices[0]
    assert choice.routed_experts_token_start == 2
    np.testing.assert_array_equal(choice.routed_experts, routes.astype(np.uint8))


@pytest.mark.parametrize("token_start", [-1, True, 1.5])
def test_capture_rejects_invalid_route_start(token_start) -> None:
    session = _session(_request("moe_routes"))

    with pytest.raises(ArtifactCaptureError, match="non-negative integer"):
        session.validate_route_start(token_start)


@pytest.mark.asyncio
async def test_capture_rejects_route_start_outside_sequence() -> None:
    session = _session(_request("moe_routes"))
    await session.admit(prompt_token_count=1, max_tokens=1)
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=np.empty((0, 1, 1), dtype=np.int64),
    )

    with pytest.raises(ArtifactCaptureError, match="index sequence_token_ids"):
        await session.finalize_choice(choice_index=0, token_start=2)


@pytest.mark.asyncio
async def test_capture_fails_closed_for_untrusted_router_layout() -> None:
    session = _session(
        _request("moe_routes"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_experts=4, num_experts_per_tok=1)
        ),
    )
    with pytest.raises(ArtifactCaptureError, match="router layout"):
        await session.admit(prompt_token_count=2, max_tokens=1)


def test_capture_rejects_unknown_contents() -> None:
    with pytest.raises(ArtifactCaptureError, match="not supported"):
        _session(_request("topk_logprobs"))


@pytest.mark.asyncio
async def test_capture_allows_token_only_artifact_without_rl(monkeypatch) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"test":{"url":"s3://generation-artifacts/token-only","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    session = _session(_request(), enable_rl=False, route_capture_enabled=False)
    await session.admit(prompt_token_count=1, max_tokens=1)
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=None,
    )
    receipt = await session.finalize_choice(choice_index=0, token_start=9)
    assert receipt["contents"] == []


@pytest.mark.asyncio
async def test_capture_rejects_malformed_contract_and_logprob_alignment() -> None:
    assert (
        VllmGenerationArtifactSession.from_backend_request(
            {},
            model_config=_model_config(),
            enable_rl=True,
            route_capture_enabled=True,
            choice_count=1,
        )
        is None
    )
    malformed = _request("selected_logprobs")
    malformed["nvext"]["generation_artifact"]["format"] = "other"
    with pytest.raises(ArtifactCaptureError, match="generation_artifact_v1"):
        _session(malformed)

    invalid_target = _request()
    invalid_target["nvext"]["generation_artifact"]["delivery"]["target"][
        "object_key"
    ] = "../escape"
    with pytest.raises(ArtifactCaptureError, match="object_key"):
        _session(invalid_target)

    session = _session(_request("selected_logprobs"))
    await session.admit(prompt_token_count=1, max_tokens=1)
    with pytest.raises(ArtifactCaptureError, match="aligned"):
        session.record_chunk(
            choice_index=0,
            prompt_token_ids=[101],
            completion_token_ids=[201],
            selected_logprobs=None,
            routed_experts=None,
        )

    with pytest.raises(ArtifactCaptureError, match="aligned"):
        session.record_chunk(
            choice_index=0,
            prompt_token_ids=[101],
            completion_token_ids=[],
            selected_logprobs=[-0.5],
            routed_experts=None,
        )


@pytest.mark.asyncio
async def test_capture_rejects_missing_and_misaligned_routes() -> None:
    session = _session(_request("moe_routes"))
    await session.admit(prompt_token_count=1, max_tokens=1)
    with pytest.raises(ArtifactCaptureError, match="prompt token IDs"):
        await session.finalize_choice(choice_index=0, token_start=0)

    session = _session(_request("moe_routes"))
    await session.admit(prompt_token_count=1, max_tokens=1)
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=np.array([[[0]], [[1]]]),
    )
    with pytest.raises(ArtifactCaptureError, match="not aligned"):
        await session.finalize_choice(choice_index=0, token_start=0)


@pytest.mark.asyncio
async def test_capture_rejects_changed_prompt_and_resolves_integer_moe_frequency() -> (
    None
):
    session = _session(_request("moe_routes"))
    await session.admit(prompt_token_count=1, max_tokens=1)
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[],
        selected_logprobs=None,
        routed_experts=None,
    )
    with pytest.raises(ArtifactCaptureError, match="changed"):
        session.record_chunk(
            choice_index=0,
            prompt_token_ids=[102],
            completion_token_ids=[],
            selected_logprobs=None,
            routed_experts=None,
        )

    config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_experts=8,
            num_hidden_layers=6,
            first_k_dense_replace=1,
            moe_layer_freq=2,
        )
    )
    assert _resolve_router_layout(config, 2) == ((2, 4), (8, 8))

    qwen_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_experts=8,
            num_hidden_layers=6,
            first_k_dense_replace=1,
            decoder_sparse_step=2,
        )
    )
    assert _resolve_router_layout(qwen_config, 3) == ((1, 3, 5), (8, 8, 8))


@pytest.mark.asyncio
async def test_capture_admission_rejects_worst_case_before_generation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_DECODED_BYTES", "32")
    session = _session(
        _request("moe_routes"),
        model_config=_model_config(num_experts_per_tok=2),
    )
    with pytest.raises(ArtifactCaptureError, match="decoded byte limit"):
        await session.admit(prompt_token_count=2, max_tokens=2)


@pytest.mark.asyncio
async def test_capture_admission_estimates_canonical_route_dtype(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_DECODED_BYTES", "64")
    session = _session(
        _request("moe_routes"),
        model_config=_model_config(num_experts_per_tok=2),
    )

    await session.admit(prompt_token_count=2, max_tokens=2)


@pytest.mark.asyncio
async def test_capture_appends_stream_chunks_without_copying_prior_tokens() -> None:
    session = _session(_request("selected_logprobs"))
    await session.admit(prompt_token_count=2, max_tokens=2)
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101, 102],
        completion_token_ids=[201],
        selected_logprobs=[-0.1],
        routed_experts=None,
    )
    first = session._choices[0]

    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[],
        completion_token_ids=[202],
        selected_logprobs=[-0.2],
        routed_experts=None,
    )

    second = session._choices[0]
    assert second.previous is first
    assert second.completion_token_ids == (202,)
    assert second.selected_logprobs == (-0.2,)


@pytest.mark.asyncio
async def test_capture_admission_does_not_hold_pipeline_slot(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_PIPELINE_CONCURRENCY", "1")
    first = _session(_request())
    second = _session(_request())
    third = _session(_request())

    await first.admit(prompt_token_count=1, max_tokens=1)
    await asyncio.wait_for(
        second.admit(prompt_token_count=1, max_tokens=1), timeout=0.1
    )
    await asyncio.wait_for(third.admit(prompt_token_count=1, max_tokens=1), timeout=0.1)


def test_router_layout_rejects_conflicting_model_metadata() -> None:
    config = SimpleNamespace(
        num_hidden_layers=2,
        hf_config=SimpleNamespace(num_hidden_layers=3, num_experts=4),
    )
    with pytest.raises(ArtifactCaptureError, match="conflicting"):
        _resolve_router_layout(config, 2)


def test_router_layout_rejects_conflicting_layout_formulas() -> None:
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=6,
            num_experts=4,
            moe_layer_freq=2,
            decoder_sparse_step=2,
        )
    )
    with pytest.raises(ArtifactCaptureError, match="ambiguous"):
        _resolve_router_layout(config, 3)


@pytest.mark.asyncio
async def test_capture_rejects_payload_over_operator_limit(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_DECODED_BYTES", "1")
    session = _session(_request())
    with pytest.raises(ArtifactCaptureError, match="decoded byte limit"):
        await session.admit(prompt_token_count=1, max_tokens=1)


def test_capture_requires_admission_before_recording() -> None:
    session = _session(_request())

    with pytest.raises(ArtifactCaptureError, match="was not admitted"):
        session.record_chunk(
            choice_index=0,
            prompt_token_ids=[101],
            completion_token_ids=[201],
            selected_logprobs=None,
            routed_experts=None,
        )


@pytest.mark.asyncio
async def test_capture_rejects_tokens_beyond_admitted_bounds() -> None:
    session = _session(_request())
    await session.admit(prompt_token_count=1, max_tokens=1)

    with pytest.raises(ArtifactCaptureError, match="admitted token bounds"):
        session.record_chunk(
            choice_index=0,
            prompt_token_ids=[101],
            completion_token_ids=[201, 202],
            selected_logprobs=None,
            routed_experts=None,
        )


@pytest.mark.asyncio
async def test_admission_rejects_limit_above_wire_format_maximum(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_DECODED_BYTES", str((64 << 20) + 1))
    session = _session(_request())

    with pytest.raises(ArtifactCaptureError, match="decoded byte limit"):
        await session.admit(prompt_token_count=1, max_tokens=1)


@pytest.mark.asyncio
async def test_admission_enforces_managed_target_encoded_limit(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_BYTES", "1024")
    session = _session(_request())

    with pytest.raises(ArtifactCaptureError, match="target max_bytes"):
        await session.admit(prompt_token_count=64, max_tokens=64)


@pytest.mark.asyncio
async def test_admission_validates_managed_target_before_generation(
    monkeypatch,
) -> None:
    monkeypatch.delenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC")
    session = _session(_request())
    with pytest.raises(ArtifactCaptureError, match="not enabled"):
        await session.admit(prompt_token_count=1, max_tokens=1)

    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", "true")
    monkeypatch.delenv("DYN_GENERATION_ARTIFACT_STORAGE_PROFILES", raising=False)
    session = _session(_request())
    with pytest.raises(ArtifactCaptureError, match="profile is unknown"):
        await session.admit(prompt_token_count=1, max_tokens=1)

    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"test":{"url":"s3://generation-artifacts/root","allowed_prefixes":["other-request"],"create_only":true}}',
    )
    session = _session(_request())
    with pytest.raises(ArtifactCaptureError, match="outside the profile prefix"):
        await session.admit(prompt_token_count=1, max_tokens=1)


@pytest.mark.asyncio
async def test_admission_bounds_adversarial_router_metadata() -> None:
    session = _session(
        _request("moe_routes"),
        model_config=_model_config(
            num_experts_per_tok=1,
            num_hidden_layers=100_000,
        ),
    )

    with pytest.raises(ArtifactCaptureError, match="router layout"):
        await session.admit(prompt_token_count=1, max_tokens=1)


@pytest.mark.asyncio
async def test_cancelled_encode_keeps_pipeline_slot(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_PIPELINE_CONCURRENCY", "1")
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    call_count = 0

    def blocking_encode(_view):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        else:
            second_started.set()
        return SimpleNamespace(data=b"artifact")

    receipt = SimpleNamespace(actual_bytes=8, sha256="digest", object_id="object")
    monkeypatch.setattr(
        "dynamo.vllm.generation_artifact.encode_generation_artifact",
        blocking_encode,
    )
    monkeypatch.setattr(
        "dynamo.vllm.generation_artifact.put_artifact",
        AsyncMock(return_value=receipt),
    )

    async def ready_session():
        session = _session(_request())
        await session.admit(prompt_token_count=1, max_tokens=1)
        session.record_chunk(
            choice_index=0,
            prompt_token_ids=[101],
            completion_token_ids=[201],
            selected_logprobs=None,
            routed_experts=None,
        )
        return session

    first = await ready_session()
    second = await ready_session()
    first_task = asyncio.create_task(
        first.finalize_choice(choice_index=0, token_start=0)
    )
    assert await asyncio.to_thread(first_started.wait, 1)
    first_task.cancel()
    with suppress(asyncio.CancelledError):
        await first_task

    second_task = asyncio.create_task(
        second.finalize_choice(choice_index=0, token_start=0)
    )
    try:
        await asyncio.sleep(0.05)
        assert not second_started.is_set()
    finally:
        release_first.set()
    await second_task
