# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from dynamo.artifacts.capture import (
    ArtifactCaptureError,
    GenerationArtifactSession,
    _resolve_router_layout,
)
from dynamo.artifacts.format_v1 import decode_generation_artifact

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture(autouse=True)
def _enable_managed_test_storage(monkeypatch):
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", "true")
    stored = {}
    session = MagicMock()
    session.__aexit__ = AsyncMock()

    async def pipe_file(path, data, mode):
        assert mode == "create"
        stored[path] = data

    filesystem = SimpleNamespace(
        protocol="s3",
        async_impl=True,
        set_session=AsyncMock(return_value=session),
        _pipe_file=pipe_file,
    )
    with patch(
        "dynamo.artifacts.storage.url_to_fs",
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
    values = {"num_experts": 4, "num_hidden_layers": 1, **overrides}
    return SimpleNamespace(hf_config=SimpleNamespace(**values))


def _session(request: dict, **overrides) -> GenerationArtifactSession:
    values = {
        "model_config": _model_config(),
        "enable_rl": True,
        "route_capture_enabled": True,
        "choice_count": 1,
        **overrides,
    }
    session = GenerationArtifactSession.from_backend_request(request, **values)
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


def test_capture_rejects_nonzero_route_start_at_admission() -> None:
    session = _session(_request("moe_routes"))
    with pytest.raises(ArtifactCaptureError, match="prompt_start=0"):
        session.validate_route_start(1)


@pytest.mark.asyncio
async def test_capture_fails_closed_for_untrusted_router_layout(monkeypatch) -> None:
    monkeypatch.setenv(
        "DYN_GENERATION_ARTIFACT_STORAGE_PROFILES",
        '{"test":{"url":"s3://generation-artifacts/root","allowed_prefixes":["request-1"],"create_only":true}}',
    )
    session = _session(
        _request("moe_routes"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_experts=4)),
    )
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101, 102],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=np.array([[[0]], [[1]]]),
    )
    with pytest.raises(ArtifactCaptureError, match="router layout"):
        await session.finalize_choice(choice_index=0, token_start=0)


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
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=None,
    )
    receipt = await session.finalize_choice(choice_index=0, token_start=9)
    assert receipt["contents"] == []


def test_capture_rejects_malformed_contract_and_logprob_alignment() -> None:
    assert (
        GenerationArtifactSession.from_backend_request(
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
    with pytest.raises(ArtifactCaptureError, match="aligned"):
        session.record_chunk(
            choice_index=0,
            prompt_token_ids=[101],
            completion_token_ids=[201],
            selected_logprobs=None,
            routed_experts=None,
        )


@pytest.mark.asyncio
async def test_capture_rejects_missing_and_misaligned_routes() -> None:
    session = _session(_request("moe_routes"))
    with pytest.raises(ArtifactCaptureError, match="prompt token IDs"):
        await session.finalize_choice(choice_index=0, token_start=0)

    session = _session(_request("moe_routes"))
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=np.array([[[0]], [[1]]]),
    )
    with pytest.raises(ArtifactCaptureError, match="not aligned"):
        await session.finalize_choice(choice_index=0, token_start=0)


def test_capture_rejects_changed_prompt_and_resolves_integer_moe_frequency() -> None:
    session = _session(_request("moe_routes"))
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
async def test_capture_rejects_payload_over_operator_limit(monkeypatch) -> None:
    monkeypatch.setenv("DYN_GENERATION_ARTIFACT_MAX_DECODED_BYTES", "1")
    session = _session(_request())
    session.record_chunk(
        choice_index=0,
        prompt_token_ids=[101],
        completion_token_ids=[201],
        selected_logprobs=None,
        routed_experts=None,
    )
    with pytest.raises(ArtifactCaptureError, match="decoded byte limit"):
        await session.finalize_choice(choice_index=0, token_start=0)
