# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the sample engine's disaggregation dispatch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo.common.backend.publisher import PushSource  # noqa: E402
from dynamo.common.backend.sample_engine import SampleLLMEngine  # noqa: E402
from dynamo.common.constants import DisaggregationMode  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _ctx(stopped: bool = False) -> MagicMock:
    """Minimal Context stand-in. The sample engine only consults
    ``is_stopped`` on each iteration — anything else is unused."""
    ctx = MagicMock()
    ctx.is_stopped.return_value = stopped
    return ctx


async def _collect(engine: SampleLLMEngine, request: dict) -> list[dict]:
    return [chunk async for chunk in engine.generate(request, _ctx())]


async def test_aggregated_mode_emits_max_tokens_chunks():
    # Aggregated mode produces one chunk per token, with the terminal
    # marked length and carrying completion_usage. No disaggregated_params.
    engine = SampleLLMEngine(
        max_tokens=4,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.AGGREGATED,
    )
    chunks = await _collect(engine, {"token_ids": [1, 2]})
    assert len(chunks) == 4
    assert chunks[-1]["finish_reason"] == "length"
    assert "disaggregated_params" not in chunks[-1]


async def test_prefill_mode_caps_to_one_token_with_disaggregated_params():
    engine = SampleLLMEngine(
        max_tokens=16,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.PREFILL,
    )
    # Even when the client asks for 8, prefill caps to 1.
    chunks = await _collect(
        engine,
        {"token_ids": [1, 2, 3], "stop_conditions": {"max_tokens": 8}},
    )
    assert len(chunks) == 1
    terminal = chunks[0]
    assert terminal["finish_reason"] == "length"
    assert terminal["completion_usage"]["completion_tokens"] == 1
    # The synthetic handle on the prefill terminal is what the frontend's
    # PrefillRouter forwards to the decode peer.
    assert "disaggregated_params" in terminal
    assert "sample_handle" in terminal["disaggregated_params"]


async def test_decode_mode_rejects_request_without_prefill_result():
    engine = SampleLLMEngine(
        max_tokens=4,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    # Decode workers can't proceed without the prefill peer's payload.
    with pytest.raises(ValueError, match="prefill_result"):
        async for _ in engine.generate({"token_ids": [1, 2, 3]}, _ctx()):
            pass


async def test_decode_mode_runs_to_completion_when_prefill_result_provided():
    engine = SampleLLMEngine(
        max_tokens=4,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    chunks = await _collect(
        engine,
        {
            "token_ids": [1, 2, 3],
            "prefill_result": {"disaggregated_params": {"sample_handle": "from-test"}},
        },
    )
    assert len(chunks) == 4
    terminal = chunks[-1]
    assert terminal["finish_reason"] == "length"
    # Decode does not stamp a fresh disaggregated_params on its response —
    # that's the prefill role.
    assert "disaggregated_params" not in terminal


async def test_decode_mode_rejects_raw_multimodal_payload():
    engine = SampleLLMEngine(
        max_tokens=1,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image": [{"url": "data:image/png;base64,AA=="}]},
        "prefill_result": {"disaggregated_params": {"sample_handle": "from-test"}},
    }

    with pytest.raises(ValueError, match="decode worker should not receive raw"):
        await _collect(engine, request)


async def test_from_args_propagates_mode_to_worker_config():
    engine, worker_config = await SampleLLMEngine.from_args(
        ["--disaggregation-mode", "prefill"]
    )
    assert engine.disaggregation_mode is DisaggregationMode.PREFILL
    assert worker_config.disaggregation_mode is DisaggregationMode.PREFILL


async def test_aggregated_mode_processes_multimodal_kwargs_locally(monkeypatch):
    engine = SampleLLMEngine(max_tokens=1, delay=0.0)
    encode = AsyncMock(wraps=engine._encode_multimodal)
    monkeypatch.setattr(engine, "_encode_multimodal", encode)
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {"image": [{"url": "data:image/png;base64,AA=="}]},
        "mm_processor_kwargs": {"min_pixels": 64},
    }

    chunks = await _collect(engine, request)

    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["engine_data"]["sample_multimodal"]["multimodal_kwargs"] == {
        "multi_modal_data": request["multi_modal_data"],
        "mm_processor_kwargs": request["mm_processor_kwargs"],
    }
    encode.assert_awaited_once_with(request)


async def test_encode_mode_emits_single_terminal_with_encoder_result():
    engine = SampleLLMEngine(
        delay=0.0,
        disaggregation_mode=DisaggregationMode.ENCODE,
    )
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image": [{"url": "data:image/png;base64,AA=="}]},
    }

    chunks = await _collect(engine, request)

    assert len(chunks) == 1
    terminal = chunks[0]
    assert terminal["token_ids"] == []
    assert terminal["finish_reason"] == "stop"
    assert terminal["completion_usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 0,
        "total_tokens": 3,
    }
    encoder_result = terminal["encoder_result"]
    assert encoder_result["handle"].startswith("sample-encoder:")
    assert (
        encoder_result["multimodal_kwargs"]["multi_modal_data"]
        == request["multi_modal_data"]
    )


@pytest.mark.parametrize("stop_checks", [[True], [False, True]])
async def test_encode_mode_observes_cancellation(stop_checks):
    engine = SampleLLMEngine(
        delay=0.0,
        disaggregation_mode=DisaggregationMode.ENCODE,
    )
    context = _ctx()
    context.is_stopped.side_effect = stop_checks

    chunks = [
        chunk
        async for chunk in engine.generate(
            {"token_ids": [1, 2, 3], "multi_modal_data": {"image": []}}, context
        )
    ]

    assert chunks == [
        {
            "token_ids": [],
            "index": 0,
            "finish_reason": "cancelled",
            "completion_usage": {
                "prompt_tokens": 3,
                "completion_tokens": 0,
                "total_tokens": 3,
            },
        }
    ]


async def test_encoder_routed_worker_requires_encoder_result():
    engine = SampleLLMEngine(
        max_tokens=1,
        delay=0.0,
        route_to_encoder=True,
    )
    request = {
        "token_ids": [1],
        "multi_modal_data": {"image": [{"url": "data:image/png;base64,AA=="}]},
    }

    with pytest.raises(ValueError, match="no encoder_result"):
        await _collect(engine, request)


async def test_encoder_routed_worker_rejects_malformed_encoder_result():
    engine = SampleLLMEngine(
        max_tokens=1,
        delay=0.0,
        route_to_encoder=True,
    )
    request = {
        "token_ids": [1],
        "multi_modal_data": {"image": [{"url": "data:image/png;base64,AA=="}]},
        "encoder_result": {"handle": "not-a-sample-handle"},
    }

    with pytest.raises(ValueError, match=r"encoder_result\.handle"):
        await _collect(engine, request)


async def test_multimodal_epd_handoff_contract():
    """Exercise Encode -> Prefill -> Decode using separate role instances."""
    encode = SampleLLMEngine(
        delay=0.0,
        disaggregation_mode=DisaggregationMode.ENCODE,
    )
    prefill = SampleLLMEngine(
        delay=0.0,
        disaggregation_mode=DisaggregationMode.PREFILL,
        route_to_encoder=True,
    )
    decode = SampleLLMEngine(
        max_tokens=2,
        delay=0.0,
        disaggregation_mode=DisaggregationMode.DECODE,
    )
    multimodal_request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image": [{"url": "data:image/png;base64,AA=="}]},
    }

    [encode_terminal] = await _collect(encode, multimodal_request)
    [prefill_terminal] = await _collect(
        prefill,
        {
            **multimodal_request,
            "encoder_result": encode_terminal["encoder_result"],
            "stop_conditions": {"max_tokens": 8},
        },
    )
    decode_chunks = await _collect(
        decode,
        {
            "token_ids": multimodal_request["token_ids"],
            "prefill_result": {
                "disaggregated_params": prefill_terminal["disaggregated_params"]
            },
        },
    )

    assert prefill_terminal["finish_reason"] == "length"
    assert (
        prefill_terminal["engine_data"]["sample_multimodal"]
        == encode_terminal["encoder_result"]
    )
    assert len(decode_chunks) == 2
    assert decode_chunks[-1]["finish_reason"] == "length"


async def test_from_args_propagates_encode_routing():
    engine, worker_config = await SampleLLMEngine.from_args(
        ["--disaggregation-mode", "prefill", "--route-to-encoder"]
    )

    assert engine.route_to_encoder is True
    assert worker_config.route_to_encoder is True


async def test_from_args_can_disable_kv_routing():
    _, worker_config = await SampleLLMEngine.from_args(["--disable-kv-routing"])

    assert worker_config.enable_kv_routing is False


async def test_encode_mode_opts_out_of_kv_publishers():
    engine = SampleLLMEngine(disaggregation_mode=DisaggregationMode.ENCODE)

    assert await engine.kv_event_sources() == []
    assert engine.component_metrics_dp_ranks() == []


async def test_source_descriptors_have_expected_shape():
    engine = SampleLLMEngine(max_tokens=4, delay=0.0)
    [kv_src] = await engine.kv_event_sources()
    assert isinstance(kv_src, PushSource) and kv_src.dp_rank == 0
    assert engine.component_metrics_dp_ranks() == [0]


async def test_attach_snapshot_publisher_stashes_handle():
    engine = SampleLLMEngine(max_tokens=2, delay=0.0)
    sentinel = object()
    engine.attach_snapshot_publisher(sentinel)
    assert engine._snapshot_publisher is sentinel


async def test_cleanup_joins_publisher_thread_started_via_on_ready():
    engine = SampleLLMEngine(max_tokens=2, delay=0.0)
    [push_src] = await engine.kv_event_sources()
    push_src.on_ready(MagicMock())
    assert engine._publish_thread is not None and engine._publish_thread.is_alive()
    await engine.cleanup()
    assert not engine._publish_thread.is_alive()


async def test_health_check_payload_decode_probe_passes_generate(monkeypatch):
    """Decode probe drives `generate()` end-to-end: `is_probe(request)`
    triggers the bypass branch, `require_prefill_result` is skipped, and
    the stream completes — even though the payload carries no synthetic
    `prefill_result`."""
    monkeypatch.delenv("DYN_HEALTH_CHECK_PAYLOAD", raising=False)
    engine = SampleLLMEngine(
        max_tokens=2, delay=0.0, disaggregation_mode=DisaggregationMode.DECODE
    )
    payload = await engine.health_check_payload()
    assert payload is not None
    assert "prefill_result" not in payload  # no payload trick
    chunks = await _collect(engine, payload)  # type: ignore[arg-type]
    assert any(c.get("finish_reason") for c in chunks)


async def test_decode_without_probe_marker_still_requires_prefill_result():
    """Negative: a real DECODE request (no `_HEALTH_CHECK` marker) must
    still trip `require_prefill_result`. Guards against the bypass branch
    swallowing real-traffic misconfigurations."""
    engine = SampleLLMEngine(
        max_tokens=2, delay=0.0, disaggregation_mode=DisaggregationMode.DECODE
    )
    with pytest.raises(ValueError, match="prefill_result"):
        await _collect(engine, {"token_ids": [1]})


async def test_logprobs_absent_when_not_requested():
    engine = SampleLLMEngine(max_tokens=2, delay=0.0)
    chunks = await _collect(engine, {"token_ids": [1]})
    for chunk in chunks:
        assert "log_probs" not in chunk
        assert "top_logprobs" not in chunk


async def test_logprobs_zero_emits_selected_only():
    # logprobs=0 means "selected token only" — no top_logprobs.
    engine = SampleLLMEngine(max_tokens=2, delay=0.0)
    chunks = await _collect(
        engine, {"token_ids": [1], "output_options": {"logprobs": 0}}
    )
    assert all("log_probs" in c for c in chunks)
    assert all("top_logprobs" not in c for c in chunks)
    assert all(len(c["log_probs"]) == len(c["token_ids"]) for c in chunks)


async def test_logprobs_with_top_k_emits_alternatives():
    engine = SampleLLMEngine(max_tokens=2, delay=0.0)
    chunks = await _collect(
        engine, {"token_ids": [1], "output_options": {"logprobs": 3}}
    )
    for chunk in chunks:
        assert len(chunk["log_probs"]) == len(chunk["token_ids"])
        assert len(chunk["top_logprobs"]) == len(chunk["token_ids"])
        for position in chunk["top_logprobs"]:
            # k=3 -> selected + 3 alternatives = 4 entries; ranks 1..4.
            assert len(position) == 4
            assert [entry["rank"] for entry in position] == [1, 2, 3, 4]
