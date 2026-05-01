# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniStageRouter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.common.utils.output_modalities import RequestType

try:
    from dynamo.vllm.omni import stage_router
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _Chunk:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


class _StageClient:
    def __init__(self, handler):
        self._handler = handler

    async def round_robin(self, request):
        async def _gen():
            payload = await self._handler(request)
            yield _Chunk(payload)

        return _gen()


def _make_stage_cfg(stage_id: int):
    return SimpleNamespace(
        stage_id=stage_id,
        engine_args=SimpleNamespace(model_stage=f"stage{stage_id}"),
    )


def _make_router(stage_configs, stage_clients, formatter=None, output_modalities=None):
    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(output_modalities=output_modalities)
    router.stage_configs = stage_configs
    router.stage_clients = stage_clients
    router._formatter = formatter or AsyncMock()
    return router


def _patched_generate(router, request, request_id="req-1", request_type="chat"):
    return (
        patch(
            "dynamo.vllm.omni.stage_router.parse_request_type",
            return_value=(None, request_type),
        ),
        patch("dynamo.vllm.omni.stage_router.uuid.uuid4", return_value=request_id),
    )


# ── issue-004: opaque router ──────────────────────────────


@pytest.mark.asyncio
async def test_generate_passes_stage_connector_refs_opaquely():
    """Router must pass stage_connector_refs from stage output to next stage unchanged."""
    stage1_received = {}

    async def stage0_handler(request):
        return {
            "original_prompt": {"prompt": "hi"},
            "stage_connector_refs": {"0": {"shm_name": "abc", "size": 42}},
            "finished": True,
        }

    async def stage1_handler(request):
        stage1_received.update(request)
        return {"shm_meta": {"x": 1}, "finished": True}

    mock_formatter = AsyncMock()
    mock_formatter.format.return_value = {"finished": True}
    router = _make_router(
        stage_configs=[_make_stage_cfg(0), _make_stage_cfg(1)],
        stage_clients={
            "stage0": _StageClient(stage0_handler),
            "stage1": _StageClient(stage1_handler),
        },
        formatter=mock_formatter,
    )

    p1, p2 = _patched_generate(router, {"prompt": "x"})
    with p1, p2:
        with patch.object(
            stage_router, "shm_deserialize", return_value=SimpleNamespace()
        ):
            [c async for c in router.generate({"prompt": "x"}, None)]

    # Router must forward stage_connector_refs and original_prompt verbatim — never inspect them.
    assert stage1_received["stage_connector_refs"] == {
        "0": {"shm_name": "abc", "size": 42}
    }
    assert stage1_received["original_prompt"] == {"prompt": "hi"}
    assert stage1_received["request_id"] == "req-1"
    # 'finished' must be stripped — it is a router signal, not a stage protocol field.
    assert "finished" not in stage1_received


@pytest.mark.asyncio
async def test_generate_concurrent_requests_have_independent_connector_refs():
    """Concurrent requests must carry independent stage_connector_refs (no cross-leakage)."""
    stage1_refs_by_request: dict = {}
    event = asyncio.Event()

    async def stage0_handler(request):
        rid = request["request_id"]
        return {
            "original_prompt": {"prompt": "x"},
            "stage_connector_refs": {"0": f"ref-for-{rid}"},
            "finished": True,
        }

    async def stage1_handler(request):
        rid = request["request_id"]
        if rid == "req-A":
            await event.wait()
        else:
            event.set()
        stage1_refs_by_request[rid] = request.get("stage_connector_refs")
        return {"shm_meta": {"x": 1}, "finished": True}

    mock_formatter = AsyncMock()
    mock_formatter.format.return_value = {"finished": True}
    router = _make_router(
        stage_configs=[_make_stage_cfg(0), _make_stage_cfg(1)],
        stage_clients={
            "stage0": _StageClient(stage0_handler),
            "stage1": _StageClient(stage1_handler),
        },
        formatter=mock_formatter,
    )

    async def run_one(request_id):
        with patch(
            "dynamo.vllm.omni.stage_router.parse_request_type",
            return_value=(None, "chat"),
        ):
            with patch(
                "dynamo.vllm.omni.stage_router.uuid.uuid4", return_value=request_id
            ):
                with patch.object(
                    stage_router, "shm_deserialize", return_value=SimpleNamespace()
                ):
                    return [c async for c in router.generate({"prompt": "x"}, None)]

    await asyncio.gather(run_one("req-A"), run_one("req-B"))

    assert stage1_refs_by_request["req-A"] == {"0": "ref-for-req-A"}
    assert stage1_refs_by_request["req-B"] == {"0": "ref-for-req-B"}


@pytest.mark.asyncio
async def test_generate_stage_error_stops_pipeline():
    """Error from any stage must immediately stop the pipeline; later stages must not run."""
    stage1_called = False

    async def stage0_handler(request):
        return {"error": "thinker exploded", "finished": True}

    async def stage1_handler(request):
        nonlocal stage1_called
        stage1_called = True
        return {"shm_meta": {"x": 1}, "finished": True}

    router = _make_router(
        stage_configs=[_make_stage_cfg(0), _make_stage_cfg(1)],
        stage_clients={
            "stage0": _StageClient(stage0_handler),
            "stage1": _StageClient(stage1_handler),
        },
    )

    p1, p2 = _patched_generate(router, {"prompt": "x"})
    with p1, p2:
        chunks = [c async for c in router.generate({"prompt": "x"}, None)]

    assert chunks == [{"error": "thinker exploded", "finished": True}]
    assert not stage1_called


# ── existing tests (formatting + error paths) ────────────


@pytest.mark.asyncio
async def test_generate_delegates_formatting_to_output_formatter():
    """Final stage output should be deserialized and passed to OutputFormatter."""
    fake_result = SimpleNamespace(final_output_type="image")
    mock_formatter = AsyncMock()
    mock_formatter.format.return_value = {"data": [{"b64_json": "abc"}]}

    async def stage0_handler(request):
        return {"shm_meta": {"some": "meta"}, "finished": True}

    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(stage0_handler)},
        formatter=mock_formatter,
    )

    request = {"prompt": "x", "response_format": "b64_json"}
    with patch.object(stage_router, "shm_deserialize", return_value=fake_result):
        with patch(
            "dynamo.vllm.omni.stage_router.parse_request_type",
            return_value=(None, "image_generation"),
        ):
            with patch(
                "dynamo.vllm.omni.stage_router.uuid.uuid4", return_value="req-fmt"
            ):
                chunks = [c async for c in router.generate(request, context=None)]

    assert chunks == [{"data": [{"b64_json": "abc"}]}]
    mock_formatter.format.assert_awaited_once_with(
        fake_result,
        "req-fmt",
        request_type="image_generation",
        response_format="b64_json",
    )


@pytest.mark.asyncio
async def test_generate_yields_error_when_no_shm_meta():
    """When final stage returns no shm_meta, generate yields an error."""

    async def stage0_handler(request):
        return {"finished": True}

    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(stage0_handler)},
    )

    with patch(
        "dynamo.vllm.omni.stage_router.parse_request_type",
        return_value=(None, "chat"),
    ):
        with patch("dynamo.vllm.omni.stage_router.uuid.uuid4", return_value="r"):
            chunks = [c async for c in router.generate({"prompt": "x"}, context=None)]

    assert chunks == [{"error": "No SHM output from final stage", "finished": True}]


# ── issue-007: router forwards raw request to stage 0 ────────────


@pytest.mark.asyncio
async def test_generate_forwards_raw_request_to_stage0():
    """Stage 0 must receive the raw request fields + request_id (no router parsing)."""
    stage0_received = {}

    async def stage0_handler(request):
        stage0_received.update(request)
        return {"shm_meta": {"x": 1}, "finished": True}

    mock_formatter = AsyncMock()
    mock_formatter.format.return_value = {"finished": True}
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(stage0_handler)},
        formatter=mock_formatter,
    )

    request = {
        "prompt": "a dog",
        "size": "832x480",
        "nvext": {"num_inference_steps": 30},
    }
    with patch(
        "dynamo.vllm.omni.stage_router.parse_request_type",
        return_value=(None, "video_generation"),
    ):
        with patch("dynamo.vllm.omni.stage_router.uuid.uuid4", return_value="req-raw"):
            with patch.object(
                stage_router, "shm_deserialize", return_value=SimpleNamespace()
            ):
                [c async for c in router.generate(request, None)]

    assert stage0_received["request_id"] == "req-raw"
    assert stage0_received["prompt"] == "a dog"
    assert stage0_received["size"] == "832x480"
    assert stage0_received["nvext"] == {"num_inference_steps": 30}


# ── Context normalization: audio data_source vs non-audio ─────────────────


class TestStageRouterContextNormalization:
    """generate() normalizes audio data_source/response_format before calling formatter."""

    def _make_router_with_formatter(self, mock_formatter):
        """One-stage router with a controllable formatter."""
        return _make_router(
            stage_configs=[_make_stage_cfg(0)],
            stage_clients={},  # overridden per test
            formatter=mock_formatter,
        )

    @pytest.mark.asyncio
    async def test_audio_request_maps_data_source_to_response_format(self):
        """data_source present: formatter sees response_format=data_source, output_format=response_format."""
        formatter_calls: list = []

        async def fake_format(result, req_id, *, request_type, **ctx):
            formatter_calls.append(ctx)
            return {"finished": True}

        mock_formatter = MagicMock()
        mock_formatter.format = fake_format

        async def stage0_handler(request):
            return {"shm_meta": {"x": 1}, "finished": True}

        router = _make_router(
            stage_configs=[_make_stage_cfg(0)],
            stage_clients={"stage0": _StageClient(stage0_handler)},
            formatter=mock_formatter,
            output_modalities=["audio"],
        )

        request = {"prompt": "hi", "data_source": "url", "response_format": "mp3"}
        p1, p2 = _patched_generate(
            router, request, request_type=RequestType.AUDIO_GENERATION
        )
        with p1, p2:
            with patch.object(
                stage_router, "shm_deserialize", return_value=SimpleNamespace()
            ):
                [c async for c in router.generate(request, None)]

        assert len(formatter_calls) == 1
        ctx = formatter_calls[0]
        assert ctx["response_format"] == "url"  # data_source
        assert ctx["output_format"] == "mp3"  # response_format (codec)

    @pytest.mark.asyncio
    async def test_audio_request_b64_json_maps_correctly(self):
        formatter_calls: list = []

        async def fake_format(result, req_id, *, request_type, **ctx):
            formatter_calls.append(ctx)
            return {"finished": True}

        mock_formatter = MagicMock()
        mock_formatter.format = fake_format

        async def stage0_handler(request):
            return {"shm_meta": {"x": 1}, "finished": True}

        router = _make_router(
            stage_configs=[_make_stage_cfg(0)],
            stage_clients={"stage0": _StageClient(stage0_handler)},
            formatter=mock_formatter,
            output_modalities=["audio"],
        )

        request = {"prompt": "hi", "data_source": "b64_json", "response_format": "opus"}
        p1, p2 = _patched_generate(
            router, request, request_type=RequestType.AUDIO_GENERATION
        )
        with p1, p2:
            with patch.object(
                stage_router, "shm_deserialize", return_value=SimpleNamespace()
            ):
                [c async for c in router.generate(request, None)]

        ctx = formatter_calls[0]
        assert ctx["response_format"] == "b64_json"
        assert ctx["output_format"] == "opus"

    @pytest.mark.asyncio
    async def test_non_audio_request_passes_through_unchanged(self):
        """No data_source: response_format and output_format passed as-is."""
        formatter_calls: list = []

        async def fake_format(result, req_id, *, request_type, **ctx):
            formatter_calls.append(ctx)
            return {"finished": True}

        mock_formatter = MagicMock()
        mock_formatter.format = fake_format

        async def stage0_handler(request):
            return {"shm_meta": {"x": 1}, "finished": True}

        router = _make_router(
            stage_configs=[_make_stage_cfg(0)],
            stage_clients={"stage0": _StageClient(stage0_handler)},
            formatter=mock_formatter,
        )

        request = {"prompt": "cat", "response_format": "url", "output_format": "mp4"}
        p1, p2 = _patched_generate(router, request)
        with p1, p2:
            with patch.object(
                stage_router, "shm_deserialize", return_value=SimpleNamespace()
            ):
                [c async for c in router.generate(request, None)]

        ctx = formatter_calls[0]
        assert ctx["response_format"] == "url"
        assert ctx["output_format"] == "mp4"

    @pytest.mark.asyncio
    async def test_no_format_fields_omitted_from_context(self):
        """Fields not present in request are not forwarded to formatter."""
        formatter_calls: list = []

        async def fake_format(result, req_id, *, request_type, **ctx):
            formatter_calls.append(ctx)
            return {"finished": True}

        mock_formatter = MagicMock()
        mock_formatter.format = fake_format

        async def stage0_handler(request):
            return {"shm_meta": {"x": 1}, "finished": True}

        router = _make_router(
            stage_configs=[_make_stage_cfg(0)],
            stage_clients={"stage0": _StageClient(stage0_handler)},
            formatter=mock_formatter,
        )

        request = {"prompt": "cat"}
        p1, p2 = _patched_generate(router, request)
        with p1, p2:
            with patch.object(
                stage_router, "shm_deserialize", return_value=SimpleNamespace()
            ):
                [c async for c in router.generate(request, None)]

        ctx = formatter_calls[0]
        assert "response_format" not in ctx
        assert "output_format" not in ctx
