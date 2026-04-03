# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniStageRouter request isolation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from dynamo.vllm.omni import stage_router

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


class _CleanupConnector:
    def __init__(self, ok=True):
        self.ok = ok
        self.cleanup_calls = []
        self.put_calls = []

    def put(self, from_stage, to_stage, put_key, data):
        self.put_calls.append((from_stage, to_stage, put_key, data))
        return self.ok, 0, {}

    def cleanup(self, request_id):
        self.cleanup_calls.append(request_id)


def _make_stage_cfg(stage_id: int):
    return SimpleNamespace(
        stage_id=stage_id,
        engine_args=SimpleNamespace(model_stage=f"stage{stage_id}"),
    )


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Tests old router design — rewrite in issue-004 (opaque router)"
)
async def test_generate_uses_request_local_proxies():
    """Concurrent requests should not leak proxy engine_outputs across requests."""
    event_stage1_b_seen = asyncio.Event()
    stage2_inputs_by_request = {}

    async def stage0_handler(request):
        return {"shm_meta": {"value": f"{request['request_id']}-s0"}}

    async def stage1_handler(request):
        request_id = request["request_id"]
        if request_id == "req-A":
            await event_stage1_b_seen.wait()
        else:
            event_stage1_b_seen.set()
        return {"shm_meta": {"value": f"{request_id}-s1"}}

    async def stage2_handler(request):
        request_id = request["request_id"]
        stage2_inputs_by_request[request_id] = request["engine_inputs"]
        return {"shm_meta": {"value": f"{request_id}-s2"}}

    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(output_modalities=None)
    router.stage_configs = [
        _make_stage_cfg(0),
        _make_stage_cfg(1),
        SimpleNamespace(
            stage_id=2,
            engine_args=SimpleNamespace(model_stage="stage2"),
            engine_input_source=[0, 1],
            requires_multimodal_data=False,
        ),
    ]
    router.processors = {
        2: lambda proxies, engine_input_source, requests, requires_mm: [
            proxies[idx].engine_outputs for idx in engine_input_source
        ]
    }
    router.stage_clients = {
        "stage0": _StageClient(stage0_handler),
        "stage1": _StageClient(stage1_handler),
        "stage2": _StageClient(stage2_handler),
    }
    router.connectors = {}
    mock_formatter = AsyncMock()
    mock_formatter.format.return_value = {"finished": True}
    router._formatter = mock_formatter

    async def run_one():
        return [chunk async for chunk in router.generate({"prompt": "x"}, context=None)]

    with patch.object(
        stage_router, "_shm_deserialize", side_effect=lambda meta: meta["value"]
    ):
        with patch.object(
            stage_router,
            "_parse_engine_inputs",
            return_value={"engine_inputs": "x", "original_prompt": {"prompt": "x"}},
        ):
            with patch(
                "dynamo.common.utils.output_modalities.parse_request_type",
                return_value=(None, "chat"),
            ):
                with patch(
                    "dynamo.vllm.omni.stage_router.uuid.uuid4",
                    side_effect=["req-A", "req-B"],
                ):
                    await asyncio.gather(run_one(), run_one())

    # engine_outputs is wrapped in a list to match monolithic orchestrator format
    assert stage2_inputs_by_request["req-A"] == [["req-A-s0"], ["req-A-s1"]]
    assert stage2_inputs_by_request["req-B"] == [["req-B-s0"], ["req-B-s1"]]


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Tests old router design — rewrite in issue-004 (opaque router)"
)
async def test_generate_cleans_connectors_when_connector_put_fails():
    connector = _CleanupConnector(ok=False)

    async def stage0_handler(request):
        return {"shm_meta": {"value": "stage0-output"}}

    async def stage1_handler(request):
        return {"shm_meta": {"value": "unexpected"}}

    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(output_modalities=None)
    router.stage_configs = [_make_stage_cfg(0), _make_stage_cfg(1)]
    router.processors = {}
    router.stage_clients = {
        "stage0": _StageClient(stage0_handler),
        "stage1": _StageClient(stage1_handler),
    }
    router.connectors = {("0", "1"): connector}
    router._formatter = AsyncMock()

    with patch.object(stage_router, "_shm_deserialize", return_value="decoded"):
        with patch.object(
            stage_router,
            "_parse_engine_inputs",
            return_value={"engine_inputs": "x", "original_prompt": {"prompt": "x"}},
        ):
            with patch(
                "dynamo.common.utils.output_modalities.parse_request_type",
                return_value=(None, "chat"),
            ):
                with patch(
                    "dynamo.vllm.omni.stage_router.uuid.uuid4",
                    return_value="req-put-fail",
                ):
                    chunks = [
                        chunk
                        async for chunk in router.generate(
                            {"prompt": "x"}, context=None
                        )
                    ]

    assert chunks == [{"error": "connector.put() failed", "finished": True}]
    assert connector.cleanup_calls == ["req-put-fail"]


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Tests old router design — rewrite in issue-004 (opaque router)"
)
async def test_generate_cleans_connectors_when_stage_returns_error():
    connector = _CleanupConnector(ok=True)

    async def stage0_handler(request):
        return {"shm_meta": {"value": "stage0-output"}}

    async def stage1_handler(request):
        return {"error": "stage exploded", "finished": True}

    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(output_modalities=None)
    router.stage_configs = [_make_stage_cfg(0), _make_stage_cfg(1)]
    router.processors = {}
    router.stage_clients = {
        "stage0": _StageClient(stage0_handler),
        "stage1": _StageClient(stage1_handler),
    }
    router.connectors = {("0", "1"): connector}
    router._formatter = AsyncMock()

    with patch.object(stage_router, "_shm_deserialize", return_value="decoded"):
        with patch.object(
            stage_router,
            "_parse_engine_inputs",
            return_value={"engine_inputs": "x", "original_prompt": {"prompt": "x"}},
        ):
            with patch(
                "dynamo.common.utils.output_modalities.parse_request_type",
                return_value=(None, "chat"),
            ):
                with patch(
                    "dynamo.vllm.omni.stage_router.uuid.uuid4",
                    return_value="req-stage-error",
                ):
                    chunks = [
                        chunk
                        async for chunk in router.generate(
                            {"prompt": "x"}, context=None
                        )
                    ]

    assert chunks == [{"error": "stage exploded", "finished": True}]
    assert connector.cleanup_calls == ["req-stage-error"]


@pytest.mark.asyncio
async def test_generate_delegates_formatting_to_output_formatter():
    """Final stage output should be deserialized and passed to OutputFormatter."""
    fake_result = SimpleNamespace(final_output_type="image")
    mock_formatter = AsyncMock()
    mock_formatter.format.return_value = {"data": [{"b64_json": "abc"}]}

    async def stage0_handler(request):
        return {"shm_meta": {"some": "meta"}, "finished": True}

    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(output_modalities=None)
    router.stage_configs = [_make_stage_cfg(0)]
    router.processors = {}
    router.stage_clients = {"stage0": _StageClient(stage0_handler)}
    router.connectors = {}
    router._formatter = mock_formatter

    request = {"prompt": "x", "response_format": "b64_json"}
    with patch.object(stage_router, "_shm_deserialize", return_value=fake_result):
        with patch.object(
            stage_router,
            "_parse_engine_inputs",
            return_value={"engine_inputs": "x", "original_prompt": {"prompt": "x"}},
        ):
            with patch(
                "dynamo.common.utils.output_modalities.parse_request_type",
                return_value=(None, "image_generation"),
            ):
                with patch(
                    "dynamo.vllm.omni.stage_router.uuid.uuid4",
                    return_value="req-fmt",
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

    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(output_modalities=None)
    router.stage_configs = [_make_stage_cfg(0)]
    router.processors = {}
    router.stage_clients = {"stage0": _StageClient(stage0_handler)}
    router.connectors = {}

    with patch.object(
        stage_router,
        "_parse_engine_inputs",
        return_value={"engine_inputs": "x", "original_prompt": {"prompt": "x"}},
    ):
        with patch(
            "dynamo.common.utils.output_modalities.parse_request_type",
            return_value=(None, "chat"),
        ):
            with patch("dynamo.vllm.omni.stage_router.uuid.uuid4", return_value="r"):
                chunks = [
                    c async for c in router.generate({"prompt": "x"}, context=None)
                ]

    assert chunks == [{"error": "No SHM output from final stage", "finished": True}]


# ── issue-002: _parse_engine_inputs builds original_prompt ────────────


class TestParseEngineInputsOriginalPrompt:
    """_parse_engine_inputs must return original_prompt with all fields
    that processor functions (ar2diffusion etc.) read from prompt."""

    def test_video_request_builds_original_prompt(self):
        """original_prompt carries geometry/params for processors; engine_inputs is just text."""
        from unittest.mock import MagicMock

        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.stage_router import _parse_engine_inputs

        config = MagicMock()
        config.default_video_fps = 16
        request = {
            "prompt": "a dog running",
            "size": "832x480",
            "nvext": {"num_frames": 30, "num_inference_steps": 20, "seed": 42},
        }
        result = _parse_engine_inputs(request, RequestType.VIDEO_GENERATION, config)

        # engine_inputs = just the text prompt for the engine
        assert result["engine_inputs"]["prompt"] == "a dog running"
        # original_prompt = richer dict with geometry/params for processors (ar2diffusion etc.)
        op = result["original_prompt"]
        assert op["prompt"] == "a dog running"
        assert op["height"] == 480
        assert op["width"] == 832
        assert op["num_inference_steps"] == 20
        assert op["seed"] == 42

    def test_image_request_builds_original_prompt(self):
        from unittest.mock import MagicMock

        from dynamo.common.utils.output_modalities import RequestType
        from dynamo.vllm.omni.stage_router import _parse_engine_inputs

        config = MagicMock()
        request = {
            "prompt": "a red apple",
            "size": "1024x1024",
            "nvext": {"num_inference_steps": 50},
        }
        result = _parse_engine_inputs(request, RequestType.IMAGE_GENERATION, config)

        op = result["original_prompt"]
        assert op["prompt"] == "a red apple"
        assert op["height"] == 1024
        assert op["width"] == 1024
        assert op["num_inference_steps"] == 50
