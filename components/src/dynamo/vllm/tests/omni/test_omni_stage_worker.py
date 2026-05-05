# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniStageWorker.

No GPU, no vllm_omni — uses mock StageEngine matching AsyncOmni.generate() signature.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from dynamo.vllm.omni.stage_worker import (
        _ASYNC_PREPARE_KEY,
        _ASYNC_PREWARM_KEY,
        _ASYNC_PREWARM_READY_KEY,
        OmniStageWorker,
        _ensure_cumulative_token_ids,
        _prepare_connector_payload,
        _Proxy,
        _stage_config_to_dict,
    )
    from dynamo.vllm.omni.utils import _build_sampling_params
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _MockEngine:
    """Satisfies StageEngine Protocol — matches AsyncOmni interface."""

    engine = None  # satisfies StageEngine.engine

    def __init__(self, output=None):
        self.received_prompt = None
        self.received_request_id = None
        self.received_sampling_params_list = None
        self._output = output or {"output": "mock", "finished": True}

    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        self.received_prompt = prompt
        self.received_request_id = request_id
        self.received_sampling_params_list = sampling_params_list

        async def _gen():
            yield self._output

        return _gen()

    async def get_tokenizer(self):
        return None


class _TokenizerEngine(_MockEngine):
    async def get_tokenizer(self):
        return SimpleNamespace(encode=lambda text, add_special_tokens=False: [9, 8, 7])


class _ErrorEngine:
    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        async def _gen():
            raise RuntimeError("engine exploded")
            yield  # make it an async generator

        return _gen()


class _MockContext:
    def id(self):
        return "test-req-id"


def _make_stage_config(**overrides):
    defaults = dict(
        stage_type="llm",
        final_output=False,
        final_output_type="text",
        engine_input_source=[],
        engine_args=SimpleNamespace(async_chunk=False),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_worker(engine=None, stage_config=None, connectors=None, stage_id=0):
    return OmniStageWorker(
        engine=engine or _MockEngine(),
        stage_config=stage_config or _make_stage_config(),
        connectors=connectors or {},
        stage_id=stage_id,
    )


@pytest.mark.asyncio
async def test_direct_input_path():
    """Stage 0 direct path: engine receives the full request dict as prompt."""
    engine = _MockEngine()
    worker = _make_worker(engine=engine)
    request = {"engine_inputs": {"prompt": "hello"}, "sampling_params_list": None}

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    # Direct path (no request_id, no stage_connector_refs) passes the whole request as prompt.
    assert engine.received_prompt == request
    assert any("shm_meta" in c for c in chunks)


@pytest.mark.asyncio
async def test_stage_connector_refs_input_path():
    """Stage N>0: engine receives output fetched from connector via stage_connector_refs."""
    engine = _MockEngine()
    fetched_prompt = {"prior_token_ids": [1, 2, 3]}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_prompt}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1", "size": 10})

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
    )
    request = {
        "request_id": "req-1",
        "original_prompt": {"prompt": "hello"},
        "stage_connector_refs": {"0": {"name": "ref0", "size": 5}},
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    in_connector.get.assert_called_once_with(
        "0", "1", "req-1", metadata={"name": "ref0", "size": 5}
    )
    assert engine.received_prompt == fetched_prompt
    assert len(chunks) == 1
    assert chunks[0]["stage_connector_refs"]["1"] == {"name": "ref1", "size": 10}
    assert chunks[0]["stage_connector_refs"]["0"] == {"name": "ref0", "size": 5}
    assert chunks[0]["original_prompt"] == {"prompt": "hello"}


@pytest.mark.asyncio
async def test_stage_connector_refs_builds_engine_core_request():
    """Stage N>0 without processor: upstream with .outputs builds OmniEngineCoreRequest."""
    engine = _MockEngine()

    # Mock upstream output that looks like a real RequestOutput (has .outputs[0].token_ids)
    mock_output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[100, 200, 300])],
        prompt_token_ids=[1, 2],
    )

    in_connector = MagicMock()
    in_connector.get.return_value = (
        mock_output  # raw object, not {"engine_inputs": ...}
    )

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
        stage_config=_make_stage_config(
            default_sampling_params={"temperature": 0.9, "max_tokens": 100},
        ),
    )
    # Mock the engine's output_processors for registration
    engine.engine = MagicMock()
    request = {
        "request_id": "req-ecr",
        "original_prompt": {"prompt": "hello"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    _ = [chunk async for chunk in worker.generate(request, _MockContext())]

    # The engine should receive an OmniEngineCoreRequest (not the raw dict)
    assert hasattr(engine.received_prompt, "prompt_token_ids")
    assert engine.received_prompt.prompt_token_ids == [100, 200, 300]
    # Output processor should have been registered
    engine.engine.output_processors[0].add_request.assert_called_once()


@pytest.mark.asyncio
async def test_stage_connector_refs_with_processor():
    """Stage N>0 with processor: processor receives stage_list built from connector output."""
    engine = _MockEngine()
    fetched_output = {"latents": [0.1, 0.2]}
    processed_prompt = {"diffusion_input": True}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_output}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    processor_calls = []

    def mock_processor(stage_list, engine_input_source, original_prompts, requires_mm):
        processor_calls.append(
            {
                "stage_list": stage_list,
                "engine_input_source": engine_input_source,
                "original_prompts": original_prompts,
            }
        )
        return [processed_prompt]

    cfg = _make_stage_config(
        stage_type="llm",
        final_output=False,
        custom_process_input_func=None,
        engine_input_source=[0],
        requires_multimodal_data=False,
    )
    worker = OmniStageWorker(
        engine=engine,
        stage_config=cfg,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
    )
    worker._processor = mock_processor

    request = {
        "request_id": "req-proc",
        "original_prompt": {"prompt": "hi", "height": 480},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert len(processor_calls) == 1
    assert processor_calls[0]["stage_list"][0].engine_outputs == [fetched_output]
    assert processor_calls[0]["original_prompts"] == [{"prompt": "hi", "height": 480}]
    assert engine.received_prompt == processed_prompt
    assert chunks[0]["stage_connector_refs"]["1"] == {"name": "ref1"}


@pytest.mark.asyncio
async def test_stage_connector_refs_processor_token_prompt_builds_engine_core_request():
    """Stage N>0 token processor outputs follow vLLM-Omni orchestrator routing."""
    engine = _MockEngine()
    engine.engine = MagicMock()
    fetched_output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7])])
    processed_prompt = {
        "prompt_token_ids": [11, 22, 33],
        "additional_information": {"speaker": ["ethan"]},
    }

    in_connector = MagicMock()
    in_connector.get.return_value = fetched_output

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    cfg = _make_stage_config(
        default_sampling_params={"temperature": 0.9, "max_tokens": 100},
        engine_input_source=[0],
    )
    worker = OmniStageWorker(
        engine=engine,
        stage_config=cfg,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
    )
    worker._processor = MagicMock(return_value=[processed_prompt])

    request = {
        "request_id": "req-token-proc",
        "original_prompt": {"prompt": "hi"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    _ = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert hasattr(engine.received_prompt, "prompt_token_ids")
    assert engine.received_prompt.prompt_token_ids == [11, 22, 33]
    engine.engine.output_processors[0].add_request.assert_called_once()


@pytest.mark.asyncio
async def test_engine_error_yields_error_chunk():
    """Engine raises → yields {error: ..., finished: True}, no crash."""
    worker = _make_worker(engine=_ErrorEngine())
    request = {"engine_inputs": {"prompt": "hello"}}

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert any("error" in c for c in chunks)
    assert any(c.get("finished") for c in chunks)


@pytest.mark.asyncio
async def test_prepare_request_returns_router_prewarm_fields():
    engine = _TokenizerEngine()
    worker = _make_worker(engine=engine)

    with patch(
        "dynamo.vllm.omni.stage_worker.parse_omni_request",
        AsyncMock(
            return_value={
                "engine_inputs": "rendered prompt",
                "original_prompt": {"prompt": "rendered prompt"},
                "sampling_params_list": {
                    "__stage_overrides__": {"0": {"max_tokens": 4}}
                },
            }
        ),
    ):
        chunks = [
            chunk
            async for chunk in worker.generate(
                {
                    "request_id": "req-prepare",
                    "prompt": "hello",
                    _ASYNC_PREPARE_KEY: True,
                },
                _MockContext(),
            )
        ]

    assert chunks == [
        {
            "original_prompt": {"prompt": "rendered prompt"},
            "sampling_params_list": {"__stage_overrides__": {"0": {"max_tokens": 4}}},
            "prompt_token_ids": [9, 8, 7],
            "finished": True,
        }
    ]


@pytest.mark.asyncio
async def test_prepare_request_uses_prepare_only_media_uuids():
    engine = _TokenizerEngine()
    worker = _make_worker(engine=engine)
    request = {
        "request_id": "req-prepare",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/audio.ogg"},
                        "uuid": "user-audio-uuid",
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        _ASYNC_PREPARE_KEY: True,
    }

    parse_mock = AsyncMock(
        return_value={
            "engine_inputs": "rendered prompt",
            "original_prompt": {"prompt": "rendered prompt"},
            "sampling_params_list": None,
        }
    )
    with patch("dynamo.vllm.omni.stage_worker.parse_omni_request", parse_mock):
        chunks = [
            chunk async for chunk in worker.generate(dict(request), _MockContext())
        ]

    parsed_request = parse_mock.call_args.args[0]
    parts = parsed_request["messages"][0]["content"]
    assert parts[0]["uuid"] == "__dynamo_prepare_req-prepare_0_0"
    assert parts[1]["uuid"] == "user-audio-uuid"
    assert "uuid" not in parts[2]
    assert _ASYNC_PREPARE_KEY not in parsed_request
    assert "uuid" not in request["messages"][0]["content"][0]
    assert chunks[-1]["finished"] is True


@pytest.mark.asyncio
async def test_async_chunk_intermediate_stage_skips_connector_put_and_shm():
    engine = _MockEngine()
    engine.engine = MagicMock()
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})
    worker = _make_worker(
        engine=engine,
        connectors={("1", "2"): out_connector},
        stage_id=1,
        stage_config=_make_stage_config(
            engine_args=SimpleNamespace(async_chunk=True),
            default_sampling_params={"temperature": 0.9, "max_tokens": 100},
        ),
    )

    with patch(
        "dynamo.vllm.omni.stage_worker.shm_write_bytes",
    ) as shm_write:
        chunks = [
            chunk
            async for chunk in worker.generate(
                {
                    "request_id": "req-prewarm",
                    "prompt_token_ids": [0, 0, 0],
                    _ASYNC_PREWARM_KEY: True,
                },
                _MockContext(),
            )
        ]

    assert engine.received_prompt.prompt_token_ids == [0, 0, 0]
    engine.engine.output_processors[0].add_request.assert_called_once()
    out_connector.put.assert_not_called()
    shm_write.assert_not_called()
    assert len(chunks) == 2
    assert chunks[0] == {_ASYNC_PREWARM_READY_KEY: True}
    assert chunks[1] == {"finished": True}


@pytest.mark.asyncio
async def test_async_chunk_router_consumed_stage_uses_stage_specific_shm_name():
    engine = _MockEngine()
    engine.engine = MagicMock()
    worker = _make_worker(
        engine=engine,
        connectors={},
        stage_id=2,
        stage_config=_make_stage_config(
            engine_args=SimpleNamespace(async_chunk=True),
            default_sampling_params={"temperature": 0.9, "max_tokens": 100},
        ),
    )

    with patch(
        "dynamo.vllm.omni.stage_worker.shm_write_bytes",
        return_value={"name": "req-final-stage-2"},
    ) as shm_write:
        chunks = [
            chunk
            async for chunk in worker.generate(
                {
                    "request_id": "req-final",
                    "prompt_token_ids": [0, 0, 0],
                    _ASYNC_PREWARM_KEY: True,
                },
                _MockContext(),
            )
        ]

    shm_write.assert_called_once()
    assert shm_write.call_args.kwargs["name"] == "req-final-stage-2"
    assert chunks[0] == {_ASYNC_PREWARM_READY_KEY: True}
    assert chunks[1]["shm_meta"]


@pytest.mark.asyncio
async def test_connector_put_failure_yields_error():
    """connector.put() returning ok=False → yields error, stops."""
    mock_connector = MagicMock()
    mock_connector.get.return_value = {"engine_inputs": {"x": 1}}
    mock_connector.put.return_value = (False, 0, {})

    worker = _make_worker(
        connectors={("1", "2"): mock_connector},
        stage_id=1,
    )
    request = {
        "request_id": "req-fail",
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }
    with patch.object(
        worker, "_fetch_stage_inputs", return_value=[_Proxy(engine_outputs=[{"x": 1}])]
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert chunks == [{"error": "connector.put() failed", "finished": True}]


# ── _fetch_stage_inputs method unit tests ──────────────────


def _make_worker_at_stage(stage_id, connectors, engine_input_source=None):
    cfg = _make_stage_config(engine_input_source=engine_input_source or [stage_id - 1])
    return OmniStageWorker(
        engine=_MockEngine(),
        stage_config=cfg,
        connectors=connectors,
        stage_id=stage_id,
    )


def test_fetch_stage_inputs_calls_correct_connector():
    meta0 = {"name": "ref0"}
    connector = MagicMock()
    connector.get.return_value = {"engine_inputs": {"tok": [1, 2]}}

    worker = _make_worker_at_stage(
        1, connectors={("0", "1"): connector}, engine_input_source=[0]
    )
    result = worker._fetch_stage_inputs({0: meta0}, "r1")

    connector.get.assert_called_once_with("0", "1", "r1", metadata=meta0)
    assert result is not None
    assert result[0].engine_outputs == [{"tok": [1, 2]}]


def test_fetch_stage_inputs_returns_sparse_stage_indexed_list():
    meta1 = {"name": "ref1"}
    connector = MagicMock()
    connector.get.return_value = {"engine_inputs": {"tok": [3, 4]}}

    worker = _make_worker_at_stage(
        2, connectors={("1", "2"): connector}, engine_input_source=[1]
    )
    result = worker._fetch_stage_inputs({1: meta1}, "r1")

    connector.get.assert_called_once_with("1", "2", "r1", metadata=meta1)
    assert len(result) == 2
    assert result[0].engine_outputs is None
    assert result[1].engine_outputs == [{"tok": [3, 4]}]


def test_fetch_stage_inputs_adds_cumulative_token_ids_for_vllm_020_outputs():
    output = SimpleNamespace(token_ids=[7, 8, 9])
    request_output = SimpleNamespace(outputs=[output])
    connector = MagicMock()
    connector.get.return_value = request_output

    worker = _make_worker_at_stage(
        1, connectors={("0", "1"): connector}, engine_input_source=[0]
    )
    result = worker._fetch_stage_inputs({0: {"name": "ref0"}}, "r1")

    assert result[0].engine_outputs[0].outputs[0].cumulative_token_ids == [7, 8, 9]


def test_ensure_cumulative_token_ids_copies_token_ids_only_when_missing():
    output = SimpleNamespace(token_ids=(1, 2, 3))
    existing = SimpleNamespace(token_ids=[4, 5], cumulative_token_ids=[4])
    result = SimpleNamespace(outputs=[output, existing])

    _ensure_cumulative_token_ids(result)

    assert output.cumulative_token_ids == [1, 2, 3]
    assert existing.cumulative_token_ids == [4]


def test_prepare_connector_payload_uses_inner_request_output_and_serializable_tokens():
    completion = SimpleNamespace(token_ids=[3], cumulative_token_ids=[1, 2, 3])
    request_output = SimpleNamespace(outputs=[completion])
    omni_output = SimpleNamespace(request_output=request_output, outputs=[completion])

    payload = _prepare_connector_payload(omni_output)

    assert payload is request_output
    assert completion.token_ids == [1, 2, 3]


def test_fetch_stage_inputs_raises_on_missing_connector():
    worker = _make_worker_at_stage(1, connectors={}, engine_input_source=[0])
    with pytest.raises(RuntimeError, match="no connector for edge"):
        worker._fetch_stage_inputs({0: {"name": "ref0"}}, "r1")


def test_fetch_stage_inputs_raises_on_missing_ref():
    worker = _make_worker_at_stage(
        1, connectors={("0", "1"): MagicMock()}, engine_input_source=[0]
    )
    with pytest.raises(RuntimeError, match="no connector ref"):
        worker._fetch_stage_inputs({}, "r1")  # ref for stage 0 missing


def test_build_sampling_params_user_overrides_yaml_defaults():
    """User overrides applied on top of YAML defaults via setattr; unspecified keys preserved."""
    stage_config = SimpleNamespace(
        stage_type="diffusion",
        default_sampling_params={
            "num_inference_steps": 20,
            "guidance_scale": 5.0,
            "height": 480,
            "width": 832,
        },
    )
    result = _build_sampling_params(
        stage_config,
        {"num_inference_steps": 50},
    )
    assert result is not None
    sp = result[0]
    assert sp.num_inference_steps == 50  # user override wins
    assert sp.guidance_scale == 5.0  # YAML default preserved


def test_build_sampling_params_applies_stage_scoped_overrides_only_to_matching_stage():
    stage0 = SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        default_sampling_params={"temperature": 0.9, "max_tokens": 100},
    )
    stage1 = SimpleNamespace(
        stage_id=1,
        stage_type="llm",
        default_sampling_params={"temperature": 0.9, "max_tokens": 100},
    )
    overrides = {"__stage_overrides__": {"0": {"max_tokens": 16}}}

    stage0_params = _build_sampling_params(stage0, overrides)[0]
    stage1_params = _build_sampling_params(stage1, overrides)[0]

    assert stage0_params.max_tokens == 16
    assert stage1_params.max_tokens == 100


def test_build_sampling_params_no_defaults_returns_none():
    """No default_sampling_params on stage_config -> returns None."""
    stage_config = SimpleNamespace(stage_type="llm")
    assert _build_sampling_params(stage_config, None) is None
    assert _build_sampling_params(stage_config, {}) is None


def test_stage_config_to_dict_preserves_async_chunk_stage_id_and_connectors():
    cfg = _make_stage_config(
        stage_id=1,
        engine_input_source=[0],
        custom_process_input_func="pkg.func",
        input_connectors={"from_stage_0": "connector"},
        engine_args=SimpleNamespace(async_chunk=True, model_stage="talker"),
    )

    result = _stage_config_to_dict(cfg, "llm", preserve_stage_id=True)

    assert result["stage_id"] == 1
    assert result["engine_input_source"] == [0]
    assert result["custom_process_input_func"] == "pkg.func"
    assert result["input_connectors"] == {"from_stage_0": "connector"}
    assert result["engine_args"]["async_chunk"] is True


def test_stage_config_to_dict_handles_omegaconf_lists():
    from omegaconf import OmegaConf

    cfg = _make_stage_config(
        stage_id=1,
        engine_input_source=OmegaConf.create([0]),
        engine_args=OmegaConf.create({"async_chunk": True, "model_stage": "talker"}),
    )

    result = _stage_config_to_dict(cfg, "llm", preserve_stage_id=True)

    assert result["engine_input_source"] == [0]
    assert result["engine_args"]["model_stage"] == "talker"


def test_stage_config_to_dict_handles_empty_omegaconf_lists():
    from omegaconf import OmegaConf

    cfg = _make_stage_config(
        stage_id=0,
        engine_input_source=OmegaConf.create([]),
        engine_args=OmegaConf.create({"async_chunk": True, "model_stage": "thinker"}),
    )

    result = _stage_config_to_dict(cfg, "llm", preserve_stage_id=True)

    assert result["engine_input_source"] == []


@pytest.mark.asyncio
async def test_image_request_with_default_sampling_params():
    """Image stage with default_sampling_params builds typed params from YAML defaults + overrides."""
    engine = _MockEngine()
    worker = OmniStageWorker(
        engine=engine,
        stage_config=_make_stage_config(
            stage_type="diffusion",
            final_output=True,
            default_sampling_params={
                "num_inference_steps": 20,
                "guidance_scale": 1.5,
                "height": 1024,
                "width": 1024,
            },
        ),
        connectors={},
        stage_id=0,
        output_modalities=["image"],
    )
    request = {
        "request_id": "img-req-1",
        "prompt": "a red apple",
        "size": "1024x1024",
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert not any("error" in c for c in chunks)
    assert engine.received_sampling_params_list is not None


@pytest.mark.asyncio
async def test_sampling_params_propagate_in_stage_output():
    """Non-final stage must include sampling_params_list in its output for downstream stages."""
    engine = _MockEngine()
    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": {"latents": [1, 2]}}
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    # Stage 1: non-final, receives stage_connector_refs from stage 0
    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
        stage_config=_make_stage_config(final_output=False),
    )
    request = {
        "request_id": "req-sp",
        "original_prompt": {"prompt": "hi"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
        "sampling_params_list": {
            "num_inference_steps": 42,
            "height": 480,
            "width": 832,
        },
    }

    with patch(
        "dynamo.vllm.omni.stage_worker._build_sampling_params", return_value=None
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert len(chunks) == 1
    assert chunks[0].get("sampling_params_list") == {
        "num_inference_steps": 42,
        "height": 480,
        "width": 832,
    }
