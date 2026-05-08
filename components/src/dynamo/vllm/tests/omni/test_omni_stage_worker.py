# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Unit tests for OmniStageWorker.

Uses mock StageEngine instances matching AsyncOmni.generate() signature.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("vllm_omni", reason="vLLM-Omni dependencies not available")

from dynamo.vllm.omni.stage_worker import (
    OmniStageWorker,
    _prepare_connector_payload,
    _Proxy,
    _stage_config_to_dict,
)
from dynamo.vllm.omni.utils import _build_sampling_params

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
        self.received_output_modalities = None
        self._output = output or {"output": "mock", "finished": True}

    def generate(
        self,
        prompt,
        request_id="",
        *,
        sampling_params_list=None,
        output_modalities=None,
    ):
        self.received_prompt = prompt
        self.received_request_id = request_id
        self.received_sampling_params_list = sampling_params_list
        self.received_output_modalities = output_modalities

        async def _gen():
            yield self._output

        return _gen()

    async def get_tokenizer(self):
        return None


class _ErrorEngine:
    def generate(
        self,
        prompt,
        request_id="",
        *,
        sampling_params_list=None,
        output_modalities=None,  # noqa: ARG002
    ):
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
    request = {
        "request_id": "req-ecr",
        "original_prompt": {"prompt": "hello"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    _ = [chunk async for chunk in worker.generate(request, _MockContext())]

    # The engine should receive an OmniEngineCoreRequest (not the raw dict)
    assert hasattr(engine.received_prompt, "prompt_token_ids")
    assert engine.received_prompt.prompt_token_ids == [100, 200, 300]
    assert engine.received_prompt.external_req_id == "req-ecr"


@pytest.mark.asyncio
async def test_stage_connector_refs_with_processor():
    """Stage N>0 with processor: v0.20 processors receive direct source outputs."""
    engine = _MockEngine()
    fetched_output = {"latents": [0.1, 0.2]}
    processed_prompt = {"diffusion_input": True}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_output}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    processor_calls = []

    def mock_processor(source_outputs, original_prompt, requires_mm):
        processor_calls.append(
            {
                "source_outputs": source_outputs,
                "original_prompt": original_prompt,
                "requires_mm": requires_mm,
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
    assert processor_calls[0]["source_outputs"] == [fetched_output]
    assert processor_calls[0]["original_prompt"] == {"prompt": "hi", "height": 480}
    assert processor_calls[0]["requires_mm"] is False
    assert engine.received_prompt == processed_prompt
    assert chunks[0]["stage_connector_refs"]["1"] == {"name": "ref1"}


@pytest.mark.asyncio
async def test_stage_connector_refs_with_stage_list_processor():
    """Stage-list processors still receive proxies and engine_input_source."""
    engine = _MockEngine()
    fetched_output = {"latents": [0.1, 0.2]}
    processed_prompt = {"stage_list_input": True}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_output}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    processor_calls = []

    def mock_processor(
        stage_list, engine_input_source, prompt, requires_multimodal_data
    ):
        processor_calls.append(
            {
                "stage_list": stage_list,
                "engine_input_source": engine_input_source,
                "prompt": prompt,
                "requires_multimodal_data": requires_multimodal_data,
            }
        )
        return [processed_prompt]

    cfg = _make_stage_config(
        custom_process_input_func=None,
        engine_input_source=[0],
        requires_multimodal_data=True,
    )
    worker = OmniStageWorker(
        engine=engine,
        stage_config=cfg,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
    )
    worker._processor = mock_processor

    request = {
        "request_id": "req-stage-list-proc",
        "original_prompt": {"prompt": "hi"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert len(processor_calls) == 1
    assert processor_calls[0]["stage_list"][0].engine_outputs == [fetched_output]
    assert processor_calls[0]["engine_input_source"] == [0]
    assert processor_calls[0]["prompt"] == [{"prompt": "hi"}]
    assert processor_calls[0]["requires_multimodal_data"] is True
    assert engine.received_prompt == processed_prompt
    assert chunks[0]["stage_connector_refs"]["1"] == {"name": "ref1"}


@pytest.mark.asyncio
async def test_engine_error_yields_error_chunk():
    """Engine raises → yields {error: ..., finished: True}, no crash."""
    worker = _make_worker(engine=_ErrorEngine())
    request = {"engine_inputs": {"prompt": "hello"}}

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert any("error" in c for c in chunks)
    assert any(c.get("finished") for c in chunks)


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


@pytest.mark.asyncio
async def test_requested_final_stage_writes_shm_instead_of_connector():
    """A requested final stage returns SHM output even when topology has a next edge."""
    out_connector = MagicMock()
    worker = _make_worker(
        engine=_MockEngine(output={"result": "done"}),
        connectors={("0", "1"): out_connector},
        stage_id=0,
    )
    request = {"request_id": "req-final", "prompt": "hello", "final_stage_id": 0}

    with (
        patch(
            "dynamo.vllm.omni.stage_worker.parse_omni_request",
            new_callable=AsyncMock,
        ) as parse_request,
        patch(
            "dynamo.vllm.omni.stage_worker.serialize_obj",
            return_value=b"serialized",
        ) as serialize_obj,
        patch(
            "dynamo.vllm.omni.stage_worker.shm_write_bytes",
            return_value={"name": "req-final", "size": 10},
        ) as shm_write_bytes,
    ):
        parse_request.return_value = {
            "engine_inputs": "hello",
            "original_prompt": {"prompt": "hello"},
            "sampling_params_list": None,
        }
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    out_connector.put.assert_not_called()
    serialize_obj.assert_called_once_with({"result": "done"})
    shm_write_bytes.assert_called_once_with(b"serialized", name="req-final")
    assert chunks == [{"shm_meta": {"name": "req-final", "size": 10}, "finished": True}]


@pytest.mark.asyncio
async def test_final_stage_extracts_text_output_for_chat_audio():
    """Final stages carry generated text back to the router for chat audio responses."""
    result = SimpleNamespace(outputs=[SimpleNamespace(text="spoken text")])
    worker = _make_worker(engine=_MockEngine(output=result))
    request = {"request_id": "req-final-text", "prompt": "hello", "final_stage_id": 0}

    with (
        patch(
            "dynamo.vllm.omni.stage_worker.parse_omni_request",
            new_callable=AsyncMock,
        ) as parse_request,
        patch(
            "dynamo.vllm.omni.stage_worker.serialize_obj",
            return_value=b"serialized",
        ),
        patch(
            "dynamo.vllm.omni.stage_worker.shm_write_bytes",
            return_value={"name": "req-final-text", "size": 10},
        ),
    ):
        parse_request.return_value = {
            "engine_inputs": "hello",
            "original_prompt": {"prompt": "hello"},
            "sampling_params_list": None,
        }
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert chunks == [
        {
            "shm_meta": {"name": "req-final-text", "size": 10},
            "stage_text_output": "spoken text",
            "finished": True,
        }
    ]


@pytest.mark.asyncio
async def test_requested_modalities_and_pipeline_final_stage_reach_engine():
    """Stage 0 keeps requested modalities and tags the pipeline final stage."""
    engine = _MockEngine(output={"result": "done"})
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref0"})
    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): out_connector},
        stage_id=0,
    )
    request = {
        "request_id": "req-audio",
        "messages": [{"role": "user", "content": "say hello"}],
        "modalities": ["text", "audio"],
        "audio": {"voice": "Cherry", "format": "wav"},
        "final_stage_id": 2,
    }

    with patch(
        "dynamo.vllm.omni.stage_worker.parse_omni_request",
        new_callable=AsyncMock,
    ) as parse_request:
        parse_request.return_value = {
            "engine_inputs": {
                "prompt": "say hello",
                "modalities": ["text", "audio"],
                "additional_information": {"speaker": ["cherry"]},
            },
            "original_prompt": {
                "prompt": "say hello",
                "modalities": ["text", "audio"],
                "additional_information": {"speaker": ["cherry"]},
            },
            "sampling_params_list": None,
        }
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert engine.received_output_modalities == ["text", "audio"]
    assert engine.received_prompt["additional_information"] == {
        "speaker": ["cherry"],
        "_dynamo_pipeline_final_stage_id": 2,
    }
    assert chunks[0]["stage_connector_refs"]["0"] == {"name": "ref0"}


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


def test_fetch_stage_inputs_restores_dynamic_completion_attrs():
    meta0 = {"name": "ref0"}
    fetched_output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[3])])
    multimodal_output = {"hidden_states": {"layers": {0: [0.1], 24: [0.2]}}}
    connector = MagicMock()
    connector.get.return_value = {
        "engine_inputs": fetched_output,
        "_dynamo_completion_output_attrs": [
            {
                "cumulative_token_ids": [1, 2, 3],
                "multimodal_output": multimodal_output,
            },
        ],
    }

    worker = _make_worker_at_stage(
        1, connectors={("0", "1"): connector}, engine_input_source=[0]
    )
    result = worker._fetch_stage_inputs({0: meta0}, "r1")

    completion = result[0].engine_outputs[0].outputs[0]
    assert completion.cumulative_token_ids == [1, 2, 3]
    assert completion.multimodal_output == multimodal_output


def test_prepare_connector_payload_preserves_dynamic_completion_attrs():
    multimodal_output = {"hidden_states": {"layers": {0: [0.1], 24: [0.2]}}}
    output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[12],
                cumulative_token_ids=(10, 11, 12),
                multimodal_output=multimodal_output,
            ),
        ]
    )

    payload = _prepare_connector_payload(output)

    assert payload["engine_inputs"] is output
    assert payload["_dynamo_completion_output_attrs"] == [
        {
            "cumulative_token_ids": [10, 11, 12],
            "multimodal_output": multimodal_output,
        },
    ]


def test_prepare_connector_payload_promotes_request_multimodal_output():
    multimodal_output = {"hidden_states": {"layers": {0: [0.1], 24: [0.2]}}}
    completion = SimpleNamespace(
        token_ids=[12],
        cumulative_token_ids=(10, 11, 12),
    )
    output = SimpleNamespace(
        request_output=SimpleNamespace(outputs=[completion]),
        multimodal_output=multimodal_output,
    )

    payload = _prepare_connector_payload(output)

    assert completion.multimodal_output == multimodal_output
    assert payload["_dynamo_completion_output_attrs"] == [
        {
            "cumulative_token_ids": [10, 11, 12],
            "multimodal_output": multimodal_output,
        },
    ]


def test_stage_config_to_dict_preserves_native_async_chunk():
    stage_config = SimpleNamespace(
        stage_id=1,
        engine_args=SimpleNamespace(
            async_chunk=True,
            custom_process_next_stage_input_func="module.next_stage",
            model_stage="thinker",
        ),
        final_output_type="text",
        engine_input_source=[0],
    )
    transfer_config = SimpleNamespace(
        connectors={
            ("0", "1"): SimpleNamespace(name="SharedMemoryConnector", extra={}),
            ("1", "2"): SimpleNamespace(name="SharedMemoryConnector", extra={}),
        }
    )

    result = _stage_config_to_dict(
        stage_config,
        "llm",
        native_async=True,
        transfer_config=transfer_config,
    )

    assert result["stage_id"] == 1
    assert result["engine_args"]["async_chunk"] is True
    assert result["engine_args"]["model_stage"] == "thinker"
    assert (
        result["engine_args"]["custom_process_next_stage_input_func"]
        == "module.next_stage"
    )
    assert result["engine_input_source"] == [0]
    assert result["input_connectors"]["from_stage_0"]["name"] == "SharedMemoryConnector"
    assert result["output_connectors"]["to_stage_2"]["name"] == "SharedMemoryConnector"


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


def test_build_sampling_params_no_defaults_returns_none():
    """No default_sampling_params on stage_config -> returns None."""
    stage_config = SimpleNamespace(stage_type="llm")
    assert _build_sampling_params(stage_config, None) is None
    assert _build_sampling_params(stage_config, {}) is None


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
