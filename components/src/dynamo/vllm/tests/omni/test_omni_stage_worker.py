# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniStageWorker.

No GPU, no vllm_omni — uses mock StageEngine matching AsyncOmni.generate() signature.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

try:
    from dynamo.vllm.omni import stage_worker
    from dynamo.vllm.omni.stage_worker import (
        _ASYNC_PREWARM_KEY,
        _ASYNC_PREWARM_READY_KEY,
        OmniStageWorker,
        _ensure_stage_connectors,
        _normalize_single_stage_runtime_devices,
        _prepare_connector_payload,
        _preserve_external_request_ids,
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
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
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


class _ErrorEngine:
    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        async def _gen():
            raise RuntimeError("engine exploded")
            yield  # make it an async generator

        return _gen()


class _MockContext:
    def id(self):
        return "test-req-id"


class _ChatTemplateTokenizer:
    chat_token_ids = [151644, 872, 100, 101, 151644, 77091]
    rendered_prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "hello"}]
        assert add_generation_prompt is True
        return list(self.chat_token_ids) if tokenize else self.rendered_prompt

    def encode(self, text, add_special_tokens=False):
        assert text == self.rendered_prompt
        return [999, 998]


class _TokenizerEngine(_MockEngine):
    def __init__(self, tokenizer):
        super().__init__()
        self._tokenizer = tokenizer

    async def get_tokenizer(self):
        return self._tokenizer


class _FakeRenderer:
    def __init__(self, engine_prompt):
        self.engine_prompt = engine_prompt
        self.messages_batch = None
        self.prompt_extras = None

    async def render_chat_async(
        self, messages_batch, chat_params, tok_params, *, prompt_extras=None
    ):
        self.messages_batch = messages_batch
        self.prompt_extras = prompt_extras
        return ([{"role": "user", "content": "hello"}],), (self.engine_prompt,)


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


def test_preserve_external_request_ids_disables_async_omni_suffixing():
    class _SuffixingEngine:
        def _get_unique_request_id(self, external_request_id):
            return f"{external_request_id}-deadbeef"

    engine = _SuffixingEngine()

    _preserve_external_request_ids(engine)

    assert engine._get_unique_request_id("req-1") == "req-1"
    assert engine._get_unique_request_id("")


@pytest.mark.asyncio
async def test_prepare_router_request_prefers_chat_template_token_ids():
    tokenizer = _ChatTemplateTokenizer()
    worker = OmniStageWorker(
        engine=_TokenizerEngine(tokenizer),
        stage_config=_make_stage_config(stage_id=0),
        connectors={},
        stage_id=0,
        output_modalities=["text"],
        pipeline_stage_configs=[
            _make_stage_config(stage_id=0, final_output_type="text"),
            _make_stage_config(stage_id=1, final_output_type="audio"),
        ],
    )

    prepared = await worker._prepare_router_request(
        {
            "request_id": "req-chat",
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "modalities": ["audio"],
        }
    )

    assert prepared["prompt_token_ids"] == tokenizer.chat_token_ids
    assert prepared["original_prompt"] == {"prompt": tokenizer.rendered_prompt}
    assert prepared["final_stage_id"] == 1


@pytest.mark.asyncio
async def test_prepare_router_request_prefers_native_renderer_prompt_token_ids():
    tokenizer = _ChatTemplateTokenizer()
    engine = _TokenizerEngine(tokenizer)
    renderer = _FakeRenderer({"prompt": tokenizer.rendered_prompt})
    engine.renderer = renderer
    engine.engine = SimpleNamespace(
        input_processor=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=1024)
        )
    )
    worker = OmniStageWorker(
        engine=engine,
        stage_config=_make_stage_config(stage_id=0),
        connectors={},
        stage_id=0,
        output_modalities=["text"],
        pipeline_stage_configs=[
            _make_stage_config(stage_id=0, final_output_type="text"),
            _make_stage_config(stage_id=1, final_output_type="audio"),
        ],
    )

    prepared = await worker._prepare_router_request(
        {
            "request_id": "req-chat",
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "modalities": ["audio"],
        }
    )

    assert prepared["prompt_token_ids"] == [999, 998]
    assert renderer.messages_batch == [[{"role": "user", "content": "hello"}]]
    assert renderer.prompt_extras == {}


@pytest.mark.asyncio
async def test_stage0_request_uses_internal_prompt_token_ids():
    tokenizer = _ChatTemplateTokenizer()
    engine = _TokenizerEngine(tokenizer)
    worker = OmniStageWorker(
        engine=engine,
        stage_config=_make_stage_config(stage_id=0),
        connectors={},
        stage_id=0,
        output_modalities=["text"],
    )

    _ = [
        chunk
        async for chunk in worker.generate(
            {
                "request_id": "req-chat",
                "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "messages": [{"role": "user", "content": "hello"}],
                "prompt_token_ids": [1, 2, 3],
            },
            _MockContext(),
        )
    ]

    assert engine.received_prompt == {
        "prompt": tokenizer.rendered_prompt,
        "prompt_token_ids": [1, 2, 3],
    }


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
    # AsyncOmni's native StagePool admission registers prebuilt EngineCoreRequests.
    engine.engine.output_processors[0].add_request.assert_not_called()


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


def test_process_stage_inputs_accepts_prompt_alias_processor():
    """vLLM-Omni processors may name the original prompt argument ."""
    fetched_output = {"latents": [0.1, 0.2]}
    processed_prompt = {"processed": True}
    processor_calls = []

    def mock_processor(source_outputs, prompt, requires_multimodal_data):
        processor_calls.append(
            {
                "source_outputs": source_outputs,
                "prompt": prompt,
                "requires_multimodal_data": requires_multimodal_data,
            }
        )
        return [processed_prompt]

    worker = OmniStageWorker(
        engine=_MockEngine(),
        stage_config=_make_stage_config(
            custom_process_input_func=None,
            engine_input_source=[0],
            requires_multimodal_data=True,
        ),
        connectors={},
        stage_id=1,
    )
    worker._processor = mock_processor

    prompt = worker._process_stage_inputs(
        [_Proxy(engine_outputs=[fetched_output])],
        {"prompt": "hi"},
    )

    assert prompt == [processed_prompt]
    assert processor_calls == [
        {
            "source_outputs": [fetched_output],
            "prompt": {"prompt": "hi"},
            "requires_multimodal_data": True,
        }
    ]


@pytest.mark.asyncio
async def test_async_prewarm_uses_native_placeholder_prompt_shape():
    """Async-chunk prewarm uses zero placeholders sized for the downstream stage."""
    engine = _MockEngine()
    engine.engine = MagicMock()
    worker = _make_worker(
        engine=engine,
        stage_id=1,
        stage_config=_make_stage_config(
            engine_args=SimpleNamespace(async_chunk=True),
            default_sampling_params={"temperature": 0.0, "max_tokens": 1},
        ),
    )
    captured = {}

    def fake_build_engine_core_request_from_tokens(**kwargs):
        prompt = kwargs["prompt"]
        captured["prompt"] = prompt
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            external_req_id=None,
            prompt_token_ids=list(prompt["prompt_token_ids"]),
            additional_information=None,
        )

    with (
        patch.object(stage_worker, "compute_talker_prompt_ids_length", return_value=4),
        patch.object(
            stage_worker,
            "build_engine_core_request_from_tokens",
            side_effect=fake_build_engine_core_request_from_tokens,
        ),
        patch.object(
            stage_worker,
            "_apply_omni_final_stage_metadata",
            side_effect=lambda prompt, _final_stage_id: prompt,
        ),
    ):
        chunks = [
            chunk
            async for chunk in worker.generate(
                {
                    "request_id": "req-talker",
                    "original_prompt": {
                        "prompt": "hello",
                        "multi_modal_data": {"audio": "raw"},
                        "mm_processor_kwargs": {"fps": 16},
                    },
                    "prompt_token_ids": [10, 11, 12],
                    "final_stage_id": 2,
                    "sampling_params_list": {"max_tokens": 1},
                    _ASYNC_PREWARM_KEY: True,
                },
                _MockContext(),
            )
        ]

    assert captured["prompt"]["prompt_token_ids"] == [0, 0, 0, 0]
    assert captured["prompt"]["prompt"] == "hello"
    assert captured["prompt"]["multi_modal_data"] is None
    assert captured["prompt"]["mm_processor_kwargs"] is None
    assert engine.received_prompt.prompt_token_ids == [0, 0, 0, 0]
    assert any(chunk.get(_ASYNC_PREWARM_READY_KEY) for chunk in chunks)


def test_async_prewarm_generation_worker_waits_for_chunk():
    worker = _make_worker(
        stage_id=2,
        stage_config=_make_stage_config(
            engine_args=SimpleNamespace(async_chunk=True, worker_type="generation"),
        ),
    )

    assert worker._async_prewarm_prompt_token_ids([10, 11, 12]) == []


def test_async_prewarm_ar_requires_prompt_token_ids():
    worker = _make_worker(
        stage_id=1,
        stage_config=_make_stage_config(
            engine_args=SimpleNamespace(async_chunk=True, worker_type="ar"),
        ),
    )

    with pytest.raises(RuntimeError, match="without prompt_token_ids"):
        worker._async_prewarm_prompt_token_ids([])


def test_stage_config_to_dict_can_preserve_async_stage_topology():
    result = _stage_config_to_dict(
        SimpleNamespace(
            stage_id=2,
            engine_args=SimpleNamespace(model_stage="code2wav", async_chunk=True),
            input_connectors={"from_stage_1": "connector_of_shared_memory"},
            output_connectors={"to_stage_3": "connector_of_shared_memory"},
            final_output_type="audio",
        ),
        "llm",
        preserve_stage_id=True,
    )

    assert result["stage_id"] == 2
    assert result["input_connectors"] == {"from_stage_1": "connector_of_shared_memory"}
    assert result["output_connectors"] == {"to_stage_3": "connector_of_shared_memory"}


def test_process_stage_inputs_ignores_var_kwargs_for_dispatch():
    """Source-output processors with **kwargs should not get a fourth positional arg."""
    fetched_output = {"latents": [0.1, 0.2]}
    processed_prompt = {"diffusion_input": True}
    processor_calls = []

    def mock_processor(source_outputs, original_prompt, requires_mm, **kwargs):
        processor_calls.append(
            {
                "source_outputs": source_outputs,
                "original_prompt": original_prompt,
                "requires_mm": requires_mm,
                "kwargs": kwargs,
            }
        )
        return [processed_prompt]

    worker = OmniStageWorker(
        engine=_MockEngine(),
        stage_config=_make_stage_config(
            custom_process_input_func=None,
            engine_input_source=[0],
            requires_multimodal_data=False,
        ),
        connectors={},
        stage_id=1,
    )
    worker._processor = mock_processor

    prompt = worker._process_stage_inputs(
        [_Proxy(engine_outputs=[fetched_output])],
        {"prompt": "hi"},
    )

    assert len(processor_calls) == 1
    assert processor_calls[0]["source_outputs"] == [fetched_output]
    assert processor_calls[0]["kwargs"] == {}
    assert prompt == [processed_prompt]


@pytest.mark.asyncio
async def test_stage_connector_refs_with_stage_list_processor():
    """Stage-list transition processors still receive proxies and sources."""
    engine = _MockEngine()
    fetched_output = {"latents": [0.1, 0.2]}
    processed_prompt = {"stage_list_input": True}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_output}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    processor_calls = []

    def mock_processor(stage_list, engine_input_source, prompt, requires_mm):
        processor_calls.append(
            {
                "stage_list": stage_list,
                "engine_input_source": engine_input_source,
                "prompt": prompt,
                "requires_mm": requires_mm,
            }
        )
        return [processed_prompt]

    cfg = _make_stage_config(
        stage_type="llm",
        final_output=False,
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
    assert processor_calls[0]["requires_mm"] is True
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


def test_fetch_stage_inputs_adds_cumulative_token_ids_when_missing():
    meta0 = {"name": "ref0"}
    fetched_output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2, 3])])
    connector = MagicMock()
    connector.get.return_value = {"engine_inputs": fetched_output}

    worker = _make_worker_at_stage(
        1, connectors={("0", "1"): connector}, engine_input_source=[0]
    )
    result = worker._fetch_stage_inputs({0: meta0}, "r1")

    completion = result[0].engine_outputs[0].outputs[0]
    assert completion.cumulative_token_ids == [1, 2, 3]


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
    multimodal_output = {"ids": {"prior_image": [1, 2, 3]}}
    completion = SimpleNamespace(token_ids=[12], cumulative_token_ids=(10, 11, 12))
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


def test_prepare_connector_payload_preserves_empty_attr_positions():
    output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[12])])

    payload = _prepare_connector_payload(output)

    assert payload["engine_inputs"] is output
    assert payload["_dynamo_completion_output_attrs"] == [{}]


def test_ensure_stage_connectors_adds_missing_shared_memory_edge(tmp_path):
    config_path = tmp_path / "glm_image.yaml"
    config_path.write_text(
        """
stages:
  - stage_id: 0
  - stage_id: 1
""".lstrip()
    )
    stage_configs = [
        SimpleNamespace(stage_id=0, engine_input_source=[]),
        SimpleNamespace(stage_id=1, engine_input_source=[0]),
    ]

    resolved_path = _ensure_stage_connectors(str(config_path), stage_configs)

    assert resolved_path != str(config_path)
    with open(resolved_path) as f:
        resolved = yaml.safe_load(f)
    assert "connector_of_shared_memory" in resolved["connectors"]
    stage_1 = next(stage for stage in resolved["stages"] if stage["stage_id"] == 1)
    assert stage_1["input_connectors"]["from_stage_0"] == ("connector_of_shared_memory")


def test_ensure_stage_connectors_rejects_non_mapping_connectors(tmp_path):
    config_path = tmp_path / "glm_image.yaml"
    config_path.write_text(
        """
connectors:
  - not-a-mapping
stages:
  - stage_id: 0
  - stage_id: 1
""".lstrip()
    )
    stage_configs = [
        SimpleNamespace(stage_id=0, engine_input_source=[]),
        SimpleNamespace(stage_id=1, engine_input_source=[0]),
    ]

    with pytest.raises(ValueError, match="connector_of_shared_memory"):
        _ensure_stage_connectors(str(config_path), stage_configs)


def test_stage_config_to_dict_includes_engine_input_source():
    result = _stage_config_to_dict(
        SimpleNamespace(
            engine_args=SimpleNamespace(model_stage="dit"),
            engine_input_source=[0],
            final_output_type="image",
        ),
        "diffusion",
    )

    assert result["engine_input_source"] == [0]


def test_stage_config_to_dict_preserves_runtime_devices():
    result = _stage_config_to_dict(
        SimpleNamespace(
            engine_args=SimpleNamespace(model_stage="dit"),
            runtime=SimpleNamespace(devices="1"),
            final_output_type="image",
        ),
        "diffusion",
    )

    assert result["runtime"]["devices"] == "1"


def test_single_stage_runtime_devices_normalized_when_visibility_is_narrowed(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    stage_arg = {
        "stage_id": 0,
        "stage_type": "diffusion",
        "runtime": {"devices": "1"},
    }

    _normalize_single_stage_runtime_devices(stage_arg)

    assert stage_arg["runtime"]["devices"] == "0"


def test_single_stage_runtime_devices_preserve_visible_subset(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    stage_arg = {
        "stage_id": 0,
        "stage_type": "diffusion",
        "runtime": {"devices": "1"},
    }

    _normalize_single_stage_runtime_devices(stage_arg)

    assert stage_arg["runtime"]["devices"] == "1"


def test_single_stage_runtime_devices_normalized_for_xpu_visibility(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "2")
    stage_arg = {
        "stage_id": 0,
        "stage_type": "diffusion",
        "runtime": {"devices": "2"},
    }

    _normalize_single_stage_runtime_devices(stage_arg)

    assert stage_arg["runtime"]["devices"] == "0"


def test_single_stage_runtime_devices_preserve_xpu_visible_subset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1")
    stage_arg = {
        "stage_id": 0,
        "stage_type": "diffusion",
        "runtime": {"devices": "1"},
    }

    _normalize_single_stage_runtime_devices(stage_arg)

    assert stage_arg["runtime"]["devices"] == "1"


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
