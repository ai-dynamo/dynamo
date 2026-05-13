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
    from dynamo.vllm.omni.stage_worker import (
        OmniStageWorker,
        _accepts_source_outputs_processor,
        _ensure_stage_connectors,
        _normalize_single_stage_runtime_devices,
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
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_worker(
    engine=None,
    stage_config=None,
    connectors=None,
    stage_id=0,
    **kwargs,
):
    return OmniStageWorker(
        engine=engine or _MockEngine(),
        stage_config=stage_config or _make_stage_config(),
        connectors=connectors or {},
        stage_id=stage_id,
        **kwargs,
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
async def test_stage_connector_refs_processor_token_prompt_builds_engine_core_request():
    """Processor OmniTokensPrompt outputs are wrapped like native inter-stage inputs."""
    engine = _MockEngine()
    engine.engine = MagicMock()
    fetched_output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1])])
    processed_prompt = {"prompt_token_ids": [10, 20, 30]}

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": fetched_output}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref1"})

    def mock_processor(source_outputs, original_prompt, requires_mm):
        return [processed_prompt]

    worker = OmniStageWorker(
        engine=engine,
        stage_config=_make_stage_config(
            custom_process_input_func=None,
            engine_input_source=[0],
            default_sampling_params={"temperature": 0.0, "max_tokens": 4},
        ),
        connectors={("0", "2"): in_connector, ("2", "3"): out_connector},
        stage_id=2,
    )
    worker._processor = mock_processor

    request = {
        "request_id": "req-token-prompt",
        "original_prompt": {"prompt": "hi"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }

    chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert hasattr(engine.received_prompt, "prompt_token_ids")
    assert engine.received_prompt.prompt_token_ids == [10, 20, 30]
    assert engine.received_prompt.external_req_id == "req-token-prompt"
    engine.engine.output_processors[0].add_request.assert_called_once_with(
        request=engine.received_prompt,
        prompt=None,
        parent_req=None,
        request_index=0,
        queue=None,
    )
    assert chunks[0]["stage_connector_refs"]["2"] == {"name": "ref1"}


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


def test_source_output_processor_signature_accepts_qwen25_names():
    assert _accepts_source_outputs_processor(
        ["source_outputs", "prompt", "requires_multimodal_data"]
    )
    assert _accepts_source_outputs_processor(
        ["source_outputs", "_prompt", "_requires_multimodal_data"]
    )


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
async def test_requested_final_stage_writes_shm_instead_of_downstream_connector():
    """A per-request final stage should return SHM even if YAML has a next edge."""
    engine = _MockEngine(output={"output": "text", "finished": True})

    in_connector = MagicMock()
    in_connector.get.return_value = {"engine_inputs": {"prompt": "ready"}}

    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref2"})

    worker = _make_worker(
        engine=engine,
        connectors={("0", "1"): in_connector, ("1", "2"): out_connector},
        stage_id=1,
    )
    request = {
        "request_id": "req-final-stage",
        "original_prompt": {"prompt": "hi"},
        "stage_connector_refs": {"0": {"name": "ref0"}},
        "final_stage_id": 1,
    }

    with (
        patch("dynamo.vllm.omni.stage_worker.serialize_obj", return_value=b"payload"),
        patch(
            "dynamo.vllm.omni.stage_worker.shm_write_bytes",
            return_value={"name": "final-shm"},
        ),
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    out_connector.put.assert_not_called()
    assert chunks == [
        {
            "shm_meta": {"name": "final-shm"},
            "finished": True,
            "final_stage_id": 1,
        }
    ]


@pytest.mark.asyncio
async def test_raw_stage0_ignores_client_final_stage_id():
    """Raw client input cannot override the server-computed final stage."""
    engine = _MockEngine(output={"output": "stage0", "finished": True})
    out_connector = MagicMock()
    out_connector.put.return_value = (True, 0, {"name": "ref0"})
    stage0 = _make_stage_config()
    stage1 = _make_stage_config()
    worker = _make_worker(
        engine=engine,
        stage_config=stage0,
        connectors={("0", "1"): out_connector},
        stage_id=0,
        output_modalities=["text", "audio"],
        pipeline_stage_configs=[stage0, stage1],
    )
    request = {
        "request_id": "router-req",
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["audio"],
        "final_stage_id": 0,
    }

    with (
        patch(
            "dynamo.vllm.omni.stage_worker.get_final_stage_id_for_e2e",
            return_value=1,
        ) as resolve_final_stage,
        patch("dynamo.vllm.omni.stage_worker.shm_write_bytes") as shm_write,
    ):
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    resolve_final_stage.assert_called_once()
    out_connector.put.assert_called_once()
    shm_write.assert_not_called()
    assert chunks == [
        {
            "original_prompt": {"prompt": "hello"},
            "stage_connector_refs": {"0": {"name": "ref0"}},
            "finished": True,
            "final_stage_id": 1,
        }
    ]


@pytest.mark.asyncio
async def test_raw_stage0_uses_vllm_omni_chat_preprocessor():
    """Stage 0 should not fall back to raw text when Omni can render chat."""
    engine = _MockEngine(output={"output": "stage0", "finished": True})
    worker = _make_worker(
        engine=engine,
        output_modalities=["text", "audio"],
    )

    class FakeOmniChatPreprocessor:
        async def _preprocess_chat(self, request, messages, **kwargs):
            assert request.messages == messages
            assert kwargs["renderer"] is renderer
            return ["conversation"], [{"prompt": "<rendered>hello"}]

    renderer = object()
    worker._omni_chat_preprocessor = FakeOmniChatPreprocessor()
    worker.engine.renderer = renderer
    request = {
        "request_id": "router-req",
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["audio"],
    }

    with patch("dynamo.vllm.omni.stage_worker.shm_write_bytes") as shm_write:
        chunks = [chunk async for chunk in worker.generate(request, _MockContext())]

    assert engine.received_prompt == {"prompt": "<rendered>hello"}
    shm_write.assert_called_once()
    assert chunks == [
        {"shm_meta": shm_write.return_value, "finished": True, "final_stage_id": 0}
    ]


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
