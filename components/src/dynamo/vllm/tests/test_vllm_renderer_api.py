#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests to verify vLLM renderer API compatibility.

These tests lock the vLLM API surface used by:
- components/src/dynamo/frontend/vllm_processor.py
- components/src/dynamo/frontend/prepost.py

If vLLM changes these APIs, these tests should fail early during upgrade.
"""

import importlib
import inspect

import pytest

# Import vllm first to ensure it's properly loaded before accessing submodules.
_vllm = importlib.import_module("vllm")
_chat_protocol = importlib.import_module(
    "vllm.entrypoints.openai.chat_completion.protocol"
)
_engine_protocol = importlib.import_module("vllm.entrypoints.openai.engine.protocol")
_inputs_data = importlib.import_module("vllm.inputs.data")
_reasoning = importlib.import_module("vllm.reasoning")
_sampling_params = importlib.import_module("vllm.sampling_params")
_tool_parsers = importlib.import_module("vllm.tool_parsers")
_engine = importlib.import_module("vllm.v1.engine")
_input_processor_mod = importlib.import_module("vllm.v1.engine.input_processor")
_output_processor_mod = importlib.import_module("vllm.v1.engine.output_processor")

ChatCompletionRequest = _chat_protocol.ChatCompletionRequest
DeltaMessage = _engine_protocol.DeltaMessage
DeltaToolCall = _engine_protocol.DeltaToolCall
TokensPrompt = _inputs_data.TokensPrompt
ReasoningParser = _reasoning.ReasoningParser
ReasoningParserManager = _reasoning.ReasoningParserManager
RequestOutputKind = _sampling_params.RequestOutputKind
SamplingParams = _sampling_params.SamplingParams
ToolParser = _tool_parsers.ToolParser
ToolParserManager = _tool_parsers.ToolParserManager
EngineCoreOutput = _engine.EngineCoreOutput
EngineCoreRequest = _engine.EngineCoreRequest
FinishReason = _engine.FinishReason
InputProcessor = _input_processor_mod.InputProcessor
OutputProcessor = _output_processor_mod.OutputProcessor
OutputProcessorOutput = _output_processor_mod.OutputProcessorOutput

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestVllmRendererApi:
    """Test vLLM APIs used by Dynamo renderer/prepost code."""

    def test_chat_completion_request_has_required_fields(self):
        """Verify ChatCompletionRequest has all fields read by prepost.py.

        preprocess_chat_request() accesses these fields after model_validate().
        New fields in vLLM are fine; missing fields break preprocessing.
        """
        expected_fields = {
            "messages",
            "model",
            "tools",
            "tool_choice",
            "reasoning_effort",
            "max_tokens",
            "max_completion_tokens",
            "logprobs",
            "top_logprobs",
            "chat_template",
            "chat_template_kwargs",
            "add_generation_prompt",
            "continue_final_message",
            "documents",
            "add_special_tokens",
            "cache_salt",
            "mm_processor_kwargs",
        }

        actual_fields = set(ChatCompletionRequest.model_fields)
        missing = sorted(expected_fields - actual_fields)
        assert not missing, (
            "ChatCompletionRequest fields changed!\n"
            f"Missing required fields: {missing}\n"
            "Update components/src/dynamo/frontend/prepost.py and "
            "components/src/dynamo/frontend/vllm_processor.py to match new vLLM API."
        )

    def test_chat_completion_request_model_validate_contract(self):
        """Verify ChatCompletionRequest supports pydantic v2 model_validate.

        prepost.py calls ChatCompletionRequest.model_validate(request) and
        then iterates model_fields to project sampling parameters.
        """
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "dummy-model",
            }
        )
        assert isinstance(request, ChatCompletionRequest)
        assert hasattr(type(request), "model_fields"), (
            "ChatCompletionRequest no longer exposes model_fields; "
            "preprocess code depends on this pydantic v2 API."
        )

    def test_delta_tool_call_fields(self):
        """Verify DeltaToolCall has fields used by _merge_tool_call in prepost.py.

        The post-processor reads id, type, index, and function from incoming
        DeltaToolCall objects. New fields in vLLM are acceptable; missing
        fields break the tool-call merge logic.
        """
        expected_fields = {
            "id",
            "type",
            "index",
            "function",
        }
        actual_fields = set(DeltaToolCall.model_fields)
        missing = sorted(expected_fields - actual_fields)
        assert not missing, (
            "DeltaToolCall fields changed!\n"
            f"Missing required fields: {missing}\n"
            "Update tool-call merge logic in components/src/dynamo/frontend/prepost.py"
        )

    def test_delta_message_fields(self):
        """Verify DeltaMessage has fields used by StreamingPostProcessor.

        process_output() reads content, reasoning, and tool_calls from
        DeltaMessage objects returned by reasoning/tool parsers. New fields
        in vLLM are acceptable; missing fields break post-processing.
        """
        expected_fields = {
            "role",
            "content",
            "reasoning",
            "reasoning_content",
            "tool_calls",
        }
        actual_fields = set(DeltaMessage.model_fields)
        missing = sorted(expected_fields - actual_fields)
        assert not missing, (
            "DeltaMessage fields changed!\n"
            f"Missing required fields: {missing}\n"
            "Update streaming post-processing logic in "
            "components/src/dynamo/frontend/prepost.py"
        )

    def test_sampling_params_annotations_and_output_kind(self):
        """Verify SamplingParams has the fields projected by vllm_processor.py.

        The processor intersects SamplingParams.__annotations__ with
        ChatCompletionRequest.model_fields to copy user-specified sampling
        fields. If annotations change, the intersection shifts silently.
        """
        required_sampling_fields = {
            "n",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "stop",
            "stop_token_ids",
            "ignore_eos",
            "max_tokens",
            "min_tokens",
            "logprobs",
            "prompt_logprobs",
            "skip_special_tokens",
            "output_kind",
            "bad_words",
            "structured_outputs",
            "extra_args",
        }

        # SamplingParams is a msgspec.Struct; all field annotations are defined
        # directly on the class (no TypedDict/dataclass inheritance to worry
        # about), so __annotations__ is the correct source here.
        annotations = set(getattr(SamplingParams, "__annotations__", {}))
        missing = sorted(required_sampling_fields - annotations)
        assert not missing, (
            "SamplingParams annotations changed!\n"
            f"Missing required fields: {missing}\n"
            "Update sampling field projection in "
            "components/src/dynamo/frontend/vllm_processor.py"
        )

        assert RequestOutputKind.DELTA.name == "DELTA"
        assert RequestOutputKind.DELTA.value == 1, (
            "RequestOutputKind.DELTA changed value; "
            "dynamo expects delta streaming behavior."
        )

    def test_tokens_prompt_supports_optional_keys_used_by_frontend(self):
        """Verify TokensPrompt TypedDict accepts the keys set by vllm_processor.py.

        The processor constructs TokensPrompt(prompt_token_ids=...) and then
        conditionally sets multi_modal_data, multi_modal_uuids, cache_salt,
        and mm_processor_kwargs.
        """
        expected_keys = {
            "prompt_token_ids",
            "multi_modal_data",
            "multi_modal_uuids",
            "cache_salt",
            "mm_processor_kwargs",
        }
        # TokensPrompt is a flat TypedDict with no inheritance, so
        # __annotations__ captures all keys.
        annotations = set(getattr(TokensPrompt, "__annotations__", {}))
        missing = sorted(expected_keys - annotations)
        assert not missing, (
            "TokensPrompt shape changed!\n"
            f"Missing keys: {missing}\n"
            "Update prompt construction in "
            "components/src/dynamo/frontend/vllm_processor.py"
        )

    def test_input_processor_method_signatures(self):
        """Verify InputProcessor.process_inputs and assign_request_id signatures.

        vllm_processor.py calls process_inputs(request_id, prompt, params)
        positionally and InputProcessor.assign_request_id(vllm_preproc) as
        a static/class method.
        """
        process_inputs_sig = inspect.signature(InputProcessor.process_inputs)
        process_inputs_params = list(process_inputs_sig.parameters)

        expected_prefix = ["self", "request_id", "prompt", "params"]
        assert process_inputs_params[:4] == expected_prefix, (
            "InputProcessor.process_inputs signature changed!\n"
            f"Expected prefix: {expected_prefix}\n"
            f"Actual: {process_inputs_params}"
        )

        for required in ["data_parallel_rank", "priority", "resumable"]:
            assert required in process_inputs_params, (
                "InputProcessor.process_inputs missing parameter "
                f"{required!r}; vllm_processor.py does not pass this today but "
                "its presence is part of the expected API surface."
            )

        # In vLLM 0.15.1 this is a @staticmethod, so the signature should be
        # exactly (request). If vLLM switches it to an instance method
        # (adding self), this assertion should fail to catch the API change.
        assign_request_id_sig = inspect.signature(InputProcessor.assign_request_id)
        assert list(assign_request_id_sig.parameters) == ["request"], (
            "InputProcessor.assign_request_id signature changed; "
            "update request-id assignment in "
            "components/src/dynamo/frontend/vllm_processor.py"
        )

    def test_input_processor_attributes(self):
        """Verify InputProcessor exposes renderer, generation_config_fields,
        and get_tokenizer used by EngineFactory and VllmProcessor.

        EngineFactory calls input_processor.get_tokenizer() and passes
        input_processor.renderer to preprocess_chat_request.
        VllmProcessor iterates input_processor.generation_config_fields.
        """
        assert hasattr(InputProcessor, "renderer"), (
            "InputProcessor no longer has 'renderer' attribute/property; "
            "update preprocess_chat_request call in "
            "components/src/dynamo/frontend/vllm_processor.py"
        )

        assert hasattr(InputProcessor, "get_tokenizer"), (
            "InputProcessor no longer has 'get_tokenizer' method; "
            "update EngineFactory in "
            "components/src/dynamo/frontend/vllm_processor.py"
        )
        assert callable(
            getattr(InputProcessor, "get_tokenizer")
        ), "InputProcessor.get_tokenizer is not callable"
        get_tok_sig = inspect.signature(InputProcessor.get_tokenizer)
        assert list(get_tok_sig.parameters) == ["self"], (
            "InputProcessor.get_tokenizer signature changed; "
            f"expected (self), got {list(get_tok_sig.parameters)}"
        )

    def test_output_processor_api_shape(self):
        """Verify OutputProcessor init, add_request, process_outputs, and
        abort_requests signatures match what vllm_processor.py calls.
        """
        init_sig = inspect.signature(OutputProcessor.__init__)
        init_params = list(init_sig.parameters)
        assert init_params[:3] == ["self", "tokenizer", "log_stats"], (
            "OutputProcessor.__init__ signature changed; "
            "update construction in components/src/dynamo/frontend/vllm_processor.py"
        )
        assert "stream_interval" in init_params

        add_request_sig = inspect.signature(OutputProcessor.add_request)
        assert list(add_request_sig.parameters) == [
            "self",
            "request",
            "prompt",
            "parent_req",
            "request_index",
            "queue",
        ], "OutputProcessor.add_request signature changed"

        process_outputs_sig = inspect.signature(OutputProcessor.process_outputs)
        process_outputs_params = list(process_outputs_sig.parameters)
        assert process_outputs_params[:2] == ["self", "engine_core_outputs"], (
            "OutputProcessor.process_outputs signature changed; "
            "update processing in components/src/dynamo/frontend/vllm_processor.py"
        )

        abort_requests_sig = inspect.signature(OutputProcessor.abort_requests)
        assert list(abort_requests_sig.parameters) == [
            "self",
            "request_ids",
            "internal",
        ], "OutputProcessor.abort_requests signature changed"

        output_fields = tuple(OutputProcessorOutput.__dataclass_fields__)
        assert output_fields == ("request_outputs", "reqs_to_abort"), (
            "OutputProcessorOutput fields changed!\n"
            f"Expected: ('request_outputs', 'reqs_to_abort')\n"
            f"Actual: {output_fields}"
        )

    def test_output_processor_request_states_attribute(self):
        """Verify OutputProcessor has a request_states dict attribute.

        vllm_processor.py checks
          vllm_preproc.request_id in self.output_processor.request_states
        in the finally block to decide whether to abort. If this attribute
        is renamed or removed, cleanup will silently break.
        """
        # request_states is an instance attribute set in __init__, so we
        # verify its presence in __init__ source rather than on the class.
        init_source = inspect.getsource(OutputProcessor.__init__)
        assert "request_states" in init_source, (
            "OutputProcessor.__init__ no longer initializes 'request_states'; "
            "update cleanup logic in components/src/dynamo/frontend/vllm_processor.py"
        )

    def test_engine_core_struct_contract(self):
        """Verify EngineCoreRequest and EngineCoreOutput field order.

        Both use msgspec array_like=True, so field ORDER determines wire
        position. vllm_processor.py constructs EngineCoreOutput by keyword
        and reads fields from EngineCoreRequest positionally.
        """
        expected_request_fields = (
            "request_id",
            "prompt_token_ids",
            "mm_features",
            "sampling_params",
            "pooling_params",
            "eos_token_id",
            "arrival_time",
            "lora_request",
            "cache_salt",
            "data_parallel_rank",
            "prompt_embeds",
            "client_index",
            "current_wave",
            "priority",
            "trace_headers",
            "resumable",
            "external_req_id",
        )
        actual_request_fields = EngineCoreRequest.__struct_fields__
        assert actual_request_fields == expected_request_fields, (
            "EngineCoreRequest fields changed!\n"
            f"Expected: {expected_request_fields}\n"
            f"Actual:   {actual_request_fields}\n"
            "Update request construction in components/src/dynamo/frontend/vllm_processor.py"
        )

        expected_output_fields = (
            "request_id",
            "new_token_ids",
            "new_logprobs",
            "new_prompt_logprobs_tensors",
            "pooling_output",
            "finish_reason",
            "stop_reason",
            "events",
            "kv_transfer_params",
            "trace_headers",
            "num_cached_tokens",
            "routed_experts",
            "num_nans_in_logits",
        )
        actual_output_fields = EngineCoreOutput.__struct_fields__
        assert actual_output_fields == expected_output_fields, (
            "EngineCoreOutput fields changed!\n"
            f"Expected: {expected_output_fields}\n"
            f"Actual:   {actual_output_fields}\n"
            "Update output mapping in components/src/dynamo/frontend/vllm_processor.py"
        )

        req_config = getattr(EngineCoreRequest, "__struct_config__", None)
        out_config = getattr(EngineCoreOutput, "__struct_config__", None)
        assert req_config is not None and req_config.array_like is True
        assert out_config is not None and out_config.array_like is True

    def test_engine_core_output_construction(self):
        """Verify EngineCoreOutput can be constructed with keyword args.

        vllm_processor.py constructs EngineCoreOutput(request_id=...,
        new_token_ids=..., finish_reason=..., stop_reason=...) from router
        responses. This smoke-tests that construction still works and fields
        land in the right positions.
        """
        import msgspec

        output = EngineCoreOutput(
            request_id="test-123",
            new_token_ids=[1, 2, 3],
            finish_reason=FinishReason.STOP,
            stop_reason="eos",
        )
        assert output.request_id == "test-123"
        assert output.new_token_ids == [1, 2, 3]
        assert output.finish_reason is FinishReason.STOP
        assert output.stop_reason == "eos"

        # Round-trip through msgpack to verify array_like serialization
        encoded = msgspec.msgpack.encode(output)
        decoded = msgspec.msgpack.decode(encoded, type=list)
        assert isinstance(decoded, list), f"Expected list, got {type(decoded)}"
        # First element is request_id
        assert decoded[0] == "test-123", f"request_id at wrong position: {decoded[0]}"
        # Second element is new_token_ids
        assert decoded[1] == [1, 2, 3], f"new_token_ids at wrong position: {decoded[1]}"

    def test_finish_reason_enum_contract(self):
        """Verify FinishReason enum names and string representations.

        vllm_processor.py maps router finish_reason strings to FinishReason
        enum values via _FINISH_REASON_MAP, and the output processor uses
        str(finish_reason) for the OpenAI response format.
        """
        assert FinishReason.STOP.name == "STOP"
        assert FinishReason.LENGTH.name == "LENGTH"
        assert FinishReason.ABORT.name == "ABORT"
        assert FinishReason.ERROR.name == "ERROR"

        assert str(FinishReason.STOP) == "stop"
        assert str(FinishReason.LENGTH) == "length"
        assert str(FinishReason.ABORT) == "abort"
        assert str(FinishReason.ERROR) == "error"

    def test_tool_and_reasoning_parser_manager_contract(self):
        """Verify ToolParserManager and ReasoningParserManager lookup signatures.

        EngineFactory calls ToolParserManager.get_tool_parser(name) and
        ReasoningParserManager.get_reasoning_parser(name) to resolve parser
        classes by name.
        """
        tool_parser_sig = inspect.signature(ToolParserManager.get_tool_parser)
        reasoning_parser_sig = inspect.signature(
            ReasoningParserManager.get_reasoning_parser
        )

        assert list(tool_parser_sig.parameters) == ["name"], (
            "ToolParserManager.get_tool_parser signature changed; "
            "update EngineFactory in components/src/dynamo/frontend/vllm_processor.py"
        )
        assert list(reasoning_parser_sig.parameters) == ["name"], (
            "ReasoningParserManager.get_reasoning_parser signature changed; "
            "update EngineFactory in components/src/dynamo/frontend/vllm_processor.py"
        )

    def test_tool_parser_method_signatures(self):
        """Verify ToolParser has adjust_request and extract_tool_calls_streaming.

        prepost.py calls tool_parser.adjust_request(request) during
        preprocessing and tool_parser.extract_tool_calls_streaming(...)
        during streaming post-processing.
        """
        assert hasattr(ToolParser, "adjust_request"), (
            "ToolParser no longer has 'adjust_request'; "
            "update preprocess_chat_request in "
            "components/src/dynamo/frontend/prepost.py"
        )
        adjust_sig = inspect.signature(ToolParser.adjust_request)
        adjust_params = list(adjust_sig.parameters)
        assert adjust_params[:2] == ["self", "request"], (
            "ToolParser.adjust_request signature changed; "
            f"expected (self, request, ...), got {adjust_params}"
        )

        assert hasattr(ToolParser, "extract_tool_calls_streaming"), (
            "ToolParser no longer has 'extract_tool_calls_streaming'; "
            "update StreamingPostProcessor in "
            "components/src/dynamo/frontend/prepost.py"
        )
        extract_sig = inspect.signature(ToolParser.extract_tool_calls_streaming)
        extract_params = list(extract_sig.parameters)
        expected_extract_params = [
            "self",
            "previous_text",
            "current_text",
            "delta_text",
            "previous_token_ids",
            "current_token_ids",
            "delta_token_ids",
            "request",
        ]
        assert extract_params == expected_extract_params, (
            "ToolParser.extract_tool_calls_streaming signature changed; "
            f"expected {expected_extract_params}, got {extract_params}"
        )

    def test_reasoning_parser_method_signatures(self):
        """Verify ReasoningParser has extract_reasoning_streaming and
        is_reasoning_end_streaming.

        prepost.py calls both during streaming post-processing to separate
        reasoning tokens from content tokens.
        """
        assert hasattr(ReasoningParser, "extract_reasoning_streaming"), (
            "ReasoningParser no longer has 'extract_reasoning_streaming'; "
            "update StreamingPostProcessor in "
            "components/src/dynamo/frontend/prepost.py"
        )
        extract_sig = inspect.signature(ReasoningParser.extract_reasoning_streaming)
        extract_params = list(extract_sig.parameters)
        expected_extract_params = [
            "self",
            "previous_text",
            "current_text",
            "delta_text",
            "previous_token_ids",
            "current_token_ids",
            "delta_token_ids",
        ]
        assert extract_params == expected_extract_params, (
            "ReasoningParser.extract_reasoning_streaming signature changed; "
            f"expected {expected_extract_params}, got {extract_params}"
        )

        assert hasattr(ReasoningParser, "is_reasoning_end_streaming"), (
            "ReasoningParser no longer has 'is_reasoning_end_streaming'; "
            "update StreamingPostProcessor in "
            "components/src/dynamo/frontend/prepost.py"
        )
        end_sig = inspect.signature(ReasoningParser.is_reasoning_end_streaming)
        end_params = list(end_sig.parameters)
        assert end_params == ["self", "input_ids", "delta_ids"], (
            "ReasoningParser.is_reasoning_end_streaming signature changed; "
            f"expected ['self', 'input_ids', 'delta_ids'], got {end_params}"
        )

    def test_preprocess_worker_result_picklability(self):
        """Verify PreprocessWorkerResult survives pickle round-trip.

        _preprocess_worker returns this dataclass via a ProcessPoolExecutor
        Future. If any field becomes unpicklable, the pool path breaks.
        """
        import pickle

        from dynamo.frontend.vllm_processor import PreprocessWorkerResult

        result = PreprocessWorkerResult(
            dynamo_preproc={
                "model": "test-model",
                "token_ids": [1, 2, 3],
                "stop_conditions": {
                    "max_tokens": 100,
                    "stop": [],
                    "stop_token_ids": [2],
                    "min_tokens": 0,
                    "ignore_eos": False,
                },
                "sampling_options": {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "min_p": 0.0,
                    "seed": None,
                },
                "output_options": {
                    "logprobs": None,
                    "prompt_logprobs": None,
                    "skip_special_tokens": True,
                },
                "eos_token_ids": [2],
                "annotations": [],
            },
            tokens=[1, 2, 3],
            vllm_preproc=EngineCoreRequest(
                request_id="test-123",
                prompt_token_ids=[1, 2, 3],
                mm_features=None,
                sampling_params=SamplingParams(),
                pooling_params=None,
                eos_token_id=2,
                arrival_time=0.0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                prompt_embeds=None,
                client_index=0,
                current_wave=0,
                priority=0,
                trace_headers=None,
            ),
            sampling_params=SamplingParams(),
            request_for_sampling={"model": "test-model", "tools": None},
            chat_template_kwargs={"reasoning_effort": None},
        )

        data = pickle.dumps(result)
        restored = pickle.loads(data)

        assert restored.dynamo_preproc == result.dynamo_preproc
        assert restored.tokens == result.tokens
        assert restored.vllm_preproc.request_id == "test-123"
        assert restored.request_for_sampling == result.request_for_sampling
        assert restored.chat_template_kwargs == result.chat_template_kwargs
