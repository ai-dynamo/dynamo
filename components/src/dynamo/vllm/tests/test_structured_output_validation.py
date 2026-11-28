# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native vLLM backend selection and Dynamo's public validation errors."""

import copy

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    # Cold vLLM model-registry initialization can exceed 30s on ARM64.
    pytest.mark.timeout(120),
]


def _object_schema(value_schema, property_name="value"):
    return {
        "type": "object",
        "properties": {property_name: value_schema},
        "required": [property_name],
        "additionalProperties": False,
    }


@pytest.fixture(scope="module")
def native_validation_config(tmp_path_factory):
    from tokenizers import Tokenizer, models
    from transformers import GPT2Config, PreTrainedTokenizerFast
    from vllm.config import ModelConfig

    directory = tmp_path_factory.mktemp("structured-output-model")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {"<unk>": 0, "<eos>": 1, "null": 2, "ok": 3}, unk_token="<unk>"
            )
        ),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(directory)
    GPT2Config(
        vocab_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=64,
        bos_token_id=0,
        eos_token_id=1,
        architectures=["GPT2LMHeadModel"],
    ).save_pretrained(directory)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_CACHE_ROOT", str(directory / "cache"))
        patch.setenv("HF_HUB_OFFLINE", "1")
        model = ModelConfig(
            model=str(directory),
            tokenizer=str(directory),
            max_model_len=64,
            dtype="float32",
        )
        yield model, tokenizer


@pytest.mark.parametrize("backend", ["auto", "xgrammar"])
@pytest.mark.parametrize(
    ("schema", "auto_backend", "auto_public_message"),
    [
        pytest.param(
            _object_schema(
                {"oneOf": [{"type": "string", "enum": []}, {"type": "null"}]}
            ),
            "guidance",
            None,
            id="oneof-empty-enum-null",
        ),
        pytest.param(
            _object_schema({"type": "string", "enum": []}, "private_status"),
            None,
            "Invalid JSON schema: a required property is unsatisfiable",
            id="required-empty-enum",
        ),
        pytest.param(
            _object_schema({"type": "string", "enum": ["ok"]}),
            "xgrammar",
            None,
            id="valid-enum",
        ),
    ],
)
def test_native_structured_output_validation_and_public_error(
    native_validation_config, backend, schema, auto_backend, auto_public_message
):
    from vllm.config import StructuredOutputsConfig
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    from dynamo.llm.exceptions import InvalidArgument
    from dynamo.vllm.errors import (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    )

    model, tokenizer = native_validation_config
    params = SamplingParams(
        max_tokens=16,
        structured_outputs=StructuredOutputsParams(json=copy.deepcopy(schema)),
    )

    def verify():
        params.verify(
            model_config=model,
            speculative_config=None,
            structured_outputs_config=(
                StructuredOutputsConfig()
                if backend == "auto"
                else StructuredOutputsConfig(backend=backend)
            ),
            tokenizer=tokenizer,
        )

    expected_backend = (
        auto_backend
        if backend == "auto"
        else ("xgrammar" if auto_backend == "xgrammar" else None)
    )
    if expected_backend is not None:
        verify()
        assert params.structured_outputs._backend == expected_backend
        assert params.structured_outputs._backend_was_auto is (backend == "auto")
        return

    with pytest.raises(VLLMValidationError) as rejected:
        verify()

    if backend == "auto":
        expected_message = auto_public_message
    else:
        expected_message = "Invalid JSON schema: enum array must not be empty"

    for translate in (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    ):
        public_error = translate(rejected.value)
        assert isinstance(public_error, InvalidArgument)
        assert str(public_error) == expected_message
        assert "private_status" not in str(public_error)
        assert "/project/" not in str(public_error)

    detail = str(rejected.value).partition(": ")[2]
    rejected.value.args = ("Upstream compiler wrapper changed: " + detail,)
    for translate in (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    ):
        public_error = translate(rejected.value)
        assert isinstance(public_error, InvalidArgument)
        assert str(public_error) == expected_message

    # Keep the native validation traceback while simulating a changed compiler
    # diagnostic; its wording must not decide whether private text is public.
    rejected.value.args = ("/private/sentinel: compiler diagnostic changed",)
    for translate in (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    ):
        public_error = translate(rejected.value)
        assert isinstance(public_error, InvalidArgument)
        assert str(public_error) == "Invalid structured-output constraint"


@pytest.mark.parametrize("backend", ["auto", "xgrammar"])
@pytest.mark.parametrize("grammar", ["", " \n\t "])
def test_native_blank_grammar_has_actionable_public_reason(
    native_validation_config, backend, grammar
):
    from vllm.config import StructuredOutputsConfig
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    from dynamo.llm.exceptions import InvalidArgument
    from dynamo.vllm.errors import (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    )

    model, tokenizer = native_validation_config
    params = SamplingParams(structured_outputs=StructuredOutputsParams(grammar=grammar))
    with pytest.raises(VLLMValidationError) as rejected:
        params.verify(model, None, StructuredOutputsConfig(backend=backend), tokenizer)

    for translate in (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    ):
        public_error = translate(rejected.value)
        assert isinstance(public_error, InvalidArgument)
        assert str(public_error) == "Invalid grammar: grammar must not be empty"


@pytest.mark.parametrize("backend", ["auto", "xgrammar"])
def test_native_regex_validation_keeps_compiler_diagnostics_private(
    native_validation_config, backend
):
    from vllm.config import StructuredOutputsConfig
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    from dynamo.llm.exceptions import InvalidArgument
    from dynamo.vllm.errors import (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    )

    model, tokenizer = native_validation_config
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(regex="(/private/regex_secret")
    )
    with pytest.raises(VLLMValidationError) as rejected:
        params.verify(
            model,
            None,
            StructuredOutputsConfig()
            if backend == "auto"
            else StructuredOutputsConfig(backend=backend),
            tokenizer,
        )

    for translate in (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    ):
        public_error = translate(rejected.value)
        assert isinstance(public_error, InvalidArgument)
        assert str(public_error) == "Invalid structured-output constraint"


def _assert_existing_request_error_mapping(error):
    from dynamo.llm.exceptions import HttpError, InvalidArgument
    from dynamo.vllm.errors import (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    )

    worker_error = vllm_client_error_to_http_error(error)
    assert isinstance(worker_error, HttpError)
    assert worker_error.code == 400
    assert worker_error.message == str(error)

    public_error = vllm_client_error_to_public_error(error)
    assert isinstance(public_error, InvalidArgument)
    assert str(public_error) == str(error)


def test_other_native_validation_preserves_public_reason_with_structured_outputs(
    native_validation_config,
):
    from vllm.config import StructuredOutputsConfig
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    model, tokenizer = native_validation_config
    params = SamplingParams(
        logprobs=max(model.max_logprobs, model.get_vocab_size()) + 1,
        structured_outputs=StructuredOutputsParams(
            json=_object_schema({"type": "string", "enum": ["ok"]})
        ),
    )
    with pytest.raises(VLLMValidationError) as rejected:
        params.verify(model, None, StructuredOutputsConfig(), tokenizer)

    assert rejected.value.parameter == "logprobs"
    assert params.structured_outputs._backend is None
    _assert_existing_request_error_mapping(rejected.value)


def test_native_prompt_length_error_preserves_public_reason(
    native_validation_config,
):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    model, _ = native_validation_config
    processor = InputProcessor(
        VllmConfig(model_config=model, device_config=DeviceConfig(device="cpu"))
    )
    try:
        with pytest.raises(VLLMValidationError) as rejected:
            processor._validate_prompt_len(model.max_model_len + 1, "decoder")
    finally:
        processor.renderer.shutdown()

    _assert_existing_request_error_mapping(rejected.value)


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"min_tokens": 8, "max_tokens": 4}, id="min-tokens-exceeds-max"),
        pytest.param({"n": 0}, id="zero-choices"),
        pytest.param({"repetition_penalty": 0}, id="zero-repetition-penalty"),
        pytest.param({"temperature": -1}, id="negative-temperature"),
    ],
)
def test_native_sampling_validation_preserves_public_reason(
    native_validation_config, options
):
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    model, tokenizer = native_validation_config
    with pytest.raises(VLLMValidationError) as rejected:
        params = SamplingParams(**options)
        params.verify(model, None, None, tokenizer)

    _assert_existing_request_error_mapping(rejected.value)
