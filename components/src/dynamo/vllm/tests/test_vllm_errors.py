# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe validation diagnostics at the vLLM worker error boundary."""

import asyncio

import pytest
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.errors import (
    _translate_vllm_client_errors,
    uuid_image_cache_miss_error,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.timeout(5),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("after_output", [False, True])
@pytest.mark.parametrize(
    ("diagnostic", "public_message"),
    [
        (
            "Failed to transform json schema into a grammar: "
            "[07:13:49] /project/cpp/json_schema_converter.cc:4175: "
            "enum array must not be empty",
            "Invalid JSON schema: enum array must not be empty",
        ),
        (
            "Grammar error: Unsatisfiable schema: required property 'status' "
            "is unsatisfiable",
            "Invalid JSON schema: a required property is unsatisfiable",
        ),
    ],
)
async def test_generator_grammar_error_has_safe_public_message(
    after_output, diagnostic, public_message
):
    from vllm.exceptions import VLLMValidationError

    from dynamo.llm.exceptions import InvalidArgument
    from dynamo.vllm.errors import _translate_vllm_client_errors

    error = VLLMValidationError(diagnostic)

    async def engine_output():
        if after_output:
            yield {"token_ids": [1]}
        raise error

    stream = _translate_vllm_client_errors(engine_output())
    if after_output:
        assert await anext(stream) == {"token_ids": [1]}
    with pytest.raises(InvalidArgument) as rejected:
        await anext(stream)

    assert str(rejected.value) == public_message
    assert rejected.value.__cause__ is error
    assert str(rejected.value.__cause__) == diagnostic


@pytest.mark.asyncio
@pytest.mark.parametrize("error_cls", [RuntimeError, asyncio.CancelledError])
async def test_generator_preserves_other_failures(error_cls):
    from dynamo.vllm.errors import _translate_vllm_client_errors

    error = error_cls("engine stopped")

    async def engine_output():
        yield {"token_ids": [1]}
        raise error

    stream = _translate_vllm_client_errors(engine_output())
    assert await anext(stream) == {"token_ids": [1]}
    with pytest.raises(error_cls) as rejected:
        await anext(stream)
    assert rejected.value is error


@pytest.mark.parametrize(
    "mapper_name",
    ["vllm_client_error_to_http_error", "vllm_client_error_to_public_error"],
)
@pytest.mark.parametrize(
    ("diagnostic", "public_message"),
    [
        (
            "Failed to transform json schema into a grammar: "
            "[07:13:49] /private/compiler.cc:123: enum array must not be empty",
            "Invalid JSON schema: enum array must not be empty",
        ),
        (
            "Grammar error: Unsatisfiable schema: required property "
            "'/private/client_schema_secret' is unsatisfiable",
            "Invalid JSON schema: a required property is unsatisfiable",
        ),
    ],
)
def test_known_grammar_reason_has_only_authored_public_text(
    mapper_name, diagnostic, public_message
):
    from vllm.exceptions import VLLMValidationError

    from dynamo.llm.exceptions import InvalidArgument
    from dynamo.vllm import errors

    translated = getattr(errors, mapper_name)(VLLMValidationError(diagnostic))

    assert isinstance(translated, InvalidArgument)
    assert str(translated) == public_message


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Failed to transform json schema into a grammar: "
        "[07:13:49] /private/compiler.cc:123: unexpected private_schema_secret",
        "Grammar error: Unsatisfiable schema: private_schema_secret",
        "Drifted schema validator: [07:13:49] /private/compiler.cc:123: "
        "enum array must not be empty",
        "/private/compiler.cc:123: enum array must not be empty",
        "[07:13:49] /private/source tree/compiler.cc:123: enum array must not be empty",
        "Template render failed: {{ private_template_secret }} at /private/template.jinja",
    ],
)
def test_unknown_synthetic_worker_error_remains_private(diagnostic):
    from vllm.exceptions import VLLMValidationError

    from dynamo.llm.exceptions import HttpError
    from dynamo.vllm.errors import vllm_client_error_to_http_error

    translated = vllm_client_error_to_http_error(VLLMValidationError(diagnostic))

    assert isinstance(translated, HttpError)
    assert translated.code == 400
    assert translated.message == diagnostic


@pytest.mark.parametrize("error_name", ["VLLMValidationError", "VLLMClientError"])
@pytest.mark.parametrize(
    "diagnostic",
    [
        "The decoder prompt (length 65) is longer than the maximum model "
        "length of 64. Make sure that `max_model_len` is no smaller than the "
        "number of text tokens.",
        "At most 2 audio(s) may be provided in one prompt. "
        "Set `--limit-mm-per-prompt` to increase this limit. (parameter=audio)",
    ],
)
def test_public_mapper_preserves_messages_without_known_compiler_location(
    error_name, diagnostic
):
    from vllm import exceptions as vllm_exceptions

    from dynamo.llm.exceptions import InvalidArgument
    from dynamo.vllm.errors import vllm_client_error_to_public_error

    error = getattr(vllm_exceptions, error_name)(diagnostic)
    translated = vllm_client_error_to_public_error(error)

    assert isinstance(translated, InvalidArgument)
    assert str(translated) == diagnostic


@pytest.mark.parametrize("error_name", ["VLLMValidationError", "VLLMClientError"])
@pytest.mark.parametrize(
    "diagnostic",
    [
        "Drifted schema validator: [07:13:49] /private/compiler.cc:123: "
        "enum array must not be empty",
        "Failed to transform json schema into a grammar: "
        "[07:13:49] /private/compiler.cc:123: unknown compiler reason",
        "[07:13:49] /private/compiler.cc:123: The decoder prompt (length 65) "
        "is longer than the maximum model length of 64. private_model_secret",
        "Future wrapper: [07:13:49] /private/compiler.cc:123: At most 1 video(s) "
        "may be provided in one prompt. private_video_secret",
    ],
)
def test_known_compiler_log_format_is_private_for_any_400_class(error_name, diagnostic):
    from vllm import exceptions as vllm_exceptions

    from dynamo.llm.exceptions import HttpError, InvalidArgument
    from dynamo.vllm.errors import (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    )

    error = getattr(vllm_exceptions, error_name)(diagnostic)
    worker_error = vllm_client_error_to_http_error(error)
    public_error = vllm_client_error_to_public_error(error)

    assert isinstance(worker_error, HttpError)
    assert worker_error.code == 400
    assert worker_error.message == diagnostic
    assert isinstance(public_error, InvalidArgument)
    assert str(public_error) == "Invalid structured-output constraint"


@pytest.mark.parametrize("error_name", ["VLLMValidationError", "VLLMClientError"])
@pytest.mark.parametrize("prefix_offset", [-32, -16, 1])
def test_compiler_location_guard_uses_untruncated_diagnostic(error_name, prefix_offset):
    from vllm import exceptions as vllm_exceptions

    from dynamo.llm.exceptions import _MAX_MESSAGE_LENGTH, HttpError, InvalidArgument
    from dynamo.vllm.errors import (
        vllm_client_error_to_http_error,
        vllm_client_error_to_public_error,
    )

    cutoff = _MAX_MESSAGE_LENGTH - len("...")
    diagnostic = (
        "x" * (cutoff + prefix_offset)
        + "[07:13:49] /private/compiler.cc:123: unknown compiler reason"
    )
    error = getattr(vllm_exceptions, error_name)(diagnostic)

    worker_error = vllm_client_error_to_http_error(error)
    assert isinstance(worker_error, HttpError)
    assert worker_error.message == diagnostic[:cutoff] + "..."
    public_error = vllm_client_error_to_public_error(error)
    assert isinstance(public_error, InvalidArgument)
    assert str(public_error) == "Invalid structured-output constraint"


def test_direct_image_cache_miss_identifies_the_aligned_uuid_only_slot():
    request = {
        "multi_modal_data": {
            "image_url": [
                {"Url": "https://example.com/image?token=SECRET"},
                {"UuidOnly": "SECRET-UUID"},
            ]
        },
        "multi_modal_uuids": {"image_url": ["other-image", "SECRET-UUID"]},
    }
    error = uuid_image_cache_miss_error(
        ValueError("Cache miss for image at index 1 but data is not provided."), request
    )
    assert isinstance(error, InvalidArgument)
    assert "Resend the request" in str(error)
    assert "same UUID" in str(error)
    assert "SECRET" not in str(error)


@pytest.mark.parametrize(
    "message,slot,uuid",
    [
        ("Invalid sampling parameters", {"UuidOnly": "image-key"}, "image-key"),
        (
            "Cache miss for image at index 0 but data is not provided.",
            {"Url": "https://example.com/image"},
            "image-key",
        ),
        (
            "Cache miss for image at index 1 but data is not provided.",
            {"UuidOnly": "image-key"},
            "image-key",
        ),
        (
            "Cache miss for image at index 0 but data is not provided.",
            {"UuidOnly": "image-key"},
            "different-key",
        ),
    ],
    ids=["unrelated-validation", "url-backed", "wrong-index", "different-uuid"],
)
def test_unrelated_errors_are_not_reported_as_uuid_cache_misses(message, slot, uuid):
    request = {
        "multi_modal_data": {"image_url": [slot]},
        "multi_modal_uuids": {"image_url": [uuid]},
    }
    cause = ValueError(message)
    wrapped = EngineGenerateError()
    wrapped.__cause__ = cause
    for error in (cause, wrapped):
        assert uuid_image_cache_miss_error(error, request) is None


def test_wrapped_server_failure_is_not_reported_as_a_cache_miss():
    request = {
        "multi_modal_data": {"image_url": [{"UuidOnly": "image-key"}]},
        "multi_modal_uuids": {"image_url": ["image-key"]},
    }
    error = EngineGenerateError()
    error.__cause__ = RuntimeError(
        "Cache miss for image at index 0 but data is not provided."
    )
    assert uuid_image_cache_miss_error(error, request) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "wrapped"])
@pytest.mark.parametrize("uuid_matches", [True, False], ids=["matching", "mismatched"])
async def test_generator_translates_only_aligned_uuid_cache_misses(
    wrapped, uuid_matches
):
    request = {
        "multi_modal_data": {"image_url": [{"UuidOnly": "SECRET-UUID"}]},
        "multi_modal_uuids": {
            "image_url": ["SECRET-UUID" if uuid_matches else "other-image"]
        },
    }
    cause = ValueError("Cache miss for image at index 0 but data is not provided.")
    error = EngineGenerateError() if wrapped else cause
    if wrapped:
        error.__cause__ = cause

    async def engine_output():
        yield {"token_ids": [1]}
        raise error

    stream = _translate_vllm_client_errors(engine_output(), request)
    assert await anext(stream) == {"token_ids": [1]}
    if uuid_matches:
        with pytest.raises(InvalidArgument, match="Resend.*same UUID") as rejected:
            await anext(stream)
        assert rejected.value.__cause__ is error
        assert "SECRET" not in str(rejected.value)
    else:
        with pytest.raises(type(error)) as rejected:
            await anext(stream)
        assert rejected.value is error
