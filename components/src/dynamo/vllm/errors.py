# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate vLLM request errors into Dynamo's HTTP error boundary."""

import re
import sys
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from vllm.exceptions import (
    VLLMClientError,
    VLLMNotFoundError,
    VLLMUnprocessableEntityError,
    VLLMValidationError,
)
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.common.constants import UUID_ONLY_VARIANT_KEY
from dynamo.llm.exceptions import HttpError, InvalidArgument

ResponseT = TypeVar("ResponseT")

_XGRAMMAR_LOG_PREFIX = re.compile(r"\[\d{2}:\d{2}:\d{2}\] \S+:\d+: ")


def _from_structured_output_validation(exc: VLLMValidationError) -> bool:
    # vLLM uses the same exception class for unrelated validation stages. Match
    # the loaded validator's code, without importing validation modules or
    # running grammar compilation twice.
    module = sys.modules.get("vllm.sampling_params")
    params_type = getattr(module, "SamplingParams", None)
    validator = getattr(params_type, "_validate_structured_outputs", None)
    code = getattr(validator, "__code__", None)
    traceback = exc.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code is code:
            return True
        traceback = traceback.tb_next
    return False


def _structured_output_message(exc: VLLMValidationError) -> str | None:
    # Removing this known log format is defense in depth, never permission to
    # publish the remaining arbitrary diagnostic.
    diagnostic = _XGRAMMAR_LOG_PREFIX.sub("", str(exc)).strip()
    structured = _from_structured_output_validation(exc)
    if (
        structured
        and diagnostic == "structured_outputs.grammar cannot be an empty string"
    ):
        return "Invalid grammar: grammar must not be empty"
    if structured or diagnostic.startswith(
        "Failed to transform json schema into a grammar: "
    ):
        if diagnostic == "enum array must not be empty" or diagnostic.endswith(
            ": enum array must not be empty"
        ):
            return "Invalid JSON schema: enum array must not be empty"
    if structured or diagnostic.startswith("Grammar error: "):
        if (
            "Unsatisfiable schema: required property '" in diagnostic
            and diagnostic.endswith("' is unsatisfiable")
        ):
            return "Invalid JSON schema: a required property is unsatisfiable"
    if structured:
        return "Invalid structured-output constraint"
    return None


async def _translate_vllm_client_errors(
    generator: AsyncIterator[ResponseT],
    request: dict[str, Any] | None = None,
) -> AsyncIterator[ResponseT]:
    """Keep request-side vLLM errors client-visible on worker endpoints."""
    try:
        async for chunk in generator:
            yield chunk
    except VLLMClientError as exc:
        raise vllm_client_error_to_http_error(exc) from exc
    except (ValueError, EngineGenerateError) as exc:
        if request is not None:
            if error := uuid_image_cache_miss_error(exc, request):
                raise error from exc
        raise


_IMAGE_CACHE_MISS = re.compile(
    r"Cache miss for (?:image|vision_chunk) at index ([0-9]{1,10}) but data is not provided\."
)


def uuid_image_cache_miss_error(
    exc: ValueError | EngineGenerateError, request: dict[str, Any]
) -> InvalidArgument | None:
    # vLLM 0.33 wraps processor ValueErrors in EngineGenerateError. Keep this
    # adapter at the engine boundary until vLLM exposes a typed cache miss.
    cause = exc.__cause__ if isinstance(exc, EngineGenerateError) else exc
    if type(cause) is not ValueError:
        return None
    match = _IMAGE_CACHE_MISS.fullmatch(str(cause))
    if match is None:
        return None
    index = int(match[1])
    images = (request.get("multi_modal_data") or {}).get("image_url") or []
    uuids = (request.get("multi_modal_uuids") or {}).get("image_url") or []
    if index >= len(images) or index >= len(uuids):
        return None
    image = images[index]
    if (
        not isinstance(image, dict)
        or len(image) != 1
        or not isinstance(uuids[index], str)
        or not uuids[index]
        or image.get(UUID_ONLY_VARIANT_KEY) != uuids[index]
    ):
        return None
    return InvalidArgument(
        "Cached image is unavailable on this worker. Resend the request with the "
        "image URL or inline image data and the same UUID; UUID-only reuse "
        "requires the image to be cached on every worker that processes the image."
    )


def vllm_client_error_to_http_error(
    exc: VLLMClientError,
) -> HttpError | InvalidArgument:
    """Preserve vLLM's status, keeping unknown worker diagnostics private."""
    if isinstance(exc, VLLMValidationError):
        if message := _structured_output_message(exc):
            return InvalidArgument(message)
    if isinstance(exc, VLLMUnprocessableEntityError):
        status_code = 422
    elif isinstance(exc, VLLMNotFoundError):
        status_code = 404
    else:
        status_code = 400
    return HttpError(status_code, str(exc))


def vllm_client_error_to_public_error(
    exc: VLLMClientError,
) -> HttpError | InvalidArgument:
    """Preserve public request reasons while withholding compiler diagnostics.

    Structured-output validation has its own authored messages. When provenance
    is unavailable, a known xgrammar log prefix still keeps compiler details
    private. Other 400 reasons retain the existing frontend/Omni public policy;
    404/422 retain private HttpError diagnostics and their original status.
    """
    error = vllm_client_error_to_http_error(exc)
    if isinstance(error, InvalidArgument):
        return error
    if error.code == 400:
        # HttpError caps its message and can cut a compiler location in half.
        if _XGRAMMAR_LOG_PREFIX.search(str(exc)):
            return InvalidArgument("Invalid structured-output constraint")
        return InvalidArgument(error.message)
    return error
