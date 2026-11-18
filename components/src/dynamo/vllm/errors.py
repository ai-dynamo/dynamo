# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate vLLM request errors into Dynamo's HTTP error boundary."""

import re
import sys
from collections.abc import AsyncIterator
from typing import TypeVar

from vllm.exceptions import (
    VLLMClientError,
    VLLMNotFoundError,
    VLLMUnprocessableEntityError,
    VLLMValidationError,
)

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
) -> AsyncIterator[ResponseT]:
    """Keep request-side vLLM errors client-visible on worker endpoints."""
    try:
        async for chunk in generator:
            yield chunk
    except VLLMClientError as exc:
        raise vllm_client_error_to_http_error(exc) from exc


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
        if _XGRAMMAR_LOG_PREFIX.search(error.message):
            return InvalidArgument("Invalid structured-output constraint")
        return InvalidArgument(error.message)
    return error
