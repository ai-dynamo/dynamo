# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate vLLM request errors into Dynamo's HTTP error boundary."""

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import (
    VLLMClientError,
    VLLMNotFoundError,
    VLLMUnprocessableEntityError,
    VLLMValidationError,
)

from dynamo.llm.exceptions import HttpError


def vllm_client_error_to_http_error(exc: VLLMClientError) -> HttpError:
    """Preserve the HTTP status assigned by vLLM's client-error hierarchy."""
    if isinstance(exc, VLLMUnprocessableEntityError):
        status_code = 422
    elif isinstance(exc, VLLMNotFoundError):
        status_code = 404
    else:
        status_code = 400
    # Native exception text can contain request values or backend diagnostics.
    # Only declared top-level parameter identities are public; unknown/nested
    # parameters remain unverified, and no field is inferred from the message.
    parameter = None
    if (
        status_code == 400
        and isinstance(exc, VLLMValidationError)
        and isinstance(exc.parameter, str)
    ):
        if (
            exc.parameter in ChatCompletionRequest.model_fields
            or exc.parameter in CompletionRequest.model_fields
        ):
            parameter = exc.parameter
    return HttpError(status_code, str(exc), param=parameter)
