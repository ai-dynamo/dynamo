# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate vLLM request errors into Dynamo's HTTP error boundary."""

from vllm.exceptions import (
    VLLMClientError,
    VLLMNotFoundError,
    VLLMUnprocessableEntityError,
)

from dynamo.llm.exceptions import HttpError, InvalidArgument


def vllm_client_error_to_http_error(exc: VLLMClientError) -> HttpError:
    """Preserve the HTTP status assigned by vLLM's client-error hierarchy."""
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
    """Translate a vLLM request error, keeping a 400's reason client-visible.

    Dynamo's Python-to-Rust error boundary treats the text of an ``HttpError``
    as private diagnostics, so a 400 raised as one reaches the client as a bare
    "Bad Request". vLLM's 400 request errors (prompt longer than the maximum
    model length, too many multimodal items, ...) are written for the caller,
    so they are raised as ``InvalidArgument``, whose message is public. 404 and
    422 keep their status through ``HttpError`` with the text still private.
    """
    http_error = vllm_client_error_to_http_error(exc)
    if http_error.code == 400:
        return InvalidArgument(str(exc))
    return http_error
