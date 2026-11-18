# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate vLLM request errors into Dynamo's HTTP error boundary."""

import logging

import torch
from vllm.exceptions import (
    VLLMClientError,
    VLLMNotFoundError,
    VLLMUnprocessableEntityError,
)
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.llm.exceptions import HttpError, InvalidArgument

logger = logging.getLogger(__name__)

# Prefix of the ValueError vLLM's InputProcessingContext.call_hf_processor
# raises around any exception from the model's HF processor.
_HF_PROCESSOR_WRAPPER_PREFIX = "Failed to apply "

PROCESSOR_REJECTED_INPUT_MESSAGE = "The model's processor rejected the multimodal input"

# Transformers' BatchFeature.convert_to_tensors re-raises any tensor-creation
# failure, including these, as ValueError.
_RESOURCE_ERRORS = (MemoryError, torch.OutOfMemoryError)


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
        return InvalidArgument(http_error.message)
    return http_error


def _caused_by_resource_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, _RESOURCE_ERRORS):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def processor_input_error(
    exc: EngineGenerateError, request_id: str
) -> InvalidArgument | None:
    """Return a 400 when the model's HF processor rejected the request input.

    vLLM wraps the processor's exception in ``ValueError("Failed to apply ...")``
    and ``AsyncLLM.generate`` wraps that in ``EngineGenerateError``, a server
    error. Only a ``ValueError`` from the processor itself (an image with an
    aspect ratio or size the processor refuses) is treated as client input;
    a processor bug (``TypeError``, ``IndexError``, ...) or a ``ValueError``
    caused by running out of memory stays a server error.
    The wrapper text embeds the prompt and media, so the public message is
    fixed and the processor's reason is only logged.
    """
    wrapper = exc.__cause__
    if type(wrapper) is not ValueError or not str(wrapper).startswith(
        _HF_PROCESSOR_WRAPPER_PREFIX
    ):
        return None
    reason = wrapper.__cause__
    if not isinstance(reason, ValueError) or _caused_by_resource_error(reason):
        return None
    logger.warning(
        "Request %s: model processor rejected request input: %s", request_id, reason
    )
    return InvalidArgument(PROCESSOR_REJECTED_INPUT_MESSAGE)
