# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate vLLM request errors into Dynamo's HTTP error boundary."""

import re
from typing import Any

from vllm.exceptions import (
    VLLMClientError,
    VLLMNotFoundError,
    VLLMUnprocessableEntityError,
)
from vllm.v1.engine.exceptions import EngineGenerateError

from dynamo.common.constants import UUID_ONLY_VARIANT_KEY
from dynamo.llm.exceptions import HttpError, InvalidArgument

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
