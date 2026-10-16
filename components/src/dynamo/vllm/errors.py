# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify native vLLM failures without choosing an HTTP response policy."""

from vllm.exceptions import (
    QueueOverflowError,
    VLLMClientError,
    VLLMError,
    VLLMNotFoundError,
)
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.llm.exceptions import DynamoException, HttpError, classified_error


def engine_error(error: Exception) -> DynamoException:
    if isinstance(error, DynamoException):
        return error
    if isinstance(error, EngineDeadError):
        identity = ("Unavailable", "backend.engine_shutdown")
    elif isinstance(error, QueueOverflowError):
        identity = ("CapacityExhausted", "capacity.worker_overloaded")
    elif isinstance(error, VLLMNotFoundError):
        identity = ("NotFound", "request.not_found")
    elif isinstance(error, VLLMClientError):
        identity = ("InvalidRequest", "backend.invalid_argument")
    else:
        # Recoverable EngineGenerateError and iteration-time ValueError are not
        # evidence of malformed client input.
        identity = ("Internal", "backend.unknown")
    public = (
        {"type": "message", "message": "The inference engine rejected the request"}
        if identity[0] == "InvalidRequest"
        else None
    )
    return classified_error(
        identity[0],
        reason=identity[1],
        diagnostic=str(error)[:8192],
        public_details=public,
    )


async def classified_stream(generator, *, classify_unknown: bool = True):
    try:
        async for chunk in generator:
            yield chunk
    except (DynamoException, HttpError):
        raise
    except EngineDeadError as error:
        if classify_unknown:
            # The engine boundary must leave fatal lifecycle handling to its caller.
            raise
        raise engine_error(error) from error
    except Exception as error:
        if not classify_unknown and not isinstance(error, VLLMError):
            # Request builders retain their explicit validation behavior. Unknown
            # engine faults are classified at the narrower iterator boundary.
            raise
        raise engine_error(error) from error


def protocol_error(message: str) -> DynamoException:
    return classified_error(
        "BackendProtocol", reason="backend.protocol", diagnostic=message
    )
