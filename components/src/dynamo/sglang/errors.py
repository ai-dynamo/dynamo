# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify SGLang failures at its structured abort and engine boundaries."""

from collections.abc import Mapping

from dynamo.llm.exceptions import DynamoException, classified_error

_MAX_DIAGNOSTIC_LENGTH = 8192


def abort_error_message(finish_reason: Mapping) -> str | None:
    if finish_reason.get("type") != "abort":
        return None
    if (
        finish_reason.get("status_code") is None
        and finish_reason.get("err_type") is None
    ):
        return None
    message = finish_reason.get("message")
    return (
        message
        if isinstance(message, str) and message
        else "SGLang aborted the request with an error"
    )


def raise_if_abort_error(finish_reason: Mapping) -> None:
    message = abort_error_message(finish_reason)
    if message is None:
        return
    # FINISH_ABORT's status is an engine signal, not Dynamo's response policy.
    status = finish_reason.get("status_code")
    identity = {
        400: ("InvalidRequest", "backend.invalid_argument"),
        422: ("InvalidRequest", "backend.invalid_argument"),
        429: ("CapacityExhausted", "capacity.worker_overloaded"),
        503: ("Unavailable", "backend.unavailable"),
    }.get(
        status if isinstance(status, int) and not isinstance(status, bool) else None,
        ("Internal", "backend.unknown"),
    )
    public = (
        {"type": "message", "message": "The inference engine rejected the request"}
        if identity[0] == "InvalidRequest"
        else None
    )
    raise classified_error(
        identity[0],
        reason=identity[1],
        diagnostic=message[:_MAX_DIAGNOSTIC_LENGTH],
        public_details=public,
    )


def engine_error(error: Exception) -> DynamoException:
    if isinstance(error, DynamoException):
        return error
    return classified_error(
        "Internal",
        reason="backend.unknown",
        diagnostic=str(error)[:_MAX_DIAGNOSTIC_LENGTH],
    )


async def submit_generation(awaitable):
    try:
        return await awaitable
    except DynamoException:
        raise
    except Exception as error:
        raise engine_error(error) from error


async def classified_stream(stream):
    try:
        async for result in stream:
            yield result
    except DynamoException:
        raise
    except Exception as error:
        raise engine_error(error) from error
