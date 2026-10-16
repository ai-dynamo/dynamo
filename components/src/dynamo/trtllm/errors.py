# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify request-local TensorRT-LLM failures; fatal lifecycle stays in handlers."""

from dynamo.llm.exceptions import DynamoException, classified_error


def engine_error(error: Exception, *, validation: bool = False) -> DynamoException:
    if isinstance(error, DynamoException):
        return error
    if validation and isinstance(error, NotImplementedError):
        identity = ("NotImplemented", "runtime.not_implemented")
    elif validation and isinstance(error, (ValueError, TypeError)):
        identity = ("InvalidRequest", "backend.invalid_argument")
    else:
        # Native RequestError denotes per-request failure, not client fault.
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


def protocol_error(message: str) -> DynamoException:
    return classified_error(
        "BackendProtocol", reason="backend.protocol", diagnostic=message
    )
