# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import logging
from typing import Optional

from dynamo._core import Cancelled as Cancelled
from dynamo._core import CannotConnect as CannotConnect
from dynamo._core import ConnectionTimeout as ConnectionTimeout
from dynamo._core import Disconnected as Disconnected
from dynamo._core import DynamoException as DynamoException
from dynamo._core import EngineShutdown as EngineShutdown
from dynamo._core import InvalidArgument as InvalidArgument
from dynamo._core import StreamIncomplete as StreamIncomplete
from dynamo._core import Unknown as Unknown

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 8192


class BackpressureError(DynamoException):
    """Router queue depth cap exceeded. Caller should retry/backoff.
    
    This exception indicates that the router's scheduler queue has reached
    its configured depth limit. The caller should:
    - Wait briefly and retry, or
    - Fall back to a smaller model/different provider, or
    - Return a 429 Too Many Requests to the client
    
    Attributes:
        reason: The specific backpressure reason (e.g., MaxQueueDepthExceeded)
        queue_depth: Current number of queued requests
        max_queue_depth: The configured limit (may be None if disabled)
    """
    
    def __init__(
        self,
        message: str,
        reason: Optional[str] = None,
        queue_depth: Optional[int] = None,
        max_queue_depth: Optional[int] = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.queue_depth = queue_depth
        self.max_queue_depth = max_queue_depth


class HttpError(Exception):
    def __init__(self, code: int, message: str):
        # These ValueErrors are easier to trace to here than the TypeErrors that
        # would be raised otherwise.
        if not isinstance(code, int) or isinstance(code, bool):
            raise ValueError("HttpError status code must be an integer")

        if not isinstance(message, str):
            raise ValueError("HttpError message must be a string")

        if not (0 <= code < 600):
            raise ValueError("HTTP status code must be an integer between 0 and 599")

        if len(message) > _MAX_MESSAGE_LENGTH:
            logger.warning(
                f"HttpError message length {len(message)} exceeds max length {_MAX_MESSAGE_LENGTH}, truncating..."
            )
            message = message[: (_MAX_MESSAGE_LENGTH - 3)] + "..."

        self.code = code
        self.message = message

        super().__init__(f"HTTP {code}: {message}")
