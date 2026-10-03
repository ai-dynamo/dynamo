# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import logging
import re

from dynamo._core import Cancelled as Cancelled
from dynamo._core import CannotConnect as CannotConnect
from dynamo._core import ConnectionTimeout as ConnectionTimeout
from dynamo._core import Disconnected as Disconnected
from dynamo._core import DynamoException as DynamoException
from dynamo._core import EngineShutdown as EngineShutdown
from dynamo._core import InvalidArgument as InvalidArgument
from dynamo._core import RouterQueueLimitExceeded as RouterQueueLimitExceeded
from dynamo._core import SelectionServiceError as SelectionServiceError
from dynamo._core import StreamIncomplete as StreamIncomplete
from dynamo._core import Unknown as Unknown

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 8192


class HttpError(Exception):
    """HTTP failure with private diagnostic text.

    ``param`` is optional, explicitly public metadata for HTTP 400 validation
    errors. Select it from a reviewed top-level request schema, never a request
    value or exception message. It does not make ``message`` client-visible.
    Older bindings ignore this metadata without changing error classification.
    """

    def __init__(self, code: int, message: str, *, param: str | None = None):
        # These ValueErrors are easier to trace to here than the TypeErrors that
        # would be raised otherwise.
        if not isinstance(code, int) or isinstance(code, bool):
            raise ValueError("HttpError status code must be an integer")

        if not isinstance(message, str):
            raise ValueError("HttpError message must be a string")

        if not (0 <= code < 600):
            raise ValueError("HTTP status code must be an integer between 0 and 599")

        if param is not None and (
            code != 400
            or not isinstance(param, str)
            or len(param) > 128
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", param) is None
        ):
            raise ValueError("HttpError param must be a top-level field for status 400")

        if len(message) > _MAX_MESSAGE_LENGTH:
            logger.warning(
                f"HttpError message length {len(message)} exceeds max length {_MAX_MESSAGE_LENGTH}, truncating..."
            )
            message = message[: (_MAX_MESSAGE_LENGTH - 3)] + "..."

        self.code = code
        self.message = message
        self.param = param

        super().__init__(f"HTTP {code}: {message}")
