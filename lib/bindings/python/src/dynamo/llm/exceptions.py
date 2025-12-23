# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import logging

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 8192


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
