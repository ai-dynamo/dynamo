# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa


class HttpError(Exception):
    def __init__(self, code: int, message: str):
        if not (isinstance(code, int) and 0 <= code < 600):
            raise ValueError("HTTP status code must be an integer between 0 and 599")
        if not (isinstance(message, str) and 0 < len(message) <= 8192):
            raise ValueError("HTTP error message must be a string of length <= 8192")
        self.code = code
        self.message = message
        super().__init__(f"HTTP {code}: {message}")
