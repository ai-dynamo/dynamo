#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from http import HTTPStatus


class DynamoException(Exception):
    """Base class for all Dynamo SDK Exception."""

    error_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_mapping: dict[HTTPStatus, type[DynamoException]] = {}

    def __init_subclass__(cls) -> None:
        if "error_code" in cls.__dict__:
            cls.error_mapping[cls.error_code] = cls

    def __init__(self, message: str, error_code: HTTPStatus | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.error_code
