# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from dynamo.llm import HttpError


def test_raise_http_error():
    with pytest.raises(HttpError):
        raise HttpError(404, "Not Found")
    with pytest.raises(Exception):
        raise HttpError(500, "Internal Server Error")


def test_invalid_http_error_code():
    with pytest.raises(ValueError):
        HttpError(1700, "Invalid Code")
