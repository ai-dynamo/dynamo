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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Payload:
    """
    Represents a test payload with expected response and log patterns.
    """

    payload_chat: Dict[str, Any]
    expected_response: List[str]
    expected_log: List[str]
    repeat_count: int = 1
    payload_completions: Optional[Dict[str, Any]] = None


def chat_completions_response_handler(response):
    """
    Process chat completions API responses.
    """
    if response.status_code != 200:
        return ""
    result = response.json()
    assert "choices" in result, "Missing 'choices' in response"
    assert len(result["choices"]) > 0, "Empty choices in response"
    assert "message" in result["choices"][0], "Missing 'message' in first choice"
    assert "content" in result["choices"][0]["message"], "Missing 'content' in message"
    return result["choices"][0]["message"]["content"]


def completions_response_handler(response):
    """
    Process completions API responses.
    """
    if response.status_code != 200:
        return ""
    result = response.json()
    assert "choices" in result, "Missing 'choices' in response"
    assert len(result["choices"]) > 0, "Empty choices in response"
    assert "text" in result["choices"][0], "Missing 'text' in first choice"
    return result["choices"][0]["text"]
