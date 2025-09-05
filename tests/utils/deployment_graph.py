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

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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

    message = result["choices"][0]["message"]

    # Check for content in all possible fields where parsers might put output:
    # 1. content - standard message content
    # 2. reasoning_content - for models with reasoning parsers
    # 3. refusal - when the model refuses to answer
    # 4. tool_calls - for function/tool calling responses

    content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")
    refusal = message.get("refusal", "")

    # Check for tool calls
    tool_calls = message.get("tool_calls", [])
    tool_content = ""
    if tool_calls:
        # Extract content from tool calls
        tool_content = ", ".join(
            call.get("function", {}).get("arguments", "")
            for call in tool_calls
            if call.get("function", {}).get("arguments")
        )

    # Return the first non-empty field in priority order
    for field_content in [content, reasoning_content, refusal, tool_content]:
        if field_content:
            return field_content

    # If all fields are empty, provide a detailed error
    raise ValueError(
        "All possible content fields are empty in message. "
        f"Checked: content={repr(content)}, reasoning_content={repr(reasoning_content)}, "
        f"refusal={repr(refusal)}, tool_calls={tool_calls}"
    )


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


def metrics_handler(response):
    """Handler to check if metrics endpoint is working and contains model label."""
    if response.status_code != 200:
        raise AssertionError(
            f"Metrics endpoint returned non-200 status code: {response.status_code}"
        )

    metrics_text = response.text

    # Check for any model label in dynamo_component_requests_total metric
    pattern = r'dynamo_component_requests_total\{[^}]*model="[^"]*"[^}]*\}\s+(\d+)'
    matches = re.findall(pattern, metrics_text)

    if not matches:
        raise AssertionError(
            "Metric 'dynamo_component_requests_total' with model label not found in metrics output"
        )

    # Since we send a request first, the counter should be > 0
    for match in matches:
        request_count = int(match)
        if request_count > 0:
            logger.info(
                f"Found dynamo_component_requests_total with count: {request_count}"
            )
            return metrics_text

    raise AssertionError(
        "dynamo_component_requests_total exists but has count of 0 - request was not tracked"
    )
