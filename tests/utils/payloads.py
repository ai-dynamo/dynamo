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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class BasePayload:
    """Generic payload body plus expectations and repeat count."""

    body: Dict[str, Any]
    expected_response: List[str]
    expected_log: List[str]
    repeat_count: int = 1
    timeout: int = 60

    # Connection info
    host: str = "localhost"
    port: int = 8000
    endpoint: str = ""
    method: str = "POST"

    def url(self) -> str:
        ep = self.endpoint.lstrip("/")
        return f"http://{self.host}:{self.port}/{ep}"

    def with_model(self, model):
        p = deepcopy(self)
        if "model" not in p.body:
            p.body = {**p.body, "model": model}
        return p

    def response_handler(self, response: Any) -> str:
        """Extract a text representation of the response for logging/validation."""
        raise NotImplementedError("Subclasses must implement response_handler()")

    def validate(self, response: Any, content: str) -> None:
        """Default validation: ensure expected substrings appear in content."""
        if self.expected_response:
            missing_expected = []
            for expected in self.expected_response:
                if not content or expected not in content:
                    missing_expected.append(expected)
            if missing_expected:
                raise AssertionError(
                    f"Expected content not found in response. Missing: {missing_expected}"
                )
        logger.info(f"SUCCESS: All expected_responses: {self.expected_response} found.")

    def process_response(self, response: Any) -> str:
        """Convenience: run response_handler then validate; return content."""
        content = self.response_handler(response)
        self.validate(response, content)
        return content


@dataclass
class ChatPayload(BasePayload):
    """Payload for chat completions endpoint."""

    endpoint: str = "/v1/chat/completions"

    @staticmethod
    def extract_content(response):
        """
        Process chat completions API responses.
        """
        response.raise_for_status()
        result = response.json()
        assert "choices" in result, "Missing 'choices' in response"
        assert len(result["choices"]) > 0, "Empty choices in response"
        assert "message" in result["choices"][0], "Missing 'message' in first choice"

        # Check for content in all possible fields where parsers might put output:
        # 1. content - standard message content
        # 2. reasoning_content - for models with reasoning parsers
        # 3. refusal - when the model refuses to answer
        # 4. tool_calls - for function/tool calling responses

        message = result["choices"][0]["message"]

        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        refusal = message.get("refusal", "")

        tool_calls = message.get("tool_calls", [])
        tool_content = ""
        if tool_calls:
            tool_content = ", ".join(
                call.get("function", {}).get("arguments", "")
                for call in tool_calls
                if call.get("function", {}).get("arguments")
            )

        for field_content in [content, reasoning_content, refusal, tool_content]:
            if field_content:
                return field_content

        raise ValueError(
            "All possible content fields are empty in message. "
            f"Checked: content={repr(content)}, reasoning_content={repr(reasoning_content)}, "
            f"refusal={repr(refusal)}, tool_calls={tool_calls}"
        )

    def response_handler(self, response: Any) -> str:
        return ChatPayload.extract_content(response)


@dataclass
class CompletionPayload(BasePayload):
    """Payload for completions endpoint."""

    endpoint: str = "/v1/completions"

    @staticmethod
    def extract_text(response):
        """
        Process completions API responses.
        """
        response.raise_for_status()
        result = response.json()
        assert "choices" in result, "Missing 'choices' in response"
        assert len(result["choices"]) > 0, "Empty choices in response"
        assert "text" in result["choices"][0], "Missing 'text' in first choice"
        return result["choices"][0]["text"]

    def response_handler(self, response: Any) -> str:
        return CompletionPayload.extract_text(response)


@dataclass
class MetricsPayload(BasePayload):
    endpoint: str = "/metrics"
    method: str = "GET"
    port: int = 8081
    min_num_requests: int = 1

    def with_model(self, model):
        # Metrics does not use model in request body
        return self

    def response_handler(self, response: Any) -> str:
        response.raise_for_status()
        return response.text

    def validate(self, response: Any, content: str) -> None:
        pattern = r'dynamo_component_requests_total\{[^}]*model="[^"]*"[^}]*\}\s+(\d+)'
        matches = re.findall(pattern, content)
        if not matches:
            raise AssertionError(
                "Metric 'dynamo_component_requests_total' with model label not found in metrics output"
            )

        for match in matches:
            request_count = int(match)
            if request_count >= self.min_num_requests:
                logger.info(
                    f"SUCCESS: Found dynamo_component_requests_total with count: {request_count}"
                )
                return

        raise AssertionError(
            f"dynamo_component_requests_total exists but has count {request_count} which is less than required {self.min_num_requests}"
        )


def check_models_api(response):
    """Check if models API is working and returns models"""
    try:
        if response.status_code != 200:
            return False
        data = response.json()
        return data.get("data") and len(data["data"]) > 0
    except Exception:
        return False


# Additional health check helpers
def check_health_generate(response):
    """Validate /health reports a 'generate' endpoint.

    Returns True if either of the following is found:
      - "endpoints" contains a string mentioning 'generate'
      - "instances" contains an object with endpoint == 'generate'
    """
    try:
        if response.status_code != 200:
            return False
        data = response.json()

        # Check endpoints list for any entry containing 'generate'
        endpoints = data.get("endpoints", []) or []
        for ep in endpoints:
            if isinstance(ep, str) and "generate" in ep:
                return True

        # Check instances for an entry with endpoint == 'generate'
        instances = data.get("instances", []) or []
        for inst in instances:
            if isinstance(inst, dict) and inst.get("endpoint") == "generate":
                return True

        return False
    except Exception:
        return False


# backwards compatiability
def completions_response_handler(response):
    return CompletionPayload.extract_text(response)


def chat_completions_response_handler(response):
    return ChatPayload.extract_content(response)
