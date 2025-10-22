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
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dynamo import prometheus_names

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
class EmbeddingPayload(BasePayload):
    """Payload for embeddings endpoint."""

    endpoint: str = "/v1/embeddings"

    @staticmethod
    def extract_embeddings(response):
        """
        Process embeddings API responses.
        """
        response.raise_for_status()
        result = response.json()
        assert "object" in result, "Missing 'object' in response"
        assert (
            result["object"] == "list"
        ), f"Expected object='list', got {result['object']}"
        assert "data" in result, "Missing 'data' in response"
        assert len(result["data"]) > 0, "Empty data in response"

        # Extract embedding vectors and validate structure
        embeddings = []
        for item in result["data"]:
            assert "object" in item, "Missing 'object' in embedding item"
            assert (
                item["object"] == "embedding"
            ), f"Expected object='embedding', got {item['object']}"
            assert "embedding" in item, "Missing 'embedding' vector in item"
            assert isinstance(
                item["embedding"], list
            ), "Embedding should be a list of floats"
            assert len(item["embedding"]) > 0, "Embedding vector should not be empty"
            embeddings.append(item["embedding"])

        # Return a summary string for validation
        return f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}"

    def response_handler(self, response: Any) -> str:
        return EmbeddingPayload.extract_embeddings(response)


@dataclass
class MetricCheck:
    """Definition of a metric validation check"""

    name: str
    pattern: Callable[[str], str]
    validator: Callable[[Any], bool]
    error_msg: Callable[[str, Any], str]
    success_msg: Callable[[str, Any], str]
    multiline: bool = False


@dataclass
class MetricsPayload(BasePayload):
    endpoint: str = "/metrics"
    method: str = "GET"
    port: int = 8081
    min_num_requests: int = 1
    backend: Optional[
        str
    ] = None  # Backend identifier for metrics validation (e.g., 'vllm', 'sglang', 'trtllm')

    def with_model(self, model):
        # Metrics does not use model in request body
        return self

    def response_handler(self, response: Any) -> str:
        response.raise_for_status()
        return response.text

    def validate(self, response: Any, content: str) -> None:
        # Use backend from payload configuration
        backend = self.backend

        # Filter out _bucket metrics from content (histogram buckets inflate counts)
        content_lines = content.split("\n")
        filtered_lines = [line for line in content_lines if "_bucket{" not in line]
        content = "\n".join(filtered_lines)

        # Build full metric names with prefix
        prefix = prometheus_names.name_prefix.COMPONENT

        # Define metrics to check
        # Pattern matches: metric_name{labels} value OR metric_name value (labels optional)
        # Examples:
        #   - dynamo_component_requests_total{model="Qwen/Qwen3-0.6B"} 6
        #   - dynamo_component_uptime_seconds 150.390999059
        def metric_pattern(name):
            return rf"{name}(?:\{{[^}}]*\}})?\s+([\d.]+)"

        metrics_to_check = [
            MetricCheck(
                # Check: Minimum count of unique dynamo_component_* metrics
                name=f"{prefix}_*",
                pattern=lambda name: rf"^{prefix}_\w+",
                validator=lambda value: len(set(value))
                >= 23,  # 80% of typical ~29 metrics (excluding _bucket) as of 2025-10-22 (but will grow)
                error_msg=lambda name, value: f"Expected at least 23 unique {prefix}_* metrics, but found only {len(set(value))}",
                success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} unique {prefix}_* metrics (minimum required: 23)",
                multiline=True,
            ),
            MetricCheck(
                name=f"{prefix}_{prometheus_names.work_handler.REQUESTS_TOTAL}",
                pattern=metric_pattern,
                validator=lambda value: int(float(value)) >= self.min_num_requests,
                error_msg=lambda name, value: f"{name} has count {value} which is less than required {self.min_num_requests}",
                success_msg=lambda name, value: f"SUCCESS: Found {name} with count: {value}",
            ),
            MetricCheck(
                name=f"{prefix}_{prometheus_names.distributed_runtime.UPTIME_SECONDS}",
                pattern=metric_pattern,
                validator=lambda value: float(value) > 0,
                error_msg=lambda name, value: f"{name} should be > 0, but got {value}",
                success_msg=lambda name, value: f"SUCCESS: Found {name} = {value}s",
            ),
            MetricCheck(
                name=f"{prefix}_{prometheus_names.kvstats.TOTAL_BLOCKS}",
                pattern=metric_pattern,
                validator=lambda value: int(float(value)) > 0,
                error_msg=lambda name, value: f"{name} should be > 0, but got {value}",
                success_msg=lambda name, value: f"SUCCESS: Found {name} = {value}",
            ),
        ]

        # Add backend-specific metric checks
        if backend == "vllm":
            metrics_to_check.append(
                MetricCheck(
                    # Check: Minimum count of unique vllm:* metrics
                    name="vllm:*",
                    pattern=lambda name: r"^vllm:\w+",
                    validator=lambda value: len(set(value))
                    >= 52,  # 80% of typical ~65 vllm metrics (excluding _bucket) as of 2025-10-22 (but will grow)
                    error_msg=lambda name, value: f"Expected at least 52 unique vllm:* metrics, but found only {len(set(value))}",
                    success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} unique vllm:* metrics (minimum required: 52)",
                    multiline=True,
                )
            )
        # TODO: Add sglang:* and trtllm:* metrics checks (similar to vllm above)

        # Check all metrics
        for metric in metrics_to_check:
            # Special handling for multiline patterns (like counting unique metrics)
            if metric.multiline:
                pattern = metric.pattern(metric.name)
                matches = re.findall(pattern, content, re.MULTILINE)
                if not matches:
                    raise AssertionError(
                        f"Could not find any matches for pattern '{metric.name}'"
                    )

                # For multiline, pass the entire list to validator
                if metric.validator(matches):
                    logger.info(metric.success_msg(metric.name, matches))
                else:
                    raise AssertionError(metric.error_msg(metric.name, matches))
            else:
                # Standard single-value metric check
                if metric.name not in content:
                    raise AssertionError(
                        f"Metric '{metric.name}' not found in metrics output"
                    )

                pattern = metric.pattern(metric.name)
                matches = re.findall(pattern, content)
                if not matches:
                    raise AssertionError(
                        f"Could not parse value for metric '{metric.name}'"
                    )

                # For metrics with multiple values (like requests_total with different labels),
                # check if any match passes validation
                validation_passed = False
                last_value = None
                for match in matches:
                    last_value = match
                    if metric.validator(match):
                        logger.info(metric.success_msg(metric.name, match))
                        validation_passed = True
                        break

                if not validation_passed:
                    raise AssertionError(
                        metric.error_msg(
                            metric.name, last_value if last_value else "N/A"
                        )
                    )


def check_models_api(response):
    """Check if models API is working and returns models"""
    try:
        if response.status_code != 200:
            return False
        data = response.json()
        time.sleep(
            1
        )  # temporary to avoid /completions race condition where we get 404 error
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
                time.sleep(
                    1
                )  # temporary to avoid /completions race condition where we get 404 error
                return True

        # Check instances for an entry with endpoint == 'generate'
        instances = data.get("instances", []) or []
        for inst in instances:
            if isinstance(inst, dict) and inst.get("endpoint") == "generate":
                time.sleep(
                    1
                )  # temporary to avoid /completions race condition where we get 404 error
                return True

        return False
    except Exception:
        return False


# backwards compatiability
def completions_response_handler(response):
    return CompletionPayload.extract_text(response)


def chat_completions_response_handler(response):
    return ChatPayload.extract_content(response)
