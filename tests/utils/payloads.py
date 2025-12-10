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
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dynamo import prometheus_names  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@dataclass
class BasePayload:
    """Generic payload body plus expectations and repeat count."""

    body: Dict[str, Any]
    expected_response: List[Any]  # Can be List[str] or List[List[str]] for alternatives
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
        """Default validation: ensure expected substrings appear in content.

        If expected_response is a list of strings, ANY one of them matching is sufficient (OR logic).
        This allows flexible validation where responses may vary but should contain at least one keyword.
        """
        if self.expected_response:
            # Check if content is empty
            if not content:
                logger.error("VALIDATION FAILED - Response content is empty")
                raise AssertionError(
                    f"Expected content not found in response. Expected any of: {self.expected_response}. Actual content is empty."
                )

            # Check if ANY of the expected strings are found (OR logic) and count matches
            found_keywords = []
            for expected in self.expected_response:
                if isinstance(expected, str) and expected.lower() in content.lower():
                    found_keywords.append(expected)

            if not found_keywords:
                logger.error(
                    f"VALIDATION FAILED - Actual content returned: {repr(content)}"
                )
                logger.error(
                    f"Expected to find at least one of: {self.expected_response}"
                )
                logger.error(f"Matches found: 0/{len(self.expected_response)}")
                raise AssertionError(
                    f"Expected content not found in response. Expected at least one of: {self.expected_response}. Actual content: {repr(content)}"
                )

            logger.info(
                f"SUCCESS: Found {len(found_keywords)}/{len(self.expected_response)} expected keywords: {found_keywords}"
            )

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

        assert (
            "choices" in result
        ), f"Missing 'choices' in response. Response keys: {list(result.keys())}"
        assert len(result["choices"]) > 0, "Empty choices in response"
        assert (
            "message" in result["choices"][0]
        ), f"Missing 'message' in first choice. Choice keys: {list(result['choices'][0].keys())}"

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
class ChatPayloadWithLogprobs(ChatPayload):
    """Chat payload that validates logprobs in response."""

    def validate(self, response: Any, content: str) -> None:
        """Validate response contains logprobs fields."""
        super().validate(response, content)

        result = response.json()
        choice = result["choices"][0]

        # Validate logprobs field exists
        assert "logprobs" in choice, "Missing 'logprobs' in choice"

        logprobs_data = choice["logprobs"]
        if logprobs_data is not None:
            assert "content" in logprobs_data, "Missing 'content' in logprobs"
            content_logprobs = logprobs_data["content"]

            if content_logprobs:
                # Validate structure of logprobs
                for item in content_logprobs:
                    assert "token" in item, "Missing 'token' in logprobs content"
                    assert "logprob" in item, "Missing 'logprob' in logprobs content"
                    assert (
                        "top_logprobs" in item
                    ), "Missing 'top_logprobs' in logprobs content"

                    # Sanity check: logprob should be valid (not nan/inf/positive)
                    logprob_val = item["logprob"]
                    assert not math.isnan(logprob_val), "logprob is NaN"
                    assert not math.isinf(logprob_val), "logprob is infinite"
                    assert (
                        logprob_val <= 0
                    ), f"logprob should be <= 0, got {logprob_val}"

                logger.info(
                    f"✓ Logprobs validation passed: found {len(content_logprobs)} tokens with logprobs"
                )


@dataclass
class ToolCallingChatPayload(ChatPayload):
    """ChatPayload that validates tool calls in the response."""

    def __init__(self, *args, expected_tool_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_tool_name = expected_tool_name

    def validate(self, response, content: str) -> None:
        """Validate that tool calls exist in the response."""
        # First run the standard validation
        super().validate(response, content)

        # Then validate tool calls specifically
        response_data = response.json()
        choices = response_data.get("choices", [])
        assert choices, "Response missing choices"

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        assert tool_calls, "Expected model to generate tool calls but none found"
        logger.info(f"Tool calls detected: {len(tool_calls)} call(s)")

        # Validate tool call structure
        for i, tc in enumerate(tool_calls):
            assert "function" in tc, f"Tool call {i} missing 'function' field"
            function = tc.get("function", {})
            assert "name" in function, f"Tool call {i} missing function name"
            assert "arguments" in function, f"Tool call {i} missing function arguments"
            logger.info(
                f"  [{i}] Function: {function.get('name')}, Args: {function.get('arguments')[:100]}..."
            )

        # If expected tool name is provided, validate it
        if self.expected_tool_name:
            tool_names = [tc.get("function", {}).get("name") for tc in tool_calls]
            assert (
                self.expected_tool_name in tool_names
            ), f"Expected tool '{self.expected_tool_name}' not found. Available tools: {tool_names}"
            logger.info(f"Expected tool '{self.expected_tool_name}' was called")


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
class CompletionPayloadWithLogprobs(CompletionPayload):
    """Completion payload that validates logprobs in response."""

    def validate(self, response: Any, content: str) -> None:
        """Validate response contains logprobs fields."""
        super().validate(response, content)

        result = response.json()
        choice = result["choices"][0]

        # Validate logprobs field exists
        assert "logprobs" in choice, "Missing 'logprobs' in choice"

        logprobs_data = choice["logprobs"]
        if logprobs_data is not None:
            assert (
                "token_logprobs" in logprobs_data
            ), "Missing 'token_logprobs' in logprobs"
            assert "tokens" in logprobs_data, "Missing 'tokens' in logprobs"

            token_logprobs = logprobs_data["token_logprobs"]
            tokens = logprobs_data["tokens"]

            if token_logprobs:
                assert len(token_logprobs) == len(
                    tokens
                ), "Mismatch between token_logprobs and tokens length"

                # Sanity check: each logprob should be valid (not nan/inf/positive)
                for i, logprob_val in enumerate(token_logprobs):
                    if logprob_val is not None:  # First token can be None
                        assert not math.isnan(
                            logprob_val
                        ), f"logprob at index {i} is NaN"
                        assert not math.isinf(
                            logprob_val
                        ), f"logprob at index {i} is infinite"
                        assert (
                            logprob_val <= 0
                        ), f"logprob at index {i} should be <= 0, got {logprob_val}"

                logger.info(
                    f"✓ Logprobs validation passed: found {len(token_logprobs)} tokens with logprobs"
                )


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
                >= 11,  # 80% of typical ~17 metrics (excluding _bucket) as of 2025-12-02
                error_msg=lambda name, value: f"Expected at least 11 unique {prefix}_* metrics, but found only {len(set(value))}",
                success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} unique {prefix}_* metrics (minimum required: 11)",
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
                validator=lambda value: int(float(value))
                >= 0,  # Allow 0 for SGLang (hardcoded issue in components/src/dynamo/sglang/publisher.py:70)
                error_msg=lambda name, value: f"{name} should be >= 0, but got {value}",
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
        elif backend == "lmcache":
            metrics_to_check.append(
                MetricCheck(
                    # Check: Minimum count of unique lmcache:* metrics
                    name="lmcache:*",
                    pattern=lambda name: r"^lmcache:\w+",
                    validator=lambda value: len(set(value))
                    >= 1,  # At least 1 lmcache metric
                    error_msg=lambda name, value: f"Expected at least 1 lmcache:* metric, but found only {len(set(value))}",
                    success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} lmcache:* metrics",
                    multiline=True,
                )
            )
        elif backend == "sglang":
            metrics_to_check.append(
                MetricCheck(
                    # Check: Minimum count of unique sglang:* metrics
                    name="sglang:*",
                    pattern=lambda name: r"^sglang:\w+",
                    validator=lambda value: len(set(value))
                    >= 20,  # 80% of typical ~25 sglang metrics (excluding _bucket) as of 2025-10-22 (but will grow)
                    error_msg=lambda name, value: f"Expected at least 20 unique sglang:* metrics, but found only {len(set(value))}",
                    success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} unique sglang:* metrics (minimum required: 20)",
                    multiline=True,
                )
            )
        elif backend == "trtllm":
            metrics_to_check.append(
                MetricCheck(
                    # Check: Minimum count of unique trtllm_* metrics
                    name="trtllm_*",
                    pattern=lambda name: r"^trtllm_\w+",
                    validator=lambda value: len(set(value))
                    >= 4,  # 80% of typical ~5 trtllm metrics (excluding _bucket) as of 2025-10-22 (but will grow)
                    error_msg=lambda name, value: f"Expected at least 4 unique trtllm_* metrics, but found only {len(set(value))}",
                    success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} unique trtllm_* metrics (minimum required: 4)",
                    multiline=True,
                )
            )

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
