# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from dynamo import prometheus_names  # type: ignore[attr-defined]
from tests.utils.constants import DefaultPort

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
    port: int = DefaultPort.FRONTEND.value
    endpoint: str = ""
    method: str = "POST"
    # Optional additional ports used by specialized payloads (e.g. LoRA system/control-plane APIs).
    # This is intentionally empty by default to preserve prior semantics.
    system_ports: list[int] = field(default_factory=list)

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
class CachedTokensChatPayload(ChatPayload):
    """
    Chat payload that validates cached tokens are populated in repeated requests.

    Used for testing KV router cache-aware routing where repeated identical prompts
    should result in cached tokens being reported in the usage field.

    Validates that usage.prompt_tokens_details.cached_tokens > 0 for requests
    after the first one (since identical prompts should hit the prefix cache).
    """

    def __init__(
        self,
        body: dict,
        repeat_count: int = 3,
        expected_response: Optional[List[str]] = None,
        expected_log: Optional[List[str]] = None,
        timeout: int = 60,
        min_cached_tokens: int = 1,
    ):
        super().__init__(
            body=body,
            repeat_count=repeat_count,
            expected_response=expected_response or [],
            expected_log=expected_log or [],
            timeout=timeout,
        )
        self.min_cached_tokens = min_cached_tokens
        self._request_count = 0
        self._cached_tokens_found = False

    def validate(self, response: Any, content: str) -> None:
        """Validate response and check for cached tokens on repeated requests."""
        # First run the standard content validation
        super().validate(response, content)

        self._request_count += 1
        result = response.json()

        # Check usage field for cached tokens
        # Expected structure: usage.prompt_tokens_details.cached_tokens
        usage = result.get("usage", {})
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0

        logger.info(
            f"Request {self._request_count}: prompt_tokens={usage.get('prompt_tokens')}, "
            f"cached_tokens={cached_tokens}, prompt_tokens_details={prompt_tokens_details}"
        )

        # For requests after the first one, we expect cached tokens > 0
        # (since identical prompts should hit the prefix cache)
        if self._request_count > 1:
            if cached_tokens >= self.min_cached_tokens:
                self._cached_tokens_found = True
                logger.info(
                    f"✓ Request {self._request_count}: Cached tokens validation PASSED - "
                    f"found {cached_tokens} cached tokens (min required: {self.min_cached_tokens})"
                )
            else:
                logger.warning(
                    f"Request {self._request_count}: cached_tokens={cached_tokens} "
                    f"(expected >= {self.min_cached_tokens})"
                )

    def final_validation(self) -> None:
        """Called after all requests are processed to ensure we saw cached tokens.

        Raises AssertionError if cached tokens were not found on any repeated request.
        """
        if self.repeat_count > 1 and not self._cached_tokens_found:
            raise AssertionError(
                f"Expected cached_tokens >= {self.min_cached_tokens} in "
                f"prompt_tokens_details for at least one repeated request, "
                f"but none found after {self._request_count} requests. "
                f"Verify that prefix caching is enabled and working correctly."
            )
        logger.info(
            "✓ Final validation PASSED: cached_tokens found in repeated requests"
        )


@dataclass
class LoraTestChatPayload(ChatPayload):
    """
    Chat payload that loads a LoRA adapter before sending inference requests.

    This payload first loads the specified LoRA adapter via the system API,
    then sends chat completion requests using the LoRA model.
    """

    def __init__(
        self,
        body: dict,
        lora_name: str,
        s3_uri: str,
        system_port: int = DefaultPort.SYSTEM1.value,
        repeat_count: int = 1,
        expected_response: Optional[list] = None,
        expected_log: Optional[list] = None,
        timeout: int = 60,
    ):
        super().__init__(
            body=body,
            repeat_count=repeat_count,
            expected_response=expected_response or [],
            expected_log=expected_log or [],
            timeout=timeout,
        )
        self.system_ports = [system_port]
        self.lora_name = lora_name
        self.s3_uri = s3_uri
        self._lora_loaded = False

    def _ensure_lora_loaded(self) -> None:
        """Ensure the LoRA adapter is loaded before making inference requests"""
        if not self._lora_loaded:
            # Import the load_lora_adapter function
            # Note: This import is done here to avoid circular dependencies
            from tests.serve.lora_utils import load_lora_adapter

            load_lora_adapter(
                system_port=self.system_ports[0],
                lora_name=self.lora_name,
                s3_uri=self.s3_uri,
                timeout=self.timeout,
            )

            # Wait for the LoRA model to appear in /v1/models
            models_url = f"http://{self.host}:{self.port}/v1/models"
            start_time = time.time()

            logger.info(
                f"Waiting for LoRA model '{self.lora_name}' to appear in /v1/models..."
            )

            while time.time() - start_time < self.timeout:
                try:
                    response = requests.get(models_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        models = data.get("data", [])
                        model_ids = [m.get("id", "") for m in models]

                        if self.lora_name in model_ids:
                            logger.info(
                                f"LoRA model '{self.lora_name}' is now available"
                            )
                            self._lora_loaded = True
                            return

                        logger.debug(
                            f"Available models: {model_ids}, waiting for '{self.lora_name}'..."
                        )
                except requests.RequestException as e:
                    logger.debug(f"Error checking /v1/models: {e}")

                time.sleep(1)

            raise RuntimeError(
                f"Timeout: LoRA model '{self.lora_name}' did not appear in /v1/models within {self.timeout}s"
            )

    def url(self) -> str:
        """Load LoRA before first request, then return URL"""
        self._ensure_lora_loaded()
        return super().url()


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
    port: int = DefaultPort.SYSTEM1.value
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
                >= 7,  # 80% of typical ~13 metrics (excluding _bucket and removed kvstats metrics)
                error_msg=lambda name, value: f"Expected at least 7 unique {prefix}_* metrics, but found only {len(set(value))}",
                success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} unique {prefix}_* metrics (minimum required: 7)",
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
                    name="sglang_*",
                    pattern=lambda name: r"^sglang_\w+",
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
