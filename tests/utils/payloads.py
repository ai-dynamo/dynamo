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
    # When True, the HTTP request is made with stream=True (for SSE responses).
    http_stream: bool = False

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
                    assert item["token"] is not None, "logprobs 'token' should not be None (frontend should detokenize logprob tokens)"
                    assert isinstance(item["token"], str) and len(item["token"]) > 0, "logprobs 'token' should be a non-empty string"
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

                    # Validate top_logprobs structure and that token text is populated
                    # This tests that the frontend populates token text in logprobs for the client
                    top_logprobs_list = item["top_logprobs"]
                    assert isinstance(
                        top_logprobs_list, list
                    ), "top_logprobs should be a list"
                    assert len(top_logprobs_list) > 0, "top_logprobs should not be empty"
                    for top_lp in top_logprobs_list:
                        assert (
                            top_lp["token"] is not None
                        ), "top_logprobs 'token' should not be None (frontend should detokenize logprob tokens)"
                        assert isinstance(top_lp["token"], str) and len(top_lp["token"]) > 0, \
                            "top_logprobs 'token' should be a non-empty string"
                        top_logprob_val = top_lp["logprob"]
                        assert not math.isnan(top_logprob_val), "top_logprobs logprob is NaN"
                        assert not math.isinf(top_logprob_val), "top_logprobs logprob is infinite"
                        assert top_logprob_val <= 0, f"top_logprobs logprob should be <= 0, got {top_logprob_val}"

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
class ResponsesPayload(BasePayload):
    """Payload for the Responses API endpoint (/v1/responses).

    For full compliance testing, use the OpenResponses bun CLI:
      bun run test:compliance --base-url http://localhost:<port>/v1 --api-key test --model <model>
    See https://www.openresponses.org/compliance
    """

    endpoint: str = "/v1/responses"

    @staticmethod
    def extract_content(response):
        """Extract text content from a Responses API response."""
        response.raise_for_status()
        result = response.json()

        assert (
            result.get("object") == "response"
        ), f"Expected object='response', got {result.get('object')}"
        assert result.get("id", "").startswith(
            "resp_"
        ), f"Expected id to start with 'resp_', got {result.get('id')}"
        assert (
            result.get("status") == "completed"
        ), f"Expected status='completed', got {result.get('status')}"

        output = result.get("output", [])
        assert len(output) > 0, "Response output is empty"

        msg = output[0]
        assert (
            msg.get("type") == "message"
        ), f"Expected output[0].type='message', got {msg.get('type')}"
        assert (
            msg.get("role") == "assistant"
        ), f"Expected role='assistant', got {msg.get('role')}"

        content_parts = msg.get("content", [])
        assert len(content_parts) > 0, "Message content is empty"
        assert (
            content_parts[0].get("type") == "output_text"
        ), f"Expected content[0].type='output_text', got {content_parts[0].get('type')}"

        return content_parts[0].get("text", "")

    def response_handler(self, response: Any) -> str:
        return ResponsesPayload.extract_content(response)


@dataclass
class ResponsesStreamPayload(BasePayload):
    """Streaming payload for the Responses API endpoint (/v1/responses).

    Validates SSE event structure and lifecycle ordering.
    """

    endpoint: str = "/v1/responses"
    http_stream: bool = True

    @staticmethod
    def extract_content(response):
        """Parse SSE stream and validate event structure."""
        import json

        response.raise_for_status()

        events = []
        event_type = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    events.append(("done", None))
                else:
                    events.append((event_type, json.loads(data_str)))

        event_types = [e[0] for e in events]

        # Validate lifecycle event ordering
        assert len(event_types) >= 2, f"Too few events: {event_types}"
        assert (
            event_types[0] == "response.created"
        ), f"First event should be response.created, got {event_types[0]}"
        assert (
            event_types[1] == "response.in_progress"
        ), f"Second event should be response.in_progress, got {event_types[1]}"

        non_done = [e for e in event_types if e != "done"]
        assert (
            non_done[-1] == "response.completed"
        ), f"Last real event should be response.completed, got {non_done[-1]}"

        # Validate text content events
        assert "response.output_item.added" in event_types, "Missing output_item.added"
        assert (
            "response.content_part.added" in event_types
        ), "Missing content_part.added"
        assert "response.output_text.delta" in event_types, "Missing output_text.delta"
        assert "response.output_text.done" in event_types, "Missing output_text.done"
        assert "response.content_part.done" in event_types, "Missing content_part.done"
        assert "response.output_item.done" in event_types, "Missing output_item.done"

        # Verify text deltas concatenate to the final text
        deltas = [e[1]["delta"] for e in events if e[0] == "response.output_text.delta"]
        done_events = [e for e in events if e[0] == "response.output_text.done"]
        assert (
            len(done_events) == 1
        ), f"Expected 1 output_text.done, got {len(done_events)}"
        full_text = "".join(deltas)
        assert (
            done_events[0][1]["text"] == full_text
        ), "Concatenated deltas don't match output_text.done text"

        return full_text

    def response_handler(self, response: Any) -> str:
        return ResponsesStreamPayload.extract_content(response)


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
    """Base class for Prometheus metrics validation payloads.

    Validates common dynamo_component_* metrics shared across all backends.
    Backend-specific subclasses handle engine-specific metrics.
    """

    endpoint: str = "/metrics"
    method: str = "GET"
    port: int = DefaultPort.SYSTEM1.value
    min_num_requests: int = 1

    def with_model(self, model):
        # Metrics does not use model in request body
        return self

    def response_handler(self, response: Any) -> str:
        response.raise_for_status()
        return response.text

    def _filter_bucket_metrics(self, content: str) -> str:
        """Filter out histogram bucket metrics to avoid count inflation"""
        content_lines = content.split("\n")
        filtered_lines = [line for line in content_lines if "_bucket{" not in line]
        return "\n".join(filtered_lines)

    def _get_common_metric_checks(self) -> list[MetricCheck]:
        """Get common dynamo_component_* metric checks shared across all backends"""
        prefix = prometheus_names.name_prefix.COMPONENT

        # Define metrics to check
        # Pattern matches: metric_name{labels} value OR metric_name value (labels optional)
        # Examples:
        #   - dynamo_component_requests_total{model="Qwen/Qwen3-0.6B"} 6
        #   - dynamo_component_uptime_seconds 150.390999059
        # Note: Supports scientific notation (e.g., 8.34e-05)
        def metric_pattern(name):
            return rf"{name}(?:\{{[^}}]*\}})?\s+([\d.eE+-]+)"

        return [
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
            MetricCheck(
                name=f"{prefix}_{prometheus_names.kvstats.TOTAL_BLOCKS}",
                pattern=metric_pattern,
                validator=lambda value: float(value) >= 0,
                error_msg=lambda name, value: f"{name} should be >= 0, but got {value}",
                success_msg=lambda name, value: f"SUCCESS: Found {name} = {value}",
            ),
            MetricCheck(
                name=f"{prefix}_{prometheus_names.kvstats.GPU_CACHE_USAGE_PERCENT}",
                pattern=metric_pattern,
                validator=lambda value: 0.0 <= float(value) <= 1.0,
                error_msg=lambda name, value: f"{name} should be between 0.0 and 1.0, but got {value}",
                success_msg=lambda name, value: f"SUCCESS: Found {name} = {value}",
            ),
            MetricCheck(
                name=f"{prefix}_{prometheus_names.model_info.LOAD_TIME_SECONDS}",
                pattern=metric_pattern,
                validator=lambda value: float(value) > 0,
                error_msg=lambda name, value: f"{name} should be > 0, but got {value}",
                success_msg=lambda name, value: f"SUCCESS: Found {name} = {float(value):.2f}s",
            ),
        ]

    def _get_backend_specific_checks(self) -> list[MetricCheck]:
        """Get backend-specific metric checks. Override in subclasses."""
        return []

    def _validate_metric_checks(
        self, metrics_to_check: list[MetricCheck], content: str
    ) -> None:
        """Run all metric checks and raise AssertionError if any fail"""
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

    def validate(self, response: Any, content: str) -> None:
        """Validate Prometheus metrics output"""
        content = self._filter_bucket_metrics(content)

        # Collect all checks: common + backend-specific
        metrics_to_check = self._get_common_metric_checks()
        metrics_to_check.extend(self._get_backend_specific_checks())

        # Run all validations
        self._validate_metric_checks(metrics_to_check, content)


@dataclass
class VLLMMetricsPayload(MetricsPayload):
    """Metrics validation for vLLM backend with auto-label checks"""

    def _get_backend_specific_checks(self) -> list[MetricCheck]:
        """vLLM-specific metric checks"""
        checks = [
            MetricCheck(
                # Check: Minimum count of unique vllm:* metrics
                name="vllm:*",
                pattern=lambda name: r"^vllm:\w+",
                validator=lambda value: len(set(value))
                >= 56,  # 80% of typical ~70 vllm metrics (excluding _bucket) as of 2026-02-05 (but will grow)
                error_msg=lambda name, value: f"Expected at least 56 unique vllm:* metrics, but found only {len(set(value))}",
                success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} unique vllm:* metrics (minimum required: 56)",
                multiline=True,
            )
        ]

        # Check required labels: auto-injected (from prometheus_names.labels) + injected by backend
        required_labels = [
            prometheus_names.labels.NAMESPACE,
            prometheus_names.labels.COMPONENT,
            prometheus_names.labels.ENDPOINT,
            prometheus_names.labels.MODEL,  # OpenAI standard (injected by all backends)
            prometheus_names.labels.MODEL_NAME,  # Alternative label (injected for compatibility)
        ]
        for label_name in required_labels:
            checks.append(
                MetricCheck(
                    name=f"vllm:* with {label_name}",
                    pattern=lambda name, lbl=label_name: rf'vllm:\w+\{{[^}}]*{lbl}="[^"]+"',
                    validator=lambda value: len(value) > 0,
                    error_msg=lambda name, value, lbl=label_name: f"vLLM metrics missing label: {lbl}",
                    success_msg=lambda name, value, lbl=label_name: f"SUCCESS: vLLM metrics include {lbl} label (found {len(value)} metrics)",
                    multiline=True,
                )
            )

        return checks


@dataclass
class LMCacheMetricsPayload(MetricsPayload):
    """Metrics validation for lmcache"""

    def _get_backend_specific_checks(self) -> list[MetricCheck]:
        """lmcache-specific metric checks"""
        return [
            MetricCheck(
                # Check: Minimum count of unique lmcache:* metrics
                name="lmcache:*",
                pattern=lambda name: r"^lmcache:\w+",
                validator=lambda value: len(set(value))
                >= 26,  # 80% of typical ~33 lmcache metrics (excluding _bucket) as of 2026-02-05 (but will grow)
                error_msg=lambda name, value: f"Expected at least 26 unique lmcache:* metrics, but found only {len(set(value))}",
                success_msg=lambda name, value: f"SUCCESS: Found {len(set(value))} lmcache:* metrics (minimum required: 26)",
                multiline=True,
            )
        ]


@dataclass
class SGLangMetricsPayload(MetricsPayload):
    """Metrics validation for SGLang backend with auto-label checks"""

    def _get_backend_specific_checks(self) -> list[MetricCheck]:
        """SGLang-specific metric checks"""
        checks = [
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
        ]

        # Check required labels: auto-injected (from prometheus_names.labels) + injected by backend
        required_labels = [
            prometheus_names.labels.NAMESPACE,
            prometheus_names.labels.COMPONENT,
            prometheus_names.labels.ENDPOINT,
            prometheus_names.labels.MODEL,  # OpenAI standard (injected by all backends)
            prometheus_names.labels.MODEL_NAME,  # Alternative label (injected for compatibility)
        ]
        for label_name in required_labels:
            checks.append(
                MetricCheck(
                    name=f"sglang:* with {label_name}",
                    pattern=lambda name, lbl=label_name: rf'sglang:\w+\{{[^}}]*{lbl}="[^"]+"',
                    validator=lambda value: len(value) > 0,
                    error_msg=lambda name, value, lbl=label_name: f"sglang metrics missing label: {lbl}",
                    success_msg=lambda name, value, lbl=label_name: f"SUCCESS: sglang metrics include {lbl} label (found {len(value)} metrics)",
                    multiline=True,
                )
            )

        return checks


@dataclass
class TRTLLMMetricsPayload(MetricsPayload):
    """Metrics validation for TensorRT-LLM backend"""

    def _get_backend_specific_checks(self) -> list[MetricCheck]:
        """TRT-LLM-specific metric checks"""
        checks = [
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
        ]

        # Check required labels: auto-injected (from prometheus_names.labels) + injected by backend
        required_labels = [
            prometheus_names.labels.NAMESPACE,
            prometheus_names.labels.COMPONENT,
            prometheus_names.labels.ENDPOINT,
            prometheus_names.labels.MODEL,  # OpenAI standard (injected by all backends)
            prometheus_names.labels.MODEL_NAME,  # Alternative label (injected for compatibility)
        ]
        for label_name in required_labels:
            checks.append(
                MetricCheck(
                    name=f"trtllm_* with {label_name}",
                    pattern=lambda name, lbl=label_name: rf'trtllm_\w+\{{[^}}]*{lbl}="[^"]+"',
                    validator=lambda value: len(value) > 0,
                    error_msg=lambda name, value, lbl=label_name: f"TRT-LLM metrics missing label: {lbl}",
                    success_msg=lambda name, value, lbl=label_name: f"SUCCESS: TRT-LLM metrics include {lbl} label (found {len(value)} metrics)",
                    multiline=True,
                )
            )

        return checks


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
