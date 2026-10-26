# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request driving and response validation for migration tests."""

import logging
import threading
import time
from dataclasses import dataclass, field

import pytest
from openai import OpenAI

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME

logger = logging.getLogger(__name__)


class OutputContinuityError(AssertionError):
    """A migrated response diverged from its stable fault-free prefix."""


@dataclass
class MigrationResponse:
    """Client-visible response state collected by the request thread."""

    observations: list[tuple[str | None | Exception, float]] = field(
        default_factory=list
    )
    finish_reason: str | None = None
    completion_tokens: int | None = None


def _make_client(frontend_port: int) -> OpenAI:
    """Build a non-retrying client pointed at the test frontend."""
    return OpenAI(
        base_url=f"http://localhost:{frontend_port}/v1",
        api_key="not-needed",
        max_retries=0,
        timeout=240,
    )


def _request_args(
    *,
    prompt: str,
    use_chat_completion: bool,
    stream: bool,
    max_tokens: int | None,
    force_max_output_tokens: bool,
) -> dict:
    args = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "stream": stream,
        "temperature": 0,
        "seed": 0,
    }
    if use_chat_completion:
        args["messages"] = [{"role": "user", "content": prompt}]
    else:
        args["prompt"] = prompt
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if force_max_output_tokens:
        if max_tokens is None:
            raise ValueError("force_max_output_tokens requires max_tokens")
        args["extra_body"] = {"ignore_eos": True, "min_tokens": max_tokens}
        if stream:
            args["stream_options"] = {"include_usage": True}
    return args


def start_request(
    frontend_port: int,
    *,
    use_chat_completion: bool,
    stream: bool,
    use_long_prompt: bool = False,
    max_tokens: int | None = None,
    long_prompt_repetitions: int = 8_000,
    force_max_output_tokens: bool = False,
) -> tuple[threading.Thread, MigrationResponse]:
    """Start one completion or chat-completion request in a background thread."""
    response = MigrationResponse()

    def send_request() -> None:
        prompt = "Tell me a long long long story about yourself?"
        if use_long_prompt:
            prompt += " Make sure it is" + " long" * long_prompt_repetitions + "!"

        request_api = "chat completion" if use_chat_completion else "completion"
        logger.info(
            "Sending %s request (stream=%s) with prompt: '%s...'",
            request_api,
            stream,
            prompt[:50],
        )
        response.observations.append((None, time.monotonic()))

        try:
            client = _make_client(frontend_port)
            args = _request_args(
                prompt=prompt,
                use_chat_completion=use_chat_completion,
                stream=stream,
                max_tokens=max_tokens,
                force_max_output_tokens=force_max_output_tokens,
            )
            if stream:
                chunks = (
                    client.chat.completions.create(**args)
                    if use_chat_completion
                    else client.completions.create(**args)
                )
                for chunk in chunks:
                    if chunk.usage is not None:
                        response.completion_tokens = chunk.usage.completion_tokens
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        continue
                    if choice.finish_reason is not None:
                        response.finish_reason = choice.finish_reason
                    content = (
                        choice.delta.content if use_chat_completion else choice.text
                    )
                    if content is not None:
                        response.observations.append((content, time.monotonic()))
            elif use_chat_completion:
                result = client.chat.completions.create(**args)
                choice = result.choices[0]
                response.finish_reason = choice.finish_reason
                if result.usage is not None:
                    response.completion_tokens = result.usage.completion_tokens
                response.observations.append((choice.message.content, time.monotonic()))
            else:
                result = client.completions.create(**args)
                choice = result.choices[0]
                response.finish_reason = choice.finish_reason
                if result.usage is not None:
                    response.completion_tokens = result.usage.completion_tokens
                response.observations.append((choice.text, time.monotonic()))
        except Exception as error:
            logger.error("Request failed with error: %s", error)
            response.observations.append((error, time.monotonic()))

    request_thread = threading.Thread(target=send_request, daemon=True)
    request_thread.start()
    return request_thread, response


def wait_for_response(
    response: MigrationResponse,
    num_responses: int = 5,
    max_wait_time: float = 10.0,
) -> None:
    """Block until the request has produced enough non-empty payload chunks."""
    deadline = time.monotonic() + max_wait_time
    while time.monotonic() < deadline:
        content_count = sum(
            1
            for content, _ in response.observations
            if isinstance(content, str) and content
        )
        if content_count >= num_responses:
            return
        time.sleep(0.001)

    content_count = sum(
        1
        for content, _ in response.observations
        if isinstance(content, str) and content
    )
    pytest.fail(
        f"Only observed {content_count}/{num_responses} non-empty response chunks "
        f"within {max_wait_time}s"
    )


def validate_response(
    request_thread: threading.Thread,
    response: MigrationResponse,
    expected_completion_tokens: int | None = None,
) -> str:
    """Wait for a terminal response and validate its client-visible contract."""
    request_thread.join(timeout=240)
    assert not request_thread.is_alive(), "Request did not complete within 240 seconds"

    observations = response.observations
    assert observations, "Missing first entry with start timestamp"
    assert observations[0][0] is None, "First entry should be start timestamp only"
    prev_timestamp = observations[0][1]
    response_parts: list[str] = []

    for result, timestamp in observations[1:]:
        delay = timestamp - prev_timestamp
        if delay > 2.0:
            logger.info("Observed %.3fs before the next response chunk", delay)
        prev_timestamp = timestamp

        assert result is not None, "Response entry should not be None"
        if isinstance(result, Exception):
            raise result
        response_parts.append(result)

    assert any(response_parts), "Request completed without non-empty response content"
    assert (
        response.finish_reason is not None
    ), "Request completed without a terminal finish reason"
    if expected_completion_tokens is not None:
        assert response.finish_reason == "length", (
            "Forced-length request terminated unexpectedly: "
            f"finish_reason={response.finish_reason!r}"
        )
        assert response.completion_tokens == expected_completion_tokens, (
            "Forced-length request returned the wrong completion-token count: "
            f"expected={expected_completion_tokens}, "
            f"actual={response.completion_tokens}"
        )

    output = "".join(response_parts)
    logger.info("Received %s response(s): %s...", len(response_parts), output[:100])
    return output


def assert_output_prefix(output: str, expected_prefix: str) -> None:
    """Require output to preserve a fault-free prefix with useful diagnostics."""
    if output.startswith(expected_prefix):
        return

    mismatch_index = next(
        (
            index
            for index, (expected, actual) in enumerate(zip(expected_prefix, output))
            if expected != actual
        ),
        min(len(expected_prefix), len(output)),
    )
    context_start = max(0, mismatch_index - 40)
    context_end = mismatch_index + 80
    raise OutputContinuityError(
        "Migrated output diverged from the stable fault-free prefix "
        f"at character {mismatch_index}: "
        f"expected={expected_prefix[context_start:context_end]!r}, "
        f"actual={output[context_start:context_end]!r}"
    )


def request_to_completion(frontend_port: int, **request_options) -> str:
    """Run one request synchronously through the shared migration request path."""
    thread, response = start_request(frontend_port, **request_options)
    expected_tokens = (
        request_options.get("max_tokens")
        if request_options.get("force_max_output_tokens")
        else None
    )
    return validate_response(
        thread, response, expected_completion_tokens=expected_tokens
    )
