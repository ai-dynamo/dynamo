# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve payloads for vLLM's public structured-output validation errors."""

import json
import time
from dataclasses import dataclass, field

import requests

from tests.utils.payloads import BasePayload
from tests.utils.prometheus import sum_metric_samples
from tests.utils.router_nvext import parse_sse_json_chunks


@dataclass
class _StructuredOutputPayload(BasePayload):
    endpoint: str = "/v1/chat/completions"
    timeout: int = 30

    def response_handler(self, response: requests.Response) -> str:
        deadline = time.monotonic() + self.timeout
        content = bytearray()
        for chunk in response.iter_content(chunk_size=1 if self.http_stream else 1024):
            assert time.monotonic() < deadline, "Structured-output response timed out"
            content.extend(chunk)
            assert (
                len(content) <= 64 * 1024
            ), "Unexpectedly large structured-output response"
        return content.decode("utf-8")


@dataclass
class StructuredOutputErrorPayload(_StructuredOutputPayload):
    worker_port_envs: tuple[str, ...] = field(kw_only=True)
    _worker_urls: tuple[str, ...] = field(default=(), init=False, repr=False)
    _before: float | None = field(default=None, init=False, repr=False)

    def bind_environment(self, env: dict[str, str]) -> None:
        self._worker_urls = tuple(
            f"http://{self.host}:{int(env[key])}/metrics"
            for key in self.worker_port_envs
        )

    def _worker_requests(self) -> float:
        assert self._worker_urls, "Worker metrics ports were not bound"
        count = 0.0
        for url in self._worker_urls:
            with requests.get(url, timeout=2) as response:
                response.raise_for_status()
                count += sum_metric_samples(
                    response.text,
                    "dynamo_component_requests_total",
                    {"dynamo_endpoint": "generate"},
                )
        return count

    def before_request(self) -> None:
        self._before = self._worker_requests()

    def accepts_status_code(self, status_code: int) -> bool:
        """Prefill can reject a stream before its HTTP headers are committed."""
        return status_code == 400 or (self.http_stream and status_code == 200)

    def validate(self, response: requests.Response, content: str) -> None:
        assert self.accepts_status_code(response.status_code), response.status_code
        assert not any(
            private in content
            for private in ("private_status", "/project/", "/private/")
        ), content
        if response.status_code == 200:
            chunks = parse_sse_json_chunks(content)
            errors = [chunk["error"] for chunk in chunks if "error" in chunk]
            assert len(errors) == 1, content
            error = errors[0]
            error_type = "invalid_request_error"
            for chunk in chunks:
                for choice in chunk.get("choices", []):
                    assert choice.get("finish_reason") is None, content
                    delta = choice.get("delta", {})
                    assert not any(
                        delta.get(key)
                        for key in (
                            "content",
                            "reasoning_content",
                            "reasoning",
                            "tool_calls",
                            "function_call",
                        )
                    ), content
        else:
            error = json.loads(content)
            error_type = "Bad Request"

        assert error == {
            "message": "Invalid JSON schema: enum array must not be empty",
            "type": error_type,
            "code": 400,
        }, error
        assert self._before is not None, "before_request must run first"
        assert (
            self._worker_requests() > self._before
        ), "The invalid schema was rejected before reaching a worker"


class StructuredOutputValidPayload(_StructuredOutputPayload):
    def validate(self, response: requests.Response, content: str) -> None:
        if self.http_stream:
            chunks = parse_sse_json_chunks(content)
            assert not any("error" in chunk for chunk in chunks), content
            choices = [
                choice for chunk in chunks for choice in chunk.get("choices", [])
            ]
            assert any(
                choice.get("finish_reason") == "stop" for choice in choices
            ), content
            text = "".join(
                choice.get("delta", {}).get("content") or "" for choice in choices
            )
        else:
            result = json.loads(content)
            assert result["choices"][0]["finish_reason"] == "stop", result
            text = result["choices"][0]["message"]["content"]
        assert json.loads(text) == {"private_status": "ok"}, text


def structured_output_validation_payloads(
    worker_port_envs: tuple[str, ...],
) -> list[BasePayload]:
    """Pair each xgrammar worker rejection with a valid constrained request."""
    payloads: list[BasePayload] = []
    for streaming in (False, True):
        body = {
            "messages": [{"role": "user", "content": "Return a JSON object."}],
            "temperature": 0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": streaming,
        }
        schema = {
            "type": "object",
            "properties": {"private_status": {"type": "string", "enum": []}},
            "required": ["private_status"],
            "additionalProperties": False,
        }
        payloads.append(
            StructuredOutputErrorPayload(
                body={**body, "guided_json": schema},
                expected_response=[],
                expected_log=[],
                expected_status_code=200 if streaming else 400,
                http_stream=streaming,
                worker_port_envs=worker_port_envs,
            )
        )
        valid_schema = {
            **schema,
            "properties": {"private_status": {"type": "string", "enum": ["ok"]}},
        }
        payloads.append(
            StructuredOutputValidPayload(
                body={**body, "guided_json": valid_schema},
                expected_response=[],
                expected_log=[],
                http_stream=streaming,
            )
        )
    return payloads
