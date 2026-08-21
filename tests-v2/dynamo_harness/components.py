# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component wrappers.

A component owns its wire interface *and* the waiting policy for it. Tests call
``frontend.query(...)``; they never build an HTTP request, and they never sleep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .transport import Http, HttpError


@dataclass
class StreamResult:
    """A streamed chat response, reassembled.

    Tool calls arrive as deltas keyed by index -- id and name in one chunk,
    argument fragments across many. Reassembling that is protocol detail and
    belongs here, not in a test.
    """

    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    model: str = ""
    chunks: int = 0
    ttft_ms: float = 0.0
    raw_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def assistant_message(self) -> Dict[str, Any]:
        """This turn, shaped for sending back as conversation history."""
        return {
            "role": "assistant",
            "content": self.content or None,
            "tool_calls": self.tool_calls,
        }


@dataclass
class Frontend:
    """The OpenAI-compatible HTTP surface."""

    http: Http
    model: Optional[str] = None

    # -- readiness ---------------------------------------------------------
    def wait_until_serving(self, timeout: float = 900.0, interval: float = 5.0) -> str:
        """Block until a model is registered, then remember it.

        Readiness only. It deliberately does not assert anything about output
        quality: a component can be serving and still be wrong.
        """
        deadline = time.monotonic() + timeout
        last = "no response yet"
        while time.monotonic() < deadline:
            try:
                names = self.models()
                if names:
                    self.model = self.model or names[0]
                    return self.model
                last = "/v1/models returned an empty list"
            except Exception as exc:  # connection refused while starting
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(interval)
        raise TimeoutError(
            f"frontend at {self.http.base_url} did not serve a model within "
            f"{timeout:.0f}s (last: {last})"
        )

    # -- inference ---------------------------------------------------------
    def models(self) -> List[str]:
        body = self.http.get_json("/v1/models")
        return [entry["id"] for entry in body.get("data", [])]

    def chat(self, prompt: str, **kw: Any) -> Dict[str, Any]:
        """Raw chat-completions response, for tests that need the envelope."""
        body: Dict[str, Any] = {
            "model": kw.pop("model", None) or self._model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kw.pop("max_tokens", 300),
        }
        body.update(kw)
        return self.http.post_json("/v1/chat/completions", body)

    def query(self, prompt: str, **kw: Any) -> str:
        """The model's answer as text.

        Falls back to ``reasoning_content`` when a reasoning deployment spends
        its whole budget inside the think block and returns empty ``content``.
        """
        message = (self.chat(prompt, **kw).get("choices") or [{}])[0].get("message", {})
        return (message.get("content") or "").strip() or (
            message.get("reasoning_content") or ""
        )

    def complete(self, prompt: str, **kw: Any) -> str:
        body: Dict[str, Any] = {
            "model": kw.pop("model", None) or self._model(),
            "prompt": prompt,
            "max_tokens": kw.pop("max_tokens", 64),
        }
        body.update(kw)
        response = self.http.post_json("/v1/completions", body)
        return (response.get("choices") or [{}])[0].get("text", "")

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kw: Any,
    ) -> StreamResult:
        """Stream a chat completion and return the reassembled result."""
        body: Dict[str, Any] = {
            "model": kw.pop("model", None) or self._model(),
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            body["tools"] = tools
        body.update(kw)

        result = StreamResult()
        by_index: Dict[int, Dict[str, Any]] = {}
        started = time.monotonic()
        content: List[str] = []
        reasoning: List[str] = []

        for chunk in self.http.post_sse("/v1/chat/completions", body):
            result.raw_chunks.append(chunk)
            result.chunks += 1
            if result.chunks == 1:
                result.ttft_ms = (time.monotonic() - started) * 1000.0
            result.model = chunk.get("model") or result.model

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for call in delta.get("tool_calls") or []:
                    self._merge_tool_delta(by_index, call)
                if choice.get("finish_reason"):
                    result.finish_reason = choice["finish_reason"]

        result.content = "".join(content)
        result.reasoning_content = "".join(reasoning)
        result.tool_calls = [by_index[i] for i in sorted(by_index)]
        return result

    @staticmethod
    def _merge_tool_delta(
        by_index: Dict[int, Dict[str, Any]], call: Dict[str, Any]
    ) -> None:
        index = call.get("index", 0)
        entry = by_index.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if call.get("id"):
            if entry["id"] and entry["id"] != call["id"]:
                raise AssertionError(
                    f"tool call id changed within index {index}: "
                    f"{entry['id']} -> {call['id']}"
                )
            entry["id"] = call["id"]
        if call.get("type"):
            entry["type"] = call["type"]
        function = call.get("function") or {}
        if function.get("name"):
            if (
                entry["function"]["name"]
                and entry["function"]["name"] != function["name"]
            ):
                raise AssertionError(
                    f"tool name changed within index {index}: "
                    f"{entry['function']['name']} -> {function['name']}"
                )
            entry["function"]["name"] = function["name"]
        if function.get("arguments"):
            entry["function"]["arguments"] += function["arguments"]

    def metrics(self) -> str:
        return self.http.get_text("/metrics")

    def metric_samples(self, prefix: str) -> Dict[str, float]:
        """Every sample whose metric name starts with ``prefix``.

        Returned keyed by the full sample line's metric name, values summed
        across label sets. Tests assert on *a* counter having advanced rather
        than on an exact metric name, which differs between builds.
        """
        totals: Dict[str, float] = {}
        for line in self.metrics().splitlines():
            if not line or line.startswith("#") or not line.startswith(prefix):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            totals[name] = totals.get(name, 0.0) + value
        return totals

    def expect_rejected(self, path: str, body: Dict[str, Any]) -> int:
        """POST expecting a 4xx; returns the status code."""
        try:
            self.http.post_json(path, body)
        except HttpError as exc:
            return exc.status
        raise AssertionError(f"{path} accepted a request that should be rejected")

    def _model(self) -> str:
        if not self.model:
            raise RuntimeError(
                "no model resolved; call wait_until_serving() or pass model=..."
            )
        return self.model
