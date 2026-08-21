# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component wrappers.

A component owns its wire interface *and* the waiting policy for it. Tests call
``frontend.query(...)``; they never build an HTTP request, and they never sleep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .transport import Http, HttpError


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
