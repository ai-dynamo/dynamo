# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a component is reached. Independent of how it was deployed."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status = status
        self.body = body


@dataclass
class Http:
    """Minimal HTTP transport. stdlib only, so the harness imports without
    ai-dynamo installed and can run from a bare pytest container."""

    base_url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 300.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def _request(self, path: str, data: Optional[bytes], content_type: Optional[str]):
        url = f"{self.base_url}{path}"
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            raise HttpError(exc.code, exc.read().decode(errors="replace"), url) from exc

    def get_text(self, path: str) -> str:
        return self._request(path, None, None)

    def get_json(self, path: str) -> Any:
        return json.loads(self.get_text(path))

    def post_json(self, path: str, body: Dict[str, Any]) -> Any:
        raw = self._request(path, json.dumps(body).encode(), "application/json")
        return json.loads(raw)

    def post_sse(self, path: str, body: Dict[str, Any]):
        """Yield decoded SSE ``data:`` payloads, skipping the [DONE] sentinel."""
        url = f"{self.base_url}{path}"
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    yield json.loads(payload)
        except urllib.error.HTTPError as exc:
            raise HttpError(exc.code, exc.read().decode(errors="replace"), url) from exc
