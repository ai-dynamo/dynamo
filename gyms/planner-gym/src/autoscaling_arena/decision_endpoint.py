# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit transport selection for System One decision models."""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlsplit


def decision_base_url(
    *,
    provider: str = "typesafe",
    base_url: str | None = None,
    api_key_env: str | None = None,
    model: str = "jev-1.13.0",
) -> str:
    """Validate before creating a client; never forward TypeSafe credentials."""
    if provider == "typesafe":
        if base_url is not None or api_key_env is not None:
            raise ValueError("typesafe uses its fixed endpoint and TYPESAFE_API_KEY")
        return "https://api.typesafe.ai"
    if provider != "self_hosted":
        raise ValueError("provider must be typesafe or self_hosted")
    if (
        not isinstance(model, str)
        or not model.strip()
        or model.lower().startswith("jev-")
    ):
        raise ValueError("self_hosted requires an explicit independent model name")
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip():
        raise ValueError("self_hosted requires an explicit base_url")
    try:
        url = urlsplit(base_url)
        port = url.port
    except ValueError as exc:
        raise ValueError("invalid decision endpoint URL") from exc
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
        or port == 0
        or any(char.isspace() for char in base_url)
    ):
        raise ValueError(
            "base_url must be an HTTP(S) origin without credentials or a path"
        )
    if url.hostname.rstrip(".").lower() in {"typesafe.ai", "api.typesafe.ai"}:
        raise ValueError("self_hosted cannot use the paid TypeSafe endpoint")
    if api_key_env is not None and (
        not isinstance(api_key_env, str)
        or re.fullmatch(r"[A-Z_][A-Z0-9_]*", api_key_env) is None
        or api_key_env == "TYPESAFE_API_KEY"
    ):
        raise ValueError("api_key_env must name a separate environment variable")
    return base_url.rstrip("/")


def decision_http_client(*, timeout_s: float, **endpoint: Any) -> Any:
    """Keep httpx optional for config-only installations."""
    import httpx

    origin = decision_base_url(**endpoint)
    provider = endpoint.get("provider", "typesafe")
    key_env = (
        "TYPESAFE_API_KEY" if provider == "typesafe" else endpoint.get("api_key_env")
    )
    headers = {}
    if key_env is not None:
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise ValueError(f"decision endpoint requires {key_env} in the environment")
        headers["Authorization"] = f"Bearer {key}"
    return httpx.AsyncClient(
        base_url=origin,
        headers=headers,
        timeout=timeout_s,
        follow_redirects=False,
        # Internal endpoints must not be routed through ambient HTTP proxies.
        trust_env=provider == "typesafe",
    )
