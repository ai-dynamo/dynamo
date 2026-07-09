# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin Python boundary for the canonical Rust Valkey JSON contract."""

from __future__ import annotations

import json
import os

from dynamo._core import normalize_router_valkey_config


def validate_router_valkey_config(raw_config: str) -> str:
    """Validate without reconstructing the Rust-owned schema in Python."""
    if not isinstance(raw_config, str):
        raise ValueError("--router-valkey-config must be a JSON string")
    try:
        normalize_router_valkey_config(raw_config)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid --router-valkey-config: {error}") from error
    return raw_config


def _legacy_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _legacy_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def legacy_tokenizer_cache_config_json() -> str:
    """Translate legacy environment names into the canonical Rust contract."""
    url = os.environ.get("DYN_TOKENIZER_CACHE_L2_URL") or None
    sentinel_urls = os.environ.get("DYN_TOKENIZER_CACHE_L2_SENTINEL_URLS") or None
    sentinel_master_name = (
        os.environ.get("DYN_TOKENIZER_CACHE_L2_SENTINEL_MASTER_NAME") or None
    )
    if bool(sentinel_urls) != bool(sentinel_master_name):
        raise ValueError(
            "DYN_TOKENIZER_CACHE_L2_SENTINEL_URLS and "
            "DYN_TOKENIZER_CACHE_L2_SENTINEL_MASTER_NAME must be configured together"
        )
    if url and sentinel_urls:
        raise ValueError(
            "DYN_TOKENIZER_CACHE_L2_URL and tokenizer Sentinel variables are mutually exclusive"
        )

    tokenizer: dict[str, object] = {
        "enabled": os.environ.get("DYN_TOKENIZER_CACHE", "1") != "0",
        "url": url,
        "sentinel_master_name": sentinel_master_name,
        "scope": os.environ.get("DYN_TOKENIZER_CACHE_L2_SCOPE", "default"),
        "key_prefix": os.environ.get(
            "DYN_TOKENIZER_CACHE_L2_KEY_PREFIX", "dynamo:tokenizer:v1"
        ),
        "ttl_seconds": _legacy_int("DYN_TOKENIZER_CACHE_L2_TTL_SECONDS", 3600),
        "timeout_ms": _legacy_int("DYN_TOKENIZER_CACHE_L2_TIMEOUT_MS", 20),
        "connection_pool_size": _legacy_int("DYN_TOKENIZER_CACHE_L2_POOL_SIZE", 8),
        "max_pending_writes": _legacy_int(
            "DYN_TOKENIZER_CACHE_L2_MAX_PENDING_WRITES", 128
        ),
        "l1_bytes": _legacy_int("DYN_TOKENIZER_CACHE_BYTES", 64 * 1024 * 1024),
        "extend": os.environ.get("DYN_TOKENIZER_CACHE_EXTEND", "1") != "0",
    }
    raw_config: dict[str, object] = {
        "allow_insecure_plaintext": _legacy_bool(
            "DYN_TOKENIZER_CACHE_L2_ALLOW_INSECURE_PLAINTEXT", False
        ),
        "tokenizer_cache": tokenizer,
    }
    if sentinel_urls and sentinel_master_name:
        urls = [part.strip() for part in sentinel_urls.split(",") if part.strip()]
        raw_config["sentinel"] = {
            "urls": urls,
            "master_name": sentinel_master_name,
            "quorum": _legacy_int(
                "DYN_TOKENIZER_CACHE_L2_SENTINEL_QUORUM", len(urls) // 2 + 1
            ),
        }
    raw = json.dumps(raw_config, separators=(",", ":"))
    validate_router_valkey_config(raw)
    return raw
