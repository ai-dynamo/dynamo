# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable HTTP response predicates for test process readiness checks."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import requests


def _json_mapping(response: requests.Response) -> Mapping[str, Any] | None:
    if response.status_code != 200:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    return payload if isinstance(payload, Mapping) else None


def check_http_ok(response: requests.Response) -> bool:
    """Return whether a response has a successful HTTP status."""
    return response.status_code == 200


def check_health_ready(response: requests.Response) -> bool:
    """Return whether an HTTP health response reports a ready component."""
    payload = _json_mapping(response)
    return payload is not None and payload.get("status") == "ready"


def models_available(response: requests.Response) -> bool:
    """Return whether a model-list response contains at least one model."""
    payload = _json_mapping(response)
    if payload is None:
        return False

    models = payload.get("data")
    return isinstance(models, list) and bool(models)


def model_registered(response: requests.Response, *, model: str) -> bool:
    """Return whether a model-list response contains the requested model ID."""
    payload = _json_mapping(response)
    if payload is None:
        return False

    models = payload.get("data")
    if not isinstance(models, list):
        return False
    return any(isinstance(item, Mapping) and item.get("id") == model for item in models)


def check_model_registered(response: requests.Response, *, model: str) -> bool:
    """Compatibility model check with the established stabilization delay."""
    if not model_registered(response, model=model):
        return False

    time.sleep(1)
    return True


def check_models_api(response: requests.Response) -> bool:
    """Compatibility readiness check with the existing stabilization delay."""
    if not models_available(response):
        return False

    # Keep the established post-registration delay until the completions 404
    # race is removed from the frontend.
    time.sleep(1)
    return True


def check_health_generate(response: requests.Response) -> bool:
    """Return whether a health response advertises a generate endpoint."""
    payload = _json_mapping(response)
    if payload is None:
        return False

    endpoints = payload.get("endpoints") or []
    if any(
        isinstance(endpoint, str) and "generate" in endpoint for endpoint in endpoints
    ):
        time.sleep(1)
        return True

    instances = payload.get("instances") or []
    if any(
        isinstance(instance, Mapping) and instance.get("endpoint") == "generate"
        for instance in instances
    ):
        time.sleep(1)
        return True

    return False
