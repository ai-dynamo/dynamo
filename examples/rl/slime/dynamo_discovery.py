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

"""Discover ready Dynamo SGLang worker control endpoints for Slime."""

from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


def _read_port() -> int:
    port = int(os.environ.get("DYNAMO_CONTROL_PORT", "9090"))
    if not 0 < port < 65536:
        raise ValueError("DYNAMO_CONTROL_PORT must be between 1 and 65535")
    return port


def _read_timeout() -> float:
    timeout = float(os.environ.get("DYNAMO_CONTROL_HEALTH_TIMEOUT_SECONDS", "5"))
    if timeout <= 0:
        raise ValueError("DYNAMO_CONTROL_HEALTH_TIMEOUT_SECONDS must be positive")
    return timeout


def _url(host: str, port: int, path: str) -> str:
    host_part = f"[{host}]" if ":" in host else host
    return f"http://{host_part}:{port}{path}"


def _resolve_hosts(service: str, port: int) -> list[str]:
    addresses: Iterable[tuple[object, ...]] = socket.getaddrinfo(
        service,
        port,
        type=socket.SOCK_STREAM,
    )
    return sorted({address[4][0] for address in addresses})


def _is_ready(host: str, port: int, timeout: float) -> bool:
    health_url = _url(host, port, "/health")
    request = Request(health_url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        LOGGER.info("Skipping unavailable Dynamo worker %s: %s", health_url, error)
        return False

    return isinstance(payload, dict) and payload.get("status") == "ready"


def discover_engine_control_urls(_args: object) -> list[str]:
    """Return ready Dynamo worker control bases for Slime's external rollout.

    Slime calls this synchronous function at startup and before each weight
    update. The Kubernetes headless Service resolves to the current worker Pod
    IPs; checking the Dynamo system server's ``/health`` route excludes workers
    that are present in DNS but not ready to receive engine administration.
    """

    service = os.environ.get("DYNAMO_CONTROL_SERVICE", "slime-sglang-control")
    port = _read_port()
    timeout = _read_timeout()
    control_urls = [
        _url(host, port, "/engine")
        for host in _resolve_hosts(service, port)
        if _is_ready(host, port, timeout)
    ]
    if not control_urls:
        raise RuntimeError(
            f"No ready Dynamo SGLang workers found through {service}:{port}/health"
        )
    return control_urls
