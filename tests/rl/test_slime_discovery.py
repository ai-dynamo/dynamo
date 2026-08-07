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

"""Unit coverage for the Slime example's Dynamo worker discovery function."""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType

import pytest
from typing_extensions import Self

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
]


class FakeResponse:
    """Minimal context-managed HTTP response for discovery unit coverage."""

    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


@pytest.fixture
def discovery_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "rl"
        / "slime"
        / "dynamo_discovery.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "slime_dynamo_discovery",
        module_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load discovery module from {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_discovery_returns_only_ready_engine_control_urls(
    discovery_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.43", 9090)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.42", 9090)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.42", 9090)),
    ]
    responses = {
        "http://10.0.0.42:9090/health": FakeResponse({"status": "ready"}),
        "http://10.0.0.43:9090/health": FakeResponse({"status": "notready"}),
    }

    monkeypatch.setenv("DYNAMO_CONTROL_SERVICE", "slime-sglang-control")
    monkeypatch.setattr(
        discovery_module.socket,
        "getaddrinfo",
        lambda *args, **kwargs: addresses,
    )
    monkeypatch.setattr(
        discovery_module,
        "urlopen",
        lambda request, timeout: responses[request.full_url],
    )

    assert discovery_module.discover_engine_control_urls(None) == [
        "http://10.0.0.42:9090/engine"
    ]
