# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend import main
from dynamo.llm import SystemRoute


class FakeEntryPoint:
    def __init__(self, name, value):
        self.name = name
        self._value = value

    def load(self):
        return self._value


def test_load_system_route_extensions_by_entry_point(monkeypatch):
    def handler(ctx):
        return {"ready": ctx.is_ready()}

    def provider():
        return [SystemRoute("GET", "/test/ready", handler)]

    monkeypatch.setattr(
        main,
        "_system_route_extension_entry_points",
        lambda: [FakeEntryPoint("test-extension", provider)],
    )

    routes = main.load_system_route_extensions(["test-extension"])

    assert len(routes) == 1
    assert routes[0].method == "GET"
    assert routes[0].path == "/test/ready"


def test_load_system_route_extensions_accepts_single_route(monkeypatch):
    def handler(ctx):
        return {"ready": ctx.is_ready()}

    def provider():
        return SystemRoute("GET", "/test/ready", handler)

    monkeypatch.setattr(
        main,
        "_system_route_extension_entry_points",
        lambda: [FakeEntryPoint("test-extension", provider)],
    )

    routes = main.load_system_route_extensions(["test-extension"])

    assert len(routes) == 1
    assert routes[0].path == "/test/ready"


def test_load_system_route_extensions_rejects_unknown_name(monkeypatch):
    monkeypatch.setattr(
        main,
        "_system_route_extension_entry_points",
        lambda: [FakeEntryPoint("known-extension", lambda: [])],
    )

    with pytest.raises(ValueError, match="Unknown system route extension 'missing'"):
        main.load_system_route_extensions(["missing"])


def test_load_system_route_extensions_rejects_ambiguous_name(monkeypatch):
    monkeypatch.setattr(
        main,
        "_system_route_extension_entry_points",
        lambda: [
            FakeEntryPoint("duplicate", lambda: []),
            FakeEntryPoint("duplicate", lambda: []),
        ],
    )

    with pytest.raises(ValueError, match="Ambiguous system route extension 'duplicate'"):
        main.load_system_route_extensions(["duplicate"])


def test_load_system_route_extensions_rejects_non_route(monkeypatch):
    def provider():
        return [{"method": "GET", "path": "/not-a-system-route"}]

    monkeypatch.setattr(
        main,
        "_system_route_extension_entry_points",
        lambda: [FakeEntryPoint("bad-extension", provider)],
    )

    with pytest.raises(TypeError, match="expected dynamo.llm.SystemRoute"):
        main.load_system_route_extensions(["bad-extension"])
