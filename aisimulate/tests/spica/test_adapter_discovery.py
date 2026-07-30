# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for lazy adapter entry-point discovery."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aisimulate.spica.adapter import API_VERSION, AdapterReplaySpec, AdapterSearchPlan
from aisimulate.spica.discovery import (
    AdapterABIError,
    AdapterFactoryError,
    AdapterLoadError,
    AdapterNotFoundError,
    DuplicateAdapterError,
    resolve_adapters,
)


class _Adapter:
    api_version = API_VERSION

    def __init__(self, name: str):
        self.name = name

    def generate_search_space(self, search_spec, context):
        return AdapterSearchPlan()

    def materialize_replay(self, plan, selection, context):
        return AdapterReplaySpec()


@dataclass
class _EntryPoint:
    name: str
    value: str
    provider: object
    dist: str = "test-distribution"
    loads: int = 0

    def load(self):
        self.loads += 1
        if isinstance(self.provider, Exception):
            raise self.provider
        return self.provider


def _factory(name: str):
    return lambda: _Adapter(name)


def test_only_selected_entry_point_is_loaded():
    planner = _EntryPoint(
        "dynamo.planner",
        "dynamo.planner.simulation:create_adapter",
        _factory("dynamo.planner"),
    )
    router = _EntryPoint(
        "dynamo.router",
        "dynamo.router.simulation:create_adapter",
        _factory("dynamo.router"),
    )

    resolved = resolve_adapters(["dynamo.router"], entry_points=[planner, router])

    assert list(resolved) == ["dynamo.router"]
    assert router.loads == 1
    assert planner.loads == 0


def test_injected_adapter_has_precedence_without_loading_installed_provider():
    installed = _EntryPoint("example", "installed:create", _factory("example"))
    injected = _Adapter("example")

    resolved = resolve_adapters(
        ["example"], injected={"example": injected}, entry_points=[installed, installed]
    )

    assert resolved == {"example": injected}
    assert installed.loads == 0


def test_repeated_config_name_is_resolved_once():
    entry_point = _EntryPoint("example", "example:create", _factory("example"))

    resolved = resolve_adapters(["example", "example"], entry_points=[entry_point])

    assert list(resolved) == ["example"]
    assert entry_point.loads == 1


def test_missing_adapter_lists_available_names_without_loading_them():
    available = _EntryPoint("available", "available:create", _factory("available"))

    with pytest.raises(
        AdapterNotFoundError, match="missing.*available adapters: available"
    ):
        resolve_adapters(["missing"], entry_points=[available])

    assert available.loads == 0


def test_duplicate_installed_adapter_is_rejected_before_loading():
    first = _EntryPoint("example", "first:create", _factory("example"), dist="first")
    second = _EntryPoint("example", "second:create", _factory("example"), dist="second")

    with pytest.raises(DuplicateAdapterError, match="first:create.*second:create"):
        resolve_adapters(["example"], entry_points=[first, second])

    assert first.loads == second.loads == 0


def test_provider_import_failure_is_distinct():
    entry_point = _EntryPoint(
        "example", "broken:create", ImportError("optional dependency missing")
    )

    with pytest.raises(
        AdapterLoadError, match="failed to load.*optional dependency missing"
    ):
        resolve_adapters(["example"], entry_points=[entry_point])


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (object(), "must resolve to a callable factory"),
        (lambda: (_ for _ in ()).throw(RuntimeError("factory boom")), "factory boom"),
    ],
)
def test_invalid_factory_errors_are_distinct(provider, message):
    entry_point = _EntryPoint("example", "example:create", provider)

    with pytest.raises(AdapterFactoryError, match=message):
        resolve_adapters(["example"], entry_points=[entry_point])


@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        (_Adapter("wrong-name"), "returned name"),
        (type("OldAdapter", (_Adapter,), {"api_version": 0})("example"), "API version"),
        (
            type("FloatVersionAdapter", (_Adapter,), {"api_version": 1.0})("example"),
            "API version",
        ),
        (
            type(
                "IncompleteAdapter",
                (),
                {"name": "example", "api_version": API_VERSION},
            )(),
            "required callable",
        ),
    ],
)
def test_invalid_adapter_abi_is_rejected(adapter, message):
    entry_point = _EntryPoint("example", "example:create", lambda: adapter)

    with pytest.raises(AdapterABIError, match=message):
        resolve_adapters(["example"], entry_points=[entry_point])
