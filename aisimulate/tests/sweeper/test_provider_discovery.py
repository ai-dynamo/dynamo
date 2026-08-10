# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for lazy adapter entry-point discovery."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aisimulate.sweeper.discovery import (
    DuplicateProviderError,
    ProviderABIError,
    ProviderFactoryError,
    ProviderLoadError,
    ProviderNotFoundError,
    resolve_providers,
)
from aisimulate.sweeper.provider import (
    API_VERSION,
    AdapterReplaySpec,
    AdapterSearchPlan,
)


class _Provider:
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
    return lambda: _Provider(name)


def test_only_selected_entry_point_is_loaded():
    planner = _EntryPoint(
        "dynamo.planner",
        "dynamo.planner.simulation:create_provider",
        _factory("dynamo.planner"),
    )
    router = _EntryPoint(
        "dynamo.router",
        "dynamo.router.simulation:create_provider",
        _factory("dynamo.router"),
    )

    resolved = resolve_providers(["dynamo.router"], entry_points=[planner, router])

    assert list(resolved) == ["dynamo.router"]
    assert router.loads == 1
    assert planner.loads == 0


def test_injected_adapter_has_precedence_without_loading_installed_provider():
    installed = _EntryPoint("example", "installed:create", _factory("example"))
    injected = _Provider("example")

    resolved = resolve_providers(
        ["example"], injected={"example": injected}, entry_points=[installed, installed]
    )

    assert resolved == {"example": injected}
    assert installed.loads == 0


def test_repeated_config_name_is_resolved_once():
    entry_point = _EntryPoint("example", "example:create", _factory("example"))

    resolved = resolve_providers(["example", "example"], entry_points=[entry_point])

    assert list(resolved) == ["example"]
    assert entry_point.loads == 1


def test_missing_provider_lists_available_names_without_loading_them():
    available = _EntryPoint("available", "available:create", _factory("available"))

    with pytest.raises(
        ProviderNotFoundError, match="missing.*available providers: available"
    ):
        resolve_providers(["missing"], entry_points=[available])

    assert available.loads == 0


def test_duplicate_installed_adapter_is_rejected_before_loading():
    first = _EntryPoint("example", "first:create", _factory("example"), dist="first")
    second = _EntryPoint("example", "second:create", _factory("example"), dist="second")

    with pytest.raises(DuplicateProviderError, match="first:create.*second:create"):
        resolve_providers(["example"], entry_points=[first, second])

    assert first.loads == second.loads == 0


def test_provider_import_failure_is_distinct():
    entry_point = _EntryPoint(
        "example", "broken:create", ImportError("optional dependency missing")
    )

    with pytest.raises(
        ProviderLoadError, match="failed to load.*optional dependency missing"
    ):
        resolve_providers(["example"], entry_points=[entry_point])


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (object(), "must resolve to a callable factory"),
        (lambda: (_ for _ in ()).throw(RuntimeError("factory boom")), "factory boom"),
    ],
)
def test_invalid_factory_errors_are_distinct(provider, message):
    entry_point = _EntryPoint("example", "example:create", provider)

    with pytest.raises(ProviderFactoryError, match=message):
        resolve_providers(["example"], entry_points=[entry_point])


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (_Provider("wrong-name"), "returned name"),
        (
            type("OldProvider", (_Provider,), {"api_version": 0})("example"),
            "API version",
        ),
        (
            type("FloatVersionProvider", (_Provider,), {"api_version": 1.0})("example"),
            "API version",
        ),
        (
            type(
                "IncompleteProvider",
                (),
                {"name": "example", "api_version": API_VERSION},
            )(),
            "required callable",
        ),
    ],
)
def test_invalid_provider_abi_is_rejected(provider, message):
    entry_point = _EntryPoint("example", "example:create", lambda: provider)

    with pytest.raises(ProviderABIError, match=message):
        resolve_providers(["example"], entry_points=[entry_point])
