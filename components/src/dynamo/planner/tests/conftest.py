# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest.py for planner tests to disable automatic test logging.
This overrides the autouse logger fixture from the parent conftest.py.
It also holds the stand-in for the native AIS engine model.
"""

import pytest


@pytest.fixture(autouse=True)
def logger(request):
    """Dummy logger fixture that does nothing - overrides the parent one."""
    yield


@pytest.fixture
def fake_engine_factory(monkeypatch):
    """Make the perf model adapter build ``next_model`` and no native engine model."""

    class FakeEngineFactory:
        next_model = None
        last_kwargs = None

        @classmethod
        def best_available(cls, **kwargs):
            assert cls.next_model is not None
            cls.last_kwargs = kwargs
            return cls.next_model

    # The class is named as text: this conftest is also loaded where
    # AISimulate is missing, and the adapter module imports it.
    monkeypatch.setattr(
        "dynamo.planner.core.perf_model.ais_adapter.AISCoreEnginePerfModel",
        FakeEngineFactory,
    )
    return FakeEngineFactory
