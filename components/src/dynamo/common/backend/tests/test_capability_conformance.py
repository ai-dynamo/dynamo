# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conformance check: bare EngineConfig declares no capabilities.

The registry-to-enum lock-step is enforced Rust-side via the
[`every_forwarded_field_has_capability`] schema test; this file
only covers the Python boundary.
"""

from __future__ import annotations

import pytest

backend = pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_default_engine_config_declares_no_capabilities():
    cfg = backend.EngineConfig(model="m")
    assert cfg.capabilities == []
