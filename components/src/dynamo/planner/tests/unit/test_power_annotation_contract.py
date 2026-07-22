# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test: the planner and Power Agent per-GPU power-limit annotation
keys must be byte-identical.

The two constants are intentionally duplicated rather than shared as a package
import: the Power Agent image copies three standalone files and does not install
the ``dynamo`` Python package, so a runtime dependency only to share one string
is not justified (see power-budget-architecture-rework.md §1.1). This test is
the safety net that keeps them in sync.
"""

import re
from pathlib import Path

import pytest

from dynamo.planner.monitoring.dgd_services import POWER_ANNOTATION_KEY

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

_LITERAL = re.compile(r'^POWER_ANNOTATION_KEY\s*=\s*"([^"]+)"', re.MULTILINE)


def _find_power_agent_source() -> Path:
    """Walk up from this test file to locate deploy/power-agent/power_agent.py."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "deploy" / "power-agent" / "power_agent.py"
        if candidate.is_file():
            return candidate
    raise AssertionError(
        "Could not locate deploy/power-agent/power_agent.py from "
        f"{Path(__file__).resolve()}"
    )


def test_planner_and_power_agent_annotation_keys_match():
    source = _find_power_agent_source().read_text(encoding="utf-8")
    match = _LITERAL.search(source)
    assert match is not None, (
        "POWER_ANNOTATION_KEY literal not found in power_agent.py; the contract "
        "test can no longer verify the duplicated constants agree."
    )
    agent_key = match.group(1)
    assert agent_key == POWER_ANNOTATION_KEY, (
        "Power Agent and planner per-GPU power-limit annotation keys diverged: "
        f"power_agent.py={agent_key!r} planner={POWER_ANNOTATION_KEY!r}. Keep the "
        "two literals identical."
    )


def test_annotation_key_value_is_the_expected_literal():
    # Pin the literal so a rename is a deliberate, reviewed change on both sides.
    assert POWER_ANNOTATION_KEY == "dynamo.nvidia.com/gpu-power-limit"
