# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared `Actuator` test double.

Exists to enforce one rule: **a double standing in for an `Actuator` must
never leave `apply_cap` bare.**

`PowerAgent._reconcile_gpu` ends in `return result.ok`, and `reconcile_once`
folds those booleans into the whole-cycle enforcement value that backs
`/readyz`. A bare `MagicMock().apply_cap(...)` returns a `MagicMock`, which is
truthy — so a cycle driven by a bare double returns a truthy `MagicMock` rather
than `True`. An `assertTrue(...)` on it passes while asserting nothing, and an
`assertIs(..., True)` fails for a reason that has nothing to do with the code
under test. This is the one migration class that fails SILENTLY, which is why
the double is centralized here rather than repeated inline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from actuator import CapWriteResult

# Any plausible in-range wattage; tests that care about the value set their own.
DEFAULT_EFFECTIVE_W = 300


def actuator_double(
    effective_w: int = DEFAULT_EFFECTIVE_W, ok: bool = True, **kwargs
) -> MagicMock:
    """Return a `MagicMock` actuator whose `apply_cap` yields a real
    `CapWriteResult`.

    `ok=False` models a cap write that was skipped or rejected — the case that
    must drive the cycle's enforcement boolean false.
    """
    actuator = MagicMock(**kwargs)
    actuator.apply_cap.return_value = CapWriteResult(effective_w, ok)
    return actuator
