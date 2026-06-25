# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for compliance.policy.validate exception behavior."""

from __future__ import annotations

from compliance.policy.validate import Policy, validate_row


def _base_policy(*, exceptions=()):
    return Policy(
        allow=frozenset({"MIT", "Apache-2.0"}),
        deny=frozenset({"GPL-3.0-only"}),
        unknown_action="deny",
        copyleft_action="deny",
        exceptions=tuple(exceptions),
    )


def test_unknown_denied_without_exception():
    policy = _base_policy()
    violation = validate_row(policy, "python", "oneccl", "2021.17.2", "UNKNOWN")
    assert violation is not None
    assert "UNKNOWN" in violation.reason


def test_unknown_allowed_with_matching_exception():
    policy = _base_policy(
        exceptions=(
            {
                "type": "python",
                "name": "oneccl",
                "allow": ["UNKNOWN"],
                "reason": "temporary local xpu exception",
            },
        )
    )
    violation = validate_row(policy, "python", "oneccl", "2021.17.2", "UNKNOWN")
    assert violation is None


def test_unknown_exception_is_package_scoped():
    policy = _base_policy(
        exceptions=(
            {
                "type": "python",
                "name": "oneccl",
                "allow": ["UNKNOWN"],
                "reason": "temporary local xpu exception",
            },
        )
    )
    violation = validate_row(
        policy,
        "python",
        "oneccl-devel",
        "2021.17.2",
        "UNKNOWN",
    )
    assert violation is not None
