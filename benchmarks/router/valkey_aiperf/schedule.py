# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from typing import Any


def arms_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the planned arms, including the policy-matched authority control."""

    if args.arm == "matched":
        return ("inprocess_immediate", "valkey_ha")
    if args.arm != "both":
        return (args.arm,)
    if args.valkey_authoritative_admission:
        return ("inprocess", "inprocess_immediate", "valkey_ha")
    return ("inprocess", "valkey_ha")


def event_plane_for_arm(args: argparse.Namespace, arm: str) -> str:
    """Hold the generic event plane constant across every comparison arm."""

    del arm
    return args.event_plane


def build_arm_schedule(arms: tuple[str, ...], runs: int) -> list[dict[str, Any]]:
    """Build a position-balanced, inspectable arm schedule.

    Three-arm authoritative comparisons use every permutation once before a
    cyclic continuation. Thus six repetitions balance both ordinal position
    and immediate predecessor, while nine repetitions put every arm in every
    position three times. Two-arm comparisons retain the conventional AB/BA
    alternation.
    """

    if not arms:
        raise ValueError("at least one benchmark arm is required")
    if runs < 1:
        raise ValueError("runs must be at least one")

    if len(arms) == 3:
        a, b, c = arms
        first_six = (
            (a, b, c),
            (b, c, a),
            (c, a, b),
            (c, b, a),
            (b, a, c),
            (a, c, b),
        )

        def order_for_run(run_number: int) -> tuple[str, ...]:
            if run_number <= len(first_six):
                return first_six[run_number - 1]
            offset = (run_number - len(first_six) - 1) % len(arms)
            return arms[offset:] + arms[:offset]

        method = "all_six_permutations_then_cyclic_rotation"
    elif len(arms) == 2:

        def order_for_run(run_number: int) -> tuple[str, ...]:
            return arms if run_number % 2 else tuple(reversed(arms))

        method = "alternating_forward_reverse"
    else:

        def order_for_run(run_number: int) -> tuple[str, ...]:
            offset = (run_number - 1) % len(arms)
            return arms[offset:] + arms[:offset]

        method = "cyclic_rotation"

    schedule: list[dict[str, Any]] = []
    sample_index = 0
    for run_number in range(1, runs + 1):
        for ordinal, arm in enumerate(order_for_run(run_number), start=1):
            sample_index += 1
            schedule.append(
                {
                    "sample_index": sample_index,
                    "run": run_number,
                    "ordinal": ordinal,
                    "arm": arm,
                    "method": method,
                }
            )
    return schedule
