# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect a hypothetical settled allocation without contacting a backend."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dynamo.global_planner.allocation import DEFAULT_MAX_SEARCH_STEPS, plan_allocation
from dynamo.global_planner.allocation_policy import AllocationPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, required=True, help="Pool policy JSON file"
    )
    parser.add_argument(
        "--allocations",
        type=Path,
        required=True,
        help="JSON object mapping every pool ID to its committed GPU allocation",
    )
    parser.add_argument("--pool", required=True, help="Requesting pool ID")
    parser.add_argument(
        "--replicas", type=int, required=True, help="Desired replica count"
    )
    parser.add_argument(
        "--unavailable-pool",
        action="append",
        default=[],
        help="Pool excluded from changes (repeat for multiple pools)",
    )
    parser.add_argument(
        "--max-search-steps", type=int, default=DEFAULT_MAX_SEARCH_STEPS
    )
    args = parser.parse_args(argv)

    try:
        policy = AllocationPolicy.model_validate_json(
            args.policy.read_text(encoding="utf-8")
        )
        allocations = json.loads(args.allocations.read_text(encoding="utf-8"))
        decision = plan_allocation(
            policy,
            allocations,
            args.pool,
            args.replicas,
            unavailable_pool_ids=args.unavailable_pool,
            max_search_steps=args.max_search_steps,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    return 0 if decision.admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
