#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute total_gpu_power_limit for the Dynamo power-aware planner.

Implements the §3.3 sizing formula from powerplanner-design.md:

    usable_W = rack_capacity_W × headroom_factor − non_gpu_overhead_W

    total_gpu_power_limit = usable_W                      (single DGD)
                          = usable_W × (dgd_gpus / total_gpus)  (multi-DGD)

Usage examples
--------------
Single DGD on a 60 kW rack with 2 kW overhead and 8 H200 SXMs:

    python tools/compute_power_budget.py \\
        --rack-capacity 60000 \\
        --nodes 2 \\
        --gpus-per-node 8 \\
        --gpu-tdp 700

Multiple DGDs sharing the same rack (two DGDs, 8+4 GPUs):

    python tools/compute_power_budget.py \\
        --rack-capacity 60000 \\
        --nodes 2 \\
        --gpus-per-node 8 \\
        --gpu-tdp 700 \\
        --dgd-gpus 8 4

Customise headroom and overhead:

    python tools/compute_power_budget.py \\
        --rack-capacity 60000 \\
        --headroom-factor 0.80 \\
        --non-gpu-overhead 3000 \\
        --gpus-per-node 8 \\
        --nodes 2 \\
        --gpu-tdp 700

Output includes the YAML snippet ready to paste into your DGD config.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Known GPU TDPs (watts) for common Dynamo-supported SKUs
# ---------------------------------------------------------------------------
_KNOWN_TDPS: dict[str, int] = {
    "h100_sxm": 700,
    "h200_sxm": 700,
    "b200_sxm": 1000,
    "a100_sxm": 400,
    "a100_pcie": 300,
    "h100_pcie": 350,
}

_NON_GPU_OVERHEAD_PER_NODE_W = 1_500  # conservative default (CPU + NIC + storage)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SizingInput:
    rack_capacity_w: int
    headroom_factor: float
    non_gpu_overhead_w: int
    total_gpus: int
    gpu_tdp_w: int
    dgd_gpu_counts: list[int] = field(default_factory=list)


@dataclass
class SizingResult:
    usable_w: int
    total_gpus: int
    gpu_tdp_w: int
    # Per-DGD allocations: list of (num_gpus, total_gpu_power_limit)
    dgd_allocations: list[tuple[int, int]]

    # Sanity checks
    sum_of_dgd_limits: int
    max_sustained_draw_at_tdp: int  # all GPUs at TDP simultaneously

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("  Dynamo Power Budget Sizing Report")
        print("=" * 60)
        print(f"  Usable rack power       : {self.usable_w:>8,} W")
        print(f"  Total GPUs on rack      : {self.total_gpus:>8}")
        print(f"  GPU TDP                 : {self.gpu_tdp_w:>8,} W")
        print(f"  Max sustained (TDP×ALL) : {self.max_sustained_draw_at_tdp:>8,} W")
        print()

        if len(self.dgd_allocations) == 1:
            _, limit = self.dgd_allocations[0]
            print(f"  total_gpu_power_limit   : {limit:>8,} W")
            print()
            print("  YAML snippet:")
            print("  ─────────────────────────────────────────────────")
            print(f"  total_gpu_power_limit: {limit}")
            print()
        else:
            print(f"  {'DGD':<6} {'GPUs':>6} {'total_gpu_power_limit':>22}")
            print("  " + "-" * 38)
            for i, (gpus, limit) in enumerate(self.dgd_allocations):
                print(f"  DGD-{i:<2} {gpus:>6} {limit:>22,} W")
            print("  " + "-" * 38)
            print(f"  {'Σ':>6}       {self.sum_of_dgd_limits:>22,} W")
            print()
            print("  YAML snippets (one per DGD):")
            print("  ─────────────────────────────────────────────────")
            for i, (_, limit) in enumerate(self.dgd_allocations):
                print(f"  # DGD-{i}")
                print(f"  total_gpu_power_limit: {limit}")
                print()

        # Warnings
        if self.sum_of_dgd_limits > self.usable_w:
            print("  ⚠  WARNING: sum of DGD limits exceeds usable rack power!")
            print(f"     {self.sum_of_dgd_limits:,} W > {self.usable_w:,} W")
            print()
        if self.max_sustained_draw_at_tdp > self.usable_w * 1.15:
            print(
                "  ℹ  NOTE: GPU TDP×ALL exceeds usable power by "
                f"{self.max_sustained_draw_at_tdp - self.usable_w:,} W."
            )
            print(
                "     Power caps will prevent all GPUs from sustaining TDP "
                "simultaneously — this is expected."
            )
            print()
        print("=" * 60)

        print()
        print("  Additional planner settings to review:")
        print(f"  power_agent_safe_default_watts: {math.ceil(self.gpu_tdp_w * 0.70)}")
        print("    (= 70% of TDP; adjust for your model's idle power profile)")
        print()
        print("  Cold-start coefficient recommendations (H200 SXM, dense):")
        print("    aic_initial_c_ttft         : 1.15")
        print("    aic_initial_c_itl          : 1.15")
        print("    aic_initial_c_power_prefill: 1.05")
        print("    aic_initial_c_power_decode : 1.15")
        print()


# ---------------------------------------------------------------------------
# Core sizing logic
# ---------------------------------------------------------------------------


def compute_budget(inp: SizingInput) -> SizingResult:
    usable_w = (
        math.floor(inp.rack_capacity_w * inp.headroom_factor) - inp.non_gpu_overhead_w
    )
    if usable_w <= 0:
        raise ValueError(
            f"usable_w={usable_w} W is non-positive. "
            f"rack_capacity_w={inp.rack_capacity_w}, "
            f"headroom_factor={inp.headroom_factor}, "
            f"non_gpu_overhead_w={inp.non_gpu_overhead_w}. "
            f"Reduce overhead or increase rack capacity."
        )

    dgd_allocations: list[tuple[int, int]] = []

    if not inp.dgd_gpu_counts:
        # Single DGD: entire usable budget
        dgd_allocations = [(inp.total_gpus, usable_w)]
    else:
        if sum(inp.dgd_gpu_counts) > inp.total_gpus:
            raise ValueError(
                f"Sum of --dgd-gpus ({sum(inp.dgd_gpu_counts)}) exceeds "
                f"total GPUs on rack ({inp.total_gpus})."
            )
        for gpus in inp.dgd_gpu_counts:
            fraction = gpus / inp.total_gpus
            limit = math.floor(usable_w * fraction)
            dgd_allocations.append((gpus, limit))

    sum_of_limits = sum(lim for _, lim in dgd_allocations)
    max_tdp = inp.total_gpus * inp.gpu_tdp_w

    return SizingResult(
        usable_w=usable_w,
        total_gpus=inp.total_gpus,
        gpu_tdp_w=inp.gpu_tdp_w,
        dgd_allocations=dgd_allocations,
        sum_of_dgd_limits=sum_of_limits,
        max_sustained_draw_at_tdp=max_tdp,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_gpu_sku(value: str) -> Optional[int]:
    slug = value.lower().replace("-", "_")
    return _KNOWN_TDPS.get(slug)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compute_power_budget",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    rack = parser.add_argument_group("Rack / facility")
    rack.add_argument(
        "--rack-capacity",
        type=int,
        required=True,
        metavar="WATTS",
        help="PDU-rated rack capacity in watts (total, all phases combined).",
    )
    rack.add_argument(
        "--headroom-factor",
        type=float,
        default=0.85,
        metavar="F",
        help="Safety headroom fraction (default: 0.85). Use ≤0.80 for very tight racks.",
    )
    rack.add_argument(
        "--non-gpu-overhead",
        type=int,
        default=None,
        metavar="WATTS",
        help=(
            f"Non-GPU overhead per rack in watts (CPUs, NICs, storage, cooling). "
            f"Default: num_nodes × {_NON_GPU_OVERHEAD_PER_NODE_W} W."
        ),
    )

    gpus = parser.add_argument_group("GPU configuration")
    gpus.add_argument(
        "--gpus-per-node",
        type=int,
        required=True,
        metavar="N",
        help="Number of GPUs per node.",
    )
    gpus.add_argument(
        "--nodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of nodes in the rack / node pool (default: 1).",
    )
    gpus.add_argument(
        "--gpu-tdp",
        type=int,
        default=None,
        metavar="WATTS",
        help="GPU TDP in watts (required unless --gpu-sku is specified).",
    )
    gpus.add_argument(
        "--gpu-sku",
        type=str,
        default=None,
        metavar="SKU",
        help=(
            f"GPU SKU shorthand; sets --gpu-tdp automatically. "
            f"Known values: {', '.join(sorted(_KNOWN_TDPS))}."
        ),
    )

    multi = parser.add_argument_group("Multi-DGD apportionment")
    multi.add_argument(
        "--dgd-gpus",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "GPU count for each DGD on this rack. Omit for single-DGD mode. "
            "Example: --dgd-gpus 8 4 (DGD-A gets 8 GPUs, DGD-B gets 4)."
        ),
    )

    args = parser.parse_args()

    # Resolve GPU TDP
    gpu_tdp: Optional[int] = args.gpu_tdp
    if args.gpu_sku is not None:
        tdp_from_sku = _parse_gpu_sku(args.gpu_sku)
        if tdp_from_sku is None:
            print(
                f"ERROR: unknown GPU SKU '{args.gpu_sku}'. "
                f"Known values: {', '.join(sorted(_KNOWN_TDPS))}. "
                f"Use --gpu-tdp to specify manually.",
                file=sys.stderr,
            )
            sys.exit(2)
        if gpu_tdp is not None and gpu_tdp != tdp_from_sku:
            print(
                f"WARNING: --gpu-sku implies TDP={tdp_from_sku} W but "
                f"--gpu-tdp={gpu_tdp} W was also specified. Using --gpu-tdp.",
            )
        else:
            gpu_tdp = tdp_from_sku

    if gpu_tdp is None:
        print("ERROR: one of --gpu-tdp or --gpu-sku is required.", file=sys.stderr)
        sys.exit(2)

    total_gpus = args.nodes * args.gpus_per_node
    non_gpu_overhead = (
        args.non_gpu_overhead
        if args.non_gpu_overhead is not None
        else args.nodes * _NON_GPU_OVERHEAD_PER_NODE_W
    )

    if not (0 < args.headroom_factor <= 1.0):
        print(
            f"ERROR: --headroom-factor must be in (0, 1]. Got {args.headroom_factor}.",
            file=sys.stderr,
        )
        sys.exit(2)

    inp = SizingInput(
        rack_capacity_w=args.rack_capacity,
        headroom_factor=args.headroom_factor,
        non_gpu_overhead_w=non_gpu_overhead,
        total_gpus=total_gpus,
        gpu_tdp_w=gpu_tdp,
        dgd_gpu_counts=args.dgd_gpus or [],
    )

    try:
        result = compute_budget(inp)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    result.print_report()


if __name__ == "__main__":
    main()
