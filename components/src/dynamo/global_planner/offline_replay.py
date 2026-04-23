# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline replay harness for GlobalPlanner's ScaleRequestHandler.

Loads a request trace (mooncake-format JSONL: ``{timestamp, input_length,
output_length, hash_ids}``) and replays simulated scale requests against the
real ``ScaleRequestHandler``. Only the Kubernetes connector layer is mocked;
the arbitration, intent cache, budget math, and cross-DGD pairing code paths
run exactly as in production.

The harness does not drive real local planners. Instead it applies a simple
capacity-based sizing heuristic per DGD per tick (desired = ceil(reqs / cap)),
which is usually enough to produce interesting floor / ceiling / pair
behaviour without pulling in the full PlannerStateMachine.

Example:

    PYTHONPATH=components/src:$PYTHONPATH \\
      python -m dynamo.global_planner.offline_replay \\
        /home/jothomson/Desktop/dynamo/traces/mooncake_trace.jsonl \\
        --num-dgds 2 --mode agg \\
        --min-total-gpus 8 --max-total-gpus 8 \\
        --initial-replicas 4 \\
        --report /tmp/gp-replay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

from dynamo.global_planner.scale_handler import ScaleRequestHandler
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleRequest


# --------------------------------------------------------------------------- #
# Mock Kubernetes connector                                                   #
# --------------------------------------------------------------------------- #


def _build_dgd_spec(pools: dict[str, tuple[int, int]]) -> dict:
    """Build a DGD spec dict. pools: {sub_type -> (replicas, gpu_per_replica)}."""
    services = {}
    for sub_type, (replicas, gpu) in pools.items():
        services[f"{sub_type}-svc"] = {
            "subComponentType": sub_type,
            "replicas": replicas,
            "resources": {"limits": {"gpu": gpu}},
        }
    return {"spec": {"services": services}}


class _MockConnector:
    """In-process stand-in for KubernetesConnector.

    Tracks applied replicas per pool so the handler's fresh reads of
    ``get_graph_deployment`` reflect the state of the world as the handler
    mutated it via ``set_component_replicas``.
    """

    def __init__(self, dgd_name: str, initial_pools: dict[str, tuple[int, int]]):
        self.parent_dgd_name = dgd_name
        # sub_type -> [replicas, gpu_per_replica] (list so we can mutate in place)
        self._state: dict[str, list[int]] = {
            k: [r, g] for k, (r, g) in initial_pools.items()
        }
        self.kube_api = MagicMock()
        self.kube_api.get_graph_deployment = MagicMock(side_effect=self._get_spec)
        self.set_component_replicas = AsyncMock(side_effect=self._apply)

    def _get_spec(self, _dgd_name):
        pools = {k: (v[0], v[1]) for k, v in self._state.items()}
        return _build_dgd_spec(pools)

    async def _apply(self, targets, blocking=False):
        for t in targets:
            sub = t.sub_component_type.value
            if sub not in self._state:
                continue
            self._state[sub][0] = t.desired_replicas

    @property
    def replicas(self) -> dict[str, int]:
        return {k: v[0] for k, v in self._state.items()}

    @property
    def total_gpus(self) -> int:
        return sum(r * g for (r, g) in self._state.values())


# --------------------------------------------------------------------------- #
# Trace loading and binning                                                   #
# --------------------------------------------------------------------------- #


def _load_trace(path: str) -> Iterator[tuple[int, int, int]]:
    """Yield (timestamp_ms, input_length, output_length) tuples from JSONL."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield (
                int(rec.get("timestamp", 0)),
                int(rec.get("input_length", 0)),
                int(rec.get("output_length", 0)),
            )


def _bin_by_tick(
    trace: Iterator[tuple[int, int, int]], tick_seconds: int
) -> Iterator[tuple[int, int]]:
    """Group requests into fixed windows. Yield (tick_index, n_requests_in_tick).

    Empty ticks between bursts are filled in so the replay cadence stays
    uniform (matters for intent cache TTL and anti-thrash behaviour).
    """
    tick_ms = tick_seconds * 1000
    current_tick: int | None = None
    count = 0
    for ts, _isl, _osl in trace:
        tick = ts // tick_ms
        if current_tick is None:
            current_tick = tick
        if tick != current_tick:
            yield current_tick, count
            # Emit empty ticks in between to preserve timing.
            for t in range(current_tick + 1, tick):
                yield t, 0
            current_tick = tick
            count = 0
        count += 1
    if current_tick is not None:
        yield current_tick, count


# --------------------------------------------------------------------------- #
# Outcome capture (classifies what the handler did)                           #
# --------------------------------------------------------------------------- #


class _OutcomeLogHandler(logging.Handler):
    """Captures the handler's decision classification from its log output."""

    def __init__(self):
        super().__init__()
        self.last_decision: str | None = None

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "Paired transfer (intra-DGD)" in msg:
            self.last_decision = "intra_dgd_pair"
        elif "Paired transfer (cross-DGD)" in msg:
            self.last_decision = "cross_dgd_pair"
        elif msg.startswith("Standalone scale request"):
            self.last_decision = "standalone"


# --------------------------------------------------------------------------- #
# Sizing heuristic                                                            #
# --------------------------------------------------------------------------- #


def _desired_replicas(
    n_requests: int, capacity_per_replica: int, min_replicas: int
) -> int:
    """ceil(n / cap), clamped to min_replicas."""
    if n_requests <= 0:
        return min_replicas
    return max(min_replicas, (n_requests + capacity_per_replica - 1) // capacity_per_replica)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline replay harness for GlobalPlanner's ScaleRequestHandler",
    )
    parser.add_argument("trace", help="Path to mooncake-format trace JSONL")
    parser.add_argument(
        "--num-dgds",
        type=int,
        default=2,
        help="Number of simulated DGDs (default: 2)",
    )
    parser.add_argument(
        "--mode",
        choices=["agg", "disagg"],
        default="agg",
        help="Pool topology per DGD. agg=one 'decode' pool; disagg=prefill+decode.",
    )
    parser.add_argument(
        "--split-disagg",
        action="store_true",
        help=(
            "In disagg mode, send prefill and decode as SEPARATE ScaleRequests "
            "to simulate split-mode local planners (one planner per pool). "
            "This exercises the intra-DGD pair path: one pool's intent gets "
            "cached, the other pool's request then pairs with it."
        ),
    )
    parser.add_argument(
        "--min-total-gpus",
        type=int,
        default=-1,
        help="GlobalPlanner floor (cluster-wide). -1 disables.",
    )
    parser.add_argument(
        "--max-total-gpus",
        type=int,
        default=-1,
        help="GlobalPlanner ceiling (cluster-wide). -1 disables.",
    )
    parser.add_argument(
        "--intent-cache-ttl-seconds",
        type=float,
        default=120.0,
        help="Intent cache TTL (default: 120s)",
    )
    parser.add_argument(
        "--initial-replicas",
        type=int,
        default=2,
        help="Starting replica count per pool per DGD",
    )
    parser.add_argument(
        "--gpu-per-replica",
        type=int,
        default=1,
        help="GPUs per replica (applied uniformly; use --prefill-gpu for asymmetry)",
    )
    parser.add_argument(
        "--prefill-gpu",
        type=int,
        default=None,
        help="Override GPUs per prefill replica (disagg only)",
    )
    parser.add_argument(
        "--decode-gpu",
        type=int,
        default=None,
        help="Override GPUs per decode replica",
    )
    parser.add_argument(
        "--tick-seconds",
        type=int,
        default=10,
        help="Trace-time bin width (default: 10s)",
    )
    parser.add_argument(
        "--capacity-per-replica",
        type=int,
        default=10,
        help="Assumed capacity (requests per tick) per replica for sizing",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Cap number of ticks to replay (useful for quick sanity)",
    )
    parser.add_argument(
        "--report",
        help="Write a JSON summary report to this path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream handler INFO logs to stderr for debugging",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    # Resolve per-pool GPU counts.
    prefill_gpu = args.prefill_gpu if args.prefill_gpu is not None else args.gpu_per_replica
    decode_gpu = args.decode_gpu if args.decode_gpu is not None else args.gpu_per_replica

    dgd_names = [f"dgd-{i}" for i in range(args.num_dgds)]
    handler = ScaleRequestHandler(
        runtime=MagicMock(),
        managed_namespaces=[f"default-{n}" for n in dgd_names],
        k8s_namespace="default",
        min_total_gpus=args.min_total_gpus,
        max_total_gpus=args.max_total_gpus,
        intent_cache_ttl_seconds=args.intent_cache_ttl_seconds,
    )

    # Install mocked connectors.
    connectors: dict[str, _MockConnector] = {}
    for name in dgd_names:
        if args.mode == "agg":
            pools = {"decode": (args.initial_replicas, decode_gpu)}
        else:
            pools = {
                "prefill": (args.initial_replicas, prefill_gpu),
                "decode": (args.initial_replicas, decode_gpu),
            }
        conn = _MockConnector(name, pools)
        handler.connectors[f"default/{name}"] = conn
        connectors[name] = conn

    # Log capture for classification.
    capture = _OutcomeLogHandler()
    sh_logger = logging.getLogger("dynamo.global_planner.scale_handler")
    sh_logger.addHandler(capture)
    if args.verbose:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        sh_logger.addHandler(stream)
    sh_logger.setLevel(logging.INFO)

    async def _fire(name: str, targets: list[TargetReplica]) -> str:
        """Emit one ScaleRequest and return the classified outcome."""
        conn = connectors[name]
        if all(
            conn._state[t.sub_component_type.value][0] == t.desired_replicas
            for t in targets
        ):
            return "stable_skipped"
        capture.last_decision = None
        req = ScaleRequest(
            caller_namespace=f"default-{name}",
            graph_deployment_name=name,
            k8s_namespace="default",
            target_replicas=targets,
        )
        results = []
        async for r in handler.scale_request(req.model_dump()):
            results.append(r)
        response = results[0] if results else {"status": "none", "message": ""}
        if response["status"] == "error":
            msg = response["message"].lower()
            if "below floor" in msg:
                return "denied_floor"
            if "ceiling" in msg or "exceeds" in msg:
                return "denied_ceiling"
            return "denied_other"
        return capture.last_decision or "standalone"

    outcomes: Counter[str] = Counter()
    per_tick_events: list[dict] = []
    initial_total = sum(c.total_gpus for c in connectors.values())

    for tick_idx, (tick, n_requests) in enumerate(
        _bin_by_tick(_load_trace(args.trace), args.tick_seconds)
    ):
        if args.max_ticks is not None and tick_idx >= args.max_ticks:
            break

        per_dgd = n_requests // max(1, args.num_dgds)
        for name in dgd_names:
            conn = connectors[name]
            if args.mode == "agg":
                desired = _desired_replicas(per_dgd, args.capacity_per_replica, 1)
                target_groups = [[
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE,
                        desired_replicas=desired,
                    )
                ]]
            else:
                half = per_dgd // 2
                prefill_target = TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    desired_replicas=_desired_replicas(
                        half, args.capacity_per_replica, 1
                    ),
                )
                decode_target = TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    desired_replicas=_desired_replicas(
                        half, args.capacity_per_replica, 1
                    ),
                )
                if args.split_disagg:
                    # Simulate split-mode local planners: one ScaleRequest per
                    # pool. Alternate ordering by tick so both arrival orders
                    # are exercised.
                    if tick_idx % 2 == 0:
                        target_groups = [[prefill_target], [decode_target]]
                    else:
                        target_groups = [[decode_target], [prefill_target]]
                else:
                    # Standard disagg: one ScaleRequest containing both pools.
                    target_groups = [[prefill_target, decode_target]]

            for targets in target_groups:
                decision = await _fire(name, targets)
                outcomes[decision] += 1
                per_tick_events.append(
                    {
                        "tick": tick,
                        "dgd": name,
                        "n_requests_in_tick": per_dgd,
                        "desired": {
                            t.sub_component_type.value: t.desired_replicas
                            for t in targets
                        },
                        "decision": decision,
                        "state_after": dict(conn.replicas),
                    }
                )

    # Summary
    final_total = sum(c.total_gpus for c in connectors.values())
    total_events = sum(outcomes.values())
    print("=" * 72)
    print("Offline replay summary")
    print("=" * 72)
    print(f"Trace:            {args.trace}")
    print(f"Mode:             {args.mode}")
    print(f"DGDs:             {args.num_dgds}")
    print(f"Min/Max GPUs:     {args.min_total_gpus} / {args.max_total_gpus}")
    print(f"Total events:     {total_events}")
    print(f"Initial GPUs:     {initial_total}")
    print(f"Final GPUs:       {final_total}")
    print()
    print("Outcomes:")
    for k in [
        "standalone",
        "intra_dgd_pair",
        "cross_dgd_pair",
        "denied_floor",
        "denied_ceiling",
        "denied_other",
        "stable_skipped",
    ]:
        v = outcomes.get(k, 0)
        print(f"  {k:20s}: {v}")
    print()
    print("Final per-DGD state:")
    for name, conn in connectors.items():
        print(f"  {name}: replicas={dict(conn.replicas)} gpus={conn.total_gpus}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "outcomes": dict(outcomes),
                    "initial_total_gpus": initial_total,
                    "final_total_gpus": final_total,
                    "final_state": {
                        name: {"replicas": dict(c.replicas), "gpus": c.total_gpus}
                        for name, c in connectors.items()
                    },
                    "events_tail": per_tick_events[-200:],
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nJSON report written to {args.report}")

    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
