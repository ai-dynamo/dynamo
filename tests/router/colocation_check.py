# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch co-location probe for predict_on_route.

Launches two mocker workers behind a KvRouter and fires NUM_PROBLEMS groups of
NUM_SAMPLES sibling requests with a shared prefix. Uses `KvRouter.best_worker`
to observe which worker each request would be dispatched to, without sending
real generation traffic (so engine KV events cannot arrive in time to inform
later sibling decisions).

Invoked once per scenario (events_only | approximate | predict_on_route) so each
run owns a fresh DistributedRuntime + etcd lease. The top-level `main` below
spawns itself three times in subprocesses and compares.

Run with `source .venv/bin/activate && python tests/router/colocation_check.py`.
Assumes etcd+nats are running (deploy/docker-compose.yml).
"""

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dynamo._core import KvRouter, KvRouterConfig  # noqa: E402
from tests.router.common import min_initial_workers_env  # noqa: E402
from tests.router.helper import get_runtime  # noqa: E402
from tests.router.test_router_e2e_with_mockers import (  # noqa: E402
    SPEEDUP_RATIO,
    MockerProcess,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
logger = logging.getLogger("colocation_check")

NUM_PROBLEMS = 32
NUM_SAMPLES = 4
PREFIX_TOKENS = 256
SUFFIX_TOKENS = 8
BLOCK_SIZE = 16
NUM_MOCKERS = 2
MODEL_NAME = "Qwen/Qwen3-0.6B"
# Stagger siblings by this many milliseconds so each one's routing decision has
# a chance to be recorded before the next sibling queries. Models "batch arrives
# roughly together" without perfect simultaneous entry.
SIBLING_STAGGER_MS = 2

SCENARIOS = {
    "events_only": dict(use_kv_events=True),
    "approximate": dict(use_kv_events=False),
    "predict_on_route": dict(
        use_kv_events=True,
        router_predict_on_route=True,
        router_predicted_ttl_secs=5.0,
    ),
}


class _DummyRequest:
    def __init__(self, name: str) -> None:
        class _Node:
            def __init__(self, n: str) -> None:
                self.name = n

        self.node = _Node(name)
        self._finalizers: list = []

    def addfinalizer(self, fn) -> None:
        self._finalizers.append(fn)


async def _wait_ready(endpoint, router: KvRouter, n: int) -> list[int]:
    client = await endpoint.client()
    deadline = time.time() + 60
    while time.time() < deadline:
        ids = client.instance_ids()
        if len(ids) >= n:
            return sorted(ids)
        await asyncio.sleep(1.0)
    raise RuntimeError(f"only {len(client.instance_ids())} workers, expected {n}")


async def _probe_once(router: KvRouter, token_ids: list[int]) -> int:
    worker_id, _dp, _overlap = await router.best_worker(
        token_ids=token_ids,
        request_id=None,
        update_indexer=True,
    )
    return worker_id


async def _scenario_inner(label: str, endpoint, cfg: KvRouterConfig) -> dict:
    with min_initial_workers_env(NUM_MOCKERS):
        router = KvRouter(
            endpoint=endpoint,
            block_size=BLOCK_SIZE,
            kv_router_config=cfg,
        )
    ids = await _wait_ready(endpoint, router, NUM_MOCKERS)
    logger.info("[%s] workers ready: %s", label, ids)

    # Small settle time so router has propagated cluster view.
    await asyncio.sleep(1.0)

    rng = random.Random(0xC0DE)
    total = 0
    colocated = 0
    per_problem: list[list[int]] = []

    async def _probe_staggered(prefix: list[int]) -> list[int]:
        async def _one(i: int) -> int:
            if i > 0:
                await asyncio.sleep(SIBLING_STAGGER_MS * 0.001 * i)
            suffix = [rng.randint(1, 10000) for _ in range(SUFFIX_TOKENS)]
            return await _probe_once(router, prefix + suffix)

        return await asyncio.gather(*[_one(i) for i in range(NUM_SAMPLES)])

    for _ in range(NUM_PROBLEMS):
        prefix = [rng.randint(1, 10000) for _ in range(PREFIX_TOKENS)]
        sample_workers = await _probe_staggered(prefix)
        per_problem.append([int(w) for w in sample_workers])
        total += NUM_SAMPLES
        if len(set(sample_workers)) == 1:
            colocated += NUM_SAMPLES

    return {
        "label": label,
        "total": total,
        "colocated": colocated,
        "ratio": colocated / total,
        "per_problem": per_problem,
    }


def _run_scenario(label: str) -> dict:
    cfg_kwargs = SCENARIOS[label]
    request = _DummyRequest(f"colocation_{label}")
    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": False,
    }
    os.environ.setdefault("ETCD_ENDPOINTS", "http://localhost:2379")
    os.environ.setdefault("NATS_SERVER", "nats://localhost:4222")
    with MockerProcess(
        request,
        mocker_args=mocker_args,
        num_mockers=NUM_MOCKERS,
        request_plane="tcp",
    ) as mockers:
        runtime = get_runtime(request_plane="tcp")
        endpoint = runtime.endpoint(
            f"{mockers.namespace}.{mockers.component_name}.generate"
        )
        return asyncio.run(
            _scenario_inner(label, endpoint, KvRouterConfig(**cfg_kwargs))
        )


def _run_self_as_scenario(label: str) -> dict:
    """Spawn this script as a subprocess for one scenario; returns parsed JSON."""
    env = os.environ.copy()
    env.setdefault("DYN_LOG", "warn")
    cmd = [sys.executable, __file__, "--scenario", label]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"scenario {label} failed rc={proc.returncode}")
    # The subprocess prints a single JSON line with prefix RESULT_JSON:.
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    raise RuntimeError(f"scenario {label} produced no RESULT_JSON")


def _format_row(res: dict) -> str:
    return (
        f"{res['label']:>20s}: {res['colocated']:>3d}/{res['total']:<3d} "
        f"co-located ({res['ratio']*100:5.1f}%)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()))
    args = parser.parse_args()

    if args.scenario:
        res = _run_scenario(args.scenario)
        print("RESULT_JSON:" + json.dumps(res))
        return 0

    # Driver: spawn one subprocess per scenario.
    results = []
    for label in ["events_only", "approximate", "predict_on_route"]:
        logger.info("=== running scenario: %s ===", label)
        results.append(_run_self_as_scenario(label))

    print()
    print("=== co-location summary ===")
    for r in results:
        print(_format_row(r))
    print()

    events_only = next(r for r in results if r["label"] == "events_only")
    approx = next(r for r in results if r["label"] == "approximate")
    predict = next(r for r in results if r["label"] == "predict_on_route")
    print(
        f"predict_on_route={predict['ratio']:.2f}  "
        f"approximate={approx['ratio']:.2f}  "
        f"events_only={events_only['ratio']:.2f}"
    )
    # Approximate is the reference good outcome (route siblings together).
    # predict_on_route should match that (within slack).
    if predict["ratio"] < approx["ratio"] - 0.1:
        print("FAIL: predict_on_route should roughly match approximate")
        return 1
    if predict["ratio"] <= events_only["ratio"]:
        print("FAIL: predict_on_route did not beat events_only baseline")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
