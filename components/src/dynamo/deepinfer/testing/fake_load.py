# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free lockstep load check: one DeepInfer engine on the CPU fake.

    DEEPINFER_FAKE_STEP_DELAY_S=0.012 DEEPINFER_FAKE_STEP_OVERLAP=1 \\
    DYN_DEEPINFER_TRACE_LOCKSTEP=1 \\
        python -m dynamo.deepinfer.testing.fake_load [--dp 32] [--seqs 18] \\
        [--max-tokens 1000] [--log PATH]

Builds the engine as the Dynamo worker does (``DeepInferEngine.from_args``,
``--no-rank0-in-process``, every rank on ``fake_deepinfer`` with the
production Cmd/Reply types), fills every rank with ``--seqs`` requests at once
(routed by ``dp_rank``) and streams them to the end from this process's event
loop, like the worker's request path. Prints one JSON line:

- ``client_period_ms``: per request, the time between its first and last
  token over the tokens in between (median over requests);
- ``full_batch``: from the controller's ``lockstep head`` lines (logged by the
  controller process with ``DYN_DEEPINFER_TRACE_LOCKSTEP=1``, or here for a
  controller thread, which has no trace of its own), the iterations with every
  rank at ``--seqs``: period, send span and receive span percentiles.

Every process's stderr goes to ``--log``. Also runs against older
``dynamo.deepinfer`` versions (it needs no module of this one).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from typing import Any

logger = logging.getLogger("dynamo.deepinfer.controller")

LINE = re.compile(
    r"lockstep head seq=(\d+) t=\S+ period_ms=([\d.]+) send_ms=([\d.]+) "
    r"recv_ms=([\d.]+) outputs=(\d+) running=(\d+)"
)


class Ctx:
    def __init__(self) -> None:
        self._id = uuid.uuid4().hex

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return False


def trace_thread(engine: Any) -> None:
    """Log a controller thread's iterations as a controller process does."""
    controller = engine.controller
    handles = sorted(controller._handles, key=lambda h: h.rank)
    marks: dict[str, float] = {}

    def timed(obj: Any, name: str, key: str, before: bool) -> None:
        inner = getattr(obj, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if before:
                marks[key] = time.monotonic()
            result = inner(*args, **kwargs)
            if not before:
                marks[key] = time.monotonic()
            return result

        setattr(obj, name, wrapper)

    timed(handles[0], "send", "started", True)
    timed(handles[-1], "send", "sent", False)
    timed(controller._recv_order[-1], "recv", "received", False)
    outputs = [0]
    last = [0.0]
    emit, iterate = engine._emit, controller._iterate

    def counting_emit(merged: dict) -> None:
        outputs[0] = len(merged)
        emit(merged)

    def traced_iterate(timeout: float) -> bool:
        outputs[0] = 0
        closing = iterate(timeout)
        started = marks["started"]
        period = (started - last[0]) * 1e3 if last[0] else 0.0
        last[0] = started
        logger.info(
            "lockstep head seq=%d t=%.6f period_ms=%.3f send_ms=%.3f "
            "recv_ms=%.3f outputs=%d running=%d",
            controller._seq - 1,
            time.time(),
            period,
            (marks["sent"] - started) * 1e3,
            (marks["received"] - marks["sent"]) * 1e3,
            outputs[0],
            sum(s.num_running for s in controller._stats),
        )
        return closing

    engine._emit = counting_emit
    controller._iterate = traced_iterate


async def client(engine: Any, rank: int, max_tokens: int) -> list[tuple[float, int]]:
    request = {
        "token_ids": [2 + (i * 7 + rank) % 900 for i in range(40)],
        "stop_conditions": {"max_tokens": max_tokens, "ignore_eos": True},
        "routing": {"dp_rank": rank},
    }
    seen, count = [], 0
    async for chunk in engine.generate(request, Ctx()):
        count += len(chunk["token_ids"])
        seen.append((time.monotonic(), count))
    return seen


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from dynamo.deepinfer.engine import DeepInferEngine

    argv = [
        *("--model", "fake-model", "--deepinfer-backend", "fake"),
        *("--dp-size", str(args.dp), "--max-num-seqs", str(args.seqs)),
        *("--max-model-len", str(args.max_tokens + 1024), "--no-rank0-in-process"),
    ]
    started = time.monotonic()
    engine, _ = await DeepInferEngine.from_args(argv)
    threaded = hasattr(engine.controller, "_iterate")
    if threaded and getattr(engine.controller, "_trace", None) is None:
        trace_thread(engine)
    await engine.start(worker_id=1)
    ready = time.monotonic()
    try:
        streams = await asyncio.gather(
            *(
                client(engine, i % args.dp, args.max_tokens)
                for i in range(args.dp * args.seqs)
            )
        )
    finally:
        await engine.cleanup()
    periods = sorted(
        (s[-1][0] - s[0][0]) * 1e3 / (s[-1][1] - s[0][1])
        for s in streams
        if s[-1][1] > s[0][1]
    )
    return {
        "controller": "thread" if threaded else "process",
        "startup_s": round(ready - started, 1),
        "requests": len(streams),
        "tokens": sum(s[-1][1] for s in streams if s),
        "client_period_ms": round(periods[len(periods) // 2], 2) if periods else None,
    }


def summarize(log_path: str, full_batch: int) -> dict[str, Any]:
    rows = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                rows.append([float(x) for x in m.groups()])
    full = [r for r in rows if r[5] >= full_batch and r[1] > 0]

    def pct(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        v = sorted(values)
        at = lambda q: v[min(len(v) - 1, len(v) * q // 100)]  # noqa: E731
        return {"p50": at(50), "p90": at(90), "p99": at(99), "max": v[-1]}

    return {
        "iterations": len(rows),
        "full_batch_iterations": len(full),
        "full_batch": {
            "period_ms": pct([r[1] for r in full]),
            "send_ms": pct([r[2] for r in full]),
            "recv_ms": pct([r[3] for r in full]),
            "outputs": pct([r[4] for r in full]),
        },
        "send_ms_all": pct([r[2] for r in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dp", type=int, default=32)
    parser.add_argument("--seqs", type=int, default=18)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    log_path = args.log or tempfile.mkstemp(prefix="fake-load-", suffix=".log")[1]
    # Every process (ranks, a controller process) inherits this stderr.
    sys.stderr.flush()
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)
    os.close(fd)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [fake_load] %(name)s: %(message)s",
    )
    os.environ.setdefault(
        "DEEPINFER_FAKE_DP_ENDPOINT", f"ipc://{tempfile.mkdtemp(prefix='di-')}/dp"
    )
    os.environ.setdefault("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "1024")
    os.environ.setdefault("DEEPINFER_FAKE_DP_TIMEOUT_S", "120")
    try:
        import uvloop  # as the Dynamo worker runs

        result = uvloop.run(run(args))
    except ImportError:
        result = asyncio.run(run(args))
    logging.shutdown()
    result.update(summarize(log_path, args.dp * args.seqs))
    result["log"] = log_path
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
