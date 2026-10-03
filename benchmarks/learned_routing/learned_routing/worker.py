# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay worker subprocess (``python -m learned_routing.worker``).

The parent (:mod:`learned_routing.pool`) keeps one worker per pool thread and sends it one job per
line on stdin. The worker runs the replay through the Python API documented in
``facts/setup.json`` (``entrypoint.how_to_call``), computes every metric from ``per_request``
(:mod:`learned_routing.goodput`), writes the record to ``job["result_path"]`` and answers with one
protocol line on its private stdout. Native code that prints to stdout is redirected to stderr, so
it cannot corrupt the protocol.

A crash or hang only costs this process: the parent records the error and starts a new worker.
Repeated replays in one process are deterministic (setup) and save ~0.5 s of startup each.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import resource
import select
import sys
import time
import traceback
from pathlib import Path

from learned_routing import goodput
from learned_routing.canon import atomic_write_bytes, atomic_write_text

ENTRYPOINT = "dynamo.replay.run_trace_replay|run_synthetic_trace_replay (python api)"

_engine_cache: dict[str, dict] = {}
_e0_tables: dict[tuple, object] = {}


def _engine(path: str) -> dict:
    if path not in _engine_cache:
        _engine_cache[path] = json.loads(Path(path).read_text())
    return _engine_cache[path]


def _engine_args(engine: dict, overrides: dict):
    from dynamo.mocker import MockEngineArgs

    args = dict(engine["mock_engine_args"])
    args.update(overrides or {})
    return MockEngineArgs.from_json(json.dumps(args))


def _router_config(replay: dict):
    if replay["router_mode"] != "kv_router":
        return None
    from dynamo.llm import KvRouterConfig

    return KvRouterConfig(
        router_policy_config=replay["policy_yaml"], **replay["router_config"]
    )


def _e0_table(engine_path: str, cache_dir: str, method: str):
    from learned_routing.e0 import E0Table

    key = (engine_path, cache_dir, method)
    if key not in _e0_tables:
        _e0_tables[key] = E0Table(_engine(engine_path), Path(cache_dir), method)
    return _e0_tables[key]


def run_replay(job: dict) -> tuple[dict, list[dict], float]:
    from dynamo.replay import run_synthetic_trace_replay, run_trace_replay

    replay = job["replay"]
    engine = _engine(job["engine_json"])
    common = {
        "extra_engine_args": _engine_args(engine, job.get("engine_overrides") or {}),
        "router_config": _router_config(replay),
        "router_mode": replay["router_mode"],
        "num_workers": replay["num_workers"],
        "sla_ttft_ms": replay.get("sla_ttft_ms"),
        "sla_itl_ms": replay.get("sla_itl_ms"),
        "capture_per_request": True,
    }
    common.update(replay["load_kwargs"])
    started = time.monotonic()
    if replay["kind"] == "synthetic":
        syn = dict(replay["synthetic"])
        report = run_synthetic_trace_replay(
            syn.pop("input_tokens"),
            syn.pop("output_tokens"),
            syn.pop("request_count"),
            arrival_seed=replay["arrival_seed"],
            **syn,
            **common,
        )
    else:
        report = run_trace_replay(
            replay["trace_path"],
            trace_format=replay["trace_format"],
            trace_block_size=replay.get("trace_block_size"),
            execution_model=replay.get("execution_model"),
            **replay.get("replay_options", {}),
            **common,
        )
    wall_s = time.monotonic() - started
    rows = [goodput.compact_row(dict(record)) for record in (report.per_request or [])]
    return dict(report.summary), rows, wall_s


def _test_job(job: dict) -> dict:
    """Pool tests only: ``_test_sleep`` sleeps, ``_test_crash`` exits without answering."""
    kind = job["replay"]["kind"]
    if kind == "_test_crash":
        os._exit(17)
    time.sleep(float(job["replay"].get("seconds", 0.0)))
    return {**job["record"], "error": None, "worker_pid": os.getpid(), "test": kind}


def handle(job: dict) -> dict:
    if job["replay"]["kind"].startswith("_test_"):
        return _test_job(job)
    record = dict(job["record"])
    started = time.monotonic()
    error = None
    try:
        summary, rows, wall_s = run_replay(job)
        rows = goodput.canonical_rows(rows)
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        record["per_request_canonical_sha256"] = hashlib.sha256(
            text.encode()
        ).hexdigest()
        scoring = job["scoring"]
        e0 = None
        if scoring["sla"].get("e2e_slowdown") is not None or scoring.get(
            "always_e0", True
        ):
            e0 = _e0_table(
                job["engine_json"], scoring["e0_cache_dir"], scoring["e0_method"]
            )
        warmup_ids = None
        warmup_trace_ms = scoring["measure"].get("warmup_trace_ms")
        if warmup_trace_ms is not None:
            if job["replay"]["kind"] == "synthetic":
                raise ValueError("measure.warmup_trace_ms needs a trace replay")
            warmup_ids = goodput.warmup_ids_from_trace(
                Path(job["replay"]["trace_path"]).read_text().splitlines(),
                float(warmup_trace_ms),
            )
        metrics = goodput.compute_metrics(
            rows,
            summary,
            sla=scoring["sla"],
            measure=scoring["measure"],
            open_loop=scoring["open_loop"],
            num_workers=scoring["num_workers"],
            e0=e0,
            warmup_ids=warmup_ids,
            occupancy_cap=scoring.get("occupancy_cap"),
        )
        if e0 is not None:
            e0.persist()
            record["e0_method"] = e0.method
        record.update(metrics)
        record["wall_s"] = wall_s
        record["native_wall_time_ms"] = summary.get("wall_time_ms")
        if (
            metrics["goodput_check_rel_err"] is not None
            and metrics["goodput_check_rel_err"] > goodput.NATIVE_REL_TOL
        ):
            error = (
                f"goodput_mismatch: recomputed token-form goodput {metrics['goodput_rps_itl']!r} "
                f"vs report {metrics['goodput_rps_report']!r} "
                f"(rel {metrics['goodput_check_rel_err']:.3e} > {goodput.NATIVE_REL_TOL})"
            )
        per_request_path = job.get("per_request_path")
        if per_request_path:
            atomic_write_bytes(
                Path(per_request_path), gzip.compress(text.encode(), mtime=0)
            )
            record["per_request_path"] = per_request_path
    except Exception as exc:  # recorded in the result, never swallowed
        error = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-4000:]
    record["error"] = error
    record["eval_wall_s"] = time.monotonic() - started
    record["worker_pid"] = os.getpid()
    record["worker_peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    )
    record["replay_entrypoint"] = ENTRYPOINT
    return record


def main() -> None:
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    os.environ.setdefault("DYN_LOG", "warn")
    ready = {"status": "ready", "pid": os.getpid()}
    check_dir = os.environ.get("LR_BUILD_CHECK_DIR")
    if check_dir:
        from learned_routing.cache import bindings_build_id

        ready["build_id"] = bindings_build_id(Path(check_dir))["build_id"]
    protocol.write(json.dumps(ready) + "\n")
    idle_exit_s = float(os.environ.get("LR_WORKER_IDLE_EXIT_S", "300"))
    stdin = sys.stdin.buffer
    while True:
        # Exit when idle, so an idle pool does not pin memory; the parent's ack handshake makes
        # a job sent just as the worker exits a resend, not a crash.
        readable, _, _ = select.select([stdin], [], [], idle_exit_s)
        if not readable:
            return
        line = stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        job = json.loads(line)
        protocol.write(json.dumps({"status": "ack", "job_id": job["job_id"]}) + "\n")
        record = handle(job)
        atomic_write_text(Path(job["result_path"]), json.dumps(record, sort_keys=True))
        protocol.write(json.dumps({"status": "done", "job_id": job["job_id"]}) + "\n")


if __name__ == "__main__":
    main()
