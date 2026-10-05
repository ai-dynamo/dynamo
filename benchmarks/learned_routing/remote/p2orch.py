# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase-2 orchestrator: resumable placement of ``lr-train`` runs on Slurm CPU nodes and the workstation.

Reads ``CR/runs/phase2/MANIFEST.json`` (every run: space, args, tier, wave) and keeps its own state in
``CR/runs/phase2/state.json``. Each ``tick`` is one idempotent pass:

1. refresh the Slurm state of our own live jobs (``lane.py alloc-refresh``, exact job IDs) and the
   liveness of local runs (exact PIDs);
2. rsync each remote run's returned directory, read its ``status.json``; a finished job whose run
   is ``done`` is ingested into the campaign cache (``lr-eval --ingest``); a run paused by its wall
   limit (exit 3) is queued again and resumes from its synced checkpoint;
3. fill every free target with the next eligible run, breadth-first by wave (a wave starts only
   after every runnable run of the earlier waves has been submitted): Slurm nodes through
   ``submit_train.sh --only RUN`` pinned with ``--nodelist``, the workstation through a local
   chunked ``lr-train`` loop over the shared CR/slots pool;
4. write per-run evaluation counts (budget equality) and the state file.

Slurm jobs run for at most ``--max-time-s``. A cluster whose jobs must end by a recurring weekly
time sets that job-end cutoff with ``init --cutoff "<weekday> HH:MM"`` (or ``LR_CUTOFF``; there is no
built-in default): ``--time`` = cutoff - now, capped at ``--max-time-s``. With ``--cluster-reopen``
(or ``LR_CLUSTER_REOPEN``) also set, no Slurm submission happens inside the maintenance window
(cutoff .. reopen time). Nothing is deleted; jobs are cancelled only by their exact IDs.

Site settings (no defaults; set them in ``site.env`` next to this file or in the environment):
``LR_CR`` (campaign root), ``LR_SSH_ALIAS``, ``LR_REMOTE_ROOT`` and ``LR_P2ORCH_TARGETS`` (a JSON
list of placement targets, see ``p2orch_targets.example.json``). ``LR_TZ`` sets the cutoff's time
zone (default UTC); ``LR_CUTOFF`` and ``LR_CLUSTER_REOPEN`` are the defaults of ``init --cutoff``
and ``init --cluster-reopen`` (unset: no cutoff, no maintenance window).

Several instances can share the nodes: ``LR_P2_DIR`` moves the manifest, state, logs and local runs
to another phase directory (run that instance's copy of this file from its own worktree, so its
bundle carries its own bindings build), and ``init --yield-to OTHER/state.json[:MAX_WAVE]`` makes
it skip the other instance's running targets and place nothing while the other has eligible
queued runs up to MAX_WAVE.

Commands: ``init``, ``status``, ``tick [--dry-run]``, ``loop``, ``budget``, ``select``,
``drift-matrix``, ``migrate RUN``, ``cancel RUN``, ``rebalance``, ``targets``. See
CR/runs/phase2/ORCHESTRATOR.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from zoneinfo import ZoneInfo

import siteenv

siteenv.load()
LANE = Path(__file__).resolve().parent
WT = LANE.parents[2]
CR = Path(siteenv.require("LR_CR"))
# A second orchestrator instance (e.g. the AIS league, run from its own worktree and bundle) keeps
# its manifest, state, logs and local runs in its own phase directory.
P2 = Path(os.environ.get("LR_P2_DIR", CR / "runs" / "phase2"))
MANIFEST = P2 / "MANIFEST.json"
STATE = P2 / "state.json"
PY = WT / ".venv" / "bin" / "python"
LR_EVAL = WT / ".venv" / "bin" / "lr-eval"
LR_TRAIN = WT / ".venv" / "bin" / "lr-train"
SSH_ALIAS = siteenv.require("LR_SSH_ALIAS")
REMOTE_ROOT = siteenv.require("LR_REMOTE_ROOT")
TZ = ZoneInfo(os.environ.get("LR_TZ") or "UTC")
SCHEMA = "learned-routing.phase2-state.v1"
NUMERICS_PIN = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR",
}
VALIDATED_IMAGES = {
    ("Ubuntu 24.04.5 LTS", "2.39")
}  # facts/remote.json parity (A9 addendum)
LIVE = {
    "SUBMITTED",
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "REQUEUED",
    "SUSPENDED",
}
MAX_ATTEMPTS = 6
PENDING_LIMIT_S = 1800
BAD_NODE_STATES = ("DOWN", "DRAIN", "FAIL", "MAINT", "RESERVED")
# Rebalance (config "rebalance": true): node_train.sh stops lr-train LR_TAIL_MARGIN_S before the
# job ends; a moved run restages its bundle, replicates and checkpoint in about RESTAGE_S.
TAIL_MARGIN_S = 600
RESTAGE_S = 240
REBALANCE_MARGIN_S = 600
MIN_SPEEDUP = 1.4
MIN_GENS_SAVED = 1.0  # a move that lets the run finish before the cutoff
MIN_GENS_SAVED_PARTIAL = 3.0  # a move after which the run still pauses at the cutoff


# Placement targets come from the site: LR_P2ORCH_TARGETS names a JSON list of target objects (see
# p2orch_targets.example.json). rank orders preference (lower first); hours = PROJECTED per-run wall.
def default_targets() -> list[dict]:
    path = Path(siteenv.require("LR_P2ORCH_TARGETS"))
    targets = json.loads(path.read_text())
    if not isinstance(targets, list) or not all(isinstance(t, dict) for t in targets):
        raise SystemExit(f"{path}: expected a JSON list of target objects")
    return targets


# -- small helpers --------------------------------------------------------------------------------
def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def stamp(t: dt.datetime | None = None) -> str:
    return (t or now()).strftime("%Y-%m-%d %H:%M:%S %Z")


def log(msg: str) -> None:
    line = f"[p2orch {stamp()}] {msg}"
    print(line, flush=True)
    with open(P2 / "orchestrator.log", "a") as handle:
        handle.write(line + "\n")


def run(
    cmd, *, env=None, timeout=600, check=True, cwd=None
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
        cwd=cwd,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(map(str, cmd))[:300]}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return proc


def rssh(command: str, timeout=120) -> str:
    return run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SSH_ALIAS, command],
        timeout=timeout,
    ).stdout


def scancel(job: str, reason: str) -> None:
    """Cancel our own job by its exact ID and record the cancel on its facts/remote.json entry."""
    rssh(f"scancel {job}")
    try:
        run(
            [
                PY,
                LANE / "lane.py",
                "alloc-record",
                "--job",
                job,
                "--json",
                json.dumps({"cancel_issued": stamp(), "cancelled_by": reason}),
            ],
            timeout=120,
        )
    except Exception as exc:  # the cancel itself happened; only the record failed
        log(f"warning: could not record the cancel of job {job} in remote.json: {exc}")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


@contextmanager
def state_lock():
    fd = os.open(str(STATE) + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def load_state() -> dict:
    state = read_json(STATE)
    if state is None:
        raise SystemExit(f"{STATE} missing: run `p2orch.py init` first")
    return state


def age_s(when: str) -> float:
    t = dt.datetime.strptime(when.rsplit(" ", 1)[0], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=TZ
    )
    return (now() - t).total_seconds()


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# -- the optional weekly job-end cutoff ----------------------------------------------------------
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def parse_cutoff(text: str | None) -> str | None:
    """Normalize ``"<weekday> HH:MM"`` (weekday as mon..sun, or its full name) to ``"ddd HH:MM"``."""
    if not text:
        return None
    m = re.fullmatch(r"\s*([A-Za-z]+)\s+(\d{1,2}):(\d{2})\s*", text)
    day = m.group(1).lower()[:3] if m else ""
    if not m or day not in WEEKDAYS or int(m.group(2)) > 23 or int(m.group(3)) > 59:
        raise SystemExit(
            f"--cutoff {text!r}: expected '<weekday> HH:MM', e.g. 'mon 06:00'"
        )
    return f"{day} {int(m.group(2)):02d}:{m.group(3)}"


def cutoff_after(t: dt.datetime, cfg: dict) -> dt.datetime | None:
    """The next weekly job-end cutoff (LR_TZ local) at or after t; None without a cutoff."""
    if not cfg.get("cutoff"):
        return None
    day_name, hhmm = cfg["cutoff"].split()
    hh, mm = (int(x) for x in hhmm.split(":"))
    day = t.replace(hour=hh, minute=mm, second=0, microsecond=0)
    day += dt.timedelta(days=(WEEKDAYS.index(day_name) - t.weekday()) % 7)
    if day < t:
        day += dt.timedelta(days=7)
    return day


def cluster_window(cfg: dict, t: dt.datetime | None = None) -> tuple[bool, float, str]:
    """(may submit, seconds until the job-end cutoff, reason)."""
    t = t or now()
    reopen = (
        dt.datetime.fromisoformat(cfg["cluster_reopen"]).astimezone(TZ)
        if cfg.get("cluster_reopen")
        else None
    )
    cutoff = cutoff_after(t, cfg)
    if cutoff is None:
        return True, math.inf, "no job-end cutoff"
    if reopen:
        # The maintenance window runs from the cutoff that precedes the reopen time to the reopen time.
        window_start = cutoff_after(reopen - dt.timedelta(days=7), cfg)
        if window_start <= t < reopen:
            return (
                False,
                0.0,
                f"maintenance window {stamp(window_start)} .. {stamp(reopen)}",
            )
    left = (cutoff - t).total_seconds()
    if left < cfg["min_job_s"]:
        return False, left, f"{left:.0f} s to the {stamp(cutoff)} cutoff < min_job_s"
    return True, left, f"cutoff {stamp(cutoff)}"


# -- state ----------------------------------------------------------------------------------------
def init_state(args) -> dict:
    if STATE.exists() and not args.force:
        raise SystemExit(
            f"{STATE} exists (use --force to rebuild targets/config, runs are kept)"
        )
    old = read_json(STATE, {}) or {}
    cutoff = parse_cutoff(args.cutoff)
    if args.cluster_reopen and not cutoff:
        raise SystemExit(
            "--cluster-reopen needs --cutoff (the window starts at the cutoff)"
        )
    targets = old.get("targets")
    if not targets and args.targets_from:
        targets = (read_json(Path(args.targets_from), {}) or {}).get("targets")
        if not targets:
            raise SystemExit(f"{args.targets_from}: no targets to copy")
    state = {
        "schema": SCHEMA,
        "created": old.get("created", stamp()),
        "config": {
            "bundle": args.bundle,
            "max_wave": args.max_wave,
            "cutoff": cutoff,
            "cluster_reopen": args.cluster_reopen,
            "min_job_s": args.min_job_s,
            "max_time_s": args.max_time_s,
            "local_chunk_s": 540,
            "loop_interval_s": 300,
            "yield_to": [parse_yield(text) for text in args.yield_to or []],
            "yield_own_from_wave": args.yield_own_from_wave,
        },
        "targets": targets or default_targets(),
        "runs": old.get("runs", {}),
        "notes": old.get("notes", []),
    }
    sync_runs(state)
    write_json(STATE, state)
    return state


def parse_yield(text: str) -> dict:
    """``PATH[:MAX_WAVE]``: another orchestrator's state.json and the waves this one yields to."""
    path, _, wave = (
        text.rpartition(":") if re.fullmatch(r".*:\d+", text) else (text, "", "")
    )
    if not Path(path).is_file():
        raise SystemExit(f"--yield-to {text}: no state file {path}")
    return {"state": str(Path(path).resolve()), "max_wave": int(wave) if wave else None}


def foreign_view(state: dict) -> tuple[set[str], list[str]]:
    """(targets held by the other orchestrators' running runs, why this instance yields).

    Read-only: another instance's state.json is replaced atomically, so no lock is needed. It
    yields while that instance has an eligible queued run at or below the configured wave.
    """
    held: set[str] = set()
    reasons: list[str] = []
    for entry in state["config"].get("yield_to") or []:
        other = read_json(Path(entry["state"]), None)
        if not other:
            reasons.append(f"{entry['state']} unreadable")
            continue
        held |= busy_targets(other)
        cap = entry.get("max_wave")
        waiting = [
            rid
            for rid in eligible(other)
            if cap is None or other["runs"][rid]["wave"] <= cap
        ]
        if waiting:
            reasons.append(
                f"{Path(entry['state']).parent.name} has {len(waiting)} eligible run(s) "
                f"up to wave {cap if cap is not None else 'max'} (e.g. {waiting[0]})"
            )
    return held, reasons


def sync_runs(state: dict) -> None:
    for i, spec in enumerate(manifest()["runs"]):
        entry = state["runs"].setdefault(spec["run"], {"status": "new", "attempts": []})
        entry["order"] = i
        entry["wave"] = spec["wave"]
        entry["tier"] = spec["tier"]
        entry["kind"] = spec["kind"]
        entry["policy_key"] = spec["policy_key"]
        runnable = spec["status"] == "planned" and (CR / spec["space"]).is_file()
        if entry["status"] in ("new", "pending", "blocked"):
            entry["status"] = (
                "queued"
                if runnable
                else ("blocked" if spec["status"] == "blocked" else "pending")
            )


def mspec(run_id: str) -> dict:
    return next(r for r in manifest()["runs"] if r["run"] == run_id)


def current(entry: dict) -> dict | None:
    return entry["attempts"][-1] if entry["attempts"] else None


def remote_run_dir(state: dict, run_id: str) -> Path:
    return CR / "runs" / "remote" / "returned" / state["config"]["bundle"] / run_id


def run_dir(state: dict, run_id: str) -> Path | None:
    att = current(state["runs"][run_id])
    if att is None:
        return None
    if att["kind"] == "local":
        return Path(att["run_dir"])
    return remote_run_dir(state, run_id) / "runs" / "train" / run_id


# -- bundle ---------------------------------------------------------------------------------------
def bundle_dir(state: dict) -> Path:
    return CR / "runs" / "remote" / "bundles" / state["config"]["bundle"]


def ensure_bundle(state: dict, dry_run: bool) -> Path:
    """Build the train bundle once; keep its spaces and train_jobs.jsonl in step with MANIFEST."""
    bundle = bundle_dir(state)
    runnable = [
        r
        for r in manifest()["runs"]
        if r["status"] == "planned" and (CR / r["space"]).is_file()
    ]
    if not (bundle / "MANIFEST.json").exists():
        if dry_run:
            log(f"dry-run: would build bundle {bundle}")
            return bundle
        log(f"building bundle {bundle}")
        train, val = CR / "cells" / "train.jsonl", CR / "cells" / "val.jsonl"
        run(
            [
                LR_EVAL,
                "--root",
                CR,
                "--policy-spec",
                "default",
                "--cells",
                train,
                val,
                "--repeats",
                "1",
                "--bundle-out",
                bundle,
            ],
            timeout=3600,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        jobs = P2 / "bundle_jobs.jsonl"
        jobs.write_text(
            "".join(
                json.dumps(
                    {"run": r["run"], "space": str(CR / r["space"]), "args": r["args"]}
                )
                + "\n"
                for r in runnable
            )
        )
        spaces = sorted({str(CR / r["space"]) for r in runnable})
        run(
            [
                PY,
                LANE / "lane.py",
                "train-bundle",
                bundle,
                "--train",
                train,
                "--val",
                val,
                "--space",
                *spaces,
                "--jobs",
                jobs,
            ],
            timeout=1800,
        )
        run([PY, LANE / "lane.py", "trace-list", bundle], timeout=1800)
        check = run(
            [
                PY,
                "-S",
                "-s",
                "-P",
                "-c",
                "import numpy, cma, learned_routing.train_cli; print('site-train ok', numpy.__version__, cma.__version__)",
            ],
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": f"{bundle}/site:{bundle}/site-train",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        log(check.stdout.strip())
    # Sync spaces and the job list (new runnable runs, e.g. M1 s2/s3 once their inits exist).
    want = []
    for r in runnable:
        src = CR / r["space"]
        dst = bundle / "spaces" / src.name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            if dry_run:
                log(f"dry-run: would copy {src.name} into the bundle")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        want.append(
            json.dumps(
                {"args": r["args"], "run": r["run"], "space": "spaces/" + src.name},
                sort_keys=True,
            )
            + "\n"
        )
    text = "".join(want)
    jobs_path = bundle / "train_jobs.jsonl"
    if not dry_run and (not jobs_path.exists() or jobs_path.read_text() != text):
        jobs_path.write_text(text)
        log(f"bundle train_jobs.jsonl: {len(want)} runs")
    return bundle


# -- Slurm ----------------------------------------------------------------------------------------
def node_free_mem_mb(nodes: list[str]) -> dict[str, tuple[int, str]]:
    """node -> (allocatable memory MB, state) from scontrol."""
    out = rssh(
        "for n in " + " ".join(nodes) + "; do scontrol show node $n -o; done",
        timeout=120,
    )
    info = {}
    for line in out.splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        name = fields.get("NodeName")
        if not name:
            continue
        real = int(fields.get("RealMemory", 0))
        alloc = int(fields.get("AllocMem", 0))
        info[name] = (real - alloc, fields.get("State", "?"))
    return info


def mem_mb(text: str) -> int:
    m = re.fullmatch(r"(\d+)([GM]?)", text)
    value = int(m.group(1))
    return value * 1024 if m.group(2) in ("G", "") else value


def submit_slurm(
    state: dict, run_id: str, target: dict, seconds: float, dry_run: bool
) -> dict | None:
    cfg = state["config"]
    seconds = min(seconds - 120, cfg["max_time_s"])
    time_s = hms(seconds)
    cmd = [
        LANE / "submit_train.sh",
        "--name",
        cfg["bundle"],
        "--bundle-dir",
        bundle_dir(state),
        "--only",
        run_id,
        "--time",
        time_s,
        "--mem",
        target["mem"],
        "--partitions",
        target["partition"],
    ]
    if target.get("slots"):
        cmd += ["--slots", str(target["slots"])]
    env = {
        "LR_SBATCH_EXTRA": f"--nodelist={target['node']}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if dry_run:
        log(
            f"dry-run: would submit {run_id} to {target['id']} --time {time_s}: {' '.join(map(str, cmd))}"
        )
        return None
    out = run(["bash", *cmd], env=env, timeout=1800).stdout
    match = re.search(rf"^run {re.escape(run_id)}: job (\d+) -> (\S+)$", out, re.M)
    if not match:
        raise RuntimeError(
            f"could not parse submit_train.sh output for {run_id}:\n{out[-1500:]}"
        )
    job = match.group(1)
    log(
        f"submitted {run_id} -> {target['id']} job {job} (--time {time_s}); cancel: ssh {SSH_ALIAS} scancel {job}"
    )
    return {
        "kind": "slurm",
        "target": target["id"],
        "node": target["node"],
        "job": job,
        "returned_remote": match.group(2),
        "time_limit": time_s,
        "submitted": stamp(),
        "slurm_state": "SUBMITTED",
        "cancel": f"ssh {SSH_ALIAS} scancel {job}",
    }


def refresh_jobs(state: dict) -> None:
    jobs = [
        att["job"]
        for e in state["runs"].values()
        for att in e["attempts"][-1:]
        if att["kind"] == "slurm" and att.get("slurm_state", "SUBMITTED") in LIVE
    ]
    if not jobs:
        return
    args = [PY, LANE / "lane.py", "alloc-refresh"]
    for job in jobs:
        args += ["--job", job]
    run(args, timeout=300)
    allocations = {
        str(a["job_id"]): a
        for a in (read_json(CR / "facts" / "remote.json", {}) or {}).get(
            "allocations", []
        )
    }
    for e in state["runs"].values():
        att = current(e)
        if att and att["kind"] == "slurm" and att["job"] in allocations:
            a = allocations[att["job"]]
            att["slurm_state"] = str(a.get("state", "?")).split()[0]
            for key in ("node", "start", "end", "elapsed"):
                if a.get(key):
                    att[f"slurm_{key}"] = a[key]


def fetch_run(state: dict, run_id: str) -> Path:
    dest = remote_run_dir(state, run_id)
    dest.mkdir(parents=True, exist_ok=True)
    src = f"{SSH_ALIAS}:{REMOTE_ROOT}/returned/{state['config']['bundle']}/{run_id}/"
    run(["rsync", "-a", src, f"{dest}/"], timeout=1800, check=False)
    return dest


def ingest(state: dict, run_id: str) -> dict:
    dest = remote_run_dir(state, run_id)
    node = read_json(dest / "node.json", {}) or {}
    image = (node.get("os"), node.get("glibc"))
    if image not in VALIDATED_IMAGES:
        raise RuntimeError(
            f"{run_id}: node image {image} not parity-validated (A9 addendum); not ingested"
        )
    proc = run(
        [
            LR_EVAL,
            "--root",
            CR,
            "--ingest",
            dest,
            "--out",
            dest / "results.ingested.jsonl",
        ],
        timeout=3600,
        check=False,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    summary = read_json_text(proc.stdout)
    (dest / "ingest.json").write_text(proc.stdout)
    if proc.returncode != 0 or summary is None or summary.get("rejected"):
        raise RuntimeError(
            f"{run_id}: ingest failed rc={proc.returncode}: {proc.stdout[-500:]} {proc.stderr[-500:]}"
        )
    return {
        k: summary.get(k)
        for k in ("ingested", "already_cached", "results_appended", "e0_values_merged")
    } | {
        "rejected": len(summary.get("rejected") or []),
        "node_image": list(image),
        "at": stamp(),
    }


def read_json_text(text: str):
    for candidate in (text.strip(), (text.strip().splitlines() or [""])[-1]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# -- local (workstation) --------------------------------------------------------------------------
def local_dir(run_id: str) -> Path:
    return P2 / "local" / run_id


def launch_local(state: dict, run_id: str, target: dict, dry_run: bool) -> dict | None:
    spec = mspec(run_id)
    rdir = local_dir(run_id)
    slots = int(target.get("slots", 20))
    script = rdir / "launch.sh"
    args = " ".join(f"'{a}'" for a in spec["args"])
    pin = " ".join(f"{k}='{v}'" for k, v in NUMERICS_PIN.items())
    text = f"""#!/usr/bin/env bash
# Local lr-train loop for {run_id} (written by p2orch.py; chunks of {state['config']['local_chunk_s']} s, resumable).
cd '{rdir}'
export PYTHONDONTWRITEBYTECODE=1 DYN_LOG=warn {pin}
rc=3
while [ "$rc" -eq 3 ]; do
  '{LR_TRAIN}' --root '{CR}' --space '{CR / spec["space"]}' --cells '{CR}/cells/train.jsonl' --val '{CR}/cells/val.jsonl' \\
    --run-dir '{rdir}' --slots {slots} --num-slots 20 --max-wall-seconds {state['config']['local_chunk_s']} {args}
  rc=$?
  echo "$(date -Is) chunk exit $rc" >> '{rdir}/chunks.log'
done
echo "$rc" > '{rdir}/exit_code'
"""
    if dry_run:
        log(f"dry-run: would launch {run_id} locally ({slots} slots) in {rdir}")
        return None
    rdir.mkdir(parents=True, exist_ok=True)
    script.write_text(text)
    script.chmod(0o755)
    with open(rdir / "node.log", "a") as out:
        # Own session (survives the orchestrator); the recorded PID is the loop's bash itself.
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (rdir / "pid").write_text(str(proc.pid))
    log(f"launched {run_id} locally pid {proc.pid} ({slots} slots) in {rdir}")
    return {
        "kind": "local",
        "target": target["id"],
        "pid": proc.pid,
        "run_dir": str(rdir),
        "submitted": stamp(),
        "launch": str(script),
    }


def pid_alive(att: dict) -> bool:
    pid = att.get("pid")
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False
    return att["launch"] in cmdline


# -- evaluation counts ----------------------------------------------------------------------------
def counts(path: Path | None) -> dict | None:
    if path is None or not (path / "status.json").exists():
        return None
    status = read_json(path / "status.json", {}) or {}
    gens = vals = reevals = 0
    val_candidates = 0
    if (path / "history.jsonl").exists():
        for line in (path / "history.jsonl").read_text().splitlines():
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            if h.get("kind") == "gen":
                gens += 1
                reevals += 1 if h.get("rank_change") else 0
            elif h.get("kind") == "val":
                val_candidates += 1
                vals += 1 if h.get("which") == "mean" else 0
    return {
        "phase": status.get("phase"),
        "generation": status.get("generation"),
        "fevals": status.get("fevals"),
        "budget_evals": status.get("budget_evals"),
        "tasks_requested": status.get("tasks_requested"),
        "replays_fresh": status.get("replays_fresh"),
        "wall_s": status.get("wall_s"),
        "gen_lines": gens,
        "val_checkpoints": vals,
        "val_candidates": val_candidates,
        "reeval_generations": reevals,
        "best_train_objective": status.get("best_train_objective"),
        "selected_val_objective": status.get("selected_val_objective"),
        "time": status.get("time"),
    }


def budget_report(state: dict) -> dict:
    rows = {rid: e.get("evals") for rid, e in state["runs"].items() if e.get("evals")}
    done = {rid: c for rid, c in rows.items() if state["runs"][rid]["status"] == "done"}
    trained = {
        rid: c for rid, c in done.items() if state["runs"][rid]["kind"] == "train"
    }
    report: dict = {"runs": rows}
    # Every tier is checked on its own (tier A, and e.g. the AIS league's tier in its own instance).
    for tier in sorted({state["runs"][rid]["tier"] for rid in trained}):
        runs = {
            rid: c for rid, c in trained.items() if state["runs"][rid]["tier"] == tier
        }
        # The UH re-evaluation is a diagnostic outside B: lr-train skips it in a generation whose
        # chunk deadline falls during it, so its count is reported but not part of the check.
        distinct = sorted({(c["fevals"], c["val_checkpoints"]) for c in runs.values()})
        report[f"tier_{tier}_done"] = len(runs)
        report[f"tier_{tier}_signatures_(fevals,val_checkpoints)"] = distinct
        report[f"tier_{tier}_equal_budget"] = len(distinct) <= 1 and all(
            v == (400, 5) for v in distinct
        )
        report[f"tier_{tier}_reeval_generations"] = {
            rid: c["reeval_generations"] for rid, c in runs.items()
        }
    report["per_policy_restarts_done"] = {
        k: sum(1 for rid in trained if state["runs"][rid]["policy_key"] == k)
        for k in sorted({state["runs"][rid]["policy_key"] for rid in trained})
    }
    return report


# -- tick -----------------------------------------------------------------------------------------
def collect(state: dict, dry_run: bool) -> None:
    try:
        refresh_jobs(state)
    except Exception as exc:  # cluster unreachable: keep going with local runs
        log(f"warning: sacct refresh failed: {exc}")
    for rid, e in sorted(state["runs"].items()):
        att = current(e)
        if att is None or e["status"] not in ("running",):
            continue
        if att["kind"] == "slurm":
            st = att.get("slurm_state", "SUBMITTED")
            if st in (
                "RUNNING",
                "COMPLETED",
                "TIMEOUT",
                "FAILED",
                "CANCELLED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "DEADLINE",
                "PREEMPTED",
                "BOOT_FAIL",
            ):
                try:
                    fetch_run(state, rid)
                except Exception as exc:
                    log(f"warning: fetch {rid} failed: {exc}")
            e["evals"] = counts(run_dir(state, rid)) or e.get("evals")
            if (
                st == "PENDING"
                and age_s(att["submitted"]) > PENDING_LIMIT_S
                and not dry_run
            ):
                # Pinned node did not start the job (memory taken meanwhile): cancel our own exact
                # job ID and place the run again.
                try:
                    reason = f"p2orch: pending > {PENDING_LIMIT_S} s"
                    scancel(att["job"], f"{reason} ({rid})")
                    att["slurm_state"] = "CANCELLED"
                    att["cancelled_by"] = reason
                    e["status"] = "queued"
                    e["resume"] = (
                        remote_run_dir(state, rid)
                        / "runs"
                        / "train"
                        / rid
                        / "checkpoint.pkl"
                    ).exists()
                    log(
                        f"{rid}: job {att['job']} pending too long on {att['node']}; cancelled (exact ID) and requeued"
                    )
                except Exception as exc:
                    log(f"warning: scancel {att['job']} failed: {exc}")
                continue
            if st in LIVE:
                continue
            refused = (
                read_json(remote_run_dir(state, rid) / "image_refused.json", {}) or {}
            )
            if str(refused.get("job")) == str(att["job"]):
                # node_train.sh refused the node's OS/glibc image before staging (A9 addendum):
                # nothing ran and the checkpoint is untouched. Take the node out of rotation (a
                # parity smoke must validate the image first; `targets --enable` restores it).
                att["image_refused"] = refused
                for t in state["targets"]:
                    if t["id"] == att["target"]:
                        t["enabled"] = False
                        t["disabled_reason"] = (
                            f"image {refused.get('image')!r} not parity-validated "
                            f"(job {att['job']}, {stamp()})"
                        )
                e["status"] = "queued"
                e["resume"] = (
                    remote_run_dir(state, rid)
                    / "runs"
                    / "train"
                    / rid
                    / "checkpoint.pkl"
                ).exists()
                log(
                    f"{rid}: job {att['job']} refused node image {refused.get('image')!r} "
                    f"(not parity-validated); {att['target']} disabled, run requeued"
                )
                continue
            timing = read_json(remote_run_dir(state, rid) / "timing.json", {}) or {}
            phase = (e.get("evals") or {}).get("phase")
            att["end_phase"], att["rc"] = phase, timing.get("rc")
            if phase == "done":
                if dry_run:
                    log(f"dry-run: would ingest {rid}")
                    continue
                try:
                    att["ingest"] = ingest(state, rid)
                    e["status"] = "done"
                    log(
                        f"{rid} done ({e['evals']['fevals']} evals); ingested {att['ingest']}"
                    )
                except Exception as exc:
                    e["status"] = "needs-attention"
                    e["error"] = str(exc)[:1000]
                    log(f"{rid}: {exc}")
            elif phase in ("paused", "running") or timing.get("rc") == 3:
                e["status"] = "queued"
                e["resume"] = True
                log(
                    f"{rid} paused at generation {(e.get('evals') or {}).get('generation')} (job {att['job']} {st}); queued to resume"
                )
            else:
                e["status"] = (
                    "queued" if len(e["attempts"]) < MAX_ATTEMPTS else "failed"
                )
                e["resume"] = bool(e.get("evals"))
                log(
                    f"{rid}: job {att['job']} ended {st} without a done/paused status; {e['status']}"
                )
        else:
            e["evals"] = counts(Path(att["run_dir"])) or e.get("evals")
            if pid_alive(att):
                continue
            phase = (e.get("evals") or {}).get("phase")
            att["end_phase"] = phase
            if phase == "done":
                e["status"] = "done"
                att["ingest"] = {"local": "results are in the campaign cache already"}
                log(f"{rid} done locally ({e['evals']['fevals']} evals)")
            else:
                e["status"] = (
                    "queued" if len(e["attempts"]) < MAX_ATTEMPTS else "failed"
                )
                e["resume"] = True
                log(
                    f"{rid}: local pid {att['pid']} exited in phase {phase}; {e['status']}"
                )


def eligible(state: dict) -> list[str]:
    max_wave = state["config"]["max_wave"]
    runs = state["runs"]
    queued = [
        rid
        for rid, e in runs.items()
        if e["status"] == "queued" and e["wave"] <= max_wave
    ]
    out = []
    for rid in sorted(queued, key=lambda r: (runs[r]["wave"], runs[r]["order"])):
        w = runs[rid]["wave"]
        # Breadth-first: every runnable run of an earlier wave must have been started once.
        if any(
            e["wave"] < w and e["status"] == "queued" and not e["attempts"]
            for e in runs.values()
        ):
            continue
        out.append(rid)
    return out


def busy_targets(state: dict) -> set[str]:
    return {
        current(e)["target"]
        for e in state["runs"].values()
        if e["status"] == "running" and current(e)
    }


def assign(state: dict, dry_run: bool) -> list:
    cfg = state["config"]
    todo = eligible(state)
    if not todo:
        return []
    held, reasons = foreign_view(state)
    # config "yield_own_from_wave" W: while another instance has eligible runs within its yield cap,
    # hold back only this instance's runs of wave >= W (instead of everything), so two instances can
    # interleave priorities without both waiting on each other.
    own_from = cfg.get("yield_own_from_wave")

    def allowed(rid: str) -> bool:
        return not reasons or (
            own_from is not None and state["runs"][rid]["wave"] < own_from
        )

    if reasons:
        todo = [r for r in todo if allowed(r)]
        log(
            f"yielding to other orchestrators{f' (waves >= {own_from})' if own_from is not None else ''}: {'; '.join(reasons)}"
        )
        if not todo:
            return []
    busy = busy_targets(state) | held
    free = [
        t
        for t in sorted(state["targets"], key=lambda t: (t["rank"], t["id"]))
        if t.get("enabled", True) and t["id"] not in busy
    ]
    ok, left, why = cluster_window(cfg)
    slurm_free = [t for t in free if t["kind"] == "slurm"]
    mem = {}
    if ok and slurm_free:
        try:
            mem = node_free_mem_mb([t["node"] for t in slurm_free])
        except Exception as exc:
            ok, why = False, f"scontrol failed: {exc}"
    placed = []
    for t in free:
        if not todo:
            break
        if t["kind"] == "slurm":
            if not ok:
                continue
            avail, nstate = mem.get(t["node"], (0, "?"))
            if avail < mem_mb(t["mem"]) or any(s in nstate for s in BAD_NODE_STATES):
                continue
        # The first queued run this target can take: a run that started on the workstation
        # finishes there; a remote run moves to the workstation only with its checkpoint.
        rid = None
        for cand in todo:
            prev = current(state["runs"][cand])
            if t["kind"] == "slurm" and prev is not None and prev["kind"] == "local":
                continue
            rid = cand
            break
        if rid is None:
            continue
        e = state["runs"][rid]
        prev = current(e)
        if (
            t["kind"] == "local"
            and prev is not None
            and prev["kind"] == "slurm"
            and e.get("resume")
        ):
            if not migrate(state, rid, dry_run):
                continue
        ensure_bundle(state, dry_run)
        att = (
            submit_slurm(state, rid, t, left, dry_run)
            if t["kind"] == "slurm"
            else launch_local(state, rid, t, dry_run)
        )
        todo.remove(rid)
        placed.append((rid, t["id"]))
        if att is not None:
            e["attempts"].append(att)
            e["status"] = "running"
            e.pop("resume", None)
            e.pop("rebalance_to", None)
            # Starting the last unstarted run of a wave unlocks the next wave in this same pass.
            placed_ids = {p[0] for p in placed}
            todo = [r for r in eligible(state) if r not in placed_ids and allowed(r)]
    if not ok:
        log(f"Slurm submissions held: {why}")
    return placed


def migrate(state: dict, rid: str, dry_run: bool) -> bool:
    """Continue a remote run on the workstation: ingest its cache records, copy its run dir."""
    src = remote_run_dir(state, rid) / "runs" / "train" / rid
    dst = local_dir(rid)
    if not (src / "checkpoint.pkl").exists():
        return False
    if dry_run:
        log(f"dry-run: would migrate {rid} {src} -> {dst}")
        return True
    if (dst / "checkpoint.pkl").exists():
        log(f"{rid}: {dst} already holds a checkpoint; not overwriting")
        return False
    state["runs"][rid].setdefault("migrations", []).append(
        {"at": stamp(), "ingest": ingest(state, rid)}
    )
    shutil.copytree(src, dst, dirs_exist_ok=True)
    log(f"migrated {rid} to the workstation ({src} -> {dst})")
    return True


# -- rebalance: move a run the cutoff would pause onto an idle faster Slurm node -------------------
def run_gens(run_id: str) -> int:
    """Generations the run's budget allows (budget_evals // popsize)."""
    args = [str(a) for a in mspec(run_id)["args"]]

    def value(flag: str, default: int) -> int:
        return int(args[args.index(flag) + 1]) if flag in args else default

    return value("--budget-evals", 400) // value("--popsize", 16)


def class_sec_per_gen(state: dict) -> dict[str, float]:
    """Measured lr-train wall per generation (validation included) by node class: the median of
    wall_s / generation over finished single-attempt training runs."""
    cls = {t["id"]: t.get("class") for t in state["targets"]}
    per: dict[str, list[float]] = {}
    for e in state["runs"].values():
        ev = e.get("evals") or {}
        if (
            e["status"] != "done"
            or e["kind"] != "train"
            or len(e["attempts"]) != 1
            or not ev.get("generation")
            or not ev.get("wall_s")
        ):
            continue
        per.setdefault(cls.get(e["attempts"][0]["target"]), []).append(
            ev["wall_s"] / ev["generation"]
        )
    return {c: statistics.median(v) for c, v in per.items() if c}


def rebalance(state: dict, dry_run: bool) -> list:
    """When nothing eligible is queued and a faster Slurm node is idle, cancel (exact job ID) one
    running Slurm run on a slower node that the cutoff would pause: preferably one the idle node
    finishes before its own pause, otherwise the one that completes the most extra generations
    there (at least MIN_GENS_SAVED_PARTIAL). The next tick's collect() requeues it with resume
    and assign() places it on the idle node (lowest rank first)."""
    cfg = state["config"]
    ok, left, _ = cluster_window(cfg)
    if not ok or eligible(state):
        return []
    by_id = {t["id"]: t for t in state["targets"]}
    reserved = {
        e["rebalance_to"] for e in state["runs"].values() if e.get("rebalance_to")
    }
    # Never move onto a node another orchestrator instance holds.
    busy = busy_targets(state) | reserved | foreign_view(state)[0]
    free = [
        t
        for t in sorted(state["targets"], key=lambda t: (t["rank"], t["id"]))
        if t["kind"] == "slurm" and t.get("enabled", True) and t["id"] not in busy
    ]
    spg = class_sec_per_gen(state)
    free = [t for t in free if spg.get(t.get("class"))]
    if not free:
        return []
    # lr-train seconds left before a running job (or one submitted now) pauses at the cutoff.
    budget = left - 120 - TAIL_MARGIN_S
    wait = cfg.get("loop_interval_s", 300) + 60 + RESTAGE_S
    mem = node_free_mem_mb([t["node"] for t in free])
    moves = []
    for t in free:
        avail, nstate = mem.get(t["node"], (0, "?"))
        if avail < mem_mb(t["mem"]) or any(s in nstate for s in BAD_NODE_STATES):
            continue
        fast = spg[t["class"]]
        best = None
        for rid, e in state["runs"].items():
            att = current(e)
            if (
                e["status"] != "running"
                or e.get("rebalance_to")
                or e["wave"] > cfg["max_wave"]
                or att is None
                or att["kind"] != "slurm"
                or att.get("slurm_state") != "RUNNING"
                or rid in {m["run"] for m in moves}
            ):
                continue
            src = by_id.get(att["target"])
            if src is None or src["rank"] <= t["rank"]:
                continue
            ev = e.get("evals") or {}
            g = ev.get("generation") or 0
            slow = (
                ev["wall_s"] / g
                if g >= 3 and ev.get("wall_s")
                else spg.get(src.get("class"))
            )
            if slow is None or slow < MIN_SPEEDUP * fast:
                continue
            remaining = run_gens(rid) - g
            here = (
                budget / slow
            )  # generations it completes before its pause where it is
            if here >= remaining:
                continue  # finishes where it is
            # +1 generation: the one in flight when the job is cancelled.
            finishes = wait + (remaining + 1) * fast <= budget - REBALANCE_MARGIN_S
            there = remaining if finishes else max(0.0, (budget - wait) / fast - 1)
            saved = there - here
            if saved < (MIN_GENS_SAVED if finishes else MIN_GENS_SAVED_PARTIAL):
                continue
            # Prefer a move that lets the run finish before the cutoff, then more generations.
            if best is None or (finishes, saved) > (
                best["finishes"],
                best["gens_saved"],
            ):
                best = {
                    "run": rid,
                    "job": att["job"],
                    "from": src["id"],
                    "to": t["id"],
                    "generation": g,
                    "remaining": remaining,
                    "sec_per_gen_from": round(slow),
                    "sec_per_gen_to": round(fast),
                    "finishes": finishes,
                    "gens_saved": round(saved, 1),
                    "eta_to": stamp(
                        now() + dt.timedelta(seconds=wait + (remaining + 1) * fast)
                    )
                    if finishes
                    else None,
                }
        if best:
            moves.append(best)
    for m in moves:
        if dry_run:
            log(f"dry-run: would rebalance {json.dumps(m, sort_keys=True)}")
            continue
        e = state["runs"][m["run"]]
        att = current(e)
        reason = f"p2orch rebalance -> {m['to']} at {stamp()}"
        try:
            scancel(m["job"], f"{reason} ({m['run']} at generation {m['generation']})")
        except Exception as exc:
            log(f"warning: rebalance scancel {m['job']} failed: {exc}")
            continue
        att["cancelled_by"] = reason
        e["rebalance_to"] = m["to"]
        e.setdefault("rebalances", []).append({"at": stamp(), **m})
        log(
            f"rebalance: cancelled {m['run']} job {m['job']} (exact ID) on {m['from']} at generation "
            f"{m['generation']} to resume on idle {m['to']} next tick: {json.dumps(m, sort_keys=True)}"
        )
    return moves


def tick(dry_run: bool = False) -> dict:
    with state_lock():
        state = load_state()
        sync_runs(state)
        collect(state, dry_run)
        if (bundle_dir(state) / "MANIFEST.json").exists():
            ensure_bundle(
                state, dry_run
            )  # keep spaces and train_jobs.jsonl in step with MANIFEST
        placed = assign(state, dry_run)
        if state["config"].get("rebalance"):
            try:
                rebalance(state, dry_run)
            except Exception as exc:  # never let the optional step break a tick
                log(f"warning: rebalance failed: {exc}")
        state["budget"] = budget_report(state)
        state["updated"] = stamp()
        state["cluster_window"] = cluster_window(state["config"])[2]
        if not dry_run:
            write_json(STATE, state)
            write_json(P2 / "budget.json", state["budget"])
        drift = [state["runs"].get(f"drift-{r}") for r in ("l1", "l3", "n4", "n8")]
        if (
            all(e is not None and e["status"] == "done" for e in drift)
            and not (P2 / "drift" / "matrix.json").exists()
        ):
            log(
                "all LR-15 drift runs done: run `p2orch.py drift-matrix` (not automatic)"
            )
        return {"placed": placed, "statuses": status_counts(state)}


def status_counts(state: dict) -> dict:
    out = {}
    for e in state["runs"].values():
        out[e["status"]] = out.get(e["status"], 0) + 1
    return out


# -- selection (LR-10) ----------------------------------------------------------------------------
def history_val_lines(path: Path | None) -> list[dict]:
    if path is None or not (path / "history.jsonl").exists():
        return []
    out = []
    for line in (path / "history.jsonl").read_text().splitlines():
        try:
            h = json.loads(line)
        except json.JSONDecodeError:
            continue
        if h.get("kind") == "val" and h.get("val_objective") is not None:
            out.append(h)
    return out


def select(state: dict, tier: str, kind: str, allow_partial: bool) -> dict:
    groups: dict[str, list[str]] = {}
    for spec in manifest()["runs"]:
        if spec["tier"] == tier and spec["kind"] == kind:
            groups.setdefault(
                spec["policy_key"] if kind != "drift" else spec["run"], []
            ).append(spec["run"])
    result = {
        "tier": tier,
        "kind": kind,
        "rule": "argmax lr-train validation objective (val k = 0..2) over the "
        "policy's restarts x validated checkpoints (best-so-far sample and CMA mean every 5 generations), LR-10",
        "written": stamp(),
        "policies": {},
    }
    specs = []
    for key, rids in sorted(groups.items()):
        candidates, included, missing = [], [], []
        for rid in rids:
            e = state["runs"].get(rid, {})
            if e.get("status") != "done" and not allow_partial:
                missing.append(rid)
                continue
            lines = history_val_lines(run_dir(state, rid))
            if lines:
                included.append(rid)
            for h in lines:
                candidates.append((h["val_objective"], rid, h))
        if not candidates:
            result["policies"][key] = {
                "selected": None,
                "runs_included": included,
                "runs_missing": missing,
            }
            continue
        value, rid, h = max(candidates, key=lambda c: c[0])
        spec = dict(h["spec"])
        spec["name"] = f"{key}@p2-{rid}-g{h['generation']}-{h['which']}"
        specs.append(spec)
        result["policies"][key] = {
            "selected": {
                "run": rid,
                "generation": h["generation"],
                "which": h["which"],
                "val_objective": value,
                "policy_sha": h["policy_sha"],
                "values": h["values"],
                "spec": spec,
            },
            "val_candidates_counted": len(candidates),
            "runs_included": included,
            "runs_missing": missing,
            "complete": not missing,
        }
    out_dir = P2 / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"tier{tier}" if kind != "drift" else "drift"
    write_json(out_dir / f"{name}.json", result)
    (out_dir / f"{name}.specs.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in specs)
    )
    return result


def drift_matrix(state: dict, slots: int) -> dict:
    sel = select(state, "A", "drift", allow_partial=False)
    if (
        any(v.get("selected") is None for v in sel["policies"].values())
        or len(sel["policies"]) != 4
    ):
        raise SystemExit("drift runs not all done")
    out_dir = P2 / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = P2 / "selection" / "drift.specs.jsonl"
    results = out_dir / "val_k3_10.jsonl"
    run(
        [
            LR_EVAL,
            "--root",
            CR,
            "--policy-spec",
            "default",
            specs,
            "--cells",
            CR / "cells" / "val.jsonl",
            "--repeats",
            "8",
            "--repeat-offset",
            "3",
            "--slots",
            str(slots),
            "--no-per-request",
            "--out",
            results,
            "--quiet",
        ],
        timeout=7200,
        env={"PYTHONDONTWRITEBYTECODE": "1", "DYN_LOG": "warn"},
    )
    regimes = {"l1": "-L1$", "l3": "-L3$", "n4": "-n4-", "n8": "-n8-"}
    recs = [
        json.loads(line) for line in results.read_text().splitlines() if line.strip()
    ]
    ref = {
        (r["cell_id"], r["repeat"]): r["goodput_rps_window"]
        for r in recs
        if r["policy_name"] == "default@defaults"
    }
    score: dict = {}
    for r in recs:
        if r["policy_name"] == "default@defaults" or r.get("error"):
            continue
        m, m0 = r["goodput_rps_window"], ref.get((r["cell_id"], r["repeat"]))
        if m is None or m0 is None:
            continue
        v = max(min(math.log((m + 1e-3) / (m0 + 1e-3)), math.log(3)), -math.log(3))
        score.setdefault(r["policy_name"], {}).setdefault(r["cell_id"], []).append(v)
    name_of = {f"drift-{k}": f"drift-{k}@p2-drift-{k}" for k in regimes}
    matrix = {}
    for trained in regimes:
        pname = next((p for p in score if p.startswith(f"drift-{trained}@")), None)
        row = {}
        for evaluated, pattern in regimes.items():
            cells = [c for c in score.get(pname, {}) if re.search(pattern, c)]
            row[evaluated] = (
                (
                    sum(sum(score[pname][c]) / len(score[pname][c]) for c in cells)
                    / len(cells)
                )
                if cells
                else None
            )
        matrix[trained] = row
    mde = 0.038
    losses = {}
    for a, b in (("l1", "l3"), ("l3", "l1"), ("n4", "n8"), ("n8", "n4")):
        if matrix[b][b] is not None and matrix[a][b] is not None:
            losses[f"{a}-tuned on {b}"] = matrix[b][b] - matrix[a][b]
    out = {
        "written": stamp(),
        "matrix_rows_trained_cols_evaluated": matrix,
        "cross_play_losses": losses,
        "mde": mde,
        "within_mde": all(abs(v) <= mde for v in losses.values()),
        "rule": "LR-15: if every cross-play loss is within the MDE, deprioritize M2 (gate cut 5)",
        "replicates": "val k = 3..10 (fresh)",
        "selection": sel,
        "names": name_of,
    }
    write_json(out_dir / "matrix.json", out)
    return out


# -- CLI ------------------------------------------------------------------------------------------
def cmd_status(args) -> int:
    state = load_state()
    print(
        f"updated {state.get('updated')}  slurm: {cluster_window(state['config'])[2]}  max_wave {state['config']['max_wave']}"
    )
    for rid, e in sorted(
        state["runs"].items(), key=lambda kv: (kv[1]["wave"], kv[1]["order"])
    ):
        if e["status"] in ("pending", "blocked") and not args.all:
            continue
        att = current(e) or {}
        ev = e.get("evals") or {}
        where = att.get("target", "-")
        job = att.get("job") or att.get("pid") or "-"
        print(
            f"{rid:18s} w{e['wave']} {e['status']:15s} {where:22s} {str(job):9s} {att.get('slurm_state', ''):10s} "
            f"gen {ev.get('generation', '-')!s:>3} fevals {ev.get('fevals', '-')!s:>4} val {ev.get('selected_val_objective')}"
        )
    print(json.dumps(status_counts(state)))
    return 0


def cmd_loop(args) -> int:
    deadline = time.time() + args.max_wall_s
    while time.time() < deadline:
        try:
            result = tick(dry_run=False)
            log(f"tick: placed {result['placed']} statuses {result['statuses']}")
            state = load_state()
            active = [
                e
                for e in state["runs"].values()
                if e["status"] in ("queued", "running")
                and e["wave"] <= state["config"]["max_wave"]
            ]
            if not active:
                log("nothing queued or running up to max_wave; loop exits")
                return 0
        except Exception:
            log("tick failed:\n" + traceback.format_exc())
        time.sleep(max(30, args.interval))
    return 0


def cmd_cancel(args) -> int:
    with state_lock():
        state = load_state()
        e = state["runs"][args.run]
        att = current(e)
        if not att or e["status"] != "running":
            raise SystemExit(f"{args.run} has no running attempt")
        if att["kind"] == "slurm":
            scancel(att["job"], f"p2orch cancel {args.run} at {stamp()}")
            log(f"cancelled {args.run} job {att['job']} (exact ID)")
        else:
            os.kill(int(att["pid"]), 15)
            log(
                f"sent SIGTERM to {args.run} local pid {att['pid']} (exact PID; lr-train checkpoints every generation)"
            )
        e["status"] = "queued"
        e["resume"] = True
        write_json(STATE, state)
    return 0


def cmd_rebalance(args) -> int:
    with state_lock():
        state = load_state()
        if args.enable or args.disable:
            state["config"]["rebalance"] = bool(args.enable)
            log(
                f"config rebalance = {state['config']['rebalance']} (applied by every tick)"
            )
            write_json(STATE, state)
            return 0
        moves = rebalance(state, args.dry_run)
        if not args.dry_run:
            write_json(STATE, state)
    print(json.dumps(moves, indent=1))
    return 0


TARGET_KEYS = {
    "slurm": {"id", "kind", "node", "class", "partition", "mem", "rank", "hours"},
    "local": {"id", "kind", "class", "slots", "rank", "hours"},
}


def cmd_targets(args) -> int:
    with state_lock():
        state = load_state()
        for text in args.add or []:
            # Only add a node whose OS/glibc image passed a parity smoke (A9 addendum).
            new = json.loads(text)
            missing = TARGET_KEYS.get(new.get("kind"), {"kind"}) - new.keys()
            if missing:
                raise SystemExit(f"target {text}: missing keys {sorted(missing)}")
            if any(t["id"] == new["id"] for t in state["targets"]):
                raise SystemExit(f"target {new['id']} already exists")
            state["targets"].append(new)
            log(f"added target {new['id']}: {json.dumps(new, sort_keys=True)}")
        for t in state["targets"]:
            if args.enable and t["id"] in args.enable:
                t["enabled"] = True
                t.pop("disabled_reason", None)
            if args.disable and t["id"] in args.disable:
                t["enabled"] = False
        write_json(STATE, state)
        for t in state["targets"]:
            print(json.dumps(t, sort_keys=True))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="p2orch.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--bundle", default="p2a")
    i.add_argument("--max-wave", type=int, default=3)
    i.add_argument(
        "--cutoff",
        default=os.environ.get("LR_CUTOFF") or None,
        metavar="'<weekday> HH:MM'",
        help="optional weekly job-end cutoff in LR_TZ local time (default LR_CUTOFF; unset: none, "
        "jobs are bounded by --max-time-s only)",
    )
    i.add_argument(
        "--cluster-reopen",
        default=os.environ.get("LR_CLUSTER_REOPEN") or None,
        help="no Slurm submission between the cutoff and this time (maintenance window)",
    )
    i.add_argument("--min-job-s", type=int, default=1800)
    i.add_argument("--max-time-s", type=int, default=12 * 3600)
    i.add_argument(
        "--yield-to",
        action="append",
        metavar="STATE[:MAX_WAVE]",
        help="another orchestrator's state.json: never place on its running targets, and place "
        "nothing while it has an eligible queued run up to MAX_WAVE (default: any wave)",
    )
    i.add_argument(
        "--targets-from",
        metavar="STATE",
        help="copy the target list of another orchestrator's state.json (new state only)",
    )
    i.add_argument(
        "--yield-own-from-wave",
        type=int,
        default=None,
        metavar="W",
        help="while yielding (--yield-to), hold back only this instance's runs of wave >= W",
    )
    i.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--all", action="store_true")
    t = sub.add_parser("tick")
    t.add_argument("--dry-run", action="store_true")
    lo = sub.add_parser("loop")
    lo.add_argument("--interval", type=int, default=300)
    lo.add_argument("--max-wall-s", type=float, default=36 * 3600)
    sub.add_parser("budget")
    se = sub.add_parser("select")
    se.add_argument("--tier", default="A")
    se.add_argument("--kind", default="train", choices=("train", "drift"))
    se.add_argument("--allow-partial", action="store_true")
    dm = sub.add_parser("drift-matrix")
    dm.add_argument("--slots", type=int, default=20)
    mg = sub.add_parser("migrate")
    mg.add_argument("run")
    c = sub.add_parser("cancel")
    c.add_argument("run")
    rb = sub.add_parser("rebalance")
    rb.add_argument("--dry-run", action="store_true")
    rb_mode = rb.add_mutually_exclusive_group()
    rb_mode.add_argument(
        "--enable", action="store_true", help="let every tick rebalance"
    )
    rb_mode.add_argument("--disable", action="store_true")
    tg = sub.add_parser("targets")
    tg.add_argument("--enable", nargs="*")
    tg.add_argument("--disable", nargs="*")
    tg.add_argument(
        "--add",
        nargs="*",
        help="JSON target objects to append (same keys as the LR_P2ORCH_TARGETS entries)",
    )
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if args.cmd == "init":
        state = init_state(args)
        print(json.dumps(status_counts(state)))
        return 0
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "tick":
        print(json.dumps(tick(dry_run=args.dry_run)))
        return 0
    if args.cmd == "loop":
        return cmd_loop(args)
    if args.cmd == "budget":
        state = load_state()
        print(json.dumps(budget_report(state), indent=1))
        return 0
    if args.cmd == "select":
        print(
            json.dumps(
                select(load_state(), args.tier, args.kind, args.allow_partial), indent=1
            )
        )
        return 0
    if args.cmd == "drift-matrix":
        print(json.dumps(drift_matrix(load_state(), args.slots), indent=1))
        return 0
    if args.cmd == "migrate":
        with state_lock():
            state = load_state()
            ok = migrate(state, args.run, dry_run=False)
            write_json(STATE, state)
        return 0 if ok else 1
    if args.cmd == "cancel":
        return cmd_cancel(args)
    if args.cmd == "rebalance":
        return cmd_rebalance(args)
    if args.cmd == "targets":
        return cmd_targets(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
