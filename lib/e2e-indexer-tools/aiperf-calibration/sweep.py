#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resumable gated sweep driver. Run inside the hold (srun step), stdlib only.

Usage: CALIB_ROOT=DIR sweep.py TRIALS.tsv  (scripts in $CALIB_ROOT/scripts, results in $CALIB_ROOT/results)
TSV columns: name, series, offered, stub_cpus, aiperf_cpus, aiperf_env, aiperf_args.
Skips trials whose summary already exists. Within a series (ordered by appearance), once a
point fails (total achieved < FAIL_RATIO x offered x instances, nonzero rc, or errors),
the remaining points of that series are skipped. Touch $R/STOP to stop between trials.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

R = os.environ["CALIB_ROOT"]
FAIL_RATIO = float(os.environ.get("FAIL_RATIO", "0.95"))


def load_summary(name: str) -> dict | None:
    p = os.path.join(R, "results", name, "summary.txt")
    if not os.path.exists(p):
        return None
    try:
        return json.loads(open(p).read().strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def failed(s: dict, offered: str) -> bool:
    if s.get("error") or s.get("rc") not in (0, "0"):
        return True
    for inst in s.get("per_instance", []):
        if inst.get("err_pct"):
            return True
    try:
        off = float(offered) * s["instances"]
    except (ValueError, KeyError):
        return False
    return s["total_rps"] < FAIL_RATIO * off


def main() -> None:
    rows = []
    for line in open(sys.argv[1]):
        if not line.strip() or line.startswith("#"):
            continue
        rows.append(line.rstrip("\n").split("\t"))
    failed_series: set[str] = set()
    log = open(os.path.join(R, "logs", "sweep-gate.log"), "a")
    for name, series, offered, stub_cpus, aiperf_cpus, aiperf_env, aiperf_args in rows:
        if os.path.exists(os.path.join(R, "STOP")):
            print("STOP present; exiting", flush=True)
            return
        s = load_summary(name)
        if s is None and series in failed_series:
            print(f"skip {name}: series {series} already failed", flush=True)
            continue
        if s is None:
            env = dict(os.environ)
            # Tokens of the form STUB_*=V in the env column configure the stub, not AIPerf.
            stub_env = dict(
                t.split("=", 1) for t in aiperf_env.split() if t.startswith("STUB_")
            )
            aiperf_env = " ".join(
                t for t in aiperf_env.split() if not t.startswith("STUB_")
            )
            env.update(stub_env)
            env.update(
                NAME=name,
                STUB_CPUS=stub_cpus,
                AIPERF_CPUS=aiperf_cpus,
                AIPERF_ENV=aiperf_env,
                AIPERF_ARGS=aiperf_args,
                OFFERED=offered,
                GATE_WAIT_S=os.environ.get("GATE_WAIT_S", "600"),
            )
            t0 = time.time()
            print(f"run {name} ...", flush=True)
            p = subprocess.run(
                [
                    "bash",
                    f"{R}/scripts/node-gate.sh",
                    "bash",
                    f"{R}/scripts/run_trial.sh",
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            gate = [ln for ln in p.stdout.splitlines() if ln.startswith("node-gate:")]
            log.write(f"{name} rc={p.returncode} {' '.join(gate)}\n")
            log.flush()
            os.makedirs(os.path.join(R, "results", name), exist_ok=True)
            with open(os.path.join(R, "results", name, "driver.log"), "w") as f:
                f.write(p.stdout)
            print(
                f"  rc={p.returncode} wall={time.time() - t0:.0f}s {' '.join(gate)}",
                flush=True,
            )
            if p.returncode == 75:
                # Gate verdict not clean: discard this trial's summary so it reruns later.
                sp = os.path.join(R, "results", name, "summary.txt")
                if os.path.exists(sp):
                    os.rename(sp, sp + ".dirty")
                continue
            s = load_summary(name)
        if s is None:
            print(f"  {name}: no summary", flush=True)
            failed_series.add(series)
            continue
        brief = {
            k: s.get(k)
            for k in ("total_rps", "total_aiperf_cores", "cpu_ms_per_req", "stub_cores")
        }
        print(f"  {name}: {brief}", flush=True)
        if failed(s, offered):
            failed_series.add(series)


if __name__ == "__main__":
    main()
