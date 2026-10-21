# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive aiperf inside the loadgen pod and pull back its summary.

aiperf runs detached (``nohup``) and is polled for an exit-code file, so a
dropped ``kubectl exec`` stream, or an exec that fails while the API server
or kubelet restarts, never loses a run. Only aiperf's summary export leaves
the pod; the full aiperf artifacts stay in the pod for inspection.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import shlex
import subprocess
import time
from typing import Any

from ..router.common import get_common_aiperf_flags
from .kube import Kube, KubeError
from .stats import LATENCY_METRICS, LATENCY_STATS, THROUGHPUT_METRICS

log = logging.getLogger(__name__)

RESULTS_ROOT = "/results"
# aiperf's per-run summary: means and percentiles of every metric.
SUMMARY_FILE = "profile_export_aiperf.json"

# aiperf renames its processes ("aiperf system_controller", "aiperf
# worker_<id>", ...), so "aiperf profile" matches only the wrapper shell.
# Killing just that leaves the real aiperf running, orphaned, still sending
# load into whatever runs next. The bracket keeps the pattern from matching a
# shell whose own command line contains it.
_AIPERF_PROCS = "^[a]iperf |[a]iperf profile"
# Far above any round's seed, so priming never pre-warms measured prompts.
PRIME_SEED_OFFSET = 1_000_000


def _num(value: float) -> str:
    """A number without exponent notation, so labels stay parseable."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


@dataclasses.dataclass(frozen=True, order=True)
class LoadPoint:
    """One load level: a closed-loop concurrency or an open-loop request rate.

    Its label (``c8``, ``rps12.5``) names the results directory, so a results
    tree can be read back without the configuration that produced it.
    """

    kind: str  # "c" or "rps"
    value: float

    def __post_init__(self) -> None:
        if self.kind not in ("c", "rps") or self.value <= 0:
            raise ValueError(f"invalid load point {self.kind}{self.value}")

    @property
    def label(self) -> str:
        return f"{self.kind}{_num(self.value)}"

    def describe(self) -> str:
        v = _num(self.value)
        return f"c={v}" if self.kind == "c" else f"{v} req/s"

    @classmethod
    def parse(cls, label: str) -> LoadPoint:
        m = re.fullmatch(r"(c|rps)(\d+(?:\.\d+)?)", label)
        if not m:
            raise ValueError(f"not a load point label: {label!r}")
        return cls(m.group(1), float(m.group(2)))


@dataclasses.dataclass
class Workload:
    """Request shape. Identical for both sides of every comparison."""

    # Open-loop Poisson arrival rates, in requests/s. Closed-loop clients
    # phase-lock with the engine's batching, which makes TTFT multi-modal and
    # its median jump between modes from one round to the next; Poisson
    # arrivals break that up. The defaults sit near 30% and 70% of what the
    # default mocker deployments sustain (about 19 req/s agg, 22 disagg).
    # Mocker capacity comes from its simulated engine, not the host, so they
    # hold on any cluster that is not CPU-starved; rescale them when changing
    # worker counts or --speedup-ratio.
    request_rates: tuple[float, ...] = (6.0, 14.0)
    # Closed-loop concurrencies, used only when request_rates is empty.
    concurrencies: tuple[int, ...] = (8, 32)
    isl: int = 1024
    osl: int = 128
    # Shared prefixes give the KV router something to route on. With random
    # unique prompts a KV-aware router degenerates to load balancing and the
    # A/B would miss most of what the EPP is for.
    prefix_length: int = 512
    num_prefixes: int = 8
    # Per measurement point; 0 means max(200, 10 x concurrency), or about 40
    # seconds of arrivals in rate mode.
    request_count: int = 0
    warmup_count: int = 0
    rounds: int = 3
    seed: int = 42
    run_timeout_s: int = 1800

    def __post_init__(self) -> None:
        if self.prefix_tokens() >= self.isl:
            raise ValueError(
                f"--prefix-length ({self.prefix_length}) must be shorter than "
                f"--isl ({self.isl}), which includes it"
            )
        points = self.points()
        if len(set(points)) != len(points):
            # Each point's runs are stored under its label.
            raise ValueError("load points must be distinct")

    def prefix_tokens(self) -> int:
        """Shared-prefix tokens at the start of every prompt (0 when disabled)."""
        return self.prefix_length if self.num_prefixes > 0 else 0

    def seed_for(self, rnd: int, point_index: int) -> int:
        """aiperf seed of one round at one load point, shared by both sides.

        Distinct per load point as well as per round: aiperf draws its prompt
        pool in order from the seed, so two points sharing one would send the
        same leading prompts, and the second would find them already cached.
        """
        return self.seed + rnd * len(self.points()) + point_index

    def points(self) -> list[LoadPoint]:
        if self.request_rates:
            return [LoadPoint("rps", float(r)) for r in self.request_rates]
        return [LoadPoint("c", float(c)) for c in self.concurrencies]

    def requests_for(self, point: LoadPoint) -> int:
        if self.request_count:
            return self.request_count
        if point.kind == "c":
            return max(200, 10 * int(point.value))
        return max(200, round(40 * point.value))

    def warmup_for(self, point: LoadPoint) -> int:
        if self.warmup_count:
            return self.warmup_count
        if point.kind == "c":
            return max(10, 2 * int(point.value))
        return max(10, round(5 * point.value))


def aiperf_command(
    *,
    model: str,
    url: str,
    headers: dict[str, str],
    workload: Workload,
    point: LoadPoint,
    seed: int,
    artifact_dir: str,
) -> list[str]:
    """aiperf invocation. Only ``url`` differs between the A and B sides."""
    if point.kind == "c":
        load = ["--concurrency", str(int(point.value))]
    else:
        load = ["--request-rate", _num(point.value), "--arrival-pattern", "poisson"]
    cmd = [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        model,
        "--url",
        url,
        # Streaming chat with ignore_eos and no GPU telemetry, as the router
        # benchmarks run it.
        *get_common_aiperf_flags(),
        *load,
        "--request-count",
        str(workload.requests_for(point)),
        "--warmup-request-count",
        str(workload.warmup_for(point)),
        # aiperf prepends the shared prefix to a prompt of this length, so
        # subtract it to keep --isl the length of the whole prompt.
        "--synthetic-input-tokens-mean",
        str(workload.isl - workload.prefix_tokens()),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(workload.osl),
        "--output-tokens-stddev",
        "0",
        # Mocker emits random token ids, so re-tokenizing its text output does
        # not round-trip; trust the server's usage counts instead.
        "--use-server-token-count",
        "--random-seed",
        str(seed),
        "--artifact-dir",
        artifact_dir,
        "--ui",
        "none",
        # Scraping /metrics works against the Frontend but not the Gateway;
        # leave it off so both sides carry the same client-side load.
        "--no-server-metrics",
    ]
    if workload.prefix_tokens() > 0:
        cmd += [
            "--prefix-prompt-length",
            str(workload.prefix_length),
            "--num-prefix-prompts",
            str(workload.num_prefixes),
        ]
    for name, value in headers.items():
        cmd += ["--header", f"{name}:{value}"]
    return cmd


def summarize(summary: dict[str, Any]) -> dict[str, Any]:
    """The per-run values the report compares, from aiperf's summary export.

    aiperf summarizes the profiling phase only, so warmup requests are left
    out. ``records`` counts the requests that succeeded.
    """

    def field(metric: str, key: str = "avg") -> Any:
        value = summary.get(metric)
        return value.get(key) if isinstance(value, dict) else None

    errors = summary.get("error_summary") or []
    return {
        "aiperf_version": summary.get("aiperf_version"),
        "records": field("request_count"),
        "errors": sum(int(e.get("count", 0)) for e in errors if isinstance(e, dict)),
        "benchmark_duration": field("benchmark_duration"),
        **{metric: field(metric) for metric in THROUGHPUT_METRICS},
        "latency": {
            metric: {key: field(metric, key) for key in LATENCY_STATS.values()}
            for metric in LATENCY_METRICS
        },
    }


def _gone(stderr: str) -> bool:
    """Whether a failed exec means the loadgen pod or container no longer exists."""
    return "(NotFound)" in stderr or "No such container" in stderr


class LoadGen:
    def __init__(self, kube: Kube, pod: str) -> None:
        self.kube = kube
        self.pod = pod

    def smoke(self, url: str, model: str, headers: dict[str, str]) -> float:
        """Send one short chat request; return its latency in seconds."""
        script = (
            "import json,sys,time,urllib.request\n"
            "url,model,headers=sys.argv[1],sys.argv[2],json.loads(sys.argv[3])\n"
            "body=json.dumps({'model':model,'max_tokens':4,"
            "'messages':[{'role':'user','content':'ping'}]}).encode()\n"
            "headers['content-type']='application/json'\n"
            "t=time.time()\n"
            "r=urllib.request.urlopen(urllib.request.Request("
            "url+'/v1/chat/completions',data=body,headers=headers),timeout=30)\n"
            "assert r.status==200, r.status\n"
            "json.loads(r.read())\n"
            "print(time.time()-t)\n"
        )
        out = self.kube.exec(
            self.pod,
            ["python3", "-c", script, url, model, json.dumps(headers)],
            timeout=60,
        ).stdout
        return float(out.strip().splitlines()[-1])

    def kill_aiperf(self) -> None:
        """Stop every aiperf process in the pod and wait until they are gone."""
        self.kube.exec(
            self.pod, ["pkill", "-9", "-f", _AIPERF_PROCS], check=False, timeout=60
        )
        # pgrep exits 1 when nothing matches, but so does a failed exec, and
        # mistaking that for "none left" would launch a run on top of a live
        # one. pgrep's own status therefore comes back on stdout.
        check = f"pgrep -f {shlex.quote(_AIPERF_PROCS)} >/dev/null; echo $?"
        for _ in range(30):
            try:
                proc = self.kube.exec(
                    self.pod, ["sh", "-c", check], check=False, timeout=60
                )
            except subprocess.TimeoutExpired:
                proc = None
            if proc is not None and proc.returncode == 0 and proc.stdout.strip() == "1":
                return
            time.sleep(1)
        raise KubeError(f"cannot confirm that aiperf in {self.pod} exited")

    def aiperf_version(self) -> str:
        return self.kube.exec(self.pod, ["aiperf", "--version"]).stdout.strip()

    def run(self, cmd: list[str], artifact_dir: str, timeout_s: int) -> dict[str, Any]:
        """Run one aiperf measurement to completion and return extracted results."""
        rc_file = f"{artifact_dir}.rc"
        log_file = f"{artifact_dir}.log"
        d, rc_q, lg = (shlex.quote(p) for p in (artifact_dir, rc_file, log_file))
        inner = f"{shlex.join(cmd)} > {lg} 2>&1; echo $? > {rc_q}"
        # Only the nohup'd run goes to the background. A trailing `&` binds
        # to the whole `&&` list, which would background the setup too: the
        # exec would return at once and, if the session closed before the
        # list reached nohup, aiperf would never start while the poll below
        # waited out the full timeout.
        launch = (
            f"rm -rf {d} {rc_q} {lg} && mkdir -p {d} && "
            f"{{ nohup sh -c {shlex.quote(inner)} > /dev/null 2>&1 & }}"
        )
        # The harness runs one aiperf at a time. A previous attempt whose
        # exec stream dropped may still be running; left alone it would add
        # its load to this one's. pkill runs as its own exec so it cannot
        # match the shell that launches the next run.
        self.kill_aiperf()
        self.kube.exec(self.pod, ["sh", "-c", launch])
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                proc = self.kube.exec(
                    self.pod, ["cat", rc_file], check=False, timeout=60
                )
            except subprocess.TimeoutExpired:
                proc = None
            if proc is not None and proc.returncode == 0 and proc.stdout.strip():
                rc = int(proc.stdout.strip())
                break
            if proc is not None and proc.returncode != 0 and _gone(proc.stderr):
                raise KubeError(f"loadgen {self.pod} is gone: {proc.stderr.strip()}")
            # Anything else, from "No such file" while aiperf runs to an exec
            # failing while the API server restarts, is waited out: aiperf
            # keeps running detached.
            if time.monotonic() > deadline:
                self.kill_aiperf()
                raise KubeError(f"aiperf did not finish within {timeout_s}s")
            time.sleep(5)
        if rc != 0:
            tail = self.kube.exec(
                self.pod, ["tail", "-n", "40", log_file], check=False
            ).stdout
            raise KubeError(f"aiperf exited {rc}:\n{tail}")
        path = f"{artifact_dir}/{SUMMARY_FILE}"
        out = self.kube.exec(self.pod, ["cat", path]).stdout
        try:
            return summarize(json.loads(out))
        except ValueError as e:
            raise KubeError(f"cannot parse {path}: {e}") from e
