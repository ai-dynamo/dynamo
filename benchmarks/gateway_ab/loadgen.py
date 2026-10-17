# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive aiperf inside the loadgen pod and pull back compact results.

aiperf runs detached (``nohup``) and is polled for an exit-code file, so a
dropped ``kubectl exec`` stream — common behind proxies that cut idle
connections — never loses a run. Only a few hundred KB of per-request samples
leave the pod; the full aiperf artifacts stay in the pod for inspection.
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

from .kube import Kube, KubeError

log = logging.getLogger(__name__)

RESULTS_ROOT = "/results"


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
        return f"{self.kind}{self.value:g}"

    def describe(self) -> str:
        return f"c={self.value:g}" if self.kind == "c" else f"{self.value:g} req/s"

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
        load = ["--request-rate", f"{point.value:g}", "--arrival-pattern", "poisson"]
    cmd = [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        model,
        "--url",
        url,
        "--endpoint-type",
        "chat",
        "--streaming",
        *load,
        "--request-count",
        str(workload.requests_for(point)),
        "--warmup-request-count",
        str(workload.warmup_for(point)),
        "--synthetic-input-tokens-mean",
        str(workload.isl),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(workload.osl),
        "--output-tokens-stddev",
        "0",
        "--extra-inputs",
        "ignore_eos:true",
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
    if workload.prefix_length > 0 and workload.num_prefixes > 0:
        cmd += [
            "--prefix-prompt-length",
            str(workload.prefix_length),
            "--num-prefix-prompts",
            str(workload.num_prefixes),
        ]
    for name, value in headers.items():
        cmd += ["--header", f"{name}:{value}"]
    return cmd


# Runs inside the loadgen pod with the image's python3. Reads aiperf's
# exports and prints one compact JSON object. Must stay standard-library only.
_EXTRACT_SCRIPT = r"""
import json, os, sys
d = sys.argv[1]
summary = json.load(open(os.path.join(d, "profile_export_aiperf.json")))
samples = {"time_to_first_token": [], "inter_token_latency": [], "request_latency": []}
errors = 0
records = 0
with open(os.path.join(d, "profile_export.jsonl")) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        meta = rec.get("metadata") or {}
        if meta.get("benchmark_phase", "profiling") != "profiling":
            continue
        records += 1
        if rec.get("error"):
            errors += 1
            continue
        metrics = rec.get("metrics") or {}
        for name, out in samples.items():
            m = metrics.get(name)
            v = m.get("value") if isinstance(m, dict) else m
            if isinstance(v, (int, float)):
                out.append(round(float(v), 4))
def scalar(name):
    m = summary.get(name)
    return m.get("avg") if isinstance(m, dict) else None
error_summary = summary.get("error_summary") or []
summary_errors = sum(int(e.get("count", 0)) for e in error_summary if isinstance(e, dict))
print(json.dumps({
    "aiperf_version": summary.get("aiperf_version"),
    "records": records,
    "errors": max(errors, summary_errors),
    "request_count": scalar("request_count"),
    "request_throughput": scalar("request_throughput"),
    "output_token_throughput": scalar("output_token_throughput"),
    "benchmark_duration": scalar("benchmark_duration"),
    "samples": samples,
}))
"""


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
        self.kube.exec(
            self.pod, ["pkill", "-f", "aiperf profile"], check=False, timeout=60
        )

    def aiperf_version(self) -> str:
        return self.kube.exec(self.pod, ["aiperf", "--version"]).stdout.strip()

    def run(self, cmd: list[str], artifact_dir: str, timeout_s: int) -> dict[str, Any]:
        """Run one aiperf measurement to completion and return extracted results."""
        rc_file = f"{artifact_dir}.rc"
        log_file = f"{artifact_dir}.log"
        d, rc_q, lg = (shlex.quote(p) for p in (artifact_dir, rc_file, log_file))
        inner = f"{shlex.join(cmd)} > {lg} 2>&1; echo $? > {rc_q}"
        launch = (
            f"rm -rf {d} {rc_q} {lg} && mkdir -p {d} && "
            f"nohup sh -c {shlex.quote(inner)} > /dev/null 2>&1 &"
        )
        # The harness runs one aiperf at a time. A previous attempt whose
        # exec stream dropped may still be running; left alone it would add
        # its load to this one's. pkill runs as its own exec so it cannot
        # match the shell that launches the next run.
        self.kill_aiperf()
        self.kube.exec(self.pod, ["sh", "-c", launch])
        deadline = time.monotonic() + timeout_s
        pod_errors = 0
        while True:
            try:
                proc = self.kube.exec(
                    self.pod, ["cat", rc_file], check=False, timeout=60
                )
            except subprocess.TimeoutExpired:
                proc = None  # transient; aiperf keeps running detached
            if proc is not None and proc.returncode == 0 and proc.stdout.strip():
                rc = int(proc.stdout.strip())
                break
            if proc is not None and proc.returncode != 0:
                # "No such file" just means aiperf is still running; anything
                # else is the exec itself failing, e.g. the pod is gone.
                if "No such file" in proc.stderr:
                    pod_errors = 0
                else:
                    pod_errors += 1
                    if pod_errors >= 3:
                        raise KubeError(
                            f"cannot reach loadgen pod {self.pod}: {proc.stderr.strip()}"
                        )
            if time.monotonic() > deadline:
                self.kill_aiperf()
                raise KubeError(f"aiperf did not finish within {timeout_s}s")
            time.sleep(5)
        if rc != 0:
            tail = self.kube.exec(
                self.pod, ["tail", "-n", "40", log_file], check=False
            ).stdout
            raise KubeError(f"aiperf exited {rc}:\n{tail}")
        out = self.kube.exec(
            self.pod, ["python3", "-", artifact_dir], input=_EXTRACT_SCRIPT
        ).stdout
        return json.loads(out)
