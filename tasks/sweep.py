#!/usr/bin/env python3
"""
TCP Request Plane benchmark sweep.

Sweeps num_requests x mocker_count x ISL/OSL profile at fixed concurrency,
builds HEAD via maturin, launches mocker+frontend, runs aiperf, collects
results into a single CSV, and prints a saturation analysis table.

Dimensions:
  A) --num-requests  : total requests sent per run (16k / 32k / 48k)
  B) --mocker-counts : backend workers via --data-parallel-size (1 / 2 / 4 / 8)

Services are restarted between mocker-count changes, not between num-requests
changes (no service restart needed for that).

Usage:
    python3 tasks/sweep.py                  # full sweep
    python3 tasks/sweep.py --dry-run        # print plan without executing
    python3 tasks/sweep.py --mocker-counts 1 4 8  # subset of mocker counts
    python3 tasks/sweep.py --num-requests 16384 32768  # subset of req counts
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

# Fixed concurrency for this sweep
CONCURRENCY = 4096

# (A) Total requests sent per aiperf run
NUM_REQUESTS_LIST = [16_384, 32_768, 49_152]

# (B) Backend worker count — maps to mocker --data-parallel-size
MOCKER_COUNTS = [1, 2, 4, 8]

COMMITS = {
    "head": "HEAD",
}


@dataclass
class Profile:
    name: str
    isl: int
    osl: int


PROFILES = [
    Profile("max-rps", 32, 32),
    # Profile("ingress-stress", 2048, 32),
    # Profile("egress-stress",  32,   1024),
]

MODEL = "Qwen/Qwen3-0.6B"

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_ACTIVATE = REPO_ROOT / "dynamo" / "bin" / "activate"
RESULTS_DIR = REPO_ROOT / "tasks" / "sweep_results"

SETTLE_SECS = 8  # wait after launching mocker/frontend
COOLDOWN_SECS = 5  # wait between runs
KILL_WAIT_SECS = 5  # wait after killing processes


def mocker_args(num_mockers: int) -> str:
    return (
        f"--model-path {MODEL} "
        "--num-gpu-blocks-override 131072 "
        "--max-num-seqs 16384 "
        "--max-num-batched-tokens 16384 "
        "--speedup-ratio 0 "
        f"--data-parallel-size {num_mockers} "
        "--request-plane tcp"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def run(
    cmd: str, *, check=True, capture=False, timeout=None
) -> subprocess.CompletedProcess:
    """Run a shell command with the virtualenv activated."""
    wrapped = f"source {VENV_ACTIVATE} && {cmd}"
    return subprocess.run(
        wrapped,
        shell=True,
        executable="/bin/bash",
        check=check,
        capture_output=capture,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def kill_services():
    """Kill any running mocker/frontend processes."""
    for pattern in ["dynamo.mocker", "dynamo.frontend"]:
        subprocess.run(
            f"pkill -f '{pattern}'",
            shell=True,
            capture_output=True,
            timeout=10,
        )
    time.sleep(KILL_WAIT_SECS)


def resolve_sha(ref: str) -> str:
    """Resolve a git ref to a short SHA without checking it out."""
    r = subprocess.run(
        f"git rev-parse --short {ref}",
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return r.stdout.strip()


def current_sha() -> str:
    return resolve_sha("HEAD")


def current_branch() -> str:
    r = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return r.stdout.strip()


def build_head() -> str:
    """Build current HEAD with maturin, return the resolved short SHA."""
    sha = current_sha()
    branch = current_branch()
    print(f"\n{'='*60}")
    print(f"  Building HEAD: {sha} ({branch})")
    print(f"{'='*60}")
    r = run(
        "cd lib/bindings/python && maturin develop --uv",
        capture=True,
        timeout=600,
        check=False,
    )
    if r.returncode != 0:
        print(f"  BUILD FAILED:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  Build OK")
    return sha


def start_services(num_mockers: int):
    """Launch mocker and frontend in background, wait for readiness."""
    print(f"  Starting mocker (data-parallel-size={num_mockers})...")
    subprocess.Popen(
        f"source {VENV_ACTIVATE} && python3 -m dynamo.mocker {mocker_args(num_mockers)}",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        preexec_fn=os.setsid,
    )
    time.sleep(SETTLE_SECS)

    print("  Starting frontend...")
    subprocess.Popen(
        f"source {VENV_ACTIVATE} && python3 -m dynamo.frontend --request-plane tcp",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        preexec_fn=os.setsid,
    )
    time.sleep(SETTLE_SECS)


def run_aiperf(profile: Profile, concurrency: int, num_requests: int) -> dict | None:
    """Run a single aiperf benchmark and parse the results."""
    cmd = (
        f"aiperf profile "
        f"--isl {profile.isl} --osl {profile.osl} "
        f"--concurrency {concurrency} "
        f"-m {MODEL} "
        f"--num-requests {num_requests} "
        f"--ui simple"
    )
    print(f"    Running: {cmd}")

    try:
        r = run(cmd, capture=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        print("    TIMEOUT after 600s", file=sys.stderr)
        return None

    if r.returncode != 0:
        print(f"    aiperf FAILED (rc={r.returncode})", file=sys.stderr)
        output = r.stdout or ""
    else:
        output = r.stdout

    metrics = parse_aiperf_output(output)

    # Add computed metrics that bypass the parser (more reliable)
    if metrics and metrics.get("duration_sec"):
        metrics["computed_req_per_sec"] = num_requests / metrics["duration_sec"]
        metrics["computed_tok_per_sec"] = (num_requests * profile.osl) / metrics[
            "duration_sec"
        ]

    return metrics


def parse_aiperf_output(output: str) -> dict | None:
    """Extract key metrics from aiperf simple UI output."""
    metrics = {}

    # Parse "Benchmark Duration: 93.42 sec"
    m = re.search(r"Benchmark Duration:\s+([\d.]+)\s+sec", output)
    if m:
        metrics["duration_sec"] = float(m.group(1))

    # Parse table rows — look for metric lines with │ delimiters
    # Example: │  Output Token │ 184,084.12 │      N/A │  ...
    lines = output.split("\n")
    current_metric = None
    for line in lines:
        cells = [c.strip() for c in line.split("│") if c.strip()]
        if len(cells) < 2:
            if current_metric and cells:
                current_metric += " " + cells[0]
            continue

        name = cells[0]
        val_str = cells[1]

        try:
            val = float(val_str.replace(",", ""))
        except ValueError:
            current_metric = name
            continue

        full_name = name
        if current_metric and not any(c.isdigit() for c in current_metric):
            full_name = current_metric + " " + name
        current_metric = name

        norm = full_name.lower().strip()
        if "request" in norm and "latency" in norm:
            metrics["req_latency_ms_avg"] = val
            if len(cells) >= 8:
                try:
                    metrics["req_latency_ms_min"] = float(cells[2].replace(",", ""))
                    metrics["req_latency_ms_max"] = float(cells[3].replace(",", ""))
                    metrics["req_latency_ms_p99"] = float(cells[4].replace(",", ""))
                    metrics["req_latency_ms_p90"] = float(cells[5].replace(",", ""))
                    metrics["req_latency_ms_p50"] = float(cells[6].replace(",", ""))
                except (ValueError, IndexError):
                    pass
        elif "output" in norm and "throughput" in norm:
            metrics["output_tok_per_sec"] = val
        elif "request" in norm and "throughput" in norm:
            metrics["req_per_sec"] = val
        elif "request count" in norm:
            metrics["request_count"] = val

    # Also try to find the JSON export path and load it
    json_match = re.search(r"JSON Export:\s*\n?\s*(.+\.json)", output)
    if json_match:
        json_path = Path(json_match.group(1).strip())
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if "output_tok_per_sec" not in metrics:
                    for entry in data if isinstance(data, list) else [data]:
                        if "output_token_throughput" in entry:
                            metrics["output_tok_per_sec"] = entry[
                                "output_token_throughput"
                            ]
                        if "request_throughput" in entry:
                            metrics["req_per_sec"] = entry["request_throughput"]
            except (json.JSONDecodeError, KeyError):
                pass

    return metrics if metrics else None


# ── Main sweep ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TCP request plane benchmark sweep")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without executing"
    )
    parser.add_argument(
        "--commits",
        nargs="+",
        default=list(COMMITS.keys()),
        help=f"Which commits to test (default: {list(COMMITS.keys())})",
    )
    parser.add_argument(
        "--num-requests",
        nargs="+",
        type=int,
        default=NUM_REQUESTS_LIST,
        dest="num_requests_list",
        help="Total request counts to sweep (dimension A)",
    )
    parser.add_argument(
        "--mocker-counts",
        nargs="+",
        type=int,
        default=MOCKER_COUNTS,
        help="Mocker data-parallel-size values to sweep (dimension B)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"Fixed concurrency level (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=[p.name for p in PROFILES],
        help="Profile names to run",
    )
    args = parser.parse_args()

    selected_profiles = [p for p in PROFILES if p.name in args.profiles]
    selected_commits = {k: v for k, v in COMMITS.items() if k in args.commits}
    mocker_counts = sorted(args.mocker_counts)
    num_requests_list = sorted(args.num_requests_list)

    total_runs = (
        len(selected_commits)
        * len(selected_profiles)
        * len(mocker_counts)
        * len(num_requests_list)
    )
    print(
        f"Sweep plan: {len(selected_commits)} commits x {len(selected_profiles)} profiles "
        f"x {len(mocker_counts)} mocker-counts x {len(num_requests_list)} num-requests "
        f"= {total_runs} runs"
    )
    print(f"  Commits:        {list(selected_commits.keys())}")
    print(f"  Profiles:       {[p.name for p in selected_profiles]}")
    print(f"  Concurrency:    {args.concurrency} (fixed)")
    print(f"  Mocker counts:  {mocker_counts}  (--data-parallel-size)")
    print(f"  Num requests:   {num_requests_list}")
    print(
        f"  Service restarts: {len(selected_commits) * len(mocker_counts)} "
        f"(once per commit x mocker-count)"
    )

    resolved_shas = {}
    for commit_name, ref in selected_commits.items():
        resolved_shas[commit_name] = resolve_sha(ref)
    sha_tag = "_".join(resolved_shas[k] for k in selected_commits)

    if args.dry_run:
        print(f"  Resolved SHAs: {resolved_shas}")
        for commit_name in selected_commits:
            for nm in mocker_counts:
                for profile in selected_profiles:
                    for nr in num_requests_list:
                        print(
                            f"  [{commit_name} ({resolved_shas[commit_name]})] "
                            f"mockers={nm} {profile.name} "
                            f"isl={profile.isl} osl={profile.osl} "
                            f"concurrency={args.concurrency} num_requests={nr}"
                        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"sweep_{sha_tag}_{timestamp}.csv"

    fieldnames = [
        "commit_name",
        "commit_sha",
        "profile",
        "isl",
        "osl",
        "concurrency",
        "num_mockers",
        "num_requests",
        "req_per_sec",
        "output_tok_per_sec",
        "req_latency_ms_avg",
        "req_latency_ms_min",
        "req_latency_ms_max",
        "req_latency_ms_p99",
        "req_latency_ms_p90",
        "req_latency_ms_p50",
        "duration_sec",
        "request_count",
        "computed_req_per_sec",
        "computed_tok_per_sec",
    ]

    results = []
    run_idx = 0

    try:
        for commit_name, ref in selected_commits.items():
            sha = build_head()

            for nm in mocker_counts:
                # Restart services for each mocker count change
                kill_services()
                start_services(nm)

                for profile in selected_profiles:
                    for nr in num_requests_list:
                        run_idx += 1
                        print(
                            f"\n--- Run {run_idx}/{total_runs}: "
                            f"[{commit_name}] {profile.name} "
                            f"mockers={nm} num_requests={nr:,} "
                            f"concurrency={args.concurrency} ---"
                        )

                        metrics = run_aiperf(profile, args.concurrency, nr)

                        row = {
                            "commit_name": commit_name,
                            "commit_sha": sha,
                            "profile": profile.name,
                            "isl": profile.isl,
                            "osl": profile.osl,
                            "concurrency": args.concurrency,
                            "num_mockers": nm,
                            "num_requests": nr,
                        }
                        if metrics:
                            row.update(metrics)
                            print(
                                f"    -> {metrics.get('req_per_sec', 'N/A')} req/s, "
                                f"{metrics.get('output_tok_per_sec', 'N/A')} tok/s, "
                                f"p99={metrics.get('req_latency_ms_p99', 'N/A')}ms, "
                                f"{metrics.get('duration_sec', 'N/A')}s"
                            )
                        else:
                            print("    -> NO RESULTS")

                        results.append(row)

                        # Write incrementally so partial results are saved
                        with open(csv_path, "w", newline="") as f:
                            writer = csv.DictWriter(
                                f, fieldnames=fieldnames, extrasaction="ignore"
                            )
                            writer.writeheader()
                            writer.writerows(results)

                        time.sleep(COOLDOWN_SECS)

                kill_services()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.")
    finally:
        kill_services()

    print(f"\n{'='*60}")
    print(f"  Sweep complete: {len(results)}/{total_runs} runs")
    print(f"  Results: {csv_path}")
    print(f"{'='*60}")

    if results:
        print_summary_table(results)
        print_saturation_analysis(results)


def print_summary_table(results: list[dict]):
    """Print mockers x num_requests matrix with req/s, tok/s, p99 columns."""
    mocker_counts = sorted(set(r["num_mockers"] for r in results))
    num_requests_list = sorted(set(r["num_requests"] for r in results))

    # Column widths
    W = 10

    print()
    print("── Results Matrix ─────────────────────────────────────────────────────")
    header = f"{'Mockers':>8}  {'NumReq':>8}  "
    header += f"{'req/s':>{W}}  {'comp_rps':>{W}}  {'tok/s':>{W}}  {'p99_ms':>{W}}  {'dur_sec':>{W}}"
    print(header)
    print("-" * len(header))

    for nm in mocker_counts:
        for nr in num_requests_list:
            rows = [
                r for r in results if r["num_mockers"] == nm and r["num_requests"] == nr
            ]
            if not rows:
                continue
            # Average across profiles (usually just one)
            r = rows[0]
            rps = r.get("req_per_sec", None)
            comp_rps = r.get("computed_req_per_sec", None)
            toks = r.get("output_tok_per_sec", None)
            p99 = r.get("req_latency_ms_p99", None)
            dur = r.get("duration_sec", None)

            rps_s = f"{rps:>{W}.1f}" if rps is not None else f"{'N/A':>{W}}"
            comp_rps_s = (
                f"{comp_rps:>{W}.1f}" if comp_rps is not None else f"{'N/A':>{W}}"
            )
            toks_s = f"{toks:>{W}.1f}" if toks is not None else f"{'N/A':>{W}}"
            p99_s = f"{p99:>{W}.1f}" if p99 is not None else f"{'N/A':>{W}}"
            dur_s = f"{dur:>{W}.2f}" if dur is not None else f"{'N/A':>{W}}"

            print(
                f"{nm:>8}  {nr:>8,}  {rps_s}  {comp_rps_s}  {toks_s}  {p99_s}  {dur_s}"
            )
        print()  # blank line between mocker groups


def print_saturation_analysis(results: list[dict]):
    """Analyze saturation and bottlenecks from sweep results."""
    mocker_counts = sorted(set(r["num_mockers"] for r in results))
    num_requests_list = sorted(set(r["num_requests"] for r in results))

    def get(nm, nr, key):
        rows = [
            r for r in results if r["num_mockers"] == nm and r["num_requests"] == nr
        ]
        return rows[0].get(key) if rows else None

    # Prefer computed_req_per_sec over parsed req_per_sec for analysis
    def get_rps(nm, nr):
        return get(nm, nr, "computed_req_per_sec") or get(nm, nr, "req_per_sec")

    print("── Saturation Analysis ────────────────────────────────────────────────")

    # (A) Does req/s scale with num_requests for each mocker count?
    print("\n[A] Throughput scaling with num_requests (fixed mockers)")
    print(f"  {'Mockers':>8}  ", end="")
    for nr in num_requests_list:
        print(f"  {nr:>8,}", end="")
    print("  <- req/s (computed)")
    for nm in mocker_counts:
        print(f"  {nm:>8}  ", end="")
        prev_rps = None
        for nr in num_requests_list:
            rps = get_rps(nm, nr)
            if rps is None:
                print(f"  {'N/A':>8}", end="")
            else:
                if prev_rps and prev_rps > 0:
                    ratio = rps / prev_rps
                    flag = " *SATURATED*" if ratio < 1.05 else ""
                    print(f"  {rps:>8.0f}{flag}", end="")
                else:
                    print(f"  {rps:>8.0f}", end="")
                prev_rps = rps
        print()

    # (B) Does req/s scale with mocker count for each num_requests?
    print("\n[B] Throughput scaling with mocker count (fixed num_requests)")
    print(f"  {'NumReq':>8}  ", end="")
    for nm in mocker_counts:
        print(f"  {nm:>6}x", end="")
    print("  <- mockers | req/s (computed)")
    for nr in num_requests_list:
        print(f"  {nr:>8,}  ", end="")
        prev_rps = None
        for nm in mocker_counts:
            rps = get_rps(nm, nr)
            if rps is None:
                print(f"  {'N/A':>6}", end="")
            else:
                if prev_rps and prev_rps > 0:
                    scale = rps / prev_rps
                    flag = (
                        " ~LINEAR"
                        if scale >= 1.8
                        else (" *BOTTLENECK*" if scale < 1.2 else "")
                    )
                    print(f"  {rps:>6.0f}{flag}", end="")
                else:
                    print(f"  {rps:>6.0f}", end="")
                prev_rps = rps
        print()

    # (C) Latency inflation: does p99 grow with num_requests?
    print("\n[C] Latency (p99 ms) vs num_requests")
    print(f"  {'Mockers':>8}  ", end="")
    for nr in num_requests_list:
        print(f"  {nr:>8,}", end="")
    print("  <- num_requests | p99 ms")
    for nm in mocker_counts:
        print(f"  {nm:>8}  ", end="")
        base_p99 = None
        for nr in num_requests_list:
            p99 = get(nm, nr, "req_latency_ms_p99")
            if p99 is None:
                print(f"  {'N/A':>8}", end="")
            else:
                if base_p99 is None:
                    base_p99 = p99
                ratio = p99 / base_p99 if base_p99 else 1.0
                flag = " *HIGH*" if ratio > 2.0 else ""
                print(f"  {p99:>8.1f}{flag}", end="")
        print()

    print()
    print("Flags: *SATURATED* = <5% gain doubling requests  |  *BOTTLENECK* = <20%")
    print("       scaling with 2x mockers  |  *HIGH* = p99 >2x baseline")


if __name__ == "__main__":
    main()
