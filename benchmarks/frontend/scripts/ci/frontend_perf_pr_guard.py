#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-pinned frontend performance PR guard.

This is intentionally small and CI-oriented: it runs a fixed frontend/mocker
benchmark point through the existing sweep runner, then compares the resulting
CSV against an optional baseline captured on the same runner class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "frontend_perf_pr_guard"
DEFAULT_MODEL = (
    REPO_ROOT / "lib" / "llm" / "tests" / "data" / "sample-models" / "TinyLlama_v1.1"
)
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] + ': ' + message['content'] + eos_token }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
)
KEY_FIELDS = ("backend", "tokenizer", "concurrency", "isl", "osl", "workers")


@dataclass(frozen=True)
class Thresholds:
    min_req_per_sec_ratio: float = 0.85
    min_output_tok_per_sec_ratio: float = 0.85
    max_ttft_p50_ratio: float = 1.25
    max_ttft_p99_ratio: float = 1.35
    max_itl_p50_ratio: float = 1.20
    max_itl_p99_ratio: float = 1.35


def parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    ordered = sorted(cpus)
    if not ordered:
        return ""

    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def allowed_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))

    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return parse_cpu_list(line.split(":", 1)[1].strip())

    return list(range(os.cpu_count() or 1))


def cpu_partition(min_cpus: int) -> tuple[str, str, list[int]]:
    cpus = allowed_cpus()
    if len(cpus) < min_cpus:
        raise RuntimeError(
            f"frontend perf guard requires at least {min_cpus} allowed CPUs; "
            f"runner only exposes {len(cpus)} ({format_cpu_list(cpus)})"
        )
    return str(cpus[0]), format_cpu_list(cpus[1:]), cpus


def row_key(row: dict[str, object]) -> str:
    return "|".join(str(row[field]) for field in KEY_FIELDS)


def to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def is_positive_finite(value: object) -> bool:
    metric = to_float(value)
    return math.isfinite(metric) and metric > 0


def normalize_row(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    for field in ("concurrency", "isl", "osl", "workers"):
        normalized[field] = int(normalized[field])
    for field in (
        "req_per_sec",
        "output_tok_per_sec",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "itl_p50_ms",
        "itl_p99_ms",
        "duration_sec",
    ):
        normalized[field] = to_float(normalized.get(field))
    return normalized


def load_csv(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            normalized = normalize_row(row)
            rows[row_key(normalized)] = normalized
    return rows


def load_points(path: Path) -> dict[str, dict[str, object]]:
    if path.suffix == ".csv":
        return load_csv(path)

    data = json.loads(path.read_text())
    if isinstance(data, list):
        points = data
    else:
        points = data.get("points", [])
    rows: dict[str, dict[str, object]] = {}
    for row in points:
        normalized = normalize_row(row)
        rows[row_key(normalized)] = normalized
    return rows


def export_baseline(
    rows: dict[str, dict[str, object]],
    path: Path,
    frontend_cpuset: str,
    load_cpuset: str,
    allowed: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "allowed_cpus": format_cpu_list(allowed),
        "frontend_cpuset": frontend_cpuset,
        "load_cpuset": load_cpuset,
        "key_fields": KEY_FIELDS,
        "points": list(rows.values()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compare_metric(
    issues: list[str],
    key: str,
    metric: str,
    baseline: float,
    candidate: float,
    ratio: float,
    lower_is_worse: bool,
) -> None:
    if baseline <= 0:
        return
    actual_ratio = candidate / baseline
    if lower_is_worse and actual_ratio < ratio:
        issues.append(
            f"{key}: {metric} dropped to {actual_ratio:.2%} of baseline "
            f"({candidate:.2f} vs {baseline:.2f})"
        )
    elif not lower_is_worse and actual_ratio > ratio:
        issues.append(
            f"{key}: {metric} rose to {actual_ratio:.2%} of baseline "
            f"({candidate:.2f} vs {baseline:.2f})"
        )


def evaluate(
    result_csv: Path,
    baseline_path: Path | None,
    thresholds: Thresholds,
    require_baseline: bool,
) -> tuple[bool, dict[str, object]]:
    candidate = load_csv(result_csv)
    issues: list[str] = []
    notes: list[str] = []

    if not candidate:
        issues.append(f"no result rows found in {result_csv}")

    for key, row in candidate.items():
        if row.get("status") != "ok":
            issues.append(f"{key}: run status is {row.get('status')!r}")
        for metric in (
            "req_per_sec",
            "output_tok_per_sec",
            "ttft_p50_ms",
            "ttft_p99_ms",
            "itl_p50_ms",
            "itl_p99_ms",
        ):
            if not is_positive_finite(row.get(metric)):
                issues.append(
                    f"{key}: {metric} is missing, non-finite, or non-positive"
                )

    baseline: dict[str, dict[str, object]] = {}
    if baseline_path is not None:
        if baseline_path.exists():
            baseline = load_points(baseline_path)
        elif require_baseline:
            issues.append(f"required baseline does not exist: {baseline_path}")
        else:
            notes.append(
                f"baseline not found; running sanity-only guard: {baseline_path}"
            )
    elif require_baseline:
        issues.append("--require-baseline was set, but no --baseline was provided")
    else:
        notes.append("no baseline supplied; running sanity-only guard")

    if baseline:
        common = sorted(set(candidate) & set(baseline))
        if not common:
            issues.append("baseline exists, but no points match the candidate run")

        missing = sorted(set(candidate) - set(baseline))
        for key in missing:
            issues.append(f"{key}: missing from baseline")

        for key in common:
            b = baseline[key]
            c = candidate[key]
            compare_metric(
                issues,
                key,
                "req_per_sec",
                to_float(b.get("req_per_sec")),
                to_float(c.get("req_per_sec")),
                thresholds.min_req_per_sec_ratio,
                lower_is_worse=True,
            )
            compare_metric(
                issues,
                key,
                "output_tok_per_sec",
                to_float(b.get("output_tok_per_sec")),
                to_float(c.get("output_tok_per_sec")),
                thresholds.min_output_tok_per_sec_ratio,
                lower_is_worse=True,
            )
            compare_metric(
                issues,
                key,
                "ttft_p50_ms",
                to_float(b.get("ttft_p50_ms")),
                to_float(c.get("ttft_p50_ms")),
                thresholds.max_ttft_p50_ratio,
                lower_is_worse=False,
            )
            compare_metric(
                issues,
                key,
                "ttft_p99_ms",
                to_float(b.get("ttft_p99_ms")),
                to_float(c.get("ttft_p99_ms")),
                thresholds.max_ttft_p99_ratio,
                lower_is_worse=False,
            )
            compare_metric(
                issues,
                key,
                "itl_p50_ms",
                to_float(b.get("itl_p50_ms")),
                to_float(c.get("itl_p50_ms")),
                thresholds.max_itl_p50_ratio,
                lower_is_worse=False,
            )
            compare_metric(
                issues,
                key,
                "itl_p99_ms",
                to_float(b.get("itl_p99_ms")),
                to_float(c.get("itl_p99_ms")),
                thresholds.max_itl_p99_ratio,
                lower_is_worse=False,
            )

    report = {
        "ok": not issues,
        "result_csv": str(result_csv),
        "baseline": str(baseline_path) if baseline_path else None,
        "candidate_points": list(candidate.values()),
        "issues": issues,
        "notes": notes,
    }
    return not issues, report


def markdown_report(report: dict[str, object]) -> str:
    lines = ["# Frontend Perf PR Guard", ""]
    lines.append(f"Result: {'PASS' if report['ok'] else 'FAIL'}")
    lines.append(f"Results CSV: `{report['result_csv']}`")
    if report.get("baseline") is not None:
        lines.append(f"Baseline: `{report['baseline']}`")
    lines.append("")

    notes = report.get("notes") or []
    if notes:
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    issues = report.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")

    points = report.get("candidate_points") or []
    if points:
        lines.append("## Candidate Points")
        lines.append("| Key | Req/s | Tok/s | TTFT p50 | TTFT p99 | ITL p50 | Status |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in points:
            key = row_key(row)
            lines.append(
                f"| `{key}` | {to_float(row.get('req_per_sec')):.2f} | "
                f"{to_float(row.get('output_tok_per_sec')):.1f} | "
                f"{to_float(row.get('ttft_p50_ms')):.1f} | "
                f"{to_float(row.get('ttft_p99_ms')):.1f} | "
                f"{to_float(row.get('itl_p50_ms')):.1f} | {row.get('status')} |"
            )
    return "\n".join(lines) + "\n"


def thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        min_req_per_sec_ratio=1.0 - args.max_req_per_sec_drop_pct / 100.0,
        min_output_tok_per_sec_ratio=1.0 - args.max_output_tok_per_sec_drop_pct / 100.0,
        max_ttft_p50_ratio=1.0 + args.max_ttft_p50_rise_pct / 100.0,
        max_ttft_p99_ratio=1.0 + args.max_ttft_p99_rise_pct / 100.0,
        max_itl_p50_ratio=1.0 + args.max_itl_p50_rise_pct / 100.0,
        max_itl_p99_ratio=1.0 + args.max_itl_p99_rise_pct / 100.0,
    )


def add_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--require-baseline", action="store_true")
    parser.add_argument("--max-req-per-sec-drop-pct", type=float, default=15.0)
    parser.add_argument("--max-output-tok-per-sec-drop-pct", type=float, default=15.0)
    parser.add_argument("--max-ttft-p50-rise-pct", type=float, default=25.0)
    parser.add_argument("--max-ttft-p99-rise-pct", type=float, default=35.0)
    parser.add_argument("--max-itl-p50-rise-pct", type=float, default=20.0)
    parser.add_argument("--max-itl-p99-rise-pct", type=float, default=35.0)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)


def write_reports(args: argparse.Namespace, report: dict[str, object]) -> None:
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    markdown = markdown_report(report)
    if args.summary_file is not None:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(markdown)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary is not None and step_summary != "":
        with open(step_summary, "a") as f:
            f.write(markdown)

    print(markdown)


def cmd_check(args: argparse.Namespace) -> int:
    ok, report = evaluate(
        args.results_csv,
        args.baseline,
        thresholds_from_args(args),
        args.require_baseline,
    )
    write_reports(args, report)
    return 0 if ok else 1


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


def materialize_default_chat_model(output_dir: Path, dry_run: bool) -> Path:
    """Create a chat-capable TinyLlama fixture under the run output directory."""
    target = output_dir / "model-fixtures" / "tinyllama-chat"
    if dry_run:
        return target

    target.mkdir(parents=True, exist_ok=True)
    for src in DEFAULT_MODEL.iterdir():
        if src.name == "tokenizer_config.json":
            continue
        dst = target / src.name
        if dst.exists():
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    tokenizer_config = json.loads((DEFAULT_MODEL / "tokenizer_config.json").read_text())
    tokenizer_config["chat_template"] = DEFAULT_CHAT_TEMPLATE
    (target / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n"
    )
    return target


def resolve_model_arg(args: argparse.Namespace, output_dir: Path) -> str:
    if _same_path(args.model, DEFAULT_MODEL):
        return str(materialize_default_chat_model(output_dir, args.dry_run))
    return args.model


def build_sweep_command(
    args: argparse.Namespace, output_dir: Path, model: str
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "sweep_runner.py"),
        "--mode",
        "local",
        "--backend",
        "mocker",
        "--model",
        model,
        "--tokenizers",
        args.tokenizers,
        "--concurrency",
        str(args.concurrency),
        "--isl",
        str(args.isl),
        "--osl",
        str(args.osl),
        "--workers",
        str(args.workers),
        "--benchmark-duration",
        str(args.benchmark_duration),
        "--speedup-ratio",
        str(args.speedup_ratio),
        "--output-dir",
        str(output_dir),
        "--cooldown",
        "0",
        "--no-report",
        "--",
        "--skip-bpf",
        "--skip-nsys",
        "--skip-flamegraph",
        "--skip-perf",
    ]


def cmd_run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    model = resolve_model_arg(args, output_dir)
    frontend_cpuset, load_cpuset, allowed = cpu_partition(args.min_cpus)
    record_processors = args.aiperf_record_processors or len(
        parse_cpu_list(load_cpuset)
    )

    env = os.environ.copy()
    env.update(
        {
            "FRONTEND_CPUSET": frontend_cpuset,
            "LOAD_CPUSET": load_cpuset,
            "INFRA_CPUSET": load_cpuset,
            "AIPERF_RECORD_PROCESSORS": str(record_processors),
            "AIPERF_WORKERS_MAX": str(args.concurrency),
            "DYN_RUNTIME_NUM_WORKER_THREADS": str(args.frontend_runtime_threads),
            "DYN_RUNTIME_MAX_BLOCKING_THREADS": str(args.frontend_max_blocking_threads),
            "DYN_COMPUTE_THREADS": str(args.frontend_compute_threads),
            "TOKENIZERS_PARALLELISM": args.tokenizers_parallelism,
        }
    )

    cmd = build_sweep_command(args, output_dir, model)
    print(f"Allowed CPUs:  {format_cpu_list(allowed)}")
    print(f"Frontend CPU:  {frontend_cpuset}")
    print(f"Load CPUs:     {load_cpuset}")
    if _same_path(args.model, DEFAULT_MODEL):
        print(f"Model fixture: {model}")
    print(f"Sweep command: {' '.join(cmd)}")

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if completed.returncode != 0:
        return completed.returncode

    result_csv = output_dir / "results.csv"
    if args.export_baseline and result_csv.exists():
        rows = load_csv(result_csv)
        export_baseline(
            rows, args.export_baseline, frontend_cpuset, load_cpuset, allowed
        )

    check_args = argparse.Namespace(
        results_csv=result_csv,
        baseline=args.baseline,
        require_baseline=args.require_baseline,
        max_req_per_sec_drop_pct=args.max_req_per_sec_drop_pct,
        max_output_tok_per_sec_drop_pct=args.max_output_tok_per_sec_drop_pct,
        max_ttft_p50_rise_pct=args.max_ttft_p50_rise_pct,
        max_ttft_p99_rise_pct=args.max_ttft_p99_rise_pct,
        max_itl_p50_rise_pct=args.max_itl_p50_rise_pct,
        max_itl_p99_rise_pct=args.max_itl_p99_rise_pct,
        summary_file=args.summary_file or output_dir / "pr_guard_summary.md",
        report_json=args.report_json or output_dir / "pr_guard_report.json",
    )
    return cmd_check(check_args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pinned sweep and check it")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--export-baseline", type=Path, default=None)
    run_parser.add_argument("--min-cpus", type=int, default=4)
    run_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    run_parser.add_argument("--tokenizers", default="hf")
    run_parser.add_argument("--concurrency", type=int, default=128)
    run_parser.add_argument("--isl", type=int, default=1024)
    run_parser.add_argument("--osl", type=int, default=128)
    run_parser.add_argument("--workers", type=int, default=2)
    run_parser.add_argument("--benchmark-duration", type=int, default=60)
    run_parser.add_argument("--speedup-ratio", type=float, default=1_000_000.0)
    run_parser.add_argument("--frontend-runtime-threads", type=int, default=1)
    run_parser.add_argument("--frontend-compute-threads", type=int, default=1)
    run_parser.add_argument("--frontend-max-blocking-threads", type=int, default=8)
    run_parser.add_argument("--aiperf-record-processors", type=int, default=0)
    run_parser.add_argument("--tokenizers-parallelism", default="false")
    run_parser.add_argument("--dry-run", action="store_true")
    add_check_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    check_parser = subparsers.add_parser("check", help="Check an existing results.csv")
    check_parser.add_argument("results_csv", type=Path)
    add_check_args(check_parser)
    check_parser.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
