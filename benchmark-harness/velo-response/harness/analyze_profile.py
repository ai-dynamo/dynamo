#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze the single Tyche AgentX frontend profile and write its deliverables."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from analyze import parse_aiperf


HEADER_RE = re.compile(
    r"^(.+?)\s+(\d+)(?:/(\d+))?\s+(?:\[\d+\]\s+)?\d+\.\d+:"
)
FRAME_RE = re.compile(r"^[ \t]+([0-9a-fA-F]+)\s+(.+)\s+\(([^()]*)\)\s*$")


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def parse_perf(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    with path.open(errors="replace") as handle:
        for line in handle:
            header = HEADER_RE.match(line)
            if header:
                if current and current["frames"]:
                    samples.append(current)
                current = {
                    "comm": header.group(1).strip(),
                    "tid": int(header.group(3) or header.group(2)),
                    "frames": [],
                }
                continue
            frame = FRAME_RE.match(line)
            if frame and current is not None:
                symbol = " ".join(frame.group(2).split())
                dso = " ".join(frame.group(3).split())
                current["frames"].append(
                    {
                        "ip": int(frame.group(1), 16),
                        "label": f"{dso}!{symbol}",
                        "symbol": symbol,
                        "dso": dso,
                    }
                )
            elif not line.strip() and current and current["frames"]:
                samples.append(current)
                current = None
    if current and current["frames"]:
        samples.append(current)
    if not samples:
        raise ValueError(f"no perf callchains found in {path}")
    return samples


def aggregate(samples: Iterable[dict[str, Any]]) -> dict[str, Counter[str]]:
    self_counts: Counter[str] = Counter()
    inclusive: Counter[str] = Counter()
    folded: Counter[str] = Counter()
    for sample in samples:
        frames = [frame["label"] for frame in sample["frames"]]
        self_counts[frames[0]] += 1
        inclusive.update(set(frames))
        folded[";".join(reversed(frames))] += 1
    return {"self": self_counts, "inclusive": inclusive, "folded": folded}


def write_rankings(path: Path, counts: Counter[str], total: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "symbol", "samples", "percent"])
        for rank, (symbol, count) in enumerate(counts.most_common(), start=1):
            writer.writerow([rank, symbol, count, 100.0 * count / total])


def write_folded(path: Path, counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for stack, count in sorted(counts.items()):
            handle.write(f"{stack} {count}\n")


def flamegraph(folded: Path, output: Path, title: str) -> None:
    script = Path(
        "/lustre/fsw/coreai_comparch_trtllm/jothomson/"
        "dynamo-numa/env/FlameGraph-v1.0/flamegraph.pl"
    )
    with folded.open("rb") as source, output.open("wb") as destination:
        completed = subprocess.run(
            [str(script), "--title", title, "--countname", "samples"],
            stdin=source,
            stdout=destination,
            stderr=subprocess.PIPE,
        )
    if completed.returncode != 0 or output.stat().st_size == 0:
        raise RuntimeError(completed.stderr.decode(errors="replace"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def measured_samples(
    samples: list[dict[str, Any]], started: str, ended: str
) -> list[dict[str, Any]]:
    lower = dt.datetime.fromisoformat(started)
    upper = dt.datetime.fromisoformat(ended)
    return [
        sample
        for sample in samples
        if lower <= dt.datetime.fromisoformat(sample["timestamp"]) <= upper
    ]


def process_cpu(samples: list[dict[str, Any]], name: str) -> dict[str, Any]:
    ticks_per_second = float(samples[0]["clock_ticks_per_second"])
    interval_cores: list[float] = []
    cumulative: dict[int, list[int]] = defaultdict(list)
    threads: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    previous: tuple[float, dict[int, int]] | None = None
    for sample in samples:
        members = sample.get("tracked_processes", {}).get(name, [])
        current = {
            int(item["pid"]): int(item["user_ticks"]) + int(item["system_ticks"])
            for item in members
        }
        for pid, ticks in current.items():
            cumulative[pid].append(ticks)
        for item in sample.get("tracked_threads", {}).get(name, []):
            pid = int(item["pid"])
            for thread in item["threads"]:
                key = (pid, int(thread["tid"]), str(thread["comm"]))
                threads[key].append(
                    int(thread["user_ticks"]) + int(thread["system_ticks"])
                )
        if previous is not None:
            old_time, old = previous
            elapsed = float(sample["monotonic_seconds"]) - old_time
            delta = sum(max(0, ticks - old.get(pid, ticks)) for pid, ticks in current.items())
            if elapsed > 0:
                interval_cores.append(delta / ticks_per_second / elapsed)
        previous = (float(sample["monotonic_seconds"]), current)
    cpu_ticks = sum(max(values) - min(values) for values in cumulative.values() if values)
    hottest = sorted(
        (
            {
                "pid": pid,
                "tid": tid,
                "comm": comm,
                "cpu_seconds": (max(values) - min(values)) / ticks_per_second,
            }
            for (pid, tid, comm), values in threads.items()
            if values
        ),
        key=lambda item: item["cpu_seconds"],
        reverse=True,
    )
    return {
        "cpu_seconds": cpu_ticks / ticks_per_second,
        "average_occupied_cores": sum(interval_cores) / len(interval_cores)
        if interval_cores
        else None,
        "p95_occupied_cores": percentile(interval_cores, 0.95),
        "hottest_thread": hottest[0] if hottest else None,
        "top_threads": hottest[:25],
    }


def addr2line_top_core(
    samples: list[dict[str, Any]], manifest: dict[str, Any], maps_path: Path
) -> list[dict[str, Any]]:
    core = str(manifest["core_path_from_proc_maps"])
    mappings: list[tuple[int, int, int]] = []
    for line in maps_path.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5].removesuffix(" (deleted)") != core:
            continue
        start, end = (int(value, 16) for value in fields[0].split("-"))
        mappings.append((start, end, int(fields[2], 16)))
    ips: Counter[int] = Counter()
    for sample in samples:
        for frame in sample["frames"]:
            if frame["dso"] == core:
                ips[frame["ip"]] += 1
    result: list[dict[str, Any]] = []
    for ip, count in ips.most_common(100):
        relative = next(
            (ip - start + offset for start, end, offset in mappings if start <= ip < end),
            None,
        )
        if relative is None:
            continue
        output = subprocess.check_output(
            ["addr2line", "-f", "-C", "-e", core, hex(relative)], text=True
        ).splitlines()
        result.append(
            {
                "runtime_ip": hex(ip),
                "dso_offset": hex(relative),
                "samples": count,
                "function": output[0] if output else "??",
                "source": output[1] if len(output) > 1 else "??:0",
            }
        )
    return result


def warning_summary(run_dir: Path) -> dict[str, Any]:
    pattern = re.compile(
        r"timing.manager|timing manager|event.loop|overrun|lagged|deadline|stalled",
        flags=re.IGNORECASE,
    )
    result: dict[str, list[str]] = {}
    for name in (
        "aiperf.log",
        "mocker.log",
        "mocker-numa0.log",
        "mocker-numa1.log",
        "frontend.log",
    ):
        path = run_dir / name
        if path.is_file():
            result[name] = [
                line for line in path.read_text(errors="replace").splitlines() if pattern.search(line)
            ]
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    latency = result["latency"]
    cpu = result["cpu"]
    capture = result["capture"]
    top_self = result["top_self_symbols"][:15]
    lines = [
        f"# Tyche AgentX c{result.get('concurrency', 2048)} frontend on-CPU profile",
        "",
        "## Latency and throughput",
        "",
        f"- TTFT p50/p95/p99: {latency.get('ttft_p50_ms')} / {latency.get('ttft_p95_ms')} / {latency.get('ttft_p99_ms')} ms",
        f"- Request throughput: {latency.get('request_throughput_rps')} requests/s",
        f"- Completed requests: {latency.get('request_count')}",
        "",
        "## CPU and saturation",
        "",
    ]
    for name in ("frontend", "mocker", "aiperf"):
        item = cpu[name]
        lines.append(
            f"- {name}: average {item.get('average_occupied_cores'):.3f} cores, "
            f"p95 {item.get('p95_occupied_cores'):.3f} cores; hottest thread "
            f"{item.get('hottest_thread')}"
            if item.get("average_occupied_cores") is not None
            else f"- {name}: CPU telemetry unavailable"
        )
    load = result["load_node"]
    if "mocker" in load and "aiperf" in load:
        load_lines = [
            f"- Mocker node total busy: average {load['mocker'].get('average')}%, p95 {load['mocker'].get('p95')}%",
            f"- AIPerf node total busy: average {load['aiperf'].get('average')}%, p95 {load['aiperf'].get('p95')}%",
        ]
    else:
        load_lines = [
            f"- Load node total busy: average {load.get('average')}%, p95 {load.get('p95')}%"
        ]
    lines.extend(
        [
            *load_lines,
            f"- Client/mocker contaminated: {result['client_mocker_contaminated']}",
            "",
            "## Perf quality",
            "",
            f"- Samples/frames: {result['perf_sample_count']} / {capture.get('frame_count')}",
            f"- Lost samples/chunks: {capture.get('lost_sample_count')} / {capture.get('lost_chunk_count_reported')}",
            f"- Unresolved frame fraction: {capture.get('unresolved_frame_fraction')}",
            f"- Mapped core DSO: `{capture.get('mapped_core_path')}`",
            f"- Core build ID: `{capture.get('mapped_core_build_id')}`",
            "",
            "## Top frontend self-time symbols",
            "",
            "| Rank | Symbol | Samples | Percent |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for index, item in enumerate(top_self, start=1):
        lines.append(
            f"| {index} | `{item['symbol']}` | {item['samples']} | {item['percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Flamegraphs",
            "",
            "- `aggregate-flamegraph.svg`",
            f"- `tid-{result['busiest_perf_tid']}-flamegraph.svg` (busiest sampled thread)",
            "",
            "## Event-loop and timing-manager evidence",
            "",
            f"- Matching warning/utilization lines: {sum(len(v) for v in result['warnings'].values())}",
            "- See `event-loop-and-timing-lines.json` for verbatim captured log lines.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    profile_root = args.job_dir / "profiles" / "agentx"
    run_dirs = sorted(path for path in profile_root.iterdir() if path.is_dir())
    if len(run_dirs) != 1:
        raise ValueError(f"expected one AgentX profile arm beneath {profile_root}, found {run_dirs}")
    run_dir = run_dirs[0]
    output = args.job_dir / "analysis"
    output.mkdir(parents=True, exist_ok=False)

    samples = parse_perf(run_dir / "perf-script.txt")
    by_tid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    comm_by_tid: dict[int, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        by_tid[sample["tid"]].append(sample)
        comm_by_tid[sample["tid"]][sample["comm"]] += 1
    aggregate_result = aggregate(samples)
    write_rankings(output / "self-time.csv", aggregate_result["self"], len(samples))
    write_rankings(output / "inclusive-time.csv", aggregate_result["inclusive"], len(samples))
    write_folded(output / "aggregate-folded.txt", aggregate_result["folded"])
    run_manifest = json.loads((run_dir / "run.json").read_text())
    concurrency = int(run_manifest.get("concurrency", 2048))
    flamegraph(
        output / "aggregate-folded.txt",
        output / "aggregate-flamegraph.svg",
        f"Dynamo frontend c{concurrency} cycles:u",
    )
    tid_dir = output / "per-tid"
    tid_dir.mkdir()
    thread_rows: list[dict[str, Any]] = []
    for tid, tid_samples in sorted(by_tid.items()):
        item = aggregate(tid_samples)
        write_folded(tid_dir / f"{tid}-folded.txt", item["folded"])
        write_rankings(tid_dir / f"{tid}-self.csv", item["self"], len(tid_samples))
        write_rankings(tid_dir / f"{tid}-inclusive.csv", item["inclusive"], len(tid_samples))
        thread_rows.append(
            {
                "tid": tid,
                "comm": comm_by_tid[tid].most_common(1)[0][0],
                "samples": len(tid_samples),
                "percent": 100.0 * len(tid_samples) / len(samples),
            }
        )
    thread_rows.sort(key=lambda item: item["samples"], reverse=True)
    with (output / "thread-ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tid", "comm", "samples", "percent"])
        writer.writeheader()
        writer.writerows(thread_rows)
    busiest_tid = int(thread_rows[0]["tid"])
    flamegraph(
        tid_dir / f"{busiest_tid}-folded.txt",
        output / f"tid-{busiest_tid}-flamegraph.svg",
        f"Dynamo frontend busiest TID {busiest_tid}",
    )

    measured_started = (run_dir / "PROFILING_STARTED").read_text().strip()
    measured_ended = (run_dir / "PROFILING_ENDED").read_text().strip()
    frontend_telemetry = measured_samples(
        load_jsonl(run_dir / "frontend-system-telemetry.jsonl"),
        measured_started,
        measured_ended,
    )
    separated_load_nodes = (run_dir / "mocker-system-telemetry.jsonl").is_file()
    if separated_load_nodes:
        mocker_telemetry = measured_samples(
            load_jsonl(run_dir / "mocker-system-telemetry.jsonl"),
            measured_started,
            measured_ended,
        )
        aiperf_telemetry = measured_samples(
            load_jsonl(run_dir / "aiperf-system-telemetry.jsonl"),
            measured_started,
            measured_ended,
        )
        mocker_parts = [
            process_cpu(mocker_telemetry, "mocker-numa0"),
            process_cpu(mocker_telemetry, "mocker-numa1"),
        ]
        mocker_cpu = {
            "cpu_seconds": sum(item["cpu_seconds"] for item in mocker_parts),
            "average_occupied_cores": sum(
                item["average_occupied_cores"] or 0.0 for item in mocker_parts
            ),
            "p95_occupied_cores": sum(
                item["p95_occupied_cores"] or 0.0 for item in mocker_parts
            ),
            "hottest_thread": max(
                (
                    item["hottest_thread"]
                    for item in mocker_parts
                    if item["hottest_thread"] is not None
                ),
                key=lambda item: item["cpu_seconds"],
                default=None,
            ),
            "top_threads": sorted(
                [thread for item in mocker_parts for thread in item["top_threads"]],
                key=lambda item: item["cpu_seconds"],
                reverse=True,
            )[:25],
            "processes": mocker_parts,
        }
        cpu = {
            "frontend": process_cpu(frontend_telemetry, "frontend"),
            "mocker": mocker_cpu,
            "aiperf": process_cpu(aiperf_telemetry, "aiperf"),
        }
    else:
        load_telemetry = measured_samples(
            load_jsonl(run_dir / "load-system-telemetry.jsonl"),
            measured_started,
            measured_ended,
        )
        cpu = {
            "frontend": process_cpu(frontend_telemetry, "frontend"),
            "mocker": process_cpu(load_telemetry, "mocker"),
            "aiperf": process_cpu(load_telemetry, "aiperf"),
        }
    records = next((run_dir / "load_artifacts").glob("**/profile_export.jsonl"), None)
    if records is None:
        raise FileNotFoundError("profile_export.jsonl")
    latency = parse_aiperf(records)
    completed = int(latency.get("request_count", 0) or 0)
    for item in cpu.values():
        item["cpu_ms_per_completed_request"] = (
            1000.0 * item["cpu_seconds"] / completed if completed else None
        )
    if separated_load_nodes:
        mocker_acceptance = json.loads((run_dir / "mocker-acceptance.json").read_text())
        aiperf_acceptance = json.loads((run_dir / "aiperf-acceptance.json").read_text())
        load_node = {
            "mocker": mocker_acceptance["node_busy_percent"],
            "aiperf": aiperf_acceptance["node_busy_percent"],
        }
        client_mocker_contaminated = not (
            mocker_acceptance["headroom_accepted"]
            and aiperf_acceptance["headroom_accepted"]
            and aiperf_acceptance["client_health_accepted"]
        )
    else:
        remote = json.loads((run_dir / "remote_acceptance.json").read_text())
        load_node = remote["load_node_busy_percent"]
        client_mocker_contaminated = remote["client_mocker_contaminated"]
    capture = json.loads((run_dir / "perf-capture.json").read_text())
    dso = json.loads((run_dir / "frontend-dso-manifest.json").read_text())
    addr2line = addr2line_top_core(samples, dso, run_dir / "frontend.maps")
    (output / "addr2line-top-core.json").write_text(
        json.dumps(addr2line, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    warnings = warning_summary(run_dir)
    (output / "event-loop-and-timing-lines.json").write_text(
        json.dumps(warnings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    top_self = [
        {"symbol": symbol, "samples": count, "percent": 100.0 * count / len(samples)}
        for symbol, count in aggregate_result["self"].most_common(100)
    ]
    result = {
        "schema_version": 1,
        "job_dir": str(args.job_dir),
        "run_dir": str(run_dir),
        "latency": latency,
        "cpu": cpu,
        "concurrency": concurrency,
        "load_node": load_node,
        "client_mocker_contaminated": client_mocker_contaminated,
        "capture": capture,
        "perf_sample_count": len(samples),
        "busiest_perf_tid": busiest_tid,
        "thread_ranking": thread_rows,
        "top_self_symbols": top_self,
        "warnings": warnings,
        "addr2line_core_frame_count": len(addr2line),
    }
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(output / "REPORT.md", result)
    artifacts = sorted(
        str(path.relative_to(args.job_dir))
        for path in args.job_dir.rglob("*")
        if path.is_file()
    )
    (args.job_dir / "ARTIFACT_INDEX.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
