# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize campaign arms over each arm's measure window (window-start .. window-end):

- frontend arms: completed requests, TTFB and duration from the Global Frontend's
  /metrics scraped at the window's edges (aiperf only generates load; it can't process
  per-request records at these rates);
- the direct arm: the completed count aiperf's two clients report for the whole
  profiling phase (settle included);
- CPU of the frontend, the mocks and the client nodes from the 1 s telemetry; the
  frontend's user/system split from its CPU ticks at the window's edges;
- TCP segments the frontend node sent per request, from its nstat snapshots.

Writes summary.json in each arm's directory and prints one Markdown table.

usage: summarize.py RESULTS_DIR [RESULTS_DIR ...]
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

CLK_TCK = os.sysconf("SC_CLK_TCK")
SAMPLE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+(\S+)$")
LABEL = re.compile(r'(\w+)="([^"]*)"')
PREFIX = "dynamo_global_frontend_"


def prometheus(path):
    """(scraped_at, [(name, labels, value)], (user, system) CPU ticks or None)."""
    scraped, samples, ticks = None, [], None
    for line in path.read_text().splitlines():
        if line.startswith("# scraped_at"):
            scraped = float(line.split()[2])
            continue
        if line.startswith("# cpu_ticks"):
            ticks = tuple(int(value) for value in line.split()[2:4])
            continue
        match = SAMPLE.match(line)
        if match:
            labels = dict(LABEL.findall(match.group(2) or ""))
            samples.append((match.group(1), labels, float(match.group(3))))
    return scraped, samples, ticks


def out_segments(path):
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "TcpOutSegs":
            return int(parts[1])
    return None


def delta(before, after, name, by=None):
    """after - before for one metric, summed over labels (grouped by the label `by`)."""
    totals = defaultdict(float)
    for sign, samples in ((-1, before), (1, after)):
        for metric, labels, value in samples:
            if metric == name:
                totals[labels.get(by) if by else None] += sign * value
    return dict(totals) if by else totals[None]


def histogram_quantile(before, after, name, quantile):
    buckets = delta(before, after, name + "_bucket", by="le")
    bounds = sorted((float(le), count) for le, count in buckets.items())
    if not bounds or bounds[-1][1] <= 0:
        return None
    target = quantile * bounds[-1][1]
    lower_bound, lower_count = 0.0, 0.0
    for bound, count in bounds:
        if count >= target:
            if bound == float("inf"):
                return lower_bound
            share = (
                (target - lower_count) / (count - lower_count)
                if count > lower_count
                else 0
            )
            return lower_bound + (bound - lower_bound) * share
        lower_bound, lower_count = bound, count
    return None


def telemetry(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def at(series, when):
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        if t0 <= when <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (when - t0) / (t1 - t0)
    return None


def rate(rows, value, start, end):
    series = [(row["t"], value(row)) for row in rows if value(row) is not None]
    lo, hi = at(series, start), at(series, end)
    return None if lo is None or hi is None else (hi - lo) / (end - start)


def busy(rows, start, end):
    out = {}
    for domain in ("numa0", "numa1"):
        busy_rate = rate(rows, lambda row: row["cpu"][domain][0], start, end)
        total_rate = rate(rows, lambda row: row["cpu"][domain][1], start, end)
        if busy_rate is not None and total_rate:
            out[domain] = round(busy_rate / total_rate, 3)
    return out


def client_completed(path):
    """(sent, completed, phase seconds) from aiperf's profiling-phase lines."""
    if not path.exists():
        return None
    text = path.read_text()
    match = re.search(r"sending complete \| sent=([\d,]+), completed=([\d,]+)", text)
    seconds = re.search(r"target: ([\d.]+)s duration", text)
    if not match or not seconds:
        return None
    return (
        *(int(group.replace(",", "")) for group in match.groups()),
        float(seconds.group(1)),
    )


def summarize(directory):
    roles = json.loads((directory / "roles.json").read_text())
    start = float((directory / "window-start").read_text())
    end = float((directory / "window-end").read_text())
    clients = [
        client_completed(directory / roles[key] / "aiperf-phase.log")
        for key in ("client_c", "client_d")
    ]
    summary = {key: roles[key] for key in ("label", "kind", "build", "cpus", "args")}
    summary["clients_sent_completed"] = clients
    if roles["kind"] == "frontend":
        (t0, before, ticks0), (t1, after, ticks1) = (
            prometheus(directory / roles["frontend"] / f"metrics-{tag}.prom")
            for tag in ("start", "end")
        )
        start, end = t0, t1
        completed = delta(before, after, PREFIX + "request_duration_seconds_count")
        statuses = delta(before, after, PREFIX + "requests_total", by="status")
        summary.update(
            requests_per_s=completed / (end - start),
            statuses={
                status: int(count) for status, count in statuses.items() if count
            },
            rejected=int(delta(before, after, PREFIX + "rejected_requests_total")),
            response_errors=int(delta(before, after, PREFIX + "response_errors_total")),
            ttfb_ms={
                q: 1000
                * (
                    histogram_quantile(
                        before, after, PREFIX + "time_to_first_byte_seconds", v
                    )
                    or 0
                )
                for q, v in (("p50", 0.5), ("p99", 0.99))
            },
            duration_ms={
                q: 1000
                * (
                    histogram_quantile(
                        before, after, PREFIX + "request_duration_seconds", v
                    )
                    or 0
                )
                for q, v in (("p50", 0.5), ("p99", 0.99))
            },
        )
    else:
        # The counters cover the whole phase, settle included.
        summary["requests_per_s"] = (
            sum(c[1] / c[2] for c in clients) if all(clients) else None
        )
    summary["window_s"] = round(end - start, 1)

    nodes = {}
    for key in ("frontend", "mocks", "client_c", "client_d"):
        rows = telemetry(directory / roles[key] / "telemetry.jsonl")
        nodes[key] = {"busy": busy(rows, start, end)}
        if key == "frontend" and roles["kind"] == "frontend":
            cores = rate(
                rows, lambda row: row["procs"].get("global-frontend"), start, end
            )
            cores = None if cores is None else cores / CLK_TCK
            window = [row for row in rows if start <= row["t"] <= end]
            nodes[key].update(
                cores=cores,
                cpu_ms_per_request=None
                if cores is None or not summary["requests_per_s"]
                else cores * 1000 / summary["requests_per_s"],
                rss_gib_max=max(
                    (row["rss"].get("global-frontend", 0) for row in window), default=0
                )
                / 2**30,
                rx_gbit_s=8e-9
                * (rate(rows, lambda row: row["nic"]["rx_bytes"], start, end) or 0),
                tx_gbit_s=8e-9
                * (rate(rows, lambda row: row["nic"]["tx_bytes"], start, end) or 0),
                retransmits_per_s=rate(rows, lambda row: row["retrans"], start, end),
            )
            if ticks0 and ticks1:
                user, system = (
                    (b - a) / CLK_TCK / (end - start) for a, b in zip(ticks0, ticks1)
                )
                nodes[key].update(user_cores=user, system_cores=system)
            host = directory / roles[key]
            segments = [
                out_segments(host / f"netstat-{tag}.txt") for tag in ("start", "end")
            ]
            netstat_s = [
                float(
                    next(
                        line
                        for line in (host / f"netstat-{tag}.txt")
                        .read_text()
                        .splitlines()
                        if line.startswith("# at")
                    ).split()[2]
                )
                for tag in ("start", "end")
            ]
            if all(segments) and summary["requests_per_s"]:
                nodes[key]["out_segments_per_request"] = (
                    (segments[1] - segments[0])
                    / (netstat_s[1] - netstat_s[0])
                    / summary["requests_per_s"]
                )
        if key == "mocks":
            cores = rate(rows, lambda row: row["procs"].get("mock-pool"), start, end)
            nodes[key]["cores"] = None if cores is None else cores / CLK_TCK
    summary["nodes"] = nodes
    (directory / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def fmt(value, digits=1):
    return "-" if value is None else f"{value:,.{digits}f}"


def main():
    print(
        "| arm | frontend CPUs | req/s | frontend cores | frontend ms/req | TTFB p50 / p99 ms "
        "| frontend user / sys ms/req | out segments/req | duration p50 / p99 ms | frontend RSS GiB "
        "| frontend NIC rx / tx Gbit/s | client busy C, D | mock cores | non-200 |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for directory in sys.argv[1:]:
        s = summarize(Path(directory))
        frontend = s["nodes"]["frontend"]
        clients = ", ".join(
            "/".join(fmt(v, 2) for v in s["nodes"][key]["busy"].values())
            for key in ("client_c", "client_d")
        )
        ttfb, duration = s.get("ttfb_ms", {}), s.get("duration_ms", {})
        rps = s["requests_per_s"] or 0
        split = (
            "-"
            if "user_cores" not in frontend or not rps
            else f"{1000 * frontend['user_cores'] / rps:.2f} / {1000 * frontend['system_cores'] / rps:.2f}"
        )
        bad = (
            sum(
                count
                for status, count in s.get("statuses", {}).items()
                if status != "200"
            )
            + s.get("rejected", 0)
            + s.get("response_errors", 0)
        )
        print(
            f"| {s['label']} | {s['cpus']} | {fmt(s['requests_per_s'], 0)} | {fmt(frontend.get('cores'))} "
            f"| {fmt(frontend.get('cpu_ms_per_request'), 2)} | {fmt(ttfb.get('p50'))} / {fmt(ttfb.get('p99'))} "
            f"| {split} | {fmt(frontend.get('out_segments_per_request'), 0)} "
            f"| {fmt(duration.get('p50'))} / {fmt(duration.get('p99'))} | {fmt(frontend.get('rss_gib_max'), 2)} "
            f"| {fmt(frontend.get('rx_gbit_s'))} / {fmt(frontend.get('tx_gbit_s'))} "
            f"| {clients} | {fmt(s['nodes']['mocks'].get('cores'))} | {bad if s['kind'] == 'frontend' else '-'} |"
        )


if __name__ == "__main__":
    main()
