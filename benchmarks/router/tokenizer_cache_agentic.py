#!/usr/bin/env python3
"""A/B the local tokenizer L1 against local L1 + shared Valkey L2.

The workload is an agentic conversation whose complete message history grows
from 2 to 32 messages.  ``pinned`` keeps every conversation on one frontend,
matching normal AIPerf multi-turn behavior.  ``handoff`` restarts AIPerf for
each depth and rotates the frontend URL order, so each full-history request is
handled by a different frontend while the topology and caches stay alive.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.router.valkey_aiperf.arm import make_request  # noqa: E402
from benchmarks.router.valkey_aiperf.common import (  # noqa: E402
    HarnessRequest,
    MockerProcessGroup,
    apply_managed_process_affinity,
)
from benchmarks.router.valkey_aiperf.frontend import wait_for_frontends  # noqa: E402
from benchmarks.router.valkey_aiperf.loadgen import run_aiperf  # noqa: E402
from benchmarks.router.valkey_aiperf.provenance import (  # noqa: E402
    enrich_dynamo_core_build_profile,
    environment,
    file_provenance,
)
from tests.router.mocker_process import MockerProcess  # noqa: E402
from tests.router.router_process import FrontendRouterProcess  # noqa: E402
from tests.utils.port_utils import allocate_ports, deallocate_ports  # noqa: E402


MODEL = "Qwen/Qwen3-0.6B"
DEPTHS = (1, 2, 4, 8, 16)
MESSAGE_COUNTS = {depth: 2 * depth for depth in DEPTHS}
CACHE_METRICS = (
    "dynamo_frontend_tokenizer_cache_hits_total",
    "dynamo_frontend_tokenizer_cache_misses_total",
    "dynamo_frontend_tokenizer_cache_l2_hits_total",
    "dynamo_frontend_tokenizer_cache_l2_misses_total",
    "dynamo_frontend_tokenizer_cache_l2_errors_total",
    "dynamo_frontend_tokenizer_cache_l2_write_drops_total",
    "dynamo_frontend_tokenizer_cache_l2_write_errors_total",
    "dynamo_frontend_tokenizer_cache_l2_lookup_seconds_count",
    "dynamo_frontend_tokenizer_cache_l2_lookup_seconds_sum",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for symbols and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded range from a source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a focused test command and return exit status and output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

SYSTEM_BLOCK = """
You are a senior coding agent operating in a production inference repository.
Preserve user changes, inspect evidence before editing, validate inputs at every
boundary, and keep changes reviewable. Build a concrete hypothesis from source
and tests. For each action, explain the expected observation, use focused tools,
and update the plan when evidence contradicts it. Never expose credentials or
silently ignore errors. Finish with validation results, performance caveats,
and exact artifact paths. The current investigation concerns asynchronous
request routing, tokenizer-prefix reuse, distributed cache availability, and
tail-latency behavior under frontend failover and horizontal scaling.
""".strip()


def system_prompt() -> str:
    sections = [
        f"Repository operating policy section {index}:\n{SYSTEM_BLOCK}"
        for index in range(1, 25)
    ]
    return "\n\n".join(sections)


def build_messages(session: int, depth: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt()},
        {
            "role": "user",
            "content": (
                f"Agent session {session:06d}: diagnose a throughput regression in "
                "the frontend tokenizer and routing path. Start from the request "
                "preprocessor, preserve exact token IDs, and propose the next focused "
                "inspection. The session nonce is intentionally unique so histories "
                "cannot share a cache entry after this message."
            ),
        },
    ]
    for step in range(1, depth):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        f"Step {step} for session {session:06d}: I inspected the prior "
                        "evidence and will search the preprocessing boundary, compare "
                        "the cached token prefix with a full encode, then check whether "
                        "the async lookup is on the request critical path. I will keep "
                        "the router policy fixed so the cache storage layer is the only "
                        "A/B variable. The next command targets the narrowest source "
                        "surface and records both correctness and timing evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Tool observation {step} for session {session:06d}: the focused "
                        "search returned tokenizer construction, chat-template rendering, "
                        "special-token boundaries, and cache counters. The test output "
                        "shows exact token parity for the inspected prefix. Continue the "
                        "investigation using this growing transcript; quantify lookup, "
                        "tokenization, TTFT, ITL, and request throughput. Preserve prior "
                        "messages verbatim because another frontend may receive the next "
                        "request after a stateless handoff. "
                        + "Evidence payload: "
                        + " ".join(
                            f"symbol_{session % 97}_{step}_{item}=validated"
                            for item in range(48)
                        )
                    ),
                },
            ]
        )
    assert len(messages) == MESSAGE_COUNTS[depth]
    return messages


def dataset_row(session: int, depth: int, session_prefix: str) -> dict[str, Any]:
    return {
        "session_id": f"{session_prefix}-{session:06d}",
        "messages": build_messages(session, depth),
        "tools": TOOLS,
        "output_length": 16,
        "extra": {
            "min_tokens": 16,
            "ignore_eos": True,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def generate_datasets(root: Path, sessions: int) -> dict[str, Any]:
    dataset_dir = root / "datasets"
    handoff: dict[int, str] = {}
    for depth in DEPTHS:
        path = dataset_dir / f"handoff-depth-{depth:02d}.jsonl"
        write_jsonl(
            path,
            [dataset_row(session, depth, "handoff") for session in range(sessions)],
        )
        handoff[depth] = str(path)

    pinned_path = dataset_dir / "pinned.jsonl"
    pinned_rows = [
        dataset_row(session, depth, "pinned")
        for session in range(sessions)
        for depth in DEPTHS
    ]
    write_jsonl(pinned_path, pinned_rows)
    manifest = {
        "sessions": sessions,
        "depths": list(DEPTHS),
        "message_counts": MESSAGE_COUNTS,
        "pinned": str(pinned_path),
        "handoff": handoff,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def add_nominal_isl(manifest: dict[str, Any]) -> None:
    """Record exact row-zero ISL using the same Qwen chat template as Dynamo."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    nominal: dict[int, int] = {}
    for depth in DEPTHS:
        path = Path(manifest["handoff"][depth])
        with path.open(encoding="utf-8") as dataset:
            row = json.loads(next(dataset))
        prompt = tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=True,
        )
        nominal[depth] = len(tokenizer.encode(prompt, add_special_tokens=False))
    manifest["nominal_isl_tokens"] = nominal
    manifest_path = Path(manifest["pinned"]).parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_port(port: int, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"port {port} did not become ready")


@contextlib.contextmanager
def valkey_server(binary: Path, port: int, log_path: Path) -> Iterator[None]:
    command = [
        "taskset",
        "--cpu-list",
        "3",
        str(binary),
        "--bind",
        "127.0.0.1",
        "--port",
        str(port),
        "--protected-mode",
        "yes",
        "--save",
        "",
        "--appendonly",
        "no",
        "--daemonize",
        "no",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_port(port)
            yield
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def scrape_cache_metrics(ports: list[int]) -> dict[str, float]:
    totals = {name: 0.0 for name in CACHE_METRICS}
    for port in ports:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            body = response.read().decode("utf-8")
        for line in body.splitlines():
            if not line or line.startswith("#") or "{" in line:
                continue
            name, _, raw_value = line.partition(" ")
            if name in totals:
                totals[name] += float(raw_value)
    return totals


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after.get(name, 0.0) - value for name, value in before.items()}


def metric_value(record: dict[str, Any], name: str) -> float | None:
    metric = (record.get("metrics") or {}).get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    return float(value) if isinstance(value, int | float) else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def parse_records(artifact_dir: Path) -> dict[int, dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    path = artifact_dir / "profile_export.jsonl"
    with path.open(encoding="utf-8") as records:
        for line in records:
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            if metadata.get("benchmark_phase") != "profiling":
                continue
            if record.get("error") is not None or metadata.get("was_cancelled"):
                continue
            groups[int(metadata.get("turn_index") or 0)].append(record)

    parsed: dict[int, dict[str, Any]] = {}
    for turn_index, records in sorted(groups.items()):
        starts = [record["metadata"]["request_start_ns"] for record in records]
        ends = [record["metadata"]["request_end_ns"] for record in records]
        wall_seconds = (max(ends) - min(starts)) / 1e9
        values = {
            name: [
                value
                for record in records
                if (value := metric_value(record, name)) is not None
            ]
            for name in (
                "request_latency",
                "time_to_first_token",
                "inter_token_latency",
                "input_sequence_length",
                "output_sequence_length",
            )
        }
        parsed[turn_index] = {
            "completed": len(records),
            "wall_seconds": wall_seconds,
            "request_throughput_rps": len(records) / wall_seconds,
            "request_latency_ms_p50": percentile(values["request_latency"], 0.50),
            "request_latency_ms_p95": percentile(values["request_latency"], 0.95),
            "ttft_ms_p50": percentile(values["time_to_first_token"], 0.50),
            "ttft_ms_p95": percentile(values["time_to_first_token"], 0.95),
            "itl_ms_p50": percentile(values["inter_token_latency"], 0.50),
            "itl_ms_p95": percentile(values["inter_token_latency"], 0.95),
            "isl_tokens_avg": mean_or_none(values["input_sequence_length"]),
            "osl_tokens_avg": mean_or_none(values["output_sequence_length"]),
        }
    return parsed


def aiperf_command(
    args: argparse.Namespace,
    dataset: Path,
    artifact_dir: Path,
    urls: list[str],
    request_count: int,
) -> list[str]:
    command = [
        "taskset",
        "--cpu-list",
        "16-23",
        str(args.aiperf),
        "profile",
        "--artifact-dir",
        str(artifact_dir),
        "--model",
        MODEL,
        "--tokenizer",
        MODEL,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--url-strategy",
        "round_robin",
    ]
    for url in urls:
        command.extend(("--url", url))
    command.extend(
        (
            "--custom-dataset-type",
            "mooncake_trace",
            "--input-file",
            str(dataset),
            "--no-fixed-schedule",
            "--dataset-sampling-strategy",
            "sequential",
            "--concurrency",
            str(args.concurrency),
            "--request-rate",
            "inf",
            "--request-count",
            str(request_count),
            "--request-timeout-seconds",
            "120",
            "--random-seed",
            "100",
            "--workers-max",
            str(min(args.concurrency, 64)),
            "--record-processors",
            "4",
            "--export-level",
            "records",
            "--no-gpu-telemetry",
            "--no-server-metrics",
            "--ui",
            "simple",
        )
    )
    return command


def run_profile(
    args: argparse.Namespace,
    run_dir: Path,
    name: str,
    dataset: Path,
    urls: list[str],
    request_count: int,
) -> dict[str, Any]:
    artifact_dir = run_dir / name
    artifact_dir.mkdir(parents=True)
    command = aiperf_command(args, dataset, artifact_dir, urls, request_count)
    (artifact_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    outcome = run_aiperf(
        command,
        run_dir=run_dir,
        log_path=artifact_dir / "aiperf.log",
        timeout_seconds=args.timeout,
    )
    if outcome["returncode"] not in (0, None) or outcome["timed_out"]:
        raise RuntimeError(f"aiperf failed for {name}: {outcome}")
    records_path = artifact_dir / "profile_export.jsonl"
    if not records_path.is_file():
        raise RuntimeError(
            f"aiperf produced no per-request records for {name}; see "
            f"{artifact_dir / 'aiperf.log'}"
        )
    parsed = parse_records(artifact_dir)
    return {
        "name": name,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "command": command,
        "outcome": outcome,
        "turns": parsed,
    }


def run_sample(
    args: argparse.Namespace,
    root: Path,
    manifest: dict[str, Any],
    scenario: str,
    arm: str,
    repetition: int,
) -> dict[str, Any]:
    run_dir = root / "runs" / f"{scenario}-r{repetition:02d}-{arm}"
    run_dir.mkdir(parents=True)
    ports = allocate_ports(args.frontends + 1, 18000)
    frontend_ports, valkey_port = ports[:-1], ports[-1]
    requests: list[HarnessRequest] = []
    cache_scope = f"agentic-{scenario}-r{repetition}-{time.time_ns()}"
    overrides = {
        "DYN_TOKENIZER": "default",
        "DYN_TOKENIZER_CACHE": "1",
        "DYN_TOKENIZER_CACHE_EXTEND": "1",
        "DYN_TOKENIZER_CACHE_BYTES": str(args.l1_bytes),
        "DYN_TOKENIZER_CACHE_L2_URL": (
            f"valkey://127.0.0.1:{valkey_port}" if arm == "valkey" else None
        ),
        "DYN_TOKENIZER_CACHE_L2_ALLOW_INSECURE_PLAINTEXT": (
            "true" if arm == "valkey" else None
        ),
        "DYN_TOKENIZER_CACHE_L2_SCOPE": cache_scope if arm == "valkey" else None,
        "DYN_TOKENIZER_CACHE_L2_POOL_SIZE": "16" if arm == "valkey" else None,
        "DYN_TOKENIZER_CACHE_L2_TIMEOUT_MS": "50" if arm == "valkey" else None,
        "DYN_TOKENIZER_CACHE_L2_MAX_PENDING_WRITES": "1024" if arm == "valkey" else None,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result: dict[str, Any] = {
        "scenario": scenario,
        "arm": arm,
        "repetition": repetition,
        "frontend_ports": frontend_ports,
        "valkey_port": valkey_port if arm == "valkey" else None,
        "cache_scope": cache_scope if arm == "valkey" else None,
        "nominal_isl_tokens": manifest["nominal_isl_tokens"],
        "profiles": [],
    }
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(environment(overrides))
            if arm == "valkey":
                stack.enter_context(
                    valkey_server(args.valkey_server, valkey_port, run_dir / "valkey.log")
                )

            mocker = MockerProcess(
                make_request(run_dir / "mocker", requests),
                mocker_args={
                    "speedup_ratio": 100000,
                    "block_size": 16,
                    "num_gpu_blocks": 131072,
                    "max_num_seqs": 16384,
                    "max_num_batched_tokens": 16384,
                    "enable_prefix_caching": True,
                },
                num_mockers=args.mockers,
                store_backend="etcd",
                request_plane="tcp",
                extra_env={
                    "DYN_TOKENIZER_CACHE": "0",
                    "DYN_TOKENIZER_CACHE_L2_URL": "",
                    "DYN_TOKENIZER_CACHE_L2_ALLOW_INSECURE_PLAINTEXT": "false",
                },
            )
            apply_managed_process_affinity(mocker._process, "0-2")
            stack.enter_context(mocker)
            workers = MockerProcessGroup([mocker])

            for index, port in enumerate(frontend_ports, start=1):
                frontend = FrontendRouterProcess(
                    make_request(run_dir / f"frontend-{index}", requests),
                    block_size=16,
                    frontend_port=port,
                    namespace=workers.namespace,
                    store_backend="etcd",
                    request_plane="tcp",
                    min_initial_workers=args.mockers,
                    router_mode="kv",
                    router_replica_sync=True,
                )
                apply_managed_process_affinity(frontend, "4-15")
                stack.enter_context(frontend)

            asyncio.run(wait_for_frontends(frontend_ports, workers, 90))
            urls = [f"http://127.0.0.1:{port}" for port in frontend_ports]

            if scenario == "pinned":
                before = scrape_cache_metrics(frontend_ports)
                profile = run_profile(
                    args,
                    run_dir,
                    "pinned",
                    Path(manifest["pinned"]),
                    urls,
                    args.sessions * len(DEPTHS),
                )
                time.sleep(0.5)
                after = scrape_cache_metrics(frontend_ports)
                profile["cache_metric_delta"] = metric_delta(before, after)
                result["profiles"].append(profile)
            else:
                for stage_index, depth in enumerate(DEPTHS):
                    rotated_urls = urls[stage_index % len(urls) :] + urls[: stage_index % len(urls)]
                    before = scrape_cache_metrics(frontend_ports)
                    profile = run_profile(
                        args,
                        run_dir,
                        f"depth-{depth:02d}",
                        Path(manifest["handoff"][depth]),
                        rotated_urls,
                        args.sessions,
                    )
                    time.sleep(0.5)
                    after = scrape_cache_metrics(frontend_ports)
                    profile["depth"] = depth
                    profile["message_count"] = MESSAGE_COUNTS[depth]
                    profile["cache_metric_delta"] = metric_delta(before, after)
                    result["profiles"].append(profile)

            result["final_cache_metrics"] = scrape_cache_metrics(frontend_ports)
            if arm == "valkey":
                dbsize = subprocess.run(
                    [str(args.valkey_cli), "-p", str(valkey_port), "DBSIZE"],
                    capture_output=True,
                    check=True,
                    text=True,
                )
                result["valkey_dbsize"] = int(dbsize.stdout.strip())
    finally:
        for request in reversed(requests):
            request.close()
        deallocate_ports(ports)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def flatten_points(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for result in results:
        if result["scenario"] == "pinned":
            turns = result["profiles"][0]["turns"]
            for turn_index, depth in enumerate(DEPTHS):
                point = dict(turns[str(turn_index)] if str(turn_index) in turns else turns[turn_index])
                point.update(
                    scenario="pinned",
                    arm=result["arm"],
                    repetition=result["repetition"],
                    depth=depth,
                    message_count=MESSAGE_COUNTS[depth],
                )
                if point.get("isl_tokens_avg") is None:
                    nominal = result["nominal_isl_tokens"]
                    point["isl_tokens_avg"] = nominal.get(str(depth), nominal.get(depth))
                points.append(point)
        else:
            for profile in result["profiles"]:
                turns = profile["turns"]
                point = dict(turns.get("0", turns.get(0)))
                point.update(
                    scenario="handoff",
                    arm=result["arm"],
                    repetition=result["repetition"],
                    depth=profile["depth"],
                    message_count=profile["message_count"],
                )
                if point.get("isl_tokens_avg") is None:
                    nominal = result["nominal_isl_tokens"]
                    depth = profile["depth"]
                    point["isl_tokens_avg"] = nominal.get(str(depth), nominal.get(depth))
                points.append(point)
    return points


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    points = flatten_points(results)
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[(point["scenario"], point["message_count"], point["arm"])].append(point)
    rows: list[dict[str, Any]] = []
    scenarios = tuple(dict.fromkeys(point["scenario"] for point in points))
    for scenario in scenarios:
        for depth in DEPTHS:
            message_count = MESSAGE_COUNTS[depth]
            row: dict[str, Any] = {
                "scenario": scenario,
                "depth": depth,
                "message_count": message_count,
            }
            for arm in ("inprocess", "valkey"):
                samples = grouped[(scenario, message_count, arm)]
                row[arm] = {
                    metric: statistics.median(float(sample[metric]) for sample in samples)
                    for metric in (
                        "request_throughput_rps",
                        "request_latency_ms_p50",
                        "request_latency_ms_p95",
                        "ttft_ms_p50",
                        "ttft_ms_p95",
                        "itl_ms_p50",
                        "itl_ms_p95",
                        "isl_tokens_avg",
                        "osl_tokens_avg",
                    )
                    if all(sample.get(metric) is not None for sample in samples)
                }
            baseline = row["inprocess"]["request_throughput_rps"]
            candidate = row["valkey"]["request_throughput_rps"]
            row["valkey_rps_delta_percent"] = (candidate / baseline - 1.0) * 100.0
            baseline_ttft = row["inprocess"].get("ttft_ms_p50")
            candidate_ttft = row["valkey"].get("ttft_ms_p50")
            row["valkey_ttft_p50_delta_percent"] = (
                (candidate_ttft / baseline_ttft - 1.0) * 100.0
                if baseline_ttft and candidate_ttft is not None
                else None
            )
            rows.append(row)
    return {"rows": rows, "points": points}


def plot_summary(summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = tuple(dict.fromkeys(row["scenario"] for row in summary["rows"]))
    figure, axes = plt.subplots(
        2, len(scenarios), figsize=(6 * len(scenarios), 8), sharex="col", squeeze=False
    )
    for column, scenario in enumerate(scenarios):
        rows = [row for row in summary["rows"] if row["scenario"] == scenario]
        messages = [row["message_count"] for row in rows]
        for arm, label, marker in (
            ("inprocess", "In-process L1", "o"),
            ("valkey", "L1 + Valkey L2", "s"),
        ):
            axes[0, column].plot(
                messages,
                [row[arm]["request_throughput_rps"] for row in rows],
                marker=marker,
                label=label,
            )
            axes[1, column].plot(
                messages,
                [row[arm]["ttft_ms_p50"] for row in rows],
                marker=marker,
                label=label,
            )
        axes[0, column].set_title(f"{scenario}: request throughput")
        axes[0, column].set_ylabel("requests/s")
        axes[1, column].set_title(f"{scenario}: TTFT p50")
        axes[1, column].set_ylabel("ms")
        axes[1, column].set_xlabel("messages in full request history")
        for row in axes[:, column]:
            row.grid(alpha=0.25)
            row.set_xscale("log", base=2)
            row.set_xticks([2, 4, 8, 16, 32], labels=["2", "4", "8", "16", "32"])
    axes[0, 0].legend()
    figure.suptitle("Agentic growing-history tokenizer-cache A/B (median of repetitions)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=384)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--frontends", type=int, default=3)
    parser.add_argument("--mockers", type=int, default=4)
    parser.add_argument("--l1-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--scenario", choices=("both", "pinned", "handoff"), default="both")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--aiperf", type=Path, default=REPO / "dynamo/bin/aiperf")
    parser.add_argument(
        "--valkey-server",
        type=Path,
        default=Path("/home/biswaranjanp/dev/valkey/src/valkey-server"),
    )
    parser.add_argument(
        "--valkey-cli",
        type=Path,
        default=Path("/home/biswaranjanp/dev/valkey/src/valkey-cli"),
    )
    args = parser.parse_args()
    if min(args.runs, args.sessions, args.concurrency, args.frontends, args.mockers) < 1:
        parser.error("run, session, concurrency, frontend, and mocker counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = generate_datasets(root, args.sessions)
    add_nominal_isl(manifest)
    if args.generate_only:
        return 0

    import dynamo._core as core

    core_provenance = file_provenance(Path(core.__file__))
    enrich_dynamo_core_build_profile(core_provenance)
    provenance = {
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain=v1"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout
        ),
        "dynamo_core": core_provenance,
        "aiperf": file_provenance(args.aiperf),
        "valkey_server": file_provenance(args.valkey_server),
        "configuration": {
            "model": MODEL,
            "depths": list(DEPTHS),
            "message_counts": MESSAGE_COUNTS,
            "runs": args.runs,
            "sessions": args.sessions,
            "concurrency": args.concurrency,
            "frontends": args.frontends,
            "mockers": args.mockers,
            "l1_bytes": args.l1_bytes,
            "request_rate": "inf",
            "output_tokens": 16,
        },
    }
    if core_provenance.get("rust_build_profile") != "release":
        raise RuntimeError(f"release build required: {core_provenance}")
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    scenarios = ("pinned", "handoff") if args.scenario == "both" else (args.scenario,)
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for repetition in range(1, args.runs + 1):
            arms = ("inprocess", "valkey") if repetition % 2 else ("valkey", "inprocess")
            for arm in arms:
                print(f"START scenario={scenario} repetition={repetition} arm={arm}", flush=True)
                result = run_sample(args, root, manifest, scenario, arm, repetition)
                results.append(result)
                (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"DONE scenario={scenario} repetition={repetition} arm={arm}", flush=True)

    summary = summarize(results)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_summary(summary, root / "agentic-cache-ab.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
