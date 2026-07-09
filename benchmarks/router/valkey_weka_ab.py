#!/usr/bin/env python3
"""Interleaved Weka-derived A/B: in-process state versus HA Valkey state."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BASE_HARNESS = REPO / "benchmarks/router/valkey_weka.py"
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.router.valkey_weka_report import summarize, write_report  # noqa: E402


def load_base_harness():
    spec = importlib.util.spec_from_file_location("weka_valkey_harness", BASE_HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base harness from {BASE_HARNESS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_harness()

MODEL = "Qwen/Qwen3-0.6B"
CACHE_METRICS = base.CACHE_METRICS
TOKENIZER_ENVIRONMENT = (
    "DYN_TOKENIZER_CACHE_L2_ALLOW_INSECURE_PLAINTEXT",
    "DYN_TOKENIZER_CACHE_L2_URL",
    "DYN_TOKENIZER_CACHE_L2_SENTINEL_URLS",
    "DYN_TOKENIZER_CACHE_L2_SENTINEL_MASTER_NAME",
    "DYN_TOKENIZER_CACHE_L2_SENTINEL_QUORUM",
    "DYN_TOKENIZER_CACHE_L2_SCOPE",
    "DYN_TOKENIZER_CACHE_L2_KEY_PREFIX",
    "DYN_TOKENIZER_CACHE_L2_TTL_SECONDS",
    "DYN_TOKENIZER_CACHE_L2_TIMEOUT_MS",
    "DYN_TOKENIZER_CACHE_L2_POOL_SIZE",
    "DYN_TOKENIZER_CACHE_L2_MAX_PENDING_WRITES",
)
ROUTER_VALKEY_ENVIRONMENT = (
    "DYN_ROUTER_VALKEY_CONFIG",
    "DYN_ROUTER_VALKEY_URLS",
    "DYN_ROUTER_VALKEY_INDEX_SCOPE",
    "DYN_ROUTER_VALKEY_CONNECTION_POOL_SIZE",
    "DYN_ROUTER_VALKEY_REQUIRED_REPLICA_ACKS",
    "DYN_ROUTER_VALKEY_SENTINEL_URLS",
    "DYN_ROUTER_VALKEY_SENTINEL_MASTER_NAME",
    "DYN_ROUTER_VALKEY_SENTINEL_QUORUM",
    "DYN_ROUTER_VALKEY_ALLOW_INSECURE_PLAINTEXT",
    "DYN_ROUTER_VALKEY_ALLOW_DEGRADED_WRITES",
    "DYN_ROUTER_VALKEY_WORKER_EVENTS",
)
INTEGRITY_FAILURE_MARKERS = (
    "DYNKV_STALE_WORKER_OWNER",
    "Failed to publish event",
    "Direct Valkey KV metadata integrity fault",
    "Direct Valkey APPLY_OWNED failed",
    "permanent integrity fence",
    "fencing direct-Valkey metadata",
    "unregister could not be proved",
    "KV event publisher drain timed out",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expand_dataset(source: Path, output: Path, copies: int) -> dict[str, Any]:
    """Repeat a captured Weka slice with distinct session identities."""
    source_rows = [json.loads(line) for line in source.read_text().splitlines()]
    with output.open("w", encoding="utf-8") as destination:
        for copy in range(copies):
            for index, source_row in enumerate(source_rows):
                row = {
                    **source_row,
                    "session_id": f"weka-ab-{copy:02d}-{index:06d}",
                }
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "source_dataset": base.DATASET,
        "captured_slice": str(source),
        "captured_slice_sha256": sha256(source),
        "source_rows": len(source_rows),
        "copies": copies,
        "derived_requests": len(source_rows) * copies,
        "path": str(output),
        "sha256": sha256(output),
        "transformation": {
            "source_block_size": base.BLOCK_SIZE,
            "derived_prompt_block_size": base.AIPERF_BLOCK_SIZE,
            "max_isl": 8192,
            "max_osl": 16,
            "session_mode": "independent full-context requests",
        },
    }


def release_core_provenance() -> dict[str, Any]:
    import dynamo._core

    path = Path(dynamo._core.__file__).resolve(strict=True)
    record = base.file_provenance(path)
    record["import_path"] = str(path)
    release = REPO / "lib/bindings/python/target/release/lib_core.so"
    release_record = base.file_provenance(release)
    record["release_artifact"] = release_record
    record["rust_build_profile"] = (
        "release" if record["sha256"] == release_record["sha256"] else None
    )
    if record.get("rust_build_profile") != "release":
        raise RuntimeError(f"release dynamo._core required: {record}")
    return record


def build_schedule(runs: int) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    sample = 0
    for repetition in range(1, runs + 1):
        arms = (
            ("inprocess", "valkey_ha")
            if repetition % 2
            else (
                "valkey_ha",
                "inprocess",
            )
        )
        for arm in arms:
            sample += 1
            schedule.append({"sample": sample, "repetition": repetition, "arm": arm})
    return schedule


def arm_environment(arm: str) -> dict[str, str | None]:
    event_plane = "nats" if arm == "inprocess" else "zmq"
    return {
        "ETCD_ENDPOINTS": "http://127.0.0.1:2379",
        "NATS_SERVER": "nats://127.0.0.1:4222" if arm == "inprocess" else None,
        "DYN_EVENT_PLANE": event_plane,
        "DYN_TOKENIZER": "default",
        "DYN_TOKENIZER_CACHE": "1",
        "DYN_TOKENIZER_CACHE_BYTES": str(16 * 1024 * 1024),
        "DYN_TOKENIZER_CACHE_EXTEND": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **{name: None for name in ROUTER_VALKEY_ENVIRONMENT},
        **{name: None for name in TOKENIZER_ENVIRONMENT},
    }


def start_valkey_topology(
    args: argparse.Namespace,
    stack: contextlib.ExitStack,
    requests: list[Any],
    run_dir: Path,
    ports: list[int],
    repetition: int,
) -> dict[str, Any]:
    router_primary_port, router_replica_port = ports[:2]
    tokenizer_primary_port, tokenizer_replica_port = ports[2:4]
    sentinel_ports = ports[4:]
    router_master = f"dynamo-router-ab-{os.getpid()}-{repetition}"
    tokenizer_master = f"dynamo-tokenizer-ab-{os.getpid()}-{repetition}"

    router_primary = base.ValkeyModuleProcess(
        base.make_request(run_dir / "logs/router-primary", requests),
        port=router_primary_port,
        data_dir=run_dir / "router-primary",
        server=str(args.valkey_server),
        module=str(args.dynkv_module),
        require_replica=True,
    )
    base.apply_managed_process_affinity(router_primary, args.valkey_cpus)
    stack.enter_context(router_primary)
    router_replica = base.ValkeyModuleProcess(
        base.make_request(run_dir / "logs/router-replica", requests),
        port=router_replica_port,
        data_dir=run_dir / "router-replica",
        server=str(args.valkey_server),
        module=str(args.dynkv_module),
        replica_of=router_primary_port,
    )
    base.apply_managed_process_affinity(router_replica, args.valkey_cpus)
    stack.enter_context(router_replica)
    base.wait_for_valkey_replica(router_primary_port, router_replica_port, 30)

    tokenizer_primary = base.ValkeyModuleProcess(
        base.make_request(run_dir / "logs/tokenizer-primary", requests),
        port=tokenizer_primary_port,
        data_dir=run_dir / "tokenizer-primary",
        server=str(args.valkey_server),
        module=str(args.dynkv_module),
    )
    base.apply_managed_process_affinity(tokenizer_primary, args.valkey_cpus)
    stack.enter_context(tokenizer_primary)
    tokenizer_replica = base.ValkeyModuleProcess(
        base.make_request(run_dir / "logs/tokenizer-replica", requests),
        port=tokenizer_replica_port,
        data_dir=run_dir / "tokenizer-replica",
        server=str(args.valkey_server),
        module=str(args.dynkv_module),
        replica_of=tokenizer_primary_port,
    )
    base.apply_managed_process_affinity(tokenizer_replica, args.valkey_cpus)
    stack.enter_context(tokenizer_replica)
    base.wait_for_valkey_replica(tokenizer_primary_port, tokenizer_replica_port, 30)
    for port in (tokenizer_primary_port, tokenizer_replica_port):
        base.valkey_cli(args.valkey_cli, port, "CONFIG", "SET", "maxmemory", "512mb")
        base.valkey_cli(
            args.valkey_cli,
            port,
            "CONFIG",
            "SET",
            "maxmemory-policy",
            "allkeys-lru",
        )

    sentinels: list[Any] = []
    for index, port in enumerate(sentinel_ports, start=1):
        sentinel = base.ValkeySentinelProcess(
            base.make_request(run_dir / f"logs/sentinel-{index}", requests),
            port=port,
            data_dir=run_dir / f"sentinel-{index}",
            server=str(args.valkey_server),
            master_port=router_primary_port,
            master_name=router_master,
            quorum=2,
            down_after_ms=1000,
            failover_timeout_ms=10000,
        )
        base.apply_managed_process_affinity(sentinel, args.valkey_cpus)
        stack.enter_context(sentinel)
        sentinels.append(sentinel)
    sentinels[0].wait_for_quorum(sentinel_count=3, timeout=30)
    for sentinel in sentinels:
        base.add_sentinel_master(
            sentinel,
            master_name=tokenizer_master,
            primary_port=tokenizer_primary_port,
        )
    sentinel_votes = {
        router_master: base.wait_for_sentinel_group(
            sentinels, router_master, router_primary_port
        ),
        tokenizer_master: base.wait_for_sentinel_group(
            sentinels, tokenizer_master, tokenizer_primary_port
        ),
    }
    scope = f"weka-ab-{os.getpid()}-{repetition}-{time.time_ns()}"
    sentinel_urls = [f"valkey://127.0.0.1:{port}" for port in sentinel_ports]
    router_urls = [
        f"valkey://127.0.0.1:{router_primary_port}",
        f"valkey://127.0.0.1:{router_replica_port}",
    ]
    config = {
        "allow_insecure_plaintext": True,
        "urls": router_urls,
        "index_scope": scope,
        "connection_pool_size": 16,
        "required_replica_acks": 1,
        "sentinel": {
            "urls": sentinel_urls,
            "master_name": router_master,
            "quorum": 2,
        },
        "worker_events": True,
        "tokenizer_cache": {
            "enabled": True,
            "sentinel_master_name": tokenizer_master,
            "scope": scope,
            "ttl_seconds": 3600,
            "timeout_ms": 100,
            "connection_pool_size": 16,
            "max_pending_writes": 1024,
            "l1_bytes": 16 * 1024 * 1024,
        },
    }
    return {
        "config": config,
        "config_json": json.dumps(config, separators=(",", ":")),
        "scope": scope,
        "router_ports": [router_primary_port, router_replica_port],
        "tokenizer_ports": [tokenizer_primary_port, tokenizer_replica_port],
        "sentinel_ports": sentinel_ports,
        "sentinel_votes": sentinel_votes,
        "router_urls": router_urls,
        "sentinel_urls": sentinel_urls,
        "router_master": router_master,
    }


def worker_environment(arm: str, valkey: dict[str, Any] | None) -> dict[str, str]:
    environment = {"DYN_TOKENIZER_CACHE": "0"}
    if arm == "inprocess":
        environment["DYN_EVENT_PLANE"] = "nats"
        return environment
    assert valkey is not None
    environment.update(
        {
            "DYN_EVENT_PLANE": "zmq",
            "DYN_ROUTER_VALKEY_CONFIG": valkey["config_json"],
        }
    )
    return environment


def prune_runtime_artifacts(run_dir: Path) -> None:
    for name in (
        "router-primary",
        "router-replica",
        "tokenizer-primary",
        "tokenizer-replica",
        "sentinel-1",
        "sentinel-2",
        "sentinel-3",
    ):
        shutil.rmtree(run_dir / name, ignore_errors=True)
    (run_dir / "aiperf/inputs.json").unlink(missing_ok=True)
    shutil.rmtree(run_dir / "logs", ignore_errors=True)


def scan_logs(logs: Path) -> dict[str, int]:
    counts = {
        "error_lines": 0,
        "teardown_channel_closed_lines": 0,
        "integrity_failure_markers": 0,
    }
    for path in logs.rglob("*.txt"):
        text = path.read_text(errors="replace")
        for line in text.splitlines():
            if "ERROR" in line:
                counts["error_lines"] += 1
                if "channel closed" in line:
                    counts["teardown_channel_closed_lines"] += 1
            if any(marker in line for marker in INTEGRITY_FAILURE_MARKERS):
                counts["integrity_failure_markers"] += 1
    return counts


def compress_raw_records(run_dir: Path) -> str | None:
    source = run_dir / "aiperf/profile_export.jsonl"
    if not source.is_file():
        return None
    destination = source.with_suffix(source.suffix + ".gz")
    with (
        source.open("rb") as input_file,
        gzip.open(destination, "wb", compresslevel=6) as output_file,
    ):
        shutil.copyfileobj(input_file, output_file)
    source.unlink()
    return str(destination)


def run_arm(
    args: argparse.Namespace,
    campaign: Path,
    dataset: Path,
    sample: dict[str, Any],
) -> dict[str, Any]:
    arm = sample["arm"]
    run_dir = campaign / "runs" / f"sample-{sample['sample']:02d}-{arm}"
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    artifact_dir = run_dir / "aiperf"
    artifact_dir.mkdir()
    frontend_ports = base.allocate_ports(3, 18000)
    valkey_ports = base.allocate_ports(7, 15000) if arm == "valkey_ha" else []
    requests: list[Any] = []
    result: dict[str, Any] = {
        **sample,
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "frontend_ports": frontend_ports,
        "dataset_sha256": sha256(dataset),
        "topology": {
            "frontends": 3,
            "logical_mocker_workers": 4,
            "mocker_processes": 1,
            "request_plane": "tcp",
            "event_plane": "nats" if arm == "inprocess" else "zmq",
            "router_state": "per-frontend" if arm == "inprocess" else "HA Valkey",
            "tokenizer_state": (
                "per-frontend L1" if arm == "inprocess" else "L1 + shared HA Valkey L2"
            ),
        },
    }
    base.write_json(run_dir / "result.json", result)
    try:
        with base.environment(arm_environment(arm)), contextlib.ExitStack() as stack:
            valkey = None
            if arm == "valkey_ha":
                valkey = start_valkey_topology(
                    args,
                    stack,
                    requests,
                    run_dir,
                    valkey_ports,
                    sample["repetition"],
                )
                result["router_valkey_config"] = valkey["config"]
                result["sentinel_votes"] = valkey["sentinel_votes"]

            mocker = base.MockerProcess(
                base.make_request(run_dir / "logs/mocker", requests),
                mocker_args={
                    "speedup_ratio": 100000,
                    "block_size": base.BLOCK_SIZE,
                    "num_gpu_blocks": 131072,
                    "max_num_seqs": 16384,
                    "max_num_batched_tokens": 16384,
                    "enable_prefix_caching": True,
                },
                num_mockers=4,
                store_backend="etcd",
                request_plane="tcp",
                extra_env=worker_environment(arm, valkey),
            )
            base.apply_managed_process_affinity(mocker._process, args.mocker_cpus)
            stack.enter_context(mocker)
            workers = base.MockerProcessGroup([mocker])
            if valkey is not None:
                index_key = base.valkey_index_key(
                    workers.namespace,
                    workers.component_name,
                    valkey["scope"],
                    base.BLOCK_SIZE,
                )
                result["registered_worker_ranks"] = base.wait_for_registered_workers(
                    valkey["router_ports"][0], index_key, 4, 60
                )

            for index, port in enumerate(frontend_ports, start=1):
                frontend = base.FrontendRouterProcess(
                    base.make_request(run_dir / f"logs/frontend-{index}", requests),
                    block_size=base.BLOCK_SIZE,
                    frontend_port=port,
                    namespace=workers.namespace,
                    store_backend="etcd",
                    request_plane="tcp",
                    min_initial_workers=4,
                    router_valkey_config=(
                        valkey["config_json"] if valkey is not None else None
                    ),
                    router_replica_sync=True,
                    event_plane="zmq" if valkey is not None else "nats",
                )
                if arm == "inprocess":
                    frontend.command.extend(
                        [
                            "--router-host-cache-hit-weight",
                            "0",
                            "--router-disk-cache-hit-weight",
                            "0",
                        ]
                    )
                base.apply_managed_process_affinity(frontend, args.frontend_cpus)
                stack.enter_context(frontend)
            asyncio.run(base.wait_for_frontends(frontend_ports, workers, 90))
            time.sleep(2)

            before = base.scrape_cache_metrics(frontend_ports)
            command = base.build_aiperf_command(
                args.aiperf,
                artifact_dir,
                frontend_ports,
                dataset,
                args.requests,
                args.concurrency,
            )
            processor_index = command.index("--record-processors") + 1
            command[processor_index] = "1"
            command.extend(("--warmup-request-count", str(args.warmup_requests)))
            base.write_json(run_dir / "aiperf-command.json", command)
            outcome = base.run_aiperf(
                command,
                run_dir=run_dir,
                log_path=artifact_dir / "aiperf.log",
                timeout_seconds=args.timeout,
            )
            time.sleep(1)
            after = base.scrape_cache_metrics(frontend_ports)
            metrics = base.parse_aiperf_metrics(artifact_dir)
            records = metrics.get("records") or {}
            result.update(
                {
                    "aiperf_outcome": outcome,
                    "aiperf_metrics": metrics,
                    "tokenizer_cache_metric_delta": {
                        name: after[name] - before[name] for name in CACHE_METRICS
                    },
                }
            )
            if valkey is not None:
                router_primary, router_replica = valkey["router_ports"]
                tokenizer_primary, tokenizer_replica = valkey["tokenizer_ports"]
                result["valkey"] = {
                    "router_primary_dbsize": int(
                        base.valkey_cli(args.valkey_cli, router_primary, "DBSIZE")
                    ),
                    "router_replica_dbsize": int(
                        base.valkey_cli(args.valkey_cli, router_replica, "DBSIZE")
                    ),
                    "tokenizer_primary_dbsize": int(
                        base.valkey_cli(args.valkey_cli, tokenizer_primary, "DBSIZE")
                    ),
                    "tokenizer_replica_dbsize": int(
                        base.valkey_cli(args.valkey_cli, tokenizer_replica, "DBSIZE")
                    ),
                }
            errors = int(records.get("errored_profiling_records", 0))
            cancelled = int(records.get("cancelled_profiling_records", 0))
            completed = int(records.get("completed_profiling_records", 0))
            result["status"] = (
                "passed"
                if outcome["returncode"] == 0
                and not outcome["timed_out"]
                and completed == args.requests
                and errors == 0
                and cancelled == 0
                else "failed"
            )
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        for request in reversed(requests):
            try:
                request.close()
            except Exception as error:
                result.setdefault("cleanup_errors", []).append(str(error))
        result["log_validation"] = scan_logs(run_dir / "logs")
        if result["log_validation"]["integrity_failure_markers"]:
            result["status"] = "failed"
        base.deallocate_ports(frontend_ports)
        base.deallocate_ports(valkey_ports)
        result["raw_records_gzip"] = compress_raw_records(run_dir)
        records = result.get("aiperf_metrics", {}).get("records")
        if result["raw_records_gzip"] and isinstance(records, dict):
            records["records_path"] = result["raw_records_gzip"]
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        base.write_json(run_dir / "result.json", result)
        prune_runtime_artifacts(run_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "bench/results/valkey-config-weka-ab",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--requests", type=int, default=15232)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=(
            REPO / "bench/results/valkey-config-weka-20260707/campaign/"
            "20260707T100019Z/weka-derived.jsonl"
        ),
    )
    parser.add_argument("--frontend-cpus", default="6-15")
    parser.add_argument("--mocker-cpus", default="0-3")
    parser.add_argument("--valkey-cpus", default="4-5")
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
    parser.add_argument(
        "--dynkv-module",
        type=Path,
        default=REPO / "lib/kv-router/valkey-module/dynkv.so",
    )
    args = parser.parse_args()
    if min(args.runs, args.requests, args.concurrency) < 1:
        parser.error("runs, requests, and concurrency must be positive")
    if args.warmup_requests < 0:
        parser.error("warmup requests cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = args.output / timestamp
    campaign.mkdir(parents=True)
    dataset = campaign / "weka-derived.jsonl"
    source_rows = sum(1 for _ in args.source_dataset.open(encoding="utf-8"))
    if args.requests % source_rows:
        raise RuntimeError(
            f"request count {args.requests} must be a multiple of {source_rows}"
        )
    dataset_info = expand_dataset(
        args.source_dataset, dataset, copies=args.requests // source_rows
    )
    if dataset_info["derived_requests"] != args.requests:
        raise RuntimeError(
            f"dataset has {dataset_info['derived_requests']} requests, expected {args.requests}"
        )
    provenance = {
        "model": MODEL,
        "dynamo_core": release_core_provenance(),
        "dynkv_module": base.file_provenance(args.dynkv_module),
        "valkey_server": base.file_provenance(args.valkey_server),
        "aiperf": base.file_provenance(args.aiperf),
        "base_harness": base.file_provenance(BASE_HARNESS),
        "dataset": dataset_info,
    }
    base.write_json(campaign / "provenance.json", provenance)
    schedule = build_schedule(args.runs)
    base.write_json(campaign / "schedule.json", schedule)
    results: list[dict[str, Any]] = []
    for sample in schedule:
        print(
            f"START sample={sample['sample']} repetition={sample['repetition']} arm={sample['arm']}",
            flush=True,
        )
        result = run_arm(args, campaign, dataset, sample)
        results.append(result)
        print(
            f"END sample={sample['sample']} arm={sample['arm']} status={result['status']}",
            flush=True,
        )
    summary = summarize(results, schedule)
    base.write_json(campaign / "summary.json", summary)
    write_report(campaign / "REPORT.md", args, summary)
    print(campaign)
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
