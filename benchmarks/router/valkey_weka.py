#!/usr/bin/env python3
"""Exercise one JSON Valkey config with two HA master sets and a Weka-derived trace."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
from benchmarks.router.valkey_aiperf.metrics import parse_aiperf_metrics  # noqa: E402
from benchmarks.router.valkey_aiperf.provenance import (  # noqa: E402
    environment,
    file_provenance,
)
from benchmarks.router.valkey_aiperf.valkey import (  # noqa: E402
    wait_for_registered_workers,
    wait_for_valkey_replica,
)
from tests.router.common import valkey_index_key  # noqa: E402
from tests.router.mocker_process import MockerProcess  # noqa: E402
from tests.router.router_process import (  # noqa: E402
    FrontendRouterProcess,
    ValkeyModuleProcess,
    ValkeySentinelProcess,
)
from tests.utils.port_utils import allocate_ports, deallocate_ports  # noqa: E402


DATASET = "semianalysisai/cc-traces-weka-062126-256k"
MODEL = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 64
AIPERF_BLOCK_SIZE = 512
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def derive_weka_slice(
    output: Path,
    *,
    traces: int,
    requests_per_trace: int,
    replicas: int,
    max_isl: int,
    max_osl: int,
) -> dict[str, Any]:
    """Preserve Weka hash topology while bounding request sizes for a CPU test."""

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train", streaming=True)
    selected: list[dict[str, Any]] = []
    source: list[dict[str, Any]] = []
    for trace_index, trace in zip(range(traces), dataset):
        converted: list[dict[str, Any]] = []
        skipped = 0
        for request_index, request in enumerate(trace["requests"]):
            hash_ids = request.get("hash_ids")
            input_length = request.get("in")
            if (
                request.get("type") != "s"
                or not isinstance(hash_ids, list)
                or not hash_ids
                or not isinstance(input_length, int)
                or input_length < 1
            ):
                skipped += 1
                continue
            bounded_isl = min(input_length, max_isl)
            block_count = min(
                math.ceil(len(hash_ids) * BLOCK_SIZE / AIPERF_BLOCK_SIZE),
                max(1, math.ceil(bounded_isl / AIPERF_BLOCK_SIZE)),
            )
            bounded_isl = min(bounded_isl, block_count * AIPERF_BLOCK_SIZE)
            namespaced_hashes = []
            source_hashes_per_derived_block = AIPERF_BLOCK_SIZE // BLOCK_SIZE
            for block_index in range(block_count):
                first = block_index * source_hashes_per_derived_block
                chunk = hash_ids[first : first + source_hashes_per_derived_block]
                digest = hashlib.blake2b(
                    f"{trace_index}:{','.join(map(str, chunk))}".encode(),
                    digest_size=8,
                ).digest()
                namespaced_hashes.append(
                    int.from_bytes(digest, "little") & ((1 << 63) - 1)
                )
            converted.append(
                {
                    "delay": 0,
                    "input_length": bounded_isl,
                    "output_length": min(max(1, int(request.get("out") or 1)), max_osl),
                    "hash_ids": namespaced_hashes,
                    "extra": {
                        "ignore_eos": True,
                        "min_tokens": min(
                            max(1, int(request.get("out") or 1)), max_osl
                        ),
                        "temperature": 0.0,
                    },
                    "source_request_index": request_index,
                }
            )
            if len(converted) >= requests_per_trace:
                break
        if not converted:
            continue
        source.append(
            {
                "trace_index": trace_index,
                "id": trace["id"],
                "models": trace["models"],
                "source_requests": len(trace["requests"]),
                "selected_requests": len(converted),
                "skipped_before_limit": skipped,
            }
        )
        for replica in range(replicas):
            for request in converted:
                row = {
                    key: value
                    for key, value in request.items()
                    if key != "source_request_index"
                }
                row["session_id"] = (
                    f"weka-{trace_index:03d}-request-"
                    f"{request['source_request_index']:04d}-replica-{replica:02d}"
                )
                selected.append(row)

    if not selected:
        raise RuntimeError("the Weka source slice produced no model requests")
    with output.open("w", encoding="utf-8") as destination:
        for row in selected:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "source_dataset": DATASET,
        "source": source,
        "transformation": {
            "trace_count": traces,
            "requests_per_trace_cap": requests_per_trace,
            "session_replication_factor": replicas,
            "max_isl": max_isl,
            "max_osl": max_osl,
            "source_block_size": BLOCK_SIZE,
            "derived_prompt_block_size": AIPERF_BLOCK_SIZE,
            "hash_namespace": (
                "BLAKE2b-64(trace index + each adjacent group of eight local hashes)"
            ),
            "timing": "all inter-turn delays set to zero",
        },
        "derived_requests": len(selected),
        "derived_sessions": len(selected),
        "path": str(output),
        "sha256": sha256(output),
    }


def reuse_weka_slice(source: Path, output: Path) -> dict[str, Any]:
    """Make each source request an independent full-context replay."""

    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as source_file:
        for index, line in enumerate(source_file):
            row = json.loads(line)
            row["session_id"] = f"weka-independent-request-{index:06d}"
            rows.append(row)
    with output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "source_dataset": DATASET,
        "source_derived_slice": str(source),
        "source_derived_slice_sha256": sha256(source),
        "transformation": {
            "source_block_size": BLOCK_SIZE,
            "derived_prompt_block_size": AIPERF_BLOCK_SIZE,
            "max_isl": max(row["input_length"] for row in rows),
            "max_osl": max(row["output_length"] for row in rows),
            "session_mode": "each row is an independent full-context request",
        },
        "derived_requests": len(rows),
        "derived_sessions": len(rows),
        "path": str(output),
        "sha256": sha256(output),
    }


def valkey_cli(cli: Path, port: int, *parts: str) -> str:
    result = subprocess.run(
        [str(cli), "--raw", "-h", "127.0.0.1", "-p", str(port), *parts],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def add_sentinel_master(
    sentinel: ValkeySentinelProcess,
    *,
    master_name: str,
    primary_port: int,
) -> None:
    commands = (
        ("SENTINEL", "MONITOR", master_name, "127.0.0.1", str(primary_port), "2"),
        ("SENTINEL", "SET", master_name, "down-after-milliseconds", "1000"),
        ("SENTINEL", "SET", master_name, "failover-timeout", "10000"),
        ("SENTINEL", "SET", master_name, "parallel-syncs", "1"),
    )
    for command in commands:
        result = sentinel.run_sentinel_command(*command)
        if result.returncode != 0 or not result.stdout.startswith("OK"):
            raise RuntimeError(
                f"Sentinel command {command!r} failed: {result.stdout}{result.stderr}"
            )


def wait_for_sentinel_group(
    sentinels: list[ValkeySentinelProcess],
    master_name: str,
    primary_port: int,
    timeout: float = 30,
) -> list[tuple[str, int]]:
    deadline = time.monotonic() + timeout
    last_votes: list[tuple[str, int]] = []
    expected = ("127.0.0.1", primary_port)
    while time.monotonic() < deadline:
        votes: list[tuple[str, int]] = []
        quorum = False
        for sentinel in sentinels:
            address = sentinel.run_sentinel_command(
                "SENTINEL", "GET-MASTER-ADDR-BY-NAME", master_name
            )
            fields = address.stdout.splitlines()
            if address.returncode == 0 and len(fields) == 2:
                votes.append((fields[0], int(fields[1])))
            check = sentinel.run_sentinel_command("SENTINEL", "CKQUORUM", master_name)
            quorum = quorum or (check.returncode == 0 and check.stdout.startswith("OK"))
        last_votes = votes
        if quorum and votes.count(expected) >= 2:
            return votes
        time.sleep(0.1)
    raise TimeoutError(
        f"Sentinels did not converge on {master_name}={expected}: {last_votes}"
    )


def scrape_cache_metrics(frontend_ports: list[int]) -> dict[str, float]:
    totals = {name: 0.0 for name in CACHE_METRICS}
    for port in frontend_ports:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=5
        ) as response:
            body = response.read().decode("utf-8")
        for line in body.splitlines():
            if not line or line.startswith("#") or "{" in line:
                continue
            name, _, raw_value = line.partition(" ")
            if name in totals:
                totals[name] += float(raw_value)
    return totals


def build_aiperf_command(
    aiperf: Path,
    artifact_dir: Path,
    frontend_ports: list[int],
    dataset: Path,
    request_count: int,
    concurrency: int,
) -> list[str]:
    command = [
        "taskset",
        "--cpu-list",
        "16-23",
        str(aiperf),
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
    for port in frontend_ports:
        command.extend(("--url", f"http://127.0.0.1:{port}"))
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
            str(concurrency),
            "--request-rate",
            "inf",
            "--request-count",
            str(request_count),
            "--request-timeout-seconds",
            "120",
            "--random-seed",
            "100",
            "--workers-max",
            str(min(concurrency, 64)),
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


def run(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output / timestamp
    logs = root / "logs"
    artifacts = root / "aiperf"
    root.mkdir(parents=True)
    logs.mkdir()
    artifacts.mkdir()
    dataset_path = root / "weka-derived.jsonl"
    if args.reuse_dataset is None:
        dataset = derive_weka_slice(
            dataset_path,
            traces=args.traces,
            requests_per_trace=args.requests_per_trace,
            replicas=args.session_replicas,
            max_isl=args.max_isl,
            max_osl=args.max_osl,
        )
    else:
        dataset = reuse_weka_slice(args.reuse_dataset, dataset_path)
    write_json(root / "dataset-provenance.json", dataset)

    ports = allocate_ports(10, 15000)
    frontend_ports = ports[:3]
    router_primary_port, router_replica_port = ports[3:5]
    tokenizer_primary_port, tokenizer_replica_port = ports[5:7]
    sentinel_ports = ports[7:10]
    requests: list[HarnessRequest] = []
    router_master = f"dynamo-router-{os.getpid()}"
    tokenizer_master = f"dynamo-tokenizer-{os.getpid()}"
    sentinel_urls = [f"valkey://127.0.0.1:{port}" for port in sentinel_ports]
    router_urls = [
        f"valkey://127.0.0.1:{router_primary_port}",
        f"valkey://127.0.0.1:{router_replica_port}",
    ]
    scope = f"weka-json-{os.getpid()}"
    router_valkey_config = {
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
    router_valkey_json = json.dumps(router_valkey_config, separators=(",", ":"))
    write_json(root / "router-valkey-config.json", router_valkey_config)

    result: dict[str, Any] = {
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "topology": {
            "frontends": 3,
            "mockers": 4,
            "router_valkey_nodes": 2,
            "tokenizer_valkey_nodes": 2,
            "shared_sentinels": 3,
            "frontend_ports": frontend_ports,
            "router_valkey_ports": [router_primary_port, router_replica_port],
            "tokenizer_valkey_ports": [
                tokenizer_primary_port,
                tokenizer_replica_port,
            ],
            "sentinel_ports": sentinel_ports,
        },
        "dataset": dataset,
        "router_valkey_config": router_valkey_config,
        "provenance": {
            "aiperf": file_provenance(args.aiperf),
            "valkey_server": file_provenance(args.valkey_server),
            "dynkv_module": file_provenance(args.dynkv_module),
        },
    }
    write_json(root / "result.json", result)
    env = {
        "ETCD_ENDPOINTS": "http://127.0.0.1:2379",
        "NATS_SERVER": None,
        "DYN_ROUTER_VALKEY_CONFIG": None,
        "DYN_ROUTER_VALKEY_URLS": None,
        "DYN_ROUTER_VALKEY_ALLOW_INSECURE_PLAINTEXT": None,
        "DYN_ROUTER_VALKEY_SENTINEL_URLS": None,
        "DYN_ROUTER_VALKEY_SENTINEL_MASTER_NAME": None,
        "DYN_TOKENIZER_CACHE_L2_URL": None,
        "DYN_TOKENIZER_CACHE_L2_ALLOW_INSECURE_PLAINTEXT": None,
        "DYN_TOKENIZER_CACHE_L2_SENTINEL_URLS": None,
        "DYN_EVENT_PLANE": "zmq",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    try:
        with environment(env), contextlib.ExitStack() as stack:
            router_primary = ValkeyModuleProcess(
                make_request(logs / "router-primary", requests),
                port=router_primary_port,
                data_dir=root / "router-primary",
                server=str(args.valkey_server),
                module=str(args.dynkv_module),
                require_replica=True,
            )
            apply_managed_process_affinity(router_primary, "4-5")
            stack.enter_context(router_primary)
            router_replica = ValkeyModuleProcess(
                make_request(logs / "router-replica", requests),
                port=router_replica_port,
                data_dir=root / "router-replica",
                server=str(args.valkey_server),
                module=str(args.dynkv_module),
                replica_of=router_primary_port,
            )
            apply_managed_process_affinity(router_replica, "4-5")
            stack.enter_context(router_replica)
            wait_for_valkey_replica(router_primary_port, router_replica_port, 30)

            tokenizer_primary = ValkeyModuleProcess(
                make_request(logs / "tokenizer-primary", requests),
                port=tokenizer_primary_port,
                data_dir=root / "tokenizer-primary",
                server=str(args.valkey_server),
                module=str(args.dynkv_module),
            )
            apply_managed_process_affinity(tokenizer_primary, "4-5")
            stack.enter_context(tokenizer_primary)
            tokenizer_replica = ValkeyModuleProcess(
                make_request(logs / "tokenizer-replica", requests),
                port=tokenizer_replica_port,
                data_dir=root / "tokenizer-replica",
                server=str(args.valkey_server),
                module=str(args.dynkv_module),
                replica_of=tokenizer_primary_port,
            )
            apply_managed_process_affinity(tokenizer_replica, "4-5")
            stack.enter_context(tokenizer_replica)
            wait_for_valkey_replica(tokenizer_primary_port, tokenizer_replica_port, 30)
            valkey_cli(
                args.valkey_cli,
                tokenizer_primary_port,
                "CONFIG",
                "SET",
                "maxmemory",
                "512mb",
            )
            valkey_cli(
                args.valkey_cli,
                tokenizer_primary_port,
                "CONFIG",
                "SET",
                "maxmemory-policy",
                "allkeys-lru",
            )

            sentinels: list[ValkeySentinelProcess] = []
            for index, port in enumerate(sentinel_ports, start=1):
                sentinel = ValkeySentinelProcess(
                    make_request(logs / f"sentinel-{index}", requests),
                    port=port,
                    data_dir=root / f"sentinel-{index}",
                    server=str(args.valkey_server),
                    master_port=router_primary_port,
                    master_name=router_master,
                    quorum=2,
                    down_after_ms=1000,
                    failover_timeout_ms=10000,
                )
                apply_managed_process_affinity(sentinel, "4-5")
                stack.enter_context(sentinel)
                sentinels.append(sentinel)
            sentinels[0].wait_for_quorum(sentinel_count=3, timeout=30)
            for sentinel in sentinels:
                add_sentinel_master(
                    sentinel,
                    master_name=tokenizer_master,
                    primary_port=tokenizer_primary_port,
                )
            result["sentinel_votes"] = {
                router_master: wait_for_sentinel_group(
                    sentinels, router_master, router_primary_port
                ),
                tokenizer_master: wait_for_sentinel_group(
                    sentinels, tokenizer_master, tokenizer_primary_port
                ),
            }

            worker_env = {
                "DYN_EVENT_PLANE": "zmq",
                "DYN_ROUTER_VALKEY_CONFIG": router_valkey_json,
                "DYN_TOKENIZER_CACHE": "0",
            }
            mocker = MockerProcess(
                make_request(logs / "mocker", requests),
                mocker_args={
                    "speedup_ratio": 100000,
                    "block_size": BLOCK_SIZE,
                    "num_gpu_blocks": 131072,
                    "max_num_seqs": 16384,
                    "max_num_batched_tokens": 16384,
                    "enable_prefix_caching": True,
                },
                num_mockers=4,
                store_backend="etcd",
                request_plane="tcp",
                extra_env=worker_env,
            )
            apply_managed_process_affinity(mocker._process, "0-3")
            stack.enter_context(mocker)
            workers = MockerProcessGroup([mocker])
            index_key = valkey_index_key(
                workers.namespace, workers.component_name, scope, BLOCK_SIZE
            )
            result["index_key"] = index_key
            result["registered_worker_ranks"] = wait_for_registered_workers(
                router_primary_port, index_key, 4, 60
            )

            for index, port in enumerate(frontend_ports, start=1):
                frontend = FrontendRouterProcess(
                    make_request(logs / f"frontend-{index}", requests),
                    block_size=BLOCK_SIZE,
                    frontend_port=port,
                    namespace=workers.namespace,
                    store_backend="etcd",
                    request_plane="tcp",
                    min_initial_workers=4,
                    router_valkey_config=router_valkey_json,
                    router_replica_sync=True,
                    event_plane="zmq",
                )
                apply_managed_process_affinity(frontend, "6-15")
                stack.enter_context(frontend)
            asyncio.run(wait_for_frontends(frontend_ports, workers, 90))

            before = scrape_cache_metrics(frontend_ports)
            command = build_aiperf_command(
                args.aiperf,
                artifacts,
                frontend_ports,
                dataset_path,
                dataset["derived_requests"],
                args.concurrency,
            )
            write_json(root / "aiperf-command.json", command)
            outcome = run_aiperf(
                command,
                run_dir=root,
                log_path=artifacts / "aiperf.log",
                timeout_seconds=args.timeout,
            )
            time.sleep(1)
            after = scrape_cache_metrics(frontend_ports)
            result["aiperf"] = {
                "outcome": outcome,
                "metrics": parse_aiperf_metrics(artifacts),
            }
            result["tokenizer_cache_metric_delta"] = {
                name: after[name] - before[name] for name in before
            }
            result["valkey"] = {
                "router_primary_dbsize": int(
                    valkey_cli(args.valkey_cli, router_primary_port, "DBSIZE")
                ),
                "router_replica_dbsize": int(
                    valkey_cli(args.valkey_cli, router_replica_port, "DBSIZE")
                ),
                "tokenizer_primary_dbsize": int(
                    valkey_cli(args.valkey_cli, tokenizer_primary_port, "DBSIZE")
                ),
                "tokenizer_replica_dbsize": int(
                    valkey_cli(args.valkey_cli, tokenizer_replica_port, "DBSIZE")
                ),
                "tokenizer_eviction_policy": valkey_cli(
                    args.valkey_cli,
                    tokenizer_primary_port,
                    "CONFIG",
                    "GET",
                    "maxmemory-policy",
                ).splitlines()[-1],
            }
            result["status"] = (
                "passed"
                if outcome["returncode"] == 0
                and not outcome["timed_out"]
                and result["registered_worker_ranks"] == 4
                else "failed"
            )
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "result.json", result)
        for request in reversed(requests):
            request.close()
        deallocate_ports(ports)
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "bench/results/valkey-config-weka",
    )
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
    parser.add_argument("--aiperf", type=Path, default=REPO / "dynamo/bin/aiperf")
    parser.add_argument("--traces", type=int, default=8)
    parser.add_argument("--requests-per-trace", type=int, default=16)
    parser.add_argument("--session-replicas", type=int, default=4)
    parser.add_argument("--max-isl", type=int, default=8192)
    parser.add_argument("--max-osl", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reuse-dataset", type=Path)
    return parser.parse_args()


def main() -> int:
    root = run(parse_args())
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
