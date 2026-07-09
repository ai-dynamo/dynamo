# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shlex
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.router.common import valkey_index_key
from tests.router.mocker_process import MockerProcess
from tests.router.router_process import (
    FrontendRouterProcess,
    ValkeyModuleProcess,
    ValkeySentinelProcess,
)
from tests.utils.port_utils import allocate_ports, deallocate_ports

from .common import (
    IMMEDIATE_LOCAL_ROUTER_FLAGS,
    ROUTER_ENV_TO_CLEAR,
    HarnessRequest,
    MockerProcessGroup,
    apply_managed_process_affinity,
    mocker_process_layout,
    retarget_mocker_process,
)
from .frontend import build_aiperf_command, discover_worker_ids, wait_for_frontends
from .loadgen import inject_primary_kill_after_profiling_starts, run_aiperf
from .metrics import aggregate_metric_value, parse_aiperf_metrics
from .provenance import environment, sha256_file, write_json
from .schedule import event_plane_for_arm
from .validation import validate_arm_result
from .valkey import (
    observe_valkey_client_pressure,
    scan_valkey_teardown_logs,
    valkey_singleton_state,
    valkey_state,
    wait_for_registered_workers,
    wait_for_valkey_replica,
    wait_for_zero_admission_reservations,
    wait_for_zero_singleton_admission_reservations,
)


def make_request(log_dir: Path, requests: list[HarnessRequest]) -> HarnessRequest:
    log_dir.mkdir(parents=True, exist_ok=True)
    request = HarnessRequest(log_dir)
    requests.append(request)
    return request


def run_arm(
    args: argparse.Namespace,
    *,
    arm: str,
    run_number: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Create, benchmark, and fully tear down exactly one topology arm."""

    run_dir = output_dir / f"run-{run_number:02d}-{arm}"
    logs_dir = run_dir / "logs"
    artifact_dir = run_dir / "aiperf"
    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir()
    artifact_dir.mkdir()

    frontend_ports = allocate_ports(args.frontend_count, 18000)
    valkey_ports: list[int] = []
    index_key: str | None = None
    valkey_primary: ValkeyModuleProcess | None = None
    valkey_replica: ValkeyModuleProcess | None = None
    sentinels: list[ValkeySentinelProcess] = []
    requests: list[HarnessRequest] = []
    process_layout = mocker_process_layout(
        args.logical_mocker_workers,
        args.mocker_processes,
        args.mocker_data_parallel_size,
    )
    authoritative_admission = arm == "valkey_ha" and args.valkey_authoritative_admission
    immediate_local_control = arm == "inprocess_immediate"
    event_plane = event_plane_for_arm(args, arm)
    started_at = datetime.now().astimezone().isoformat()
    result: dict[str, Any] = {
        "arm": arm,
        "run": run_number,
        "started_at": started_at,
        "run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
        "frontend_ports": frontend_ports,
        "request_plane": "tcp",
        "tcp_request_timeout_seconds": args.tcp_request_timeout_seconds,
        "offered_load": {
            "mode": "closed_loop",
            "concurrency": args.concurrency,
            # Omitting aiperf --request-rate means a replacement request is
            # issued immediately when one completes: effectively infinite
            # offered rate, bounded by the configured in-flight concurrency.
            "request_rate_rps": "inf",
        },
        "event_plane": event_plane,
        "control_event_plane": args.event_plane,
        "valkey_authoritative_admission": authoritative_admission,
        "valkey_gc_interval_ms": (
            args.valkey_gc_interval_ms if arm == "valkey_ha" else None
        ),
        "valkey_gc_inspection_budget": (
            args.valkey_gc_inspection_budget if arm == "valkey_ha" else None
        ),
        "comparison_profile": (
            "valkey_authoritative_immediate"
            if authoritative_admission
            else (
                "inprocess_immediate_control"
                if immediate_local_control
                else "normal_router_policy"
            )
        ),
        "topology": {
            "frontend_processes": args.frontend_count,
            "frontend_ports": frontend_ports,
            "mocker_processes": args.mocker_processes,
            "logical_mocker_workers": args.logical_mocker_workers,
            "data_parallel_ranks_per_worker": args.mocker_data_parallel_size,
            "routing_ranks": (
                args.logical_mocker_workers * args.mocker_data_parallel_size
            ),
            "mocker_process_layout": process_layout,
            "dp_rank_scope": "worker_identity",
            "worker_identity_source": "dynamo_runtime_connection_id",
            "mocker_request_ports": "runtime_assigned_and_discovery_advertised",
            "configured_cpu_affinity": {
                "frontend": args.frontend_cpus,
                "mocker": args.mocker_cpus,
                "valkey": args.valkey_cpus,
                "aiperf": args.aiperf_cpus,
            },
        },
        "status": "startup_failed",
    }
    topology_path = run_dir / "topology.json"
    result["topology_artifact"] = str(topology_path)
    write_json(topology_path, result["topology"])

    env: dict[str, str | None] = {
        "ETCD_ENDPOINTS": args.etcd_endpoints,
        "DYN_EVENT_PLANE": event_plane,
        "DYN_TCP_REQUEST_TIMEOUT": str(args.tcp_request_timeout_seconds),
        # The harness must not inherit a developer's Valkey settings into the
        # in-process control arm. The HA arm supplies its settings explicitly
        # below through the worker environment and frontend CLI flags.
        "DYN_ROUTER_VALKEY_URLS": None,
        "DYN_ROUTER_VALKEY_ALLOW_INSECURE_PLAINTEXT": None,
        "DYN_ROUTER_VALKEY_INDEX_SCOPE": None,
        "DYN_ROUTER_VALKEY_CONNECTION_POOL_SIZE": None,
        "DYN_ROUTER_VALKEY_REQUIRED_REPLICA_ACKS": None,
        "DYN_ROUTER_VALKEY_WORKER_EVENTS": None,
        "DYN_ROUTER_VALKEY_EVENT_BATCHING_TIMEOUT_MS": None,
        "DYN_ROUTER_VALKEY_AUTHORITATIVE_ADMISSION": None,
        "DYN_ROUTER_VALKEY_ADMISSION_LEASE_MS": None,
        "NATS_SERVER": args.nats_server if event_plane == "nats" else None,
        **{name: None for name in ROUTER_ENV_TO_CLEAR},
    }
    try:
        with environment(env), contextlib.ExitStack() as stack:
            valkey_urls: str | None = None
            valkey_index_scope: str | None = None
            valkey_sentinel_urls: str | None = None
            valkey_sentinel_master_name: str | None = None
            router_valkey_config: str | None = None
            worker_env = {
                "DYN_EVENT_PLANE": event_plane,
                "DYN_ROUTER_VALKEY_EVENT_BATCHING_TIMEOUT_MS": str(
                    args.valkey_event_batching_timeout_ms
                ),
            }

            if arm == "valkey_ha":
                valkey_ports = allocate_ports(
                    5 if args.kill_valkey_primary else 2, 15000
                )
                data_ports = valkey_ports[:2]
                sentinel_ports = valkey_ports[2:]
                result["valkey_ports"] = valkey_ports
                result["topology"]["valkey_data_ports"] = data_ports
                if sentinel_ports:
                    result["topology"]["valkey_sentinel_ports"] = sentinel_ports
                valkey_data_node_urls = [
                    f"valkey://127.0.0.1:{port}" for port in data_ports
                ]
                # Static HA exposes only the primary. Sentinel mode keeps both
                # data nodes as bootstrap addresses and independently proves
                # the elected endpoint through a strict Sentinel majority.
                valkey_urls = (
                    ",".join(valkey_data_node_urls)
                    if args.kill_valkey_primary
                    else valkey_data_node_urls[0]
                )
                valkey_index_scope = f"valkey-router-aiperf-run-{run_number}"
                result["valkey_urls"] = valkey_urls
                result["valkey_data_node_urls"] = valkey_data_node_urls
                result["valkey_required_replica_acks"] = 1
                result["topology"].update(
                    {
                        "valkey_client_endpoint_count": len(
                            valkey_data_node_urls
                            if args.kill_valkey_primary
                            else valkey_data_node_urls[:1]
                        ),
                        "valkey_data_node_count": 2,
                        "valkey_sentinel_count": len(sentinel_ports),
                        "valkey_required_replica_acks": 1,
                    }
                )
                result["valkey_index_scope"] = valkey_index_scope
                primary_dir = run_dir / "valkey-primary"
                replica_dir = run_dir / "valkey-replica"
                valkey_primary = ValkeyModuleProcess(
                    make_request(logs_dir / "valkey-primary", requests),
                    port=valkey_ports[0],
                    data_dir=primary_dir,
                    server=str(args.valkey_server),
                    module=str(args.dynkv_module),
                    # A two-data-node failover necessarily has no replica after
                    # promotion. The explicit degraded profile retains WAIT 1
                    # during healthy operation, then permits ack-0 only after a
                    # fresh Sentinel majority and ROLE check confirm promotion.
                    require_replica=not args.kill_valkey_primary,
                )
                apply_managed_process_affinity(valkey_primary, args.valkey_cpus)
                stack.enter_context(valkey_primary)
                valkey_replica = ValkeyModuleProcess(
                    make_request(logs_dir / "valkey-replica", requests),
                    port=valkey_ports[1],
                    data_dir=replica_dir,
                    server=str(args.valkey_server),
                    module=str(args.dynkv_module),
                    replica_of=valkey_ports[0],
                )
                apply_managed_process_affinity(valkey_replica, args.valkey_cpus)
                stack.enter_context(valkey_replica)
                wait_for_valkey_replica(
                    valkey_ports[0], valkey_ports[1], args.replica_ready_timeout
                )
                result["valkey_initial_state"] = valkey_state(
                    data_ports[0], data_ports[1]
                )
                if args.kill_valkey_primary:
                    valkey_sentinel_master_name = (
                        f"dynkv-aiperf-{os.getpid()}-{run_number}"
                    )
                    for sentinel_number, sentinel_port in enumerate(
                        sentinel_ports, start=1
                    ):
                        sentinel = ValkeySentinelProcess(
                            make_request(
                                logs_dir / f"valkey-sentinel-{sentinel_number}",
                                requests,
                            ),
                            port=sentinel_port,
                            data_dir=run_dir / f"valkey-sentinel-{sentinel_number}",
                            server=str(args.valkey_server),
                            master_port=data_ports[0],
                            master_name=valkey_sentinel_master_name,
                            quorum=2,
                            down_after_ms=args.sentinel_down_after_ms,
                            failover_timeout_ms=args.sentinel_failover_timeout_ms,
                        )
                        apply_managed_process_affinity(sentinel, args.valkey_cpus)
                        stack.enter_context(sentinel)
                        sentinels.append(sentinel)
                    sentinels[0].wait_for_quorum(
                        sentinel_count=3,
                        timeout=min(args.ready_timeout, 30),
                    )
                    initial_primary = ("127.0.0.1", data_ports[0])
                    initial_votes = [
                        sentinel.get_master_addr(timeout=2.0) for sentinel in sentinels
                    ]
                    if sum(vote == initial_primary for vote in initial_votes) < 2:
                        raise RuntimeError(
                            "Sentinel majority does not identify the initial primary: "
                            f"{initial_votes!r}"
                        )
                    valkey_sentinel_urls = ",".join(
                        f"valkey://127.0.0.1:{port}" for port in sentinel_ports
                    )
                    result.update(
                        {
                            "valkey_sentinel_urls": valkey_sentinel_urls,
                            "valkey_sentinel_master_name": valkey_sentinel_master_name,
                            "valkey_sentinel_quorum": 2,
                            "valkey_allow_degraded_writes": True,
                            "valkey_initial_sentinel_votes": initial_votes,
                        }
                    )
                config: dict[str, Any] = {
                    "allow_insecure_plaintext": True,
                    "urls": valkey_data_node_urls
                    if args.kill_valkey_primary
                    else valkey_data_node_urls[:1],
                    "index_scope": valkey_index_scope,
                    "connection_pool_size": args.valkey_connection_pool_size,
                    "required_replica_acks": 1,
                    "worker_events": True,
                    "authoritative_admission": authoritative_admission,
                    "admission_lease_ms": args.valkey_admission_lease_ms,
                    "gc_interval_ms": args.valkey_gc_interval_ms,
                    "gc_inspection_budget": args.valkey_gc_inspection_budget,
                }
                if args.kill_valkey_primary:
                    assert valkey_sentinel_urls is not None
                    assert valkey_sentinel_master_name is not None
                    config.update(
                        {
                            "sentinel": {
                                "urls": [
                                    f"valkey://127.0.0.1:{port}"
                                    for port in sentinel_ports
                                ],
                                "master_name": valkey_sentinel_master_name,
                                "quorum": 2,
                            },
                            "allow_degraded_writes": True,
                        }
                    )
                router_valkey_config = json.dumps(
                    config, separators=(",", ":"), sort_keys=True
                )
                worker_env["DYN_ROUTER_VALKEY_CONFIG"] = router_valkey_config

            mocker_shards: list[MockerProcess] = []
            shared_namespace: str | None = None
            for shard_layout in process_layout:
                process_index = shard_layout["process_index"]
                mocker_log_name = (
                    "mocker"
                    if args.mocker_processes == 1
                    else f"mocker-process-{process_index}"
                )
                shard = MockerProcess(
                    make_request(logs_dir / mocker_log_name, requests),
                    mocker_args={
                        "block_size": args.block_size,
                        "kv_bytes_per_token": args.kv_bytes_per_token,
                        "speedup_ratio": args.speedup_ratio,
                        "num_gpu_blocks": args.num_gpu_blocks,
                        "max_num_seqs": args.mocker_max_num_seqs,
                        "max_num_batched_tokens": args.mocker_max_num_batched_tokens,
                        "dp_size": args.mocker_data_parallel_size,
                    },
                    num_mockers=shard_layout["logical_worker_count"],
                    store_backend="etcd",
                    request_plane="tcp",
                    extra_env=worker_env,
                )
                if shared_namespace is None:
                    shared_namespace = shard.namespace
                retarget_mocker_process(shard, shared_namespace, args.mocker_cpus)
                stack.enter_context(shard)
                mocker_shards.append(shard)

            workers = MockerProcessGroup(mocker_shards)
            result["namespace"] = workers.namespace

            if arm == "valkey_ha":
                assert valkey_index_scope is not None
                index_key = valkey_index_key(
                    workers.namespace,
                    workers.component_name,
                    valkey_index_scope,
                    args.block_size,
                )
                expected_ranks = workers.num_workers * args.mocker_data_parallel_size
                registered_ranks = wait_for_registered_workers(
                    valkey_ports[0],
                    index_key,
                    expected_ranks,
                    args.ready_timeout,
                )
                result["valkey_registered_physical_workers"] = len(workers.shards)
                result["valkey_registered_logical_workers"] = workers.num_workers
                result["valkey_registered_mocker_processes"] = len(workers.shards)
                result["valkey_expected_registered_ranks"] = expected_ranks
                result["valkey_registered_ranks"] = registered_ranks

            for frontend_number, port in enumerate(frontend_ports, start=1):
                frontend = FrontendRouterProcess(
                    make_request(logs_dir / f"frontend-{frontend_number}", requests),
                    block_size=args.block_size,
                    frontend_port=port,
                    namespace=workers.namespace,
                    store_backend="etcd",
                    request_plane="tcp",
                    min_initial_workers=workers.num_workers,
                    router_valkey_config=router_valkey_config,
                    # The normal HA index arm holds local active-sequence
                    # replication constant with the in-process baseline. The
                    # authority arm and immediate-local control intentionally
                    # disable it because they measure immediate admission.
                    router_replica_sync=not (
                        authoritative_admission or immediate_local_control
                    ),
                    event_plane=event_plane,
                )
                if immediate_local_control:
                    # FrontendRouterProcess has a focused test-wrapper API;
                    # append only the matching local-policy flags here rather
                    # than widening that shared helper for one benchmark.
                    frontend.command.extend(IMMEDIATE_LOCAL_ROUTER_FLAGS)
                apply_managed_process_affinity(frontend, args.frontend_cpus)
                stack.enter_context(frontend)

            asyncio.run(wait_for_frontends(frontend_ports, workers, args.ready_timeout))
            worker_ids = asyncio.run(discover_worker_ids(workers, args.ready_timeout))
            result["topology"].update(
                {
                    "discovered_worker_ids": worker_ids,
                    "discovered_worker_identity_count": len(worker_ids),
                    "routing_targets": [
                        {
                            "worker_id": worker_id,
                            "dp_rank_start": 0,
                            "dp_rank_end_exclusive": args.mocker_data_parallel_size,
                        }
                        for worker_id in worker_ids
                    ],
                }
            )
            write_json(topology_path, result["topology"])
            if args.settle_seconds:
                time.sleep(args.settle_seconds)

            command = build_aiperf_command(args, frontend_ports, artifact_dir)
            command_path = run_dir / "aiperf.command.sh"
            command_path.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n"
            )
            aiperf_log = run_dir / "aiperf.log"
            fault_hook: Callable[[threading.Event], dict[str, Any]] | None = None
            if args.kill_valkey_primary:
                if valkey_primary is None or valkey_replica is None:
                    raise RuntimeError("Valkey failover processes were not initialized")
                if len(sentinels) != 3:
                    raise RuntimeError(
                        "Valkey failover requires exactly three Sentinels"
                    )

                def kill_primary(stop: threading.Event) -> dict[str, Any]:
                    return inject_primary_kill_after_profiling_starts(
                        stop,
                        records_path=artifact_dir / "profile_export.jsonl",
                        completed_records=args.fault_after_completed_records,
                        primary=valkey_primary,
                        sentinels=sentinels,
                        promoted_port=valkey_ports[1],
                        timeout_seconds=args.fault_timeout_seconds,
                    )

                fault_hook = kill_primary
            client_pressure_context = (
                observe_valkey_client_pressure(valkey_ports[:2])
                if arm == "valkey_ha"
                else contextlib.nullcontext(None)
            )
            with client_pressure_context as client_pressure:
                aiperf_result = run_aiperf(
                    command,
                    run_dir=run_dir,
                    log_path=aiperf_log,
                    timeout_seconds=args.aiperf_timeout_seconds,
                    fault_hook=fault_hook,
                    # Leave room for the final bounded Sentinel/ROLE poll which
                    # may begin just before the hook's own deadline.
                    fault_join_timeout_seconds=args.fault_timeout_seconds + 5.0,
                )
            if client_pressure is not None:
                result["valkey_client_pressure"] = dict(client_pressure)
            inputs_path = artifact_dir / "inputs.json"
            input_sha256 = sha256_file(inputs_path) if inputs_path.is_file() else None
            if arm == "valkey_ha":
                assert index_key is not None
                if args.settle_seconds:
                    time.sleep(args.settle_seconds)
                if args.kill_valkey_primary:
                    fault = aiperf_result.get("fault_injection")
                    if isinstance(fault, Mapping) and fault.get("status") == "promoted":
                        result["valkey_final_admission_stats"] = {
                            "promoted": wait_for_zero_singleton_admission_reservations(
                                valkey_ports[1],
                                index_key,
                                min(args.replica_ready_timeout, 30.0),
                            )
                        }
                        result["valkey_final_state"] = valkey_singleton_state(
                            valkey_ports[1], index_key
                        )
                else:
                    wait_for_valkey_replica(
                        valkey_ports[0], valkey_ports[1], args.replica_ready_timeout
                    )
                    result["valkey_final_admission_stats"] = (
                        wait_for_zero_admission_reservations(
                            valkey_ports[0],
                            valkey_ports[1],
                            index_key,
                            min(args.replica_ready_timeout, 30.0),
                        )
                    )
                    result["valkey_final_state"] = valkey_state(
                        valkey_ports[0], valkey_ports[1]
                    )
            aiperf_metrics = parse_aiperf_metrics(artifact_dir)
            raw_records = (
                aiperf_metrics.get("records")
                if isinstance(aiperf_metrics, dict)
                else None
            )
            completed_records = (
                raw_records.get("completed_profiling_records", 0)
                if isinstance(raw_records, dict)
                else 0
            )
            summary_metrics = (
                aiperf_metrics.get("summary")
                if isinstance(aiperf_metrics, dict)
                else None
            )
            summary_error_count = (
                aggregate_metric_value(summary_metrics.get("error_request_count"))
                if isinstance(summary_metrics, dict)
                else None
            )
            raw_error_count = (
                raw_records.get("errored_profiling_records", 0)
                if isinstance(raw_records, dict)
                else 0
            )
            observed_error_count = max(
                float(summary_error_count or 0), float(raw_error_count)
            )
            if aiperf_result["timed_out"]:
                status = "aiperf_timeout"
            elif aiperf_result["returncode"] != 0:
                status = "aiperf_failed"
            elif observed_error_count > 0:
                status = "aiperf_errors"
            elif completed_records != args.requests:
                status = "aiperf_incomplete"
            else:
                status = "ok"
            result.update(
                {
                    "aiperf_command": command,
                    "aiperf_command_path": str(command_path),
                    "aiperf_log": str(aiperf_log),
                    "aiperf": aiperf_result,
                    "aiperf_expected_profiling_requests": args.requests,
                    "aiperf_summary_error_request_count": summary_error_count,
                    "aiperf_jsonl_error_request_count": raw_error_count,
                    "aiperf_observed_error_request_count": observed_error_count,
                    "aiperf_metrics": aiperf_metrics,
                    "aiperf_input_path": str(inputs_path),
                    "aiperf_input_sha256": input_sha256,
                    "status": status,
                }
            )
            validation_errors = validate_arm_result(result, args)
            result["validation_errors"] = validation_errors
            if validation_errors:
                result["status"] = "validation_failed"
    except Exception as error:  # Keep other interleaved arms inspectable.
        result.update(
            {
                "status": "startup_failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        (run_dir / "error.txt").write_text(result["traceback"])
    finally:
        for request in reversed(requests):
            try:
                request.close()
            except Exception as error:  # Do not hide the primary benchmark result.
                result.setdefault("cleanup_errors", []).append(str(error))
        if result.get("cleanup_errors") and result.get("status") == "ok":
            result["status"] = "cleanup_failed"
            result.setdefault("validation_errors", []).append(
                "managed process cleanup reported errors"
            )
        if arm == "valkey_ha":
            teardown_logs = scan_valkey_teardown_logs(logs_dir)
            result["valkey_teardown_log_validation"] = teardown_logs
            expected_log_files = args.frontend_count + args.mocker_processes + 2
            if teardown_logs["files_scanned"] < expected_log_files:
                result["status"] = "validation_failed"
                result.setdefault("validation_errors", []).append(
                    "Valkey topology teardown log scan found only "
                    f"{teardown_logs['files_scanned']} of at least "
                    f"{expected_log_files} expected process logs"
                )
            if teardown_logs["failure_count"]:
                result["status"] = "validation_failed"
                result.setdefault("validation_errors", []).append(
                    "Valkey teardown logs contain "
                    f"{teardown_logs['failure_count']} owner/publisher failure marker(s)"
                )
        deallocate_ports(frontend_ports)
        deallocate_ports(valkey_ports)
        result["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(topology_path, result["topology"])
        write_json(run_dir / "result.json", result)

    return result
