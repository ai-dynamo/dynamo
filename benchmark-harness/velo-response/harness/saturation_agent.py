#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tyche AgentX saturation selector, profiler, and fixed-concurrency NUMA ablations.

Rank 0 owns the single-NUMA frontend and experiment order. One or more middle
ranks own mocker processes, with exactly one also owning etcd. The final rank
owns one unbound AIPerf controller. Only job-scoped files on Lustre are used for
coordination.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import node_agent as base
from analyze import parse_aiperf
from analyze_profile import load_jsonl, measured_samples, process_cpu


class CampaignError(RuntimeError):
    pass


KV_ZMQ_ACTIVE_SOURCES_METRIC = (
    "dynamo_component_router_kv_zmq_ingress_sources"
)
KV_CACHE_EVENTS_APPLIED_METRIC = "dynamo_component_kv_cache_events_applied"
KV_ZMQ_SOCKET_GROUPS_METRIC = (
    "dynamo_component_router_kv_zmq_ingress_socket_groups"
)
KV_ZMQ_CONNECTED_ENDPOINTS_METRIC = (
    "dynamo_component_router_kv_zmq_ingress_connected_endpoints"
)


def parse_kv_zmq_active_sources(payload: str) -> int | None:
    """Return the single active-source gauge from a Prometheus scrape."""
    values: list[int] = []
    for line in payload.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        sample = fields[0]
        if not sample.startswith(KV_ZMQ_ACTIVE_SOURCES_METRIC):
            continue
        labels = sample[len(KV_ZMQ_ACTIVE_SOURCES_METRIC) :]
        if re.search(r'(?:\{|,)state="active"(?:,|\})', labels) is None:
            continue
        try:
            value = float(fields[1])
        except ValueError as error:
            raise CampaignError(
                f"invalid {KV_ZMQ_ACTIVE_SOURCES_METRIC} value: {fields[1]}"
            ) from error
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise CampaignError(
                f"invalid {KV_ZMQ_ACTIVE_SOURCES_METRIC} gauge: {value}"
            )
        values.append(int(value))
    if not values:
        return None
    if len(values) != 1:
        raise CampaignError(
            f"expected one {KV_ZMQ_ACTIVE_SOURCES_METRIC} active sample, "
            f"found {len(values)}"
        )
    return values[0]


def parse_kv_cache_events_applied(payload: str) -> float | None:
    """Sum successful KV-event applications in one Prometheus scrape."""
    values: list[float] = []
    for line in payload.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        sample = fields[0]
        if not sample.startswith(f"{KV_CACHE_EVENTS_APPLIED_METRIC}{{"):
            continue
        labels = sample[len(KV_CACHE_EVENTS_APPLIED_METRIC) :]
        if re.search(r'(?:\{|,)status="ok"(?:,|\})', labels) is None:
            continue
        try:
            value = float(fields[1])
        except ValueError as error:
            raise CampaignError(
                f"invalid {KV_CACHE_EVENTS_APPLIED_METRIC} value: {fields[1]}"
            ) from error
        if not math.isfinite(value) or value < 0:
            raise CampaignError(
                f"invalid {KV_CACHE_EVENTS_APPLIED_METRIC} counter: {value}"
            )
        values.append(value)
    return math.fsum(values) if values else None


def parse_stream_gauge(payload: str, metric: str, stream: str) -> int | None:
    values: list[int] = []
    for line in payload.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0].split("{", 1)[0] != metric:
            continue
        labels = fields[0][len(metric) :]
        if re.search(rf'(?:\{{|,)stream="{re.escape(stream)}"(?:,|\}})', labels) is None:
            continue
        value = float(fields[1])
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise CampaignError(f"invalid {metric} gauge: {value}")
        values.append(int(value))
    if not values:
        return None
    if len(values) != 1:
        raise CampaignError(f"expected one {metric} sample, found {len(values)}")
    return values[0]


def load_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    parent = raw.pop("extends", None)
    if parent is not None:
        parent_path = (path.parent / str(parent)).resolve()
        config = merge_config(json.loads(parent_path.read_text()), raw)
    else:
        config = raw
    validate_config(config)
    return config


def merge_config(
    original: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(original))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config):
    assert config['schema_version'] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", config["pins"]["dynamo"])
    small_smoke = config['campaign'].get('smoke_only', False)
    assert config['runtime']['num_mockers'] == (12 if small_smoke else 2048)
    assert config['runtime']['mocker_speedup_ratio'] == 10
    assert config['runtime']['tokenizer'] in ('default', 'fastokens')
    assert config['runtime']['aiperf_workers_max'] == 128
    assert config['runtime']['router_mode'] == 'kv'
    assert config['runtime']['router_replica_sync']
    assert config['runtime']['router_zmq_endpoints_per_sub'] == 1
    assert config['workload']['kind'] == 'agentx'
    assert config['workload']['block_size'] == 16
    assert config['campaign']['fixed_concurrency'] == (8192 if config['campaign'].get('preflight_saturation') else 6144)
    assert config['topology']['frontend'] == {'cpus':'0-71','memory':'bind:0'}
    assert config['topology']['frontend_control'] == {'cpus':'72-143','memory':'bind:1'}
    nodes=normalized_mocker_nodes(config)
    assert [sum(p['workers'] for p in n['processes']) for n in nodes] == ([4,4,4] if small_smoke else [684,684,680])
    assert all([p['cpus'] for p in n['processes']] == ['0-35','36-71','72-107','108-143'] for n in nodes)
    assert config['saturation']['settle_seconds'] == 30
    assert config['saturation']['measurement_seconds'] == 120
    assert config['workload']['grace_period_seconds'] == 90
    assert not any(k in config['runtime'] for k in ('bpe_threads','rayon_threads'))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def interface_ipv4(interface: str) -> str:
    try:
        addresses = json.loads(
            subprocess.check_output(
                ["ip", "-j", "address", "show", "dev", interface], text=True
            )
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise CampaignError(
            f"failed to inspect interface {interface}: {error}"
        ) from error
    candidates = [
        item["local"]
        for address in addresses
        for item in address.get("addr_info", [])
        if item.get("family") == "inet" and item.get("scope") == "global"
    ]
    if len(candidates) != 1:
        raise CampaignError(
            f"expected one global IPv4 address on {interface}, found {candidates}"
        )
    return str(candidates[0])


def discover_networks(
    config: Mapping[str, Any], peer: str
) -> dict[str, dict[str, Any]]:
    configured = config.get("network", {}).get("numa_interfaces")
    if configured is None:
        interface, address = base._route_to(peer, config["network"]["interface"])
        return {
            "primary": {
                "interface": interface,
                "address": address,
                "numa_node": None,
            }
        }
    networks: dict[str, dict[str, Any]] = {}
    for name, raw_spec in configured.items():
        spec = dict(raw_spec)
        interface = str(spec["interface"])
        networks[str(name)] = {
            "interface": interface,
            "address": interface_ipv4(interface),
            "numa_node": int(spec["numa_node"]),
        }
    return networks


def primary_network(
    config: Mapping[str, Any], networks: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    name = str(config.get("network", {}).get("primary", "primary"))
    if name not in networks:
        if len(networks) == 1:
            name = next(iter(networks))
        else:
            raise CampaignError(f"primary network {name!r} is not configured")
    return dict(networks[name])


def select_network(
    networks: Mapping[str, Mapping[str, Any]], name: str | None
) -> dict[str, Any]:
    if name is None and len(networks) == 1:
        return dict(next(iter(networks.values())))
    if name not in networks:
        raise CampaignError(f"network {name!r} is not available: {sorted(networks)}")
    return dict(networks[str(name)])


def percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[
        min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    ]


def atomic_json(path: Path, value: Any) -> None:
    base.atomic_json(path, value)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def wait_for(path: Path, timeout: float, shutdown: threading.Event) -> None:
    base.wait_for_path(path, timeout, shutdown)


class Channel:
    def __init__(self, state_dir: Path, name: str, shutdown: threading.Event) -> None:
        self.root = state_dir / "channels" / name
        self.commands = self.root / "commands"
        self.acks = self.root / "acks"
        self.shutdown = shutdown
        self.sequence = 0

    def send(
        self, action: str, payload: Mapping[str, Any], timeout: float
    ) -> dict[str, Any]:
        self.sequence += 1
        name = f"{self.sequence:04d}"
        atomic_json(
            self.commands / f"{name}.json",
            {
                "sequence": self.sequence,
                "action": action,
                "payload": dict(payload),
                "issued_at": now(),
            },
        )
        ack_path = self.acks / f"{name}.json"
        wait_for(ack_path, timeout, self.shutdown)
        ack = json.loads(ack_path.read_text())
        if not ack.get("ok"):
            raise CampaignError(
                f"{self.root.name} action {action} failed: {ack.get('error')}"
            )
        return dict(ack.get("result", {}))

    def receive(self, sequence: int) -> dict[str, Any]:
        path = self.commands / f"{sequence:04d}.json"
        wait_for(path, 24 * 60 * 60, self.shutdown)
        return json.loads(path.read_text())

    def acknowledge(
        self,
        sequence: int,
        action: str,
        *,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        atomic_json(
            self.acks / f"{sequence:04d}.json",
            {
                "sequence": sequence,
                "action": action,
                "ok": error is None,
                "result": dict(result or {}),
                "error": error,
                "acknowledged_at": now(),
            },
        )


class MockerFanoutChannel:
    """Present multiple mocker ranks as the coordinator's single lifecycle peer."""

    def __init__(self, channels: Sequence[tuple[Mapping[str, Any], Channel]]) -> None:
        self.channels = list(channels)
        owners = [item for item in self.channels if item[0].get("manage_etcd")]
        if len(owners) != 1:
            raise CampaignError("mocker fanout requires exactly one etcd owner")
        self.etcd_owner = owners[0]

    def send(
        self, action: str, payload: Mapping[str, Any], timeout: float
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        ordered = sorted(
            self.channels, key=lambda item: not bool(item[0].get("manage_etcd"))
        )
        for spec, channel in ordered:
            result = channel.send(action, payload, timeout)
            results.append(
                {"rank": int(spec["rank"]), "name": spec["name"], "result": result}
            )
        if action == "prepare":
            _, owner = self.etcd_owner
            verified = owner.send("verify_mockers", {}, timeout)
            return {
                "nodes": results,
                "registered_mockers": verified["registered_mockers"],
            }
        if action == "finish_monitor":
            node_results = [item["result"] for item in results]
            return {
                "nodes": results,
                "accepted": all(bool(item["accepted"]) for item in node_results),
                "headroom_accepted": all(
                    bool(item["headroom_accepted"]) for item in node_results
                ),
                "process_cpu": {
                    name: value
                    for item in node_results
                    for name, value in item["process_cpu"].items()
                },
                "memory": {
                    name: value
                    for item in node_results
                    for name, value in item["memory"].items()
                },
            }
        return {"nodes": results}


def normalized_mocker_nodes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    configured = config["topology"].get("mocker_nodes")
    if configured is not None:
        return [dict(node) for node in configured]
    return [
        {
            "rank": 1,
            "name": "mocker",
            "artifact_prefix": "mocker",
            "manage_etcd": True,
            "processes": list(config["topology"]["mocker_processes"]),
        }
    ]


def slurm_hosts(config: Mapping[str, Any]) -> list[str]:
    node_list = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get(
        "SLURM_JOB_NODELIST"
    )
    if not node_list:
        raise CampaignError("SLURM node list is unavailable")
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list], text=True
    ).split()
    expected = int(
        config.get("campaign", {}).get(
            "world_size", len(normalized_mocker_nodes(config)) + 2
        )
    )
    if len(hosts) != expected or len(set(hosts)) != expected:
        raise CampaignError(f"expected {expected} exclusive hosts, found {hosts}")
    return hosts


def role_preflight(
    config: Mapping[str, Any],
    rank: int,
    interface: str,
    networks: Mapping[str, Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_online = set(range(144))
    online = base.parse_cpu_set(
        Path("/sys/devices/system/cpu/online").read_text().strip()
    )
    if online != expected_online:
        raise CampaignError(f"expected CPUs 0-143, found {sorted(online)}")
    observed_nodes: dict[str, str] = {}
    for path in Path("/sys/devices/system/node").glob("node[0-9]*"):
        cpulist = path / "cpulist"
        if not cpulist.is_file():
            continue
        value = cpulist.read_text().strip()
        # Grace Hopper nodes expose GPU-memory NUMA nodes without CPUs. They
        # are real memory targets but are irrelevant to this CPU-placement
        # contract, which covers the two CPU-bearing NUMA domains only.
        if value:
            observed_nodes[path.name.removeprefix("node")] = value
    if observed_nodes != config["topology"]["node_cpus"]:
        raise CampaignError(f"unexpected NUMA topology: {observed_nodes}")
    required = ["git", "ip", "mpstat", "pidstat", "numactl", "ping"]
    mocker_ranks = {int(node["rank"]) for node in normalized_mocker_nodes(config)}
    aiperf_rank = (
        int(config.get("campaign", {}).get("world_size", len(mocker_ranks) + 2)) - 1
    )
    mocker_source = rank in mocker_ranks and config["paths"].get(
        "mocker_dynamo_repo"
    )
    dynamo_repo = str(
        config["paths"]["mocker_dynamo_repo"]
        if mocker_source
        else config["paths"]["dynamo_repo"]
    )
    dynamo_python = str(
        config["paths"]["mocker_dynamo_python"]
        if mocker_source
        else config["paths"]["dynamo_python"]
    )
    dynamo_sha = str(
        config["pins"]["mocker_dynamo"]
        if mocker_source
        else config["pins"]["dynamo"]
    )
    if rank == 0:
        required += ["perf", dynamo_python]
    elif rank in mocker_ranks:
        required += ["etcd", "etcdctl", dynamo_python]
    elif rank == aiperf_rank:
        required += [config["paths"]["aiperf"]]
    else:
        raise CampaignError(f"rank {rank} has no configured role")
    missing = [item for item in required if shutil.which(str(item)) is None]
    if missing:
        raise CampaignError(f"missing executables on rank {rank}: {missing}")
    repo = Path(dynamo_repo)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != dynamo_sha:
        raise CampaignError(f"Dynamo HEAD is {head}, expected {dynamo_sha}")
    diff_result = subprocess.run(["git", "-C", str(repo), "diff", "--quiet"])
    if diff_result.returncode != 0:
        expected_dirty_files = config["pins"].get("dirty_source_files")
        if not isinstance(expected_dirty_files, Mapping) or not expected_dirty_files:
            raise CampaignError("Dynamo checkout has tracked modifications")
        changed_files = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--name-only"], text=True
        ).splitlines()
        if changed_files != list(expected_dirty_files):
            raise CampaignError(
                f"unexpected tracked modifications: {changed_files}"
            )
        observed_dirty_files = {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in changed_files
        }
        if observed_dirty_files != dict(expected_dirty_files):
            raise CampaignError(
                "tracked modification checksums do not match the approved patch: "
                f"{observed_dirty_files}"
            )
    marker = repo / ".dynamo-native-rustflags"
    if (
        not marker.is_file()
        or marker.read_text().strip() != config["pins"]["rustflags"]
    ):
        raise CampaignError("native build marker is absent or incorrect")
    for name in ("model", "hf_home", "frontend_jemalloc"):
        if not Path(config["paths"][name]).exists():
            raise CampaignError(f"missing staged path {name}: {config['paths'][name]}")
    if rank == aiperf_rank and bool(
        config["runtime"].get("aiperf_native", False)
    ):
        checksum = Path(config["paths"]["aiperf_sha256"])
        if not checksum.is_file():
            raise CampaignError(f"missing native AIPerf checksum: {checksum}")
        completed = subprocess.run(
            ["sha256sum", "--check", str(checksum)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise CampaignError(
                f"native AIPerf checksum validation failed: {completed.stdout} "
                f"{completed.stderr}"
            )
        if not Path(config["paths"]["agentx_trace"]).is_file():
            raise CampaignError("native AgentX JSONL export is missing")
    if rank == 0 or rank in mocker_ranks:
        subprocess.run(
            [
                dynamo_python,
                "-c",
                "import dynamo._core, dynamo.frontend, dynamo.mocker",
            ],
            check=True,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            timeout=60,
        )
    network_checks: dict[str, Any] = {}
    for name, spec in networks.items():
        selected_interface = str(spec["interface"])
        selected_address = str(spec["address"])
        interface_path = Path("/sys/class/net") / selected_interface
        if not interface_path.exists():
            raise CampaignError(
                f"network interface does not exist: {selected_interface}"
            )
        operstate = (interface_path / "operstate").read_text().strip()
        if operstate != "up":
            raise CampaignError(
                f"network interface {selected_interface} is {operstate}, not up"
            )
        link_type = int((interface_path / "type").read_text().strip())
        expected_node = spec.get("numa_node")
        numa_path = interface_path / 'device' / 'numa_node'
        actual_node = int(numa_path.read_text().strip()) if numa_path.exists() else None
        if selected_interface == 'enP6p3s0f1np1' and actual_node != 0:
            raise CampaignError(f'Expected preserved Ethernet on NUMA0, found {actual_node}')
        if expected_node is not None:
            if link_type != 32:
                raise CampaignError(
                    f"{selected_interface} is not an IPoIB interface (ARPHRD type {link_type})"
                )
            numa_path = interface_path / "device" / "numa_node"
            actual_node = int(numa_path.read_text().strip())
            if actual_node != int(expected_node):
                raise CampaignError(
                    f"{selected_interface} is on NUMA node {actual_node}, expected {expected_node}"
                )
        peers: list[dict[str, Any]] = []
        for peer_agent in agents:
            if int(peer_agent["rank"]) == rank:
                continue
            peer_network = peer_agent.get("networks", {}).get(name)
            if peer_network is None:
                continue
            peer_address = str(peer_network["address"])
            route = json.loads(
                subprocess.check_output(
                    ["ip", "-j", "route", "get", peer_address], text=True
                )
            )[0]
            route_source = str(route.get("prefsrc") or route.get("src") or "")
            if (
                route.get("dev") != selected_interface
                or route_source != selected_address
            ):
                raise CampaignError(
                    f"route to {peer_address} uses dev={route.get('dev')} src={route_source}; "
                    f"expected dev={selected_interface} src={selected_address}"
                )
            ping = subprocess.run(
                [
                    "ping",
                    "-c",
                    "2",
                    "-W",
                    "2",
                    "-I",
                    selected_address,
                    peer_address,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if ping.returncode != 0:
                raise CampaignError(
                    f"{name} IPoIB reachability failed from {selected_address} "
                    f"to {peer_address}: {ping.stdout} {ping.stderr}"
                )
            peers.append(
                {
                    "rank": int(peer_agent["rank"]),
                    "address": peer_address,
                    "route": route,
                    "ping": ping.stdout,
                }
            )
        network_checks[str(name)] = {
            "interface": selected_interface,
            "address": selected_address,
            "link_type": link_type,
            "numa_node": actual_node,
            "mtu": int((interface_path / "mtu").read_text().strip()),
            "peers": peers,
        }
    return {
        "ok": True,
        "rank": rank,
        "hostname": socket.gethostname(),
        "interface": interface,
        "networks": network_checks,
        "online_cpus": sorted(online),
        "numa_nodes": observed_nodes,
        "dynamo_head": head,
        "checked_at": now(),
    }


def process_busy(
    path: Path, tracked_name: str, started: str, ended: str
) -> dict[str, Any]:
    samples = measured_samples(load_jsonl(path), started, ended)
    if len(samples) < 2:
        raise CampaignError(f"insufficient process telemetry in {path}")
    return process_cpu(samples, tracked_name)


def combined_process_busy(
    path: Path, tracked_names: Sequence[str], started: str, ended: str
) -> dict[str, Any]:
    samples = measured_samples(load_jsonl(path), started, ended)
    if len(samples) < 2:
        raise CampaignError(f"insufficient process telemetry in {path}")
    ticks_per_second = float(samples[0]["clock_ticks_per_second"])
    interval_cores: list[float] = []
    previous: tuple[float, dict[tuple[str, int], int]] | None = None
    for sample in samples:
        current: dict[tuple[str, int], int] = {}
        for name in tracked_names:
            for item in sample.get("tracked_processes", {}).get(name, []):
                current[(name, int(item["pid"]))] = int(item["user_ticks"]) + int(
                    item["system_ticks"]
                )
        if previous is not None:
            old_time, old = previous
            elapsed = float(sample["monotonic_seconds"]) - old_time
            delta = sum(
                max(0, ticks - old.get(key, ticks)) for key, ticks in current.items()
            )
            if elapsed > 0:
                interval_cores.append(delta / ticks_per_second / elapsed)
        previous = (float(sample["monotonic_seconds"]), current)
    by_process = {name: process_cpu(samples, name) for name in tracked_names}
    threads = sorted(
        (
            thread
            for result in by_process.values()
            for thread in result.get("top_threads", [])
        ),
        key=lambda item: float(item["cpu_seconds"]),
        reverse=True,
    )
    return {
        "cpu_seconds": sum(float(item["cpu_seconds"]) for item in by_process.values()),
        "average_occupied_cores": (
            sum(interval_cores) / len(interval_cores) if interval_cores else None
        ),
        "p95_occupied_cores": percentile(interval_cores, 0.95),
        "hottest_thread": threads[0] if threads else None,
        "top_threads": threads[:25],
        "by_process": by_process,
    }


def node_busy(path: Path, pool: str, started: str, ended: str) -> dict[str, Any]:
    lower = dt.datetime.fromisoformat(started)
    upper = dt.datetime.fromisoformat(ended)
    values: list[float] = []
    for sample in base._read_samples(path):
        stamp = dt.datetime.fromisoformat(sample["timestamp"])
        value = sample.get("pools", {}).get(pool, {}).get("busy_percent")
        if lower <= stamp <= upper and isinstance(value, (int, float)):
            values.append(float(value))
    return {
        "sample_count": len(values),
        "average": sum(values) / len(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copy2(source, destination)


def mocker_command(
    config: Mapping[str, Any], placement: Mapping[str, Any]
) -> list[str]:
    workload = config["workload"]
    runtime = config["runtime"]
    return base.process_prefix(str(placement["cpus"]), str(placement["memory"])) + [
        config["paths"].get(
            "mocker_dynamo_python", config["paths"]["dynamo_python"]
        ),
        "-c",
        "from dynamo.mocker.main import main; main()",
        "--response-plane", str(config["runtime"]["response_plane"]),
        "--model-path",
        config["paths"]["model"],
        "--model-name",
        config["model"]["name"],
        "--endpoint",
        "dyn://dynamo.backend.generate",
        "--num-workers",
        str(placement["workers"]),
        "--num-gpu-blocks-override",
        str(runtime["mocker_num_gpu_blocks"]),
        "--speedup-ratio",
        str(runtime["mocker_speedup_ratio"]),
        "--max-num-seqs",
        "100000",
        "--max-num-batched-tokens",
        "10000000",
        "--block-size",
        str(workload["block_size"]),
    ]


def aiperf_command(
    config: Mapping[str, Any],
    frontend_ip: str | Mapping[str, str],
    concurrency: int,
    duration: int,
    artifact_dir: Path,
) -> list[str]:
    workload = config["workload"]
    runtime = config["runtime"]
    weka_mmap_compat = bool(runtime.get("aiperf_weka_mmap_compat", False))
    dual_frontend = bool(runtime.get("frontend_dual_numa", False))
    default_frontend_ip = (
        str(next(iter(frontend_ip.values())))
        if isinstance(frontend_ip, Mapping)
        else frontend_ip
    )
    urls = [f"http://{default_frontend_ip}:{config['ports']['frontend_http']}"]
    if dual_frontend:
        if isinstance(frontend_ip, Mapping):
            endpoints = {
                "numa0": f"http://{frontend_ip['numa0']}:8001",
                "numa1": f"http://{frontend_ip['numa1']}:8002",
            }
        else:
            endpoints = {
                "numa0": f"http://{frontend_ip}:8001",
                "numa1": f"http://{frontend_ip}:8002",
            }
        endpoint_weights = runtime.get(
            "aiperf_endpoint_weights", {"numa0": 1, "numa1": 1}
        )
        urls = [
            endpoints[name]
            for name in ("numa0", "numa1")
            for _ in range(int(endpoint_weights[name]))
        ]
    if bool(runtime.get("aiperf_native", False)):
        native_config = {
            "schemaVersion": "2.0",
            "randomSeed": 12345,
            "benchmark": {
                "model": config["model"]["name"],
                # Match legacy AIPerf: record rare request failures and continue
                # instead of graph-mode's native fail-fast default.
                "failurePolicy": "continue",
                "endpoint": {
                    "urls": urls,
                    "urlStrategy": "round_robin",
                    "type": str(runtime.get("aiperf_endpoint_type", "chat")),
                    "streaming": True,
                    "timeout": 21600.0,
                    "useServerTokenCount": True,
                    "extra": {"ignore_eos": True},
                },
                "dataset": {
                    "type": "file",
                    "path": config["paths"]["agentx_trace"],
                    "format": "weka_trace",
                    "entries": int(workload["dataset_entries"]),
                    "sampling": "sequential",
                    "synthesis": {
                        "speedup_ratio": 1.0,
                        "prefix_len_multiplier": 1.0,
                        "prefix_root_multiplier": 1,
                        "prompt_len_multiplier": 1.0,
                        "output_len_multiplier": 1.0,
                        "allow_dataset_wrap": True,
                        "idle_gap_cap_seconds": 60.0,
                        "ignore_trace_delays": True,
                        "dataset_sampling_strategy": "sequential",
                    },
                },
                "tokenizer": {"name": config["paths"]["model"]},
                "warmup": {
                    "type": "concurrency",
                    "requests": int(workload["warmup_request_count"]),
                    "concurrency": concurrency,
                },
                "profiling": {
                    "type": "concurrency",
                    "duration": duration,
                    "concurrency": concurrency,
                    "gracePeriod": int(workload["grace_period_seconds"]),
                },
                "artifacts": {
                    "dir": str(artifact_dir),
                    "summary": ["json"],
                    "records": ["jsonl"],
                },
                "gpuTelemetry": {"enabled": False},
                "serverMetrics": {"enabled": False},
            },
            "runtime": {
                "workers": int(runtime["aiperf_native_workers"]),
                "dispatch": "global",
                "ui": "none",
                "statsInterval": 1,
            },
        }
        native_config_path = artifact_dir / "rust-aiperf-config.json"
        atomic_json(native_config_path, native_config)
        return [
            config["paths"]["aiperf"],
            "profile",
            "--config",
            str(native_config_path),
            "--ui",
            "none",
        ]
    command = [
        config["paths"]["aiperf"],
        "profile",
        "--model",
        config["model"]["name"],
    ]
    if not weka_mmap_compat:
        command.extend(["--tokenizer", config["paths"]["model"]])
    else:
        command.extend(["--tokenizer-revision", config["pins"]["model_revision"]])
    command.extend(
        [
            "--url",
            urls[0],
            "--url-strategy",
            "round-robin",
            "--endpoint-type",
            str(runtime.get("aiperf_endpoint_type", "chat")),
            "--streaming",
            "--concurrency",
            str(concurrency),
            "--benchmark-duration",
            str(duration),
            "--benchmark-grace-period",
            str(workload["grace_period_seconds"]),
            "--warmup-request-count",
            str(workload["warmup_request_count"]),
            "--workers-max",
            str(runtime["aiperf_workers_max"]),
            "--record-processors",
            str(runtime["aiperf_record_processors"]),
            "--random-seed",
            "12345",
            "--output-artifact-dir",
            str(artifact_dir),
            "--export-level",
            "records",
            "--no-server-metrics",
            "--stats-interval",
            "0",
            "--ui",
            "none",
            "--extra-inputs",
            "ignore_eos:true",
            "--use-server-token-count",
            "--no-fixed-schedule",
        ]
    )
    if weka_mmap_compat:
        command.extend(
            [
                "--input-file",
                config["paths"]["weka_mmap_cache"],
                "--custom-dataset-type",
                "weka_mmap_compat",
            ]
        )
    else:
        command.extend(
            [
                "--ignore-trace-delays",
                "--public-dataset",
                "weka_hf",
                "--hf-weka-repo",
                config["model"]["agentx_dataset"],
                "--num-dataset-entries",
                str(workload["dataset_entries"]),
            ]
        )
    for url in urls[1:]:
        command.extend(["--url", url])
    return command


def wait_for_native_aiperf_process_tree(
    process: base.ManagedProcess,
    run_dir: Path,
    *,
    expected_workers: int,
    timeout: int,
) -> dict[str, Any]:
    """Wait for the native execute child and its thread-per-core workers."""

    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    pgid = os.getpgid(process.pid)
    while time.monotonic() < deadline:
        process.require_alive()
        processes = []
        for item in Path("/proc").iterdir():
            if not item.name.isdigit():
                continue
            pid = int(item.name)
            try:
                if os.getpgid(pid) != pgid:
                    continue
                cmdline = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    errors="replace"
                )
                tids = sorted(
                    int(task.name)
                    for task in (item / "task").iterdir()
                    if task.name.isdigit()
                )
                processes.append(
                    {
                        "pid": pid,
                        "cmdline": cmdline.strip(),
                        "thread_count": len(tids),
                        "thread_ids": tids,
                        "affinity": sorted(os.sched_getaffinity(pid)),
                    }
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        last = {
            "controller_pid": process.pid,
            "pgid": pgid,
            "expected_native_workers": expected_workers,
            "processes": sorted(processes, key=lambda item: item["pid"]),
            "total_threads": sum(item["thread_count"] for item in processes),
            "max_process_threads": max(
                (item["thread_count"] for item in processes), default=0
            ),
            "captured_at": now(),
        }
        if last["max_process_threads"] >= expected_workers:
            atomic_json(run_dir / "aiperf_process_tree.json", last)
            return last
        time.sleep(1)
    atomic_json(run_dir / "aiperf_process_tree.json", last)
    raise CampaignError(
        "native AIPerf did not expose its thread-per-core worker set: "
        f"expected at least {expected_workers} threads in one process, observed {last}"
    )


def wait_for_mockers(
    config: Mapping[str, Any], endpoint: str, processes: Iterable[base.ManagedProcess]
) -> None:
    deadline = time.monotonic() + int(
        config["workload"].get("mocker_registration_timeout_seconds", 240)
    )
    while time.monotonic() < deadline:
        for process in processes:
            process.require_alive()
        completed = subprocess.run(
            [
                "etcdctl",
                f"--endpoints={endpoint}",
                "get",
                "--prefix",
                "v1/instances/dynamo/backend/generate/",
                "--keys-only",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        count = completed.stdout.count("generate/") if completed.returncode == 0 else 0
        if count == config["runtime"]["num_mockers"]:
            return
        time.sleep(1)
    raise CampaignError(
        f"{config['runtime']['num_mockers']} mockers did not register"
    )


class MockerAgent:
    def __init__(
        self,
        config: Mapping[str, Any],
        channel: Channel,
        shutdown: threading.Event,
        *,
        job_id: str,
        rank: int,
        node_spec: Mapping[str, Any],
        advertised_ip: str,
        etcd_ip: str,
        interface: str,
        networks: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.channel = channel
        self.shutdown = shutdown
        self.job_id = job_id
        self.rank = rank
        self.node_spec = dict(node_spec)
        self.advertised_ip = advertised_ip
        self.etcd_ip = etcd_ip
        self.interface = interface
        self.networks = (
            {name: dict(spec) for name, spec in networks.items()}
            if networks is not None
            else {
                "primary": {
                    "interface": interface,
                    "address": advertised_ip,
                    "numa_node": None,
                }
            }
        )
        self.processes: dict[str, base.ManagedProcess] = {}
        self.metrics_urls: dict[str, str] = {}
        self.telemetry: list[base.ManagedProcess] = []
        self.sampler: base.Sampler | None = None
        self.perf_processes: list[
            tuple[str, subprocess.Popen[Any], Any, Path, Path, Path]
        ] = []
        self.run_dir: Path | None = None
        self.scratch: Path | None = None

    def serve(self) -> None:
        sequence = 1
        try:
            while not self.shutdown.is_set():
                command = self.channel.receive(sequence)
                action = command["action"]
                try:
                    if action == "prepare":
                        result = self.prepare(command["payload"])
                    elif action == "verify_mockers":
                        result = self.verify_mockers()
                    elif action == "start_monitor":
                        result = self.start_monitor(command["payload"])
                    elif action == "finish_monitor":
                        result = self.finish_monitor(command["payload"])
                    elif action == "start_perf":
                        result = self.start_perf()
                    elif action == "finish_perf":
                        result = self.finish_perf()
                    elif action == "stop":
                        self.cleanup()
                        result = {"stopped_at": now()}
                    elif action == "terminate":
                        self.cleanup()
                        self.channel.acknowledge(sequence, action, result={})
                        return
                    else:
                        raise CampaignError(f"unknown mocker action {action}")
                    self.channel.acknowledge(sequence, action, result=result)
                except Exception as error:
                    detail = (
                        f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                    )
                    try:
                        self.cleanup()
                    except Exception as cleanup_error:
                        detail += f"\ncleanup: {cleanup_error}"
                    self.channel.acknowledge(sequence, action, error=detail)
                sequence += 1
        finally:
            self.cleanup()

    def prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.cleanup()
        self.run_dir = Path(payload["run_dir"])
        scratch_root = Path(os.environ["SLURM_TMPDIR"])
        self.scratch = (
            scratch_root
            / "dynamo-saturation"
            / self.job_id
            / f"rank{self.rank}"
            / payload["run_id"]
        )
        self.scratch.mkdir(parents=True, exist_ok=False)
        client = int(self.config["ports"]["etcd_client"])
        peer = int(self.config["ports"]["etcd_peer"])
        etcd_env = os.environ | {"ETCD_UNSUPPORTED_ARCH": "arm64"}
        if self.node_spec.get("manage_etcd"):
            etcd_data = self.scratch / "etcd"
            etcd_data.mkdir()
            etcd_cmd = [
                "etcd",
                "--name",
                "default",
                "--data-dir",
                str(etcd_data),
                "--listen-client-urls",
                f"http://{self.advertised_ip}:{client}",
                "--advertise-client-urls",
                f"http://{self.advertised_ip}:{client}",
                "--listen-peer-urls",
                f"http://{self.advertised_ip}:{peer}",
                "--initial-advertise-peer-urls",
                f"http://{self.advertised_ip}:{peer}",
                "--initial-cluster",
                f"default=http://{self.advertised_ip}:{peer}",
            ]
            self.processes["etcd"] = base.ManagedProcess(
                "etcd", etcd_cmd, self.scratch / "etcd.log", env=etcd_env
            ).start()
        endpoint = f"http://{self.etcd_ip}:{client}"
        event_host_override = bool(
            self.config.get("network", {}).get("event_plane_host_override", False)
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if (
                subprocess.run(
                    ["etcdctl", f"--endpoints={endpoint}", "endpoint", "health"],
                    capture_output=True,
                    env=etcd_env,
                ).returncode
                == 0
            ):
                break
            if "etcd" in self.processes:
                self.processes["etcd"].require_alive()
            time.sleep(1)
        else:
            raise CampaignError("etcd did not become healthy")
        placements = list(self.node_spec["processes"])
        process_networks: dict[str, dict[str, Any]] = {}
        process_environments: dict[str, dict[str, str]] = {}
        metrics_base_port = self.config["runtime"].get(
            "mocker_system_metrics_base_port"
        )
        for index, placement in enumerate(placements):
            name = placement["name"]
            network = select_network(self.networks, placement.get("network"))
            env = base._base_runtime_env(
                self.config,
                str(network["address"]),
                self.etcd_ip,
                str(network["interface"]),
            )
            if env.get("DYN_VELO_RESPONSE_TRANSPORT") == "ucx":
                node = str(placement["memory"]).removeprefix("bind:")
                env["UCX_NET_DEVICES"] = self.config["network"]["ucx_numa_devices"][node]
            env["DYN_RUNTIME_NUM_WORKER_THREADS"] = "36"
            env["DYN_SELF_HOST_METADATA"] = (
                "1"
                if bool(
                    self.config["runtime"].get("mocker_self_host_metadata", True)
                )
                else "0"
            )
            mocker_zmq_max_sockets = self.config["runtime"].get(
                "mocker_zmq_max_sockets",
                self.config["runtime"].get("zmq_max_sockets"),
            )
            if mocker_zmq_max_sockets is not None:
                env["DYN_ZMQ_MAX_SOCKETS"] = str(
                    mocker_zmq_max_sockets
                )
            process_networks[name] = network
            process_environments[name] = {
                key: env[key]
                for key in (
                    "ETCD_ENDPOINTS",
                    "DYN_TCP_RPC_HOST",
                    "DYN_RESPONSE_STREAM_HOST",
                    "DYN_RESPONSE_STREAM_PORT",
                    "DYN_RESPONSE_STREAM_HIGH_WINDOW",
                    "DYN_TCP_RESPONSE_STREAM_HOST",
                    "DYN_REQUEST_PLANE",
                    "DYN_REQUEST_PLANE_CODEC",
                    "DYN_EVENT_PLANE",
                    "DYN_EVENT_PLANE_CODEC",
                    "DYN_EVENT_PLANE_HOST",
                    "DYN_TCP_RESPONSE_MUX",
                    "DYN_TCP_RESPONSE_BATCH_INTERVAL_MS",
                    "DYN_SELF_HOST_METADATA",
                    "DYN_QUIC_RESPONSE_CONNECTIONS",
                    "DYN_QUIC_RESPONSE_LANES",
                    "DYN_QUIC_RESPONSE_BATCH_INTERVAL_US",
                    "DYN_ZMQ_BROKER_ENABLED",
                    "DYN_ZMQ_BROKER_URL",
                    "DYN_ZMQ_MAX_SOCKETS",
                    "DYN_RESPONSE_PLANE",
                    "DYN_VELO_RESPONSE_TRANSPORT",
                    "UCX_TLS", "UCX_NET_DEVICES", "UCX_LOG_LEVEL",
                    "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB",
                )
                if key in env
            }
            if self.config["runtime"].get("tcp_response_batch_interval_ms") is not None:
                env["DYN_TCP_RESPONSE_BATCH_INTERVAL_MS"] = str(
                    self.config["runtime"].get("tcp_response_batch_interval_ms", 5)
                )
                process_environments[name]["DYN_TCP_RESPONSE_BATCH_INTERVAL_MS"] = env[
                    "DYN_TCP_RESPONSE_BATCH_INTERVAL_MS"
                ]
            if self.config["runtime"].get("tcp_response_pool_size") is not None:
                env["DYN_TCP_RESPONSE_POOL_SIZE"] = str(
                    self.config["runtime"]["tcp_response_pool_size"]
                )
                process_environments[name]["DYN_TCP_RESPONSE_POOL_SIZE"] = env[
                    "DYN_TCP_RESPONSE_POOL_SIZE"
                ]
            if bool(self.config["runtime"].get("response_packet_metrics", False)):
                env["DYN_TCP_RESPONSE_PACKET_METRICS"] = "1"
                process_environments[name]["DYN_TCP_RESPONSE_PACKET_METRICS"] = "1"
            if metrics_base_port is not None:
                metrics_port = int(metrics_base_port) + index
                env["DYN_SYSTEM_HOST"] = str(network["address"])
                env["DYN_SYSTEM_PORT"] = str(metrics_port)
                process_environments[name]["DYN_SYSTEM_HOST"] = env["DYN_SYSTEM_HOST"]
                process_environments[name]["DYN_SYSTEM_PORT"] = env["DYN_SYSTEM_PORT"]
                self.metrics_urls[name] = (
                    f"http://{network['address']}:{metrics_port}/metrics"
                )
            if self.config["runtime"].get("quic_response_lanes") is not None:
                env["DYN_QUIC_RESPONSE_CONNECTIONS"] = str(
                    self.config["runtime"].get("quic_response_connections", 1)
                )
                env["DYN_QUIC_RESPONSE_LANES"] = str(
                    self.config["runtime"]["quic_response_lanes"]
                )
                env["DYN_QUIC_RESPONSE_BATCH_INTERVAL_US"] = str(
                    self.config["runtime"].get("quic_response_batch_interval_us", 1000)
                )
                process_environments[name]["DYN_QUIC_RESPONSE_CONNECTIONS"] = env[
                    "DYN_QUIC_RESPONSE_CONNECTIONS"
                ]
                process_environments[name]["DYN_QUIC_RESPONSE_LANES"] = env[
                    "DYN_QUIC_RESPONSE_LANES"
                ]
                process_environments[name]["DYN_QUIC_RESPONSE_BATCH_INTERVAL_US"] = env[
                    "DYN_QUIC_RESPONSE_BATCH_INTERVAL_US"
                ]
            self.processes[name] = base.ManagedProcess(
                name,
                mocker_command(self.config, placement),
                self.scratch / f"{name}.log",
                env=env,
            ).start()
            if index + 1 < len(placements):
                if bool(
                    self.config["runtime"].get(
                        "serialize_mocker_process_startup", False
                    )
                ):
                    process = self.processes[name]
                    expected = int(placement["workers"])
                    deadline = time.monotonic() + 300
                    while time.monotonic() < deadline:
                        process.require_alive()
                        log_text = process.log_path.read_text(errors="replace")
                        if "panicked at" in log_text:
                            raise CampaignError(
                                f"mocker {name} panicked during serialized startup"
                            )
                        # Fanout starts nodes and processes in order. Check discovery,
                        # independent of the binary's compiled log level.
                        expected = sum(
                            int(p["workers"])
                            for n in self.config["topology"]["mocker_nodes"]
                            if int(n["rank"]) < self.rank
                            for p in n["processes"]
                        ) + sum(int(p["workers"]) for p in placements[:index + 1])
                        discovery = subprocess.run(
                            ["etcdctl", f"--endpoints={endpoint}", "get", "--prefix",
                             "v1/event_channels/", "--keys-only"],
                            capture_output=True, text=True, timeout=10,
                            stdin=subprocess.DEVNULL,
                        )
                        registered = sum(
                            "kv-events" in line for line in discovery.stdout.splitlines()
                        ) if discovery.returncode == 0 else 0
                        if registered >= expected:
                            break
                        time.sleep(0.5)
                    else:
                        raise CampaignError(
                            f"mocker {name} did not bind {expected} KV publishers "
                            "during serialized startup"
                        )
                else:
                    time.sleep(1)
        affinities: dict[str, dict[str, Any]] = {}
        for placement in placements:
            deadline = time.monotonic() + 60
            while True:
                process = self.processes[placement["name"]]
                process.require_alive()
                observed = base._validate_process_affinity(process, placement["cpus"])
                if observed["valid"]:
                    affinities[placement["name"]] = observed
                    break
                if time.monotonic() >= deadline:
                    affinities[placement["name"]] = observed
                    break
                time.sleep(0.25)
        if not all(value["valid"] for value in affinities.values()):
            raise CampaignError(f"mocker affinity validation failed: {affinities}")
        prefix = str(self.node_spec.get("artifact_prefix", self.node_spec["name"]))
        atomic_json(self.run_dir / f"{prefix}-affinity.json", affinities)
        atomic_json(
            self.run_dir / f"{prefix}-pids.json",
            {
                name: {
                    "pid": process.pid,
                    "pgid": os.getpgid(process.pid),
                    "argv": process.command,
                }
                for name, process in self.processes.items()
            },
        )
        atomic_json(
            self.run_dir / f"{prefix}-network-bindings.json",
            {
                "etcd": {
                    "address": self.advertised_ip,
                    "endpoint": endpoint,
                },
                "processes": process_networks,
                "environments": process_environments,
                "event_plane_note": (
                    "DYN_TCP_RPC_HOST and DYN_EVENT_PLANE_HOST pin request and "
                    "direct-ZMQ traffic to each process's configured interface. "
                    "The frontend, not the mocker, owns the streaming-response listener."
                    if self.config.get("network", {}).get("response_stream_interface")
                    else "DYN_TCP_RPC_HOST, DYN_TCP_RESPONSE_STREAM_HOST, and "
                    "DYN_EVENT_PLANE_HOST pin request, streaming-response, and "
                    "direct-ZMQ traffic to each process's configured interface."
                    if event_host_override
                    else "DYN_TCP_RPC_HOST pins request-plane advertisement. Direct "
                    "ZMQ publication uses Dynamo local_ip_for_advertise()."
                ),
            },
        )
        return {
            "ready_at": now(),
            "local_workers": sum(int(item["workers"]) for item in placements),
            "affinity": affinities,
        }

    def verify_mockers(self) -> dict[str, Any]:
        if not self.node_spec.get("manage_etcd"):
            raise CampaignError(
                "only the etcd owner can verify global mocker registration"
            )
        endpoint = f"http://{self.etcd_ip}:{self.config['ports']['etcd_client']}"
        local = [self.processes[item["name"]] for item in self.node_spec["processes"]]
        wait_for_mockers(self.config, endpoint, local)
        for process in local:
            process.require_alive()
            log_text = process.log_path.read_text(errors="replace")
            if "panicked at" in log_text or "Failed to bind ZMQ publisher" in log_text:
                raise CampaignError(
                    f"mocker {process.name} reported a runtime panic during startup"
                )
        return {
            "verified_at": now(),
            "registered_mockers": self.config["runtime"]["num_mockers"],
        }

    def start_monitor(self, _payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.run_dir is None or self.scratch is None:
            raise CampaignError("mocker monitor started before prepare")
        tracked = {
            name: os.getpgid(process.pid) for name, process in self.processes.items()
        }
        prefix = str(self.node_spec.get("artifact_prefix", self.node_spec["name"]))
        self.telemetry = base._start_system_telemetry(
            self.run_dir, role=prefix, tracked=tracked
        )
        for name, url in self.metrics_urls.items():
            self.telemetry.append(
                base.ManagedProcess(
                    f"{name}-metrics",
                    [
                        sys.executable,
                        str(Path(__file__).with_name("metrics_capture.py")),
                        "--role",
                        name,
                        "--url",
                        url,
                        "--output",
                        str(self.run_dir / f"{name}-metrics.jsonl"),
                    ],
                    self.run_dir / f"{name}-metrics-collector.log",
                ).start()
            )
        self.sampler = base.Sampler(
            self.scratch / f"{prefix}-samples.jsonl",
            pools={"mocker_node": "0-143", "numa0": "0-71", "numa1": "72-143"},
            interface=[
                str(
                    select_network(self.networks, placement.get("network"))["interface"]
                )
                for placement in self.node_spec["processes"]
            ],
            interval=1.0,
        )
        self.sampler.start()
        return {"monitoring_started_at": now()}

    def finish_monitor(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.run_dir is None or self.scratch is None or self.sampler is None:
            raise CampaignError("mocker monitor is not running")
        self.sampler.stop()
        self.sampler = None
        base._stop_processes(self.telemetry)
        self.telemetry = []
        prefix = str(self.node_spec.get("artifact_prefix", self.node_spec["name"]))
        samples_path = self.scratch / f"{prefix}-samples.jsonl"
        shutil.copy2(samples_path, self.run_dir / f"{prefix}-samples.jsonl")
        started, ended = (
            str(payload["measurement_started"]),
            str(payload["measurement_ended"]),
        )
        node = node_busy(samples_path, "mocker_node", started, ended)
        process = {
            placement["name"]: process_busy(
                self.run_dir / f"{prefix}-system-telemetry.jsonl",
                placement["name"],
                started,
                ended,
            )
            for placement in self.node_spec["processes"]
        }
        memory = {
            placement["name"]: base._validate_process_memory(
                self.processes[placement["name"]],
                int(str(placement["memory"]).removeprefix("bind:")),
                0.99,
            )
            for placement in self.node_spec["processes"]
        }
        limits = self.config["acceptance"]
        headroom = (
            node["average"] is not None
            and node["p95"] is not None
            and node["average"] < limits["load_node_average_busy_max_percent"]
            and node["p95"] < limits["load_node_p95_busy_max_percent"]
        )
        result = {
            "node_busy_percent": node,
            "process_cpu": process,
            "memory": memory,
            "headroom_accepted": headroom,
            "accepted": headroom and all(value["valid"] for value in memory.values()),
        }
        atomic_json(self.run_dir / f"{prefix}-acceptance.json", result)
        return result

    def start_perf(self) -> dict[str, Any]:
        if self.run_dir is None or self.scratch is None:
            raise CampaignError("mocker perf started before prepare")
        if self.perf_processes:
            raise CampaignError("mocker perf is already running")
        selected = self.config["profile"].get("mocker_process")
        started: dict[str, Any] = {}
        for placement in self.node_spec["processes"]:
            name = str(placement["name"])
            if selected is not None and name != selected:
                continue
            target = self.processes[name]
            target.require_alive()
            profile_dir = self.run_dir / "profiles" / "mocker" / name
            profile_dir.mkdir(parents=True, exist_ok=True)
            base._capture_frontend_dso_manifest(target.pid, profile_dir)
            capture_cpus = self.config["profile"].get("mocker_perf_cpus")
            atomic_json(
                profile_dir / "capture-target.json",
                {
                    "pid": target.pid,
                    "cpus": capture_cpus,
                    "mode": "per-cpu" if capture_cpus is not None else "per-process",
                },
            )
            data = self.scratch / f"oncpu-{name}.data"
            error_path = self.scratch / f"perf-{name}.err"
            error = error_path.open("w")
            target_args = (
                ["-a", "-C", str(capture_cpus)]
                if capture_cpus is not None
                else ["-p", str(target.pid)]
            )
            process = subprocess.Popen(
                [
                    "perf",
                    "record",
                        "--clockid", "mono",
                    "-e",
                    str(self.config["profile"].get("event", "cycles:u")),
                    "-F",
                    str(self.config["profile"]["frequency_hz"]),
                    "-m",
                    str(self.config["profile"]["perf_mmap_pages"]),
                    "--call-graph",
                    "dwarf,16384",
                    *target_args,
                    "-o",
                    str(data),
                    "--",
                    "sleep",
                    str(self.config["profile"]["record_seconds"]),
                ],
                stderr=error,
            )
            self.perf_processes.append(
                (name, process, error, data, error_path, profile_dir)
            )
            started[name] = {"pid": target.pid, "started_at": now()}
        return {"profiles": started}

    def finish_perf(self) -> dict[str, Any]:
        if not self.perf_processes:
            return {"profiles": {}}
        timeout = int(self.config["profile"]["record_seconds"]) + 300
        results: dict[str, Any] = {}
        try:
            for name, process, error, data, error_path, profile_dir in self.perf_processes:
                returncode = process.wait(timeout=timeout)
                error.flush()
                copy_if_present(error_path, profile_dir / "perf.err")
                copy_if_present(data, profile_dir / "oncpu.data")
                output = profile_dir / "oncpu.data"
                nonempty = output.is_file() and output.stat().st_size > 0
                if returncode != 0 or not nonempty:
                    raise CampaignError(
                        f"mocker perf capture for {name} failed: "
                        f"rc={returncode}, nonempty={nonempty}"
                    )
                base._capture_mapped_dso_checksums(profile_dir)
                results[name] = {
                    "returncode": returncode,
                    "nonempty": nonempty,
                    "data_bytes": output.stat().st_size,
                    "finished_at": now(),
                }
        finally:
            for _, process, error, _, _, _ in self.perf_processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)
                error.close()
            self.perf_processes = []
        return {"profiles": results}

    def cleanup(self) -> None:
        for _, process, error, data, error_path, profile_dir in self.perf_processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            error.flush()
            copy_if_present(error_path, profile_dir / "perf.err")
            copy_if_present(data, profile_dir / "oncpu.data")
            error.close()
        self.perf_processes = []
        if self.sampler is not None:
            self.sampler.stop(raise_on_error=False)
            self.sampler = None
        try:
            base._stop_processes(self.telemetry)
        finally:
            self.telemetry = []
        run_dir, scratch = self.run_dir, self.scratch
        names = [item["name"] for item in reversed(self.node_spec["processes"])] + [
            "etcd"
        ]
        for name in names:
            process = self.processes.pop(name, None)
            if process is not None:
                process.stop()
        if scratch is not None and run_dir is not None:
            names = ["etcd.log"] + [
                f"{item['name']}.log" for item in self.node_spec["processes"]
            ]
            for name in names:
                copy_if_present(scratch / name, run_dir / name)
            shutil.rmtree(scratch, ignore_errors=True)
        self.run_dir = None
        self.scratch = None
        self.metrics_urls = {}


def aiperf_special_process_health(
    path: Path, started: str, ended: str
) -> dict[str, Any]:
    samples = measured_samples(load_jsonl(path), started, ended)
    if len(samples) < 2:
        return {"accepted": False, "reason": "insufficient telemetry"}
    ticks_per_second = float(samples[0]["clock_ticks_per_second"])
    elapsed = float(samples[-1]["monotonic_seconds"]) - float(
        samples[0]["monotonic_seconds"]
    )
    by_identity: dict[tuple[int, str], list[int]] = {}
    for sample in samples:
        for item in sample.get("tracked_processes", {}).get("aiperf", []):
            command = str(item.get("cmdline", ""))
            if not re.search(
                r"timing.manager|timing_manager|record_processor", command, re.I
            ):
                continue
            key = (int(item["pid"]), command)
            by_identity.setdefault(key, []).append(
                int(item["user_ticks"]) + int(item["system_ticks"])
            )
    rows = []
    for (pid, command), values in by_identity.items():
        cpu_fraction = (
            (max(values) - min(values)) / ticks_per_second / elapsed
            if elapsed > 0
            else 1.0
        )
        rows.append(
            {"pid": pid, "cmdline": command, "cpu_fraction_of_one_core": cpu_fraction}
        )
    return {
        "processes": rows,
        "max_core_fraction": max(
            (row["cpu_fraction_of_one_core"] for row in rows), default=0.0
        ),
    }


def aiperf_warning_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if re.search(
            r"event.?loop.*(?:overrun|lag|stalled|taking too long)|timing.?manager.*(?:overrun|lag|stalled|taking too long)",
            line,
            re.I,
        )
    ]


def wait_for_file_quiescence(
    path: Path, *, timeout_seconds: float = 60.0, stable_seconds: float = 5.0
) -> bool:
    """Wait for worker-side JSONL flushes after the AIPerf controller exits."""

    deadline = time.monotonic() + timeout_seconds
    previous: tuple[int, int] | None = None
    stable_since: float | None = None
    while time.monotonic() < deadline:
        try:
            stat = path.stat()
        except FileNotFoundError:
            previous = None
            stable_since = None
            time.sleep(1)
            continue
        signature = (stat.st_size, stat.st_mtime_ns)
        current = time.monotonic()
        if signature != previous:
            previous = signature
            stable_since = current
        elif stable_since is not None and current - stable_since >= stable_seconds:
            return True
        time.sleep(1)
    return False


def validate_profile_artifact(
    path: Path, workload: Mapping[str, Any], *, sending_ended: str | None = None, expected_records: int | None = None
) -> dict[str, Any]:
    """Validate completed records; report cancellations after sending separately."""

    if not path.is_file() or path.stat().st_size == 0:
        raise CampaignError(f"missing AIPerf record export: {path}")
    records = errors = cancelled = mismatched_tokens = wrong_synthetic_length = 0
    drain_cancelled = 0
    request_ids = set()
    cutoff_ns = int(dt.datetime.fromisoformat(sending_ended).timestamp() * 1e9) if sending_ended else None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CampaignError(f"invalid JSONL in {path}: {error}") from error
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase") != "profiling":
                continue
            records += 1
            request_id = metadata.get("x_request_id")
            if not request_id or request_id in request_ids:
                raise CampaignError(f"missing or duplicate exported request ID: {request_id}")
            request_ids.add(request_id)
            if metadata.get("context_overflow_skip"):
                raise CampaignError("client exported a skipped context-overflow request")
            if metadata.get("was_cancelled") and cutoff_ns is not None and metadata.get("request_end_ns", 0) >= cutoff_ns:
                drain_cancelled += 1
                continue
            cancelled += int(bool(metadata.get("was_cancelled")))
            record_error = (
                record.get("error")
                or metadata.get("error")
                or metadata.get("error_code")
            )
            if record_error:
                errors += 1
                continue
            metrics = record.get("metrics", {})
            output = metrics.get("output_token_count", {}).get("value")
            usage = metrics.get("usage_completion_tokens", {}).get("value")
            if output is None or usage is None or int(output) != int(usage):
                mismatched_tokens += 1
            if str(workload.get("kind", "")) == "synthetic" and output is not None:
                expected = int(workload.get("osl", workload.get("output_tokens", 1024)))
                wrong_synthetic_length += int(int(output) != expected)
    allowed_error_fraction = float(workload.get("allowed_error_fraction", 0.0))
    error_fraction = errors / records if records else None
    if (
        records == 0
        or (expected_records is not None and records != expected_records)
        or (error_fraction is not None and error_fraction > allowed_error_fraction)
        or cancelled
        or wrong_synthetic_length
        or mismatched_tokens
    ):
        raise CampaignError(
            "AIPerf record validation failed: "
            f"records={records}, expected_records={expected_records}, errors={errors}, cancelled={cancelled}, "
            f"error_fraction={error_fraction}, "
            f"allowed_error_fraction={allowed_error_fraction}, "
            f"token_mismatches={mismatched_tokens}, "
            f"wrong_synthetic_length={wrong_synthetic_length}"
        )
    return {
        "profiling_records": records,
        "error_records": errors,
        "cancelled": cancelled,
        "drain_cancelled": drain_cancelled,
        "token_count_mismatches": mismatched_tokens,
        "wrong_synthetic_length": wrong_synthetic_length,
        "error_fraction": error_fraction,
        "allowed_error_fraction": allowed_error_fraction,
    }


class AiperfAgent:
    def __init__(
        self,
        config: Mapping[str, Any],
        channel: Channel,
        shutdown: threading.Event,
        *,
        job_id: str,
        advertised_ip: str,
        frontend_ip: str | Mapping[str, str],
        interface: str,
        networks: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.channel = channel
        self.shutdown = shutdown
        self.job_id = job_id
        self.advertised_ip = advertised_ip
        self.frontend_ip = frontend_ip
        self.interface = interface
        self.networks = (
            {name: dict(spec) for name, spec in networks.items()}
            if networks is not None
            else {
                "primary": {
                    "interface": interface,
                    "address": advertised_ip,
                    "numa_node": None,
                }
            }
        )
        self.run_dir: Path | None = None
        self.scratch: Path | None = None
        self.load: base.ManagedProcess | None = None
        self.telemetry: list[base.ManagedProcess] = []
        self.sampler: base.Sampler | None = None
        self.aiperf_profile_dir: Path | None = None
        self.aiperf_profile_scratch: Path | None = None

    def serve(self) -> None:
        sequence = 1
        try:
            while not self.shutdown.is_set():
                command = self.channel.receive(sequence)
                action = command["action"]
                try:
                    if action == "prepare":
                        result = self.prepare(command["payload"])
                    elif action == "run_load":
                        result = self.run_load(command["payload"])
                    elif action == "stop":
                        self.cleanup()
                        result = {"stopped_at": now()}
                    elif action == "terminate":
                        self.cleanup()
                        self.channel.acknowledge(sequence, action, result={})
                        return
                    else:
                        raise CampaignError(f"unknown AIPerf action {action}")
                    self.channel.acknowledge(sequence, action, result=result)
                except Exception as error:
                    detail = (
                        f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                    )
                    if (
                        self.run_dir is not None
                        and self.load is not None
                        and self.load.log_path.is_file()
                    ):
                        shutil.copy2(
                            self.load.log_path,
                            self.run_dir / "aiperf-failed.log",
                        )
                    try:
                        self.cleanup()
                    except Exception as cleanup_error:
                        detail += f"\ncleanup: {cleanup_error}"
                    self.channel.acknowledge(sequence, action, error=detail)
                sequence += 1
        finally:
            self.cleanup()

    def prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.cleanup()
        self.run_dir = Path(payload["run_dir"])
        self.scratch = (
            Path(os.environ["SLURM_TMPDIR"])
            / "dynamo-saturation"
            / self.job_id
            / "rank2"
            / payload["run_id"]
        )
        self.scratch.mkdir(parents=True, exist_ok=False)
        return {"ready_at": now(), "affinity": sorted(os.sched_getaffinity(0))}

    def run_load(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.run_dir is None or self.scratch is None:
            raise CampaignError("AIPerf run_load received before prepare")
        concurrency = int(payload["concurrency"])
        duration = int(payload["duration_seconds"])
        artifact_dir = self.scratch / "load_artifacts"
        artifact_dir.mkdir()
        command = aiperf_command(
            self.config, self.frontend_ip, concurrency, duration, artifact_dir
        )
        env = os.environ | {
            "HF_HOME": self.config["paths"]["hf_home"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        if bool(self.config["runtime"].get("aiperf_compact_chat_result_batch", False)):
            env["AIPERF_COMPACT_CHAT_RESULT_BATCH"] = "1"
        if bool(self.config["runtime"].get("aiperf_native", False)):
            env["AIPERF_STATS_INTERVAL"] = "1"
        if bool(self.config.get("profile", {}).get("aiperf_perf_enabled", False)):
            # Python 3.12's perf trampoline emits per-process symbol maps that
            # let perf resolve Python frames without ptrace sampling.
            env["PYTHONPERFSUPPORT"] = "1"
        self.load = base.ManagedProcess(
            "aiperf", command, self.scratch / "aiperf.log", env=env
        ).start()
        atomic_json(self.run_dir / "aiperf-command.json", {"argv": command})
        if bool(self.config["runtime"].get("aiperf_native", False)):
            tree = wait_for_native_aiperf_process_tree(
                self.load,
                self.run_dir,
                expected_workers=int(
                    self.config["runtime"]["aiperf_native_workers"]
                ),
                timeout=int(
                    self.config["workload"].get(
                        "aiperf_process_tree_timeout_seconds", 900
                    )
                ),
            )
        else:
            tree = base._wait_for_aiperf_process_tree(
                self.load,
                self.run_dir,
                expected_workers=self.config["runtime"]["aiperf_workers_max"],
                expected_record_processors=self.config["runtime"][
                    "aiperf_record_processors"
                ],
                timeout=int(
                    self.config["workload"].get(
                        "aiperf_process_tree_timeout_seconds", 180
                    )
                ),
            )
        if sorted(os.sched_getaffinity(self.load.pid)) != list(range(144)):
            raise CampaignError(
                "AIPerf controller did not inherit the full node CPU mask"
            )
        self.telemetry = base._start_system_telemetry(
            self.run_dir, role="aiperf", tracked={"aiperf": os.getpgid(self.load.pid)}
        )
        samples_path = self.scratch / "aiperf-samples.jsonl"
        self.sampler = base.Sampler(
            samples_path,
            pools={"aiperf_node": "0-143"},
            interface=[str(spec["interface"]) for spec in self.networks.values()],
            interval=1.0,
        )
        self.sampler.start()
        deadline = time.monotonic() + 420
        profiling_start_pattern = re.compile(
            r"Phase profiling(?: \(profiling\))? started", re.I
        )
        native_initialized_pattern = re.compile(
            r'Initialized \d+ phase\(s\): \["warmup", "profiling"\]', re.I
        )
        native_aiperf = bool(
            self.config["runtime"].get("aiperf_native", False)
        )

        def load_log_text() -> str:
            text = self.load.log_path.read_text(errors="replace")
            native_log = artifact_dir / "logs" / "aiperf.log"
            if native_log.is_file():
                text += "\n" + native_log.read_text(errors="replace")
            return text

        while time.monotonic() < deadline:
            self.load.require_alive()
            log_text = load_log_text()
            if profiling_start_pattern.search(log_text):
                break
            if native_aiperf and native_initialized_pattern.search(log_text):
                # Graph mode does not currently attach the phase observer that
                # emits the normal "phase profiling started" marker. The fixed
                # 32-request warmup completes within a few seconds; delay past
                # it before opening the measurement/capture window.
                time.sleep(5)
                self.load.require_alive()
                break
            time.sleep(1)
        else:
            raise CampaignError("AIPerf did not enter profiling")
        profiling_started = now()
        atomic_text(self.run_dir / "PROFILING_STARTED", profiling_started + "\n")
        if bool(self.config.get("profile", {}).get("aiperf_perf_enabled", False)):
            self.capture_aiperf_perf(tree)
        sending_ended: str | None = None
        profiling_ended: str | None = None
        finalizer_deadline: float | None = None
        finalizer_terminated = False
        hard_deadline = (
            time.monotonic()
            + duration
            + self.config["workload"]["grace_period_seconds"]
            + int(
                self.config["workload"].get(
                    "native_finalization_budget_seconds", 300
                )
            )
        )
        phase_prefix = r"Phase profiling(?: \(profiling\))?"
        sending_end_pattern = re.compile(rf"{phase_prefix} sending complete", re.I)
        sending_summary_pattern = re.compile(
            rf"{phase_prefix} sending complete\s*\|\s*"
            r"sent=([0-9,]+),\s*completed=([0-9,]+),\s*in_flight=([0-9,]+)",
            re.I,
        )
        end_pattern = re.compile(
            rf"{phase_prefix} (?:complete|completed|ended|finished)|"
            r"Phase complete: PhaseRecordsStats\(phase=CreditPhase\.PROFILING",
            re.I,
        )
        while self.load.poll() is None:
            if self.shutdown.is_set() or time.monotonic() > hard_deadline:
                raise CampaignError(
                    "AIPerf exceeded its duration/grace/finalization budget"
                )
            text = load_log_text()
            if sending_ended is None and sending_end_pattern.search(text):
                sending_ended = now()
                atomic_text(self.run_dir / "SENDING_ENDED", sending_ended + "\n")
            if profiling_ended is None and end_pattern.search(text):
                profiling_ended = now()
                atomic_text(self.run_dir / "PROFILING_ENDED", profiling_ended + "\n")
                finalizer_deadline = time.monotonic() + int(
                    self.config["workload"].get("finalizer_timeout_seconds", 90)
                )
            # The pinned client can remain alive after all results are exported.
            # Stop only after its shutdown message and durable exports; the full
            # record/count validation below remains the acceptance gate.
            exported = artifact_dir / "profile_export.jsonl"
            if (
                profiling_ended is not None
                and "All results received, initiating shutdown" in text
                and (artifact_dir / "profile_export_aiperf.json").is_file()
                and exported.is_file()
                and time.time() - exported.stat().st_mtime > 10
            ):
                self.load.stop(grace=10)
                finalizer_terminated = True
                break
            if finalizer_deadline is not None and time.monotonic() > finalizer_deadline:
                self.load.stop(grace=10)
                finalizer_terminated = True
                break
            time.sleep(1)
        rc = self.load.wait()
        text = load_log_text()
        if sending_ended is None and sending_end_pattern.search(text):
            sending_ended = now()
            atomic_text(self.run_dir / "SENDING_ENDED", sending_ended + "\n")
        if profiling_ended is None and end_pattern.search(text):
            profiling_ended = now()
            atomic_text(self.run_dir / "PROFILING_ENDED", profiling_ended + "\n")
        if native_aiperf and rc == 0:
            # Native graph mode currently omits the legacy phase-completion
            # console markers. A clean process exit follows its duration and
            # grace handling and is the authoritative terminal signal.
            if sending_ended is None:
                sending_ended = now()
                atomic_text(self.run_dir / "SENDING_ENDED", sending_ended + "\n")
            if profiling_ended is None:
                profiling_ended = sending_ended
                atomic_text(self.run_dir / "PROFILING_ENDED", profiling_ended + "\n")
        if profiling_ended is None:
            raise CampaignError("AIPerf exited without a profiling-complete marker")
        if sending_ended is None:
            raise CampaignError(
                "AIPerf exited without a profiling-sending-complete marker"
            )
        if rc != 0 and not finalizer_terminated:
            raise CampaignError(f"AIPerf exited with status {rc}")
        self.sampler.stop()
        self.sampler = None
        base._stop_processes(self.telemetry)
        self.telemetry = []
        profile_path = artifact_dir / "profile_export.jsonl"
        profile_quiescent = wait_for_file_quiescence(profile_path)
        # Preserve the durable client evidence even when validation below rejects
        # the run. Slurm scratch is removed after allocation teardown.
        shutil.copytree(
            artifact_dir, self.run_dir / "load_artifacts", dirs_exist_ok=True
        )
        shutil.copy2(samples_path, self.run_dir / "aiperf-samples.jsonl")
        shutil.copy2(self.load.log_path, self.run_dir / "aiperf.log")
        native_log = artifact_dir / "logs" / "aiperf.log"
        if native_log.is_file():
            shutil.copy2(native_log, self.run_dir / "aiperf-native.log")
        aiperf_perf_result: dict[str, Any] | None = None
        if self.aiperf_profile_dir is not None:
            aiperf_perf_result = self.process_aiperf_perf()
        if not profile_quiescent:
            raise CampaignError(
                f"AIPerf record export did not quiesce after finalization: {profile_path}"
            )
        terminal = re.findall(r"Phase profiling complete\s*\|\s*completed=([0-9,]+), cancelled=([0-9,]+), errors=([0-9,]+)", text)
        if not terminal:
            raise CampaignError("missing terminal client count for export completeness")
        completed_count, cancelled_count, _ = [int(v.replace(",", "")) for v in terminal[-1]]
        # The pinned client's record export covers completed credits. Credits
        # cancelled at the drain deadline have a separate phase count and do
        # not produce records. Keep that count visible for tail interpretation.
        profile = validate_profile_artifact(profile_path, self.config["workload"], sending_ended=sending_ended, expected_records=completed_count)
        profile["phase_cancelled_requests"] = cancelled_count
        if not finalizer_terminated and not native_aiperf:
            summary = re.search(
                r"Processed [0-9,]+ valid requests and ([0-9,]+) errors|"
                r"completed=[0-9,]+, cancelled=0, errors=([0-9,]+)",
                text,
            )
            if summary is None:
                raise CampaignError("AIPerf lacks a completion summary")
            # AIPerf's phase summary can report zero even when the durable
            # record export contains a negligible HTTP error. Per-record
            # validation above is authoritative.
        overflow = [
            line
            for line in text.splitlines()
            if re.search(
                r"(?:context|token).*(?:overflow|exceed).*(?:skip|drop)", line, re.I
            )
        ]
        if overflow:
            raise CampaignError(
                f"AIPerf reported context-overflow skips: {overflow[:5]}"
            )
        measurement_marker = self.run_dir / "MEASUREMENT_STARTED"
        wait_for(measurement_marker, duration + 120, self.shutdown)
        measurement_started = measurement_marker.read_text().strip()
        sending_summaries = list(sending_summary_pattern.finditer(text))
        if not sending_summaries:
            raise CampaignError("AIPerf lacks a profiling-sending summary")
        sending_summary = sending_summaries[-1]
        sending_cutoff_sent = int(sending_summary.group(1).replace(",", ""))
        sending_cutoff_completed = int(sending_summary.group(2).replace(",", ""))
        sending_cutoff_in_flight = int(sending_summary.group(3).replace(",", ""))
        sending_window_seconds = (
            dt.datetime.fromisoformat(sending_ended)
            - dt.datetime.fromisoformat(measurement_started)
        ).total_seconds()
        if sending_window_seconds <= 0:
            raise CampaignError(
                f"invalid AIPerf sending window: {sending_window_seconds} seconds"
            )
        sending_cutoff_throughput_rps = (
            sending_cutoff_completed / sending_window_seconds
        )
        node = node_busy(
            samples_path, "aiperf_node", measurement_started, sending_ended
        )
        cpu = process_busy(
            self.run_dir / "aiperf-system-telemetry.jsonl",
            "aiperf",
            measurement_started,
            sending_ended,
        )
        special = aiperf_special_process_health(
            self.run_dir / "aiperf-system-telemetry.jsonl",
            measurement_started,
            sending_ended,
        )
        warning_lines = aiperf_warning_lines(text)
        limits = self.config["acceptance"]
        headroom = (
            node["average"] is not None
            and node["p95"] is not None
            and node["average"] < limits["load_node_average_busy_max_percent"]
            and node["p95"] < limits["load_node_p95_busy_max_percent"]
        )
        client_health = (
            len(warning_lines) <= limits["aiperf_repeated_overrun_max_count"]
            and special.get("max_core_fraction", 1.0)
            < limits["aiperf_hot_process_max_core_fraction"]
        )
        result = {
            **profile,
            "profiling_started": profiling_started,
            "measurement_ended": sending_ended,
            "profiling_ended": profiling_ended,
            "sending_cutoff_sent": sending_cutoff_sent,
            "sending_cutoff_completed": sending_cutoff_completed,
            "sending_cutoff_in_flight": sending_cutoff_in_flight,
            "sending_window_seconds": sending_window_seconds,
            "sending_cutoff_throughput_rps": sending_cutoff_throughput_rps,
            "node_busy_percent": node,
            "process_cpu": cpu,
            "special_process_health": special,
            "event_loop_warning_count": len(warning_lines),
            "event_loop_warning_lines": warning_lines[:200],
            "headroom_accepted": headroom,
            "client_health_accepted": client_health,
            "finalizer_terminated_after_records": finalizer_terminated,
            "profile_export_quiescent": profile_quiescent,
            "process_tree": tree,
            # Selection is based on aggregate load-node headroom.  Per-process
            # record saturation and event-loop warnings are retained as
            # diagnostics but do not invalidate an otherwise usable load arm.
            "accepted": headroom
            if limits["aiperf_client_health_is_informational"]
            else headroom and client_health,
        }
        if aiperf_perf_result is not None:
            result["aiperf_perf"] = aiperf_perf_result
        atomic_json(self.run_dir / "aiperf-acceptance.json", result)
        return result

    @staticmethod
    def _aiperf_process_role(cmdline: str) -> str:
        if re.search(r"(?:^| )aiperf worker_[0-9a-f]+(?: |$)", cmdline):
            return "request_workers"
        if re.search(r"(?:^| )aiperf record_processor_[0-9a-f]+(?: |$)", cmdline):
            return "record_processors"
        if re.search(r"(?:^| )aiperf timing_manager(?: |$)", cmdline):
            return "timing_manager"
        if re.search(r"(?:^| )aiperf records_manager(?: |$)", cmdline):
            return "records_manager"
        return "controller_other"

    def _snapshot_aiperf_profile_targets(
        self, pgid: int
    ) -> tuple[list[int], dict[str, Any]]:
        processes: list[dict[str, Any]] = []
        target_pids: list[int] = []
        tid_roles: dict[str, str] = {}
        for entry in sorted(
            (path for path in Path("/proc").iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        ):
            pid = int(entry.name)
            try:
                if os.getpgid(pid) != pgid:
                    continue
                cmdline = (
                    (entry / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode(errors="replace")
                    .strip()
                )
                role = self._aiperf_process_role(cmdline)
                tids = sorted(
                    int(task.name)
                    for task in (entry / "task").iterdir()
                    if task.name.isdigit()
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            target_pids.append(pid)
            for tid in tids:
                tid_roles[str(tid)] = role
            processes.append(
                {
                    "pid": pid,
                    "cmdline": cmdline,
                    "role": role,
                    "tids": tids,
                }
            )
        if not target_pids:
            raise CampaignError(f"no live AIPerf processes found in pgid {pgid}")
        return target_pids, {
            "captured_at": now(),
            "pgid": pgid,
            "processes": processes,
            "target_pid_count": len(target_pids),
            "tid_roles": tid_roles,
        }

    def capture_aiperf_perf(self, tree: Mapping[str, Any]) -> None:
        if self.run_dir is None or self.scratch is None or self.load is None:
            raise CampaignError("AIPerf perf capture started before load setup")
        profile = self.config["profile"]
        deadline = time.monotonic() + int(
            profile.get(
                "aiperf_start_delay_seconds",
                profile["start_delay_seconds"],
            )
        )
        while time.monotonic() < deadline:
            self.load.require_alive()
            if self.shutdown.wait(0.25):
                raise CampaignError("interrupted before AIPerf perf capture")

        profile_dir = self.run_dir / "profiles" / "aiperf"
        kernel_dir = self.run_dir / "profiles" / "aiperf-kernel"
        scratch_dir = self.scratch / "aiperf-profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=False)
        self.aiperf_profile_dir = profile_dir
        self.aiperf_profile_scratch = scratch_dir

        target_pids, manifest = self._snapshot_aiperf_profile_targets(int(tree["pgid"]))
        atomic_json(profile_dir / "process-manifest.json", manifest)
        record_seconds = int(profile["record_seconds"])
        frequency = int(profile.get("aiperf_frequency_hz", profile["frequency_hz"]))
        dwarf_stack_bytes = int(profile.get("aiperf_dwarf_stack_bytes", 16384))
        mmap_pages = int(
            profile.get("aiperf_perf_mmap_pages", profile["perf_mmap_pages"])
        )
        kernel_mmap_pages = int(
            profile.get(
                "aiperf_kernel_perf_mmap_pages",
                profile.get("kernel_perf_mmap_pages", mmap_pages),
            )
        )
        captures = [
            (
                "aiperf",
                [
                    "perf",
                    "record",
                        "--clockid", "mono",
                    "--no-buildid",
                    "-a",
                    "-e",
                    "cycles:u",
                    "-F",
                    str(frequency),
                    "-m",
                    str(mmap_pages),
                    "--call-graph",
                    f"dwarf,{dwarf_stack_bytes}",
                    "-o",
                    str(scratch_dir / "aiperf.data"),
                    "--",
                    "sleep",
                    str(record_seconds),
                ],
                scratch_dir / "aiperf.data",
                scratch_dir / "aiperf-perf.err",
            ),
            (
                "kernel",
                [
                    "perf",
                    "record",
                        "--clockid", "mono",
                    "-a",
                    "-e",
                    "cycles:k",
                    "-F",
                    str(frequency),
                    "-m",
                    str(kernel_mmap_pages),
                    "--call-graph",
                    "fp",
                    "-o",
                    str(scratch_dir / "kernel.data"),
                    "--",
                    "sleep",
                    str(record_seconds),
                ],
                scratch_dir / "kernel.data",
                scratch_dir / "kernel-perf.err",
            ),
        ]
        started_at = now()
        running: list[tuple[str, subprocess.Popen[Any], Any, Path, Path]] = []
        try:
            for name, command, data_path, error_path in captures:
                error = error_path.open("w")
                process = subprocess.Popen(command, stderr=error)
                running.append((name, process, error, data_path, error_path))
            timeout = record_seconds + 300
            for name, process, error, data_path, error_path in running:
                returncode = process.wait(timeout=timeout)
                if returncode != 0:
                    error.flush()
                    shutil.copy2(
                        error_path,
                        profile_dir / f"{name}-capture-failed.err",
                    )
                    raise CampaignError(
                        f"AIPerf {name} perf capture failed: rc={returncode}"
                    )
                if not data_path.is_file() or data_path.stat().st_size == 0:
                    raise CampaignError(f"AIPerf {name} perf capture produced no data")
        finally:
            for _, process, error, _, _ in running:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)
                error.close()

        maps_dir = scratch_dir / "python-perf-maps"
        maps_dir.mkdir()
        copied_maps: list[str] = []
        for pid in target_pids:
            source = Path(f"/tmp/perf-{pid}.map")
            if source.is_file():
                destination = maps_dir / source.name
                shutil.copy2(source, destination)
                copied_maps.append(source.name)
        if not copied_maps:
            raise CampaignError(
                "PYTHONPERFSUPPORT=1 produced no Python perf symbol maps"
            )

        atomic_json(
            profile_dir / "capture.json",
            {
                "started_at": started_at,
                "ended_at": now(),
                "frequency_hz": frequency,
                "dwarf_stack_bytes": dwarf_stack_bytes,
                "record_seconds": record_seconds,
                "target_pid_count": len(target_pids),
                "python_perf_map_count": len(copied_maps),
                "python_perf_maps": copied_maps,
            },
        )

    def process_aiperf_perf(self) -> dict[str, Any]:
        if (
            self.run_dir is None
            or self.aiperf_profile_dir is None
            or self.aiperf_profile_scratch is None
        ):
            raise CampaignError("AIPerf perf processing lacks a run directory")
        profile_dir = self.aiperf_profile_dir
        scratch_dir = self.aiperf_profile_scratch
        kernel_dir = self.run_dir / "profiles" / "aiperf-kernel"
        timeout = int(self.config["profile"]["perf_processing_timeout_seconds"])
        maps_dir = profile_dir / "python-perf-maps"
        maps_dir.mkdir(exist_ok=False)
        shutil.copy2(scratch_dir / "aiperf.data", profile_dir / "aiperf.data")
        shutil.copy2(scratch_dir / "aiperf-perf.err", profile_dir / "perf.err")
        shutil.copy2(scratch_dir / "kernel.data", kernel_dir / "kernel.data")
        shutil.copy2(scratch_dir / "kernel-perf.err", kernel_dir / "perf.err")
        for source in (scratch_dir / "python-perf-maps").glob("perf-*.map"):
            shutil.copy2(source, maps_dir / source.name)
        # CPython may remove its live /tmp maps during interpreter shutdown.
        # Restore captured copies under the original names for perf script, on
        # the same compute node where the profile was recorded.
        restored_maps: list[Path] = []
        for source in (profile_dir / "python-perf-maps").glob("perf-*.map"):
            destination = Path("/tmp") / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
                restored_maps.append(destination)
        commands = [
            (
                profile_dir,
                [
                    "perf",
                    "script",
                    "--no-inline",
                    "--show-lost-events",
                    "-i",
                    str(profile_dir / "aiperf.data"),
                ],
                "perf-script.txt",
                "perf-script.err",
            ),
            (
                profile_dir,
                ["perf", "buildid-list", "-i", str(profile_dir / "aiperf.data")],
                "perf-buildids.txt",
                "perf-buildids.err",
            ),
            (
                profile_dir,
                [
                    "perf",
                    "report",
                    "--stdio",
                    "--header-only",
                    "-i",
                    str(profile_dir / "aiperf.data"),
                ],
                "perf-header.txt",
                "perf-header.err",
            ),
            (
                kernel_dir,
                [
                    "perf",
                    "script",
                    "--show-lost-events",
                    "-i",
                    str(kernel_dir / "kernel.data"),
                ],
                "perf-script.txt",
                "perf-script.err",
            ),
            (
                kernel_dir,
                [
                    "perf",
                    "report",
                    "--stdio",
                    "--sort",
                    "comm,dso,symbol",
                    "--percent-limit",
                    "0.1",
                    "-i",
                    str(kernel_dir / "kernel.data"),
                ],
                "perf-report.txt",
                "perf-report.err",
            ),
        ]
        for directory, command, stdout_name, stderr_name in commands:
            with (
                (directory / stdout_name).open("w") as stdout,
                (directory / stderr_name).open("w") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                )
            if completed.returncode != 0:
                raise CampaignError(
                    f"AIPerf perf processing failed: {' '.join(command)}"
                )

        def quality(directory: Path) -> dict[str, Any]:
            script = (directory / "perf-script.txt").read_text(errors="replace")
            lost_lines = [
                line for line in script.splitlines() if "PERF_RECORD_LOST" in line
            ]
            warning_lines = [
                line
                for path in (directory / "perf.err", directory / "perf-script.err")
                for line in path.read_text(errors="replace").splitlines()
                if re.search(r"\blost\b", line, flags=re.IGNORECASE)
            ]
            frame_count = sum(
                bool(
                    re.match(
                        r"^\s*[0-9a-fA-F]+\s+.+\s+\([^)]*\)\s*$",
                        line,
                    )
                )
                for line in script.splitlines()
            )
            unresolved = script.count("[unknown]")
            return {
                "accepted": bool(script.strip())
                and not lost_lines
                and not warning_lines,
                "frame_count": frame_count,
                "lost_sample_count": len(lost_lines),
                "unresolved_frame_count": unresolved,
                "unresolved_frame_fraction": (
                    unresolved / frame_count if frame_count else None
                ),
                "warning_lines": warning_lines,
            }

        user_quality = quality(profile_dir)
        kernel_quality = quality(kernel_dir)
        if not user_quality["accepted"] or not kernel_quality["accepted"]:
            raise CampaignError(
                "AIPerf perf quality failed: "
                f"user={user_quality}, kernel={kernel_quality}"
            )
        atomic_json(profile_dir / "perf-capture.json", user_quality)
        atomic_json(kernel_dir / "perf-capture.json", kernel_quality)
        analysis_dir = self.run_dir / "aiperf-perf-analysis"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("analyze_aiperf_profile.py")),
                "--profile-dir",
                str(profile_dir),
                "--output-dir",
                str(analysis_dir),
                "--flamegraph",
                str(self.config["paths"]["flamegraph"]),
            ],
            check=True,
            timeout=timeout,
        )
        for path in restored_maps:
            path.unlink(missing_ok=True)
        return {
            "accepted": True,
            "user": user_quality,
            "kernel": kernel_quality,
            "analysis_dir": str(analysis_dir),
        }

    def cleanup(self) -> None:
        if self.sampler is not None:
            self.sampler.stop(raise_on_error=False)
            self.sampler = None
        try:
            base._stop_processes(self.telemetry)
        finally:
            self.telemetry = []
        if self.load is not None:
            self.load.stop()
            self.load = None
        if self.scratch is not None:
            shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch = None
        self.run_dir = None
        self.aiperf_profile_dir = None
        self.aiperf_profile_scratch = None


class Coordinator:
    def __init__(
        self,
        config: Mapping[str, Any],
        mocker_channel: Channel,
        aiperf_channel: Channel,
        shutdown: threading.Event,
        *,
        job_id: str,
        result_dir: Path,
        advertised_ip: str,
        mocker_ip: str,
        interface: str,
        networks: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.mocker_channel = mocker_channel
        self.aiperf_channel = aiperf_channel
        self.shutdown = shutdown
        self.job_id = job_id
        self.result_dir = result_dir
        self.advertised_ip = advertised_ip
        self.mocker_ip = mocker_ip
        self.interface = interface
        self.networks = (
            {name: dict(spec) for name, spec in networks.items()}
            if networks is not None
            else {
                "primary": {
                    "interface": interface,
                    "address": advertised_ip,
                    "numa_node": None,
                }
            }
        )
        self.frontend: base.ManagedProcess | None = None
        self.frontend_secondary: base.ManagedProcess | None = None
        self.telemetry: list[base.ManagedProcess] = []
        self.sampler: base.Sampler | None = None
        self.scratch: Path | None = None
        self.run_dir: Path | None = None
        self.frontend_networks: dict[str, dict[str, Any]] = {}

    def run(self) -> None:
        try:
            fixed = self.config.get("campaign", {}).get("fixed_concurrency")
            if fixed is not None:
                result = self.run_one(
                    int(fixed),
                    profile=bool(self.config.get("campaign", {}).get("profile", False)),
                )
                self.write_artifact_index()
                marker = (
                    "COMPLETE" if result["accepted"] else "COMPLETE_WITH_REJECTED_ARM"
                )
                (self.result_dir / marker).write_text(now() + "\n")
                return
            selected = self.select_saturation()
            if selected is None:
                (self.result_dir / "NO_SATURATION_POINT").write_text(now() + "\n")
                return
            final_result = self.run_one(selected, profile=True)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_profile.py")),
                    "--job-dir",
                    str(self.result_dir),
                ],
                check=True,
                timeout=1800,
                stdout=(self.result_dir / "analysis.log").open("w"),
                stderr=subprocess.STDOUT,
            )
            self.write_artifact_index()
            marker = (
                "COMPLETE"
                if final_result["accepted"]
                else "COMPLETE_WITH_INVALID_PROFILE"
            )
            (self.result_dir / marker).write_text(now() + "\n")
        finally:
            self.stop_frontend()
            for channel in (self.mocker_channel, self.aiperf_channel):
                try:
                    channel.send("terminate", {}, 60)
                except Exception:
                    if not self.shutdown.is_set():
                        traceback.print_exc()

    def select_saturation(self) -> int | None:
        saturation = self.config["saturation"]
        queue = list(saturation["initial_candidates"])
        tested: dict[int, dict[str, Any]] = {}
        seed_lower = saturation.get("seed_lower")
        lower: tuple[int, float] | None = (
            (int(seed_lower["concurrency"]), float(seed_lower["frontend_mean_cores"]))
            if seed_lower is not None
            else None
        )
        upper: tuple[int, float] | None = None
        while queue:
            concurrency = int(queue.pop(0))
            if concurrency in tested:
                continue
            result = self.run_one(concurrency, profile=False)
            tested[concurrency] = result
            mean = float(result["frontend_cpu"]["average_occupied_cores"] or 0.0)
            p95 = float(result["frontend_cpu"]["p95_occupied_cores"] or 0.0)
            qualified = (
                saturation["frontend_mean_cores_min"]
                <= mean
                <= saturation["frontend_mean_cores_max"]
                and p95 >= saturation["frontend_p95_cores_min"]
                and result["mocker"]["accepted"]
                and result["aiperf"]["accepted"]
            )
            atomic_json(
                self.result_dir / "saturation-progress.json",
                {
                    "seed_lower": seed_lower,
                    "tested": tested,
                    "latest": concurrency,
                    "qualified": qualified,
                    "updated_at": now(),
                },
            )
            if qualified:
                atomic_json(
                    self.result_dir / "saturation-selection.json",
                    {
                        "selected_concurrency": concurrency,
                        "seed_lower": seed_lower,
                        "tested": tested,
                        "selected_at": now(),
                    },
                )
                return concurrency
            if mean < saturation["frontend_mean_cores_min"]:
                lower = (concurrency, mean)
            elif mean > saturation["frontend_mean_cores_max"]:
                upper = (concurrency, mean)
            if lower and upper and lower[0] < upper[0]:
                step = int(saturation["refinement_step"])
                midpoint = int(round(((lower[0] + upper[0]) / 2) / step) * step)
                if midpoint not in tested and lower[0] < midpoint < upper[0]:
                    queue.insert(0, midpoint)
                    continue
            if concurrency == max(saturation["initial_candidates"]):
                break
        atomic_json(
            self.result_dir / "saturation-selection.json",
            {
                "selected_concurrency": None,
                "seed_lower": seed_lower,
                "tested": tested,
                "completed_at": now(),
            },
        )
        return None

    def require_frontends_alive(self) -> None:
        if self.frontend is None:
            raise CampaignError("frontend is not running")
        self.frontend.require_alive()
        if self.frontend_secondary is not None:
            self.frontend_secondary.require_alive()

    def run_one(self, concurrency: int, *, profile: bool) -> dict[str, Any]:
        perf_enabled = profile and bool(
            self.config.get("profile", {}).get("perf_enabled", True)
        )
        offcpu_enabled = profile and bool(
            self.config.get("profile", {}).get("offcpu_enabled", False)
        )
        kind = "profile" if profile else "calibration"
        run_id = f"{kind}-c{concurrency}"
        campaign = self.config.get("campaign", {})
        if campaign.get("fixed_concurrency") is not None:
            run_dir = self.result_dir / "ablations" / str(campaign["name"])
        else:
            run_dir = (
                self.result_dir / "profiles" / "agentx" / "numa0"
                if profile
                else self.result_dir / "calibration" / f"c{concurrency}"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        duration = (
            int(self.config["profile"]["duration_seconds"])
            if profile
            else int(
                self.config["saturation"]["settle_seconds"]
                + self.config["saturation"]["measurement_seconds"]
            )
        )
        manifest = {
            "run_id": run_id,
            "kind": kind,
            "concurrency": concurrency,
            "duration_seconds": duration,
            "started_at": now(),
            "config": self.config,
        }
        atomic_json(run_dir / "run.json", manifest)
        self.mocker_channel.send(
            "prepare", {"run_dir": str(run_dir), "run_id": run_id}, 900
        )
        self.aiperf_channel.send(
            "prepare", {"run_dir": str(run_dir), "run_id": run_id}, 120
        )
        self.start_frontend(run_dir, run_id)
        self.wait_for_frontend_kv_sources(run_dir)
        broker_kv_baseline = (
            self.capture_broker_kv_baseline(run_dir)
            if bool(self.config["runtime"].get("zmq_broker_enabled", False))
            else None
        )
        smoke = self.smoke(run_dir)
        if self.config['campaign'].get('smoke_only', False):
            return {'accepted':True, 'smoke_only':True}
        if broker_kv_baseline is not None:
            self.wait_for_broker_kv_events(run_dir, broker_kv_baseline)
        self.mocker_channel.send("start_monitor", {}, 60)
        self.start_frontend_monitor(run_dir)
        load_result: dict[str, Any] = {}
        load_error: list[BaseException] = []
        load_timeout = (
            duration
            + self.config["workload"]["grace_period_seconds"]
            + int(self.config["workload"].get("finalizer_timeout_seconds", 1200)) + 120
            + (
                int(self.config["profile"]["perf_processing_timeout_seconds"])
                if bool(
                    self.config.get("profile", {}).get("aiperf_perf_enabled", False)
                )
                else 0
            )
        )

        def run_load() -> None:
            try:
                load_result.update(
                    self.aiperf_channel.send(
                        "run_load",
                        {
                            "concurrency": concurrency,
                            "duration_seconds": duration,
                        },
                        load_timeout,
                    )
                )
            except BaseException as error:
                load_error.append(error)

        load_thread = threading.Thread(target=run_load, daemon=True)
        load_thread.start()
        wait_for(run_dir / "PROFILING_STARTED", 600, self.shutdown)
        profiling_started = (run_dir / "PROFILING_STARTED").read_text().strip()
        settle_deadline = time.monotonic() + int(self.config['saturation']['settle_seconds'])
        while time.monotonic() < settle_deadline:
            self.require_frontends_alive()
            if self.shutdown.wait(0.25):
                raise CampaignError('interrupted during ramp')
        measurement_started = now()
        atomic_text(run_dir / "MEASUREMENT_STARTED", measurement_started + "\n")
        if perf_enabled:
            self.capture_perf(run_dir, load_thread)
        if offcpu_enabled:
            self.capture_offcpu_perf(run_dir, load_thread)
        load_thread.join(timeout=load_timeout)
        if load_thread.is_alive():
            raise CampaignError("AIPerf load thread did not finish")
        if load_error:
            raise load_error[0]
        profiling_ended = (run_dir / "PROFILING_ENDED").read_text().strip()
        measurement_ended = str(load_result["measurement_ended"])
        atomic_text(run_dir / "MEASUREMENT_ENDED", measurement_ended + "\n")
        self.finish_frontend_monitor()
        dual_frontend = bool(self.config["runtime"].get("frontend_dual_numa", False))
        if dual_frontend:
            frontend_cpu = combined_process_busy(
                run_dir / "frontend-system-telemetry.jsonl",
                ("frontend-numa0", "frontend-numa1"),
                measurement_started,
                measurement_ended,
            )
            frontend_cpu_by_process = frontend_cpu.pop("by_process")
        else:
            frontend_cpu = process_busy(
                run_dir / "frontend-system-telemetry.jsonl",
                "frontend",
                measurement_started,
                measurement_ended,
            )
            frontend_cpu_by_process = {"frontend": dict(frontend_cpu)}
        frontend_domain_cpu = {
            "numa0": node_busy(
                run_dir / "frontend-samples.jsonl",
                "numa0",
                measurement_started,
                measurement_ended,
            ),
            "numa1": node_busy(
                run_dir / "frontend-samples.jsonl",
                "numa1",
                measurement_started,
                measurement_ended,
            ),
        }
        mocker_result = self.mocker_channel.send(
            "finish_monitor",
            {
                "measurement_started": measurement_started,
                "measurement_ended": measurement_ended,
            },
            120,
        )
        if perf_enabled:
            mocker_result["perf"] = self.mocker_channel.send(
                "finish_perf", {}, 600
            )
        self.stop_frontend()
        atomic_text(run_dir / "FRONTENDS_STOPPED", now() + "\n")
        self.mocker_channel.send("stop", {}, 120)
        self.aiperf_channel.send("stop", {}, 120)
        warning_analysis_path = run_dir / "direct-zmq-warning-analysis.json"
        with warning_analysis_path.open("w") as output:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_active_sequence_warnings.py")),
                    str(run_dir),
                ],
                stdout=output,
                check=True,
            )
        warning_analysis = json.loads(warning_analysis_path.read_text())
        if not warning_analysis["coverage"][
            "frontend_log_capture_covers_full_grace"
        ]:
            raise CampaignError("frontend logs do not cover the full AIPerf grace period")
        warning_gate_accepted = (
            int(warning_analysis["lane_full"]["measurement_and_grace_count"]) == 0
        )
        # CPU samples stop at SENDING_ENDED, so use completions observed at that
        # same cutoff. Durable records can continue to arrive during grace.
        completed = int(load_result.get("sending_cutoff_completed", 0))
        frontend_cpu["cpu_ms_per_completed_request"] = (
            1000.0 * frontend_cpu["cpu_seconds"] / completed if completed else None
        )
        for item in frontend_cpu_by_process.values():
            item["cpu_ms_per_all_completed_requests"] = (
                1000.0 * item["cpu_seconds"] / completed if completed else None
            )
        saturation = self.config["saturation"]
        frontend_saturation_accepted = (
            saturation["frontend_mean_cores_min"]
            <= float(frontend_cpu["average_occupied_cores"] or 0.0)
            <= saturation["frontend_mean_cores_max"]
            and float(frontend_cpu["p95_occupied_cores"] or 0.0)
            >= saturation["frontend_p95_cores_min"]
        )
        require_saturation = bool(
            self.config["profile"].get("require_saturation", True)
        )
        result = {
            "accepted": (
                bool(smoke["accepted"])
                and bool(mocker_result["accepted"])
                and bool(load_result["accepted"])
                and warning_gate_accepted
                and (
                    not profile
                    or not require_saturation
                    or frontend_saturation_accepted
                )
            ),
            "concurrency": concurrency,
            "profile": profile,
            "perf_enabled": perf_enabled,
            "measurement_started": measurement_started,
            "measurement_ended": measurement_ended,
            "sending_cutoff_completed": completed,
            "throughput_rps": load_result.get("sending_cutoff_throughput_rps"),
            "direct_zmq_warning_analysis": warning_analysis,
            "direct_zmq_warning_gate_accepted": warning_gate_accepted,
            "frontend_cpu": frontend_cpu,
            "frontend_cpu_by_process": frontend_cpu_by_process,
            "frontend_domain_cpu": frontend_domain_cpu,
            "frontend_saturation_accepted": frontend_saturation_accepted,
            "profile_saturation_required": require_saturation,
            "mocker": mocker_result,
            "aiperf": load_result,
        }
        atomic_json(run_dir / "campaign-result.json", result)
        if perf_enabled:
            self.process_perf(run_dir)
        (run_dir / "COMPLETE").write_text(now() + "\n")
        return result

    def start_frontend(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.scratch = (
            Path(os.environ["SLURM_TMPDIR"])
            / "dynamo-saturation"
            / self.job_id
            / "rank0"
            / run_id
        )
        self.scratch.mkdir(parents=True, exist_ok=False)
        dual_frontend = bool(self.config["runtime"].get("frontend_dual_numa", False))
        if dual_frontend:
            if {"numa0", "numa1"} <= set(self.networks):
                frontend_networks = {
                    "numa0": select_network(self.networks, "numa0"),
                    "numa1": select_network(self.networks, "numa1"),
                }
            else:
                selected = select_network(self.networks, None)
                frontend_networks = {
                    "numa0": dict(selected),
                    "numa1": dict(selected),
                }
        else:
            selected = select_network(
                self.networks, self.config["topology"]["frontend"].get("network")
            )
            frontend_networks = {"frontend": selected}
        frontend_envs: dict[str, dict[str, str]] = {}
        for name, network in frontend_networks.items():
            frontend_envs[name] = base._base_runtime_env(
                self.config,
                str(network["address"]),
                self.mocker_ip,
                str(network["interface"]),
            )
            if frontend_envs[name].get("DYN_VELO_RESPONSE_TRANSPORT") == "ucx":
                node = "1" if name == "numa1" else "0"
                frontend_envs[name]["UCX_NET_DEVICES"] = self.config["network"]["ucx_numa_devices"][node]
            frontend_envs[name]["DYN_RUNTIME_NUM_WORKER_THREADS"] = "72"
            if self.config["runtime"].get("zmq_max_sockets") is not None:
                frontend_envs[name]["DYN_ZMQ_MAX_SOCKETS"] = str(
                    self.config["runtime"]["zmq_max_sockets"]
                )
            response_stream_interface = self.config.get("network", {}).get(
                "response_stream_interface"
            )
            if response_stream_interface:
                response_stream_ip = interface_ipv4(str(response_stream_interface))
                frontend_envs[name]["DYN_RESPONSE_STREAM_HOST"] = response_stream_ip
                frontend_envs[name]["DYN_TCP_RESPONSE_STREAM_HOST"] = response_stream_ip
            if self.config["runtime"].get("tcp_response_batch_interval_ms") is not None:
                frontend_envs[name]["DYN_TCP_RESPONSE_BATCH_INTERVAL_MS"] = str(
                    self.config["runtime"].get("tcp_response_batch_interval_ms", 5)
                )
            if bool(self.config["runtime"].get("response_packet_metrics", False)):
                frontend_envs[name]["DYN_TCP_RESPONSE_PACKET_METRICS"] = "1"
            if bool(
                self.config["runtime"].get(
                    "axum_connection_local_shutdown", False
                )
            ):
                frontend_envs[name]["DYN_AXUM_CONNECTION_LOCAL_SHUTDOWN"] = "1"
            if self.config["runtime"].get("tcp_channel_buffer") is not None:
                frontend_envs[name]["DYN_TCP_CHANNEL_BUFFER"] = str(
                    self.config["runtime"]["tcp_channel_buffer"]
                )
                frontend_envs[name]["DYN_TCP_POOL_SIZE"] = str(
                    self.config["runtime"]["tcp_pool_size"]
                )
            if self.config["runtime"].get("quic_response_lanes") is not None:
                frontend_envs[name]["DYN_QUIC_RESPONSE_CONNECTIONS"] = str(
                    self.config["runtime"].get("quic_response_connections", 1)
                )
                frontend_envs[name]["DYN_QUIC_RESPONSE_LANES"] = str(
                    self.config["runtime"]["quic_response_lanes"]
                )
                frontend_envs[name]["DYN_QUIC_RESPONSE_BATCH_INTERVAL_US"] = str(
                    self.config["runtime"].get("quic_response_batch_interval_us", 1000)
                )
            if bool(
                self.config["runtime"].get("frontend_jemalloc_enabled", True)
            ):
                frontend_envs[name]["LD_PRELOAD"] = self.config["paths"][
                    "frontend_jemalloc"
                ]
        placement = self.config["topology"]["frontend"]
        secondary_placement = self.config["topology"]["frontend_control"]
        frontend_argv = [
            self.config["paths"]["dynamo_python"],
            "-m",
            "dynamo.frontend",
            "--response-plane", str(self.config["runtime"]["response_plane"]),
            "--router-mode",
            str(self.config["runtime"].get("router_mode", "kv")),
            "--kv-cache-block-size",
            str(self.config["workload"]["block_size"]),
            "--http-host",
            str(next(iter(frontend_networks.values()))["address"]),
            "--http-port",
            str(self.config["ports"]["frontend_http"]),
            "--migration-limit",
            "0",
            "--active-decode-blocks-threshold",
            "None",
            "--active-prefill-tokens-threshold",
            "None",
            "--active-prefill-tokens-threshold-frac",
            "None",
        ]
        if bool(self.config["runtime"].get("router_replica_sync", False)):
            frontend_argv.append("--router-replica-sync")
        # The coordinator is intentionally isolated to the other NUMA domain.
        # numactl validates --physcpubind against the affinity mask inherited by
        # its own process, so let the launcher child inherit the allocation-wide
        # mask and immediately constrain itself to the configured domain. Restore the
        # coordinator before waiting for frontend readiness.
        coordinator_affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, range(144))
            if dual_frontend:
                primary_command = base.process_prefix(
                    str(placement["cpus"]), str(placement["memory"])
                ) + list(frontend_argv)
                primary_command[primary_command.index("--http-host") + 1] = str(
                    frontend_networks["numa0"]["address"]
                )
                primary_command[primary_command.index("--http-port") + 1] = "8001"
                secondary_command = base.process_prefix(
                    str(secondary_placement["cpus"]),
                    str(secondary_placement["memory"]),
                ) + list(frontend_argv)
                secondary_command[secondary_command.index("--http-host") + 1] = str(
                    frontend_networks["numa1"]["address"]
                )
                secondary_command[secondary_command.index("--http-port") + 1] = "8002"
                self.frontend = base.ManagedProcess(
                    "frontend-numa0",
                    primary_command,
                    self.scratch / "frontend-numa0.log",
                    env=frontend_envs["numa0"],
                ).start()
                self.frontend_secondary = base.ManagedProcess(
                    "frontend-numa1",
                    secondary_command,
                    self.scratch / "frontend-numa1.log",
                    env=frontend_envs["numa1"],
                ).start()
                commands = {"numa0": primary_command, "numa1": secondary_command}
            else:
                command = base.process_prefix(
                    str(placement["cpus"]), str(placement["memory"])
                ) + list(frontend_argv)
                self.frontend = base.ManagedProcess(
                    "frontend",
                    command,
                    self.scratch / "frontend.log",
                    env=frontend_envs["frontend"],
                ).start()
                commands = {"frontend": command}
        finally:
            os.sched_setaffinity(0, coordinator_affinity)
        assert self.frontend is not None
        if self.frontend_secondary is not None:
            base._wait_http(
                f"http://{frontend_networks['numa0']['address']}:8001/v1/models",
                self.config["model"]["name"],
                self.frontend,
                timeout=180,
            )
            base._wait_http(
                f"http://{frontend_networks['numa1']['address']}:8002/v1/models",
                self.config["model"]["name"],
                self.frontend_secondary,
                timeout=180,
            )
            affinities = {
                "numa0": base._validate_process_affinity(
                    self.frontend, str(placement["cpus"])
                ),
                "numa1": base._validate_process_affinity(
                    self.frontend_secondary, str(secondary_placement["cpus"])
                ),
            }
        else:
            base._wait_http(
                f"http://{frontend_networks['frontend']['address']}:"
                f"{self.config['ports']['frontend_http']}/v1/models",
                self.config["model"]["name"],
                self.frontend,
                timeout=180,
            )
            affinities = {
                "frontend": base._validate_process_affinity(
                    self.frontend, str(placement["cpus"])
                )
            }
        if not all(bool(item["valid"]) for item in affinities.values()):
            raise CampaignError(f"frontend affinity failed: {affinities}")
        effective={}
        for label, proc in [('numa0', self.frontend), ('numa1', self.frontend_secondary)]:
            if proc is None: continue
            raw=Path(f'/proc/{proc.pid}/environ').read_bytes().split(b'\0')
            env=dict(item.decode().split('=',1) for item in raw if b'=' in item)
            unset=['RAYON_NUM_THREADS','RAYON_RS_NUM_THREADS','TOKENIZERS_PARALLELISM','FASTOKENS_BPE_THREADS']
            assert all(k not in env for k in unset),env.keys()
            effective[label]={'environment':{k:v for k,v in env.items() if k.startswith(('DYN_','OTEL_'))},
                'thread_overrides_absent':unset, 'threads':len(list(Path(f'/proc/{proc.pid}/task').iterdir()))}
            (run_dir/f'{label}-numa-maps.txt').write_text(Path(f'/proc/{proc.pid}/numa_maps').read_text())
        atomic_json(run_dir/'effective-process-settings.json',effective)
        atomic_json(run_dir / "frontend-commands.json", commands)
        atomic_json(
            run_dir / "frontend-pid.json",
            {
                "pid": self.frontend.pid,
                "pgid": os.getpgid(self.frontend.pid),
                "argv": self.frontend.command,
                "primary": {
                    "pid": self.frontend.pid,
                    "pgid": os.getpgid(self.frontend.pid),
                    "argv": self.frontend.command,
                },
                "secondary": (
                    {
                        "pid": self.frontend_secondary.pid,
                        "pgid": os.getpgid(self.frontend_secondary.pid),
                        "argv": self.frontend_secondary.command,
                    }
                    if self.frontend_secondary is not None
                    else None
                ),
                "affinity": affinities,
            },
        )
        self.frontend_networks = frontend_networks
        environment_keys = (
            "DYN_RESPONSE_PLANE", "DYN_RUNTIME_NUM_WORKER_THREADS",
            "DYN_QUIC_RESPONSE_BUFFER_CAPACITY",
            "RAYON_NUM_THREADS", "RAYON_RS_NUM_THREADS", "TOKENIZERS_PARALLELISM", "FASTOKENS_BPE_THREADS",
            "ETCD_ENDPOINTS",
            "DYN_TCP_RPC_HOST",
            "DYN_TCP_RESPONSE_STREAM_HOST",
            "DYN_REQUEST_PLANE",
            "DYN_REQUEST_PLANE_CODEC",
            "DYN_EVENT_PLANE",
            "DYN_EVENT_PLANE_CODEC",
            "DYN_EVENT_PLANE_HOST",
            "DYN_TCP_RESPONSE_MUX",
            "DYN_TCP_RESPONSE_BATCH_INTERVAL_MS",
            "DYN_AXUM_CONNECTION_LOCAL_SHUTDOWN",
            "DYN_TCP_CHANNEL_BUFFER",
            "DYN_TCP_POOL_SIZE",
            "DYN_QUIC_RESPONSE_CONNECTIONS",
            "DYN_QUIC_RESPONSE_LANES",
            "DYN_QUIC_RESPONSE_BATCH_INTERVAL_US",
            "DYN_ZMQ_BROKER_ENABLED",
            "DYN_ZMQ_BROKER_URL",
            "DYN_ZMQ_MAX_SOCKETS",
            "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB",
            "DYN_TOKENIZER",
            "DYN_TOKENIZER_CACHE",
            "DYN_TOKENIZER_CACHE_BYTES",
            "HF_HOME",
            "LD_PRELOAD",
        )
        atomic_json(
            run_dir / "frontend-environment.json",
            {
                name: {key: env[key] for key in environment_keys if key in env}
                for name, env in frontend_envs.items()
            },
        )
        atomic_json(run_dir / "frontend-network-bindings.json", frontend_networks)

    def wait_for_frontend_kv_sources(self, run_dir: Path) -> None:
        timeout_value = self.config["workload"].get(
            "frontend_kv_source_ready_timeout_seconds"
        )
        if timeout_value is None:
            return

        timeout_seconds = int(timeout_value)
        if timeout_seconds <= 0:
            raise CampaignError(
                "frontend_kv_source_ready_timeout_seconds must be positive"
            )
        expected = int(self.config["runtime"]["num_mockers"])
        expected_groups = self.config["runtime"].get(
            "expected_kv_zmq_socket_groups"
        )
        if expected_groups is not None:
            expected_groups = int(expected_groups)
        if self.frontend_secondary is not None:
            endpoints = {
                "numa0": (
                    str(self.frontend_networks["numa0"]["address"]),
                    8001,
                ),
                "numa1": (
                    str(self.frontend_networks["numa1"]["address"]),
                    8002,
                ),
            }
        else:
            endpoints = {
                "frontend": (
                    str(self.frontend_networks["frontend"]["address"]),
                    int(self.config["ports"]["frontend_http"]),
                )
            }

        samples_path = run_dir / "frontend-kv-source-readiness.jsonl"
        started = time.monotonic()
        deadline = started + timeout_seconds
        while True:
            self.require_frontends_alive()
            sample: dict[str, Any] = {
                "timestamp": now(),
                "elapsed_seconds": time.monotonic() - started,
                "metric": KV_ZMQ_ACTIVE_SOURCES_METRIC,
                "expected_active_sources": expected,
                "frontends": {},
            }
            for name, (address, port) in endpoints.items():
                url = f"http://{address}:{port}/metrics"
                observed: dict[str, Any] = {"url": url}
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        payload = response.read().decode("utf-8", errors="replace")
                    observed["active_sources"] = parse_kv_zmq_active_sources(payload)
                    observed["event_socket_groups"] = parse_stream_gauge(
                        payload, KV_ZMQ_SOCKET_GROUPS_METRIC, "events"
                    )
                    observed["metric_socket_groups"] = parse_stream_gauge(
                        payload, KV_ZMQ_SOCKET_GROUPS_METRIC, "metrics"
                    )
                    observed["event_connected_endpoints"] = parse_stream_gauge(
                        payload, KV_ZMQ_CONNECTED_ENDPOINTS_METRIC, "events"
                    )
                    observed["metric_connected_endpoints"] = parse_stream_gauge(
                        payload, KV_ZMQ_CONNECTED_ENDPOINTS_METRIC, "metrics"
                    )
                except Exception as error:
                    observed["active_sources"] = None
                    observed["event_socket_groups"] = None
                    observed["metric_socket_groups"] = None
                    observed["event_connected_endpoints"] = None
                    observed["metric_connected_endpoints"] = None
                    observed["error"] = f"{type(error).__name__}: {error}"
                sample["frontends"][name] = observed

            with samples_path.open("a") as output:
                output.write(json.dumps(sample, sort_keys=True) + "\n")

            if all(
                item["active_sources"] == expected
                and (
                    expected_groups is None
                    or (
                        item["event_socket_groups"] == expected_groups
                        and item["metric_socket_groups"] == expected_groups
                        and item["event_connected_endpoints"] == expected
                        and item["metric_connected_endpoints"] == expected
                    )
                )
                for item in sample["frontends"].values()
            ):
                sample["status"] = "ready"
                sample["ready_at"] = now()
                atomic_json(run_dir / "frontend-kv-source-readiness.json", sample)
                return

            if time.monotonic() >= deadline:
                sample["status"] = "timeout"
                atomic_json(run_dir / "frontend-kv-source-readiness.json", sample)
                raise CampaignError(
                    "frontends did not simultaneously reach "
                    f"{expected} active KV ZMQ sources within {timeout_seconds} seconds: "
                    f"{sample['frontends']}"
                )
            if self.shutdown.wait(1):
                raise CampaignError("interrupted while waiting for KV ZMQ sources")

    def frontend_metric_endpoints(self) -> dict[str, tuple[str, int]]:
        if self.frontend_secondary is not None:
            return {
                "numa0": (str(self.frontend_networks["numa0"]["address"]), 8001),
                "numa1": (str(self.frontend_networks["numa1"]["address"]), 8002),
            }
        return {
            "frontend": (
                str(self.frontend_networks["frontend"]["address"]),
                int(self.config["ports"]["frontend_http"]),
            )
        }

    def scrape_kv_events_applied(self) -> dict[str, dict[str, Any]]:
        observed: dict[str, dict[str, Any]] = {}
        for name, (address, port) in self.frontend_metric_endpoints().items():
            url = f"http://{address}:{port}/metrics"
            item: dict[str, Any] = {"url": url}
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                item["successful_applications"] = parse_kv_cache_events_applied(
                    payload
                )
            except Exception as error:
                item["successful_applications"] = None
                item["error"] = f"{type(error).__name__}: {error}"
            observed[name] = item
        return observed

    def capture_broker_kv_baseline(
        self, run_dir: Path
    ) -> dict[str, dict[str, Any]]:
        observed = self.scrape_kv_events_applied()
        if any(
            item["successful_applications"] is None for item in observed.values()
        ):
            raise CampaignError(
                f"broker KV readiness baseline was unavailable: {observed}"
            )
        atomic_json(
            run_dir / "broker-kv-readiness-baseline.json",
            {"timestamp": now(), "frontends": observed},
        )
        return observed

    def wait_for_broker_kv_events(
        self,
        run_dir: Path,
        baseline: Mapping[str, Mapping[str, Any]],
    ) -> None:
        timeout_seconds = int(
            self.config["workload"].get("broker_kv_ready_timeout_seconds", 120)
        )
        if timeout_seconds <= 0:
            raise CampaignError("broker_kv_ready_timeout_seconds must be positive")
        samples_path = run_dir / "broker-kv-readiness.jsonl"
        started = time.monotonic()
        deadline = started + timeout_seconds
        while True:
            self.require_frontends_alive()
            observed = self.scrape_kv_events_applied()
            sample: dict[str, Any] = {
                "timestamp": now(),
                "elapsed_seconds": time.monotonic() - started,
                "metric": KV_CACHE_EVENTS_APPLIED_METRIC,
                "frontends": {},
            }
            ready = True
            for name, item in observed.items():
                before = baseline[name]["successful_applications"]
                current = item["successful_applications"]
                delta = None if current is None else current - before
                sample["frontends"][name] = item | {
                    "baseline": before,
                    "delta": delta,
                }
                ready = ready and delta is not None and delta > 0
            with samples_path.open("a") as output:
                output.write(json.dumps(sample, sort_keys=True) + "\n")
            if ready:
                sample["status"] = "ready"
                sample["ready_at"] = now()
                atomic_json(run_dir / "broker-kv-readiness.json", sample)
                return
            if time.monotonic() >= deadline:
                sample["status"] = "timeout"
                atomic_json(run_dir / "broker-kv-readiness.json", sample)
                raise CampaignError(
                    "KV-cache-applied counters did not increase on every frontend "
                    f"within {timeout_seconds} seconds: {sample['frontends']}"
                )
            if self.shutdown.wait(1):
                raise CampaignError("interrupted while waiting for broker KV events")

    def smoke(self, run_dir: Path) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.config["model"]["name"],
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hello."},
                ],
                "stream": True,
                "max_tokens": 8,
                "ignore_eos": True,
            }
        ).encode()
        from validate_sse import validate

        if self.frontend_secondary is not None:
            endpoints = {
                "numa0": (str(self.frontend_networks["numa0"]["address"]), 8001),
                "numa1": (str(self.frontend_networks["numa1"]["address"]), 8002),
            }
        else:
            endpoints = {
                "frontend": (
                    str(self.frontend_networks["frontend"]["address"]),
                    int(self.config["ports"]["frontend_http"]),
                ),
            }
        results: dict[str, Any] = {}
        for name, (address, port) in endpoints.items():
            suffix = "" if len(endpoints) == 1 else f"-{name}"
            attempts: list[dict[str, Any]] = []
            for attempt in range(1, 4):
                request = urllib.request.Request(
                    f"http://{address}:{port}/v1/chat/completions",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        raw = response.read()
                    result = dict(validate(raw, 8))
                    result["accepted"] = bool(result["valid"])
                    (run_dir / f"sse-witness{suffix}-attempt{attempt}.txt").write_bytes(
                        raw
                    )
                except Exception as error:
                    result = {
                        "accepted": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                attempts.append(result)
                atomic_json(
                    run_dir / f"sse-witness{suffix}-attempt{attempt}.json", result
                )
                if result["accepted"]:
                    break
                time.sleep(0.25)
            selected = dict(attempts[-1])
            selected["attempts"] = attempts
            selected["accepted"] = any(bool(item["accepted"]) for item in attempts)
            atomic_json(run_dir / f"sse-witness{suffix}.json", selected)
            results[name] = selected
        summary = {
            "accepted": all(bool(item["accepted"]) for item in results.values()),
            "frontends": results,
        }
        atomic_json(run_dir / "sse-witness.json", summary)
        if not summary["accepted"]:
            raise CampaignError(f"streaming smoke failed: {summary}")
        return summary

    def start_frontend_monitor(self, run_dir: Path) -> None:
        assert self.frontend is not None and self.scratch is not None
        tracked = {"frontend": os.getpgid(self.frontend.pid)}
        metrics_port = int(self.config["ports"]["frontend_http"])
        if self.frontend_secondary is not None:
            tracked = {
                "frontend-numa0": os.getpgid(self.frontend.pid),
                "frontend-numa1": os.getpgid(self.frontend_secondary.pid),
            }
            metrics_port = 8001
            metrics_address = str(self.frontend_networks["numa0"]["address"])
        else:
            metrics_address = str(self.frontend_networks["frontend"]["address"])
        self.telemetry = base._start_system_telemetry(
            run_dir,
            role="frontend",
            tracked=tracked,
            metrics_url=f"http://{metrics_address}:{metrics_port}/metrics",
        )
        if self.frontend_secondary is not None:
            secondary_metrics_command = [
                sys.executable,
                str(Path(__file__).with_name("telemetry_capture.py")),
                "--role",
                "frontend-numa1",
                "--output",
                str(run_dir / "frontend-numa1-system-telemetry.jsonl"),
                "--tracked-pgid",
                f"frontend-numa1:{os.getpgid(self.frontend_secondary.pid)}",
                "--metrics-url",
                f"http://{self.frontend_networks['numa1']['address']}:8002/metrics",
            ]
            self.telemetry.append(
                base.ManagedProcess(
                    "frontend-numa1-collector",
                    secondary_metrics_command,
                    run_dir / "frontend-numa1-collector.log",
                ).start()
            )
        self.sampler = base.Sampler(
            self.scratch / "frontend-samples.jsonl",
            pools={
                "frontend_domain": str(self.config["topology"]["frontend"]["cpus"]),
                "control_domain": str(
                    self.config["topology"]["frontend_control"]["cpus"]
                ),
                "numa0": "0-71",
                "numa1": "72-143",
            },
            interface=[
                str(spec["interface"]) for spec in self.frontend_networks.values()
            ],
            process=self.frontend,
            interval=1.0,
        )
        self.sampler.start()

    def finish_frontend_monitor(self) -> None:
        for name, proc in [('numa0', self.frontend), ('numa1', self.frontend_secondary)]:
            if proc is not None and proc.poll() is None and self.run_dir is not None:
                maps = Path(f'/proc/{proc.pid}/numa_maps').read_text()
                (self.run_dir / f'{name}-settled-numa-maps.txt').write_text(maps)
                (self.run_dir / f'{name}-settled-status.txt').write_text(Path(f'/proc/{proc.pid}/status').read_text())
                atomic_json(self.run_dir / f'{name}-settled-residency.json', base.private_anonymous_residency(maps.splitlines()))
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler = None
        base._stop_processes(self.telemetry)
        self.telemetry = []
        assert self.scratch is not None and self.run_dir is not None
        shutil.copy2(
            self.scratch / "frontend-samples.jsonl",
            self.run_dir / "frontend-samples.jsonl",
        )

    def capture_perf(self, run_dir: Path, load_thread: threading.Thread) -> None:
        assert self.frontend is not None and self.scratch is not None
        deadline = time.monotonic() + int(self.config["profile"]["start_delay_seconds"])
        while time.monotonic() < deadline:
            self.require_frontends_alive()
            if not load_thread.is_alive():
                raise CampaignError("load ended before perf capture")
            if self.shutdown.wait(0.25):
                raise CampaignError("interrupted before perf capture")
        atomic_json(run_dir / 'profile-window-start.json', {'epoch':time.time(), 'monotonic':time.monotonic(), 'measurement_start':(run_dir/'MEASUREMENT_STARTED').read_text().strip()})
        captures = [("numa0", self.frontend)]
        if self.frontend_secondary is not None:
            captures.append(("numa1", self.frontend_secondary))
        elif not bool(self.config["runtime"].get("frontend_dual_numa", False)):
            captures = [("frontend", self.frontend)]
        requested_arms = self.config["profile"].get("capture_arms")
        if requested_arms is not None:
            requested = {str(name) for name in requested_arms}
            captures = [item for item in captures if item[0] in requested]
            if not captures:
                raise CampaignError(
                    f"profile.capture_arms selected no running frontend: {requested}"
                )
        processes: list[tuple[str, subprocess.Popen[Any], Any, Path, Path]] = []
        try:
            for name, frontend in captures:
                profile_dir = (
                    run_dir / "profiles" / "agentx" / name
                    if self.frontend_secondary is not None
                    else run_dir
                )
                profile_dir.mkdir(parents=True, exist_ok=True)
                base._capture_frontend_dso_manifest(frontend.pid, profile_dir)
                dual_frontend = self.frontend_secondary is not None
                data_name = f"oncpu-{name}.data" if dual_frontend else "oncpu.data"
                error_name = f"perf-{name}.err" if dual_frontend else "perf.err"
                data = self.scratch / data_name
                error_path = self.scratch / error_name
                error = error_path.open("w")
                process = subprocess.Popen(
                    [
                        "perf",
                        "record",
                        "--clockid", "mono",
                        "-e",
                        str(self.config["profile"].get("event", "cycles:u")),
                        "-F",
                        str(self.config["profile"]["frequency_hz"]),
                        "-m",
                        str(self.config["profile"]["perf_mmap_pages"]),
                        "--call-graph",
                        "dwarf,16384",
                        "-p",
                        str(frontend.pid),
                        "-o",
                        str(data),
                        "--",
                        "sleep",
                        str(self.config["profile"]["record_seconds"]),
                    ],
                    stderr=error,
                )
                processes.append((name, process, error, data, profile_dir))
            if bool(
                self.config["profile"].get("kernel_perf_enabled", True)
            ):
                kernel_profile_dir = run_dir / "profiles" / "kernel"
                kernel_profile_dir.mkdir(parents=True, exist_ok=True)
                kernel_data = self.scratch / "kernel.data"
                kernel_error = (self.scratch / "perf-kernel.err").open("w")
                kernel_process = subprocess.Popen(
                    [
                        "perf",
                        "record",
                        "--clockid", "mono",
                        "-a",
                        "-e",
                        "cycles:k",
                        "-F",
                        str(self.config["profile"]["frequency_hz"]),
                        "-m",
                        str(
                            self.config["profile"].get(
                                "kernel_perf_mmap_pages",
                                self.config["profile"]["perf_mmap_pages"],
                            )
                        ),
                        "--call-graph",
                        "fp",
                        "-o",
                        str(kernel_data),
                        "--",
                        "sleep",
                        str(self.config["profile"]["record_seconds"]),
                    ],
                    stderr=kernel_error,
                )
                processes.append(
                    (
                        "kernel",
                        kernel_process,
                        kernel_error,
                        kernel_data,
                        kernel_profile_dir,
                    )
                )
            timeout = int(self.config["profile"]["record_seconds"]) + 300
            for name, process, _, data, profile_dir in processes:
                returncode = process.wait(timeout=timeout)
                if returncode != 0:
                    raise CampaignError(
                        f"perf capture for {name} failed: rc={returncode}"
                    )
                if not data.is_file() or data.stat().st_size == 0:
                    raise CampaignError(f"perf capture for {name} produced no data")
                if name != "kernel":
                    base._capture_mapped_dso_checksums(profile_dir)
        finally:
            for _, process, error, _, _ in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)
                error.close()

    def capture_offcpu_perf(
        self, run_dir: Path, load_thread: threading.Thread
    ) -> None:
        """Best-effort non-root context-switch stack capture for each frontend."""
        assert self.frontend is not None and self.scratch is not None
        deadline = time.monotonic() + int(
            self.config["profile"].get(
                "offcpu_start_delay_seconds",
                self.config["profile"]["start_delay_seconds"],
            )
        )
        while time.monotonic() < deadline:
            self.require_frontends_alive()
            if not load_thread.is_alive():
                raise CampaignError("load ended before off-CPU capture")
            if self.shutdown.wait(0.25):
                raise CampaignError("interrupted before off-CPU capture")
        captures = [("numa0", self.frontend)]
        if self.frontend_secondary is not None:
            captures.append(("numa1", self.frontend_secondary))
        elif not bool(self.config["runtime"].get("frontend_dual_numa", False)):
            captures = [("frontend", self.frontend)]
        seconds = int(
            self.config["profile"].get(
                "offcpu_record_seconds",
                self.config["profile"]["record_seconds"],
            )
        )
        running: list[
            tuple[str, subprocess.Popen[Any], Any, Path, Path, Path]
        ] = []
        results: dict[str, Any] = {}
        try:
            for name, frontend in captures:
                profile_dir = (
                    run_dir / "profiles" / "agentx" / name
                    if self.frontend_secondary is not None
                    else run_dir
                )
                profile_dir.mkdir(parents=True, exist_ok=True)
                base._capture_frontend_dso_manifest(frontend.pid, profile_dir)
                data = self.scratch / f"offcpu-{name}.data"
                error_path = self.scratch / f"perf-offcpu-{name}.err"
                error = error_path.open("w")
                process = subprocess.Popen(
                    [
                        "perf",
                        "record",
                        "--clockid", "mono",
                        "--switch-events",
                        "-e",
                        "context-switches:u",
                        "-g",
                        "--call-graph",
                        "dwarf,16384",
                        "-m",
                        str(self.config["profile"]["perf_mmap_pages"]),
                        "-p",
                        str(frontend.pid),
                        "-o",
                        str(data),
                        "--",
                        "sleep",
                        str(seconds),
                    ],
                    stderr=error,
                )
                running.append(
                    (name, process, error, data, error_path, profile_dir)
                )
            timeout = seconds + 120
            for name, process, error, data, error_path, profile_dir in running:
                returncode = process.wait(timeout=timeout)
                error.flush()
                error_text = error_path.read_text(errors="replace")
                copied_data = profile_dir / "offcpu.data"
                copied_error = profile_dir / "perf-offcpu.err"
                shutil.copy2(error_path, copied_error)
                if data.is_file() and data.stat().st_size > 0:
                    shutil.copy2(data, copied_data)
                results[name] = {
                    "returncode": returncode,
                    "nonempty": copied_data.is_file()
                    and copied_data.stat().st_size > 0,
                    "error": error_text,
                }
        finally:
            for _, process, error, _, _, _ in running:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)
                error.close()
        atomic_json(
            run_dir / "offcpu-capture.json",
            {
                "accepted": bool(results)
                and all(
                    item["returncode"] == 0 and item["nonempty"]
                    for item in results.values()
                ),
                "profiles": results,
            },
        )

    def process_perf(self, run_dir: Path) -> None:
        profile_dirs = {"frontend": run_dir}
        if bool(self.config["runtime"].get("frontend_dual_numa", False)):
            requested_arms = self.config["profile"].get(
                "capture_arms", ("numa0", "numa1")
            )
            profile_dirs = {
                name: run_dir / "profiles" / "agentx" / name for name in requested_arms
            }
        results: dict[str, Any] = {}
        for name, profile_dir in profile_dirs.items():
            data = profile_dir / "oncpu.data"
            commands = [
                (
                    [
                        "perf",
                        "script",
                        "--no-inline",
                        "--show-lost-events",
                        "-i",
                        str(data),
                    ],
                    "perf-script.txt",
                    "perf-script.err",
                ),
                (
                    ["perf", "buildid-list", "-i", str(data)],
                    "perf-buildids.txt",
                    "perf-buildids.err",
                ),
                (
                    ["perf", "report", "--stdio", "--header-only", "-i", str(data)],
                    "perf-header.txt",
                    "perf-header.err",
                ),
            ]
            for command, stdout_name, stderr_name in commands:
                with (
                    (profile_dir / stdout_name).open("w") as stdout,
                    (profile_dir / stderr_name).open("w") as stderr,
                ):
                    completed = subprocess.run(
                        command,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=self.config["profile"][
                            "perf_processing_timeout_seconds"
                        ],
                    )
                if completed.returncode != 0:
                    raise CampaignError(
                        f"perf processing failed for {name}: {' '.join(command)}"
                    )
            script = (profile_dir / "perf-script.txt").read_text(errors="replace")
            lost_lines = [
                line for line in script.splitlines() if "PERF_RECORD_LOST" in line
            ]
            warning_lines = [
                line
                for path in (profile_dir / "perf.err", profile_dir / "perf-script.err")
                if path.is_file()
                for line in path.read_text(errors="replace").splitlines()
                if re.search(r"\blost\b", line, flags=re.IGNORECASE)
            ]
            lost_chunks = sum(
                int(match.group(1))
                for line in warning_lines
                if (
                    match := re.search(
                        r"\blost\s+(\d+)\s+chunks?\b",
                        line,
                        flags=re.IGNORECASE,
                    )
                )
            )
            frame_count = sum(
                bool(re.match(r"^\s*[0-9a-fA-F]+\s+.+\s+\([^)]*\)\s*$", line))
                for line in script.splitlines()
            )
            unresolved = script.count("[unknown]")
            manifest = json.loads(
                (profile_dir / "frontend-dso-manifest.json").read_text()
            )
            buildids = (profile_dir / "perf-buildids.txt").read_text(errors="replace")
            build_match = (
                manifest["core_build_id"] in buildids and Path(manifest["core_path_from_proc_maps"]).name in buildids
            )
            result = {
                "accepted": frame_count > 0
                and not lost_lines
                and not warning_lines
                and build_match
                and (unresolved / frame_count)
                < self.config["acceptance"]["unresolved_frame_max_fraction"],
                "frame_count": frame_count,
                "unresolved_frame_count": unresolved,
                "unresolved_frame_fraction": unresolved / frame_count
                if frame_count
                else None,
                "lost_sample_count": len(lost_lines),
                "lost_chunk_count_reported": lost_chunks,
                "warning_lines": warning_lines,
                "mapped_core_path": manifest["core_path_from_proc_maps"],
                "mapped_core_build_id": manifest["core_build_id"],
                "mapped_core_sha256": manifest["core_sha256"],
            }
            atomic_json(profile_dir / "perf-capture.json", result)
            if not result["accepted"]:
                raise CampaignError(f"perf quality failed for {name}: {result}")
            results[name] = result
        atomic_json(
            run_dir / "perf-capture.json",
            {
                "accepted": all(item["accepted"] for item in results.values()),
                "profiles": results,
            },
        )
        for name, profile_dir in profile_dirs.items():
            with (profile_dir/'perf-script.txt').open('rb') as src, (profile_dir/'folded-stacks.txt').open('wb') as dst:
                subprocess.run(['perl', str(Path(__file__).with_name('stackcollapse-perf.pl'))],stdin=src,stdout=dst,check=True,timeout=300)
            with (profile_dir/'folded-stacks.txt').open('rb') as src, (profile_dir/'flamegraph.svg').open('wb') as dst:
                subprocess.run([self.config['paths']['flamegraph'],'--title',f'Latest main frontend {name}, AgentX c6144','--countname','cycles'],stdin=src,stdout=dst,check=True,timeout=300)
        if not bool(self.config["profile"].get("kernel_perf_enabled", True)):
            return
        kernel_dir = run_dir / "profiles" / "kernel"
        kernel_data = kernel_dir / "kernel.data"
        kernel_commands = [
            (
                [
                    "perf",
                    "script",
                    "--show-lost-events",
                    "-i",
                    str(kernel_data),
                ],
                "perf-script.txt",
                "perf-script.err",
            ),
            (
                ["perf", "buildid-list", "-i", str(kernel_data)],
                "perf-buildids.txt",
                "perf-buildids.err",
            ),
            (
                [
                    "perf",
                    "report",
                    "--stdio",
                    "--sort",
                    "comm,dso,symbol",
                    "--percent-limit",
                    "0.1",
                    "-i",
                    str(kernel_data),
                ],
                "perf-report.txt",
                "perf-report.err",
            ),
        ]
        for command, stdout_name, stderr_name in kernel_commands:
            with (
                (kernel_dir / stdout_name).open("w") as stdout,
                (kernel_dir / stderr_name).open("w") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.config["profile"]["perf_processing_timeout_seconds"],
                )
            if completed.returncode != 0:
                raise CampaignError(
                    f"kernel perf processing failed: {' '.join(command)}"
                )
        kernel_script = (kernel_dir / "perf-script.txt").read_text(errors="replace")
        lost_lines = [
            line for line in kernel_script.splitlines() if "PERF_RECORD_LOST" in line
        ]
        warning_lines = [
            line
            for path in (kernel_dir / "perf.err", kernel_dir / "perf-script.err")
            if path.is_file()
            for line in path.read_text(errors="replace").splitlines()
            if re.search(r"\blost\b", line, flags=re.IGNORECASE)
        ]
        kernel_result = {
            "accepted": bool(kernel_script.strip())
            and not lost_lines
            and not warning_lines,
            "lost_sample_count": len(lost_lines),
            "warning_lines": warning_lines,
            "unknown_frame_count": kernel_script.count("[unknown]"),
        }
        atomic_json(kernel_dir / "perf-capture.json", kernel_result)
        if not kernel_result["accepted"]:
            raise CampaignError(f"kernel perf quality failed: {kernel_result}")
        stackcollapse = str(Path(__file__).with_name("stackcollapse-perf.pl"))
        with (
            (kernel_dir / "perf-script.txt").open("rb") as source,
            (kernel_dir / "folded-stacks.txt").open("wb") as destination,
        ):
            subprocess.run(
                [stackcollapse],
                stdin=source,
                stdout=destination,
                check=True,
                timeout=300,
            )
        with (
            (kernel_dir / "folded-stacks.txt").open("rb") as source,
            (kernel_dir / "flamegraph.svg").open("wb") as destination,
        ):
            subprocess.run(
                [
                    self.config["paths"]["flamegraph"],
                    "--title",
                    "AgentX c6144 frontend-node kernel cycles",
                    "--countname",
                    "samples",
                ],
                stdin=source,
                stdout=destination,
                check=True,
                timeout=300,
            )

    def process_mocker_perf(self, run_dir: Path) -> None:
        selected = self.config["profile"].get("mocker_process")
        if not selected:
            return
        profile_dir = run_dir / "profiles" / "mocker" / str(selected)
        data = profile_dir / "oncpu.data"
        if not data.is_file():
            raise CampaignError(f"missing mocker perf data: {data}")
        capture_target = json.loads(
            (profile_dir / "capture-target.json").read_text()
        )
        commands = [
            (
                [
                    "perf",
                    "script",
                    "--pid",
                    str(capture_target["pid"]),
                    "--no-inline",
                    "--show-lost-events",
                    "-i",
                    str(data),
                ],
                "perf-script.txt",
                "perf-script.err",
            ),
            (["perf", "buildid-list", "-i", str(data)], "perf-buildids.txt", "perf-buildids.err"),
            (
                ["perf", "report", "--stdio", "--header-only", "-i", str(data)],
                "perf-header.txt",
                "perf-header.err",
            ),
        ]
        for command, stdout_name, stderr_name in commands:
            with (
                (profile_dir / stdout_name).open("w") as stdout,
                (profile_dir / stderr_name).open("w") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.config["profile"]["perf_processing_timeout_seconds"],
                )
            if completed.returncode != 0:
                raise CampaignError(
                    f"mocker perf processing failed: {' '.join(command)}"
                )
        script = (profile_dir / "perf-script.txt").read_text(errors="replace")
        lost_lines = [
            line for line in script.splitlines() if "PERF_RECORD_LOST" in line
        ]
        warning_lines = [
            line
            for path in (profile_dir / "perf.err", profile_dir / "perf-script.err")
            for line in path.read_text(errors="replace").splitlines()
            if re.search(r"\blost\b", line, flags=re.IGNORECASE)
        ]
        frame_count = sum(
            bool(re.match(r"^\s*[0-9a-fA-F]+\s+.+\s+\([^)]*\)\s*$", line))
            for line in script.splitlines()
        )
        unresolved = script.count("[unknown]")
        manifest = json.loads((profile_dir / "frontend-dso-manifest.json").read_text())
        buildids = (profile_dir / "perf-buildids.txt").read_text(errors="replace")
        build_match = (
            manifest["core_build_id"] in buildids and Path(manifest["core_path_from_proc_maps"]).name in buildids
        )
        result = {
            "accepted": frame_count > 0
            and not lost_lines
            and not warning_lines
            and build_match
            and (unresolved / frame_count)
            < self.config["acceptance"]["unresolved_frame_max_fraction"],
            "frame_count": frame_count,
            "unresolved_frame_count": unresolved,
            "unresolved_frame_fraction": unresolved / frame_count if frame_count else None,
            "lost_sample_count": len(lost_lines),
            "warning_lines": warning_lines,
            "mapped_core_path": manifest["core_path_from_proc_maps"],
            "mapped_core_build_id": manifest["core_build_id"],
            "mapped_core_sha256": manifest["core_sha256"],
        }
        atomic_json(profile_dir / "perf-capture.json", result)
        if not result["accepted"]:
            raise CampaignError(f"mocker perf quality failed: {result}")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("analyze_single_perf.py")),
                "--profile-dir",
                str(profile_dir),
                "--output-dir",
                str(run_dir / "single-perf-analysis" / str(selected)),
                "--flamegraph",
                str(self.config["paths"]["flamegraph"]),
                "--title",
                f"AgentX c{self.config['campaign']['fixed_concurrency']} mocker {selected}",
            ],
            check=True,
            timeout=int(self.config["profile"]["perf_processing_timeout_seconds"]),
        )

    def stop_frontend(self) -> None:
        if self.sampler is not None:
            self.sampler.stop(raise_on_error=False)
            self.sampler = None
        try:
            base._stop_processes(self.telemetry)
        finally:
            self.telemetry = []
        frontend, secondary, scratch, run_dir = (
            self.frontend,
            self.frontend_secondary,
            self.scratch,
            self.run_dir,
        )
        self.frontend = None
        self.frontend_secondary = None
        self.scratch = None
        self.run_dir = None
        if frontend is not None:
            frontend.stop()
        if secondary is not None:
            secondary.stop()
        if scratch is not None and run_dir is not None:
            for name in (
                "frontend.log",
                "frontend-numa0.log",
                "frontend-numa1.log",
                "oncpu.data",
                "perf.err",
                "kernel.data",
                "perf-kernel.err",
            ):
                if name == "kernel.data":
                    destination = run_dir / "profiles" / "kernel" / name
                elif name == "perf-kernel.err":
                    destination = run_dir / "profiles" / "kernel" / "perf.err"
                else:
                    destination = run_dir / name
                copy_if_present(scratch / name, destination)
            for name in ("numa0", "numa1"):
                if (scratch / f"oncpu-{name}.data").is_file():
                    profile_dir = run_dir / "profiles" / "agentx" / name
                    copy_if_present(
                        scratch / f"oncpu-{name}.data", profile_dir / "oncpu.data"
                    )
                    copy_if_present(
                        scratch / f"perf-{name}.err", profile_dir / "perf.err"
                    )
            shutil.rmtree(scratch, ignore_errors=True)

    def write_artifact_index(self) -> None:
        artifacts = sorted(
            str(path.relative_to(self.result_dir))
            for path in self.result_dir.rglob("*")
            if path.is_file()
        )
        atomic_json(self.result_dir / "ARTIFACT_INDEX.json", {"artifacts": artifacts})


def initialize_paths(
    config: Mapping[str, Any], job_id: str, rank: int, shutdown: threading.Event
) -> tuple[Path, Path]:
    root = Path(config["root"])
    execution_id = f"{job_id}-{config['campaign']['name']}"
    state = root / "state" / execution_id
    results = root / "results" / execution_id
    if rank == 0:
        state.mkdir(parents=True, exist_ok=False)
        results.mkdir(parents=True, exist_ok=False)
        channel_names = [
            f"mocker-rank{node['rank']}" for node in normalized_mocker_nodes(config)
        ] + ["aiperf"]
        for name in channel_names:
            (state / "channels" / name / "commands").mkdir(parents=True)
            (state / "channels" / name / "acks").mkdir()
        (state / "agents").mkdir()
        (state / "preflight").mkdir()
    else:
        wait_for(state / "agents", 180, shutdown)
        wait_for(results, 180, shutdown)
    return state, results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("saturation-config.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise CampaignError("real execution requires a Slurm allocation")
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    mocker_nodes = normalized_mocker_nodes(config)
    expected_world = int(
        config.get("campaign", {}).get("world_size", len(mocker_nodes) + 2)
    )
    if world != expected_world or rank not in range(expected_world):
        raise CampaignError(
            f"expected ranks 0-{expected_world - 1}, got rank={rank}, world={world}"
        )
    job_id = os.environ["SLURM_JOB_ID"]
    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    state, results = initialize_paths(config, job_id, rank, shutdown)
    hosts = slurm_hosts(config)
    peer = hosts[1] if rank == 0 else hosts[0]
    networks = discover_networks(config, peer)
    primary = primary_network(config, networks)
    interface = str(primary["interface"])
    advertised_ip = str(primary["address"])
    if rank == 0:
        os.sched_setaffinity(
            0, base.parse_cpu_set(config["topology"]["frontend_control"]["cpus"])
        )
    atomic_json(
        state / "agents" / f"rank{rank}.json",
        {
            "rank": rank,
            "hostname": hosts[rank],
            "advertised_ip": advertised_ip,
            "interface": interface,
            "networks": networks,
            "pid": os.getpid(),
            "registered_at": now(),
        },
    )
    for expected in range(expected_world):
        wait_for(state / "agents" / f"rank{expected}.json", 180, shutdown)
    agents = [
        json.loads((state / "agents" / f"rank{item}.json").read_text())
        for item in range(expected_world)
    ]
    try:
        preflight = role_preflight(config, rank, interface, networks, agents)
    except Exception as error:
        preflight = {
            "ok": False,
            "rank": rank,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    atomic_json(state / "preflight" / f"rank{rank}.json", preflight)
    for expected in range(expected_world):
        wait_for(state / "preflight" / f"rank{expected}.json", 300, shutdown)
    preflights = [
        json.loads((state / "preflight" / f"rank{item}.json").read_text())
        for item in range(expected_world)
    ]
    if not all(item.get("ok") for item in preflights):
        raise CampaignError(f"preflight failed: {preflights}")
    try:
        if rank == 0:
            fanout = MockerFanoutChannel(
                [
                    (node, Channel(state, f"mocker-rank{node['rank']}", shutdown))
                    for node in mocker_nodes
                ]
            )
            etcd_node = next(node for node in mocker_nodes if node.get("manage_etcd"))
            Coordinator(
                config,
                fanout,
                Channel(state, "aiperf", shutdown),
                shutdown,
                job_id=job_id,
                result_dir=results,
                advertised_ip=agents[0]["advertised_ip"],
                mocker_ip=agents[int(etcd_node["rank"])]["advertised_ip"],
                interface=interface,
                networks=networks,
            ).run()
        elif any(int(node["rank"]) == rank for node in mocker_nodes):
            node = next(node for node in mocker_nodes if int(node["rank"]) == rank)
            etcd_node = next(item for item in mocker_nodes if item.get("manage_etcd"))
            MockerAgent(
                config,
                Channel(state, f"mocker-rank{rank}", shutdown),
                shutdown,
                job_id=job_id,
                rank=rank,
                node_spec=node,
                advertised_ip=agents[rank]["advertised_ip"],
                etcd_ip=agents[int(etcd_node["rank"])]["advertised_ip"],
                interface=interface,
                networks=networks,
            ).serve()
        elif rank == expected_world - 1:
            frontend_networks = {
                name: str(spec["address"])
                for name, spec in agents[0]["networks"].items()
            }
            frontend_ip: str | Mapping[str, str] = (
                frontend_networks
                if {"numa0", "numa1"} <= set(frontend_networks)
                else str(agents[0]["advertised_ip"])
            )
            AiperfAgent(
                config,
                Channel(state, "aiperf", shutdown),
                shutdown,
                job_id=job_id,
                advertised_ip=agents[rank]["advertised_ip"],
                frontend_ip=frontend_ip,
                interface=interface,
                networks=networks,
            ).serve()
        else:
            raise CampaignError(f"rank {rank} has no role")
        atomic_json(state / f"rank{rank}.done.json", {"completed_at": now()})
        return 0
    except Exception as error:
        atomic_json(
            state / f"rank{rank}.failed.json",
            {
                "failed_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"rank {rank} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
