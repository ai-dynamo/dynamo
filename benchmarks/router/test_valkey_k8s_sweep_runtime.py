# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.router.kubernetes.valkey_sweep import cluster as sweep_cluster
from benchmarks.router.kubernetes.valkey_sweep import start_valkey
from benchmarks.router.kubernetes.valkey_sweep import sweep
from benchmarks.router.valkey_k8s_sweep_test_support import (
    BINDING,
    DIGEST_IMAGE,
    completed,
)


def test_cluster_adapters_and_ha_helpers(monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def fake_run(command, *, input_text=None, timeout=None, check=True):
        calls.append((tuple(command), input_text))
        if "valkey-cli" in command:
            return completed("value-1\n\nvalue-2\n")
        return completed(json.dumps({"items": []}))

    monkeypatch.setattr(sweep_cluster, "run_command", fake_run)
    cluster = sweep_cluster.Cluster("test-ns")
    cluster.kubectl(("get", "pods"), input_text="input")
    cluster.client_exec(("true",))
    assert cluster.valkey("router", "PING") == ["value-1", "value-2"]
    assert cluster.sentinel(1, "PING") == ["value-1", "value-2"]
    assert calls[0] == (("kubectl", "-n", "test-ns", "get", "pods"), "input")
    assert "deployment/valkey-sweep-client" in calls[1][0]

    class HealthyCluster:
        def sentinel(self, ordinal, *arguments):
            if ordinal == 2:
                raise OSError("temporarily unavailable")
            if "CKQUORUM" in arguments:
                return ["OK 3 usable Sentinels"]
            if arguments[-1] == "dynamo-tokenizer":
                return ["valkey-sweep-tokenizer-primary", "6379"]
            return ["valkey-sweep-router-primary", "6379"]

        def valkey(self, host, command, *_arguments):
            if command == "INFO":
                if host.endswith("primary"):
                    return ["role:master", "connected_slaves:1"]
                return ["role:slave", "master_link_status:up"]
            return ["0", "10", "10"]

    healthy = HealthyCluster()
    assert sweep_cluster.sentinel_master(healthy, "dynamo-router") == (
        "valkey-sweep-router-primary",
        6379,
    )
    monkeypatch.setattr(sweep_cluster.time, "sleep", lambda _: None)
    sweep_cluster.wait_for_ha(healthy)
    assert sweep_cluster.wait_for_registered_mockers(healthy, "run/ns", 10) == [
        0,
        10,
        10,
    ]
    assert "%2F" in sweep_cluster.index_key("run/ns")

    empty = SimpleNamespace(
        kubectl=lambda *_args, **_kwargs: completed(json.dumps({"items": []}))
    )
    assert sweep_cluster.pods(empty, "app=test") == []
    sweep_cluster.wait_for_no_pods(empty, "deployment")
    sweep_cluster.rollout(empty, "deployment")


def test_apply_stack_and_sentinel_fail_closed(monkeypatch) -> None:
    operations: list[tuple[str, ...]] = []

    class FakeCluster:
        def kubectl(self, arguments, **_kwargs):
            operations.append(tuple(arguments))
            if arguments[:2] == ("get", "service/etcd"):
                return completed(
                    json.dumps(
                        {"spec": {"selector": {"app.kubernetes.io/component": "etcd"}}}
                    )
                )
            return completed()

    monkeypatch.setattr(
        sweep_cluster, "wait_for_ha", lambda _: operations.append(("ha",))
    )
    sweep_cluster.apply_stack(FakeCluster(), DIGEST_IMAGE)
    assert ("apply", "-f", "-") in operations
    assert operations[-1] == ("ha",)
    assert sum(operation[:2] == ("rollout", "status") for operation in operations) == 6

    failing = SimpleNamespace(sentinel=lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(RuntimeError, match="no Sentinel"):
        sweep_cluster.sentinel_master(failing, "dynamo-router")


def test_start_valkey_queries_majority_and_executes_server(
    monkeypatch, tmp_path: Path
) -> None:
    responses = iter(("master\n6379\n", "master\n6379\n", "other\n6379\n"))
    monkeypatch.setattr(
        start_valkey.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=next(responses)),
    )
    assert start_valkey.sentinel_master("dynamo-router", frozenset({"master"})) == (
        "master",
        6379,
    )
    monkeypatch.setattr(
        start_valkey.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("10.0.0.1", 0))],
    )
    assert start_valkey.resolves_to_self("master", "10.0.0.1", "self")
    with pytest.raises(ValueError, match="invalid bootstrap role"):
        start_valkey.role_arguments(
            None,
            bootstrap_role="invalid",
            bootstrap_primary_host="primary",
            pod_ip="10.0.0.2",
            pod_dns="self",
        )
    args, server_args = start_valkey.parse_args(
        [
            "--master-name",
            "dynamo-router",
            "--bootstrap-role",
            "primary",
            "--bootstrap-primary-host",
            "primary",
            "--allowed-master-host",
            "primary",
            "--",
            "--appendonly",
            "yes",
        ]
    )
    assert args.master_name == "dynamo-router"
    assert server_args == ["--appendonly", "yes"]

    monkeypatch.setattr(
        start_valkey, "wait_for_master", lambda *_args, **_kwargs: ("primary", 6379)
    )
    monkeypatch.setattr(
        start_valkey, "wait_for_server_ready", lambda *_args, **_kwargs: None
    )
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("POD_IP", "10.0.0.1")
    monkeypatch.setenv("POD_DNS", "primary")
    command: list[str] = []

    class FakeProcess:
        returncode = 0

        def poll(self):
            return None

        def wait(self, **_kwargs):
            return 0

        def terminate(self):
            raise AssertionError("healthy child must not be terminated")

        def send_signal(self, _signum):
            raise AssertionError("test does not send a signal")

    def fake_popen(arguments):
        command.extend(arguments)
        return FakeProcess()

    monkeypatch.setattr(start_valkey.subprocess, "Popen", fake_popen)
    start_valkey.main(
        [
            "--master-name",
            "dynamo-router",
            "--bootstrap-role",
            "primary",
            "--bootstrap-primary-host",
            "primary",
            "--allowed-master-host",
            "primary",
            "--",
            "--appendonly",
            "yes",
        ]
    )
    assert command[0].endswith("valkey-server")
    assert command[-2:] == ["--appendonly", "yes"]
    assert (tmp_path / "bootstrap-complete").is_file()


def test_start_valkey_does_not_mark_unadopted_bootstrap(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeProcess:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = -15

        def wait(self, **_kwargs):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = FakeProcess()
    monkeypatch.setattr(start_valkey.subprocess, "Popen", lambda *_: process)
    monkeypatch.setattr(start_valkey, "wait_for_server_ready", lambda *_: None)
    monkeypatch.setattr(start_valkey, "wait_for_master", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="Sentinel did not adopt"):
        start_valkey.start_initial_server(
            ["valkey-server"],
            state_dir=tmp_path,
            master_name="dynamo-router",
            allowed_hosts=frozenset({"primary"}),
        )
    assert not (tmp_path / "bootstrap-complete").exists()
    assert process.returncode == -15


def test_configure_topology_uses_fresh_namespace_and_exact_counts(monkeypatch) -> None:
    operations: list[tuple[str, ...]] = []

    class FakeCluster:
        def kubectl(self, arguments, **_kwargs):
            operations.append(tuple(arguments))
            return completed()

        def valkey(self, _host, command, *_arguments):
            return ["OK"] if command == "FLUSHDB" else ["1"]

        def client_exec(self, _arguments, **_kwargs):
            if _arguments[0] == "/bin/sh":
                return completed("1048576\n1024 65535\n")
            return completed()

    fake = FakeCluster()
    point = sweep.MatrixPoint(2, 4096, 10)
    monkeypatch.setattr(
        sweep,
        "elected_identity",
        lambda _cluster, name: (
            f"{name}-primary",
            f"{name}-replica",
            {},
            (f"{name}-primary", 6379),
        ),
    )
    monkeypatch.setattr(
        sweep,
        "wait_for_valkey_value",
        lambda _cluster, _host, _arguments, expected: expected,
    )
    monkeypatch.setattr(sweep, "wait_for_no_pods", lambda *_: None)
    monkeypatch.setattr(sweep, "rollout", lambda *_: None)
    monkeypatch.setattr(sweep, "wait_for_ha", lambda *_: None)
    monkeypatch.setattr(sweep, "wait_for_registered_mockers", lambda *_: [0, 10, 10])
    monkeypatch.setattr(sweep, "verify_active_images", lambda *_: {"pod": {"uid": "1"}})
    monkeypatch.setattr(sweep, "ha_snapshot", lambda *_: {"ha": "healthy"})
    monkeypatch.setattr(
        sweep,
        "pods",
        lambda *_: [
            {
                "metadata": {"name": "frontend-a"},
                "status": {"phase": "Running", "podIP": "10.0.0.1"},
            },
            {
                "metadata": {"name": "frontend-b"},
                "status": {"phase": "Running", "podIP": "10.0.0.2"},
            },
        ],
    )
    topology = sweep.configure_topology(fake, "campaign", point, BINDING, "d" * 16)
    assert topology["frontend_urls"] == ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
    expected_reset = {
        "flush": ["OK"],
        "primary_dbsize": ["0"],
        "replica_dbsize": ["0"],
    }
    assert topology["tokenizer_reset"] == expected_reset
    assert topology["router_reset"] == expected_reset
    assert topology["load_generator_capacity"]["ephemeral_ports"] == 64512
    assert "d" * 8 in topology["runtime_namespace"]
    assert any("--replicas=10" in operation for operation in operations)
    assert any("--replicas=2" in operation for operation in operations)
