# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.router.kubernetes.valkey_sweep import cluster as sweep_cluster
from benchmarks.router.kubernetes.valkey_sweep import ha as sweep_ha
from benchmarks.router.kubernetes.valkey_sweep import model as sweep_model
from benchmarks.router.kubernetes.valkey_sweep import start_valkey


ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "benchmarks" / "router" / "kubernetes" / "valkey_sweep"


def _load_sweep_module():
    return importlib.import_module("benchmarks.router.kubernetes.valkey_sweep.sweep")


def _resources() -> dict[tuple[str, str], dict]:
    documents = [
        document
        for path in sorted(SWEEP_DIR.glob("stack-*.yaml"))
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if document
    ]
    return {
        (document["kind"], document["metadata"]["name"]): document
        for document in documents
    }


def test_requested_matrix_is_exact_and_request_count_covers_concurrency() -> None:
    sweep = _load_sweep_module()

    points = sweep.matrix_points()

    assert len(points) == 36
    assert {point.frontends for point in points} == {1, 2, 4}
    assert {point.concurrency for point in points} == {4096, 8192, 16384, 32768}
    assert {point.mockers for point in points} == {10, 40, 80}
    assert all(
        sweep_model.request_count(point.concurrency) >= 4 * point.concurrency
        for point in points
    )
    assert all(
        sweep_model.warmup_count(point.concurrency) == point.concurrency
        for point in points
    )
    assert len({point.slug for point in points}) == 36
    first_namespace = sweep.runtime_namespace("campaign", points[0], "a" * 16)
    second_namespace = sweep.runtime_namespace("campaign", points[0], "b" * 16)
    assert "c4096" in first_namespace
    assert first_namespace != second_namespace


def test_manifest_rendering_is_structural_and_rejects_yaml_injection() -> None:
    sweep = _load_sweep_module()

    digest_image = "registry.example/dynamo@sha256:" + "a" * 64
    rendered = list(yaml.safe_load_all(sweep.render_stack(digest_image)))

    assert rendered
    assert {document["metadata"]["namespace"] for document in rendered} == {"bis-rl-3"}
    images = {
        container["image"]
        for document in rendered
        for container in document.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    }
    assert images == {digest_image}
    assert [path.name for path in sweep_cluster.MANIFEST_FILES] == [
        "stack-config.yaml",
        "stack-data.yaml",
        "stack-policy.yaml",
        "stack-workload.yaml",
    ]
    with pytest.raises(ValueError, match="digest image reference"):
        sweep.render_stack("registry.example/dynamo:mutable")
    with pytest.raises(ValueError, match="image reference"):
        sweep.render_stack("safe:image\n---\nkind: Secret")


def test_manifest_rendering_rejects_weakened_pod_security(
    monkeypatch, tmp_path: Path
) -> None:
    documents = [
        document
        for path in sweep_cluster.MANIFEST_FILES
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if document
    ]
    workload = next(
        document
        for document in documents
        if document["kind"] == "Deployment"
        and document["metadata"]["name"] == "valkey-sweep-client"
    )
    manifest = tmp_path / "stack.yaml"
    workload["spec"]["template"]["spec"]["securityContext"]["seccompProfile"] = {
        "type": "Unconfined"
    }
    manifest.write_text(yaml.safe_dump_all(documents), encoding="utf-8")
    monkeypatch.setattr(sweep_cluster, "MANIFEST_FILES", (manifest,))
    with pytest.raises(ValueError, match="RuntimeDefault seccomp"):
        sweep_cluster.render_stack("registry.example/dynamo@sha256:" + "a" * 64)

    workload["spec"]["template"]["spec"]["securityContext"]["seccompProfile"] = {
        "type": "RuntimeDefault"
    }
    workload["spec"]["template"]["spec"]["containers"][0]["securityContext"][
        "capabilities"
    ] = {"drop": []}
    manifest.write_text(yaml.safe_dump_all(documents), encoding="utf-8")
    with pytest.raises(ValueError, match="retains Linux capabilities"):
        sweep_cluster.render_stack("registry.example/dynamo@sha256:" + "a" * 64)


def test_manifest_rendering_rejects_unscoped_network_policy(
    monkeypatch, tmp_path: Path
) -> None:
    documents = [
        document
        for path in sweep_cluster.MANIFEST_FILES
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if document
    ]
    policy = next(document for document in documents if document["kind"] == "NetworkPolicy")
    policy["spec"]["podSelector"]["matchLabels"].pop("app.kubernetes.io/part-of")
    manifest = tmp_path / "stack.yaml"
    manifest.write_text(yaml.safe_dump_all(documents), encoding="utf-8")
    monkeypatch.setattr(sweep_cluster, "MANIFEST_FILES", (manifest,))
    with pytest.raises(ValueError, match="network policy selector.*not sweep-scoped"):
        sweep_cluster.render_stack("registry.example/dynamo@sha256:" + "a" * 64)


def test_resume_manifest_and_point_binding_are_immutable(tmp_path: Path) -> None:
    sweep = _load_sweep_module()
    point = sweep.MatrixPoint(1, 4096, 10)
    binding = {
        "image": "registry.example/dynamo@sha256:" + "a" * 64,
        "core_revision": "b" * 40,
        "image_ids": ["registry.example/dynamo@sha256:" + "a" * 64],
    }
    contract = sweep.campaign_contract(
        campaign="campaign",
        image=binding["image"],
        points=[point],
        methodology={"methodology_digest": "sha256:method"},
        prove_failover=False,
    )
    manifest = sweep.campaign_manifest(
        contract=contract,
        binding=binding,
        network_isolation={"status": "passed"},
        sentinel_policy={"status": "expected"},
    )
    path = tmp_path / "manifest.json"
    sweep.write_json_atomic(path, manifest)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    sweep.validate_resume_manifest(manifest, manifest)

    changed = {**manifest, "contract": {**contract, "image": "other"}}
    with pytest.raises(ValueError, match="resume manifest"):
        sweep.validate_resume_manifest(manifest, changed)

    result = {
        "status": "ok",
        "campaign": "campaign",
        "manifest_digest": manifest["manifest_digest"],
        "attempt_generation": "generation",
        "point": point._asdict(),
        "provenance": binding,
        "aiperf_summary_digest": "sha256:summary",
        "metrics": {"request_throughput_rps": 1.0},
    }
    attempt = {
        "status": "complete",
        "campaign": "campaign",
        "manifest_digest": manifest["manifest_digest"],
        "generation": "generation",
        "point": point._asdict(),
        "provenance": binding,
        "result": "result.json",
        "result_digest": sweep.canonical_digest(result),
        "aiperf_summary_digest": "sha256:summary",
    }
    assert sweep.completed_result_matches(
        result,
        attempt,
        point,
        binding,
        "campaign",
        manifest["manifest_digest"],
        "sha256:summary",
        result["metrics"],
    )
    assert not sweep.completed_result_matches(
        {**result, "provenance": {**binding, "core_revision": "c" * 40}},
        attempt,
        point,
        binding,
        "campaign",
        manifest["manifest_digest"],
        "sha256:summary",
        result["metrics"],
    )

    first_attempt = sweep.begin_attempt(
        tmp_path, point, binding, "campaign", manifest["manifest_digest"]
    )
    second_attempt = sweep.begin_attempt(
        tmp_path, point, binding, "campaign", manifest["manifest_digest"]
    )
    assert first_attempt["generation"] != second_attempt["generation"]
    assert json.loads((tmp_path / "attempt.json").read_text()) == second_attempt


def test_valkey_launcher_bootstrap_is_one_shot_and_fail_closed(tmp_path: Path) -> None:
    calls = iter((None, ("elected", 6379)))
    master = start_valkey.wait_for_master(
        "router",
        allowed_hosts=frozenset({"elected"}),
        initialized=True,
        lookup=lambda *_: next(calls),
        sleep=lambda _: None,
    )
    assert master == ("elected", 6379)
    assert (
        start_valkey.wait_for_master(
            "router",
            allowed_hosts=frozenset({"elected"}),
            initialized=False,
            bootstrap_timeout=0,
            lookup=lambda *_: None,
            sleep=lambda _: None,
        )
        is None
    )
    assert (
        start_valkey.role_arguments(
            ("10.0.0.1", 6379),
            bootstrap_role="primary",
            bootstrap_primary_host="bootstrap",
            pod_ip="10.0.0.1",
            pod_dns="self",
        )
        == []
    )
    assert start_valkey.role_arguments(
        None,
        bootstrap_role="replica",
        bootstrap_primary_host="bootstrap",
        pod_ip="10.0.0.2",
        pod_dns="self",
    ) == ["--replicaof", "bootstrap", "6379"]
    start_valkey.mark_initialized(tmp_path)
    start_valkey.mark_initialized(tmp_path)
    assert (tmp_path / "bootstrap-complete").read_text() == "initialized\n"


def test_failover_proof_can_run_twice(monkeypatch) -> None:
    class FakeCluster:
        groups = sweep_ha.GROUP_IDENTITIES

        def __init__(self):
            self.current = {
                name: identities[0] for name, identities in self.groups.items()
            }
            self.values: dict[str, str] = {}

        def kubectl(self, arguments, **_kwargs):
            if arguments[0] == "scale" and arguments[-1] == "--replicas=0":
                identity = arguments[1].split("/", 1)[1]
                for name, identities in self.groups.items():
                    if identity in identities and self.current[name] == identity:
                        self.current[name] = next(
                            item for item in identities if item != identity
                        )
            return SimpleNamespace(stdout="", returncode=0)

        def valkey(self, _host, command, *arguments):
            if command == "SET":
                self.values[arguments[0]] = arguments[1]
                return ["OK"]
            if command == "GET":
                value = self.values.get(arguments[0])
                return [] if value is None else [value]
            if command == "DEL":
                del self.values[arguments[0]]
                return ["1"]
            if command == "WAIT":
                return ["1"]
            raise AssertionError(command)

    fake = FakeCluster()

    def sentinel_master(_cluster, name):
        return (fake.current[name], 6379)

    def selected_info(_cluster, identity):
        name = "dynamo-router" if "router" in identity else "dynamo-tokenizer"
        if fake.current[name] == identity:
            return {"role": "master", "connected_slaves": "1"}
        return {"role": "slave", "master_link_status": "up"}

    def replication_snapshot(_cluster, name):
        elected = fake.current[name]
        replica = next(
            identity for identity in fake.groups[name] if identity != elected
        )
        return (
            elected,
            replica,
            {identity: selected_info(fake, identity) for identity in fake.groups[name]},
            (elected, 6379),
        )

    monkeypatch.setattr(sweep_ha, "sentinel_master", sentinel_master)
    monkeypatch.setattr(sweep_ha, "selected_valkey_info", selected_info)
    monkeypatch.setattr(sweep_ha, "replication_snapshot", replication_snapshot)
    monkeypatch.setattr(sweep_ha, "wait_for_no_pods", lambda *_: None)
    monkeypatch.setattr(sweep_ha, "wait_for_ha", lambda *_: None)
    monkeypatch.setattr(
        sweep_ha, "pods", lambda *_: [{"metadata": {"name": "sentinel"}}]
    )

    canaries: list[int] = []

    def application_probe():
        canaries.append(len(canaries) + 1)
        return {"sequence": canaries[-1]}

    first = sweep_ha.prove_failover_and_restart(fake, application_probe)
    second = sweep_ha.prove_failover_and_restart(fake)

    assert first["status"] == second["status"] == "passed"
    assert first["scope"] == "storage-and-dynamo"
    assert second["scope"] == "storage-only"
    assert [item["phase"] for item in first["application_canaries"]] == [
        "before-failover",
        "during-dynamo-router-outage",
        "after-dynamo-router-failover",
        "during-dynamo-tokenizer-outage",
        "after-dynamo-tokenizer-failover",
        "after-sentinel-restart",
    ]
    assert fake.current == {
        name: identities[0] for name, identities in fake.groups.items()
    }


def test_runtime_snapshot_requires_exact_healthy_base_and_workload() -> None:
    image_id = "registry/dynamo@sha256:" + "a" * 64

    def pod(name: str, *, ready: bool = True) -> dict:
        return {
            "metadata": {"name": name, "uid": f"uid-{name}"},
            "status": {
                "phase": "Running",
                "conditions": [
                    {"type": "Ready", "status": "True" if ready else "False"}
                ],
                "containerStatuses": [
                    {
                        "imageID": image_id,
                        "restartCount": 0,
                        "ready": ready,
                    }
                ],
            },
        }

    names = [
        "valkey-sweep-router-primary-0",
        "valkey-sweep-router-replica-0",
        "valkey-sweep-tokenizer-primary-0",
        "valkey-sweep-tokenizer-replica-0",
        "valkey-sweep-sentinel-0",
        "valkey-sweep-sentinel-1",
        "valkey-sweep-sentinel-2",
        "valkey-sweep-client-abc",
        "valkey-sweep-frontend-abc",
        *(f"valkey-sweep-mocker-{number}" for number in range(10)),
    ]

    class FakeCluster:
        def __init__(self):
            self.items = [pod(name) for name in names]

        def kubectl(self, *_args, **_kwargs):
            return SimpleNamespace(stdout=json.dumps({"items": self.items}))

    fake = FakeCluster()
    binding = {"image_ids": [image_id]}
    point = _load_sweep_module().MatrixPoint(1, 4096, 10)
    snapshot = sweep_cluster.verify_active_images(fake, binding, point)
    assert len(snapshot) == len(names)
    fake.items[-1] = pod(names[-1], ready=False)
    with pytest.raises(RuntimeError, match="not healthy"):
        sweep_cluster.verify_active_images(fake, binding, point)


def test_ha_snapshot_compares_topology_without_volatile_telemetry(monkeypatch) -> None:
    masters = {
        name: identities[0] for name, identities in sweep_ha.GROUP_IDENTITIES.items()
    }
    counter = 0

    def sentinel_master(_cluster, master_name):
        return (masters[master_name], 6379)

    def selected_info(_cluster, identity):
        nonlocal counter
        counter += 1
        group = "dynamo-router" if "router" in identity else "dynamo-tokenizer"
        if identity == masters[group]:
            return {
                "role": "master",
                "connected_slaves": "1",
                "total_commands_processed": str(counter),
            }
        return {
            "role": "slave",
            "master_link_status": "up",
            "used_memory": str(counter),
        }

    def replication_snapshot(_cluster, master_name):
        elected = masters[master_name]
        replica = next(
            identity
            for identity in sweep_ha.GROUP_IDENTITIES[master_name]
            if identity != elected
        )
        infos = {
            identity: selected_info(_cluster, identity)
            for identity in sweep_ha.GROUP_IDENTITIES[master_name]
        }
        return elected, replica, infos, (elected, 6379)

    monkeypatch.setattr(sweep_ha, "sentinel_master", sentinel_master)
    monkeypatch.setattr(sweep_ha, "selected_valkey_info", selected_info)
    monkeypatch.setattr(sweep_ha, "replication_snapshot", replication_snapshot)
    first = sweep_ha.ha_snapshot(object())
    second = sweep_ha.ha_snapshot(object())
    assert first == second
    assert (
        "total_commands_processed"
        not in first["dynamo-router"]["identities"]["valkey-sweep-router-primary"]
    )


def test_stack_is_namespaced_labeled_and_uses_distinct_valkey_memory_policies() -> None:
    resources = _resources()
    expected_stateful_sets = {
        "valkey-sweep-router-primary",
        "valkey-sweep-router-replica",
        "valkey-sweep-tokenizer-primary",
        "valkey-sweep-tokenizer-replica",
        "valkey-sweep-sentinel",
    }
    actual_stateful_sets = {name for kind, name in resources if kind == "StatefulSet"}
    assert actual_stateful_sets == expected_stateful_sets

    sentinels = resources[("StatefulSet", "valkey-sweep-sentinel")]
    assert sentinels["spec"]["replicas"] == 3
    assert sentinels["spec"]["volumeClaimTemplates"]
    assert resources[("ConfigMap", "valkey-sweep-config")]["immutable"] is True

    disruption_budgets = {
        name for kind, name in resources if kind == "PodDisruptionBudget"
    }
    assert disruption_budgets == {
        "valkey-sweep-router",
        "valkey-sweep-tokenizer",
        "valkey-sweep-sentinel",
    }

    for document in resources.values():
        assert document["metadata"]["namespace"] == "bis-rl-3"
        assert document["metadata"]["labels"]["app.kubernetes.io/part-of"] == (
            "valkey-router-sweep"
        )
        selector = document.get("spec", {}).get("selector")
        if selector is not None:
            labels = selector.get("matchLabels", selector)
            assert labels["app.kubernetes.io/part-of"] == "valkey-router-sweep"

    router_command = " ".join(
        resources[("StatefulSet", "valkey-sweep-router-primary")]["spec"]["template"][
            "spec"
        ]["containers"][0]["args"]
    )
    tokenizer_command = " ".join(
        resources[("StatefulSet", "valkey-sweep-tokenizer-primary")]["spec"][
            "template"
        ]["spec"]["containers"][0]["args"]
    )
    assert "--maxmemory-policy noeviction" in router_command
    assert "--appendonly yes" in router_command
    assert "--loadmodule /usr/local/lib/dynkv.so" in router_command
    assert "--maxmemory-policy allkeys-lru" in tokenizer_command
    assert "--appendonly no" in tokenizer_command
    assert "dynkv.so" not in tokenizer_command
    for stateful_set in expected_stateful_sets - {"valkey-sweep-sentinel"}:
        data_command = " ".join(
            resources[("StatefulSet", stateful_set)]["spec"]["template"]["spec"][
                "containers"
            ][0]["args"]
        )
        assert "--replicaof" not in data_command

    sentinel_init = " ".join(
        sentinels["spec"]["template"]["spec"]["initContainers"][0]["args"]
    )
    assert "test -s /sentinel-data/sentinel.conf" in sentinel_init


def test_stack_has_scalable_clients_network_isolation_and_no_embedded_secret() -> None:
    resources = _resources()
    for name in (
        "valkey-sweep-frontend",
        "valkey-sweep-mocker",
        "valkey-sweep-client",
    ):
        deployment = resources[("Deployment", name)]
        assert deployment["spec"]["template"]["spec"]["imagePullSecrets"] == [
            {"name": "nvcr-pull-secret"}
        ]

    for document in resources.values():
        pod_spec = document.get("spec", {}).get("template", {}).get("spec", {})
        if pod_spec:
            assert pod_spec["automountServiceAccountToken"] is False
            assert pod_spec["securityContext"]["runAsNonRoot"] is True
            assert pod_spec["securityContext"]["seccompProfile"] == {
                "type": "RuntimeDefault"
            }
        containers = pod_spec.get("initContainers", []) + pod_spec.get("containers", [])
        for container in containers:
            assert all(
                isinstance(argument, str) for argument in container.get("args", [])
            )
            assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]

    policies = {name for kind, name in resources if kind == "NetworkPolicy"}
    assert policies == {
        "valkey-sweep-router-ingress",
        "valkey-sweep-tokenizer-ingress",
        "valkey-sweep-sentinel-ingress",
        "valkey-sweep-frontend-ingress",
        "valkey-sweep-mocker-ingress",
        "valkey-sweep-router-egress",
        "valkey-sweep-tokenizer-egress",
        "valkey-sweep-sentinel-egress",
        "valkey-sweep-frontend-egress",
        "valkey-sweep-mocker-egress",
        "valkey-sweep-client-egress",
    }
    for name in policies:
        policy = resources[("NetworkPolicy", name)]
        assert (
            policy["spec"]["podSelector"]["matchLabels"]["app.kubernetes.io/part-of"]
            == "valkey-router-sweep"
        )
        for rule in policy["spec"].get("ingress", []) + policy["spec"].get(
            "egress", []
        ):
            peers = rule.get("from", []) + rule.get("to", [])
            for peer in peers:
                selector = peer.get("podSelector")
                if selector is None:
                    continue
                labels = selector.get("matchLabels", {})
                if labels.get("k8s-app") == "kube-dns":
                    continue
                if labels.get("app.kubernetes.io/component") == "etcd":
                    assert labels == {"app.kubernetes.io/component": "etcd"}
                    assert rule["ports"] == [{"protocol": "TCP", "port": 2379}]
                    continue
                assert labels["app.kubernetes.io/part-of"] == "valkey-router-sweep"

    manifest_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SWEEP_DIR.glob("stack-*.yaml"))
    )
    dockerfile_text = (SWEEP_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "${VALKEY_SWEEP_IMAGE}" in manifest_text
    assert "password:" not in manifest_text.lower()
    assert "router-valkey-config:" in manifest_text
    assert "ARG GIT_REVISION" in dockerfile_text
    assert "ARG VALKEY_GIT_REVISION" not in dockerfile_text
    assert "VALKEY_GIT_REVISION=5b690cefd6cad707a748879c2bab6b72e18efcb7" in (
        dockerfile_text
    )
    assert "pylock.aiperf.toml" in dockerfile_text
    assert "\n        curl " in dockerfile_text
    assert "cargo install --locked --version 1.9.4 maturin" in dockerfile_text
    assert dockerfile_text.count("@sha256:") >= 3
    assert "rustup.rs" not in dockerfile_text
    assert "snapshot.debian.org" in dockerfile_text
    assert "--no-build-isolation" in dockerfile_text
    assert "pylock.build.toml" in dockerfile_text
    assert "pylock.crick-build.toml" in dockerfile_text
    assert "crick-0.0.8.tar.gz" in dockerfile_text
    assert "--no-build" in dockerfile_text
    assert "DYNAMO_BUILD_GIT_DIRTY=false" not in dockerfile_text
    assert "COPY source.bundle" in dockerfile_text
    assert "GIT_LFS_SKIP_SMUDGE=1 git clone" in dockerfile_text
    assert "git -C /source fsck --strict" in dockerfile_text
    assert "git -C /source rev-parse --is-shallow-repository" in dockerfile_text

    for deployment_name in ("valkey-sweep-mocker", "valkey-sweep-frontend"):
        env = resources[("Deployment", deployment_name)]["spec"]["template"]["spec"][
            "containers"
        ][0]["env"]
        config = next(
            item for item in env if item["name"] == "DYN_ROUTER_VALKEY_CONFIG"
        )
        assert config["valueFrom"]["configMapKeyRef"]["name"] == ("valkey-sweep-config")
    client = resources[("Deployment", "valkey-sweep-client")]["spec"]["template"][
        "spec"
    ]
    model_mount = next(
        item
        for item in client["containers"][0]["volumeMounts"]
        if item["name"] == "models"
    )
    assert model_mount["readOnly"] is True
    results_volume = next(
        item for item in client["volumes"] if item["name"] == "results"
    )
    assert results_volume["emptyDir"]["sizeLimit"] == "10Gi"
    assert client["securityContext"]["sysctls"] == [
        {"name": "net.ipv4.ip_local_port_range", "value": "1024 65535"}
    ]

    assert (
        "PersistentVolumeClaim",
        "valkey-sweep-tokenizer-primary-state",
    ) in resources
    assert (
        "PersistentVolumeClaim",
        "valkey-sweep-tokenizer-replica-state",
    ) in resources
