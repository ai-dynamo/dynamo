# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.router.kubernetes.valkey_sweep import cluster as sweep_cluster
from benchmarks.router.kubernetes.valkey_sweep import ha as sweep_ha
from benchmarks.router.kubernetes.valkey_sweep import methodology as sweep_methodology
from benchmarks.router.kubernetes.valkey_sweep import model as sweep_model
from benchmarks.router.kubernetes.valkey_sweep import network as sweep_network
from benchmarks.router.valkey_k8s_sweep_test_support import DIGEST_IMAGE, completed


def test_sentinel_policy_and_network_isolation_proofs() -> None:
    expected_policy = [
        "quorum",
        "2",
        "down-after-milliseconds",
        "5000",
        "failover-timeout",
        "30000",
        "parallel-syncs",
        "1",
    ]
    policy_cluster = SimpleNamespace(sentinel=lambda *_: expected_policy)
    policy = sweep_cluster.sentinel_policy_snapshot(policy_cluster)
    assert set(policy) == {"0", "1", "2"}
    assert set(policy["0"]) == {"dynamo-router", "dynamo-tokenizer"}
    stale = SimpleNamespace(sentinel=lambda *_: [*expected_policy[:-1], "3"])
    with pytest.raises(RuntimeError, match="stale policy"):
        sweep_cluster.sentinel_policy_snapshot(stale)

    operations: list[tuple[str, ...]] = []
    created_documents: list[dict] = []

    class NetworkCluster:
        namespace = "test-ns"

        def __init__(self):
            self.resources: dict[str, dict] = {}
            self.uid = 0

        def client_exec(self, _arguments, **_kwargs):
            return completed()

        def delete_with_uid_precondition(self, resource, uid):
            assert self.resources[resource]["metadata"]["uid"] == uid
            self.resources.pop(resource)
            return True

        def kubectl(self, arguments, **kwargs):
            operations.append(tuple(arguments))
            if arguments[0] == "create":
                document = yaml.safe_load(kwargs["input_text"])
                created_documents.append(document)
                resource = f"{document['kind'].lower()}/{document['metadata']['name']}"
                self.uid += 1
                document["metadata"]["uid"] = f"uid-{self.uid}"
                self.resources[resource] = document
                return completed(json.dumps(document))
            if arguments[0] == "get":
                document = self.resources.get(arguments[1])
                return completed(json.dumps(document) if document else "")
            if arguments[0] == "exec" and arguments[-1] != "2379":
                return completed("EXPECTED_NETWORK_DENIAL\n", returncode=23)
            return completed()

    cluster = NetworkCluster()
    proof = sweep_network.prove_network_isolation(cluster, DIGEST_IMAGE)
    assert proof["status"] == "passed"
    assert proof["negative_egress_control"] == "etcd:2379"
    assert cluster.resources == {}
    assert ("create", "-f", "-", "-o", "json") in operations
    probe_policy = next(
        document
        for document in created_documents
        if document["kind"] == "NetworkPolicy"
    )
    permitted_ports = {
        port["port"]
        for rule in probe_policy["spec"]["egress"]
        for port in rule["ports"]
    }
    assert permitted_ports == {53, 2379}
    assert 6379 not in permitted_ports
    assert 26379 not in permitted_ports


def test_dynamo_application_canary_checks_router_and_tokenizer(monkeypatch) -> None:
    sizes = iter(("1", "2"))

    class CanaryCluster:
        def valkey(self, _host, command, *_arguments):
            assert command == "DBSIZE"
            return [next(sizes)]

        def client_exec(self, arguments, **_kwargs):
            assert arguments[-2].startswith("http://frontend")
            return completed(json.dumps({"choices": 1, "model": sweep_model.MODEL}))

    monkeypatch.setattr(
        sweep_ha, "sentinel_master", lambda *_: ("tokenizer-primary", 6379)
    )
    monkeypatch.setattr(sweep_ha, "registered_mocker_stats", lambda *_: [0, 10, 10])
    evidence = sweep_ha.dynamo_application_canary(
        CanaryCluster(), "http://frontend:8000", "runtime", 10
    )
    assert evidence["registered_index_stats"] == [0, 10, 10]
    assert evidence["tokenizer_dbsize_after"] > evidence["tokenizer_dbsize_before"]


def test_stack_ownership_preflight_rejects_unrelated_resources() -> None:
    class CollisionCluster:
        def kubectl(self, _arguments, **_kwargs):
            return completed(json.dumps({"metadata": {"labels": {}}}))

    with pytest.raises(RuntimeError, match="refusing to replace unowned"):
        sweep_cluster.assert_stack_ownership(CollisionCluster())

    sentinel_claim = "persistentvolumeclaim/sentinel-data-valkey-sweep-sentinel-0"

    class SentinelCollisionCluster:
        def kubectl(self, arguments, **_kwargs):
            if arguments[1] == sentinel_claim:
                return completed(json.dumps({"metadata": {"labels": {}}}))
            return completed()

    with pytest.raises(RuntimeError, match="sentinel-data-valkey-sweep-sentinel-0"):
        sweep_cluster.assert_stack_ownership(SentinelCollisionCluster())


def test_network_probe_does_not_delete_a_colliding_pod() -> None:
    operations: list[tuple[str, ...]] = []
    creates = 0

    class CollisionCluster:
        namespace = "test-ns"

        def __init__(self):
            self.resources: dict[str, dict] = {}

        def client_exec(self, _arguments, **_kwargs):
            return completed()

        def delete_with_uid_precondition(self, resource, uid):
            assert self.resources[resource]["metadata"]["uid"] == uid
            self.resources.pop(resource)
            return True

        def kubectl(self, arguments, **kwargs):
            nonlocal creates
            operations.append(tuple(arguments))
            if arguments[0] == "create":
                creates += 1
                if creates == 2:
                    raise subprocess.CalledProcessError(1, arguments)
                document = yaml.safe_load(kwargs["input_text"])
                resource = f"{document['kind'].lower()}/{document['metadata']['name']}"
                document["metadata"]["uid"] = "uid-policy"
                self.resources[resource] = document
                return completed(json.dumps(document))
            if arguments[0] == "get":
                document = self.resources.get(arguments[1])
                return completed(json.dumps(document) if document else "")
            return completed()

    cluster = CollisionCluster()
    with pytest.raises(subprocess.CalledProcessError):
        sweep_network.prove_network_isolation(cluster, DIGEST_IMAGE)
    assert cluster.resources == {}
    assert not [operation for operation in operations if operation[0] == "delete"]


def test_network_probe_cleans_up_after_create_response_reconciliation_fails() -> None:
    operations: list[tuple[str, ...]] = []

    class TransientReadCluster:
        namespace = "test-ns"

        def __init__(self):
            self.resources: dict[str, dict] = {}
            self.gets = 0

        def client_exec(self, _arguments, **_kwargs):
            return completed()

        def delete_with_uid_precondition(self, resource, uid):
            assert self.resources[resource]["metadata"]["uid"] == uid
            self.resources.pop(resource)
            return True

        def kubectl(self, arguments, **kwargs):
            operations.append(tuple(arguments))
            if arguments[0] == "create":
                document = yaml.safe_load(kwargs["input_text"])
                resource = f"{document['kind'].lower()}/{document['metadata']['name']}"
                document["metadata"]["uid"] = "uid-policy"
                self.resources[resource] = document
                return completed("truncated create response")
            if arguments[0] == "get":
                self.gets += 1
                if self.gets == 1:
                    raise OSError("transient API read failure")
                document = self.resources.get(arguments[1])
                return completed(json.dumps(document) if document else "")
            return completed()

    cluster = TransientReadCluster()
    with pytest.raises(OSError, match="transient API read failure"):
        sweep_network.prove_network_isolation(cluster, DIGEST_IMAGE)
    assert cluster.resources == {}
    assert not [operation for operation in operations if operation[0] == "delete"]


def test_network_probe_never_deletes_a_uid_replacement() -> None:
    probe_id = "probe-id"
    resource = "pod/valkey-sweep-network-probe-probe-id"
    original = {
        "metadata": {
            "uid": "original",
            "labels": {
                "app.kubernetes.io/part-of": sweep_model.PART_OF,
                "valkey-sweep-probe-id": probe_id,
            },
        }
    }

    class ReplacementCluster:
        def __init__(self):
            self.resources = {resource: original}

        def kubectl(self, arguments, **_kwargs):
            document = self.resources.get(arguments[1])
            return completed(json.dumps(document) if document else "")

        def delete_with_uid_precondition(self, _resource, uid):
            assert uid == "original"
            self.resources[resource] = {
                "metadata": {
                    "uid": "replacement",
                    "labels": original["metadata"]["labels"],
                }
            }
            return False

    cluster = ReplacementCluster()
    with pytest.raises(RuntimeError, match="refusing to delete replaced"):
        sweep_network._delete_probe_resource(cluster, resource, "original", probe_id)
    assert cluster.resources[resource]["metadata"]["uid"] == "replacement"


def test_uid_preconditioned_delete_posts_delete_options(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Proxy:
        def __enter__(self):
            return "http://127.0.0.1:12345"

        def __exit__(self, *_args):
            return False

    request = None

    def fake_urlopen(observed, **_kwargs):
        nonlocal request
        request = observed
        return Response()

    monkeypatch.setattr(sweep_cluster, "kubectl_proxy", Proxy)
    monkeypatch.setattr(sweep_cluster, "urlopen", fake_urlopen)
    assert sweep_cluster.delete_with_uid_precondition("test ns", "pod/test-pod", "uid")
    assert (
        request.full_url
        == "http://127.0.0.1:12345/api/v1/namespaces/test%20ns/pods/test-pod"
    )
    assert request.get_method() == "DELETE"
    assert json.loads(request.data) == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {"uid": "uid"},
    }


def test_methodology_binding_is_source_and_render_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        sweep_methodology,
        "_git",
        lambda *arguments: "" if arguments[0] == "status" else "c" * 40,
    )
    monkeypatch.setattr(
        sweep_methodology,
        "_git_bytes",
        lambda _command, revision_path: (
            sweep_methodology.REPOSITORY / revision_path.split(":", 1)[1]
        ).read_bytes(),
    )
    binding = sweep_methodology.methodology_binding(DIGEST_IMAGE)
    assert binding["git_revision"] == "c" * 40
    assert len(binding["files"]) == len(sweep_methodology.METHOD_FILES)
    sweep_methodology.verify_methodology(binding, "c" * 40)
    with pytest.raises(RuntimeError, match="does not match image core"):
        sweep_methodology.verify_methodology(binding, "d" * 40)
    monkeypatch.setattr(
        sweep_methodology,
        "_git",
        lambda *arguments: " M changed.py" if arguments[0] == "status" else "c" * 40,
    )
    with pytest.raises(RuntimeError, match="not a clean checkout"):
        sweep_methodology.methodology_binding(DIGEST_IMAGE)
