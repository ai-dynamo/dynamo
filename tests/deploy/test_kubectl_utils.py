# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the object lifetimes in tests/deploy/kubectl_utils.py, run
against a fake kubectl."""

import json
import re
import shlex
import subprocess
from typing import Any

import pytest
import yaml

from tests.deploy.kubectl_utils import (
    Kubectl,
    KubectlError,
    Owner,
    created,
    delete_leftovers,
    use_namespace,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

LABEL = "example.com/e2e-test"
OWNER = Owner(LABEL, "run-2")
EARLIER_RUN = {LABEL: "run-1"}
QUOTA_ERROR = (
    'Error from server (Forbidden): services "engine-1" is forbidden: '
    "exceeded quota: services"
)


def _document(kind: str, name: str) -> dict[str, Any]:
    api_version = "batch/v1" if kind == "Job" else "v1"
    return {"apiVersion": api_version, "kind": kind, "metadata": {"name": name}}


def _key(document: dict[str, Any]) -> str:
    return f"{document['kind']}/{document['metadata']['name']}"


class FakeKubectl(Kubectl):
    """Serves the helpers' kubectl commands from live objects, each "Kind/name"
    mapped to its labels.

    create adds the documents in order and, like kubectl, goes on past one
    that fails: one whose name is taken, or one in rejected, which maps
    "Kind/name" to the API server's error. With timed_out, create then raises
    TimeoutExpired, as when kubectl times out after the API server accepted
    the objects. With unreadable, get fails.
    """

    def __init__(
        self,
        live: dict[str, dict[str, str]] | None = None,
        *,
        rejected: dict[str, str] | None = None,
        timed_out: bool = False,
        unreadable: bool = False,
    ) -> None:
        super().__init__("e2e")
        self.live = dict(live or {})
        self.rejected = rejected or {}
        self.timed_out = timed_out
        self.unreadable = unreadable
        self.deleted: list[str] = []

    def run(
        self,
        *args: str,
        input_: str | None = None,
        check: bool = True,
        timeout: float = 0,
    ) -> subprocess.CompletedProcess[str]:
        documents = [
            document for document in yaml.safe_load_all(input_ or "") if document
        ]
        if args[0] == "create":
            stdout, stderr = self._create(documents, args, timeout)
        elif args[0] == "get":
            stdout, stderr = self._get(documents)
        elif args[0] == "delete":
            stdout, stderr = self._delete(documents)
        else:
            raise AssertionError(f"unexpected kubectl command: {args}")
        if check and stderr:
            raise KubectlError(f"kubectl {shlex.join(args)} exited with 1: {stderr}")
        return subprocess.CompletedProcess(args, 1 if stderr else 0, stdout, stderr)

    def _create(
        self, documents: list[dict[str, Any]], args: tuple[str, ...], timeout: float
    ) -> tuple[str, str]:
        errors = []
        for document in documents:
            key = _key(document)
            if key in self.live:
                errors.append(f'Error from server (AlreadyExists): "{key}" exists')
            elif key in self.rejected:
                errors.append(self.rejected[key])
            else:
                self.live[key] = dict(document["metadata"].get("labels", {}))
        if self.timed_out:
            raise subprocess.TimeoutExpired(["kubectl", *args], timeout)
        return "", "\n".join(errors)

    def _get(self, documents: list[dict[str, Any]]) -> tuple[str, str]:
        (document,) = documents
        if self.unreadable:
            return "", "Unable to connect to the server: dial tcp: i/o timeout"
        labels = self.live.get(_key(document))
        if labels is None:
            return "", f'Error from server (NotFound): "{_key(document)}" not found'
        name = document["metadata"]["name"]
        live = {**document, "metadata": {"name": name, "labels": labels}}
        return json.dumps({"apiVersion": "v1", "kind": "List", "items": [live]}), ""

    def _delete(self, documents: list[dict[str, Any]]) -> tuple[str, str]:
        for document in documents:
            if self.live.pop(_key(document), None) is not None:
                self.deleted.append(_key(document))
        return "", ""


def _services() -> list[dict[str, Any]]:
    return [_document("Service", "engine-0"), _document("Service", "engine-1")]


def test_created_deletes_its_objects_after_saving_diagnostics() -> None:
    kubectl = FakeKubectl()
    live_when_saved: list[set[str]] = []

    with created(
        kubectl,
        _services(),
        OWNER,
        before_delete=lambda: live_when_saved.append(set(kubectl.live)),
    ):
        assert kubectl.live == {
            "Service/engine-0": {LABEL: "run-2"},
            "Service/engine-1": {LABEL: "run-2"},
        }

    assert live_when_saved == [{"Service/engine-0", "Service/engine-1"}]
    assert kubectl.deleted == ["Service/engine-0", "Service/engine-1"]
    assert kubectl.cleanup_errors == []


def test_created_deletes_what_a_partly_failed_create_made() -> None:
    kubectl = FakeKubectl(rejected={"Service/engine-1": QUOTA_ERROR})

    with pytest.raises(KubectlError, match="exceeded quota"):
        with created(kubectl, _services(), OWNER):
            pytest.fail("created() entered its body after a failed create")

    assert kubectl.deleted == ["Service/engine-0"]
    assert kubectl.live == {}
    assert kubectl.cleanup_errors == []


def test_created_deletes_what_the_api_server_accepted_before_a_timeout() -> None:
    kubectl = FakeKubectl(timed_out=True)

    with pytest.raises(subprocess.TimeoutExpired):
        with created(kubectl, [_document("Job", "setup")], OWNER):
            pytest.fail("created() entered its body after a failed create")

    assert kubectl.deleted == ["Job/setup"]
    assert kubectl.cleanup_errors == []


@pytest.mark.parametrize("labels", [{}, EARLIER_RUN], ids=["unlabeled", "earlier-run"])
def test_created_keeps_an_object_that_already_had_the_name(
    labels: dict[str, str],
) -> None:
    kubectl = FakeKubectl({"Service/engine-0": labels})

    with pytest.raises(KubectlError, match="AlreadyExists"):
        with created(kubectl, _services(), OWNER):
            pytest.fail("created() entered its body after a failed create")

    assert kubectl.live == {"Service/engine-0": labels}
    assert kubectl.deleted == ["Service/engine-1"]


def test_a_failed_cleanup_does_not_replace_the_create_error() -> None:
    kubectl = FakeKubectl(rejected={"Service/engine-1": QUOTA_ERROR}, unreadable=True)

    with pytest.raises(KubectlError, match="exceeded quota"):
        with created(kubectl, _services(), OWNER):
            pytest.fail("created() entered its body after a failed create")

    (error,) = kubectl.cleanup_errors
    assert error.startswith("deleting Service/engine-0, Service/engine-1: ")
    assert "Unable to connect to the server" in error


def test_use_namespace_deletes_only_a_namespace_that_it_created() -> None:
    kubectl = FakeKubectl()
    with use_namespace(kubectl, OWNER) as is_new:
        assert is_new
        assert kubectl.live == {"Namespace/e2e": {LABEL: "run-2"}}
    assert kubectl.deleted == ["Namespace/e2e"]

    kubectl = FakeKubectl({"Namespace/e2e": {}})
    with use_namespace(kubectl, OWNER) as is_new:
        assert not is_new
    assert kubectl.deleted == []


def _creation_order() -> list[list[dict[str, Any]]]:
    return [
        [_document("PersistentVolumeClaim", "data")],
        [_document("Job", "setup")],
        _services(),
    ]


def test_delete_leftovers_deletes_earlier_runs_objects_in_reverse() -> None:
    kubectl = FakeKubectl(
        {"PersistentVolumeClaim/data": EARLIER_RUN, "Service/engine-1": EARLIER_RUN}
    )

    delete_leftovers(kubectl, _creation_order(), OWNER)

    assert kubectl.deleted == ["Service/engine-1", "PersistentVolumeClaim/data"]


def test_delete_leftovers_deletes_nothing_if_an_object_lacks_the_label() -> None:
    kubectl = FakeKubectl(
        {"Service/engine-0": EARLIER_RUN, "PersistentVolumeClaim/data": {}}
    )

    with pytest.raises(AssertionError, match="PersistentVolumeClaim/data"):
        delete_leftovers(kubectl, _creation_order(), OWNER)

    assert kubectl.deleted == []


def test_each_run_gets_its_own_valid_label_value() -> None:
    runs = {Owner.new(LABEL).run for _ in range(2)}

    assert len(runs) == 2
    for run in runs:
        # The syntax and length limit of a Kubernetes label value.
        assert re.fullmatch(r"[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?", run)
