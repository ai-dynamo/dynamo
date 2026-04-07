# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from dynamo.common.utils import snapshot


@pytest.fixture(autouse=True)
def _clean_restore_env(monkeypatch):
    for var in (
        "DYN_NAMESPACE",
        "DYN_COMPONENT",
        "DYN_PARENT_DGD_K8S_NAME",
        "DYN_PARENT_DGD_K8S_NAMESPACE",
        "DYN_NAMESPACE_WORKER_SUFFIX",
        "DYN_DISCOVERY_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)


def test_reload_snapshot_restore_identity_keeps_non_kubernetes_backend(
    tmp_path,
    monkeypatch,
):
    podinfo_root = tmp_path / "podinfo"
    podinfo_root.mkdir()
    monkeypatch.setattr(snapshot, "PODINFO_ROOT", str(podinfo_root))

    namespace, backend = snapshot.reload_snapshot_restore_identity(
        "restore-ns",
        "etcd",
    )

    assert namespace == "restore-ns"
    assert backend == "etcd"
    assert "DYN_DISCOVERY_BACKEND" not in os.environ


def test_reload_snapshot_restore_identity_requires_podinfo_for_kubernetes(
    tmp_path,
    monkeypatch,
):
    podinfo_root = tmp_path / "podinfo"
    podinfo_root.mkdir()
    monkeypatch.setattr(snapshot, "PODINFO_ROOT", str(podinfo_root))

    with pytest.raises(RuntimeError, match="snapshot restore requires"):
        snapshot.reload_snapshot_restore_identity("ignored", "kubernetes")


def test_reload_snapshot_restore_identity_reads_kubernetes_podinfo(
    tmp_path,
    monkeypatch,
):
    podinfo_root = tmp_path / "podinfo"
    podinfo_root.mkdir()
    monkeypatch.setattr(snapshot, "PODINFO_ROOT", str(podinfo_root))

    for filename, value in (
        ("dyn_namespace", "restore-ns"),
        ("dyn_component", "worker"),
        ("dyn_parent_dgd_k8s_name", "graph"),
        ("dyn_parent_dgd_k8s_namespace", "restore"),
        ("dyn_namespace_worker_suffix", "abcd"),
    ):
        (podinfo_root / filename).write_text(value, encoding="utf-8")

    namespace, backend = snapshot.reload_snapshot_restore_identity(
        "ignored",
        "kubernetes",
    )

    assert namespace == "restore-ns-abcd"
    assert backend == "kubernetes"
    assert os.environ["DYN_NAMESPACE"] == "restore-ns"
    assert os.environ["DYN_COMPONENT"] == "worker"
    assert os.environ["DYN_PARENT_DGD_K8S_NAME"] == "graph"
    assert os.environ["DYN_PARENT_DGD_K8S_NAMESPACE"] == "restore"
    assert os.environ["DYN_NAMESPACE_WORKER_SUFFIX"] == "abcd"
    assert os.environ["DYN_DISCOVERY_BACKEND"] == "kubernetes"
