# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

_DYN_NAMESPACE_PODINFO_FILE = Path("/etc/podinfo/dyn_namespace")
_DYN_NAMESPACE_WORKER_SUFFIX_PODINFO_FILE = Path(
    "/etc/podinfo/dyn_namespace_worker_suffix"
)
_DYN_COMPONENT_PODINFO_FILE = Path("/etc/podinfo/dyn_component")
_DYN_PARENT_DGD_NAME_PODINFO_FILE = Path("/etc/podinfo/dyn_parent_dgd_k8s_name")
_DYN_PARENT_DGD_NAMESPACE_PODINFO_FILE = Path(
    "/etc/podinfo/dyn_parent_dgd_k8s_namespace"
)


def get_worker_namespace(
    namespace: Optional[str] = None, suffix: Optional[str] = None
) -> str:
    """Get the Dynamo namespace for a worker.

    Uses the provided namespace, then DYN_NAMESPACE, then "dynamo". If a suffix is
    provided, or DYN_NAMESPACE_WORKER_SUFFIX is set, it is appended as
    "{namespace}-{suffix}" to support multiple sets of workers for the same model.
    """
    if not namespace:
        namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    if suffix is None:
        suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


def reload_snapshot_restore_identity() -> tuple[str, str]:
    """Reload worker identity after snapshot restore on Kubernetes."""
    if not _DYN_NAMESPACE_PODINFO_FILE.is_file():
        raise RuntimeError("snapshot restore requires /etc/podinfo/dyn_namespace")

    namespace = _DYN_NAMESPACE_PODINFO_FILE.read_text(encoding="utf-8").strip()
    if not namespace:
        raise RuntimeError(
            "snapshot restore requires a non-empty /etc/podinfo/dyn_namespace"
        )

    suffix = None
    if _DYN_NAMESPACE_WORKER_SUFFIX_PODINFO_FILE.is_file():
        suffix = _DYN_NAMESPACE_WORKER_SUFFIX_PODINFO_FILE.read_text(
            encoding="utf-8"
        ).strip()
        if not suffix:
            suffix = None

    component = None
    if _DYN_COMPONENT_PODINFO_FILE.is_file():
        component = _DYN_COMPONENT_PODINFO_FILE.read_text(encoding="utf-8").strip()
        if not component:
            component = None

    parent_dgd_name = None
    if _DYN_PARENT_DGD_NAME_PODINFO_FILE.is_file():
        parent_dgd_name = _DYN_PARENT_DGD_NAME_PODINFO_FILE.read_text(
            encoding="utf-8"
        ).strip()
        if not parent_dgd_name:
            parent_dgd_name = None

    parent_dgd_namespace = None
    if _DYN_PARENT_DGD_NAMESPACE_PODINFO_FILE.is_file():
        parent_dgd_namespace = _DYN_PARENT_DGD_NAMESPACE_PODINFO_FILE.read_text(
            encoding="utf-8"
        ).strip()
        if not parent_dgd_namespace:
            parent_dgd_namespace = None

    os.environ["DYN_NAMESPACE"] = namespace
    if suffix is None:
        os.environ.pop("DYN_NAMESPACE_WORKER_SUFFIX", None)
    else:
        os.environ["DYN_NAMESPACE_WORKER_SUFFIX"] = suffix
    if component is None:
        os.environ.pop("DYN_COMPONENT", None)
    else:
        os.environ["DYN_COMPONENT"] = component
    if parent_dgd_name is None:
        os.environ.pop("DYN_PARENT_DGD_K8S_NAME", None)
    else:
        os.environ["DYN_PARENT_DGD_K8S_NAME"] = parent_dgd_name
    if parent_dgd_namespace is None:
        os.environ.pop("DYN_PARENT_DGD_K8S_NAMESPACE", None)
    else:
        os.environ["DYN_PARENT_DGD_K8S_NAMESPACE"] = parent_dgd_namespace
    os.environ["DYN_DISCOVERY_BACKEND"] = "kubernetes"

    return get_worker_namespace(namespace, suffix), "kubernetes"
