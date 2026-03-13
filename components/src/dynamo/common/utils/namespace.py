# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

_DYN_NAMESPACE_PODINFO_FILE = Path("/etc/podinfo/dyn_namespace")
_DYN_NAMESPACE_WORKER_SUFFIX_PODINFO_FILE = Path(
    "/etc/podinfo/dyn_namespace_worker_suffix"
)
_DYN_SYSTEM_PORT_ENV = "DYN_SYSTEM_PORT"
_DEFAULT_WORKER_SYSTEM_PORT = "9090"


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


def reload_snapshot_restore_identity(
    namespace: Optional[str], discovery_backend: str
) -> tuple[str, str]:
    """Reload worker identity after snapshot restore on Kubernetes."""
    if not _DYN_NAMESPACE_PODINFO_FILE.is_file():
        return get_worker_namespace(namespace), discovery_backend

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

    if int(os.environ.get(_DYN_SYSTEM_PORT_ENV, "-1")) < 0:
        os.environ[_DYN_SYSTEM_PORT_ENV] = _DEFAULT_WORKER_SYSTEM_PORT

    return get_worker_namespace(namespace, suffix), "kubernetes"
