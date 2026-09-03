# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path helpers for DGDR model-cache PVC configuration."""


def normalize_model_cache_path(
    pvc_mount_path: str | None,
    pvc_model_path: str | None,
) -> str:
    """Resolve a DGDR model-cache path to the container-visible model path.

    ``pvcModelPath`` is normally relative to the PVC. For compatibility, a leading
    slash is still treated as PVC-relative unless the value begins with the
    configured mount path, in which case it is an already container-visible path.
    To address a PVC directory that happens to match the mount-path prefix, use the
    unambiguous relative form without a leading slash.
    """
    mount = (pvc_mount_path or "").rstrip("/")
    raw_path = (pvc_model_path or "").strip()
    if not raw_path:
        return mount

    absolute_path = raw_path.rstrip("/")
    if mount and (absolute_path == mount or absolute_path.startswith(f"{mount}/")):
        return absolute_path

    sub_path = raw_path.lstrip("/")
    if not sub_path:
        return mount
    return f"{mount}/{sub_path}"


def model_cache_path_in_pvc(
    pvc_mount_path: str | None,
    pvc_model_path: str | None,
) -> str | None:
    """Return the model path form expected by PVC-relative generators.

    AIC's Kubernetes generator expects ``k8s_model_path_in_pvc`` to be relative
    to the PVC mount. Preserve existing relative and legacy leading-slash inputs,
    but strip the mount prefix from already container-visible paths. A path inside
    the PVC that matches the mount prefix must use the relative form without a
    leading slash. Return ``None`` when no model path was provided so callers
    preserve the prior unset-state behavior. Return ``"."`` when the model path
    explicitly points at the PVC mount root, which keeps AIC from falling back to
    its HF_HOME default while still resolving under the mount.
    """
    mount = (pvc_mount_path or "").rstrip("/")
    raw_path = (pvc_model_path or "").strip()
    if not raw_path:
        return None

    absolute_path = raw_path.rstrip("/")
    if mount and absolute_path == mount:
        return "."
    if mount and absolute_path.startswith(f"{mount}/"):
        return absolute_path[len(mount) :].lstrip("/")
    return raw_path
