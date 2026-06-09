# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants shared by Dynamo snapshot helpers."""

PODINFO_ROOT = "/etc/podinfo"
KUBERNETES_REQUIRED_PODINFO_FILES = {
    "DYN_NAMESPACE": "dyn_namespace",
    "DYN_COMPONENT": "dyn_component",
    "DYN_PARENT_DGD_K8S_NAME": "dyn_parent_dgd_k8s_name",
    "DYN_PARENT_DGD_K8S_NAMESPACE": "dyn_parent_dgd_k8s_namespace",
}
KUBERNETES_OPTIONAL_PODINFO_FILES = {
    "DYN_NAMESPACE_WORKER_SUFFIX": "dyn_namespace_worker_suffix",
}
SNAPSHOT_CONTROL_DIR_ENV = "DYN_SNAPSHOT_CONTROL_DIR"
SNAPSHOT_CONTROL_DIR = "/snapshot-control"
SNAPSHOT_RESTORE_CONTEXT_FILE = "restore-context.json"
SNAPSHOT_RESTORE_PLACEHOLDER_ENV = "DYN_SNAPSHOT_RESTORE_PLACEHOLDER"

# Must match snapshotprotocol.{SnapshotCompleteFile,RestoreCompleteFile,
# ReadyForCheckpointFile}.
SNAPSHOT_COMPLETE_FILE = "snapshot-complete"
RESTORE_COMPLETE_FILE = "restore-complete"
READY_FOR_CHECKPOINT_FILE = "ready-for-checkpoint"

RESTORE_RUNTIME_ENV_NAMES = {
    # Parsed Python runtime config that must also refresh the in-memory config
    # passed to create_runtime().
    "DYN_DISCOVERY_BACKEND",
    "DYN_REQUEST_PLANE",
    "DYN_EVENT_PLANE",
    # DistributedRuntime infrastructure env read after restore.
    "NATS_SERVER",
    "ETCD_ENDPOINTS",
    # Runtime system server/readiness env read after restore.
    "DYN_SYSTEM_PORT",
    "DYN_HEALTH_CHECK_ENABLED",
    "DYN_SYSTEM_STARTING_HEALTH_STATUS",
    "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
    "DYN_SYSTEM_HOST",
    "DYN_SYSTEM_HEALTH_PATH",
    "DYN_SYSTEM_LIVE_PATH",
    # Kubernetes discovery mode env read when the restored runtime registers.
    "DYN_KUBE_DISCOVERY_MODE",
    "CONTAINER_NAME",
    # Optional non-secret platform endpoints that may be consumed after restore.
    "MODEL_EXPRESS_URL",
    "PROMETHEUS_ENDPOINT",
}
