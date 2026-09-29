# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render an untimed staging Pod using the capture's exact CUDA installer.

The staging Pod executes only file copies and checksums. The PVC-sourced core
shim is recorded by SHA256SUMS alongside the two image-sourced binaries, so a
caller can verify bytes before mounting the bundle into a timed restore Pod.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

INSTALLER = "snapshot-cuda-install"
BUNDLE_VOLUME = "snapshot-cuda"
BUNDLE_FILES = ("cuda-checkpoint", "libcuinterpose.so", "libcuinterpose_core.so")


def validate_bundle_path(path):
    path = Path(path)
    if not path.is_absolute() or str(path) == "/" or ".." in path.parts:
        raise ValueError(
            "CUDA bundle must use an absolute, non-root experiment directory"
        )
    return str(path)


def use_preinstalled_cuda(pod, host_path):
    """Replace only installer lifecycle and bundle backing; retain all mounts."""
    host_path = validate_bundle_path(host_path)
    result = copy.deepcopy(pod)
    installers = [
        c for c in result["spec"].get("initContainers", []) if c["name"] == INSTALLER
    ]
    if len(installers) != 1:
        raise ValueError(
            "expected exactly one captured snapshot-cuda-install init container"
        )
    result["spec"]["initContainers"] = [
        c for c in result["spec"]["initContainers"] if c["name"] != INSTALLER
    ]
    volume = next(v for v in result["spec"]["volumes"] if v["name"] == BUNDLE_VOLUME)
    if "emptyDir" not in volume:
        raise ValueError("captured CUDA bundle must originate in an emptyDir")
    volume.clear()
    volume.update(name=BUNDLE_VOLUME, hostPath={"path": host_path, "type": "Directory"})
    mounted = False
    for container in result["spec"]["containers"] + result["spec"]["initContainers"]:
        for mount in container.get("volumeMounts", []):
            if mount["name"] == BUNDLE_VOLUME:
                mount["readOnly"] = True
                mounted = True
    if not mounted:
        raise ValueError("restored workload does not mount the CUDA bundle")
    return result


def render_staging_pod(source, host_path, node_name, name):
    host_path = validate_bundle_path(host_path)
    source_pods = [item for item in source["items"] if item["kind"] == "Pod"]
    if len(source_pods) != 1:
        raise ValueError("capture manifest must identify exactly one source Pod")
    source_pod = source_pods[0]
    original = next(
        c for c in source_pod["spec"]["initContainers"] if c["name"] == INSTALLER
    )
    if "@sha256:" not in original["image"]:
        raise ValueError("capture installer image must be pinned by digest")
    if original["command"] != ["/bin/sh", "-c"] or len(original.get("args", [])) != 1:
        raise ValueError(
            "unsupported captured installer command; cannot append checksums safely"
        )
    resources = original.get("resources", {})
    if resources.get("claims") or any(
        key.startswith("nvidia.com/")
        for kind in ("requests", "limits")
        for key in resources.get(kind, {})
    ):
        raise ValueError("staging installer must not request GPUs")
    container = copy.deepcopy(original)
    container["args"][0] += (
        " && cd /snapshot-cuda && sha256sum "
        + " ".join(BUNDLE_FILES)
        + " > SHA256SUMS && cat SHA256SUMS"
    )
    mounted = {mount["name"] for mount in container["volumeMounts"]}
    volumes = [
        copy.deepcopy(v) for v in source_pod["spec"]["volumes"] if v["name"] in mounted
    ]
    bundle = next(v for v in volumes if v["name"] == BUNDLE_VOLUME)
    bundle.clear()
    bundle.update(
        name=BUNDLE_VOLUME, hostPath={"path": host_path, "type": "DirectoryOrCreate"}
    )
    artifact_volume = next(v for v in volumes if v["name"] == "artifacts")
    if "persistentVolumeClaim" not in artifact_volume:
        raise ValueError(
            "capture installer must read its core shim from the artifacts PVC"
        )
    installer_hash = hashlib.sha256(
        json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    spec = {
        key: copy.deepcopy(source_pod["spec"][key])
        for key in (
            "runtimeClassName",
            "securityContext",
            "imagePullSecrets",
            "tolerations",
            "terminationGracePeriodSeconds",
        )
        if key in source_pod["spec"]
    }
    spec.update(
        nodeSelector={"kubernetes.io/hostname": node_name},
        restartPolicy="Never",
        containers=[container],
        volumes=volumes,
    )
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": source_pod["metadata"]["namespace"],
            "labels": {"experiment": "gms-v1-cuda-preinstall-0929"},
            "annotations": {
                "nvidia.com/gms-prototype-cuda-installer-sha256": installer_hash
            },
        },
        "spec": spec,
    }
    provenance = {
        "source_pod_name": source_pod["metadata"]["name"],
        "source_installer": original,
        "source_installer_sha256": installer_hash,
        "source_artifacts_volume": artifact_volume,
        "bundle_files": list(BUNDLE_FILES),
        "bundle_host_path": host_path,
        "node_name": node_name,
        "checksum_file": "/snapshot-cuda/SHA256SUMS",
        "limitation": "PVC source bytes must be compared to the qualified bundle after staging; the installer image digest alone does not pin the PVC shim.",
    }
    return pod, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--host-path", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--name", default="gms-v1-cuda-preinstall-0929")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_manifest.read_bytes()
    pod, provenance = render_staging_pod(
        json.loads(raw), args.host_path, args.node, args.name
    )
    provenance["source_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(pod, indent=2))
    args.provenance_output.write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
