#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a frontend-sidecar workload with partial probes as released 1.6.0 rendered it."""

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml
from bootstrap_upgrade_fixtures import OPERATOR
from capture_missing_origin import archive_release, export_snapshot

RELEASE = "1.6.0"
REVISION = "701da25292ecca8f64d770cc0b17d389200019a1"
NAMESPACE = "upgrade-compatibility-frontend-sidecar-probes"
SIDECAR = "sidecar-frontend"
HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)

# Operator 1.6.0 field-merged these frontend defaults into partially authored
# sidecar probes. Main replaces an authored probe as a complete value, so the
# merged fields are what an operator-only upgrade must preserve.
MERGED_PROBES = {
    "readinessProbe": {
        "httpGet": {"path": "/health", "port": "http"},
        "initialDelaySeconds": 30,
        "periodSeconds": 10,
        "timeoutSeconds": 3,
        "failureThreshold": 3,
    },
    "livenessProbe": {
        "httpGet": {"path": "/live", "port": "http"},
        "initialDelaySeconds": 15,
        "periodSeconds": 30,
        "timeoutSeconds": 1,
        "failureThreshold": 3,
    },
}


def authored_dgd():
    """The CREATE intent: a Grove worker with a frontend sidecar and partial probes."""
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": "workers",
            "namespace": NAMESPACE,
            "annotations": {"nvidia.com/enable-grove": "true"},
        },
        "spec": {
            "backendFramework": "vllm",
            "components": [
                {
                    "name": "frontend",
                    "type": "frontend",
                    "replicas": 1,
                    "minAvailable": 1,
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.6.0",
                                    "command": ["python3"],
                                    "args": ["-m", "dynamo.frontend"],
                                }
                            ]
                        }
                    },
                },
                {
                    "name": "worker",
                    "type": "worker",
                    "replicas": 2,
                    "minAvailable": 1,
                    "frontendSidecar": SIDECAR,
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0",
                                    "command": ["python3", "-m", "dynamo.vllm"],
                                    "args": ["--model", "Qwen/Qwen3-0.6B"],
                                    "resources": {"limits": {"nvidia.com/gpu": "1"}},
                                },
                                {
                                    "name": SIDECAR,
                                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0",
                                    "args": [
                                        "-m",
                                        "dynamo.frontend",
                                        "--router-mode",
                                        "direct",
                                    ],
                                    "readinessProbe": {
                                        "httpGet": {"path": "/health", "port": "http"},
                                        "initialDelaySeconds": 30,
                                    },
                                    "livenessProbe": {
                                        "httpGet": {"path": "/live", "port": "http"},
                                        "periodSeconds": 30,
                                    },
                                },
                            ]
                        }
                    },
                },
            ],
        },
    }


def sidecar_probes(objects):
    """Return the rendered sidecar probes from the worker clique template."""
    pcs = next(o for o in objects if o["kind"] == "PodCliqueSet")
    for clique in pcs["spec"]["template"]["cliques"]:
        if clique.get("labels", {}).get("nvidia.com/dynamo-component") != "worker":
            continue
        for container in clique["spec"]["podSpec"]["containers"]:
            if container["name"] == SIDECAR:
                return {
                    key: container[key] for key in ("readinessProbe", "livenessProbe")
                }
    raise ValueError("worker clique has no frontend sidecar container")


def render_harness():
    """Reuse the bootstrap observer for one released 1.6.0 stage."""
    template = (OPERATOR / "hack/upgrade_capture_test.go.tmpl").read_text()
    barrier_source = (
        OPERATOR / "internal/controller/upgrade_clustertest_test.go"
    ).read_text()
    barrier = barrier_source.split("func completedUpgradeReconciles", 1)[1].split(
        "\nfunc captureUpgrade(", 1
    )[0]
    template += "\nfunc captureReconciles" + barrier
    template = template.replace(
        'metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"',
        'metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"\n"sigs.k8s.io/controller-runtime/pkg/metrics"',
    )
    template = template.replace(
        "stop := captureStartManager(t, env, dgd.Namespace)",
        "baseline := captureReconciles(t); stop := captureStartManager(t, env, dgd.Namespace)",
    )
    template = template.replace(
        "require.Eventually(t, func() bool {",
        "require.Eventually(t, func() bool { if captureReconciles(t) <= baseline { return false };",
    )
    template = template.replace("// SCALE_CLIENT", "")
    template = template.replace(
        "dynamo.PCSNameForDGD(observed.Name, observed.Spec.Components)",
        "dynamo.PCSNameForDGD(&observed, nil)",
    )
    return template, barrier


def capture(destination):
    if destination.exists():
        raise ValueError("Destination must be a new staging directory")
    destination.mkdir(parents=True)
    official = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/ai-dynamo/dynamo/commits/v{RELEASE}"], text=True
        )
    )
    if official["sha"] != REVISION:
        raise ValueError(f"Upstream v{RELEASE} differs from its pinned revision")
    rendered, barrier = render_harness()
    provenance = {
        "releases": {RELEASE: REVISION},
        "creation": f"{RELEASE} CREATE with mutating admission; origin stamped by the released operator",
        "scope": "frontendSidecar container with partially authored probes, field-merged by 1.6.0",
        "harness": {RELEASE: hashlib.sha256(rendered.encode()).hexdigest()},
    }
    scenario = destination / "create.yaml"
    scenario.write_text(HEADER + yaml.safe_dump(authored_dgd(), sort_keys=False))
    with tempfile.TemporaryDirectory(
        prefix="dynamo-frontend-sidecar-probes-"
    ) as temporary:
        work = Path(temporary)
        data = archive_release(REVISION, ["deploy/operator", "deploy/helm"])
        checkout = work / RELEASE
        checkout.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(checkout, filter="data")
        operator = checkout / "deploy/operator"
        (operator / "internal/controller/zz_capture_upgrade_test.go").write_text(
            rendered
        )
        observer = (OPERATOR / "internal/testing/golden/capture.go").read_text()
        (operator / "internal/controller/zz_capture_owned_test.go").write_text(
            observer.replace("package golden", "package controller", 1)
        )
        env = dict(os.environ, GOWORK="off", GOTOOLCHAIN="auto")
        info = json.loads(
            subprocess.check_output(
                [
                    "go",
                    "mod",
                    "download",
                    "-json",
                    "github.com/ai-dynamo/grove/operator/api",
                ],
                cwd=operator,
                env=env,
            )
        )
        import shutil

        shutil.copytree(
            Path(info["Dir"]) / "core/v1alpha1/crds",
            operator / "internal/controller/testing/grove.io",
            dirs_exist_ok=True,
        )
        env.update(
            USE_EXISTING_CLUSTER="false",
            UPGRADE_CAPTURE_ROOT=str(destination),
            UPGRADE_CAPTURE_VERSION=RELEASE,
            UPGRADE_CAPTURE_INPUT=str(scenario),
            UPGRADE_CAPTURE_TREES=str(work),
        )
        with (destination / f"{RELEASE}.log").open("w") as log:
            subprocess.run(
                [
                    "go",
                    "test",
                    "-p",
                    "2",
                    "./internal/controller",
                    "-run",
                    "^TestCaptureUpgradeBaseline$",
                    "-count=1",
                    "-v",
                    "-timeout",
                    "15m",
                ],
                cwd=operator,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    observed = destination / f"{RELEASE}.json"
    objects = json.loads(observed.read_text())
    dgd = next(o for o in objects if o["kind"] == "DynamoGraphDeployment")
    origin = dgd["metadata"]["annotations"].get(
        "nvidia.com/dynamo-operator-origin-version"
    )
    if origin != RELEASE:
        raise ValueError(
            f"Released admission stamped origin {origin!r}, expected {RELEASE!r}"
        )
    rendered_probes = sidecar_probes(objects)
    if rendered_probes != MERGED_PROBES:
        raise ValueError(
            "Released 1.6.0 did not field-merge the sidecar probes as expected: "
            + json.dumps(rendered_probes, sort_keys=True)
        )
    provenance["observations"] = {
        RELEASE: {
            "sha256": hashlib.sha256(observed.read_bytes()).hexdigest(),
            "workerHash": dgd["metadata"]["annotations"][
                "nvidia.com/current-worker-hash-v2"
            ],
            "sidecarProbes": rendered_probes,
        }
    }
    path = destination / "input.yaml"
    export_snapshot(observed, path)
    provenance["inputSHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["reconcileBarrierSHA256"] = hashlib.sha256(barrier.encode()).hexdigest()
    for name in (
        "hack/capture_frontend_sidecar_probes.py",
        "hack/capture_missing_origin.py",
        "hack/upgrade_capture_test.go.tmpl",
        "internal/testing/golden/capture.go",
        "internal/testing/golden/snapshot.go",
        "hack/normalize_upgrade_snapshot/main.go",
    ):
        provenance.setdefault("captureSources", {})[name] = hashlib.sha256(
            (OPERATOR / name).read_bytes()
        ).hexdigest()
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args().output.resolve())
