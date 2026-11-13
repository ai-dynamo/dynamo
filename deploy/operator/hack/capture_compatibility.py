#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a compatibility snapshot by running released operators on an authored DGD.

Each scenario under testdata/upgrade/compatibility records how pinned released
operators persisted one workload shape that the canonical history matrix does
not cover. The replay seeds that graph and checks that the current operator
leaves it alone. Run one scenario at a time:

    KUBEBUILDER_ASSETS=/abs/envtest/bin python3 hack/capture_compatibility.py \
        frontend-sidecar-probes --output /new/staging/directory
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml
from bootstrap_upgrade_fixtures import OPERATOR, RELEASES, ROOT

ORIGIN_ANNOTATION = "nvidia.com/dynamo-operator-origin-version"
WORKER_HASH_ANNOTATION = "nvidia.com/current-worker-hash-v2"
RELEASE_1_6_0 = "701da25292ecca8f64d770cc0b17d389200019a1"
HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)
CAPTURE_SOURCES = (
    "hack/capture_compatibility.py",
    "hack/upgrade_capture_test.go.tmpl",
    "internal/testing/golden/capture.go",
    "internal/testing/golden/snapshot.go",
    "hack/normalize_upgrade_snapshot/main.go",
)


@dataclass
class Scenario:
    """One compatibility snapshot: which releases run, what they create, what to record."""

    name: str
    releases: dict  # ordered version -> pinned revision
    creation: str
    scope: str
    create: Callable[[], dict]
    origin: str | None  # expected origin annotation after every stage; None means absent
    mutating_on_create: bool = True
    observe: Callable[[list], dict] = field(default=lambda objects: {})

    @property
    def namespace(self):
        return f"upgrade-compatibility-{self.name}"


def archive_release(revision, paths):
    """Fetch a missing pinned release object without changing the checkout."""
    if subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    ).returncode:
        subprocess.run(
            [
                "git",
                "fetch",
                "--no-tags",
                "https://github.com/ai-dynamo/dynamo.git",
                revision,
            ],
            cwd=ROOT,
            check=True,
        )
    return subprocess.check_output(["git", "archive", revision, *paths], cwd=ROOT)


def export_snapshot(source, destination):
    """Publish observations with the same Go writer as release advancement."""
    subprocess.run(
        [
            "go",
            "run",
            "./hack/normalize_upgrade_snapshot",
            str(source),
            str(destination),
        ],
        cwd=OPERATOR,
        env=dict(os.environ, GOWORK="off"),
        check=True,
    )


def minor(version):
    return tuple(int(part) for part in version.split(".")[:2])


def dgd_of(objects):
    return next(o for o in objects if o["kind"] == "DynamoGraphDeployment")


def safe_members(archive, destination):
    """Accept only regular files and directories that resolve inside destination.

    This is the subset of the tarfile data filter that matters for a git archive:
    no links, no devices, and no path that escapes the checkout.
    """
    root = destination.resolve()
    members = []
    for member in archive.getmembers():
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"Refusing archive member of unexpected type: {member.name}"
            )
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(
                f"Refusing archive member outside the checkout: {member.name}"
            )
        members.append(member)
    return members


def render_harness(scenario, version):
    """Render the bootstrap observer for one released stage of scenario."""
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

    # Later stages attach to the first stage's API server; each stage knows its successor.
    versions = list(scenario.releases)
    successors = ", ".join(
        f'"{current}": "{following}"'
        for current, following in zip(versions, versions[1:])
    )
    template = template.replace(
        'if version == "1.4.0" {',
        "if next := map[string]string{" + successors + '}[version]; next != "" {',
    )
    template = template.replace(
        'cmd.Dir = os.Getenv("UPGRADE_CAPTURE_15_OPERATOR")',
        'cmd.Dir = filepath.Join(os.Getenv("UPGRADE_CAPTURE_TREES"), next, "deploy/operator")',
    )
    template = template.replace(
        '"UPGRADE_CAPTURE_VERSION=1.5.0"', '"UPGRADE_CAPTURE_VERSION="+next'
    )
    template = template.replace(
        'filepath.Join(root, "1.5.log")', 'filepath.Join(root, next+".log")'
    )

    # Mutating admission on CREATE stamps the origin; a legacy installation lacks it.
    if not scenario.mutating_on_create:
        template = template.replace("Mutating: true", "Mutating: target")
    if scenario.origin is None:
        template = template.replace(
            'captureWrite(t, filepath.Join(root, version+".json"), objects)',
            "require.NotContains(t, observed.Annotations, consts.KubeAnnotationDynamoOperatorOriginVersion); "
            'captureWrite(t, filepath.Join(root, version+".json"), objects)',
        )

    # Released API differences per operator version.
    scale = (
        "scaleClient,err:=env.ScaleClient();require.NoError(t,err);dgdOpts.ScaleClient=scaleClient"
        if version == "1.4.0"
        else ""
    )
    template = template.replace("// SCALE_CLIENT", scale)
    if minor(version) >= (1, 6):
        template = template.replace(
            "dynamo.PCSNameForDGD(observed.Name, observed.Spec.Components)",
            "dynamo.PCSNameForDGD(&observed, nil)",
        )
    return template, barrier


def prepare_release_tree(scenario, version, revision, work, provenance):
    """Extract the pinned release and install the rendered harness and Grove CRDs."""
    official = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/ai-dynamo/dynamo/commits/v{version}"], text=True
        )
    )
    if official["sha"] != revision:
        raise ValueError(f"Upstream v{version} differs from its pinned revision")
    paths = ["deploy/operator", "deploy/helm"]
    if version == "1.4.0":
        paths.append("deploy/snapshot")
    checkout = work / version
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive_release(revision, paths))) as archive:
        # The data filter exists from Python 3.11.4; older 3.11 patch releases reject the keyword.
        if hasattr(tarfile, "data_filter"):
            archive.extractall(checkout, filter="data")
        else:
            archive.extractall(checkout, members=safe_members(archive, checkout))
    operator = checkout / "deploy/operator"
    rendered, barrier = render_harness(scenario, version)
    (operator / "internal/controller/zz_capture_upgrade_test.go").write_text(rendered)
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
    shutil.copytree(
        Path(info["Dir"]) / "core/v1alpha1/crds",
        operator / "internal/controller/testing/grove.io",
        dirs_exist_ok=True,
    )
    provenance.setdefault("harness", {})[version] = hashlib.sha256(
        rendered.encode()
    ).hexdigest()
    provenance["reconcileBarrierSHA256"] = hashlib.sha256(barrier.encode()).hexdigest()
    return operator


def takeover_safety(objects):
    """The fields an unchanged workload must keep across released stages."""
    dgd = dgd_of(objects)
    pcs = next(o for o in objects if o["kind"] == "PodCliqueSet")
    return {
        "dgdUID": dgd["metadata"]["uid"],
        "spec": dgd["spec"],
        "workerHash": dgd["metadata"]["annotations"][WORKER_HASH_ANNOTATION],
        "pcsUID": pcs["metadata"]["uid"],
        "pcsGeneration": pcs["metadata"]["generation"],
        "rollingUpdate": dgd.get("status", {}).get("rollingUpdate"),
        "restart": dgd.get("status", {}).get("restart"),
        "updateStrategy": pcs["spec"].get("updateStrategy"),
        "templates": pcs["spec"]["template"]["cliques"],
    }


def capture(scenario, destination):
    if destination.exists():
        raise ValueError("Destination must be a new staging directory")
    destination.mkdir(parents=True)
    provenance = {
        "releases": dict(scenario.releases),
        "creation": scenario.creation,
        "scope": scenario.scope,
    }
    manifest = destination / "create.yaml"
    manifest.write_text(HEADER + yaml.safe_dump(scenario.create(), sort_keys=False))
    with tempfile.TemporaryDirectory(prefix=f"dynamo-{scenario.name}-") as temporary:
        work = Path(temporary)
        trees = {
            version: prepare_release_tree(scenario, version, revision, work, provenance)
            for version, revision in scenario.releases.items()
        }
        first = next(iter(scenario.releases))
        env = dict(
            os.environ,
            GOWORK="off",
            GOTOOLCHAIN="auto",
            USE_EXISTING_CLUSTER="false",
            UPGRADE_CAPTURE_ROOT=str(destination),
            UPGRADE_CAPTURE_VERSION=first,
            UPGRADE_CAPTURE_INPUT=str(manifest),
            UPGRADE_CAPTURE_TREES=str(work),
        )
        with (destination / f"{first}.log").open("w") as log:
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
                cwd=trees[first],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    # Every stage must keep the workload and record the expected origin.
    previous = None
    for version in scenario.releases:
        path = destination / f"{version}.json"
        objects = json.loads(path.read_text())
        dgd = dgd_of(objects)
        origin = dgd["metadata"]["annotations"].get(ORIGIN_ANNOTATION)
        if origin != scenario.origin:
            raise ValueError(
                f"{version} left origin {origin!r}, expected {scenario.origin!r}"
            )
        safety = takeover_safety(objects)
        if previous is not None and safety != previous:
            raise ValueError(f"Unchanged workload changed during {version} takeover")
        previous = safety
        provenance.setdefault("observations", {})[version] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "workerHash": safety["workerHash"],
            **scenario.observe(objects),
        }
    last = list(scenario.releases)[-1]
    path = destination / "input.yaml"
    export_snapshot(destination / f"{last}.json", path)
    provenance["inputSHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["captureSources"] = {
        name: hashlib.sha256((OPERATOR / name).read_bytes()).hexdigest()
        for name in CAPTURE_SOURCES
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


def missing_origin_dgd():
    """Authored CREATE intent for a 1.4 installation before mutating admission."""
    source = (
        OPERATOR
        / "internal/controller/testdata/upgrade/latest/sglang/aggregated/single/grove/input.yaml"
    )
    dgd = yaml.safe_load(source.read_text())
    dgd["metadata"] = {
        "name": "workers",
        "namespace": "upgrade-compatibility-missing-origin",
        "annotations": {"nvidia.com/enable-grove": "true"},
    }
    for component in dgd["spec"]["components"]:
        component["minAvailable"] = 1
        container = component["podTemplate"]["spec"]["containers"][0]
        container["image"] = container["image"].replace(":1.6.0", ":1.4.0")
        if component["type"] == "frontend":
            container["command"] = ["python3"]
            container["args"] = ["-m", "dynamo.frontend"]
    return dgd


SIDECAR = "sidecar-frontend"

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


def frontend_sidecar_probes_dgd():
    """A Grove worker with a frontend sidecar whose probes set only some fields."""
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": "workers",
            "namespace": "upgrade-compatibility-frontend-sidecar-probes",
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


def observe_sidecar_probes(objects):
    rendered = sidecar_probes(objects)
    if rendered != MERGED_PROBES:
        raise ValueError(
            "Released operator did not field-merge the sidecar probes as expected: "
            + json.dumps(rendered, sort_keys=True)
        )
    return {"sidecarProbes": rendered}


SCENARIOS = {
    "missing-origin": Scenario(
        name="missing-origin",
        releases={**RELEASES, "1.6.0": RELEASE_1_6_0},
        creation="1.4.0 CREATE before mutating admission is enabled",
        scope="installation-level missing origin; not pre-0.7.1 ancestry",
        create=missing_origin_dgd,
        origin=None,
        mutating_on_create=False,
    ),
    "frontend-sidecar-probes": Scenario(
        name="frontend-sidecar-probes",
        releases={"1.6.0": RELEASE_1_6_0},
        creation="1.6.0 CREATE with mutating admission; origin stamped by the released operator",
        scope="frontendSidecar container with partially authored probes, field-merged by 1.6.0",
        create=frontend_sidecar_probes_dgd,
        origin="1.6.0",
        observe=observe_sidecar_probes,
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    capture(SCENARIOS[arguments.scenario], arguments.output.resolve())
