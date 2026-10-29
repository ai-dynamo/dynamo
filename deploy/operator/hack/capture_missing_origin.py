#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture an unstamped installation's 1.4 -> 1.5 -> 1.6 resource graph."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml
from bootstrap_upgrade_fixtures import OPERATOR, RELEASES, ROOT, dump


def capture(destination):
    if destination.exists():
        raise ValueError("Destination must be a new staging directory")
    destination.mkdir(parents=True)
    releases = {**RELEASES, "1.6.0": "701da25292ecca8f64d770cc0b17d389200019a1"}
    provenance = {
        "releases": releases,
        "creation": "1.4.0 CREATE before mutating admission is enabled",
        "scope": "installation-level missing origin; not pre-0.7.1 ancestry",
    }
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
    # Reuse the bootstrap observer, extending its same-API-server chain by one minor.
    template = template.replace("Mutating: true", "Mutating: target")
    template = template.replace('if version == "1.4.0" {', 'if version != "1.6.0" {')
    template = template.replace(
        'cmd.Dir = os.Getenv("UPGRADE_CAPTURE_15_OPERATOR")',
        'next := "1.5.0"; if version == "1.5.0" { next = "1.6.0" }; '
        'cmd.Dir = filepath.Join(os.Getenv("UPGRADE_CAPTURE_TREES"), next, "deploy/operator")',
    )
    template = template.replace(
        '"UPGRADE_CAPTURE_VERSION=1.5.0"', '"UPGRADE_CAPTURE_VERSION="+next'
    )
    template = template.replace(
        'filepath.Join(root, "1.5.log")', 'filepath.Join(root, next+".log")'
    )
    template = template.replace(
        'captureWrite(t, filepath.Join(root, version+".json"), objects)',
        "require.NotContains(t, observed.Annotations, consts.KubeAnnotationDynamoOperatorOriginVersion); "
        'captureWrite(t, filepath.Join(root, version+".json"), objects)',
    )
    with tempfile.TemporaryDirectory(prefix="dynamo-missing-origin-") as temporary:
        work = Path(temporary)
        for version, revision in releases.items():
            official = json.loads(
                subprocess.check_output(
                    ["gh", "api", f"repos/ai-dynamo/dynamo/commits/v{version}"],
                    text=True,
                )
            )
            if official["sha"] != revision:
                raise ValueError(
                    f"Upstream v{version} differs from its pinned revision"
                )
            paths = ["deploy/operator", "deploy/helm"]
            if version == "1.4.0":
                paths.append("deploy/snapshot")
            data = subprocess.check_output(
                ["git", "archive", revision, *paths], cwd=ROOT
            )
            checkout = work / version
            checkout.mkdir()
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                archive.extractall(checkout, filter="data")
            operator = checkout / "deploy/operator"
            scale = (
                "scaleClient,err:=env.ScaleClient();require.NoError(t,err);dgdOpts.ScaleClient=scaleClient"
                if version == "1.4.0"
                else ""
            )
            rendered = template.replace("// SCALE_CLIENT", scale)
            if version == "1.6.0":
                rendered = rendered.replace(
                    "dynamo.PCSNameForDGD(observed.Name, observed.Spec.Components)",
                    "dynamo.PCSNameForDGD(&observed, nil)",
                )
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
            shutil.copytree(
                Path(info["Dir"]) / "core/v1alpha1/crds",
                operator / "internal/controller/testing/grove.io",
                dirs_exist_ok=True,
            )
            provenance.setdefault("harness", {})[version] = hashlib.sha256(
                rendered.encode()
            ).hexdigest()
        # This is authored CREATE intent, not an edited historical snapshot.
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
        scenario = destination / "create.yaml"
        scenario.write_text(
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            + yaml.safe_dump(dgd, sort_keys=False)
        )
        env = dict(
            os.environ,
            GOWORK="off",
            GOTOOLCHAIN="auto",
            USE_EXISTING_CLUSTER="false",
            UPGRADE_CAPTURE_ROOT=str(destination),
            UPGRADE_CAPTURE_VERSION="1.4.0",
            UPGRADE_CAPTURE_INPUT=str(scenario),
            UPGRADE_CAPTURE_TREES=str(work),
        )
        with (destination / "1.4.0.log").open("w") as log:
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
                cwd=work / "1.4.0/deploy/operator",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    previous = None
    for version in releases:
        path = destination / f"{version}.json"
        objects = json.loads(path.read_text())
        dgd = next(o for o in objects if o["kind"] == "DynamoGraphDeployment")
        if (
            "nvidia.com/dynamo-operator-origin-version"
            in dgd["metadata"]["annotations"]
        ):
            raise ValueError(f"Origin was materialized by {version}")
        pcs = next(o for o in objects if o["kind"] == "PodCliqueSet")
        safety = {
            "dgdUID": dgd["metadata"]["uid"],
            "spec": dgd["spec"],
            "workerHash": dgd["metadata"]["annotations"][
                "nvidia.com/current-worker-hash-v2"
            ],
            "pcsUID": pcs["metadata"]["uid"],
            "pcsGeneration": pcs["metadata"]["generation"],
            "rollingUpdate": dgd.get("status", {}).get("rollingUpdate"),
            "restart": dgd.get("status", {}).get("restart"),
            "updateStrategy": pcs["spec"].get("updateStrategy"),
            "templates": pcs["spec"]["template"]["cliques"],
        }
        if previous is not None and safety != previous:
            raise ValueError(
                f"Unchanged legacy workload changed during {version} takeover"
            )
        previous = safety
        provenance.setdefault("observations", {})[version] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "workerHash": dgd["metadata"]["annotations"][
                "nvidia.com/current-worker-hash-v2"
            ],
        }
    dump(destination / "input.yaml", objects)
    # Use this command's name in the generated artifact, rather than the shared writer's bootstrap name.
    path = destination / "input.yaml"
    path.write_text(
        path.read_text().replace(
            "Generated by bootstrap_upgrade_fixtures.py",
            "Generated by capture_missing_origin.py",
        )
    )
    provenance["inputSHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["reconcileBarrierSHA256"] = hashlib.sha256(barrier.encode()).hexdigest()
    for name in (
        "hack/capture_missing_origin.py",
        "hack/upgrade_capture_test.go.tmpl",
        "internal/testing/golden/capture.go",
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
