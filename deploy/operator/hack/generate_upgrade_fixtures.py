#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate or check immutable upgrade snapshots using pinned released source."""

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

OPERATOR = Path(__file__).resolve().parents[1]
ROOT = OPERATOR.parents[1]
FIXTURES = OPERATOR / "internal/controller/testdata/upgrade"
TEMPLATE = OPERATOR / "hack/upgrade_capture_test.go.tmpl"


def sha256(data):
    """Hash exact file bytes for the snapshot provenance lock."""
    return hashlib.sha256(data).hexdigest()


def generate(check):
    """Capture baselines with pinned release code, never candidate rendering.

    With check=True, compare regenerated snapshots and provenance without
    writing fixtures. Otherwise, write both snapshots and their lock; even
    comment-only generator edits require regeneration because it is hashed.
    """
    catalog = json.loads((FIXTURES / "catalog.json").read_text())
    chart = (ROOT / "deploy/helm/charts/platform/Chart.yaml").read_text()
    version = next(
        line.split(":", 1)[1].strip()
        for line in chart.splitlines()
        if line.startswith("version:")
    )
    if not version.startswith(catalog["candidateMinor"] + "."):
        raise SystemExit("Update the historical catalog for the candidate release line")
    lock = {
        "generated": "DO NOT EDIT: run hack/generate_upgrade_fixtures.py",
        "inputs": {},
        "snapshots": {},
    }
    for path in [
        FIXTURES / "catalog.json",
        FIXTURES / "input.yaml",
        TEMPLATE,
        Path(__file__).resolve(),
    ]:
        lock["inputs"][str(path.relative_to(OPERATOR))] = sha256(path.read_bytes())

    with tempfile.TemporaryDirectory(prefix="dynamo-upgrade-fixtures-") as directory:
        workspace = Path(directory)
        for version, release in catalog["releases"].items():
            revision = release["revision"]
            subprocess.run(
                ["git", "-C", str(ROOT), "cat-file", "-e", revision + "^{commit}"],
                check=True,
            )
            checkout = workspace / version
            checkout.mkdir()
            paths = ["deploy/operator"]
            if version == "1.4.0":
                paths.append("deploy/snapshot")
            archive = subprocess.check_output(
                ["git", "-C", str(ROOT), "archive", revision, *paths]
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(checkout, filter="data")
            operator = checkout / "deploy/operator"
            capture = TEMPLATE.read_text()
            if release["newWorkerSuffix"]:
                capture = capture.replace(
                    "// WORKER_SUFFIX",
                    "require.NoError(t,applyGroveWorkerHashSuffix(renderDGD,&dgd))",
                )
            else:
                capture = capture.replace("// WORKER_SUFFIX", "")
            capture = capture.replace(
                "// FEATURES_IMPORT",
                '"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"',
            )
            capture = capture.replace(
                "RUNTIME_CONFIG",
                "&common.RuntimeConfig{Gate:features.Gates{Grove:true}}",
            )
            (operator / "internal/controller/zz_capture_upgrade_test.go").write_text(
                capture
            )

            for profile in catalog["profiles"]:
                if profile["release"] != version:
                    continue
                output = workspace / (profile["name"] + ".json")
                env = dict(
                    os.environ,
                    UPGRADE_INPUT=str(FIXTURES / "input.yaml"),
                    UPGRADE_IMAGE=profile["image"],
                    UPGRADE_REVISION=revision,
                    UPGRADE_OUTPUT=str(output),
                    GOTOOLCHAIN="go1.26.8",
                )
                print(f"Capture {profile['name']} from {revision}", flush=True)
                subprocess.run(
                    [
                        "go",
                        "test",
                        "./internal/controller",
                        "-run",
                        "^TestCaptureUpgradeBaseline$",
                        "-count=1",
                    ],
                    cwd=operator,
                    env=env,
                    check=True,
                )
                data = output.read_bytes()
                lock["snapshots"][profile["name"] + ".json"] = sha256(data)
                destination = FIXTURES / (profile["name"] + ".json")
                if check:
                    if not destination.exists() or destination.read_bytes() != data:
                        raise SystemExit(
                            f"Historical snapshot differs: {destination}; inspect the released-source capture, do not bless candidate rendering"
                        )
                else:
                    destination.write_bytes(data)

    data = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    path = FIXTURES / "snapshots.lock.json"
    if check:
        if not path.exists() or path.read_bytes() != data:
            raise SystemExit(f"Snapshot provenance differs: {path}")
    else:
        path.write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate from released source and compare without changing fixtures",
    )
    generate(parser.parse_args().check)
