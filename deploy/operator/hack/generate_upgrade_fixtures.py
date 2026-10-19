#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate rolling history or capture the current operator before its minor bump."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

OPERATOR = Path(__file__).resolve().parents[1]
ROOT = OPERATOR.parents[1]
FIXTURES = OPERATOR / "internal/controller/testdata/upgrade"
CHART = "deploy/helm/charts/platform/components/operator/Chart.yaml"
MARKER = "deploy/operator/internal/controller/testdata/upgrade/RECONCILED_VERSION"


def app_version(chart):
    match = re.search(r"^appVersion:\s*[\"']?([^\s\"']+)", chart, re.MULTILINE)
    if not match:
        raise ValueError("operator chart has no appVersion")
    return match.group(1)


def minor(version):
    match = re.fullmatch(r"(\d+)\.(\d+)\.\d+(?:[-+].*)?", version)
    if not match:
        raise ValueError(f"Invalid operator version: {version}")
    return tuple(map(int, match.groups()))


def validate_versions(current, reconciled, previous=None):
    major, number = minor(current)
    old_major, old_number = minor(reconciled)
    if major != old_major or number - old_number not in (0, 1):
        raise ValueError(
            "Revert the premature appVersion bump, run clustertest-upgrade-fixtures-regen with the previous operator, commit the history, then reapply the bump"
        )
    if (
        previous
        and minor(previous) != minor(current)
        and (
            minor(current) != (minor(previous)[0], minor(previous)[1] + 1)
            or minor(reconciled) != minor(previous)
        )
    ):
        raise ValueError(
            "A minor bump requires history captured by the immediately previous operator; revert the bump and run clustertest-upgrade-fixtures-regen first. Skipped-minor and major bumps are unsupported"
        )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_inputs(role):
    return sorted((FIXTURES / role).rglob("input.yaml"))


def require_complete_matrix():
    required = set(
        json.loads((FIXTURES / "required-matrix.json").read_text())["requiredScenarios"]
    )
    available = {
        p.parent.relative_to(FIXTURES / "latest").as_posix()
        for p in fixture_inputs("latest")
    }
    missing = required - available
    if missing:
        raise ValueError(
            "Incomplete release matrix; missing: " + ", ".join(sorted(missing))
        )


def check(base=None):
    current = app_version((ROOT / CHART).read_text())
    reconciled = (FIXTURES / "RECONCILED_VERSION").read_text().strip()
    previous = None
    if base:
        previous = app_version(
            subprocess.check_output(
                ["git", "show", f"{base}:{CHART}"], cwd=ROOT, text=True
            )
        )
    validate_versions(current, reconciled, previous)
    if previous and minor(previous) != minor(current):
        require_complete_matrix()
    lock = json.loads((FIXTURES / "history.lock.json").read_text())
    if lock["reconciledVersion"] != reconciled:
        raise ValueError("History provenance and RECONCILED_VERSION disagree")
    inputs = {
        str(p.relative_to(FIXTURES)): digest(p) for p in fixture_inputs("history")
    }
    if not inputs or inputs != lock["inputs"]:
        raise ValueError(
            "Historical input inventory or bytes changed; regenerate through production controllers, not by editing inputs"
        )
    for path in fixture_inputs("history"):
        if not path.with_name("output.yaml").exists():
            raise ValueError(f"Missing golden: {path.parent}")
    latest = fixture_inputs("latest")
    if not latest:
        raise ValueError("No latest scenarios")
    for path in latest:
        if not path.with_name("output.yaml").is_file():
            raise ValueError(f"Missing golden: {path.parent}")
    for origin in (FIXTURES / "history").iterdir():
        if not origin.is_dir():
            continue
        if minor(origin.name) > minor(reconciled):
            raise ValueError("Origin cannot be newer than the last reconciler")
    print(
        f"operator appVersion={current}; history last reconciled={reconciled}; {len(inputs)} historical inputs"
    )
    return current, reconciled


def golden_objects(objects, latest=False):
    """Only server-assigned metadata, allocated Service IPs and condition times vary."""
    objects = json.loads(json.dumps(objects))
    for obj in objects:
        metadata = obj["metadata"]
        if latest:
            bind_latest_variables(obj)
            if obj["kind"] == "PodCliqueSet":
                metadata["annotations"][
                    "nvidia.com/last-applied-hash"
                ] = "$var:pcs-spec-hash"
        metadata.pop("managedFields", None)
        for field in ("uid", "resourceVersion", "generation", "creationTimestamp"):
            if field in metadata:
                metadata[field] = "$ignore"
        for owner in metadata.get("ownerReferences", []):
            owner["uid"] = "$ignore"
        for condition in obj.get("status", {}).get("conditions", []):
            if "lastTransitionTime" in condition:
                condition["lastTransitionTime"] = "$ignore"
        if obj["kind"] == "Service" and obj.get("spec", {}).get("clusterIP") != "None":
            obj["spec"]["clusterIP"] = "$ignore"
            obj["spec"]["clusterIPs"] = ["$ignore" for _ in obj["spec"]["clusterIPs"]]
    return objects


def bind_latest_variables(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "nvidia.com/dynamo-operator-origin-version":
                value[key] = "$var:operator-version"
            elif key in (
                "nvidia.com/current-worker-hash-v2",
                "nvidia.com/dynamo-worker-hash",
            ):
                value[key] = "$var:worker-hash"
            elif key == "value" and value.get("name") == "DYN_NAMESPACE_WORKER_SUFFIX":
                value[key] = "$var:worker-hash"
            else:
                bind_latest_variables(child)
    elif isinstance(value, list):
        for child in value:
            bind_latest_variables(child)


def write_golden(path, objects, latest=False):
    import yaml

    header = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
    path.write_text(
        header + yaml.safe_dump_all(golden_objects(objects, latest), sort_keys=True)
    )


def regenerate():
    import yaml

    current, reconciled = check()
    require_complete_matrix()
    if minor(current) == minor(reconciled):
        raise ValueError(
            "History is already reconciled by this minor; do not advance it twice"
        )
    with tempfile.TemporaryDirectory(prefix="upgrade-capture-") as temporary:
        capture = Path(temporary)
        env = dict(os.environ, DYNAMO_UPGRADE_CAPTURE_DIR=str(capture))
        subprocess.run(
            [
                "go",
                "test",
                "-tags=clustertest upgradecontract",
                "./internal/controller",
                "-run",
                "^TestClusterOperatorUpgrade$",
                "-count=1",
                "-v",
                "-timeout=15m",
            ],
            cwd=OPERATOR,
            env=env,
            check=True,
        )
        expected = list(fixture_inputs("history")) + list(fixture_inputs("latest"))
        captures = {}
        for original in expected:
            relative = original.relative_to(FIXTURES)
            if relative.parts[0] == "latest":
                relative = (
                    Path("history")
                    / f"{minor(current)[0]}.{minor(current)[1]}.0"
                    / Path(*relative.parts[1:])
                )
            captured = capture / relative
            if not captured.is_file():
                raise ValueError(f"Capture incomplete: {relative}")
            captures[relative] = captured
        # Publish only after every scenario, golden and independent safety check passed.
        backup = FIXTURES.with_name(".upgrade-backup")
        if backup.exists():
            raise ValueError(f"Previous backup exists: {backup}")
        stage = Path(tempfile.mkdtemp(prefix=".upgrade-staged-", dir=FIXTURES.parent))
        try:
            shutil.copytree(
                FIXTURES,
                stage,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.new"),
            )
            for relative, captured in captures.items():
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(captured, destination)
                write_golden(
                    destination.with_name("output.yaml"),
                    list(yaml.safe_load_all(captured.read_text())),
                )
            for original in fixture_inputs("latest"):
                write_golden(
                    stage / original.relative_to(FIXTURES).with_name("output.yaml"),
                    list(
                        yaml.safe_load_all(
                            (capture / original.relative_to(FIXTURES)).read_text()
                        )
                    ),
                    latest=True,
                )
            namespaces = set()
            for historical in list((stage / "history").rglob("input.yaml")) + list(
                (stage / "latest").rglob("input.yaml")
            ):
                objects = list(yaml.safe_load_all(historical.read_text()))
                names = {obj["metadata"]["namespace"] for obj in objects}
                if len(names) != 1 or names & namespaces:
                    raise ValueError(
                        "Fixture namespace collision: history and latest must use distinct namespaces"
                    )
                namespaces.update(names)
                if "history" in historical.relative_to(stage).parts[:1] and any(
                    not obj["metadata"].get("uid") for obj in objects
                ):
                    raise ValueError(f"Non-concrete capture: {historical}")
            lock = {
                "reconciledVersion": current,
                "sourceRevision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "inputs": {
                    str(p.relative_to(stage)): digest(p)
                    for p in (stage / "history").rglob("input.yaml")
                },
            }
            (stage / "history.lock.json").write_text(
                json.dumps(lock, indent=2, sort_keys=True) + "\n"
            )
            (stage / "RECONCILED_VERSION").write_text(current + "\n")
            FIXTURES.rename(backup)
            try:
                stage.rename(FIXTURES)
            except BaseException:
                backup.rename(FIXTURES)
                raise
            shutil.rmtree(backup)
        finally:
            shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--regenerate", action="store_true")
    group.add_argument("--check-release-matrix", action="store_true")
    parser.add_argument(
        "--base", help="Base git revision for the minor-bump smoke check"
    )
    args = parser.parse_args()
    try:
        if args.regenerate:
            regenerate()
        elif args.check_release_matrix:
            check(args.base)
            require_complete_matrix()
        else:
            check(args.base)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
