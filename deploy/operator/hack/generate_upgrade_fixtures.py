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
            "Revert the premature appVersion bump, run regen-upgrade-fixtures with the previous operator, commit the history, then reapply the bump"
        )
    if previous and minor(previous) != minor(current):
        if minor(current) != (minor(previous)[0], minor(previous)[1] + 1) or minor(
            reconciled
        ) != minor(previous):
            raise ValueError(
                "A minor bump requires history captured by the immediately previous operator; skipped-minor and major bumps are unsupported"
            )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    lock = json.loads((FIXTURES / "history.lock.json").read_text())
    if lock["reconciledVersion"] != reconciled:
        raise ValueError("History provenance and RECONCILED_VERSION disagree")
    inputs = {
        str(p.relative_to(FIXTURES)): digest(p)
        for p in FIXTURES.glob("history/*/*/input.yaml")
    }
    if not inputs or inputs != lock["inputs"]:
        raise ValueError(
            "Historical input inventory or bytes changed; regenerate through production controllers, not by editing inputs"
        )
    for path in FIXTURES.glob("history/*/*/input.yaml"):
        if not path.with_name("output.yaml").exists():
            raise ValueError(f"Missing golden: {path.parent}")
    latest = {p.parent.name for p in FIXTURES.glob("latest/*/input.yaml")}
    required = {"sglang-1.4", "sglang-1.5", "sglang-1.5-custom"}
    if latest != required:
        raise ValueError("The initial three Grove scenarios must all be present")
    for origin in (FIXTURES / "history").iterdir():
        if not origin.is_dir():
            continue
        if minor(origin.name) > minor(reconciled):
            raise ValueError("Origin cannot be newer than the last reconciler")
    print(
        f"operator appVersion={current}; history last reconciled={reconciled}; {len(inputs)} historical inputs"
    )
    return current, reconciled


def golden_objects(objects):
    """Only server-assigned metadata, allocated Service IPs and condition times vary."""
    objects = json.loads(json.dumps(objects))
    for obj in objects:
        metadata = obj["metadata"]
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


def write_golden(path, objects):
    import yaml

    header = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
    path.write_text(
        header + yaml.safe_dump_all(golden_objects(objects), sort_keys=True)
    )


def regenerate():
    import yaml

    current, reconciled = check()
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
        expected = list(FIXTURES.glob("history/*/*/input.yaml")) + list(
            FIXTURES.glob("latest/*/input.yaml")
        )
        for original in expected:
            if not (capture / original.relative_to(FIXTURES)).is_file():
                raise ValueError(f"Capture incomplete: {original}")
        # Publish only after every scenario, golden and independent safety check passed.
        stage = Path(tempfile.mkdtemp(prefix=".upgrade-staged-", dir=FIXTURES.parent))
        backup = FIXTURES.with_name(".upgrade-backup")
        if backup.exists():
            raise ValueError(f"Previous backup exists: {backup}")
        try:
            shutil.copytree(
                FIXTURES,
                stage,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.new"),
            )
            for original in expected:
                relative = original.relative_to(FIXTURES)
                if relative.parts[0] == "latest":
                    relative = (
                        Path("history")
                        / f"{minor(current)[0]}.{minor(current)[1]}.0"
                        / relative.parts[1]
                        / "input.yaml"
                    )
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                captured = capture / original.relative_to(FIXTURES)
                shutil.copyfile(captured, destination)
                write_golden(
                    destination.with_name("output.yaml"),
                    list(yaml.safe_load_all(captured.read_text())),
                )
            namespaces = set()
            for historical in stage.glob("history/*/*/input.yaml"):
                objects = list(yaml.safe_load_all(historical.read_text()))
                names = {obj["metadata"]["namespace"] for obj in objects}
                if len(names) != 1 or names & namespaces:
                    raise ValueError(
                        "Historical namespace collision: give latest scenarios fresh namespaces before promoting a new origin"
                    )
                namespaces.update(names)
                if any(not obj["metadata"].get("uid") for obj in objects):
                    raise ValueError(f"Non-concrete capture: {historical}")
            lock = {
                "reconciledVersion": current,
                "sourceRevision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "inputs": {
                    str(p.relative_to(stage)): digest(p)
                    for p in stage.glob("history/*/*/input.yaml")
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
    parser.add_argument(
        "--base", help="Base git revision for the minor-bump smoke check"
    )
    args = parser.parse_args()
    try:
        regenerate() if args.regenerate else check(args.base)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
