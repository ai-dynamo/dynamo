#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate immutable history or append a pinned release baseline."""

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


# The reviewed harness may be carried to a release checkout; production inputs may not.
CAPTURE_PATHS = [
    "deploy/operator",
    "deploy/helm/charts/platform",
    ":(exclude)deploy/operator/hack",
    ":(exclude)deploy/operator/Makefile",
    ":(exclude)deploy/operator/internal/testing",
    ":(exclude)deploy/operator/internal/controller/testdata",
    ":(exclude,glob)**/*_test.go",
    ":(exclude,glob)**/README*",
    ":(exclude,glob)**/AGENTS.md",
    ":(exclude,glob)**/CLAUDE.md",
]


def capture_source(current, release_tag=None, candidate_sha=None):
    if bool(release_tag) == bool(candidate_sha):
        raise ValueError(
            "Select exactly one released --release-tag or provisional --candidate-sha"
        )
    if release_tag:
        if not re.fullmatch(r"v\d+\.\d+\.\d+", release_tag):
            raise ValueError("Use a published stable release tag such as v1.6.0")
        ref = f"refs/tags/{release_tag}"
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", candidate_sha):
            raise ValueError("--candidate-sha requires a full immutable commit SHA")
        ref = candidate_sha
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    version = app_version(
        subprocess.check_output(
            ["git", "show", f"{revision}:{CHART}"], cwd=ROOT, text=True
        )
    )
    if version != current or (release_tag and release_tag != f"v{version}"):
        raise ValueError("Capture source must match the checkout's operator appVersion")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", revision, "--", *CAPTURE_PATHS],
        cwd=ROOT,
        text=True,
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *CAPTURE_PATHS],
        cwd=ROOT,
        text=True,
    )
    if changed.strip() or untracked.strip():
        raise ValueError(
            "Capture production tree differs from the selected release source: "
            + changed
            + untracked
        )
    if release_tag:
        release = json.loads(
            subprocess.check_output(
                ["gh", "api", f"repos/ai-dynamo/dynamo/releases/tags/{release_tag}"],
                cwd=ROOT,
                text=True,
                timeout=30,
            )
        )
        official = json.loads(
            subprocess.check_output(
                ["gh", "api", f"repos/ai-dynamo/dynamo/commits/{release_tag}"],
                cwd=ROOT,
                text=True,
                timeout=30,
            )
        )
        if (
            release["draft"]
            or release["prerelease"]
            or not release["published_at"]
            or release["tag_name"] != release_tag
            or official["sha"] != revision
        ):
            raise ValueError(
                "The tag must identify a published stable upstream release"
            )
    return {
        "sourceKind": "release-tag" if release_tag else "release-candidate",
        "sourceRef": release_tag or candidate_sha,
        "sourceRevision": revision,
    }


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


def validate_versions(current, baseline_version, previous=None):
    major, number = minor(current)
    old_major, old_number = minor(baseline_version)
    if major != old_major or number - old_number not in (0, 1):
        raise ValueError(
            f"Operator {current} cannot replay newest baseline {baseline_version}; history must be the same minor or the minor immediately before it. Revert the premature appVersion bump, run clustertest-upgrade-fixtures-regen UPGRADE_RELEASE_TAG=vN.x.y in the selected released operator tree, commit the history, then reapply the bump"
        )
    if (
        previous
        and minor(previous) != minor(current)
        and (
            minor(current) != (minor(previous)[0], minor(previous)[1] + 1)
            or minor(baseline_version) != minor(previous)
        )
    ):
        raise ValueError(
            "A minor bump requires history captured by the immediately previous operator; revert the bump and capture the published predecessor tag with clustertest-upgrade-fixtures-regen first. Skipped-minor and major bumps are unsupported"
        )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_inputs(role):
    return sorted((FIXTURES / role).rglob("input.yaml"))


def require_complete_matrix(release_tag=None):
    """Require full coverage or the exact one-time v1.6.0 input inventory."""
    required = set(
        json.loads((FIXTURES / "required-matrix.json").read_text())["requiredScenarios"]
    )
    available = {
        p.parent.relative_to(FIXTURES / "latest").as_posix()
        for p in fixture_inputs("latest")
    }
    missing = required - available
    if not missing:
        return None
    if (
        release_tag == "v1.6.0"
        and app_version((ROOT / CHART).read_text()) == "1.6.0"
        and (FIXTURES / "RECONCILED_VERSION").read_text().strip() == "1.5.0"
    ):
        inventory = json.loads((FIXTURES / "capture-v1.6.0.json").read_text())
        inputs = {
            str(p.relative_to(FIXTURES)): digest(p)
            for role in ("history", "latest")
            for p in fixture_inputs(role)
        }
        if (
            inventory["sourceRef"] != release_tag
            or inventory["inputs"] != inputs
            or inventory["requiredScenarios"] != sorted(required)
            or inventory["missingScenarios"] != sorted(missing)
            or inventory["promotedScenarios"] != sorted(required & available)
        ):
            raise ValueError("The frozen v1.6.0 capture inventory has changed")
        return inventory
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
    lock = json.loads((FIXTURES / "history.lock.json").read_text())
    bootstrap = json.loads((FIXTURES / "bootstrap.json").read_text())
    # Bootstrap provenance describes only the original, immutable three inputs.
    original = lock.get("bootstrap", lock)
    if (
        original["reconciledVersion"] != reconciled
        or original["inputs"] != bootstrap["inputs"]
        or original["sourceRevision"] != bootstrap["releases"].get(reconciled)
    ):
        raise ValueError("Original bootstrap provenance or input hashes changed")
    inputs = {
        str(p.relative_to(FIXTURES)): digest(p) for p in fixture_inputs("history")
    }
    if not inputs or inputs != lock["inputs"]:
        raise ValueError(
            "Historical input inventory or bytes changed; inputs are immutable"
        )
    owned = dict(original["inputs"])
    captures = lock.get("captures", {})
    for version, capture in captures.items():
        if (
            capture["sourceKind"] != "release-tag"
            or capture["sourceRef"] != f"v{version}"
            or not re.fullmatch(r"[0-9a-f]{40}", capture["sourceRevision"])
            or not capture["inputs"]
            or any(
                not p.startswith(f"history/{minor(version)[0]}.{minor(version)[1]}.0/")
                for p in capture["inputs"]
            )
            or owned.keys() & capture["inputs"].keys()
        ):
            raise ValueError(
                "Each baseline requires distinct inputs and released-tag provenance"
            )
        if "captureInventory" in capture:
            inventory = json.loads((FIXTURES / "capture-v1.6.0.json").read_text())
            expected = {
                f"history/1.6.0/{scenario}/input.yaml"
                for scenario in inventory["promotedScenarios"]
            }
            frozen_history = {
                p: checksum
                for p, checksum in inventory["inputs"].items()
                if p.startswith("history/")
            }
            if (
                version != "1.6.0"
                or capture["sourceRevision"] != inventory["sourceRevision"]
                or capture["captureInventory"] != inventory
                or set(capture["inputs"]) != expected
                or any(
                    inputs.get(p) != checksum for p, checksum in frozen_history.items()
                )
            ):
                raise ValueError(
                    "Published v1.6.0 baseline differs from its frozen scope"
                )
        owned.update(capture["inputs"])
    if owned != inputs:
        raise ValueError(
            "Historical inputs differ from their individual baseline provenance"
        )
    latest_baseline = max([reconciled, *captures], key=minor)
    validate_versions(current, latest_baseline, previous)
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
        if minor(origin.name) > minor(latest_baseline):
            raise ValueError("Origin cannot be newer than the newest captured baseline")
    print(
        f"operator appVersion={current}; newest captured baseline={latest_baseline}; {len(inputs)} historical inputs"
    )
    return current, latest_baseline


def golden_objects(objects, latest=False):
    """Keep lowering intent exact while excluding server bookkeeping and runtime status."""
    objects = json.loads(json.dumps(objects))
    component_latest = latest and any(
        obj["kind"] == "DynamoComponentDeployment" for obj in objects
    )
    worker_hash = next(
        (
            obj["metadata"]
            .get("annotations", {})
            .get("nvidia.com/current-worker-hash-v2")
            for obj in objects
            if obj["kind"] == "DynamoGraphDeployment"
        ),
        None,
    )
    for obj in objects:
        metadata = obj["metadata"]
        if latest:
            bind_latest_variables(obj)
            if obj["kind"] == "PodCliqueSet":
                metadata["annotations"][
                    "nvidia.com/last-applied-hash"
                ] = "$var:pcs-spec-hash"
        if component_latest:
            if obj["kind"] in (
                "Deployment",
                "LeaderWorkerSet",
                "DynamoComponentDeployment",
                "Service",
            ):
                # Desired-spec hashes predate API defaulting; the complete defaulted spec is compared below.
                annotations = metadata.get("annotations", {})
                if "nvidia.com/last-applied-hash" in annotations:
                    annotations["nvidia.com/last-applied-hash"] = "$ignore"
            if obj["kind"] in ("Deployment", "LeaderWorkerSet"):
                # Workload-controller availability is outside the operator lowering boundary.
                obj["status"] = {"$ignore": True}
            if obj["kind"] in ("DynamoComponentDeployment", "DynamoGraphDeployment"):
                status = obj.get("status", {})
                components = (
                    [status["component"]]
                    if obj["kind"] == "DynamoComponentDeployment"
                    else status.get("components", {}).values()
                )
                for component in components:
                    # GPU/replica observations can change after the workload graph converges.
                    for field in (
                        "gpusPerEngine",
                        "gpusPerReplica",
                        "availableReplicas",
                        "readyReplicas",
                        "replicas",
                        "updatedReplicas",
                    ):
                        component.pop(field, None)
                    component["$strict"] = False
                for condition in status.get("conditions", []):
                    if condition["type"] == "Available":
                        # A subsequent status conflict can replace the availability diagnostic.
                        condition["message"] = "$ignore"
                        condition["reason"] = "$ignore"
            if worker_hash:
                bind_worker_strings(obj, worker_hash)
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


def bind_worker_strings(value, worker_hash):
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and worker_hash in child:
                value[key] = "$var:worker-string:" + child.replace(
                    worker_hash, "{{worker-hash}}"
                )
            else:
                bind_worker_strings(child, worker_hash)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str) and worker_hash in child:
                value[index] = "$var:worker-string:" + child.replace(
                    worker_hash, "{{worker-hash}}"
                )
            else:
                bind_worker_strings(child, worker_hash)


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

    # Go's YAML parser also recognizes exponent-only numbers such as worker hashes.
    # Quote those strings so generated goldens retain their Kubernetes string type.
    class GoldenDumper(yaml.SafeDumper):
        pass

    GoldenDumper.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$"),
        list("-+0123456789."),
    )
    header = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
    path.write_text(
        header
        + yaml.dump_all(
            golden_objects(objects, latest), Dumper=GoldenDumper, sort_keys=True
        )
    )


def regenerate(release_tag=None, candidate_sha=None, candidate_output=None):
    import yaml

    current, latest_baseline = check()
    if minor(current) == minor(latest_baseline):
        raise ValueError(
            "A baseline is already captured for this minor; existing inputs cannot be replaced"
        )
    inventory = require_complete_matrix(release_tag)
    source = capture_source(current, release_tag, candidate_sha)
    if inventory and source["sourceRevision"] != inventory["sourceRevision"]:
        raise ValueError("The release source differs from the frozen v1.6.0 source")
    if candidate_sha:
        if (
            candidate_output is None
            or Path(candidate_output).exists()
            or Path(candidate_output).resolve().is_relative_to(FIXTURES.resolve())
        ):
            raise ValueError(
                "Candidate capture requires a new --candidate-output directory outside the fixture tree"
            )
    elif candidate_output is not None:
        raise ValueError("--candidate-output is only valid with --candidate-sha")
    with tempfile.TemporaryDirectory(prefix="upgrade-capture-") as temporary:
        capture = Path(temporary)
        env = dict(os.environ, GOWORK="off", DYNAMO_UPGRADE_CAPTURE_DIR=str(capture))
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
        # Only catalog scenarios become new baselines; legacy latest remains CREATE coverage.
        catalog = set(
            json.loads((FIXTURES / "required-matrix.json").read_text())[
                "requiredScenarios"
            ]
        )
        expected = [
            p
            for p in fixture_inputs("latest")
            if p.parent.relative_to(FIXTURES / "latest").as_posix() in catalog
        ]
        captures = {}
        for original in expected:
            relative = (
                Path("history")
                / f"{minor(current)[0]}.{minor(current)[1]}.0"
                / original.relative_to(FIXTURES / "latest")
            )
            captured = capture / relative
            if not captured.is_file():
                raise ValueError(f"Capture incomplete: {relative}")
            captures[relative] = captured
        if candidate_sha:
            (capture / "capture-source.json").write_text(
                json.dumps(
                    {"operatorVersion": current, **source}, indent=2, sort_keys=True
                )
                + "\n"
            )
            shutil.copytree(capture, candidate_output)
            print(
                "Provisional candidate capture saved; history and RECONCILED_VERSION are unchanged"
            )
            return
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
            # Latest describes the development tree; release capture never rewrites it.
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
            previous = json.loads((FIXTURES / "history.lock.json").read_text())
            # Retain each prior baseline's source and exact input bytes independently.
            lock = {
                "bootstrap": previous.get("bootstrap", previous),
                "captures": dict(previous.get("captures", {})),
                "inputs": {
                    str(p.relative_to(stage)): digest(p)
                    for p in (stage / "history").rglob("input.yaml")
                },
            }
            baseline = {
                **source,
                "inputs": {str(p): digest(stage / p) for p in captures},
            }
            if inventory:
                baseline["captureInventory"] = inventory
            lock["captures"][current] = baseline
            (stage / "history.lock.json").write_text(
                json.dumps(lock, indent=2, sort_keys=True) + "\n"
            )
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
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--release-tag", help="Published stable upstream tag for formal history"
    )
    sources.add_argument(
        "--candidate-sha",
        help="Full provisional release-candidate SHA; does not publish history",
    )
    parser.add_argument(
        "--candidate-output", type=Path, help="New directory for provisional captures"
    )
    args = parser.parse_args()
    try:
        if args.regenerate:
            regenerate(args.release_tag, args.candidate_sha, args.candidate_output)
        elif args.check_release_matrix:
            check(args.base)
            require_complete_matrix(args.release_tag)
        else:
            check(args.base)
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise SystemExit(str(error)) from error
