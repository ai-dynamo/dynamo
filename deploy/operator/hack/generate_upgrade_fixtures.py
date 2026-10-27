#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate rolling history or capture a pinned release operator."""

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


# One-time exception: main moved to 1.7 before history/1.6.0 was captured (#16650).
# Delete this entry once history/1.6.0 lands; every other minor keeps the one-minor rule.
HISTORY_EXCEPTIONS = {(1, 7): (1, 5)}


def validate_versions(current, reconciled, previous=None):
    major, number = minor(current)
    old_major, old_number = minor(reconciled)
    exception = HISTORY_EXCEPTIONS.get(minor(current))
    if minor(reconciled) != exception and (
        major != old_major or number - old_number not in (0, 1)
    ):
        raise ValueError(
            f"Operator {current} cannot replay history reconciled by {reconciled}; history must be the same minor or the minor immediately before it. Revert the premature appVersion bump, run clustertest-upgrade-fixtures-regen UPGRADE_RELEASE_TAG=vN.x.y in the selected released operator tree, commit the history, then reapply the bump"
        )
    if (
        previous
        and minor(previous) != minor(current)
        and (
            minor(current) != (minor(previous)[0], minor(previous)[1] + 1)
            or minor(reconciled) not in (minor(previous), exception)
        )
    ):
        raise ValueError(
            f"The bump from {previous} to {current} must advance exactly one minor and needs history captured by the published {previous} operator; revert the bump and capture that tag with clustertest-upgrade-fixtures-regen first. Skipped-minor and major bumps are unsupported"
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
    lock = json.loads((FIXTURES / "history.lock.json").read_text())
    # The original bootstrap predates sourceKind; only its exact pinned inventory is exempt.
    if "sourceKind" not in lock:
        bootstrap = json.loads((FIXTURES / "bootstrap.json").read_text())
        if lock["inputs"] != bootstrap["inputs"] or lock["sourceRevision"] != bootstrap[
            "releases"
        ].get(reconciled):
            raise ValueError(
                "Non-bootstrap history requires explicit released-tag provenance"
            )
    if "sourceKind" in lock and (
        lock["sourceKind"] != "release-tag"
        or lock.get("sourceRef") != f"v{reconciled}"
        or not re.fullmatch(r"[0-9a-f]{40}", lock.get("sourceRevision", ""))
    ):
        raise ValueError(
            "Published history requires released-tag provenance, not a candidate capture"
        )
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
            if obj["kind"] in ("Deployment", "LeaderWorkerSet") and "status" in obj:
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
            elif key == "disaggregatedset.x-k8s.io/revision":
                value[key] = "$var:ds-revision"
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


def regenerate(release_tag=None, candidate_sha=None, candidate_output=None):
    import yaml

    current, reconciled = check()
    if minor(current) == minor(reconciled):
        raise ValueError(
            "History is already reconciled by this minor; do not advance it twice"
        )
    if minor(current) != (minor(reconciled)[0], minor(reconciled)[1] + 1):
        raise ValueError(
            f"Capturing {current} over history reconciled by {reconciled} skips a minor; capture each missing minor from its own released operator tree first"
        )
    require_complete_matrix()
    source = capture_source(current, release_tag, candidate_sha)
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
            lock = {
                "reconciledVersion": current,
                **source,
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
            require_complete_matrix()
        else:
            check(args.base)
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise SystemExit(str(error)) from error
