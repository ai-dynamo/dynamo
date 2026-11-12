# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind image inventories to producer runs and audit immutable platform rootfses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .prepare_diff import select_osrb_csv, validate_csv

SCHEMA_VERSION = 1
SYFT_VERSION = "1.54.1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_repository(ref: str) -> str:
    """Drop a tag or digest without treating a registry port as a tag."""
    if "@" in ref:
        return ref.split("@", 1)[0]
    head, separator, tail = ref.rpartition("/")
    return (head + separator if separator else "") + tail.split(":", 1)[0]


def inspect_image(ref: str) -> dict:
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            ref,
            "--format",
            "{{json .Manifest}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def platform_digests(manifest: dict, expected: set[str]) -> dict[str, str]:
    """Reject extra, duplicate, or missing runnable platforms; ignore attestations."""
    if not _DIGEST.fullmatch(manifest.get("digest", "")):
        raise ValueError("Image index has no valid immutable digest")
    found = {}
    for descriptor in manifest.get("manifests", []):
        platform = descriptor.get("platform", {})
        if (
            descriptor.get("annotations", {}).get("vnd.docker.reference.type")
            == "attestation-manifest"
        ):
            continue
        name = f"{platform.get('os')}/{platform.get('architecture')}"
        digest = descriptor.get("digest", "")
        if name not in expected or name in found or not _DIGEST.fullmatch(digest):
            raise ValueError(
                f"Unexpected, duplicate, or invalid image platform: {name}"
            )
        found[name] = digest
    if set(found) != expected:
        raise ValueError(
            f"Image platforms {sorted(found)} do not match {sorted(expected)}"
        )
    return found


def safe_file(root: Path, relative: str) -> Path:
    path = root / relative
    if (
        not path.resolve().is_relative_to(root.resolve())
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(f"Invalid evidence file: {relative}")
    return path


def create_provenance(
    image_ref: str,
    artifact_root: Path,
    prefix: str,
    source_sha: str,
    repository: str,
    run_id: str,
    run_attempt: str,
    platforms: set[str],
    cuda_version: str = "",
    baseline_stem: str = "",
) -> dict:
    if not image_ref or not run_id.isdigit() or not run_attempt.isdigit():
        raise ValueError("A pushed image and exact producer run/attempt are required")
    manifest = inspect_image(image_ref)
    digests = platform_digests(manifest, platforms)
    records = []
    for platform, digest in sorted(digests.items()):
        arch = platform.split("/", 1)[1]
        csv_path = select_osrb_csv(
            artifact_root, prefix, arch, source_sha, cuda_version
        )
        validate_csv(csv_path)
        baseline = None
        if baseline_stem:
            if not re.fullmatch(r"[a-zA-Z0-9_.@-]+", baseline_stem):
                raise ValueError("Invalid baseline SBOM stem")
            src = (
                Path(__file__).parent
                / "base_sboms"
                / f"{baseline_stem}-{arch}.cdx.json"
            )
            dest = artifact_root / "provenance-base" / f"{arch}.cdx.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            baseline = {
                "path": str(dest.relative_to(artifact_root)),
                "sha256": sha256(dest),
            }
        records.append(
            {
                "platform": platform,
                "digest": digest,
                "csv_path": str(csv_path.relative_to(artifact_root)),
                "csv_sha256": sha256(csv_path),
                "baseline": baseline,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "source_sha": source_sha,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "artifact_prefix": prefix,
        "cuda_version": cuda_version,
        "image_ref": image_ref,
        "index_digest": manifest["digest"],
        "platforms": records,
    }
    (artifact_root / "build-provenance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def validate_provenance(
    root: Path,
    repository: str,
    run_id: str,
    source_sha: str,
    prefix: str,
    image_ref: str,
    platforms: set[str],
    expected_attempt: str | None = None,
) -> dict:
    provenance = json.loads(safe_file(root, "build-provenance.json").read_text())
    expected = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "run_id": run_id,
        "source_sha": source_sha,
        "artifact_prefix": prefix,
        "image_ref": image_ref,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Image provenance belongs to a different producer, SHA, or image"
        )
    if (
        expected_attempt is not None
        and provenance.get("run_attempt") != expected_attempt
    ):
        raise ValueError("Image provenance belongs to a different run attempt")
    current = inspect_image(image_ref)
    digests = platform_digests(current, platforms)
    if current["digest"] != provenance.get("index_digest"):
        raise ValueError("Image tag no longer identifies the producer's index digest")
    records = provenance.get("platforms", [])
    if len(records) != len(platforms) or {r["platform"] for r in records} != platforms:
        raise ValueError("Producer inventory does not cover every published platform")
    for record in records:
        arch = record["platform"].split("/")[1]
        selected = select_osrb_csv(
            root, prefix, arch, source_sha, provenance["cuda_version"]
        )
        if (
            selected != safe_file(root, record["csv_path"])
            or sha256(selected) != record["csv_sha256"]
        ):
            raise ValueError("Producer inventory identity or hash mismatch")
        validate_csv(selected)
        if record["digest"] != digests[record["platform"]]:
            raise ValueError("Producer platform digest does not match published index")
        if record["baseline"]:
            baseline = safe_file(root, record["baseline"]["path"])
            if sha256(baseline) != record["baseline"]["sha256"]:
                raise ValueError("Producer baseline SBOM hash mismatch")
    return provenance


def scan_rootfs(rootfs: Path, artifact_root: Path, platform: str, output: Path) -> None:
    """Called only in the native helper stage; no shipped executable is run."""
    from .audit_image_sbom import main as audit

    provenance = json.loads((artifact_root / "build-provenance.json").read_text())
    matches = [r for r in provenance["platforms"] if r["platform"] == platform]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous audit platform")
    record = matches[0]
    inventory = safe_file(rootfs, "legal/osrb-deps.csv")
    if sha256(inventory) != record["csv_sha256"]:
        raise ValueError("Final image inventory differs from its producer artifact")
    validate_csv(inventory)
    policy = Path(__file__).parent / "policy/licenses.toml"
    output.mkdir(parents=True, exist_ok=True)
    cdx, native = output / "syft.cdx.json", output / "syft.json"
    subprocess.run(
        [
            "syft",
            f"dir:{rootfs}",
            "--select-catalogers",
            "installed,+rust-cargo-lock-cataloger",
            "-o",
            f"cyclonedx-json={cdx}",
            "-o",
            f"syft-json={native}",
        ],
        check=True,
    )
    args = [
        "--syft-sbom",
        str(cdx),
        "--syft-json",
        str(native),
        "--our-csv",
        str(inventory),
        "--policy",
        str(policy),
        "--rootfs-prefix",
        str(rootfs),
        "--image",
        provenance["image_ref"],
        "--report",
        str(output / "audit-report.txt"),
        "--fail-on-findings",
    ]
    if record["baseline"]:
        baseline = safe_file(artifact_root, record["baseline"]["path"])
        if sha256(baseline) != record["baseline"]["sha256"]:
            raise ValueError("Baseline changed before final-image audit")
        args += ["--base-sbom", str(baseline)]
    if audit(args):
        raise ValueError(f"Independent final-image attribution failed for {platform}")
    (output / "accepted.json").write_text(
        json.dumps(
            {
                "platform": platform,
                "digest": record["digest"],
                "index_digest": provenance["index_digest"],
                "csv_sha256": record["csv_sha256"],
                "syft_version": SYFT_VERSION,
                "policy_sha256": sha256(policy),
            },
            indent=2,
        )
        + "\n"
    )


def audit_platforms(
    artifact_root: Path, provenance: dict, output: Path, builder: str
) -> None:
    """Scan platform digests on a native BuildKit helper, including distroless images."""
    dockerfile = Path(__file__).parents[1] / "Dockerfile.compliance-audit"
    for record in provenance["platforms"]:
        platform = record["platform"]
        destination = output / platform.replace("/", "_")
        destination.mkdir(parents=True, exist_ok=True)
        accepted_path = destination / "accepted.json"
        accepted_path.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="image-compliance-") as temp:
            context = Path(temp)
            shutil.copytree(
                Path(__file__).parent,
                context / "compliance",
                ignore=shutil.ignore_patterns("__pycache__", "tests"),
            )
            shutil.copytree(artifact_root, context / "input")
            image = image_repository(provenance["image_ref"]) + "@" + record["digest"]
            command = [
                "docker",
                "buildx",
                "build",
                "--builder",
                builder,
                "--platform",
                "linux/amd64",
                "--target",
                "result",
                "--build-arg",
                f"AUDIT_IMAGE={image}",
                "--build-arg",
                f"AUDIT_PLATFORM={platform}",
                "--output",
                f"type=local,dest={destination}",
                "--progress=plain",
                "-f",
                str(dockerfile),
                str(context),
            ]
            with (destination / "build.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print((destination / "build.log").read_text()[-20000:])
                raise subprocess.CalledProcessError(result.returncode, command)
            if not accepted_path.is_file():
                raise ValueError("Scanner completed without a successful audit result")
            accepted = json.loads(accepted_path.read_text())
            expected = {
                "platform": platform,
                "digest": record["digest"],
                "index_digest": provenance["index_digest"],
                "csv_sha256": record["csv_sha256"],
                "syft_version": SYFT_VERSION,
                "policy_sha256": sha256(context / "compliance/policy/licenses.toml"),
            }
            if accepted != expected:
                raise ValueError("Scanner result does not match this image and policy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    for name in (
        "image-ref",
        "prefix",
        "sha",
        "repository",
        "run-id",
        "run-attempt",
        "platforms",
    ):
        create.add_argument("--" + name, required=True)
    create.add_argument("--artifact-root", type=Path, required=True)
    create.add_argument("--cuda-version", default="")
    create.add_argument("--baseline-stem", default="")
    scan = sub.add_parser("scan-rootfs")
    scan.add_argument("--rootfs", type=Path, required=True)
    scan.add_argument("--artifact-root", type=Path, required=True)
    scan.add_argument("--platform", required=True)
    scan.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        create_provenance(
            args.image_ref,
            args.artifact_root,
            args.prefix,
            args.sha,
            args.repository,
            args.run_id,
            args.run_attempt,
            set(args.platforms.split(",")),
            args.cuda_version,
            args.baseline_stem,
        )
    else:
        scan_rootfs(args.rootfs, args.artifact_root, args.platform, args.output)


if __name__ == "__main__":
    main()
