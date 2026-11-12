# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit exact-run image evidence before any release copy, then pin copy sources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode
from zipfile import ZipFile

from .image_provenance import audit_platforms, image_repository, validate_provenance

IMAGE_PREFIXES = {
    "vllm-runtime": "vllm-runtime",
    "sglang-runtime": "sglang-runtime",
    "trtllm-runtime": "trtllm-runtime",
    "vllm-efa": "vllm-runtime-efa",
    "sglang-efa": "sglang-runtime-efa",
    "trtllm-efa": "trtllm-runtime-efa",
    "frontend": "dynamo-frontend",
    "operator": "operator",
    "planner": "dynamo-planner",
    "sidecar": "dynamo-sidecar",
}
_REQUIRED_NIGHTLY = {"vllm-runtime", "sglang-runtime", "trtllm-runtime"}


def api(path: str) -> dict:
    result = subprocess.run(["gh", "api", path], check=True, capture_output=True)
    return json.loads(result.stdout)


def run_metadata(repo: str, run_id: str, sha: str, nightly: bool | None) -> dict:
    if not run_id.isdigit():
        raise ValueError("An exact producer workflow run ID is required")
    run = api(f"/repos/{repo}/actions/runs/{run_id}")
    if (
        run.get("head_sha") != sha
        or run.get("head_repository", {}).get("full_name") != repo
    ):
        raise ValueError(
            "Producer workflow belongs to a different repository or commit"
        )
    path = run.get("path", "").split("@", 1)[0]
    if nightly is not None:
        expected = (
            ".github/workflows/nightly-ci.yml"
            if nightly
            else ".github/workflows/post-merge-ci.yml"
        )
        if path != expected:
            raise ValueError(
                "Release producer is not the expected nightly/post-merge workflow"
            )
        branch = run.get("head_branch", "")
        if nightly and branch != "main":
            raise ValueError("Nightly producer must build main")
        if not nightly and not branch.startswith("release/"):
            raise ValueError("Release producer must build a release branch")
    elif path not in {
        ".github/workflows/pr.yaml",
        ".github/workflows/nightly-ci.yml",
        ".github/workflows/post-merge-ci.yml",
    }:
        raise ValueError("Unrecognized image-producing workflow")
    return run


def download_inventory(
    repo: str, run_id: str, sha: str, prefix: str, destination: Path
) -> None:
    name = f"compliance-{sha}-{prefix}"
    endpoint = f"/repos/{repo}/actions/runs/{run_id}/artifacts?" + urlencode(
        {"name": name, "per_page": 100}
    )
    artifacts = api(endpoint).get("artifacts", [])
    candidates = [a for a in artifacts if a["name"] == name and not a["expired"]]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one retained {name} from producer run {run_id}; rebuild its image"
        )
    artifact = candidates[0]
    if artifact.get("workflow_run", {}).get("id") != int(run_id):
        raise ValueError("Artifact generating run does not match requested producer")
    data = subprocess.run(
        ["gh", "api", f"/repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
        check=True,
        capture_output=True,
    ).stdout
    digest = artifact.get("digest")
    if digest and digest != "sha256:" + hashlib.sha256(data).hexdigest():
        raise ValueError("Downloaded artifact does not match GitHub's digest")
    with ZipFile(io.BytesIO(data)) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError("Compliance artifact exceeds the inventory size limit")
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError("Unsafe path in compliance artifact")
        archive.extractall(destination)


def selected_images(
    subset: str,
    sha: str,
    registry_repo: str,
    nightly: bool,
    build_results: dict[str, str],
) -> dict[str, str]:
    selected = (
        set(IMAGE_PREFIXES)
        if subset == "all"
        else set()
        if subset == "none"
        else set(subset.split(","))
    )
    if selected - IMAGE_PREFIXES.keys():
        raise ValueError(
            f"Unknown selected images: {sorted(selected - IMAGE_PREFIXES.keys())}"
        )
    result = {}
    for token in sorted(selected):
        # Optional nightly builds can be omitted only when the caller explicitly
        # says they did not succeed. A manual restage has no such exemption.
        status = build_results.get(token, "")
        if (
            nightly
            and token not in _REQUIRED_NIGHTLY
            and status
            and status != "success"
        ):
            continue
        suffix = "-nightly" if nightly else ""
        result[token] = f"{registry_repo}:{sha}-{IMAGE_PREFIXES[token]}{suffix}"
    return result


def audit_one(
    repo: str,
    run_id: str,
    sha: str,
    prefix: str,
    image_ref: str,
    builder: str,
    output: Path,
    nightly: bool | None = None,
) -> str:
    run = run_metadata(repo, run_id, sha, nightly)
    with tempfile.TemporaryDirectory(prefix="compliance-artifact-") as temp:
        root = Path(temp)
        download_inventory(repo, run_id, sha, prefix, root)
        provenance = validate_provenance(
            root, repo, run_id, sha, prefix, image_ref, {"linux/amd64", "linux/arm64"}
        )
        attempt = provenance.get("run_attempt", "")
        if not attempt.isdigit() or not 0 < int(attempt) <= run["run_attempt"]:
            raise ValueError("Invalid producing run attempt")
        audit_platforms(root, provenance, output, builder)
        return image_repository(image_ref) + "@" + provenance["index_digest"]


def resolve_producer(repo: str, sha: str, nightly: bool, images: dict[str, str]) -> str:
    """Choose a trusted producer only when every selected digest matches its evidence."""
    workflow = "nightly-ci.yml" if nightly else "post-merge-ci.yml"
    query = urlencode({"head_sha": sha, "per_page": 50})
    runs = api(f"/repos/{repo}/actions/workflows/{workflow}/runs?{query}").get(
        "workflow_runs", []
    )
    for run in runs:
        run_id = str(run["id"])
        try:
            run_metadata(repo, run_id, sha, nightly)
            for token, image_ref in images.items():
                with tempfile.TemporaryDirectory(prefix="producer-evidence-") as temp:
                    root = Path(temp)
                    download_inventory(repo, run_id, sha, IMAGE_PREFIXES[token], root)
                    validate_provenance(
                        root,
                        repo,
                        run_id,
                        sha,
                        IMAGE_PREFIXES[token],
                        image_ref,
                        {"linux/amd64", "linux/arm64"},
                    )
        except (ValueError, FileNotFoundError):
            continue
        return run_id
    raise ValueError(
        "No trusted producer run has matching evidence for every selected image; rebuild or supply its exact run ID"
    )


def preflight(
    repo: str,
    run_id: str,
    sha: str,
    images: dict[str, str],
    nightly: bool,
    builder: str,
    output: Path,
) -> dict:
    # Do not leave an accepted manifest from an earlier partial/failed attempt.
    manifest_path = output / "publication-manifest.json"
    manifest_path.unlink(missing_ok=True)
    if images and not run_id:
        run_id = resolve_producer(repo, sha, nightly, images)
    pinned = {}
    for token, image_ref in images.items():
        pinned[image_ref] = audit_one(
            repo,
            run_id,
            sha,
            IMAGE_PREFIXES[token],
            image_ref,
            builder,
            output / token,
            nightly,
        )
    result = {"source_sha": sha, "producer_run_id": run_id, "images": pinned}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def pinned_source(manifest: Path, source: str) -> str:
    images = json.loads(manifest.read_text())["images"]
    if source not in images:
        raise ValueError(
            "Copy source was not accepted by the complete publication preflight"
        )
    return images[source]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--repo", required=True)
        cmd.add_argument("--run-id", default="")
        cmd.add_argument("--sha", required=True)
        cmd.add_argument("--builder", required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "audit":
            cmd.add_argument("--prefix", required=True)
            cmd.add_argument("--image-ref", required=True)
        else:
            cmd.add_argument("--registry-repo", required=True)
            cmd.add_argument("--nightly", action="store_true")
            cmd.add_argument("--containers", default="all")
    source = sub.add_parser("source")
    source.add_argument("--manifest", type=Path, required=True)
    source.add_argument("--image-ref", required=True)
    args = parser.parse_args()
    if args.command == "source":
        print(pinned_source(args.manifest, args.image_ref))
    elif args.command == "audit":
        audit_one(
            args.repo,
            args.run_id,
            args.sha,
            args.prefix,
            args.image_ref,
            args.builder,
            args.output,
        )
    else:
        results = {
            token: os.environ.get(token.upper().replace("-", "_") + "_BUILD_RESULT", "")
            for token in IMAGE_PREFIXES
        }
        images = selected_images(
            args.containers, args.sha, args.registry_repo, args.nightly, results
        )
        preflight(
            args.repo,
            args.run_id,
            args.sha,
            images,
            args.nightly,
            args.builder,
            args.output,
        )


if __name__ == "__main__":
    main()
