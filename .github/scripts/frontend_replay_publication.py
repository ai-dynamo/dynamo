#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe staging without disclosing its registry, then verify public promotion."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "dynamo-frontend-replay-publication/v1"
REPOSITORY = "dynamo-frontend-nightly"
PUBLIC_REPOSITORY = "nvcr.io/nvidia/ai-dynamo/" + REPOSITORY
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
TAG = re.compile(r"[0-9]{8}-[0-9a-f]{7}\Z")
ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


class ProvenanceError(ValueError):
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


def verify_run(value: Any, repository: str, run_id: str) -> dict[str, Any]:
    """Only completed default-branch nightlies can supply publication evidence."""
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ProvenanceError(
            "publication run ID must be a positive integer", "invalid_run_id"
        )
    if (
        not isinstance(value, dict)
        or type(value.get("id")) is not int
        or value["id"] != int(run_id)
        or not isinstance(value.get("repository"), dict)
        or not isinstance(value.get("head_repository"), dict)
        or value.get("repository", {}).get("full_name") != repository
        or value.get("head_repository", {}).get("full_name") != repository
        or value.get("path") != ".github/workflows/nightly-ci.yml"
        or value.get("name") != "Nightly CI Pipeline"
        or value.get("head_branch") != "main"
        or value.get("event") not in ("schedule", "workflow_dispatch")
        or value.get("status") != "completed"
        or value.get("pull_requests") != []
        or not isinstance(value.get("head_sha"), str)
        or not COMMIT.fullmatch(value["head_sha"])
        or type(value.get("run_attempt")) is not int
        or value["run_attempt"] < 1
    ):
        raise ProvenanceError(
            "publication evidence must come from a completed main nightly",
            "ineligible_nightly_run",
        )
    return {
        key: value[key]
        for key in (
            "id",
            "repository",
            "head_repository",
            "path",
            "name",
            "head_branch",
            "event",
            "status",
            "pull_requests",
            "head_sha",
            "run_attempt",
        )
    }


def validate_run(
    repository: str, run_id: str, output: Path, target: Path, event_attempt: str = ""
) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ProvenanceError("invalid GitHub repository", "invalid_repository")
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ProvenanceError(
            "publication run ID must be a positive integer", "invalid_run_id"
        )
    if event_attempt and not re.fullmatch(r"[1-9][0-9]*", event_attempt):
        raise ProvenanceError("invalid triggering run attempt", "invalid_run_attempt")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
        "User-Agent": "dynamo-frontend-replay",
    }
    base = f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
    url = base + f"/attempts/{event_attempt}" if event_attempt else base
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = verify_run(json.load(response), repository, run_id)
    if event_attempt and metadata["run_attempt"] != int(event_attempt):
        raise ProvenanceError(
            "nightly run attempt differs from triggering event", "run_attempt_mismatch"
        )
    write_json(output, metadata)
    artifact_name = f"frontend-replay-publication-{metadata['run_attempt']}"
    query = urllib.parse.urlencode({"name": artifact_name, "per_page": 100})
    request = urllib.request.Request(base + "/artifacts?" + query, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        artifacts = json.load(response).get("artifacts", [])
    if not any(
        item.get("name") == artifact_name and item.get("expired") is False
        for item in artifacts
    ):
        write_json(
            target,
            {
                "status": "target_unavailable",
                "reason": "publication_not_rerun_in_attempt"
                if metadata["run_attempt"] > 1
                else "publication_descriptor_missing",
            },
        )
        return 2
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        stream.write(f"run_id={metadata['id']}\n")
        stream.write(f"run_attempt={metadata['run_attempt']}\n")
    return 0


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def platforms(manifest: dict[str, Any]) -> dict[str, str]:
    result = {}
    for entry in manifest.get("manifests", []):
        platform = entry.get("platform", {})
        key = "/".join((platform.get("os", ""), platform.get("architecture", "")))
        digest = entry.get("digest", "")
        if key.startswith("linux/") and DIGEST.fullmatch(digest):
            result[key] = digest
    return result


def record(output: Path) -> None:
    sha, tag = os.environ["SOURCE_SHA"], os.environ["NGC_VERSION_TAG"]
    if not COMMIT.fullmatch(sha) or not TAG.fullmatch(tag) or tag[9:] != sha[:7]:
        raise ValueError("invalid nightly source identity")
    staged = os.environ.get("FRONTEND_STAGED") == "true"
    descriptor: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "tag": tag,
        "source_sha": sha,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "frontend_staged": staged,
        "index_digest": None,
        "platforms": {},
    }
    if staged:
        # Never persist command stderr or this reference: staging's org is secret.
        reference = os.environ["STAGING_REFERENCE"]
        try:
            digest = subprocess.run(
                ["crane", "digest", reference],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.strip()
            if not DIGEST.fullmatch(digest):
                raise ValueError("invalid image digest")
            manifest = subprocess.run(
                ["crane", "manifest", reference.rsplit(":", 1)[0] + "@" + digest],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            descriptor["index_digest"] = digest
            descriptor["platforms"] = platforms(json.loads(manifest.stdout))
        except (OSError, ValueError, subprocess.SubprocessError):
            descriptor["reason"] = "staged_digest_unavailable"
    else:
        descriptor["reason"] = "frontend_not_staged"
    write_json(output, descriptor)


def public_manifest(tag: str) -> tuple[str, dict[str, Any]]:
    url = f"https://nvcr.io/v2/nvidia/ai-dynamo/{REPOSITORY}/manifests/{tag}"
    headers = {"Accept": ACCEPT}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise
        challenge = error.headers.get("WWW-Authenticate", "")
        values = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        realm = values.get("realm", "")
        if not realm.startswith("https://nvcr.io/"):
            raise ValueError("unexpected registry authentication realm") from error
        query = urllib.parse.urlencode(
            {key: value for key, value in values.items() if key in ("service", "scope")}
        )
        separator = "&" if "?" in realm else "?"
        with urllib.request.urlopen(realm + separator + query, timeout=30) as auth:
            token = json.load(auth)
        headers["Authorization"] = "Bearer " + (
            token.get("token") or token["access_token"]
        )
        response = urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=30
        )
    with response:
        body = response.read()
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        advertised = response.headers.get("Docker-Content-Digest")
        if advertised and advertised != digest:
            raise ValueError("registry manifest digest mismatch")
        return digest, json.loads(body)


def resolve(
    descriptor_path: Path,
    output: Path,
    platform: str,
    wait_seconds: int,
    run_metadata_path: Path,
) -> int:
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("schema") != SCHEMA
        or descriptor.get("repository") != REPOSITORY
        or not COMMIT.fullmatch(descriptor.get("source_sha", ""))
        or not TAG.fullmatch(descriptor.get("tag", ""))
        or descriptor["tag"][9:] != descriptor["source_sha"][:7]
    ):
        raise ProvenanceError(
            "invalid publication descriptor", "invalid_publication_descriptor"
        )
    if (
        descriptor.get("run_id") != str(metadata["id"])
        or descriptor.get("run_attempt") != str(metadata["run_attempt"])
        or descriptor["source_sha"] != metadata["head_sha"]
    ):
        raise ProvenanceError(
            "publication descriptor does not match its trusted nightly run",
            "publication_run_identity_mismatch",
        )
    expected = descriptor.get("index_digest", "") or ""
    if descriptor.get("frontend_staged") is not True or not DIGEST.fullmatch(expected):
        write_json(
            output,
            {
                "status": "target_unavailable",
                "reason": descriptor.get("reason", "no_staged_frontend"),
            },
        )
        return 2
    started = time.monotonic()
    deadline = started + wait_seconds
    while True:
        reason = "public_promotion_unavailable"
        try:
            digest, manifest = public_manifest(descriptor["tag"])
            available = platforms(manifest)
            selected = available.get(platform)
            if digest != expected:
                reason = "public_digest_differs_from_staging"
            elif (
                not selected
                or descriptor.get("platforms", {}).get(platform) != selected
            ):
                reason = "platform_digest_unavailable_or_mismatched"
                write_json(
                    output,
                    {
                        "status": "target_unavailable",
                        "reason": reason,
                        "resolution_seconds": round(time.monotonic() - started, 3),
                    },
                )
                return 2
            else:
                target = {
                    "status": "ready",
                    "source_sha": descriptor["source_sha"],
                    "tag": descriptor["tag"],
                    "index_digest": digest,
                    "platform": platform,
                    "platform_digest": selected,
                    "image": PUBLIC_REPOSITORY + "@" + selected,
                    "resolution_seconds": round(time.monotonic() - started, 3),
                }
                write_json(output, target)
                github_output = os.environ.get("GITHUB_OUTPUT")
                if github_output:
                    with open(github_output, "a", encoding="utf-8") as stream:
                        stream.write(f"image={target['image']}\n")
                return 0
        except (OSError, ValueError, KeyError, http.client.HTTPException):
            pass
        if time.monotonic() >= deadline:
            write_json(
                output,
                {
                    "status": "target_unavailable",
                    "reason": reason,
                    "resolution_seconds": round(time.monotonic() - started, 3),
                },
            )
            return 2
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def ensure_report(directory: Path, target: Path | None = None) -> None:
    if not (directory / "summary.json").is_file():
        observation = (
            json.loads(target.read_text()) if target and target.is_file() else {}
        )
        rejected = observation.get("status") == "provenance_rejected"
        unavailable = observation.get("status") == "target_unavailable" or (
            target is not None and not target.is_file()
        )
        write_json(
            directory / "summary.json",
            {
                "schema": "dynamo-frontend-replay-result/v1",
                "exit_code": 2,
                "status": "provenance_rejected"
                if rejected
                else "target_unavailable"
                if unavailable
                else "infra_error",
                "reason": observation.get("reason", "publication_evidence_unavailable")
                if unavailable or rejected
                else "replay_did_not_complete",
                "executions": 0,
            },
        )
    if (directory / "junit.xml").is_file():
        return
    (directory / "junit.xml").write_text(
        '<testsuite name="frontend-replay" tests="1" errors="1">'
        '<testcase name="execution"><error message="Replay did not complete"/>'
        "</testcase></testsuite>\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("record")
    capture.add_argument("--output", type=Path, required=True)
    trusted = commands.add_parser("validate-run")
    trusted.add_argument("--run-id", required=True)
    trusted.add_argument("--repository", required=True)
    trusted.add_argument("--output", type=Path, required=True)
    trusted.add_argument("--target", type=Path, required=True)
    trusted.add_argument("--event-attempt", default="")
    verify = commands.add_parser("resolve")
    verify.add_argument("--descriptor", type=Path, required=True)
    verify.add_argument("--run-metadata", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--platform", default="linux/amd64")
    verify.add_argument("--wait-seconds", type=int, default=600)
    report = commands.add_parser("ensure-report")
    report.add_argument("--directory", type=Path, required=True)
    report.add_argument("--target", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "record":
            record(args.output)
        elif args.command == "validate-run":
            return validate_run(
                args.repository,
                args.run_id,
                args.output,
                args.target,
                args.event_attempt,
            )
        elif args.command == "resolve":
            return resolve(
                args.descriptor,
                args.output,
                args.platform,
                args.wait_seconds,
                args.run_metadata,
            )
        else:
            ensure_report(args.directory, args.target)
    except (ValueError, KeyError, TypeError) as exc:
        if args.command not in ("validate-run", "resolve"):
            raise
        write_json(
            args.target if args.command == "validate-run" else args.output,
            {
                "status": "provenance_rejected",
                "reason": exc.reason
                if isinstance(exc, ProvenanceError)
                else "invalid_publication_evidence",
            },
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
