#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe staging without disclosing its registry, then verify public promotion."""

from __future__ import annotations

import argparse
import hashlib
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
    descriptor_path: Path, output: Path, platform: str, wait_seconds: int
) -> int:
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if (
        descriptor.get("schema") != SCHEMA
        or descriptor.get("repository") != REPOSITORY
        or not COMMIT.fullmatch(descriptor.get("source_sha", ""))
        or not TAG.fullmatch(descriptor.get("tag", ""))
        or descriptor["tag"][9:] != descriptor["source_sha"][:7]
    ):
        raise ValueError("invalid publication descriptor")
    expected = descriptor.get("index_digest", "") or ""
    if descriptor.get("frontend_staged") is not True or not DIGEST.fullmatch(expected):
        write_json(
            output, {"status": "target_unavailable", "reason": "no_staged_frontend"}
        )
        return 2
    deadline = time.monotonic() + wait_seconds
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
            else:
                target = {
                    "status": "ready",
                    "source_sha": descriptor["source_sha"],
                    "tag": descriptor["tag"],
                    "index_digest": digest,
                    "platform": platform,
                    "platform_digest": selected,
                    "image": PUBLIC_REPOSITORY + "@" + selected,
                }
                write_json(output, target)
                github_output = os.environ.get("GITHUB_OUTPUT")
                if github_output:
                    with open(github_output, "a", encoding="utf-8") as stream:
                        stream.write(f"image={target['image']}\n")
                return 0
        except (OSError, ValueError, KeyError):
            pass
        if time.monotonic() >= deadline:
            write_json(output, {"status": "target_unavailable", "reason": reason})
            return 2
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def ensure_report(directory: Path) -> None:
    if (directory / "summary.json").is_file():
        return
    write_json(
        directory / "summary.json",
        {"status": "infra_error", "reason": "replay_did_not_complete", "executions": 0},
    )
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
    verify = commands.add_parser("resolve")
    verify.add_argument("--descriptor", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--platform", default="linux/amd64")
    verify.add_argument("--wait-seconds", type=int, default=600)
    report = commands.add_parser("ensure-report")
    report.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        record(args.output)
    elif args.command == "resolve":
        return resolve(args.descriptor, args.output, args.platform, args.wait_seconds)
    else:
        ensure_report(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
