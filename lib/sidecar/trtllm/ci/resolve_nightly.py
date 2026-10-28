#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the latest published regular TRT-LLM nightly to an immutable image."""

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY = "nvcr.io/nvidia/tensorrt-llm/release"
REGISTRY_PATH = "https://nvcr.io/v2/nvidia/tensorrt-llm/release"
NIGHTLY_TAG = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?\.dev(\d{12})")
MANIFEST_TYPES = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def latest_tag(tags):
    """Select by publication timestamp, excluding feature and release aliases."""
    regular = [tag for tag in tags if NIGHTLY_TAG.fullmatch(tag)]
    if not regular:
        raise ValueError("No regular TRT-LLM nightly tags were published")
    return max(regular, key=lambda tag: (NIGHTLY_TAG.fullmatch(tag)[1], tag))


class Registry:
    def __init__(self):
        self.token = None

    def get(self, path):
        headers = {"Accept": MANIFEST_TYPES}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(f"{REGISTRY_PATH}/{path}", headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            if error.code != 401 or self.token:
                raise
            challenge = dict(
                re.findall(r'(\w+)="([^"]+)"', error.headers["WWW-Authenticate"])
            )
            realm = challenge.pop("realm")
            if urllib.parse.urlparse(realm).netloc != "nvcr.io" or not realm.startswith(
                "https://"
            ):
                raise ValueError("Unexpected NGC authentication endpoint") from error
            with urllib.request.urlopen(
                realm + "?" + urllib.parse.urlencode(challenge), timeout=30
            ) as auth_response:
                auth = json.load(auth_response)
            self.token = auth.get("token") or auth["access_token"]
            return self.get(path)
        with response:
            return response.read(), response.headers

    def manifest(self, reference):
        data, headers = self.get(f"manifests/{reference}")
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if headers.get("Docker-Content-Digest", digest) != digest:
            raise ValueError("Registry manifest digest does not match its content")
        if reference.startswith("sha256:") and reference != digest:
            raise ValueError("Registry returned a different manifest than requested")
        return json.loads(data), digest


def resolve(config, registry):
    if config["repository"] != REPOSITORY:
        raise ValueError("Expected the official TRT-LLM release repository")
    data, _ = registry.get("tags/list")
    tag = latest_tag(json.loads(data)["tags"])
    manifest, digest = registry.manifest(tag)
    if "manifests" in manifest:
        amd64 = [
            item
            for item in manifest["manifests"]
            if item.get("platform", {}).get("os") == "linux"
            and item.get("platform", {}).get("architecture") == "amd64"
        ]
        if len(amd64) != 1:
            raise ValueError("Expected exactly one linux/amd64 nightly image")
        manifest, _ = registry.manifest(amd64[0]["digest"])
    data, _ = registry.get(f"blobs/{manifest['config']['digest']}")
    environment = dict(
        entry.split("=", 1) for entry in json.loads(data)["config"]["Env"]
    )
    commit = environment.get("TRT_LLM_GIT_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Nightly image does not identify its TRT-LLM source commit")
    return {
        **config,
        "image": f"{REPOSITORY}:{tag}@{digest}",
        "upstream_commit": commit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("nightly.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    resolved = resolve(json.loads(args.config.read_text()), Registry())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resolved, indent=2) + "\n")
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(f"image={resolved['image']}\n")
    print(json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()
