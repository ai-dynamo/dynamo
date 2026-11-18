#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose the engines whose mocker parity test a change needs.

An engine is selected when one of its CUDA runtime images in
`container/context.yaml` changed, or when its mocker server or the shared
parity code changed.
"""

import argparse
import json
import subprocess

import yaml

ENGINES = ("vllm", "sglang", "trtllm")
SHARED = ("lib/mocker/servers/tests/", "lib/mocker/servers/parity/")


def git(*args):
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def images(revision):
    """Each engine's CUDA runtime images, newest CUDA version first."""
    context = yaml.safe_load(git("show", f"{revision}:container/context.yaml"))
    return {
        engine: [
            "{runtime_image}:{runtime_image_tag}".format(**context[engine][key])
            for key in sorted(
                (key for key in context[engine] if key.startswith("cuda")),
                key=lambda key: tuple(int(part) for part in key[4:].split(".")),
                reverse=True,
            )
        ]
        for engine in ENGINES
    }


def mirrored(image, dockerhub_mirror):
    """Pulls a Docker Hub image, which names no registry host, through the mirror."""
    if dockerhub_mirror and "." not in image.split("/")[0]:
        return dockerhub_mirror.rstrip("/") + "/" + image
    return image


def plan(base, head, dockerhub_mirror=None):
    changed = git("diff", "--name-only", base, head).splitlines()
    shared = any(path.startswith(SHARED) for path in changed) or (
        ".github/workflows/shared-mocker-parity.yml" in changed
    )
    before, after = images(base), images(head)
    return [
        {"engine": engine, "image": mirrored(after[engine][0], dockerhub_mirror)}
        for engine in ENGINES
        if shared
        or before[engine] != after[engine]
        or any(path.startswith(f"lib/mocker/servers/{engine}/") for path in changed)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--github-output")
    parser.add_argument(
        "--dockerhub-mirror", help="Registry prefix for Docker Hub images"
    )
    args = parser.parse_args()
    matrix = plan(args.base, args.head, args.dockerhub_mirror)
    print(json.dumps(matrix, indent=2))
    if args.github_output:
        with open(args.github_output, "a") as output:
            output.write(f"matrix={json.dumps(matrix)}\n")


if __name__ == "__main__":
    main()
