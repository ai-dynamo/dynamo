# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the AISimulate wheel from Dynamo's exact Rust source patch."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def build(root: Path, source: Path, output: Path) -> None:
    with (root / "Cargo.toml").open("rb") as handle:
        patch = tomllib.load(handle)["patch"]["crates-io"]["aisimulate-core"]
    repository, revision = patch["git"], patch["rev"]
    if repository != "https://github.com/ai-dynamo/aisimulate.git" or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise ValueError(
            "AISimulate source patch must use the canonical repository and a full SHA"
        )

    source.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "fetch", "--depth=1", patch["git"], patch["rev"]],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True
    )
    project = source / "python/aisimulate"
    with (project / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    with (root / "pyproject.toml").open("rb") as handle:
        requirements = tomllib.load(handle)["project"]["dependencies"]
    requirement = next(
        Requirement(item)
        for item in requirements
        if Requirement(item).name == "aisimulate"
    )
    if requirement.url is not None or not requirement.specifier.contains(version):
        raise ValueError(
            f"Source wheel version {version} does not satisfy {requirement}"
        )
    print(f"Building AISimulate {version} from {patch['rev']}", flush=True)
    subprocess.run(
        [
            "maturin",
            "build",
            "--release",
            "--locked",
            "--interpreter",
            sys.executable,
            "--out",
            str(output.resolve()),
        ],
        cwd=project,
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.source, args.output)
