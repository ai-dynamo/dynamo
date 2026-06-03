# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a local Fern workspace that does not require private global themes."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "fern"
LOCAL_DIR = REPO_ROOT / ".fern-local"


def copy_path(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def write_local_docs_yml() -> None:
    source = SOURCE_DIR / "docs.yml"
    destination = LOCAL_DIR / "docs.yml"

    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    filtered = [line for line in lines if not line.startswith("global-theme:")]
    destination.write_text("".join(filtered), encoding="utf-8")


def prepare_workspace() -> Path:
    LOCAL_DIR.mkdir(exist_ok=True)

    for name in ("fern.config.json", "main.css", "components"):
        copy_path(SOURCE_DIR / name, LOCAL_DIR / name)

    write_local_docs_yml()
    return LOCAL_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or run a local Fern docs workspace without global-theme."
    )
    parser.add_argument(
        "fern_args",
        nargs=argparse.REMAINDER,
        help="Optional Fern CLI arguments to run from .fern-local, e.g. docs dev.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate .fern-local and do not run the Fern CLI.",
    )
    args = parser.parse_args()

    workspace = prepare_workspace()
    print(f"Prepared local Fern workspace: {workspace}", flush=True)

    if args.prepare_only:
        return 0

    fern_args = args.fern_args or ["docs", "dev"]
    if fern_args and fern_args[0] == "--":
        fern_args = fern_args[1:]

    try:
        return subprocess.call(["fern", *fern_args], cwd=workspace)
    except FileNotFoundError:
        print(
            "Fern CLI not found. Install it with `npm install -g fern-api`, "
            "then rerun this command.",
            file=sys.stderr,
        )
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
