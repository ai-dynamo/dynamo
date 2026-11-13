#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Purge a reviewed orphan set only when APT plans no other package changes."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys

_PACKAGE = re.compile(r"^[a-z0-9][a-z0-9+.-]+$")
_REVIEWED_PACKAGES = frozenset(
    {
        "libplacebo338",
        "libgl1",
        "libglx0",
        "libgbm1",
        "libgl1-mesa-dri",
        "libglx-mesa0",
        "mesa-libgallium",
    }
)
_GL_BINDINGS = frozenset(
    {
        "pyopengl",
        "pyopengl-accelerate",
        "glfw",
        "moderngl",
        "pyrender",
        "pyglet",
        "open3d",
        "opencv-python",
        "opencv-contrib-python",
    }
)


def validate_headless(build_information: str, distributions: list[str]) -> None:
    gui = re.search(
        r"^\s*GUI:\s*(\S+)", build_information, re.MULTILINE | re.IGNORECASE
    )
    opengl = re.findall(
        r"^\s*OpenGL(?: support)?:\s*(\S+)",
        build_information,
        re.MULTILINE | re.IGNORECASE,
    )
    if (
        not gui
        or gui.group(1).upper() not in {"NONE", "NO"}
        or any(value.upper() != "NO" for value in opengl)
    ):
        raise RuntimeError(
            "Refusing GL library removal: OpenCV must remain headless with OpenGL disabled"
        )
    installed = {re.sub(r"[-_.]+", "-", name).lower() for name in distributions}
    consumers = installed & _GL_BINDINGS
    if consumers:
        raise RuntimeError(
            f"Refusing GL library removal: installed graphics bindings {sorted(consumers)}"
        )


def remove_orphans(packages: list[str]) -> None:
    requested = set(packages)
    if (
        not requested
        or not requested <= _REVIEWED_PACKAGES
        or any(not _PACKAGE.fullmatch(package) for package in requested)
    ):
        raise ValueError(
            f"Only the reviewed orphan package set is supported: {sorted(requested)}"
        )
    environment = {**os.environ, "LC_ALL": "C", "DEBIAN_FRONTEND": "noninteractive"}
    native_arch = subprocess.run(
        ["dpkg", "--print-architecture"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    if native_arch not in {"amd64", "arm64"}:
        raise RuntimeError(f"Unreviewed native architecture: {native_arch}")
    status = subprocess.run(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${db:Status-Status}\n"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    states = dict(line.split("\t", 1) for line in status.splitlines())
    installed = {}
    for package in requested:
        identity = (
            f"{package}:{native_arch}"
            if f"{package}:{native_arch}" in states
            else package
        )
        if identity in states and states[identity] != "not-installed":
            installed[package] = identity
    if not installed:
        print("Reviewed orphan packages are absent; no removal needed")
        return
    for package in installed:
        if states[installed[package]] != "installed":
            raise RuntimeError(
                f"Unexpected {package} package state: {states[installed[package]]}"
            )
    command = [
        "apt-get",
        "-o",
        "APT::Get::AutomaticRemove=false",
        "purge",
        *[f"{package}:{native_arch}" for package in sorted(installed)],
    ]
    plan = subprocess.run(
        [*command[:1], "--simulate", *command[1:]],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    changes = [
        line.split()
        for line in plan.splitlines()
        if line.startswith(("Remv ", "Purg ", "Inst ", "Conf "))
    ]
    removed = {
        row[1].removesuffix(f":{native_arch}")
        for row in changes
        if row[0] in {"Remv", "Purg"}
    }
    if removed != set(installed) or any(row[0] in {"Inst", "Conf"} for row in changes):
        raise RuntimeError(
            f"Refusing orphan removal: APT proposes unrelated changes: {changes}"
        )
    subprocess.run([*command[:1], "--yes", *command[1:]], check=True, env=environment)
    print(
        f"Removed only reviewed orphans {sorted(installed)}; automatic removal disabled"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--require-headless-opencv", action="store_true")
    parser.add_argument("--check-headless-only", action="store_true")
    args = parser.parse_args()
    if args.require_headless_opencv or args.check_headless_only:
        cv2 = importlib.import_module("cv2")
        validate_headless(
            cv2.getBuildInformation(),
            [
                dist.metadata["Name"]
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            ],
        )
        if args.check_headless_only:
            numpy = importlib.import_module("numpy")
            cv2.resize(numpy.zeros((2, 2, 3), dtype=numpy.uint8), (1, 1))
            print("Fresh-process headless OpenCV import, build info and resize passed")
            return
    remove_orphans(args.package)
    if args.require_headless_opencv:
        # A fresh interpreter cannot keep a removed library mapped from preflight.
        subprocess.run([sys.executable, __file__, "--check-headless-only"], check=True)


if __name__ == "__main__":
    main()
