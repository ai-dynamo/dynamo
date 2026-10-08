# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the temporary Git-pinned C API against the installed native NIXL core.

Run with Python 3.11+, a C++20 compiler, and Git. The output directory contains
libnixl_capi.so and its source checkout. Put the output on LD_LIBRARY_PATH
before the installed NIXL directory. The native core and plugins are unchanged.
Remove this helper when nixl-sys and the native wheel share a released version.
"""

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--native-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = [
        tomllib.loads((args.repo / f"lib/{name}/Cargo.toml").read_text())[
            "dependencies"
        ]["nixl-sys"]
        for name in ("memory", "llm")
    ]
    repo, rev = pins[0]["git"], pins[0]["rev"]
    if (repo, rev) != (pins[1]["git"], pins[1]["rev"]):
        raise ValueError("both nixl-sys dependencies must use the same source")
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ValueError("an immutable commit is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    if not (source / ".git").exists():
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "fetch", "--depth", "1", repo, rev], check=True
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != rev:
        raise ValueError("C API source does not match the Cargo pin")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
        raise ValueError("C API source checkout must be clean")
    version = tomllib.loads((source / "Cargo.toml").read_text())["workspace"][
        "package"
    ]["version"]
    if version != args.native_version:
        raise ValueError("keep the native core at the binding's version")
    libraries = args.library_dir.resolve()
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            str(source / "src/bindings/rust/wrapper.cpp"),
            *[f"-I{source / 'src' / name}" for name in ("api/cpp", "infra", "core")],
            f"-L{libraries}",
            "-lnixl",
            "-lnixl_build",
            "-Wl,-z,defs",
            "-o",
            str(output / "libnixl_capi.so"),
        ],
        check=True,
    )
    symbols = subprocess.check_output(
        ["nm", "-D", str(output / "libnixl_capi.so")], text=True
    )
    if " T nixl_capi_opt_args_set_include_conn_info\n" not in symbols:
        raise RuntimeError("the connection-info setter is missing from the C API")
    (output / "source.json").write_text(
        json.dumps(
            {
                "repository": repo,
                "commit": rev,
                "native_version": version,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
