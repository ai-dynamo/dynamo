# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make NIXL's C API loadable by SONAME.

nixl-sys resolves it with a bare dlopen("libnixl_capi.so") and the Dynamo
runtime extension carries no RPATH, so without this the Rust bindings silently
fall back to stub mode.

The library directory is either an installed nixl-cu* wheel (default, found
through this interpreter's package metadata) or an existing NIXL install
(--lib-dir). It is registered with ldconfig and, optionally, exposed at a
stable path for LD_LIBRARY_PATH. Run as root, with the interpreter that has
the nixl-cu* wheel installed, after the last step that installs or upgrades it.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import NoReturn

CAPI = "libnixl_capi.so"
LD_CONF = Path("/etc/ld.so.conf.d/nixl.conf")
AUX_CACHE = Path("/var/cache/ldconfig/aux-cache")


def die(message: str) -> NoReturn:
    raise SystemExit(f"register_nixl_libs: {message}")


def wheel_lib_dir(cuda_major: str | None) -> Path:
    """Return the directory holding libnixl_capi.so in the nixl-cu* wheel."""
    cuda = re.escape(cuda_major) if cuda_major else "[0-9]+"
    dists = [
        d
        for d in distributions()
        if re.fullmatch(rf"nixl[-_.]cu{cuda}", d.metadata.get("Name", ""), re.I)
    ]
    if len(dists) != 1:
        names = sorted(d.metadata.get("Name", "") for d in dists)
        die(f"expected one nixl-cu* distribution, found {names}")
    paths = {
        Path(dists[0].locate_file(f)).resolve()
        for f in dists[0].files or ()
        if Path(f).name == CAPI
    }
    paths = {p for p in paths if p.is_file()}
    if len(paths) != 1:
        die(f"expected one {CAPI}, found {sorted(map(str, paths))}")
    return next(iter(paths)).parent


def link_dir(link: Path, lib_dir: Path) -> None:
    """Symlink the directory, not single files: ld.so expands $ORIGIN from the
    resolved path, so libnixl_capi.so still finds its siblings through it."""
    if link.is_symlink() or link.exists():
        if not link.is_symlink() or link.resolve() != lib_dir:
            die(f"refusing to replace existing path: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(lib_dir)


def dlopens(env_updates: dict[str, str | None]) -> bool:
    """Run the dlopen check in a child process, since the loader reads
    LD_LIBRARY_PATH only at startup."""
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    for key, value in env_updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run(
        [sys.executable, __file__, "--check-dlopen"],
        env=env,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cuda-major", help="Use the nixl-cu<N> wheel when several are installed."
    )
    parser.add_argument("--lib-dir", type=Path, help="Use this NIXL library directory.")
    parser.add_argument(
        "--link", type=Path, help="Symlink this stable path to the library directory."
    )
    parser.add_argument("--check-dlopen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.check_dlopen:
        ctypes.CDLL(CAPI)
        return

    lib_dir = (args.lib_dir or wheel_lib_dir(args.cuda_major)).resolve()
    if not lib_dir.is_dir():
        die(f"missing NIXL library directory: {lib_dir}")
    for lib in (CAPI, "libnixl.so"):
        if not (lib_dir / lib).is_file():
            die(f"missing {lib_dir / lib}; NIXL layout changed")
    if not (lib_dir / "plugins").is_dir():
        die(f"missing {lib_dir / 'plugins'}; NIXL layout changed")

    if args.link:
        link_dir(args.link, lib_dir)
        if not dlopens({"LD_LIBRARY_PATH": str(args.link)}):
            die(f"{CAPI} could not be dlopened via {args.link}")

    LD_CONF.write_text(f"{lib_dir}\n")
    subprocess.run(["ldconfig"], check=True)

    # Processes with a cleared environment rely on ld.so.cache. Inside an image
    # build ldconfig can skip a directory: it reuses aux-cache entries keyed on
    # (st_dev, st_ino, st_size, st_ctime), which overlayfs recycles across
    # layers, and drops a directory whose (st_dev, st_ino) it already queued.
    # Retry once without the aux-cache, then fail.
    if not dlopens({"LD_LIBRARY_PATH": None}):
        AUX_CACHE.unlink(missing_ok=True)
        subprocess.run(["ldconfig"], check=True)
        if not dlopens({"LD_LIBRARY_PATH": None}):
            verbose = subprocess.run(
                ["ldconfig", "-v"], capture_output=True, text=True
            ).stderr
            for line in verbose.splitlines():
                if "given more than once" in line:
                    print(line, file=sys.stderr)
            die(f"ldconfig will not cache {CAPI} from {lib_dir}")

    via = f" via {args.link}" if args.link else ""
    print(f"register_nixl_libs: registered {lib_dir}{via}")


if __name__ == "__main__":
    main()
