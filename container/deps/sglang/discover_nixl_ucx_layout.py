# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover NIXL's private UCX libraries and native UCX consumers."""

from __future__ import annotations

import re
from importlib.metadata import Distribution, distributions
from pathlib import Path


def cuda_distributions(
    package_prefix: str,
) -> list[tuple[str, str, Distribution]]:
    """Return installed CUDA distributions matching a package prefix."""
    pattern = re.compile(rf"{re.escape(package_prefix)}-cu([0-9]+)", re.IGNORECASE)
    matches = []
    for distribution in distributions():
        name = distribution.metadata.get("Name", "")
        normalized_name = re.sub(r"[-_.]+", "-", name)
        match = pattern.fullmatch(normalized_name)
        if match is not None:
            matches.append((name, match.group(1), distribution))
    return matches


def main() -> None:
    """Print the discovered layout as tab-separated records."""
    nixl_matches = cuda_distributions("nixl")
    if len(nixl_matches) != 1:
        names = sorted(name for name, _, _ in nixl_matches)
        raise SystemExit(
            f"nixl-ucx-compat: expected one nixl-cu* distribution, found {names}"
        )

    nixl_name, nixl_cuda_major, nixl = nixl_matches[0]
    nixl_files = list(nixl.files or ())

    plugins = set()
    for _, _, nvshmem in cuda_distributions("nvidia-nvshmem"):
        for item in nvshmem.files or ():
            if Path(item).name.startswith("nvshmem_transport_ucx.so"):
                path = Path(nvshmem.locate_file(item)).resolve()
                if path.is_file():
                    plugins.add(path)

    aliases = []
    for stem in ("libucp", "libucs"):
        pattern = re.compile(rf"{stem}-[0-9A-Fa-f]+[.]so[.]([0-9]+)(?:[.][0-9]+)*")
        candidates = {}
        for item in nixl_files:
            match = pattern.fullmatch(Path(item).name)
            if match is None:
                continue
            path = Path(nixl.locate_file(item)).resolve()
            if path.is_file():
                candidates[path] = match.group(1)
        if len(candidates) != 1:
            raise SystemExit(
                f"nixl-ucx-compat: expected one private {stem} library, "
                f"found {sorted(map(str, candidates))}"
            )
        target, major = next(iter(candidates.items()))
        aliases.append((f"{stem}.so.{major}", target))

    lib_dirs = {target.parent for _, target in aliases}
    if len(lib_dirs) != 1:
        raise SystemExit(
            "nixl-ucx-compat: UCX libraries span multiple directories: "
            f"{sorted(map(str, lib_dirs))}"
        )
    nixl_lib_dir = next(iter(lib_dirs))

    print(f"nixl\t{nixl_name}\t{nixl_cuda_major}")
    print(f"libdir\t{nixl_lib_dir}")
    for plugin in sorted(plugins):
        print(f"plugin\t{plugin}")
    for alias, target in aliases:
        print(f"alias\t{alias}\t{target}")


if __name__ == "__main__":
    main()
