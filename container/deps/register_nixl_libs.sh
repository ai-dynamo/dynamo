#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Make NIXL's C API loadable by SONAME. nixl-sys resolves it with a bare
# dlopen("libnixl_capi.so") and the Dynamo runtime extension carries no RPATH,
# so without this the Rust bindings silently fall back to stub mode.
#
# The library directory is either an installed nixl-cu* wheel (default) or an
# existing NIXL install (--lib-dir). It is registered with ldconfig and,
# optionally, exposed at a stable path for LD_LIBRARY_PATH. Run as root after
# the last step that installs or upgrades the nixl-cu* wheel.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: register_nixl_libs.sh [options]

Options:
  --python <path>    Python used to locate the nixl-cu* wheel. Default: python3.
  --cuda-major <N>   Use the nixl-cu<N> wheel when several are installed.
  --lib-dir <path>   Use this NIXL library directory instead of the wheel's.
  --link <path>      Symlink this stable path to the library directory.
  -h, --help         Show this help text.
USAGE
}

die() {
    echo "register_nixl_libs: $*" >&2
    exit 1
}

python="python3"
cuda_major=""
lib_dir=""
link=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --python)
            [ "$#" -ge 2 ] || die "--python requires a value"
            python="$2"
            shift 2
            ;;
        --cuda-major)
            [ "$#" -ge 2 ] || die "--cuda-major requires a value"
            cuda_major="$2"
            shift 2
            ;;
        --lib-dir)
            [ "$#" -ge 2 ] || die "--lib-dir requires a value"
            lib_dir="${2%/}"
            shift 2
            ;;
        --link)
            [ "$#" -ge 2 ] || die "--link requires a value"
            link="${2%/}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [ -z "${lib_dir}" ]; then
    lib_dir="$(NIXL_CUDA_MAJOR="${cuda_major}" "${python}" - <<'EOF'
import os
import re
import sys
from importlib.metadata import distributions
from pathlib import Path

cuda = os.environ["NIXL_CUDA_MAJOR"] or "[0-9]+"
dists = [
    d for d in distributions()
    if re.fullmatch(rf"nixl[-_.]cu{cuda}", d.metadata.get("Name", ""), re.IGNORECASE)
]
if len(dists) != 1:
    names = sorted(d.metadata.get("Name", "") for d in dists)
    sys.exit(f"register_nixl_libs: expected one nixl-cu* distribution, found {names}")
paths = {
    Path(dists[0].locate_file(f)).resolve()
    for f in dists[0].files or ()
    if Path(f).name == "libnixl_capi.so"
}
paths = {p for p in paths if p.is_file()}
if len(paths) != 1:
    sys.exit(f"register_nixl_libs: expected one libnixl_capi.so, found {sorted(map(str, paths))}")
print(next(iter(paths)).parent)
EOF
)"
fi

[ -d "${lib_dir}" ] || die "missing NIXL library directory: ${lib_dir}"
for lib in libnixl_capi.so libnixl.so; do
    [ -f "${lib_dir}/${lib}" ] || die "missing ${lib_dir}/${lib}; NIXL layout changed"
done
[ -d "${lib_dir}/plugins" ] || die "missing ${lib_dir}/plugins; NIXL layout changed"

# Link the directory, not single files: ld.so expands $ORIGIN from the resolved
# path, so libnixl_capi.so still finds its siblings through the link.
if [ -n "${link}" ]; then
    if [ -e "${link}" ] || [ -L "${link}" ]; then
        [ -L "${link}" ] && [ "$(readlink -f "${link}")" = "$(readlink -f "${lib_dir}")" ] || \
            die "refusing to replace existing path: ${link}"
    else
        mkdir -p "$(dirname "${link}")"
        ln -s "${lib_dir}" "${link}"
    fi
fi

echo "${lib_dir}" > /etc/ld.so.conf.d/nixl.conf
ldconfig

dlopen_capi() {
    env -u LD_PRELOAD "$@" "${python}" -c 'import ctypes; ctypes.CDLL("libnixl_capi.so")'
}

if [ -n "${link}" ]; then
    dlopen_capi LD_LIBRARY_PATH="${link}" || \
        die "libnixl_capi.so could not be dlopened via ${link}"
fi

# Processes with a cleared environment rely on ld.so.cache. Inside an image
# build ldconfig can skip a directory: it reuses aux-cache entries keyed on
# (st_dev, st_ino, st_size, st_ctime), which overlayfs recycles across layers,
# and drops a directory whose (st_dev, st_ino) it already queued. Retry once
# without the aux-cache, then fail.
if ! dlopen_capi -u LD_LIBRARY_PATH 2>/dev/null; then
    rm -f /var/cache/ldconfig/aux-cache
    ldconfig
    if ! dlopen_capi -u LD_LIBRARY_PATH; then
        ldconfig -v 2>&1 >/dev/null | grep -F "given more than once" >&2 || true
        die "ldconfig will not cache libnixl_capi.so from ${lib_dir}"
    fi
fi

echo "register_nixl_libs: registered ${lib_dir}${link:+ via ${link}}"
