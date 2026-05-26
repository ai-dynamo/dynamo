#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install a stable NIXL_PREFIX that points at the native libraries shipped by a
# nixl-cu* Python wheel. This lets non-Python consumers use the same NIXL copy
# that Python imports from the wheel.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: install_nixl_from_wheel [options]

Options:
  --cuda-major <major>       CUDA major used by the nixl-cu wheel, e.g. 12 or 13.
  --python-version <version> Python version used to infer site-packages.
  --site-packages <path>     Python site-packages/dist-packages path.
  --wheel-lib-dir <path>     Explicit .nixl_cu*.mesonpy.libs directory.
  --headers-src <path>       NIXL include directory to copy beside wheel libs.
  --prefix <path>            Stable symlink prefix to create. Default: /opt/dynamo/nixl.
  --skip-headers             Do not install or require headers.
  -h, --help                 Show this help text.

NIXL_REQUIRED_LIBS may override the required library list.
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

prefix="/opt/dynamo/nixl"
cuda_major="${CUDA_MAJOR:-}"
python_version="${PYTHON_VERSION:-}"
site_packages=""
wheel_lib_dir=""
headers_src=""
skip_headers=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cuda-major)
            [ "$#" -ge 2 ] || die "--cuda-major requires a value"
            cuda_major="$2"
            shift 2
            ;;
        --python-version)
            [ "$#" -ge 2 ] || die "--python-version requires a value"
            python_version="$2"
            shift 2
            ;;
        --site-packages)
            [ "$#" -ge 2 ] || die "--site-packages requires a value"
            site_packages="${2%/}"
            shift 2
            ;;
        --wheel-lib-dir)
            [ "$#" -ge 2 ] || die "--wheel-lib-dir requires a value"
            wheel_lib_dir="${2%/}"
            shift 2
            ;;
        --headers-src)
            [ "$#" -ge 2 ] || die "--headers-src requires a value"
            headers_src="${2%/}"
            shift 2
            ;;
        --prefix)
            [ "$#" -ge 2 ] || die "--prefix requires a value"
            prefix="${2%/}"
            shift 2
            ;;
        --skip-headers)
            skip_headers=1
            shift
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

if [ -z "${wheel_lib_dir}" ]; then
    if [ -z "${site_packages}" ]; then
        if [ -z "${python_version}" ]; then
            die "set --site-packages, --python-version, PYTHON_VERSION, or --wheel-lib-dir"
        fi
        site_packages="/usr/local/lib/python${python_version}/dist-packages"
    fi
    [ -n "${cuda_major}" ] || die "set --cuda-major, CUDA_MAJOR, or --wheel-lib-dir"
    wheel_lib_dir="${site_packages}/.nixl_cu${cuda_major}.mesonpy.libs"
fi

if [ ! -d "${wheel_lib_dir}" ]; then
    die "expected NIXL wheel libs at ${wheel_lib_dir}; upstream NIXL wheel layout changed"
fi

read -r -a required_libs <<< "${NIXL_REQUIRED_LIBS:-libnixl.so libnixl_build.so libnixl_common.so libserdes.so libstream.so}"
for lib in "${required_libs[@]}"; do
    if [ ! -f "${wheel_lib_dir}/${lib}" ]; then
        die "missing ${wheel_lib_dir}/${lib}; upstream NIXL wheel layout changed"
    fi
done

if [ ! -d "${wheel_lib_dir}/plugins" ]; then
    die "missing ${wheel_lib_dir}/plugins; upstream NIXL wheel layout changed"
fi

if [ "${skip_headers}" -eq 0 ]; then
    if [ -n "${headers_src}" ]; then
        [ -d "${headers_src}" ] || die "missing NIXL headers source directory: ${headers_src}"
        [ -f "${headers_src}/nixl.h" ] || die "missing ${headers_src}/nixl.h"
        if [ ! -f "${wheel_lib_dir}/include/nixl.h" ]; then
            rm -rf "${wheel_lib_dir}/include"
            cp -a "${headers_src}" "${wheel_lib_dir}/include"
        fi
    fi
    [ -f "${wheel_lib_dir}/include/nixl.h" ] || die "missing ${wheel_lib_dir}/include/nixl.h; pass --headers-src or --skip-headers"
fi

if [ -e "${prefix}" ] && [ ! -L "${prefix}" ]; then
    die "${prefix} already exists and is not a symlink"
fi

mkdir -p "$(dirname "${prefix}")"
ln -sfn "${wheel_lib_dir}" "${prefix}"

echo "Installed NIXL wheel prefix: ${prefix} -> $(readlink -f "${prefix}")"
