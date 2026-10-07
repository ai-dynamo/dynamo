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
  --ucx-prefix <path>        Also expose the wheel's UCX through generic SONAMEs.
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
ucx_prefix=""
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
        --ucx-prefix)
            [ "$#" -ge 2 ] || die "--ucx-prefix requires a value"
            ucx_prefix="${2%/}"
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
        if [ ! -e "${wheel_lib_dir}/include" ]; then
            cp -a "${headers_src}" "${wheel_lib_dir}/include"
        elif [ ! -f "${wheel_lib_dir}/include/nixl.h" ]; then
            die "unexpected header layout under ${wheel_lib_dir}/include; upstream NIXL wheel layout changed"
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

if [ -n "${ucx_prefix}" ]; then
    [ -n "${cuda_major}" ] || die "--ucx-prefix requires --cuda-major"
    command -v patchelf >/dev/null || die "--ucx-prefix requires patchelf"
    ucx_lib_dir="${wheel_lib_dir%/*}/nixl_cu${cuda_major}.libs"
    [ -d "${ucx_lib_dir}/ucx" ] || die "missing wheel UCX modules: ${ucx_lib_dir}/ucx"

    # MPI asks for generic UCX SONAMEs; the wheel uses auditwheel-renamed ones.
    # A symlink alone does not register MPI's name when LD_PRELOAD uses an
    # absolute path. Give the preloaded file the generic SONAME as well, so
    # MPI cannot load system UCX through its RPATH or an inherited search path.
    # Keep the hashed filenames for the wheel's DT_NEEDED references: the
    # loader recognizes the same inode and reuses the already loaded library.
    # Keep aliases beside their targets so UCX can find its modules via $ORIGIN.
    shopt -s nullglob
    for library in libucm libucs libuct libucp; do
        candidates=("${ucx_lib_dir}/${library}-"*.so.0.*)
        [ "${#candidates[@]}" -eq 1 ] || die "expected one wheel ${library}: ${candidates[*]}"
        if [ "$(patchelf --print-soname "${candidates[0]}")" != "${library}.so.0" ]; then
            # Do not modify a file that uv may have hard-linked from its cache.
            patched_library="$(mktemp "${candidates[0]}.XXXXXX")"
            cp --preserve=mode,timestamps "${candidates[0]}" "${patched_library}"
            patchelf --set-soname "${library}.so.0" "${patched_library}"
            mv -f "${patched_library}" "${candidates[0]}"
        fi
        [ "$(patchelf --print-soname "${candidates[0]}")" = "${library}.so.0" ] || \
            die "failed to set ${library} SONAME"
        alias_path="${ucx_lib_dir}/${library}.so.0"
        [ ! -e "${alias_path}" ] || [ -L "${alias_path}" ] || die "refusing to replace ${alias_path}"
        ln -sfn "${candidates[0]##*/}" "${alias_path}"
    done

    # Loading libucs through .so.0 makes its module loader use that suffix too.
    # The wheels retain only fully versioned module files, so restore the links.
    for module_path in "${ucx_lib_dir}"/ucx/*.so.0.*; do
        ln -sfn "${module_path##*/}" "${module_path%%.so.*}.so.0"
    done
    shopt -u nullglob
    for module in libuct_cuda libucm_cuda; do
        [ -f "${ucx_lib_dir}/ucx/${module}.so.0" ] || die "missing wheel UCX module: ${module}"
    done

    [ ! -e "${ucx_prefix}" ] || [ -L "${ucx_prefix}" ] || die "refusing to replace ${ucx_prefix}"
    mkdir -p "$(dirname "${ucx_prefix}")"
    ln -sfn "${ucx_lib_dir}" "${ucx_prefix}"
    echo "Installed UCX wheel prefix: ${ucx_prefix} -> $(readlink -f "${ucx_prefix}")"
fi
