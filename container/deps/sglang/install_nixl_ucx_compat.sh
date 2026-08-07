#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly OUTPUT_DIR="${1:-/opt/dynamo/nixl-ucx-compat}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

die() {
    echo "nixl-ucx-compat: $*" >&2
    exit 1
}

validate_ucx_dependencies() {
    local consumer="$1"
    local resolved="$2"
    local dependency_count=0
    local soname arrow resolved_path remainder

    while read -r soname arrow resolved_path remainder; do
        [[ "${arrow}" == "=>" ]] || continue
        ((dependency_count += 1))
        if [[ "${resolved_path}" == "not" && "${remainder}" == found* ]]; then
            die "${consumer}: unresolved UCX dependency ${soname}; the generic alias set may need updating"
        fi
        case "${resolved_path}" in
            "${OUTPUT_DIR}/"*|"${NIXL_LIB_DIR}/"*) ;;
            *)
                die "${consumer}: ${soname} resolved outside NIXL's UCX: ${resolved_path}"
                ;;
        esac
    done < <(grep -E '(^|[[:space:]/])libuc[mpst][-.]' <<<"${resolved}" || true)

    [[ "${dependency_count}" -gt 0 ]] || \
        die "${consumer}: no UCX dependencies found in ldd output"
}

[[ "${OUTPUT_DIR}" == /* && "${OUTPUT_DIR}" != "/" ]] || \
    die "output directory must be an absolute non-root path"

for tool in dirname env grep ldd ln mkdir python3 readlink; do
    command -v "${tool}" >/dev/null || die "required tool not found: ${tool}"
done

readonly DISCOVER_SCRIPT="${SCRIPT_DIR}/discover_nixl_ucx_layout.py"
[[ -f "${DISCOVER_SCRIPT}" ]] || die "layout discovery script not found: ${DISCOVER_SCRIPT}"

# NIXL CUDA wheels carry auditwheel-renamed UCX libraries, while other native
# components can request the generic UCX SONAMEs. Resolve both names to the same
# wheel-owned files so one process cannot load two UCX implementations.
LAYOUT_OUTPUT="$(python3 "${DISCOVER_SCRIPT}")"

mapfile -t LAYOUT <<<"${LAYOUT_OUTPUT}"

declare -a PLUGINS=()
declare -A ALIASES=()
NIXL_PACKAGE=""
NIXL_LIB_DIR=""
UCX_VERSION=""
for record in "${LAYOUT[@]}"; do
    IFS=$'\t' read -r kind first second <<<"${record}"
    case "${kind}" in
        nixl) NIXL_PACKAGE="${first}" ;;
        ucx) UCX_VERSION="${first}" ;;
        libdir) NIXL_LIB_DIR="${first}" ;;
        plugin) PLUGINS+=("${first}") ;;
        alias) ALIASES["${first}"]="${second}" ;;
        *) die "unexpected layout record: ${record}" ;;
    esac
done

[[ -n "${NIXL_PACKAGE}" ]] || die "failed to identify the NIXL CUDA wheel"
[[ -n "${UCX_VERSION}" ]] || die "failed to identify the NIXL UCX version"
[[ -d "${NIXL_LIB_DIR}" ]] || die "failed to locate the NIXL private library directory"
# The NVSHMEM transport directly needs generic libucp and libucs only. Their
# libuct and libucm dependencies use auditwheel-mangled names resolved by
# $ORIGIN, so adding generic aliases for those libraries would be unnecessary.
[[ "${#ALIASES[@]}" -eq 2 ]] || die "failed to locate both NIXL UCX core libraries"
if [[ "${#PLUGINS[@]}" -eq 0 ]] && \
    [[ "${NIXL_UCX_COMPAT_ALLOW_NO_CONSUMER:-0}" != "1" ]]; then
    die "no NVSHMEM UCX transport found; aliases were not validated against a consumer;" \
        "set NIXL_UCX_COMPAT_ALLOW_NO_CONSUMER=1 to accept this"
fi

# Keep the generic names beside NIXL's private libraries. Loading libucp or
# libucs through a separate symlink directory changes $ORIGIN and prevents UCX
# from finding both its hashed core dependencies and its ucx/ module directory.
# These aliases are not tracked by the wheel's RECORD, so a later NIXL uninstall
# or upgrade can leave stale or dangling links. Run this after the final install
# step that can modify the nixl-cu* wheel.
for alias in "${!ALIASES[@]}"; do
    target="${ALIASES[${alias}]}"
    [[ -f "${target}" ]] || die "alias target is missing: ${target}"
    [[ "${target%/*}" == "${NIXL_LIB_DIR}" ]] || \
        die "alias target is outside the NIXL private library directory: ${target}"
    alias_path="${NIXL_LIB_DIR}/${alias}"
    if [[ -e "${alias_path}" || -L "${alias_path}" ]]; then
        [[ -L "${alias_path}" && "$(readlink -f "${alias_path}")" == "${target}" ]] || \
            die "refusing to replace existing path: ${alias_path}"
    else
        ln -s "${target##*/}" "${alias_path}"
    fi
    [[ "$(readlink -f "${alias_path}")" == "${target}" ]] || \
        die "failed to install ${alias}"
done

# Expose the discovered wheel directory through a stable image path for ENV.
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
    [[ -L "${OUTPUT_DIR}" && "$(readlink -f "${OUTPUT_DIR}")" == "${NIXL_LIB_DIR}" ]] || \
        die "refusing to replace existing output path: ${OUTPUT_DIR}"
else
    mkdir -p "$(dirname "${OUTPUT_DIR}")"
    ln -s "${NIXL_LIB_DIR}" "${OUTPUT_DIR}"
fi

# Do not inherit the image's ambient NIXL paths here: they would mask a broken
# $ORIGIN closure in the compatibility layout.
for plugin in "${PLUGINS[@]}"; do
    resolved="$(env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="${OUTPUT_DIR}" ldd "${plugin}")"
    validate_ucx_dependencies "${plugin}" "${resolved}"
    grep -E '^[[:space:]]*libucp[.]so[^[:space:]]*[[:space:]]+=>' <<<"${resolved}" >/dev/null || \
        die "${plugin}: generic libucp dependency was not found"
    grep -E '^[[:space:]]*libucs[.]so[^[:space:]]*[[:space:]]+=>' <<<"${resolved}" >/dev/null || \
        die "${plugin}: generic libucs dependency was not found"
done

# UCX discovers loadable transports relative to libucs at runtime. Validate the
# CUDA modules and their private UCX dependency closure through the stable path.
for module_name in libuct_cuda.so libucm_cuda.so; do
    module_path="${OUTPUT_DIR}/ucx/${module_name}"
    [[ -e "${module_path}" ]] || die "missing NIXL UCX CUDA module: ${module_path}"
    resolved="$(env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="${OUTPUT_DIR}" ldd "${module_path}")"
    validate_ucx_dependencies "${module_path}" "${resolved}"
done

if [[ "${#PLUGINS[@]}" -eq 0 ]]; then
    echo "nixl-ucx-compat: no NVSHMEM UCX transport found; consumer validation explicitly skipped"
fi
echo "nixl-ucx-compat: installed ${#ALIASES[@]} aliases for" \
    "${NIXL_PACKAGE} UCX ${UCX_VERSION} in ${NIXL_LIB_DIR}"
