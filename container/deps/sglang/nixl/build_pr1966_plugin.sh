#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly SOURCE_FILES=(
    "src/plugins/libfabric/libfabric_backend.cpp"
    "src/utils/libfabric/libfabric_common.h"
    "src/utils/libfabric/libfabric_rail_manager.cpp"
    "src/utils/libfabric/libfabric_rail_manager.h"
)
readonly ACTIVE_REL=".nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so"
readonly COMPAT_REL="nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so"
readonly REQUIRED_CORE_RELS=(
    ".nixl_cu13.mesonpy.libs/libnixl.so"
    ".nixl_cu13.mesonpy.libs/libnixl_common.so"
    ".nixl_cu13.mesonpy.libs/libserdes.so"
)

die() {
    echo "nixl-pr1966: $*" >&2
    exit 1
}

require_var() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "required environment variable ${name} is empty"
}

sha256() {
    sha256sum "$1" | awk '{print $1}'
}

dynamic_path() {
    local plugin="$1"
    readelf -d "$plugin" | sed -n \
        's/.*Library \(rpath\|runpath\): \[\(.*\)\]/\2/p'
}

dynamic_path_tag() {
    local plugin="$1"
    readelf -d "$plugin" | sed -n 's/.*(\(RPATH\|RUNPATH\)).*/\1/p'
}

needed_libraries() {
    local plugin="$1"
    readelf -d "$plugin" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p'
}

validate_plugin() {
    local plugin="$1"
    local expected_tag="$2"
    local expected_path="$3"
    local expected_numa="$4"
    local machine
    local actual_tag
    local actual_path
    local soname
    local required

    machine="$(readelf -h "$plugin" | sed -n 's/^ *Machine: *//p')"
    case "${TARGETARCH}" in
        arm64) [[ "$machine" == "AArch64" ]] || die "${plugin}: unexpected machine ${machine}" ;;
        amd64) [[ "$machine" == "Advanced Micro Devices X86-64" ]] || die "${plugin}: unexpected machine ${machine}" ;;
        *) die "unsupported TARGETARCH ${TARGETARCH}" ;;
    esac
    if readelf --wide -S "$plugin" | grep -F ".nv_fatbin" >/dev/null; then
        die "${plugin}: unexpected CUDA fatbin would make the plugin GPU-architecture-specific"
    fi

    soname="$(patchelf --print-soname "$plugin")"
    [[ "$soname" == "libplugin_LIBFABRIC.so" ]] || \
        die "${plugin}: unexpected SONAME ${soname}"

    actual_tag="$(dynamic_path_tag "$plugin")"
    actual_path="$(dynamic_path "$plugin")"
    [[ "$actual_tag" == "$expected_tag" ]] || \
        die "${plugin}: expected ${expected_tag}, found ${actual_tag:-none}"
    [[ "$actual_path" == "$expected_path" ]] || \
        die "${plugin}: loader path changed from ${expected_path} to ${actual_path}"
    [[ "$actual_path" != *"/tmp/"* && "$actual_path" != *"/build"* ]] || \
        die "${plugin}: build-tree path leaked into ${actual_tag}"

    for required in libnixl_common.so libserdes.so libfabric.so.1; do
        needed_libraries "$plugin" | grep -Fx "$required" >/dev/null || \
            die "${plugin}: missing NEEDED entry ${required}"
    done
    needed_libraries "$plugin" | grep -Fx "$expected_numa" >/dev/null || \
        die "${plugin}: missing stock NUMA dependency ${expected_numa}"
    if needed_libraries "$plugin" | grep -Fx "libnuma.so.1" >/dev/null; then
        die "${plugin}: unbundled libnuma.so.1 dependency remains"
    fi

    nm -D --defined-only "$plugin" | awk '{print $3}' | grep -Fx nixl_plugin_init >/dev/null || \
        die "${plugin}: nixl_plugin_init is not exported"
    nm -D --defined-only "$plugin" | awk '{print $3}' | grep -Fx nixl_plugin_fini >/dev/null || \
        die "${plugin}: nixl_plugin_fini is not exported"

    if readelf --version-info "$plugin" | grep -F "FABRIC_1.9" >/dev/null; then
        die "${plugin}: requires FABRIC_1.9 and is incompatible with EFA 1.49"
    fi
    for required in fi_getinfo fi_freeinfo fi_dupinfo; do
        readelf --wide -Ws "$plugin" | grep -E \
            "[[:space:]]${required}@FABRIC_1[.]8([[:space:]]|$)" >/dev/null || \
            die "${plugin}: ${required} does not resolve at FABRIC_1.8"
    done
}

build_variant() {
    local stock="$1"
    local output="$2"
    local stock_tag
    local stock_path
    local stock_numa
    local -a stock_numa_entries

    stock_tag="$(dynamic_path_tag "$stock")"
    stock_path="$(dynamic_path "$stock")"
    [[ "$stock_tag" == "RPATH" || "$stock_tag" == "RUNPATH" ]] || \
        die "${stock}: expected exactly one RPATH or RUNPATH entry"
    [[ -n "$stock_path" ]] || die "${stock}: loader path is empty"

    mapfile -t stock_numa_entries < <(needed_libraries "$stock" | grep '^libnuma')
    [[ "${#stock_numa_entries[@]}" -eq 1 ]] || \
        die "${stock}: expected one NUMA dependency, found ${#stock_numa_entries[@]}"
    stock_numa="${stock_numa_entries[0]}"
    [[ "$stock_numa" == libnuma-*.so.* ]] || \
        die "${stock}: expected a wheel-hashed NUMA dependency, found ${stock_numa}"

    needed_libraries "$BUILT_PLUGIN" | grep -Fx "libnuma.so.1" >/dev/null || \
        die "built plugin no longer has the expected libnuma.so.1 dependency"

    mkdir -p "$(dirname "$output")"
    install -m 0755 "$BUILT_PLUGIN" "$output"
    patchelf --replace-needed libnuma.so.1 "$stock_numa" "$output"
    if [[ "$stock_tag" == "RPATH" ]]; then
        patchelf --force-rpath --set-rpath "$stock_path" "$output"
    else
        patchelf --set-rpath "$stock_path" "$output"
    fi

    validate_plugin "$output" "$stock_tag" "$stock_path" "$stock_numa"
}

for name in \
    TARGETARCH \
    SGLANG_EFA_NIXL_WHEEL_BUILD_REF \
    SGLANG_EFA_NIXL_WHEEL_BUILD_TREE \
    SGLANG_EFA_NIXL_BASE_REF \
    SGLANG_EFA_NIXL_BASE_TREE \
    SGLANG_EFA_NIXL_UPSTREAM_PR_HEAD \
    SGLANG_EFA_NIXL_UPSTREAM_PR_TREE \
    SGLANG_EFA_NIXL_PATCH_REF \
    SGLANG_EFA_NIXL_PATCH_SHA256 \
    SGLANG_EFA_NIXL_PATCH_ID \
    SGLANG_EFA_NIXL_PATCHED_TREE \
    SGLANG_EFA_NIXL_BUILDER_SHA256 \
    SGLANG_EFA_NIXL_VALIDATOR_SHA256 \
    SGLANG_EFA_NIXL_VERSION \
    SGLANG_EFA_NIXL_MESON_VERSION; do
    require_var "$name"
done

[[ "$#" -eq 3 ]] || die "usage: $0 PATCH_FILE OUTPUT_ROOT SEMANTICS_VALIDATOR"
readonly PATCH_FILE="$1"
readonly OUT_ROOT="$2"
readonly SEMANTICS_VALIDATOR="$3"
[[ -f "$PATCH_FILE" ]] || die "patch file not found: ${PATCH_FILE}"
[[ -f "$SEMANTICS_VALIDATOR" ]] || \
    die "semantics validator not found: ${SEMANTICS_VALIDATOR}"
[[ "$OUT_ROOT" == /* && "$OUT_ROOT" != "/" ]] || \
    die "output root must be an absolute non-root path"

for tool in git meson ninja patchelf readelf nm sha256sum strings python3; do
    command -v "$tool" >/dev/null || die "required tool not found: ${tool}"
done

[[ "$(sha256 "$PATCH_FILE")" == "$SGLANG_EFA_NIXL_PATCH_SHA256" ]] || \
    die "vendored patch checksum mismatch"
[[ "$(sha256 "$0")" == "$SGLANG_EFA_NIXL_BUILDER_SHA256" ]] || \
    die "plugin builder checksum mismatch"
[[ "$(sha256 "$SEMANTICS_VALIDATOR")" == "$SGLANG_EFA_NIXL_VALIDATOR_SHA256" ]] || \
    die "semantics validator checksum mismatch"
[[ "$(git patch-id --stable < "$PATCH_FILE" | awk 'NR == 1 {print $1}')" == \
    "$SGLANG_EFA_NIXL_PATCH_ID" ]] || die "vendored patch ID mismatch"

[[ "$(meson --version)" == "$SGLANG_EFA_NIXL_MESON_VERSION" ]] || \
    die "expected Meson ${SGLANG_EFA_NIXL_MESON_VERSION}, found $(meson --version)"

case "$TARGETARCH" in
    arm64|amd64) ;;
    *) die "unsupported TARGETARCH ${TARGETARCH}" ;;
esac

readonly TMP_ROOT="$(mktemp -d /tmp/nixl-pr1966.XXXXXX)"
trap 'rm -rf "$TMP_ROOT"' EXIT
readonly SOURCE_ROOT="${TMP_ROOT}/nixl"

mapfile -t WHEEL_PATHS < <(
    EXPECTED_VERSION="$SGLANG_EFA_NIXL_VERSION" \
    EXPECTED_ACTIVE_REL="$ACTIVE_REL" \
    EXPECTED_COMPAT_REL="$COMPAT_REL" \
    python3 - <<'PY'
import importlib.metadata
import os
from pathlib import Path

dist = importlib.metadata.distribution("nixl-cu13")
expected_version = os.environ["EXPECTED_VERSION"]
if dist.version != expected_version:
    raise SystemExit(
        f"nixl-pr1966: expected nixl-cu13=={expected_version}, found {dist.version}"
    )

expected = [os.environ["EXPECTED_ACTIVE_REL"], os.environ["EXPECTED_COMPAT_REL"]]
files = {str(item): item for item in (dist.files or ())}
actual_plugins = sorted(
    name for name in files if Path(name).name == "libplugin_LIBFABRIC.so"
)
if actual_plugins != sorted(expected):
    raise SystemExit(
        "nixl-pr1966: unexpected nixl-cu13 LIBFABRIC layout: "
        + repr(actual_plugins)
    )
for name in expected:
    path = Path(dist.locate_file(files[name]))
    if not path.is_file():
        raise SystemExit(f"nixl-pr1966: wheel file is missing: {path}")
    print(path)
PY
)
[[ "${#WHEEL_PATHS[@]}" -eq 2 ]] || die "failed to locate both stock plugins"
readonly ACTIVE_PLUGIN="${WHEEL_PATHS[0]}"
readonly COMPAT_PLUGIN="${WHEEL_PATHS[1]}"

mapfile -t CORE_PATHS < <(
    EXPECTED_RELS="$(IFS=:; echo "${REQUIRED_CORE_RELS[*]}")" \
    python3 - <<'PY'
import importlib.metadata
import os
from pathlib import Path

dist = importlib.metadata.distribution("nixl-cu13")
files = {str(item): item for item in (dist.files or ())}
for name in os.environ["EXPECTED_RELS"].split(":"):
    if name not in files:
        raise SystemExit(f"nixl-pr1966: required core wheel file is missing: {name}")
    path = Path(dist.locate_file(files[name]))
    if not path.is_file():
        raise SystemExit(f"nixl-pr1966: required core library is missing: {path}")
    print(path)
PY
)
[[ "${#CORE_PATHS[@]}" -eq "${#REQUIRED_CORE_RELS[@]}" ]] || \
    die "failed to locate the stock NIXL core libraries"
readonly WHEEL_BUILD_SHORT="${SGLANG_EFA_NIXL_WHEEL_BUILD_REF:0:8}"
[[ "$(strings "${CORE_PATHS[1]}" | grep -Fxc "$WHEEL_BUILD_SHORT")" -eq 1 ]] || \
    die "stock nixl-cu13 wheel does not embed build revision ${WHEEL_BUILD_SHORT}"

git clone --filter=blob:none --no-checkout https://github.com/ai-dynamo/nixl.git "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" checkout --detach "$SGLANG_EFA_NIXL_BASE_REF"
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SGLANG_EFA_NIXL_BASE_REF" ]] || \
    die "NIXL base commit mismatch"
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD^{tree})" == "$SGLANG_EFA_NIXL_BASE_TREE" ]] || \
    die "NIXL base tree mismatch"

git -C "$SOURCE_ROOT" fetch --quiet --filter=blob:none --no-tags \
    origin "$SGLANG_EFA_NIXL_WHEEL_BUILD_REF"
[[ "$(git -C "$SOURCE_ROOT" rev-parse FETCH_HEAD)" == \
    "$SGLANG_EFA_NIXL_WHEEL_BUILD_REF" ]] || die "wheel build commit mismatch"
[[ "$(git -C "$SOURCE_ROOT" rev-parse FETCH_HEAD^{tree})" == \
    "$SGLANG_EFA_NIXL_WHEEL_BUILD_TREE" ]] || die "wheel build tree mismatch"
[[ "$SGLANG_EFA_NIXL_WHEEL_BUILD_TREE" == "$SGLANG_EFA_NIXL_BASE_TREE" ]] || \
    die "wheel build and source-anchor trees differ"

git -C "$SOURCE_ROOT" fetch --quiet --filter=blob:none --no-tags \
    origin "$SGLANG_EFA_NIXL_UPSTREAM_PR_HEAD"
[[ "$(git -C "$SOURCE_ROOT" rev-parse FETCH_HEAD)" == \
    "$SGLANG_EFA_NIXL_UPSTREAM_PR_HEAD" ]] || die "upstream PR head mismatch"
[[ "$(git -C "$SOURCE_ROOT" rev-parse FETCH_HEAD^{tree})" == \
    "$SGLANG_EFA_NIXL_UPSTREAM_PR_TREE" ]] || die "upstream PR tree mismatch"

git -C "$SOURCE_ROOT" apply --index --whitespace=error-all "$PATCH_FILE"
git -C "$SOURCE_ROOT" diff --cached --check
mapfile -t CHANGED_FILES < <(git -C "$SOURCE_ROOT" diff --cached --name-only)
[[ "${#CHANGED_FILES[@]}" -eq "${#SOURCE_FILES[@]}" ]] || \
    die "PR #1966 backport changed an unexpected number of files"
for i in "${!SOURCE_FILES[@]}"; do
    [[ "${CHANGED_FILES[$i]}" == "${SOURCE_FILES[$i]}" ]] || \
        die "PR #1966 backport changed unexpected file ${CHANGED_FILES[$i]}"
done
[[ "$(git -C "$SOURCE_ROOT" write-tree)" == "$SGLANG_EFA_NIXL_PATCHED_TREE" ]] || \
    die "patched NIXL tree does not match ${SGLANG_EFA_NIXL_PATCH_REF}"

python3 "$SEMANTICS_VALIDATOR" "$SOURCE_ROOT"

# Guard the endpoint-selection correction that superseded the first tested PR
# head: local rails and remote endpoints must each index using their own count.
grep -Fq "selected_rails.size())" \
    "${SOURCE_ROOT}/src/utils/libfabric/libfabric_rail_manager.cpp" || \
    die "local-rail selection guard is missing"
grep -Fq "remote_selected_endpoints.size())" \
    "${SOURCE_ROOT}/src/utils/libfabric/libfabric_rail_manager.cpp" || \
    die "remote-endpoint selection guard is missing"

# The Python bindings and telemetry plugin are unrelated to LIBFABRIC and pull
# in build-only dependencies absent from the runtime image. Guard their exact
# Meson entries before removing them from this plugin-only build tree.
[[ "$(grep -Fxc "subdir('bindings')" "${SOURCE_ROOT}/src/meson.build")" -eq 1 ]] || \
    die "unexpected NIXL bindings Meson layout"
[[ "$(grep -Fxc "subdir('telemetry')" "${SOURCE_ROOT}/src/plugins/meson.build")" -eq 1 ]] || \
    die "unexpected NIXL telemetry Meson layout"
sed -i "/^subdir('bindings')$/d" "${SOURCE_ROOT}/src/meson.build"
sed -i "/^subdir('telemetry')$/d" "${SOURCE_ROOT}/src/plugins/meson.build"

meson setup "${SOURCE_ROOT}/build" "$SOURCE_ROOT" \
    --buildtype=release \
    --prefix=/opt/nvidia/nvda_nixl \
    -Denable_plugins=LIBFABRIC \
    -Dbuild_tests=false \
    -Dbuild_examples=false \
    -Dinstall_headers=false \
    -Dlibfabric_path=/opt/amazon/efa \
    -Dcudapath_inc=/usr/local/cuda/include \
    -Dcudapath_lib=/usr/local/cuda/lib64 \
    -Dcudapath_stub=/usr/local/cuda/lib64/stubs \
    -Dnixl_cuda_arch_list=100
ninja -C "${SOURCE_ROOT}/build" -j2 \
    src/plugins/libfabric/libplugin_LIBFABRIC.so

readonly BUILT_PLUGIN="${SOURCE_ROOT}/build/src/plugins/libfabric/libplugin_LIBFABRIC.so"
[[ -f "$BUILT_PLUGIN" ]] || die "Meson did not produce the LIBFABRIC plugin"

mkdir -p "$OUT_ROOT"
rm -rf "${OUT_ROOT}/plugins" "${OUT_ROOT}/provenance"
mkdir -p "${OUT_ROOT}/plugins" "${OUT_ROOT}/provenance"
build_variant "$ACTIVE_PLUGIN" "${OUT_ROOT}/plugins/mesonpy/libplugin_LIBFABRIC.so"
build_variant "$COMPAT_PLUGIN" "${OUT_ROOT}/plugins/compat/libplugin_LIBFABRIC.so"

# Validate each ABI-adjusted artifact at its real wheel path. The CUDA stub is
# used only for the build-time dlopen check; it is not copied into the image.
install -m 0755 "${OUT_ROOT}/plugins/mesonpy/libplugin_LIBFABRIC.so" "$ACTIVE_PLUGIN"
install -m 0755 "${OUT_ROOT}/plugins/compat/libplugin_LIBFABRIC.so" "$COMPAT_PLUGIN"
mkdir -p "${TMP_ROOT}/cuda-stubs"
ln -s /usr/local/cuda/lib64/stubs/libcuda.so "${TMP_ROOT}/cuda-stubs/libcuda.so.1"
readonly CORE_DIR="$(dirname "${CORE_PATHS[0]}")"
readonly CHECK_LD_LIBRARY_PATH="${TMP_ROOT}/cuda-stubs:/opt/amazon/efa/lib:${CORE_DIR}:${LD_LIBRARY_PATH:-}"
for plugin in "$ACTIVE_PLUGIN" "$COMPAT_PLUGIN"; do
    ldd_output="$(env -u LD_PRELOAD LD_LIBRARY_PATH="$CHECK_LD_LIBRARY_PATH" ldd -v "$plugin" 2>&1)"
    echo "$ldd_output"
    if grep -q "not found" <<<"$ldd_output"; then
        die "${plugin}: unresolved runtime dependency"
    fi
    grep -F "libfabric.so.1 => /opt/amazon/efa/lib/libfabric.so.1" <<<"$ldd_output" >/dev/null || \
        die "${plugin}: did not resolve EFA 1.49 libfabric"
    env -u LD_PRELOAD LD_LIBRARY_PATH="$CHECK_LD_LIBRARY_PATH" PLUGIN="$plugin" python3 - <<'PY'
import ctypes
import os

ctypes.CDLL(os.environ["PLUGIN"], mode=os.RTLD_NOW | os.RTLD_LOCAL)
PY
done

readonly PROVENANCE="${OUT_ROOT}/provenance"
cp "$PATCH_FILE" "${PROVENANCE}/pr1966-1.3.0-backport.patch"
cp "$0" "${PROVENANCE}/build_pr1966_plugin.sh"
cp "$SEMANTICS_VALIDATOR" "${PROVENANCE}/validate_pr1966_semantics.py"
cp "${SOURCE_ROOT}/LICENSE" "${PROVENANCE}/NIXL-LICENSE"
printf '%s\n' "$SGLANG_EFA_NIXL_WHEEL_BUILD_REF" > "${PROVENANCE}/WHEEL_BUILD_COMMIT"
printf '%s\n' "$SGLANG_EFA_NIXL_WHEEL_BUILD_TREE" > "${PROVENANCE}/WHEEL_BUILD_TREE"
printf '%s\n' "$SGLANG_EFA_NIXL_BASE_REF" > "${PROVENANCE}/BASE_COMMIT"
printf '%s\n' "$SGLANG_EFA_NIXL_BASE_TREE" > "${PROVENANCE}/BASE_TREE"
printf '%s\n' "$SGLANG_EFA_NIXL_UPSTREAM_PR_HEAD" > "${PROVENANCE}/UPSTREAM_PR_HEAD"
printf '%s\n' "$SGLANG_EFA_NIXL_UPSTREAM_PR_TREE" > "${PROVENANCE}/UPSTREAM_PR_TREE"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCH_REF" > "${PROVENANCE}/BACKPORT_REF"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCHED_TREE" > "${PROVENANCE}/PATCHED_TREE"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCH_SHA256" > "${PROVENANCE}/PATCH_SHA256"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCH_ID" > "${PROVENANCE}/PATCH_ID"
printf '%s\n' "$SGLANG_EFA_NIXL_BUILDER_SHA256" > "${PROVENANCE}/BUILDER_SHA256"
printf '%s\n' "$SGLANG_EFA_NIXL_VALIDATOR_SHA256" > "${PROVENANCE}/VALIDATOR_SHA256"
printf '%s\n' "$SGLANG_EFA_NIXL_VERSION" > "${PROVENANCE}/NIXL_VERSION"
printf '%s\n' "$SGLANG_EFA_NIXL_MESON_VERSION" > "${PROVENANCE}/MESON_VERSION"
printf '%s\n' "$TARGETARCH" > "${PROVENANCE}/TARGETARCH"
printf '%s\n' "https://github.com/ai-dynamo/nixl" > "${PROVENANCE}/SOURCE_URL"
(
    cd "$SOURCE_ROOT"
    for source_file in "${SOURCE_FILES[@]}"; do
        git show "HEAD:${source_file}" | sha256sum | awk -v file="$source_file" \
            '{print $1 "  " file}'
    done
) > "${PROVENANCE}/BASE_SOURCE_SHA256SUMS"
(
    cd "$SOURCE_ROOT"
    sha256sum "${SOURCE_FILES[@]}"
) > "${PROVENANCE}/PATCHED_SOURCE_SHA256SUMS"
sha256sum "${CORE_PATHS[@]}" > "${PROVENANCE}/NIXL_CORE_SHA256SUMS"
{
    printf '%s  %s\n' \
        "$(sha256 "${OUT_ROOT}/plugins/mesonpy/libplugin_LIBFABRIC.so")" \
        "$ACTIVE_PLUGIN"
    printf '%s  %s\n' \
        "$(sha256 "${OUT_ROOT}/plugins/compat/libplugin_LIBFABRIC.so")" \
        "$COMPAT_PLUGIN"
} > "${PROVENANCE}/SHA256SUMS"

sha256sum -c "${PROVENANCE}/NIXL_CORE_SHA256SUMS"
sha256sum -c "${PROVENANCE}/SHA256SUMS"
echo "nixl-pr1966: built and validated ${TARGETARCH} SGLang EFA plugins"
cat "${PROVENANCE}/SHA256SUMS"
