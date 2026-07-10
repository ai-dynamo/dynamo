#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly SOURCE_FILE="src/utils/libfabric/libfabric_rail_manager.cpp"
readonly ACTIVE_REL=".nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so"
readonly COMPAT_REL="nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so"
readonly REQUIRED_CORE_RELS=(
    ".nixl_cu13.mesonpy.libs/libnixl.so"
    ".nixl_cu13.mesonpy.libs/libnixl_common.so"
    ".nixl_cu13.mesonpy.libs/libserdes.so"
)

die() {
    echo "nixl-efa-fi-more: $*" >&2
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
    SGLANG_EFA_NIXL_BASE_REF \
    SGLANG_EFA_NIXL_BASE_TREE \
    SGLANG_EFA_NIXL_BASE_SOURCE_SHA256 \
    SGLANG_EFA_NIXL_PATCH_REF \
    SGLANG_EFA_NIXL_PATCH_SHA256 \
    SGLANG_EFA_NIXL_PATCHED_TREE \
    SGLANG_EFA_NIXL_PATCHED_SOURCE_SHA256 \
    SGLANG_EFA_NIXL_VERSION \
    SGLANG_EFA_NIXL_MESON_VERSION; do
    require_var "$name"
done

[[ "$#" -eq 2 ]] || die "usage: $0 PATCH_FILE OUTPUT_ROOT"
readonly PATCH_FILE="$1"
readonly OUT_ROOT="$2"
[[ -f "$PATCH_FILE" ]] || die "patch file not found: ${PATCH_FILE}"
[[ "$OUT_ROOT" == /* && "$OUT_ROOT" != "/" ]] || \
    die "output root must be an absolute non-root path"
[[ "$(sha256 "$PATCH_FILE")" == "$SGLANG_EFA_NIXL_PATCH_SHA256" ]] || \
    die "vendored patch checksum mismatch"

for tool in git meson ninja patchelf readelf nm sha256sum python3; do
    command -v "$tool" >/dev/null || die "required tool not found: ${tool}"
done
[[ "$(meson --version)" == "$SGLANG_EFA_NIXL_MESON_VERSION" ]] || \
    die "expected Meson ${SGLANG_EFA_NIXL_MESON_VERSION}, found $(meson --version)"

case "$TARGETARCH" in
    arm64|amd64) ;;
    *) die "unsupported TARGETARCH ${TARGETARCH}" ;;
esac

readonly TMP_ROOT="$(mktemp -d /tmp/nixl-efa-fi-more.XXXXXX)"
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
        "nixl-efa-fi-more: expected "
        f"nixl-cu13=={expected_version}, found {dist.version}"
    )

expected = [os.environ["EXPECTED_ACTIVE_REL"], os.environ["EXPECTED_COMPAT_REL"]]
files = {str(item): item for item in (dist.files or ())}
actual_plugins = sorted(
    name for name in files if Path(name).name == "libplugin_LIBFABRIC.so"
)
if actual_plugins != sorted(expected):
    raise SystemExit(
        "nixl-efa-fi-more: unexpected nixl-cu13 LIBFABRIC layout: "
        + repr(actual_plugins)
    )
for name in expected:
    path = Path(dist.locate_file(files[name]))
    if not path.is_file():
        raise SystemExit(f"nixl-efa-fi-more: wheel file is missing: {path}")
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
        raise SystemExit(
            f"nixl-efa-fi-more: required core wheel file is missing: {name}"
        )
    path = Path(dist.locate_file(files[name]))
    if not path.is_file():
        raise SystemExit(
            f"nixl-efa-fi-more: required core library is missing: {path}"
        )
    print(path)
PY
)
[[ "${#CORE_PATHS[@]}" -eq "${#REQUIRED_CORE_RELS[@]}" ]] || \
    die "failed to locate the stock NIXL core libraries"

git clone --filter=blob:none --no-checkout https://github.com/ai-dynamo/nixl.git "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" checkout --detach "$SGLANG_EFA_NIXL_BASE_REF"
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SGLANG_EFA_NIXL_BASE_REF" ]] || \
    die "NIXL base commit mismatch"
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD^{tree})" == "$SGLANG_EFA_NIXL_BASE_TREE" ]] || \
    die "NIXL base tree mismatch"
[[ "$(sha256 "${SOURCE_ROOT}/${SOURCE_FILE}")" == "$SGLANG_EFA_NIXL_BASE_SOURCE_SHA256" ]] || \
    die "NIXL base source checksum mismatch"

git -C "$SOURCE_ROOT" apply --index --whitespace=error-all "$PATCH_FILE"
git -C "$SOURCE_ROOT" diff --cached --check
mapfile -t CHANGED_FILES < <(git -C "$SOURCE_ROOT" diff --cached --name-only)
[[ "${#CHANGED_FILES[@]}" -eq 1 && "${CHANGED_FILES[0]}" == "$SOURCE_FILE" ]] || \
    die "EFA FI_MORE patch changed files other than ${SOURCE_FILE}"
[[ "$(git -C "$SOURCE_ROOT" write-tree)" == "$SGLANG_EFA_NIXL_PATCHED_TREE" ]] || \
    die "patched NIXL tree does not match ${SGLANG_EFA_NIXL_PATCH_REF}"
[[ "$(sha256 "${SOURCE_ROOT}/${SOURCE_FILE}")" == "$SGLANG_EFA_NIXL_PATCHED_SOURCE_SHA256" ]] || \
    die "patched NIXL source checksum mismatch"

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
    ldd_output="$(LD_LIBRARY_PATH="$CHECK_LD_LIBRARY_PATH" ldd "$plugin" 2>&1)"
    echo "$ldd_output"
    if grep -q "not found" <<<"$ldd_output"; then
        die "${plugin}: unresolved runtime dependency"
    fi
    LD_LIBRARY_PATH="$CHECK_LD_LIBRARY_PATH" PLUGIN="$plugin" python3 - <<'PY'
import ctypes
import os

ctypes.CDLL(os.environ["PLUGIN"], mode=os.RTLD_NOW | os.RTLD_LOCAL)
PY
done

readonly PROVENANCE="${OUT_ROOT}/provenance"
cp "$PATCH_FILE" "${PROVENANCE}/disable_efa_fi_more.patch"
cp "${SOURCE_ROOT}/LICENSE" "${PROVENANCE}/NIXL-LICENSE"
printf '%s\n' "$SGLANG_EFA_NIXL_BASE_REF" > "${PROVENANCE}/BASE_COMMIT"
printf '%s\n' "$SGLANG_EFA_NIXL_BASE_TREE" > "${PROVENANCE}/BASE_TREE"
printf '%s\n' "$SGLANG_EFA_NIXL_BASE_SOURCE_SHA256" > "${PROVENANCE}/BASE_SOURCE_FILE_SHA256"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCH_REF" > "${PROVENANCE}/SOURCE_COMMIT"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCHED_TREE" > "${PROVENANCE}/PATCHED_TREE"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCH_SHA256" > "${PROVENANCE}/PATCH_SHA256"
printf '%s\n' "$SGLANG_EFA_NIXL_PATCHED_SOURCE_SHA256" > "${PROVENANCE}/SOURCE_FILE_SHA256"
printf '%s\n' "$SGLANG_EFA_NIXL_VERSION" > "${PROVENANCE}/NIXL_VERSION"
printf '%s\n' "$SGLANG_EFA_NIXL_MESON_VERSION" > "${PROVENANCE}/MESON_VERSION"
printf '%s\n' "$TARGETARCH" > "${PROVENANCE}/TARGETARCH"
printf '%s\n' "https://github.com/ai-dynamo/nixl" > "${PROVENANCE}/SOURCE_URL"
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
echo "nixl-efa-fi-more: built and validated ${TARGETARCH} SGLang EFA plugins"
cat "${PROVENANCE}/SHA256SUMS"
