#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <nixl-version> <nixl-source-ref> <patch-sha256> <patch-file>" >&2
    exit 2
fi

nixl_version=$1
source_ref=$2
patch_sha256=$3
patch_file=$4
source_root=/tmp/nixl-pr1966
active_plugin=/usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so
compat_plugin=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so
core_library=/usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/libnixl_common.so

EXPECTED_NIXL_VERSION="${nixl_version}" python3 - <<'PY'
import importlib.metadata as metadata
import os

expected_nixl = os.environ["EXPECTED_NIXL_VERSION"]
expected = {"sglang": "0.5.14", "nixl": expected_nixl, "nixl-cu13": expected_nixl}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise SystemExit(f"expected {package}=={version}, found {actual}")
PY

for plugin in "${active_plugin}" "${compat_plugin}"; do
    [[ -f "${plugin}" ]]
done
[[ -f "${core_library}" ]]
[[ "$(strings "${core_library}" | grep -Fxc "${source_ref:0:8}")" -eq 1 ]]
printf '%s  %s\n' "${patch_sha256}" "${patch_file}" | sha256sum -c -

git clone --filter=blob:none https://github.com/ai-dynamo/nixl.git "${source_root}"
git -C "${source_root}" checkout --detach "${source_ref}"
[[ "$(git -C "${source_root}" rev-parse HEAD)" == "${source_ref}" ]]
git -C "${source_root}" apply --check --whitespace=error-all "${patch_file}"
git -C "${source_root}" apply --whitespace=error-all "${patch_file}"
git -C "${source_root}" diff --check

# These targets are unrelated to the native LIBFABRIC plugin and require
# additional dependencies that are intentionally absent from this builder.
sed -i "/^subdir('bindings')$/d" "${source_root}/src/meson.build"
sed -i "/^subdir('telemetry')$/d" "${source_root}/src/plugins/meson.build"

meson setup "${source_root}/build" "${source_root}" \
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
ninja -C "${source_root}/build" -j2 \
    src/plugins/libfabric/libplugin_LIBFABRIC.so

built_plugin=${source_root}/build/src/plugins/libfabric/libplugin_LIBFABRIC.so
[[ -f "${built_plugin}" ]]

replace_plugin() {
    local destination=$1
    local loader_path
    local loader_tag
    local numa_library
    local tmp_plugin

    loader_path=$(patchelf --print-rpath "${destination}")
    if readelf -d "${destination}" | grep -Fq '(RPATH)'; then
        loader_tag=RPATH
    elif readelf -d "${destination}" | grep -Fq '(RUNPATH)'; then
        loader_tag=RUNPATH
    else
        echo "${destination}: missing RPATH/RUNPATH" >&2
        return 1
    fi

    numa_library=$(patchelf --print-needed "${destination}" | grep '^libnuma' | head -n1)
    [[ "${numa_library}" == libnuma-*.so.* ]]

    tmp_plugin=${destination}.new
    install -m 0755 "${built_plugin}" "${tmp_plugin}"
    patchelf --replace-needed libnuma.so.1 "${numa_library}" "${tmp_plugin}"
    if [[ "${loader_tag}" == RPATH ]]; then
        patchelf --force-rpath --set-rpath "${loader_path}" "${tmp_plugin}"
    else
        patchelf --set-rpath "${loader_path}" "${tmp_plugin}"
    fi
    mv "${tmp_plugin}" "${destination}"
}

replace_plugin "${active_plugin}"
replace_plugin "${compat_plugin}"
