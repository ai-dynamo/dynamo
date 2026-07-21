#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 <efa-version> <archive-sha256> <archive-size> [--skip-nccl]" >&2
    exit 2
fi

efa_version=$1
archive_sha256=$2
archive_size=$3
skip_nccl=${4:-}
if [[ -n "${skip_nccl}" && "${skip_nccl}" != "--skip-nccl" ]]; then
    echo "unsupported option: ${skip_nccl}" >&2
    exit 2
fi
cache_dir=${EFA_INSTALLER_CACHE_DIR:-/var/cache/efa-installer}
archive="${cache_dir}/aws-efa-installer-${efa_version}.tar.gz"
archive_url="https://efa-installer.amazonaws.com/aws-efa-installer-${efa_version}.tar.gz"

archive_is_valid() {
    [[ -f "${archive}" ]] &&
        [[ "$(wc -c < "${archive}")" == "${archive_size}" ]] &&
        printf '%s  %s\n' "${archive_sha256}" "${archive}" | sha256sum -c - >/dev/null
}

mkdir -p "${cache_dir}"
if ! archive_is_valid; then
    rm -f "${archive}" "${archive}.tmp"
    curl --retry 3 --retry-delay 2 -fsSL -o "${archive}.tmp" "${archive_url}"
    mv "${archive}.tmp" "${archive}"
fi
archive_is_valid

work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' EXIT
tar -xf "${archive}" -C "${work_dir}"
cd "${work_dir}/aws-efa-installer"

command -v apt-get >/dev/null 2>&1 || {
    echo "EFA Installer --build-ngc requires a Debian-family image" >&2
    exit 1
}
apt-get update

# Force the installer-owned libfabric prefix back to package provenance. This
# safely removes the legacy 2.5.1 overlay only after its underlying EFA package
# registrations have been purged, so the installer cannot skip a reinstall.
efa_libfabric_packages=()
for package in libfabric-aws-dev libfabric-aws-bin libfabric1-aws; do
    if dpkg-query -W "${package}" >/dev/null 2>&1; then
        efa_libfabric_packages+=("${package}")
    fi
done
if [[ ${#efa_libfabric_packages[@]} -gt 0 ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get purge -y "${efa_libfabric_packages[@]}"
fi
rm -rf /opt/amazon/efa

# Framework base images may already contain an older or differently named NCCL
# OFI package. The 1.49 NGC package does not declare Conflicts/Replaces even
# though it owns the same files, so normalize that package family first.
nccl_packages=()
for package in \
    libnccl-ofi \
    libnccl-ofi-ngc-v1 \
    libnccl-ofi-ngc-v2 \
    libnccl-ofi-ngc-v3; do
    if dpkg-query -W "${package}" >/dev/null 2>&1; then
        nccl_packages+=("${package}")
    fi
done
if [[ ${#nccl_packages[@]} -gt 0 ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get purge -y "${nccl_packages[@]}"
fi
rm -rf /opt/amazon/ofi-nccl /opt/amazon/aws-ofi-nccl \
    /etc/ld.so.conf.d/aws-ofi-nccl.conf

installer_args=(
    -y
    --build-ngc
    --skip-kmod
    --skip-limit-conf
    --skip-mpi
    --no-verify
)
if [[ "${skip_nccl}" == "--skip-nccl" ]]; then
    installer_args+=(--skip-plugin)
fi
./efa_installer.sh "${installer_args[@]}"

# Remove the unowned payload left by the retired source-built 2.5.1 overlay.
# The installer restores its own package-managed symlink below; fail closed if
# any other part of that overlay remains active.
rm -f /opt/amazon/efa/lib/libfabric.so.1.31.1
printf '%s\n' /opt/amazon/efa/lib > /etc/ld.so.conf.d/000_efa.conf
ldconfig

libfabric=/opt/amazon/efa/lib/libfabric.so.1.30.0
[[ "$(readlink /opt/amazon/efa/lib/libfabric.so.1)" == "libfabric.so.1.30.0" ]]
[[ ! -e /opt/amazon/efa/lib/libfabric.so.1.31.1 ]]
grep -aFq efa_data_path_direct_post_read "${libfabric}"
grep -aFq efa_data_path_direct_post_write "${libfabric}"
/opt/amazon/efa/bin/fi_info --version | grep -F "libfabric: 2.4.0amzn5.0"

if [[ "${skip_nccl}" == "--skip-nccl" ]]; then
    ! dpkg-query -W 'libnccl-ofi*' >/dev/null 2>&1
    [[ ! -e /opt/amazon/ofi-nccl ]]
else
    [[ "$(dpkg-query -W -f='${Version}' libnccl-ofi-ngc-v3)" == "1.20.0-1" ]]
    dpkg-query -S /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so |
        grep -F 'libnccl-ofi-ngc-v3:' >/dev/null
fi

rm -rf /var/lib/apt/lists/*
