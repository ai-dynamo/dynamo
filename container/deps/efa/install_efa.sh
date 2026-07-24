#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <efa-version> <archive-sha256> [--skip-plugin]" >&2
    exit 2
fi

efa_version=$1
archive_sha256=$2
skip_plugin=${3:-}
if [[ -n "${skip_plugin}" && "${skip_plugin}" != "--skip-plugin" ]]; then
    echo "unsupported option: ${skip_plugin}" >&2
    exit 2
fi

cache_dir=${EFA_INSTALLER_CACHE_DIR:-/var/cache/efa-installer}
archive="${cache_dir}/aws-efa-installer-${efa_version}.tar.gz"
url="https://efa-installer.amazonaws.com/aws-efa-installer-${efa_version}.tar.gz"

mkdir -p "${cache_dir}"
if ! printf '%s  %s\n' "${archive_sha256}" "${archive}" | sha256sum -c - >/dev/null 2>&1; then
    rm -f "${archive}"
    curl --retry 3 --retry-delay 2 -fsSL -o "${archive}" "${url}"
fi
printf '%s  %s\n' "${archive_sha256}" "${archive}" | sha256sum -c -

work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' EXIT
tar -xf "${archive}" -C "${work_dir}"
cd "${work_dir}/aws-efa-installer"

apt-get update

# Framework base images can contain older EFA or NCCL-OFI packages. Remove
# those package families before installing the pinned NGC userspace stack.
installed=()
for package in \
    libfabric-aws-dev libfabric-aws-bin libfabric1-aws \
    libnccl-ofi libnccl-ofi-ngc-v1 libnccl-ofi-ngc-v2 libnccl-ofi-ngc-v3; do
    if dpkg-query -W "${package}" >/dev/null 2>&1; then
        installed+=("${package}")
    fi
done
if [[ ${#installed[@]} -gt 0 ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get purge -y "${installed[@]}"
fi
rm -rf /opt/amazon/efa /opt/amazon/ofi-nccl /opt/amazon/aws-ofi-nccl
# The NGC plugin package does not own this loader entry. Some framework base
# images leave it behind after the package and plugin directory are removed,
# and EFA 1.49 rejects the stale path instead of replacing it.
rm -f /etc/ld.so.conf.d/aws-ofi-nccl.conf

installer_args=(
    -y
    --build-ngc
    --skip-kmod
    --skip-limit-conf
    --skip-mpi
    --no-verify
)
if [[ "${skip_plugin}" == "--skip-plugin" ]]; then
    installer_args+=(--skip-plugin)
fi
./efa_installer.sh "${installer_args[@]}"

rm -rf /var/lib/apt/lists/*
