#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export PATH="/opt/amazon/efa/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

usage() {
  cat <<'EOF'
Usage:
  check-efa-userspace.sh check
  EFA_INSTALLER_VERSION=<version> EFA_INSTALLER_SHA256=<sha256> check-efa-userspace.sh install

The default mode is check. Install mode downloads a versioned AWS EFA installer
archive, verifies its SHA-256 checksum, and runs the installer without kernel
module or limit configuration changes.
EOF
}

check_efa_userspace() {
  if ! command -v fi_info >/dev/null 2>&1; then
    cat >&2 <<'EOF'
EFA userspace is not installed: fi_info was not found.

Install the AWS EFA userspace stack in the runtime image or by using your
cluster's approved node/container bootstrap flow before deploying the H100
disaggregated manifests.
EOF
    exit 1
  fi

  if ! fi_info -p efa >/tmp/dynamo-efa-fi-info.txt 2>&1; then
    cat >&2 <<'EOF'
The EFA provider is not available to libfabric.

Check that the runtime image contains the EFA userspace libraries and that the
pod requests the EFA device resources exposed by the cluster.
EOF
    cat /tmp/dynamo-efa-fi-info.txt >&2
    exit 1
  fi

  echo "EFA userspace detected."
  fi_info -p efa | head -80
}

install_efa_userspace() {
  if [ "$(id -u)" != "0" ]; then
    echo "Install mode must run as root." >&2
    exit 1
  fi

  : "${EFA_INSTALLER_VERSION:?Set EFA_INSTALLER_VERSION to a pinned AWS EFA installer version.}"
  : "${EFA_INSTALLER_SHA256:?Set EFA_INSTALLER_SHA256 to the installer archive SHA-256.}"

  for tool in curl sha256sum tar; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "Install mode requires ${tool} in the runtime image." >&2
      exit 1
    fi
  done

  workdir="$(mktemp -d)"
  trap 'rm -rf "${workdir}"' EXIT

  archive="${workdir}/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
  curl -fsSL \
    "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
    -o "${archive}"
  printf '%s  %s\n' "${EFA_INSTALLER_SHA256}" "${archive}" | sha256sum -c -
  tar -xzf "${archive}" -C "${workdir}"

  cd "${workdir}/aws-efa-installer"
  ./efa_installer.sh -y --skip-kmod --skip-limit-conf
  check_efa_userspace
}

case "${1:-check}" in
  check)
    check_efa_userspace
    ;;
  install)
    install_efa_userspace
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
