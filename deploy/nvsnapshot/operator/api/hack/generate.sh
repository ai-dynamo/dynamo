#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Generates deepcopy methods and CRD manifests for the operator/api
# module. Runs controller-gen only (no kube_codegen.sh) — clientsets
# and informers are not produced by this script.
#
# Caller may override controller-gen via the CONTROLLER_GEN env var
# (defaulting to whatever is on PATH). Module Makefile passes the
# version-pinned binary from $LOCALBIN.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONTROLLER_GEN="${CONTROLLER_GEN:-controller-gen}"
BOILERPLATE="${REPO_ROOT}/hack/boilerplate.generatego.txt"

if ! command -v "${CONTROLLER_GEN}" >/dev/null 2>&1 && [[ ! -x "${CONTROLLER_GEN}" ]]; then
  echo >&2 "controller-gen not found at: ${CONTROLLER_GEN}"
  echo >&2 "run 'make controller-gen' from the repo root to install it"
  exit 1
fi

cd "${MODULE_DIR}"

echo "> Generating deepcopy methods..."
"${CONTROLLER_GEN}" \
  object:headerFile="${BOILERPLATE}" \
  paths="./v1alpha1/..."

echo "> Generating CRD manifests..."
"${CONTROLLER_GEN}" \
  crd \
  paths="./v1alpha1/..." \
  output:crd:dir="./v1alpha1/crds"

echo "> done."
