#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -o errexit
set -o nounset
set -o pipefail

GO_CMD=${1:-go}
CURRENT_DIR=$(dirname "${BASH_SOURCE[0]}")
OPERATOR_ROOT=$(realpath "${CURRENT_DIR}/../../..")
CODEGEN_PKG=$(cd "${OPERATOR_ROOT}"; "${GO_CMD}" list -m -mod=readonly -f "{{.Dir}}" k8s.io/code-generator)

# Keep generated-tool binaries inside the Operator workspace, matching the
# rest of the Makefile-managed tooling.
export GOBIN=${GOBIN:-"${OPERATOR_ROOT}/bin"}

cd "${OPERATOR_ROOT}"

# shellcheck source=/dev/null
source "${CODEGEN_PKG}/kube_codegen.sh"

kube::codegen::gen_helpers \
  --boilerplate "${OPERATOR_ROOT}/hack/boilerplate.go.txt_" \
  "${OPERATOR_ROOT}/api"
