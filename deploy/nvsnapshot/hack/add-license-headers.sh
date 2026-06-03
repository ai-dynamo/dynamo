#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Adds (or verifies, with --check) the SPDX license header on in-scope
# source files: .go, .c, .sh, .py.
# Makefile, *.mk, Dockerfile, *.md, *.yaml, *.yml are out of scope.

set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADDLICENSE="${ADDLICENSE:-addlicense}"
BOILERPLATE="${REPO_ROOT}/hack/boilerplate.go.txt"
YEAR="${YEAR:-$(date +%Y)}"

MODE="${1:-add}"
CHECK_ARGS=()
if [[ "${MODE}" == "--check" ]]; then
  CHECK_ARGS=("-check")
fi

TEMPLATE="$(mktemp)"
trap 'rm -f "${TEMPLATE}"' EXIT

# Boilerplate is raw SPDX text (no comment markers); addlicense applies
# language-appropriate comment markers per file extension.
sed -e "s/YEAR/${YEAR}/g" "${BOILERPLATE}" > "${TEMPLATE}"

cd "${REPO_ROOT}"
# The ${arr[@]+"${arr[@]}"} form is bash 3.2-safe under `nounset`
# (plain ${arr[@]} expansion errors on an empty array in macOS bash).
"${ADDLICENSE}" \
  ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"} \
  -f "${TEMPLATE}" \
  -ignore "vendor/**" \
  -ignore "bin/**" \
  -ignore ".idea/**" \
  -ignore ".vscode/**" \
  -ignore "**/*.md" \
  -ignore "**/*.yaml" \
  -ignore "**/*.yml" \
  -ignore "**/Dockerfile" \
  -ignore "**/Makefile" \
  -ignore "**/*.mk" \
  -ignore "**/testdata/**" \
  -ignore "**/*.pb.go" \
  -ignore ".github/**" \
  -ignore "**/zz_generated*.go" \
  .

echo "License headers OK."
