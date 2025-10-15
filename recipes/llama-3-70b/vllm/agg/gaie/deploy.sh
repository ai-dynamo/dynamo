#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env bash
set -Eeuo pipefail

# ===== Config (env overridable) =====
: "${NAMESPACE:=dynamo}"
: "${RELEASE:=dynamo-gaie}"
: "${EPP_IMAGE:?EPP_IMAGE must be set, e.g. nvcr.io/your/epp:tag}"

# Per-recipe values
: "${RECIPE_VALUES_1:=model-gaie.yaml}"
: "${RECIPE_VALUES_2:=values-epp-aware.yaml}"

# ===== Paths =====
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find repo root
if GIT_TOP=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
  REPO_ROOT="$GIT_TOP"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
fi

CHART_DIR="$REPO_ROOT/deploy/inference-gateway/helm/dynamo-gaie"

if [[ ! -d "$CHART_DIR" ]]; then
  echo "ERROR: GAIE chart not found at: $CHART_DIR"
  exit 1
fi

# ===== Pre-flight checks =====
command -v helm >/dev/null 2>&1 || { echo "ERROR: helm not found"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not found"; exit 1; }

# ===== Namespace ensure =====
if ! kubectl get ns "$NAMESPACE" >/dev/null 2>&1; then
  kubectl create namespace "$NAMESPACE"
fi

# ===== Build chart deps (if any) =====
helm dependency build "$CHART_DIR" >/dev/null

# ===== Compose -f args from local files if present =====
VALUES_ARGS=()
if [[ -f "$SCRIPT_DIR/$RECIPE_VALUES_1" ]]; then
  VALUES_ARGS+=(-f "$SCRIPT_DIR/$RECIPE_VALUES_1")
fi
if [[ -f "$SCRIPT_DIR/$RECIPE_VALUES_2" ]]; then
  VALUES_ARGS+=(-f "$SCRIPT_DIR/$RECIPE_VALUES_2")
fi

# Allow caller to add more -f/--set/etc via passthrough args
# Example:
#   ./deploy.sh --set eppAware.extraEnv[0].name=FOO --set eppAware.extraEnv[0].value=bar
EXTRA_ARGS=( "$@" )

# ===== Install/upgrade =====
echo "==> Deploying GAIE chart"
echo "    Release:   $RELEASE"
echo "    Namespace: $NAMESPACE"
echo "    Chart:     $CHART_DIR"
echo "    EPP_IMAGE: $EPP_IMAGE"
helm upgrade --install "$RELEASE" "$CHART_DIR" \
  -n "$NAMESPACE" \
  "${VALUES_ARGS[@]}" \
  --set eppAware.enabled=true \
  --set-string eppAware.eppImage="$EPP_IMAGE" \
  "${EXTRA_ARGS[@]}"

echo "Done."

