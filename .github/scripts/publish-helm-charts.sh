#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Packages and pushes Dynamo Helm charts to an OCI registry (NGC).
#
# Required env:
#   CHART_VERSION   - Version string for chart packaging (e.g., 0.9.0rc0)
#   NGC_HELM_REPO   - OCI registry URL (e.g., oci://nvcr.io/nvidia/ai-dynamo)
#
# Optional env:
#   DRY_RUN=1       - Package charts but skip the push step
#   GITHUB_OUTPUT   - If set, writes helm_success_count / helm_failed_count
#
# Usage:
#   .github/scripts/publish-helm-charts.sh
#   DRY_RUN=1 .github/scripts/publish-helm-charts.sh
#
# Exit codes:
#   0  At least one chart published (or dry-run completed)
#   1  Missing env vars or all charts failed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG_DIR="/tmp/helm-packages"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------

if [ -z "${CHART_VERSION:-}" ]; then
  echo "ERROR: CHART_VERSION env var is required" >&2
  exit 1
fi

if [ -z "${NGC_HELM_REPO:-}" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "ERROR: NGC_HELM_REPO env var is required (or set DRY_RUN=1)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

HELM_SUCCESS=()
HELM_FAILED=()

helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
mkdir -p "${PKG_DIR}"

# ---------------------------------------------------------------------------
# publish_chart <chart_dir> <chart_name>
#
# Package and push a chart with no sub-chart dependencies.
# Skips gracefully if the chart directory doesn't exist.
# ---------------------------------------------------------------------------
publish_chart() {
  local chart_dir="$1"
  local chart_name="$2"

  if [ ! -d "${chart_dir}" ]; then
    echo "Skipping ${chart_name}: ${chart_dir} not found"
    return 0
  fi

  echo "Publishing ${chart_name}:${CHART_VERSION} ..."
  helm package "${chart_dir}" \
    --version "${CHART_VERSION}" \
    --app-version "${CHART_VERSION}" \
    --destination "${PKG_DIR}"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] Would push ${chart_name}-${CHART_VERSION}.tgz to ${NGC_HELM_REPO:-<unset>}"
    HELM_SUCCESS+=("${chart_name}:${CHART_VERSION}")
    return 0
  fi

  if helm push "${PKG_DIR}/${chart_name}-${CHART_VERSION}.tgz" "${NGC_HELM_REPO}"; then
    HELM_SUCCESS+=("${chart_name}:${CHART_VERSION}")
  else
    HELM_FAILED+=("${chart_name}:${CHART_VERSION}")
  fi
}

# ---------------------------------------------------------------------------
# publish_platform <chart_dir>
#
# dynamo-platform requires special handling:
#   - Operator sub-chart version must match the parent dependency spec
#   - External dependencies (bitnami, nats) require helm dep build
# ---------------------------------------------------------------------------
publish_platform() {
  local platform_dir="$1"

  echo "Publishing dynamo-platform:${CHART_VERSION} ..."

  # Sync operator sub-chart version so helm dep build resolves the file:// dependency
  yq -i ".version = \"${CHART_VERSION}\"" "${platform_dir}/components/operator/Chart.yaml"
  yq -i ".appVersion = \"${CHART_VERSION}\"" "${platform_dir}/components/operator/Chart.yaml"
  yq -i "(.dependencies[] | select(.name == \"dynamo-operator\")).version = \"${CHART_VERSION}\"" "${platform_dir}/Chart.yaml"

  pushd "${platform_dir}" > /dev/null
  helm dep build .
  popd > /dev/null

  helm package "${platform_dir}" \
    --version "${CHART_VERSION}" \
    --app-version "${CHART_VERSION}" \
    --destination "${PKG_DIR}"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] Would push dynamo-platform-${CHART_VERSION}.tgz to ${NGC_HELM_REPO:-<unset>}"
    HELM_SUCCESS+=("dynamo-platform:${CHART_VERSION}")
    return 0
  fi

  if helm push "${PKG_DIR}/dynamo-platform-${CHART_VERSION}.tgz" "${NGC_HELM_REPO}"; then
    HELM_SUCCESS+=("dynamo-platform:${CHART_VERSION}")
  else
    HELM_FAILED+=("dynamo-platform:${CHART_VERSION}")
  fi
}

# ---------------------------------------------------------------------------
# Charts to publish
# To add a new simple chart, add a publish_chart line below.
# ---------------------------------------------------------------------------

publish_chart   "${REPO_ROOT}/deploy/helm/charts/crds"     dynamo-crds
publish_platform "${REPO_ROOT}/deploy/helm/charts/platform"
publish_chart   "${REPO_ROOT}/deploy/helm/charts/chrek"    chrek

# ---------------------------------------------------------------------------
# Summary & outputs
# ---------------------------------------------------------------------------

echo "Helm publish summary: ${#HELM_SUCCESS[@]} succeeded, ${#HELM_FAILED[@]} failed"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "helm_success_count=${#HELM_SUCCESS[@]}" >> "$GITHUB_OUTPUT"
  echo "helm_failed_count=${#HELM_FAILED[@]}" >> "$GITHUB_OUTPUT"
fi

if [ ${#HELM_SUCCESS[@]} -eq 0 ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "ERROR: No charts were successfully published" >&2
  exit 1
fi
