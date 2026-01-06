#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# route_buildkit.sh - Discover and route BuildKit pods for CI builds
# =============================================================================
#
# DESCRIPTION:
#   Discovers active BuildKit pods via Kubernetes DNS and assigns them to
#   different framework flavors (vllm, trtllm, sglang, general) using a
#   modulo-based routing strategy. Outputs are written to GITHUB_OUTPUT for
#   use in GitHub Actions workflows.
#
# USAGE:
#   ./route_buildkit.sh --arch amd64 --flavor all    # Route all flavors for AMD64
#   ./route_buildkit.sh --arch arm64 --flavor vllm   # Route vllm for ARM64
#   ./route_buildkit.sh --arch all --flavor all      # Route all flavors for both architectures
#
# ARGUMENTS:
#   --arch <arch>     Target architecture: amd64, arm64, or all
#   --flavor <flavor> Target flavor: vllm, trtllm, sglang, general, or all
#
# ENVIRONMENT VARIABLES:
#   MAX_RETRIES   Max attempts to wait for pods (default: 8)
#   RETRY_DELAY   Seconds between retry attempts (default: 30)
#
# OUTPUTS (written to GITHUB_OUTPUT):
#   <flavor>_<arch>=tcp://<pod>.<svc>.<ns>.svc.cluster.local:<port>[,...]
#
#   Examples:
#     vllm_amd64=tcp://buildkit-amd64-0.buildkit-amd64-headless.buildkit.svc.cluster.local:1234
#     trtllm_arm64=tcp://buildkit-arm64-1.buildkit-arm64-headless.buildkit.svc.cluster.local:1234
#
# ROUTING STRATEGY:
#   Pods are assigned to flavors based on pod index modulo 3:
#     - vllm:    pod_index % 3 == 0  (pods 0, 3, 6, ...)
#     - trtllm:  pod_index % 3 == 1  (pods 1, 4, 7, ...)
#     - sglang:  pod_index % 3 == 2  (pods 2, 5, 8, ...)
#     - general: pod_index % 3 == 2  (same as sglang)
#
#   If no pods match a flavor's modulo, all available pods are used as fallback.
#
# REQUIREMENTS:
#   - nslookup (from dnsutils or bind-tools)
#   - Access to Kubernetes DNS (run inside cluster)
#   - GITHUB_OUTPUT environment variable set (GitHub Actions)
#
# EXAMPLES:
#   # In GitHub Actions workflow:
#   - name: Route Buildkit Workers
#     run: |
#       .github/scripts/route_buildkit.sh --arch amd64 --flavor all
#       .github/scripts/route_buildkit.sh --arch arm64 --flavor all
#
#   # Route specific flavor for specific arch:
#   - name: Route vllm for AMD64
#     run: .github/scripts/route_buildkit.sh --arch amd64 --flavor vllm
#
#   # Route all flavors for all architectures:
#   - name: Route all
#     run: .github/scripts/route_buildkit.sh --arch all --flavor all
#
#   # Then use outputs:
#   buildkit_worker_addresses: ${{ steps.route.outputs.vllm_amd64 }}
#
# =============================================================================

set -e

# --- ARGUMENT PARSING ---
ARCH_INPUT=""
FLAVOR_INPUT=""
ALL_FLAVORS=("vllm" "trtllm" "sglang" "general")

while [[ $# -gt 0 ]]; do
  case $1 in
    --arch)
      ARCH_INPUT="$2"
      shift 2
      ;;
    --flavor)
      FLAVOR_INPUT="$2"
      shift 2
      ;;
    *)
      echo "❌ Error: Unknown argument '$1'. Use --arch <amd64|arm64|all> --flavor <vllm|trtllm|sglang|general|all>."
      exit 1
      ;;
  esac
done

if [ -z "$ARCH_INPUT" ]; then
  echo "❌ Error: Must specify --arch <amd64|arm64|all>."
  exit 1
fi

if [ -z "$FLAVOR_INPUT" ]; then
  echo "❌ Error: Must specify --flavor <vllm|trtllm|sglang|general|all>."
  exit 1
fi

# Validate arch input
case $ARCH_INPUT in
  amd64|arm64|all) ;;
  *)
    echo "❌ Error: Invalid arch '$ARCH_INPUT'. Must be amd64, arm64, or all."
    exit 1
    ;;
esac

# Validate flavor input
case $FLAVOR_INPUT in
  vllm|trtllm|sglang|general|all) ;;
  *)
    echo "❌ Error: Invalid flavor '$FLAVOR_INPUT'. Must be vllm, trtllm, sglang, general, or all."
    exit 1
    ;;
esac

# Determine architectures to process
if [ "$ARCH_INPUT" = "all" ]; then
  ARCHS=("amd64" "arm64")
else
  ARCHS=("$ARCH_INPUT")
fi

# Determine flavors to process
if [ "$FLAVOR_INPUT" = "all" ]; then
  FLAVORS=("${ALL_FLAVORS[@]}")
else
  FLAVORS=("$FLAVOR_INPUT")
fi

# --- CONFIGURATION ---
NAMESPACE="buildkit"
PORT="1234"
# ---------------------

if ! command -v nslookup &> /dev/null; then
    echo "❌ Error: nslookup not found. Please install dnsutils or bind-tools."
    exit 1
fi

# --- RETRY CONFIGURATION ---
MAX_RETRIES=${MAX_RETRIES:-8}
RETRY_DELAY=${RETRY_DELAY:-30}
# ---------------------------

# Function to count active IPs for a headless service
get_pod_count() {
  local service_name=$1
  local ip_count
  ip_count=$(nslookup ${service_name}.${NAMESPACE}.svc.cluster.local 2>/dev/null | grep -i "Address" | grep -v "#53" | wc -l)
  echo $((ip_count))
}

# Function to get all pod indices for a flavor based on Modulo 3
get_target_indices() {
  local flavor=$1
  local count=$2

  # Target remainder:
  # vllm   -> % 3 == 0
  # trtllm -> % 3 == 1
  # sglang -> % 3 == 2 (and others)
  local target_mod
  case $flavor in
    vllm)           target_mod=0 ;;
    trtllm)         target_mod=1 ;;
    sglang|general)  target_mod=2 ;;
    *)              target_mod=2 ;; # Default others to 2
  esac

  # Find all valid indices [0 ... count-1] that match the modulo
  local candidates=()
  for (( i=0; i<count; i++ )); do
    if [ $(( i % 3 )) -eq "$target_mod" ]; then
      candidates+=("$i")
    fi
  done

  # If no pods match the specific modulo, fallback to all pods
  if [ "${#candidates[@]}" -eq "0" ]; then
    for (( i=0; i<count; i++ )); do
      candidates+=("$i")
    done
  fi

  echo "${candidates[@]}"
}

# Process each architecture
for ARCH in "${ARCHS[@]}"; do
  SERVICE_NAME="buildkit-${ARCH}-headless"
  POD_PREFIX="buildkit-${ARCH}"

  echo "🔍 Discovering Buildkit pods for ${ARCH} via DNS..."

  # Initial count
  COUNT=$(get_pod_count "$SERVICE_NAME")

  # Retry loop if no pods found
  if [ "$COUNT" -eq "0" ]; then
    echo "⚠️  DNS returned 0 records for ${ARCH}. KEDA should be triggering a new buildkit pod."

    for (( retry=1; retry<=MAX_RETRIES; retry++ )); do
      echo "⏳ Waiting ${RETRY_DELAY}s for BuildKit pods to become available (attempt ${retry}/${MAX_RETRIES})..."
      sleep "$RETRY_DELAY"

      COUNT=$(get_pod_count "$SERVICE_NAME")
      if [ "$COUNT" -gt "0" ]; then
        echo "✅ BuildKit pods for ${ARCH} are now available!"
        break
      fi

      if [ "$retry" -eq "$MAX_RETRIES" ]; then
        echo "❌ Error: No BuildKit pods available for ${ARCH} after ${MAX_RETRIES} attempts ($(( MAX_RETRIES * RETRY_DELAY ))s total)."
        echo "   Please check KEDA scaling configuration and BuildKit deployment status."
        exit 1
      fi
    done
  fi

  echo "✅ Found $COUNT active pod(s) in service $SERVICE_NAME."

  # Iterate over flavors and set outputs
  for flavor in "${FLAVORS[@]}"; do
    TARGET_INDICES=($(get_target_indices "$flavor" "$COUNT"))

    ADDRS=""
    for idx in "${TARGET_INDICES[@]}"; do
      POD_NAME="${POD_PREFIX}-${idx}"
      ADDR="tcp://${POD_NAME}.${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}"
      if [ -z "$ADDRS" ]; then
        ADDRS="$ADDR"
      else
        ADDRS="${ADDRS},${ADDR}"
      fi
    done

    echo "   -> Routing ${flavor}_${ARCH} to pod indices: ${TARGET_INDICES[*]}"

    # Write to GitHub Output
    echo "${flavor}_${ARCH}=$ADDRS" >> "$GITHUB_OUTPUT"
  done
done