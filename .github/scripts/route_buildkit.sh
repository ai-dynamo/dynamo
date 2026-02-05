#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# route_buildkit.sh - Discover and route BuildKit pods for CI builds
# =============================================================================
#
# ROUTING LOGIC:
# --------------
# Routing is optimized for Docker layer caching based on shared base images:
#   - vLLM and SGLang share the same base image (cuda-dl-base) when CUDA versions match
#   - TensorRT-LLM uses a different base (pytorch), so it's isolated
#   - General builds have no framework, grouped with trtllm for isolation
#
# Flavors are routed to BuildKit pods using modulo 3 on the pod index:
#   - Pool 0 (idx % 3 == 0): vllm-cuda12, sglang-cuda12  (share cuda-dl-base + wheel_builder cache)
#   - Pool 1 (idx % 3 == 1): vllm-cuda13, sglang-cuda13  (share cuda-dl-base + wheel_builder cache)
#   - Pool 2 (idx % 3 == 2): trtllm-cuda13, general      (isolated - different/no framework base)
#
# FALLBACK: If no pods match the target pool, the highest available index is used.
#
# EXPECTED ROUTING TABLE (pod indices returned for each flavor):
# +------+-------------+---------------+-------------+---------------+---------------+---------+
# | Pods | vllm-cuda12 | sglang-cuda12 | vllm-cuda13 | sglang-cuda13 | trtllm-cuda13 | general |
# |      | (mod 0)     | (mod 0)       | (mod 1)     | (mod 1)       | (mod 2)       | (mod 2) |
# +------+-------------+---------------+-------------+---------------+---------------+---------+
# |  1   | 0           | 0             | 0 (fb)      | 0 (fb)        | 0 (fb)        | 0 (fb)  |
# |  2   | 0           | 0             | 1           | 1             | 1 (fb)        | 1 (fb)  |
# |  3   | 0           | 0             | 1           | 1             | 2             | 2       |
# |  4   | 0, 3        | 0, 3          | 1           | 1             | 2             | 2       |
# |  5   | 0, 3        | 0, 3          | 1, 4        | 1, 4          | 2             | 2       |
# |  6   | 0, 3        | 0, 3          | 1, 4        | 1, 4          | 2, 5          | 2, 5    |
# +------+-------------+---------------+-------------+---------------+---------------+---------+
# (fb) = fallback - no pods matched target pool, returns max available index
#
# =============================================================================

set -e

# --- ARGUMENT PARSING ---
ARCH_INPUT=""
FLAVOR_INPUT=""
CUDA_VERSION=""
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
    --cuda)
      CUDA_VERSION="$2"
      shift 2
      ;;
    *)
      echo "❌ Error: Unknown argument '$1'. Use --arch <amd64|arm64|all> --flavor <vllm|trtllm|sglang|general|all> [--cuda <12.9|13.0>]."
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

# CUDA version is required for all flavors except "general"
if [ -z "$CUDA_VERSION" ] && [ "$FLAVOR_INPUT" != "general" ]; then
  echo "❌ Error: Must specify --cuda <12.9|13.0> for flavor '$FLAVOR_INPUT'."
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

# Validate CUDA version input (allow empty for general flavor)
if [ -n "$CUDA_VERSION" ]; then
  case $CUDA_VERSION in
    12.9|13.0|13.1) ;;
    *)
      echo "❌ Error: Invalid CUDA version '$CUDA_VERSION'. Must be 12.9, 13.0, or 13.1."
      exit 1
      ;;
  esac
fi

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
MAX_POD_CHECK=10  # How many pod indices to probe (e.g., 0 to 3)
# ---------------------

if ! command -v nslookup &> /dev/null; then
    echo "❌ Error: nslookup not found. Please install dnsutils or bind-tools."
    exit 1
fi

# --- RETRY CONFIGURATION ---
MAX_RETRIES=${MAX_RETRIES:-8}
RETRY_DELAY=${RETRY_DELAY:-30}
# ---------------------------

# Function to discover SPECIFIC active pod indices
# This handles gaps (e.g., if pod-0 and pod-2 are up, but pod-1 is down)
get_active_indices() {
  local arch=$1
  local service_name=$2
  local active_indices=()

  # Loop through theoretical indices to see which ones actually resolve via DNS.
  for (( i=0; i<MAX_POD_CHECK; i++ )); do
    local pod_dns="buildkit-${arch}-${i}.${service_name}.${NAMESPACE}.svc.cluster.local"

    # Check if this specific pod resolves
    if nslookup "$pod_dns" >/dev/null 2>&1; then
      active_indices+=("$i")
    fi
  done

  echo "${active_indices[@]}"
}

# Function to route flavors to specific active indices based on Modulo 3
get_target_indices() {
  local flavor=$1
  local cuda_version=$2
  # Read remaining arguments as an array of available indices
  local -a available_indices=("${@:3}")

  if [ ${#available_indices[@]} -eq 0 ]; then
    echo ""
    return
  fi

  local cuda_major=${cuda_version%%.*}
  local route_key="${flavor}-cuda${cuda_major}"
  local target_mod

  case "$route_key" in
    # --- POOL 0: CUDA 12 builds (vLLM + SGLang share cuda-dl-base:cuda12.9) ---
    vllm-cuda12|sglang-cuda12)
      target_mod=0
      ;;
    # --- POOL 1: CUDA 13 builds (vLLM + SGLang share cuda-dl-base:cuda13.0) ---
    vllm-cuda13|sglang-cuda13)
      target_mod=1
      ;;
    # --- POOL 2: Isolated builds (TensorRT-LLM uses pytorch base, general has no framework) ---
    trtllm-cuda13|general-*)
      target_mod=2
      ;;
    # --- FALLBACK ---
    *)
      target_mod=2
      ;;
  esac

  echo "    [DEBUG] Routing Key: '$route_key' -> Worker Index Modulo: $target_mod" >&2

  local final_targets=()

  # Filter the AVAILABLE indices (not just 0..count)
  for idx in "${available_indices[@]}"; do
    if [ $(( idx % 3 )) -eq "$target_mod" ]; then
      final_targets+=("$idx")
    fi
  done

  # If no pods match the specific modulo, fallback to the highest available index
  if [ "${#final_targets[@]}" -eq "0" ]; then
    local max_idx=${available_indices[0]}
    for idx in "${available_indices[@]}"; do
      if [ "$idx" -gt "$max_idx" ]; then
        max_idx=$idx
      fi
    done
    echo "$max_idx"
  else
    echo "${final_targets[@]}"
  fi
}

# Process each architecture
for ARCH in "${ARCHS[@]}"; do
  SERVICE_NAME="buildkit-${ARCH}-headless"
  POD_PREFIX="buildkit-${ARCH}"

  echo "🔍 Discovering active Buildkit pods for ${ARCH} via DNS (checking indices 0-$((MAX_POD_CHECK-1)))..."

  # Get the actual list of alive indices (e.g., "0 2 5")
  ACTIVE_INDICES=($(get_active_indices "$ARCH" "$SERVICE_NAME"))
  COUNT=${#ACTIVE_INDICES[@]}

  # Retry loop if no pods found
  if [ "$COUNT" -eq "0" ]; then
    echo "⚠️  DNS returned 0 records for ${ARCH}. KEDA should be triggering a new buildkit pod."

    for (( retry=1; retry<=MAX_RETRIES; retry++ )); do
      echo "⏳ Waiting ${RETRY_DELAY}s for BuildKit pods to become available (attempt ${retry}/${MAX_RETRIES})..."
      sleep "$RETRY_DELAY"

      # Re-probe for active indices
      ACTIVE_INDICES=($(get_active_indices "$ARCH" "$SERVICE_NAME"))
      COUNT=${#ACTIVE_INDICES[@]}

      if [ "$COUNT" -gt "0" ]; then
        echo "✅ BuildKit pods for ${ARCH} are now available!"
        break
      fi

      if [ "$retry" -eq "$MAX_RETRIES" ]; then
        echo "::warning::No remote BuildKit pods available for ${ARCH} after ${MAX_RETRIES} attempts. Falling back to Kubernetes driver."
        echo "⚠️  Warning: No remote BuildKit pods available for ${ARCH}."

        for flavor in "${FLAVORS[@]}"; do
          echo "${flavor}_${ARCH}=" >> "$GITHUB_OUTPUT"
        done
        exit 1
      fi
    done
  fi

  echo "✅ Found $COUNT active pod(s) (Indices: ${ACTIVE_INDICES[*]})."

  # Iterate over flavors and set outputs
  for flavor in "${FLAVORS[@]}"; do
    # Pass the discovered ACTIVE_INDICES to the routing function
    TARGET_INDICES=($(get_target_indices "$flavor" "$CUDA_VERSION" "${ACTIVE_INDICES[@]}"))

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

    echo "    -> Routing ${flavor}_${ARCH} to pod indices: ${TARGET_INDICES[*]}"

    # Write to GitHub Output
    echo "${flavor}_${ARCH}=$ADDRS" >> "$GITHUB_OUTPUT"
  done
done