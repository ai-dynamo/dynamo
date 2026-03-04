#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# =============================================================================
# route_buildkit.sh - Discover and route BuildKit pods for CI builds
# =============================================================================
#
# ROUTING LOGIC (Coverage-Aware Ranked Rendezvous Hashing with SHA-256):
# ---------------------------------------------------------
# Routing is optimized for Docker layer caching, linear scaling, and
# 100% pod utilization across any number of BuildKit pods.
#
# CACHE GROUPS (3 distinct groups to maximize layer reuse):
#   - Group 0 (cuda-dl-base-13):    vLLM & SGLang (CUDA 13.x)
#   - Group 1 (cuda-dl-base-12):    vLLM & SGLang (CUDA 12.x)
#   - Group 2 (general-trt-combined): TRT-LLM & General Builds
#
# ALGORITHM:
# 1. SCORING: Each group key is hashed with every active pod index (SHA-256)
#    to produce a uniformly distributed score per (group, arch, pod) triple.
# 2. RANKING: Pods are sorted by score (descending) per group. StatefulSet
#    pod names are constant, so rankings are stable across invocations.
# 3. POOL SIZING: Pool Size = ceil(Active Pods / 3) ensures even distribution.
# 4. COVERAGE-AWARE SELECTION: Pools are built round-by-round across all 3
#    groups simultaneously. In each round, each group picks its highest-ranked
#    pod that is NOT YET in any group's pool (preferring uncovered pods).
#    This guarantees every active pod appears in at least one group's pool.
# 5. RANDOM PICK: ONE pod is randomly selected from the candidate pool.
#
# LOAD DISTRIBUTION (cksum-based, all pods utilized):
# +------+------+-------------------+-------------------+---------------------+
# | Pods | Pool | G0: vLLM/SGL C13  | G1: vLLM/SGL C12  | G2: TRT-LLM/General |
# +------+------+-------------------+-------------------+---------------------+
# |  1   |  1   | {0}               | {0}               | {0}                 |
# |  2   |  1   | {0}               | {1}               | {1}                 |
# |  3   |  1   | {0}               | {2}               | {1}                 |
# |  4   |  2   | {0, 3}            | {2, 1}            | {1, 2}             |
# |  5   |  2   | {0, 3}            | {2, 4}            | {1, 2}             |
# |  6   |  2   | {0, 3}            | {5, 1}            | {2, 4}             |
# |  7   |  3   | {0, 3, 4}         | {5, 1, 2}         | {2, 6, 5}          |
# |  8   |  3   | {7, 0, 3}         | {5, 1, 4}         | {2, 6, 5}          |
# |  9   |  3   | {7, 0, 3}         | {8, 5, 1}         | {2, 6, 4}          |
# +------+------+-------------------+-------------------+---------------------+
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
MAX_POD_CHECK=10
# ---------------------

if ! command -v nslookup &> /dev/null; then
    echo "❌ Error: nslookup not found. Please install dnsutils or bind-tools."
    exit 1
fi

if ! command -v sha256sum &> /dev/null; then
    echo "❌ Error: sha256sum not found. Please install coreutils."
    exit 1
fi

# --- RETRY CONFIGURATION ---
MAX_RETRIES=${MAX_RETRIES:-2}
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

GROUP_KEYS=("cuda-dl-base-13" "cuda-dl-base-12" "general-trt-combined")

# Map a flavor + CUDA version to a group index (0, 1, or 2)
flavor_to_group() {
  local flavor=$1
  local cuda_major=${2%%.*}
  case "$flavor" in
    vllm|sglang)
      case "$cuda_major" in
        13) echo 0 ;;
        *)  echo 1 ;;
      esac
      ;;
    trtllm|general|*) echo 2 ;;
  esac
}

# Compute coverage-aware pool assignments for all 3 groups.
# Outputs pipe-separated pools: "pool0|pool1|pool2"
compute_group_pools() {
  local arch=$1
  local -a available_indices=("${@:2}")
  local count=${#available_indices[@]}

  if [ "$count" -eq 0 ]; then
    echo "||"
    return
  fi

  local pool_size=$(( (count + 2) / 3 ))

  local rank0="" rank1="" rank2=""
  for g in 0 1 2; do
    local scored_list=()
    for idx in "${available_indices[@]}"; do
      local combo="${GROUP_KEYS[$g]}-buildkit-${arch}-${idx}"
      local score=$(echo -n "$combo" | sha256sum | awk '{print $1}')
      scored_list+=("${score}:${idx}")
    done
    local sorted_str=$(printf "%s\n" "${scored_list[@]}" | sort -r | cut -d':' -f2 | tr '\n' ' ')
    if [ "$g" -eq 0 ]; then rank0="$sorted_str"; fi
    if [ "$g" -eq 1 ]; then rank1="$sorted_str"; fi
    if [ "$g" -eq 2 ]; then rank2="$sorted_str"; fi
  done

  local pool0=" " pool1=" " pool2=" "
  local covered=" "

  for (( round=0; round<pool_size; round++ )); do
    for g in 0 1 2; do
      local current_rank="" current_pool=""
      if [ "$g" -eq 0 ]; then current_rank="$rank0"; current_pool="$pool0"; fi
      if [ "$g" -eq 1 ]; then current_rank="$rank1"; current_pool="$pool1"; fi
      if [ "$g" -eq 2 ]; then current_rank="$rank2"; current_pool="$pool2"; fi

      local picked=""
      for candidate in $current_rank; do
        [[ "$current_pool" == *" $candidate "* ]] && continue
        if [[ "$covered" != *" $candidate "* ]]; then
          picked=$candidate; break
        fi
      done
      if [ -z "$picked" ]; then
        for candidate in $current_rank; do
          [[ "$current_pool" == *" $candidate "* ]] && continue
          picked=$candidate; break
        done
      fi

      if [ -n "$picked" ]; then
        current_pool="${current_pool}${picked} "
        covered="${covered}${picked} "
        if [ "$g" -eq 0 ]; then pool0="$current_pool"; fi
        if [ "$g" -eq 1 ]; then pool1="$current_pool"; fi
        if [ "$g" -eq 2 ]; then pool2="$current_pool"; fi
      fi
    done
  done

  pool0=$(echo "$pool0" | xargs)
  pool1=$(echo "$pool1" | xargs)
  pool2=$(echo "$pool2" | xargs)
  echo "${pool0}|${pool1}|${pool2}"
}

# Route a flavor to its group's pre-computed pool.
get_target_indices() {
  local flavor=$1
  local cuda_version=$2
  local arch=$3
  local -a available_indices=("${@:4}")

  if [ ${#available_indices[@]} -eq 0 ]; then
    echo ""
    return
  fi

  local group=$(flavor_to_group "$flavor" "$cuda_version")
  local cuda_major=${cuda_version%%.*}
  echo "    [DEBUG] Routing Key: '$flavor-cuda$cuda_major' -> Group: $group (${GROUP_KEYS[$group]})" >&2

  local all_pools=$(compute_group_pools "$arch" "${available_indices[@]}")
  echo "$all_pools" | cut -d'|' -f$((group + 1))
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
    # Pass the discovered ACTIVE_INDICES to the routing function to get the candidate pool
    TARGET_INDICES=($(get_target_indices "$flavor" "$CUDA_VERSION" "$ARCH" "${ACTIVE_INDICES[@]}"))

    ADDRS=""
    # 2. Get the number of elements in the candidate pool array
    TARGET_INDICES_LENGTH=${#TARGET_INDICES[@]}

    # 3. Generate a random index between 0 and length-1
    # The $RANDOM variable provides a number between 0 and 32767.
    RANDOM_INDEX=$(($RANDOM % $TARGET_INDICES_LENGTH))
    RANDOM_VALUE="${TARGET_INDICES[$RANDOM_INDEX]}"
    POD_NAME="${POD_PREFIX}-${RANDOM_VALUE}"
    ADDRS="tcp://${POD_NAME}.${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}"

    echo "    -> Routing ${flavor}_${ARCH} to Candidate Pool: {${TARGET_INDICES[*]}} | Selected: ${RANDOM_VALUE}"

    # Write to GitHub Output
    echo "${flavor}_${ARCH}=$ADDRS" >> "$GITHUB_OUTPUT"
  done
done