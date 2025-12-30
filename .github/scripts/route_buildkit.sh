#!/bin/bash
set -e

# --- ARGUMENT PARSING ---
ARCH=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --amd)
      ARCH="amd64"
      shift
      ;;
    --arm)
      ARCH="arm64"
      shift
      ;;
    *)
      echo "❌ Error: Unknown argument '$1'. Use --amd or --arm."
      exit 1
      ;;
  esac
done

if [ -z "$ARCH" ]; then
  echo "❌ Error: Must specify --amd or --arm."
  exit 1
fi

# --- CONFIGURATION ---
SERVICE_NAME="buildkit-${ARCH}-headless"
NAMESPACE="buildkit"
POD_PREFIX="buildkit-${ARCH}"
PORT="1234"
FLAVORS=("vllm" "trtllm" "sglang" "general")
# ---------------------

echo "🔍 Discovering Buildkit pods via DNS..."

if ! command -v nslookup &> /dev/null; then
    echo "❌ Error: nslookup not found. Please install dnsutils or bind-tools."
    exit 1
fi

# Count active IPs for the headless service
IP_COUNT=$(nslookup ${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local | grep -i "Address" | grep -v "#53" | wc -l)
COUNT=$((IP_COUNT))

if [ "$COUNT" -eq "0" ]; then
  echo "⚠️ Warning: DNS returned 0 records. KEDA might be warming up."
  # Trigger KEDA scale-up by hitting the endpoint (fire and forget)
  curl -s --max-time 2 "http://${POD_PREFIX}-0.${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}" &>/dev/null &
  for flavor in "${FLAVORS[@]}"; do
    echo "${flavor}_${ARCH}=tcp://${POD_PREFIX}-0.${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:${PORT}" >> "$GITHUB_OUTPUT"
  done
  exit 0
fi

echo "✅ Found $COUNT active pod(s) in service $SERVICE_NAME."

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

  echo "   -> Routing $flavor to pod indices: ${TARGET_INDICES[*]}"

  # Write to GitHub Output
  echo "${flavor}_${ARCH}=$ADDRS" >> "$GITHUB_OUTPUT"
done