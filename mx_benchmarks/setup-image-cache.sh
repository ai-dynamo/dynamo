#!/usr/bin/env bash
# Finds node(s) with all GPUs available for a given role (source or target),
# labels them, and deploys the image cache DaemonSet.
#
# If a pod gets evicted during rollout (e.g., DiskPressure), the bad node is
# automatically un-labeled and replaced with a healthy one.
#
# Usage:
#   ./setup-image-cache.sh source     # Find and label 1 source node
#   ./setup-image-cache.sh target     # Find and label 1 target node (excludes source)
#   ./setup-image-cache.sh target 3   # Find and label 3 target nodes
set -euo pipefail

ROLE="${1:-}"
COUNT="${2:-1}"
if [[ "$ROLE" != "source" && "$ROLE" != "target" ]]; then
  echo "Usage: $0 {source|target} [count]"
  exit 1
fi

NAMESPACE="${NAMESPACE:-hwoo}"
RUN_ID="${RUN_ID:-default}"
LABEL_KEY="dynamo/${RUN_ID}-image-cache"
LABEL_VALUE="true"
ROLE_LABEL="dynamo/${RUN_ID}-role"
_TP=$(( ${TP_SIZE:-8} > 0 ? ${TP_SIZE:-8} : 1 ))
_DP=$(( ${DP_SIZE:-1} > 0 ? ${DP_SIZE:-1} : 1 ))
TOTAL_GPUS=$(( _TP * _DP ))
RDMA_RESOURCE="${RDMA_RESOURCE:-rdma/ib}"
RDMA_COUNT="${RDMA_COUNT:-$TOTAL_GPUS}"
REQUIRED_GPUS="$TOTAL_GPUS"
REQUIRED_RDMA="$RDMA_COUNT"
EXCLUDED_NODES="${EXCLUDED_NODES:-}"  # space-separated nodes to never select (e.g., known-bad nodes)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMONSET_FILE="${SCRIPT_DIR}/image-cache-daemonset.yaml"

if [[ ! -f "$DAEMONSET_FILE" ]]; then
  echo "Error: $DAEMONSET_FILE not found"
  exit 1
fi

# Collect nodes to exclude: cordoned nodes + this RUN_ID's labeled nodes + nodes tainted by other runs
get_excluded_nodes() {
  local cordoned labeled tainted
  # Exclude unschedulable (cordoned/drained) nodes
  cordoned=$(kubectl get nodes --field-selector spec.unschedulable=true -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
  labeled=$(kubectl get nodes -l "$ROLE_LABEL" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
  # Exclude nodes with any dynamo/*-reserved taint (from any RUN_ID)
  tainted=$(kubectl get nodes -o json 2>/dev/null \
    | python3 -c 'import json,sys; data=json.load(sys.stdin);
nodes=[n["metadata"]["name"] for n in data.get("items",[]) for t in n.get("spec",{}).get("taints",[]) if "dynamo/" in t.get("key","") and "reserved" in t.get("key","")]
print(" ".join(set(nodes)))' 2>/dev/null || true)
  echo "$cordoned $labeled $tainted"
}

get_free_resources() {
  local node=$1
  local gpu_cap=$2

  local gpu_alloc rdma_cap rdma_alloc

  # Count resources from all non-terminated pods (Running + Pending both hold reservations)
  gpu_alloc=$(kubectl get pods --all-namespaces --field-selector="spec.nodeName=$node,status.phase!=Succeeded,status.phase!=Failed" \
    -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.requests.nvidia\.com/gpu}{"\n"}{end}{end}' 2>/dev/null \
    | awk '{s+=$1} END {print s+0}')

  rdma_cap=$(kubectl get node "$node" -o jsonpath="{.status.capacity.${RDMA_RESOURCE}}" 2>/dev/null)
  rdma_cap=${rdma_cap:-0}

  rdma_alloc=$(kubectl get pods --all-namespaces --field-selector="spec.nodeName=$node,status.phase!=Succeeded,status.phase!=Failed" \
    -o jsonpath="{range .items[*]}{range .spec.containers[*]}{.resources.requests.${RDMA_RESOURCE}}{\"\\n\"}{end}{end}" 2>/dev/null \
    | awk '{s+=$1} END {print s+0}')

  echo "$((gpu_cap - gpu_alloc)) $((rdma_cap - rdma_alloc))"
}

find_node() {
  local excluded="$1"
  local gpu_nodes
  gpu_nodes=$(kubectl get nodes -o jsonpath='{range .items[?(@.status.capacity.nvidia\.com/gpu)]}{.metadata.name} {.status.capacity.nvidia\.com/gpu}{"\n"}{end}')

  selected_node=""

  # Try GPU + RDMA first
  while read -r node capacity; do
    [[ -z "$node" ]] && continue
    [[ "$capacity" -lt "$REQUIRED_GPUS" ]] && continue
    # Skip already-labeled nodes
    if [[ " $excluded " == *" $node "* ]]; then
      continue
    fi

    read -r gpu_free rdma_free <<< "$(get_free_resources "$node" "$capacity")"

    if [[ "$gpu_free" -ge "$REQUIRED_GPUS" ]] && [[ "$rdma_free" -ge "$REQUIRED_RDMA" ]]; then
      selected_node="$node"
      return 0
    fi
  done <<< "$gpu_nodes"

  # Fallback: GPU-only
  while read -r node capacity; do
    [[ -z "$node" ]] && continue
    [[ "$capacity" -lt "$REQUIRED_GPUS" ]] && continue
    if [[ " $excluded " == *" $node "* ]]; then
      continue
    fi

    read -r gpu_free _ <<< "$(get_free_resources "$node" "$capacity")"
    if [[ "$gpu_free" -ge "$REQUIRED_GPUS" ]]; then
      selected_node="$node"
      return 0
    fi
  done <<< "$gpu_nodes"

  return 1
}

excluded_nodes="$(get_excluded_nodes) $EXCLUDED_NODES"

echo "Finding $COUNT $ROLE node(s) with $REQUIRED_GPUS free GPUs each..."
if [[ -n "${excluded_nodes// /}" ]]; then
  echo "  Excluding nodes: $excluded_nodes"
fi

found_nodes=()
for (( i=1; i<=COUNT; i++ )); do
  wait_elapsed=0
  while ! find_node "$excluded_nodes"; do
    sleep 30
    wait_elapsed=$((wait_elapsed + 30))
    if (( wait_elapsed % 60 == 0 )); then
      echo "  waiting for node $i/$COUNT... (${wait_elapsed}s)"
    fi
  done
  found_nodes+=("$selected_node")
  excluded_nodes="$excluded_nodes $selected_node"
  echo "  [$i/$COUNT] Found: $selected_node"
done

echo ""
echo "Selected $ROLE node(s): ${found_nodes[*]}"

# Remove stale labels for this role only
echo ""
echo "Removing stale $ROLE labels..."
stale_nodes=$(kubectl get nodes -l "${ROLE_LABEL}=${ROLE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
for node in $stale_nodes; do
  kubectl label node "$node" "$LABEL_KEY-" "$ROLE_LABEL-" 2>/dev/null || true
done

echo ""
echo "Labeling $ROLE node(s)..."
for node in "${found_nodes[@]}"; do
  kubectl label node "$node" "$LABEL_KEY=$LABEL_VALUE" "$ROLE_LABEL=$ROLE"
  echo "  $node -> $ROLE_LABEL=$ROLE"
done

echo ""
DAEMONSET_NAME="image-cache-${RUN_ID}"
echo "Deploying image cache DaemonSet ($DAEMONSET_NAME) to namespace $NAMESPACE..."
DYNAMO_IMAGE_TAG="${DYNAMO_IMAGE_TAG:-dynamo-mx-runtime}"
sed -e "s|name: dynamo-image-cache|name: ${DAEMONSET_NAME}|g" \
    -e "s|app: dynamo-image-cache|app: ${DAEMONSET_NAME}|g" \
    -e "s|dynamo/image-cache|${LABEL_KEY}|g" \
    -e "s|image: \(.*\):.*|image: \1:${DYNAMO_IMAGE_TAG}|g" \
    "$DAEMONSET_FILE" | kubectl apply -n "$NAMESPACE" -f -

echo ""
echo "Waiting for DaemonSet pods to be ready..."

# Poll for readiness, checking for evicted pods every cycle.
# If a pod gets evicted (e.g., DiskPressure), un-label the bad node,
# find a replacement, re-label, and continue waiting.
MAX_RECOVER_ATTEMPTS=3
recover_attempt=0
evicted_nodes=""  # track nodes evicted during rollout so we never pick them again
wait_elapsed=0
timeout=600
while true; do
  sleep 10
  wait_elapsed=$((wait_elapsed + 10))

  # Check for evicted/failed pods first (fast detection)
  evicted=$(kubectl get pods -n "$NAMESPACE" --no-headers \
    -l "app=${DAEMONSET_NAME}" \
    --field-selector="status.phase=Failed" \
    -o custom-columns=":metadata.name,:spec.nodeName" 2>/dev/null || true)

  if [[ -n "$evicted" ]]; then
    recover_attempt=$((recover_attempt + 1))
    if [[ "$recover_attempt" -gt "$MAX_RECOVER_ATTEMPTS" ]]; then
      echo "ERROR: Exceeded $MAX_RECOVER_ATTEMPTS eviction recovery attempts, giving up"
      exit 1
    fi

    while read -r pod node; do
      [[ -z "$pod" ]] && continue
      echo "Evicted pod detected: $pod on node $node (attempt $recover_attempt/$MAX_RECOVER_ATTEMPTS)"

      # Remember this node so we never pick it again
      evicted_nodes="$evicted_nodes $node"

      # Delete the failed pod and un-label the bad node
      kubectl delete pod "$pod" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
      kubectl label node "$node" "${LABEL_KEY}-" "${ROLE_LABEL}-" 2>/dev/null || true

      # Find a replacement (exclude evicted nodes in addition to normal exclusions)
      local_excluded="$(get_excluded_nodes) $EXCLUDED_NODES $evicted_nodes"
      if find_node "$local_excluded"; then
        echo "  Replacement found: $selected_node (replacing $node)"
        kubectl label node "$selected_node" "$LABEL_KEY=$LABEL_VALUE" "$ROLE_LABEL=$ROLE"
      else
        echo "ERROR: No replacement node available for evicted $node"
        exit 1
      fi
    done <<< "$evicted"

    # Reset timeout after recovery to give the new pod time
    wait_elapsed=0
    continue
  fi

  # Check if DaemonSet is fully rolled out
  desired=$(kubectl get "daemonset/${DAEMONSET_NAME}" -n "$NAMESPACE" \
    -o jsonpath='{.status.desiredNumberScheduled}' 2>/dev/null || echo "0")
  ready=$(kubectl get "daemonset/${DAEMONSET_NAME}" -n "$NAMESPACE" \
    -o jsonpath='{.status.numberReady}' 2>/dev/null || echo "0")

  if [[ "$desired" -gt 0 ]] && [[ "$ready" -ge "$desired" ]]; then
    echo "  DaemonSet ready ($ready/$desired pods)"
    break
  fi

  if [[ "$wait_elapsed" -ge "$timeout" ]]; then
    echo "ERROR: Timed out waiting for DaemonSet rollout after ${timeout}s ($ready/$desired ready)"
    exit 1
  fi

  if (( wait_elapsed % 60 == 0 )); then
    echo "  waiting for DaemonSet... ($ready/$desired ready, ${wait_elapsed}s)"
  fi
done

echo ""
echo "Image cache is ready ($ROLE: $(kubectl get nodes -l "${ROLE_LABEL}=${ROLE}" -o jsonpath='{.items[*].metadata.name}'))"
