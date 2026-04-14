#!/bin/bash
# test_k8s_failover.sh — End-to-end single-node failover test on K8s
#
# Deploys a failover-enabled DGD, validates convergence, runs inference,
# kills the active engine, verifies failover, and checks standby recovery.
#
# Usage:
#   ./test_k8s_failover.sh [options]
#
# Options:
#   --engine-image IMAGE    Engine image (required)
#   --operator-image IMAGE  Operator image (if provided, upgrades helm release)
#   --namespace NS          K8s namespace (default: failover-e2e-test)
#   --node NODE             Pin worker to this node via nodeSelector
#   --model MODEL           Model to serve (default: Qwen/Qwen3-0.6B)
#   --tp TP                 Tensor parallel size (default: 2)
#   --gpu-count N           GPU count for DRA (default: same as --tp)
#   --dgd-name NAME         DGD resource name (default: failover-e2e)
#   --hf-secret NAME        HF token secret name (default: hf-token-secret)
#   --skip-deploy           Skip DGD deployment (test existing deployment)
#   --skip-operator         Skip operator upgrade
#   --cleanup               Delete DGD after test
#   --help                  Show this help

set -u

# ── Defaults ──────────────────────────────────────────────────────────────────
NAMESPACE="failover-e2e-test"
MODEL="Qwen/Qwen3-0.6B"
TP=2
GPU_COUNT=""
DGD_NAME="failover-e2e"
HF_SECRET="hf-token-secret"
NODE=""
ENGINE_IMAGE=""
OPERATOR_IMAGE=""
SKIP_DEPLOY=false
SKIP_OPERATOR=true
CLEANUP=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --engine-image)   ENGINE_IMAGE="$2"; shift 2 ;;
    --operator-image) OPERATOR_IMAGE="$2"; SKIP_OPERATOR=false; shift 2 ;;
    --namespace)      NAMESPACE="$2"; shift 2 ;;
    --node)           NODE="$2"; shift 2 ;;
    --model)          MODEL="$2"; shift 2 ;;
    --tp)             TP="$2"; shift 2 ;;
    --gpu-count)      GPU_COUNT="$2"; shift 2 ;;
    --dgd-name)       DGD_NAME="$2"; shift 2 ;;
    --hf-secret)      HF_SECRET="$2"; shift 2 ;;
    --skip-deploy)    SKIP_DEPLOY=true; shift ;;
    --skip-operator)  SKIP_OPERATOR=true; shift ;;
    --cleanup)        CLEANUP=true; shift ;;
    --help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

GPU_COUNT="${GPU_COUNT:-$TP}"

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
TOTAL=0

phase() { echo -e "\n\033[1;34m━━━ Phase $1: $2 ━━━\033[0m"; }
check() {
  TOTAL=$((TOTAL + 1))
  local desc="$1"; shift
  local rc=0
  "$@" >/dev/null 2>&1 || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "  ✓ $desc"
    PASS=$((PASS + 1))
  else
    echo "  ✗ $desc"
    FAIL=$((FAIL + 1))
  fi
}

wait_for_pods() {
  local label="$1" expected_ready="$2" timeout_s="${3:-300}"
  local deadline=$((SECONDS + timeout_s))
  while [[ $SECONDS -lt $deadline ]]; do
    local ready
    ready=$(kubectl get pods -n "$NAMESPACE" -l "$label" \
      -o jsonpath='{range .items[*]}{range .status.containerStatuses[*]}{.ready}{"\n"}{end}{end}' 2>/dev/null \
      | grep -c "true" || true)
    [[ "$ready" -ge "$expected_ready" ]] && return 0
    sleep 3
  done
  return 1
}

get_worker_pod() {
  kubectl get pod -n "$NAMESPACE" -l nvidia.com/dynamo-component=VllmWorker \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

get_frontend_svc() {
  kubectl get svc -n "$NAMESPACE" -l nvidia.com/dynamo-component=Frontend \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

engine_log_grep() {
  local pod="$1" container="$2" pattern="$3"
  kubectl logs "$pod" -c "$container" -n "$NAMESPACE" 2>/dev/null | grep -qE "$pattern"
}

do_inference() {
  local svc="$1"
  kubectl run "failover-curl-$RANDOM" --rm -i --restart=Never -n "$NAMESPACE" \
    --image=curlimages/curl -- \
    curl -sf --max-time 30 "http://${svc}:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":10}" \
    2>&1 | grep "^{" | head -1
}

# ── Phase 0: Operator upgrade (optional) ──────────────────────────────────────
if [[ "$SKIP_OPERATOR" == false && -n "$OPERATOR_IMAGE" ]]; then
  phase 0 "Upgrade operator"
  OPERATOR_REPO="${OPERATOR_IMAGE%:*}"
  OPERATOR_TAG="${OPERATOR_IMAGE##*:}"

  # Find the chart — look relative to this script, then common locations
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CHART_DIR=""
  for candidate in \
    "${SCRIPT_DIR}/deploy/helm/charts/platform" \
    "${SCRIPT_DIR}/../deploy/helm/charts/platform"; do
    [[ -f "${candidate}/Chart.yaml" ]] && CHART_DIR="$candidate" && break
  done
  if [[ -z "$CHART_DIR" ]]; then
    echo "  ⚠ Could not find platform helm chart, skipping operator upgrade"
  else
    helm upgrade dynamo-platform "$CHART_DIR" \
      -n "$NAMESPACE" --install --reuse-values \
      --set "dynamo-operator.controllerManager.manager.image.repository=${OPERATOR_REPO}" \
      --set "dynamo-operator.controllerManager.manager.image.tag=${OPERATOR_TAG}" \
      --wait --timeout 5m 2>&1 | tail -1

    kubectl rollout restart deployment dynamo-platform-dynamo-operator-controller-manager \
      -n "$NAMESPACE" >/dev/null 2>&1
    kubectl rollout status deployment dynamo-platform-dynamo-operator-controller-manager \
      -n "$NAMESPACE" --timeout=120s >/dev/null 2>&1
    echo "  Operator upgraded to ${OPERATOR_IMAGE}"
  fi
fi

# ── Phase 1: Deploy DGD ──────────────────────────────────────────────────────
if [[ "$SKIP_DEPLOY" == false ]]; then
  phase 1 "Deploy failover DGD"

  if [[ -z "$ENGINE_IMAGE" ]]; then
    echo "  ERROR: --engine-image is required (or use --skip-deploy)"
    exit 1
  fi

  # Delete existing DGD if present
  kubectl delete dgd "$DGD_NAME" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1
  sleep 5

  # Build nodeSelector block
  NODE_SELECTOR=""
  if [[ -n "$NODE" ]]; then
    NODE_SELECTOR="
        nodeSelector:
          kubernetes.io/hostname: ${NODE}"
  fi

  cat <<YAML | kubectl apply -n "$NAMESPACE" -f -
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: ${DGD_NAME}
spec:
  services:
    Frontend:
      envFromSecret: ${HF_SECRET}
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: ${ENGINE_IMAGE}
    VllmWorker:
      envFromSecret: ${HF_SECRET}
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "${GPU_COUNT}"
      failover:
        enabled: true
      extraPodSpec:${NODE_SELECTOR}
        mainContainer:
          image: ${ENGINE_IMAGE}
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - ${MODEL}
            - --tensor-parallel-size
            - "${TP}"
            - --load-format
            - gms
YAML
  echo "  DGD ${DGD_NAME} applied"
fi

# ── Phase 2: Wait for convergence ────────────────────────────────────────────
phase 2 "Wait for convergence"

echo "  Waiting for worker pod (up to 5 min)..."
if ! wait_for_pods "nvidia.com/dynamo-component=VllmWorker" 2 300; then
  echo "  ERROR: Worker pod did not become ready in 5 min"
  POD=$(get_worker_pod)
  if [[ -n "$POD" ]]; then
    echo "  --- engine-0 tail ---"
    kubectl logs "$POD" -c engine-0 -n "$NAMESPACE" 2>&1 | tail -10
    echo "  --- engine-1 tail ---"
    kubectl logs "$POD" -c engine-1 -n "$NAMESPACE" 2>&1 | tail -10
  fi
  exit 1
fi

POD=$(get_worker_pod)
FRONTEND_SVC=$(get_frontend_svc)
echo "  Worker pod: ${POD}"
echo "  Worker node: $(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')"

# ── Phase 3: Validate operator behavior ──────────────────────────────────────
phase 3 "Validate operator behavior"

check "ResourceClaimTemplate exists" \
  kubectl get resourceclaimtemplate -n "$NAMESPACE" -o name

check "Pod has 2 engine containers" \
  test "$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[*].name}' | wc -w)" -eq 2

check "Pod has GMS sidecar init container" \
  test "$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.initContainers[0].name}')" = "gms-weights"

check "Engine-0 has CONTAINER_NAME=engine-0" \
  test "$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].env[?(@.name=="CONTAINER_NAME")].value}')" = "engine-0"

check "Engine-1 has CONTAINER_NAME=engine-1" \
  test "$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[1].env[?(@.name=="CONTAINER_NAME")].value}')" = "engine-1"

check "Engine-1 has staggered --master-port" \
  test -n "$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[1].args[*]}' | grep master-port)"

check "DYN_VLLM_GMS_SHADOW_MODE=true on engine-0" \
  test "$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].env[?(@.name=="DYN_VLLM_GMS_SHADOW_MODE")].value}')" = "true"

# ── Phase 4: Validate engine lifecycle ────────────────────────────────────────
phase 4 "Validate engine lifecycle"

check "Engine-0: shadow mode enabled" \
  engine_log_grep "$POD" engine-0 "Shadow.*Enabled shadow mode"

check "Engine-0: GMS RW allocator" \
  engine_log_grep "$POD" engine-0 "Created RW allocator"

check "Engine-1: GMS RO mode" \
  engine_log_grep "$POD" engine-1 "gms_read_only=True"

# Determine which engine is active (has the lock)
ACTIVE=""
STANDBY=""
if engine_log_grep "$POD" engine-1 "Lock acquired"; then
  ACTIVE="engine-1"
  STANDBY="engine-0"
elif engine_log_grep "$POD" engine-0 "Lock acquired"; then
  ACTIVE="engine-0"
  STANDBY="engine-1"
fi

if [[ -n "$ACTIVE" ]]; then
  echo "  Active engine: ${ACTIVE}, Standby: ${STANDBY}"
  check "${ACTIVE}: lock acquired" true
  check "${ACTIVE}: KV cache allocated on wake" \
    engine_log_grep "$POD" "$ACTIVE" "Allocated KV cache on wake|Successfully allocated KV cache"
  check "${STANDBY}: sleeping, waiting for lock" \
    engine_log_grep "$POD" "$STANDBY" "Engine sleeping.*waiting for lock"
else
  echo "  ⚠ Could not determine active engine"
  FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1))
fi

RESTART_SUM=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[*].restartCount}' | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
check "0 restarts" \
  test "${RESTART_SUM:-99}" -eq 0

# ── Phase 5: Inference test ──────────────────────────────────────────────────
phase 5 "Inference test"

RESPONSE=$(do_inference "$FRONTEND_SVC" 2>&1 || true)
if echo "$RESPONSE" | grep -q '"choices"'; then
  check "Inference returns valid response" true
  echo "  Response: $(echo "$RESPONSE" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"][:80])' 2>/dev/null || echo "(parse error)")"
else
  check "Inference returns valid response" false
  echo "  Response: ${RESPONSE:0:200}"
fi

# ── Phase 6: Failover test ───────────────────────────────────────────────────
if [[ -z "$ACTIVE" ]]; then
  echo -e "\n  ⚠ Skipping failover test — could not determine active engine"
else
  phase 6 "Failover test (kill ${ACTIVE})"

  KILL_TIME=$(date +%s)
  echo "  Killing ${ACTIVE} at $(date +%H:%M:%S)"
  kubectl exec "$POD" -c "$ACTIVE" -n "$NAMESPACE" -- kill 1 2>/dev/null || true

  # Wait for standby to wake and register
  echo "  Waiting for ${STANDBY} to take over..."
  FAILOVER_OK=false
  for i in $(seq 1 30); do
    sleep 2
    if engine_log_grep "$POD" "$STANDBY" "Lock acquired"; then
      WAKE_TIME=$(kubectl logs "$POD" -c "$STANDBY" -n "$NAMESPACE" 2>/dev/null \
        | grep "Lock acquired" | tail -1 | grep -oP '\d{2}:\d{2}:\d{2}\.\d{3}' | head -1)
      echo "  ${STANDBY} acquired lock at ${WAKE_TIME}"
      FAILOVER_OK=true
      break
    fi
  done

  check "${STANDBY} acquired lock after kill" $FAILOVER_OK

  # Wait a moment for registration
  sleep 3

  check "${STANDBY}: KV cache allocated" \
    engine_log_grep "$POD" "$STANDBY" "Allocated KV cache on wake|Successfully allocated KV cache"

  check "${STANDBY}: generate endpoint registered" \
    engine_log_grep "$POD" "$STANDBY" "Registered endpoint.*generate"

  # Inference after failover
  RESPONSE2=$(do_inference "$FRONTEND_SVC" 2>&1 || true)
  if echo "$RESPONSE2" | grep -q '"choices"'; then
    check "Inference works after failover" true
  else
    check "Inference works after failover" false
    echo "  Response: ${RESPONSE2:0:200}"
  fi

  # ── Phase 7: Verify killed engine recovers as standby ────────────────────
  phase 7 "Verify ${ACTIVE} recovers as standby"

  echo "  Waiting for ${ACTIVE} to restart and go to sleep (up to 3 min)..."
  RECOVERY_OK=false
  for i in $(seq 1 60); do
    sleep 3
    # Check current (post-restart) logs for the sleeping message
    if kubectl logs "$POD" -c "$ACTIVE" -n "$NAMESPACE" 2>/dev/null \
        | grep -q "Engine sleeping.*waiting for lock"; then
      RECOVERY_OK=true
      break
    fi
  done

  check "${ACTIVE} restarted and went to sleep" $RECOVERY_OK

  RESTARTS=$(kubectl get pod "$POD" -n "$NAMESPACE" \
    -o jsonpath='{.status.containerStatuses[*].restartCount}' 2>/dev/null)
  echo "  Container restarts: ${RESTARTS}"
  RESTART_SUM=$(echo "$RESTARTS" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
  check "Exactly 1 total restart (from the kill)" \
    test "${RESTART_SUM:-99}" -eq 1

  # Wait for both containers to become ready (startup probes need to pass)
  BOTH_READY=false
  for i in $(seq 1 30); do
    READY_COUNT=$(kubectl get pod "$POD" -n "$NAMESPACE" \
      -o jsonpath='{range .status.containerStatuses[*]}{.ready}{"\n"}{end}' 2>/dev/null \
      | grep -c "true" || true)
    if [[ "$READY_COUNT" -eq 2 ]]; then
      BOTH_READY=true
      break
    fi
    sleep 5
  done
  check "Both containers ready" $BOTH_READY
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
if [[ "$CLEANUP" == true ]]; then
  echo -e "\n  Cleaning up DGD..."
  kubectl delete dgd "$DGD_NAME" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "\n\033[1m━━━ Results: ${PASS}/${TOTAL} passed"
if [[ $FAIL -gt 0 ]]; then
  echo -e "\033[1;31m    ${FAIL} FAILED\033[0m"
  exit 1
else
  echo -e "\033[1;32m    ALL PASSED\033[0m"
  exit 0
fi
