#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-pod integration test for --throughput-metrics-source=frontend
#
# Runs ALL components (etcd, nats, frontend, prefill worker, decode worker,
# Prometheus, planner) on a single pod — no DGD operator, no Kubernetes
# scaling.
#
# What this validates:
#   1. dynamo.planner.planner_sla starts with --throughput-metrics-source=frontend
#   2. The planner connects to frontend/worker components via etcd/nats
#   3. Prometheus scrapes dynamo_frontend_* metrics from the HTTP server
#   4. The planner observes non-zero num_req / ttft / itl after traffic is sent
#
# Usage:
#   ./single-pod-frontend-metrics-test.sh [NAMESPACE]
#
# Prerequisites:
#   - kubectl configured to your cluster
#   - docker-imagepullsecret exists in the target namespace
#
# Corrections vs. the original plan spec:
#   - The plan used --environment=local, which is NOT a valid argparse choice.
#     Valid choices are: kubernetes, virtual, global-planner.
#     We use --environment=virtual --no-operation, which is the correct equivalent:
#       virtual   = use etcd/nats for service discovery (not K8s API)
#       no-operation = skip scaling connector (no VirtualConnectorCoordinator
#                      blocking), worker counts come from runtime etcd discovery.
#   - The plan listed --backend mocker for the frontend; the frontend has no
#     --backend flag (it's backend-agnostic). Removed.
#   - Added --metric-pulling-prometheus-endpoint=http://localhost:9090.
#   - Added in-pod Prometheus (downloaded if not present) so the planner can
#     actually see non-zero metrics.

set -euo pipefail

NAMESPACE="${1:-darfeen-dynamo-cloud}"
POD="dynamo-frontend-metrics-test"
IMAGE="dynamoci.azurecr.io/ai-dynamo/dynamo:451e3d9ea6b1a4bceb55f6a798fd857a59dfd319-vllm-cuda12-amd64"
BRANCH="feat/throughput-metrics-source"
REPO="https://github.com/ai-dynamo/dynamo"
MODEL="nvidia/Llama-3.1-8B-Instruct-FP8"
PROFILE_DIR="/workspace/tests/planner/profiling_results/H200_TP1P_TP1D"
SP="/opt/dynamo/venv/lib/python3.12/site-packages"
PYTHON="/opt/dynamo/venv/bin/python3"
DYN_NS="dynamo"
PROM_VERSION="2.54.1"

# ─── Helpers ──────────────────────────────────────────────────────────────────

exec_pod() {
  # Run a bash script inside the pod. Double quotes allow host-variable expansion;
  # pod-side shell variables must be escaped: \$!, \$PATH, etc.
  kubectl exec "$POD" -n "$NAMESPACE" -- bash -c "$1"
}

cleanup() {
  echo ""
  echo "--- Cleanup ---"
  kubectl delete pod "$POD" -n "$NAMESPACE" --ignore-not-found --wait=false 2>/dev/null || true
}
trap cleanup EXIT

# ─── Step 1: Launch pod ────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: Launching pod ==="
kubectl run "$POD" \
  --image="$IMAGE" \
  --restart=Never \
  --overrides='{"spec":{"imagePullSecrets":[{"name":"docker-imagepullsecret"}]}}' \
  -n "$NAMESPACE" \
  -- sleep infinity

kubectl wait "pod/$POD" --for=condition=Ready -n "$NAMESPACE" --timeout=120s
echo "Pod ready."

# ─── Step 2: Clone PR and patch the installed venv ────────────────────────────
echo ""
echo "=== Step 2: Cloning PR branch and patching venv ==="
exec_pod "
set -e
echo 'Cloning $BRANCH...'
git clone --depth=1 --branch '$BRANCH' '$REPO' /tmp/dynamo-pr 2>&1 | tail -3

SRC_LIB=/tmp/dynamo-pr/lib/bindings/python/src/dynamo
SRC_COMP=/tmp/dynamo-pr/components/src/dynamo

echo 'Applying patches to installed venv ($SP/dynamo/)...'
cp \$SRC_LIB/prometheus_names.py               $SP/dynamo/prometheus_names.py
cp \$SRC_COMP/planner/utils/prometheus.py       $SP/dynamo/planner/utils/prometheus.py
cp \$SRC_COMP/planner/utils/planner_argparse.py $SP/dynamo/planner/utils/planner_argparse.py
cp \$SRC_COMP/planner/utils/planner_core.py     $SP/dynamo/planner/utils/planner_core.py
cp \$SRC_COMP/planner/defaults.py               $SP/dynamo/planner/defaults.py
echo 'Patches applied.'

echo 'Smoke-test: --throughput-metrics-source choices must include frontend and router'
$PYTHON - <<'PYEOF'
from dynamo.planner.utils.planner_argparse import create_sla_planner_parser
p = create_sla_planner_parser()
choices = next(a.choices for a in p._actions if a.dest == 'throughput_metrics_source')
assert 'frontend' in choices and 'router' in choices, f'Unexpected choices: {choices}'
print(f'OK -- throughput_metrics_source choices: {choices}')
PYEOF
"

# ─── Step 3: Start etcd and nats-server ───────────────────────────────────────
echo ""
echo "=== Step 3: Starting etcd and nats-server ==="
exec_pod "
set -e
if ! command -v etcd &>/dev/null; then
  echo 'ERROR: etcd not found. Install it in the image or run: apt-get install -y etcd' >&2
  exit 1
fi
if ! command -v nats-server &>/dev/null; then
  echo 'ERROR: nats-server not found. Install it or download from https://github.com/nats-io/nats-server/releases' >&2
  exit 1
fi

etcd \
  --data-dir=/tmp/etcd-data \
  --listen-client-urls=http://127.0.0.1:2379 \
  --advertise-client-urls=http://127.0.0.1:2379 \
  --listen-peer-urls=http://127.0.0.1:2380 \
  --initial-advertise-peer-urls=http://127.0.0.1:2380 \
  --initial-cluster=default=http://127.0.0.1:2380 \
  > /tmp/etcd.log 2>&1 &
echo \"etcd pid=\$!\"

nats-server > /tmp/nats.log 2>&1 &
echo \"nats-server pid=\$!\"

sleep 3
curl -sf http://127.0.0.1:2379/health >/dev/null \
  || { echo 'ERROR: etcd not healthy'; cat /tmp/etcd.log; exit 1; }
echo 'etcd + nats-server are up.'
"

# ─── Step 4a: Start frontend ──────────────────────────────────────────────────
echo ""
echo "=== Step 4a: Starting frontend (port 8000) ==="
exec_pod "
DYN_NAMESPACE=$DYN_NS \
$PYTHON -m dynamo.frontend \
  --model-name '$MODEL' \
  --http-port 8000 \
  > /tmp/frontend.log 2>&1 &
echo \"frontend pid=\$!\"
"

# ─── Step 4b: Start prefill worker ────────────────────────────────────────────
echo ""
echo "=== Step 4b: Starting prefill worker ==="
exec_pod "
cd /workspace
DYN_NAMESPACE=$DYN_NS \
$PYTHON -m dynamo.mocker \
  --model-path '$MODEL' \
  --model-name '$MODEL' \
  --speedup-ratio 5.0 \
  --planner-profile-data '$PROFILE_DIR' \
  --is-prefill-worker \
  > /tmp/prefill.log 2>&1 &
echo \"prefill worker pid=\$!\"
"

# ─── Step 4c: Start decode worker ─────────────────────────────────────────────
echo ""
echo "=== Step 4c: Starting decode worker ==="
exec_pod "
cd /workspace
DYN_NAMESPACE=$DYN_NS \
$PYTHON -m dynamo.mocker \
  --model-path '$MODEL' \
  --model-name '$MODEL' \
  --speedup-ratio 5.0 \
  --planner-profile-data '$PROFILE_DIR' \
  --is-decode-worker \
  > /tmp/decode.log 2>&1 &
echo \"decode worker pid=\$!\"
"

echo ""
echo "Waiting 15s for workers to register with etcd..."
sleep 15

# ─── Step 4d: Start in-pod Prometheus ─────────────────────────────────────────
# Prometheus scrapes localhost:8000/metrics where the frontend HTTP server
# exposes dynamo_frontend_* histograms.  We add dynamo_namespace: dynamo as a
# static label because the frontend metrics do NOT auto-inject this label
# (the DGD operator would normally copy it from the pod label via a Kubernetes
# relabel rule, which isn't available in a single-pod static-config setup).
# honor_labels=true ensures the scraped model label is preserved as-is.
# The planner queries localhost:9090 via --metric-pulling-prometheus-endpoint.
echo ""
echo "=== Step 4d: Starting in-pod Prometheus ==="
exec_pod "
set -e
if ! command -v prometheus &>/dev/null; then
  echo 'Downloading Prometheus $PROM_VERSION...'
  curl -fsSL \
    'https://github.com/prometheus/prometheus/releases/download/v$PROM_VERSION/prometheus-$PROM_VERSION.linux-amd64.tar.gz' \
    | tar -xz --strip-components=1 \
          -C /tmp \
          'prometheus-$PROM_VERSION.linux-amd64/prometheus'
  chmod +x /tmp/prometheus
  export PATH=/tmp:\$PATH
  echo 'Downloaded to /tmp/prometheus'
fi

cat > /tmp/prometheus.yml << 'PROMEOF'
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: dynamo-frontend
    static_configs:
      - targets: ['localhost:8000']
        labels:
          # Inject dynamo_namespace so the planner's label filter matches.
          # (The DGD operator normally copies this from the Kubernetes pod
          # label via a relabel rule; static config must do it explicitly.)
          dynamo_namespace: dynamo
    honor_labels: true
PROMEOF

prometheus \
  --config.file=/tmp/prometheus.yml \
  --storage.tsdb.path=/tmp/prometheus-data \
  --storage.tsdb.retention.time=1h \
  --web.listen-address=:9090 \
  --log.level=warn \
  > /tmp/prometheus.log 2>&1 &
echo \"Prometheus pid=\$!\"

sleep 5
curl -sf 'http://localhost:9090/-/ready' >/dev/null \
  && echo 'Prometheus is ready.' \
  || { echo 'WARNING: Prometheus did not become ready.'; tail -10 /tmp/prometheus.log; }
"

# ─── Step 4e: Start planner ────────────────────────────────────────────────────
# --environment=virtual --no-operation:
#   * no-operation  → skips validate_deployment / wait_for_deployment_ready /
#                     connector initialization entirely; worker counts are
#                     discovered via etcd through the DistributedRuntime.
#   * virtual       → valid argparse choice (the plan spec said "local" which
#                     does not exist); with no-operation the choice is
#                     effectively ignored since no connector object is created.
#   * --model-name  → required when --no-operation is set (no DGD to read from)
echo ""
echo "=== Step 4e: Starting planner ==="
exec_pod "
DYN_NAMESPACE=$DYN_NS \
$PYTHON -m dynamo.planner.planner_sla \
  --environment=virtual \
  --backend=mocker \
  --mode=disagg \
  --no-operation \
  --model-name='$MODEL' \
  --adjustment-interval=30 \
  --ttft=2000 \
  --itl=200 \
  --max-gpu-budget=-1 \
  --prefill-engine-num-gpu=1 \
  --decode-engine-num-gpu=1 \
  --no-correction \
  --profile-results-dir='$PROFILE_DIR' \
  --metric-pulling-prometheus-endpoint=http://localhost:9090 \
  --throughput-metrics-source=frontend \
  > /tmp/planner.log 2>&1 &
echo \"planner pid=\$!\"
"

# ─── Step 5: Send traffic ──────────────────────────────────────────────────────
echo ""
echo "=== Step 5: Sending test traffic ==="
echo "Waiting 10s for the frontend HTTP server to start serving..."
sleep 10

exec_pod "
PAYLOAD='{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 20.\"}],\"max_tokens\":64}'
echo 'Sending 3 waves x 10 concurrent requests...'
for wave in 1 2 3; do
  for i in \$(seq 1 10); do
    curl -sf -X POST http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d \"\$PAYLOAD\" \
      -o /dev/null &
  done
  wait
  echo \"Wave \$wave/3 complete (10 requests).\"
  sleep 3
done
echo 'Traffic complete.'
"

# ─── Step 6: Wait for planner's first adjustment ──────────────────────────────
echo ""
echo "Waiting 65s for the planner first adjustment interval..."
echo "(30s INIT_PLANNER_START_DELAY + 30s adjustment-interval + 5s margin)"
sleep 65

# ─── Step 7: Check outputs ────────────────────────────────────────────────────
echo ""
echo "=== Step 7: Results ==="

exec_pod "
echo '──────────────────────────────────────────────────'
echo 'Planner log (last 60 lines)'
echo '──────────────────────────────────────────────────'
tail -60 /tmp/planner.log 2>/dev/null || echo '(planner log not found)'

echo ''
echo '──────────────────────────────────────────────────'
echo 'Key planner lines (Observed / adjustment / scaling)'
echo '──────────────────────────────────────────────────'
grep -iE 'observed|num_req|ttft|itl|adjustment started|scaling|replicas|prometheus|frontend|throughput' \
  /tmp/planner.log 2>/dev/null | tail -30 \
  || echo '(no matching lines yet — planner may still be in initial 30s delay)'

echo ''
echo '──────────────────────────────────────────────────'
echo 'Prometheus: dynamo_frontend_requests_total'
echo '──────────────────────────────────────────────────'
curl -sG 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=dynamo_frontend_requests_total' \
  | $PYTHON -m json.tool 2>/dev/null \
  | head -40 \
  || echo '(query failed — Prometheus may not be running)'

echo ''
echo '──────────────────────────────────────────────────'
echo 'Frontend /metrics (dynamo_frontend_requests_total lines)'
echo '──────────────────────────────────────────────────'
curl -sf http://localhost:8000/metrics 2>/dev/null \
  | grep 'dynamo_frontend_requests_total' \
  || echo '(no dynamo_frontend_requests_total — frontend may not be serving metrics yet)'
"

# ─── Cleanup note ─────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo ""
echo "To keep the pod alive for manual debugging, remove or comment the trap cleanup EXIT line."
echo "Log files inside the pod: /tmp/{frontend,prefill,decode,planner,prometheus,etcd,nats}.log"
echo ""
echo "Expected success indicators:"
echo "  - Planner log shows 'New throughput adjustment interval started!'"
echo "  - Planner log shows 'Observed: num_req=X.X' with X > 0"
echo "  - Prometheus query returns data for dynamo_frontend_requests_total"
echo ""
echo "If metrics show 0:"
echo "  1. Verify frontend is emitting:  curl http://localhost:8000/metrics | grep dynamo_frontend"
echo "  2. Verify Prometheus is scraping: curl http://localhost:9090/targets"
echo "  3. Check /tmp/prometheus.log for scrape errors"
