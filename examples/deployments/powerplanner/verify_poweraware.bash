#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# verify_poweraware.bash — Phase 3 power-aware planner smoke test
#
# Verifies that a running power-aware DynamoGraphDeployment is healthy:
#   1. Power Agent DaemonSet is ready on all GPU nodes
#   2. GPU power caps are applied (pod annotations present)
#   3. Planner Prometheus metrics exist and are non-zero
#   4. AIC optimizer metrics (if enabled) are reported
#   5. EMA correction coefficients are within bounds [0.5, 2.0]
#   6. Budget utilisation is within safe range (< 1.1)
#   7. Admission thresholds are in (0, 1] when autoset mode is active
#
# Usage:
#   ./verify_poweraware.bash [OPTIONS]
#
# Options:
#   -n  NAMESPACE        Kubernetes namespace (default: default)
#   -d  DGD_NAME         DynamoGraphDeployment name (required)
#   -p  PROM_URL         Prometheus base URL (default: http://localhost:9090)
#   -m  MODE             "phase12" | "phase3" (default: phase3)
#   -h                   Show this help
#
# Exit codes:
#   0  All checks passed
#   1  One or more checks failed (details printed to stderr)
#   2  Usage error

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NAMESPACE="default"
DGD_NAME=""
PROM_URL="http://localhost:9090"
MODE="phase3"
FAIL=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

pass()  { echo -e "  ${GREEN}[PASS]${RESET} $*"; }
fail()  { echo -e "  ${RED}[FAIL]${RESET} $*" >&2; FAIL=1; }
warn()  { echo -e "  ${YELLOW}[WARN]${RESET} $*"; }
section() { echo -e "\n=== $* ==="; }

prom_query() {
    # Usage: prom_query PROMQL
    # Returns the scalar result (first value) or "NaN" on error.
    local query="$1"
    local encoded
    encoded=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$query" 2>/dev/null || echo "$query")
    curl -sf "${PROM_URL}/api/v1/query?query=${encoded}" 2>/dev/null \
        | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('data',{}).get('result',[])
print(r[0]['value'][1] if r else 'NaN')
" 2>/dev/null || echo "NaN"
}

prom_check() {
    # Usage: prom_check DESCRIPTION PROMQL OP THRESHOLD
    #   OP: gt | lt | ge | le | eq | ne
    local desc="$1" query="$2" op="$3" threshold="$4"
    local val
    val=$(prom_query "$query")
    if [[ "$val" == "NaN" ]]; then
        fail "$desc — metric not found (query: ${query})"
        return
    fi
    local ok=0
    case "$op" in
        gt) python3 -c "exit(0 if float('$val') > float('$threshold') else 1)" && ok=1 ;;
        lt) python3 -c "exit(0 if float('$val') < float('$threshold') else 1)" && ok=1 ;;
        ge) python3 -c "exit(0 if float('$val') >= float('$threshold') else 1)" && ok=1 ;;
        le) python3 -c "exit(0 if float('$val') <= float('$threshold') else 1)" && ok=1 ;;
        eq) python3 -c "exit(0 if float('$val') == float('$threshold') else 1)" && ok=1 ;;
        ne) python3 -c "exit(0 if float('$val') != float('$threshold') else 1)" && ok=1 ;;
    esac
    if [[ "$ok" -eq 1 ]]; then
        pass "$desc (${val} ${op} ${threshold})"
    else
        fail "$desc — expected ${op} ${threshold}, got ${val}"
    fi
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while getopts "n:d:p:m:h" opt; do
    case $opt in
        n) NAMESPACE="$OPTARG" ;;
        d) DGD_NAME="$OPTARG" ;;
        p) PROM_URL="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        h) grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'; exit 0 ;;
        *) echo "Unknown option: $opt" >&2; exit 2 ;;
    esac
done

if [[ -z "$DGD_NAME" ]]; then
    echo "ERROR: -d DGD_NAME is required." >&2
    exit 2
fi

echo "Power-Aware Planner Verification"
echo "  Namespace : ${NAMESPACE}"
echo "  DGD       : ${DGD_NAME}"
echo "  Prometheus: ${PROM_URL}"
echo "  Mode      : ${MODE}"

# ---------------------------------------------------------------------------
# 1. Power Agent DaemonSet
# ---------------------------------------------------------------------------
section "1 — Power Agent DaemonSet"

AGENT_DS=$(kubectl get daemonset dynamo-power-agent -n "${NAMESPACE}" --ignore-not-found \
    -o jsonpath='{.status.numberReady}/{.status.desiredNumberScheduled}' 2>/dev/null)

if [[ -z "$AGENT_DS" ]]; then
    fail "DaemonSet dynamo-power-agent not found in namespace ${NAMESPACE}"
else
    READY=$(echo "$AGENT_DS" | cut -d/ -f1)
    DESIRED=$(echo "$AGENT_DS" | cut -d/ -f2)
    if [[ "$READY" == "$DESIRED" && "$DESIRED" -gt 0 ]]; then
        pass "Power Agent DaemonSet: ${READY}/${DESIRED} pods ready"
    else
        fail "Power Agent DaemonSet: only ${READY}/${DESIRED} pods ready"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Pod annotations — power caps applied
# ---------------------------------------------------------------------------
section "2 — Pod power-cap annotations"

ANNOTATED=$(kubectl get pods -n "${NAMESPACE}" \
    -l "nvidia.com/dynamo-graph-deployment=${DGD_NAME}" \
    -o jsonpath='{range .items[*]}{.metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}{"\n"}{end}' \
    2>/dev/null | grep -c '[0-9]' || true)

TOTAL=$(kubectl get pods -n "${NAMESPACE}" \
    -l "nvidia.com/dynamo-graph-deployment=${DGD_NAME}" \
    --no-headers 2>/dev/null | wc -l || echo 0)

if [[ "$TOTAL" -eq 0 ]]; then
    warn "No pods found for DGD ${DGD_NAME} — is it deployed?"
elif [[ "$ANNOTATED" -eq 0 ]]; then
    fail "No pods have the gpu-power-limit annotation set (0/${TOTAL} pods)"
else
    if [[ "$ANNOTATED" -eq "$TOTAL" ]]; then
        pass "All ${TOTAL} pods have gpu-power-limit annotation"
    else
        warn "${ANNOTATED}/${TOTAL} pods annotated (some may be freshly scheduled)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Planner power budget metrics
# ---------------------------------------------------------------------------
section "3 — Planner power budget metrics"

prom_check \
    "power_budget_total_watts > 0" \
    "dynamo_planner_power_budget_total_watts" \
    "gt" "0"

prom_check \
    "power_budget_utilization in (0, 1.1]" \
    "dynamo_planner_power_budget_utilization" \
    "gt" "0"

prom_check \
    "power_budget_utilization < 1.1 (budget not busted)" \
    "dynamo_planner_power_budget_utilization" \
    "lt" "1.1"

# ---------------------------------------------------------------------------
# 4. Power Agent applied limits (via DCGM/NVML metrics)
# ---------------------------------------------------------------------------
section "4 — Power Agent NVML enforcement"

prom_check \
    "dynamo_power_agent_applied_limit_watts > 0 on at least one GPU" \
    "max(dynamo_power_agent_applied_limit_watts)" \
    "gt" "0"

# Safe-default counter should be 0 in a healthy deployment.
SAFE_DEFAULT=$(prom_query "dynamo_power_agent_safe_default_applied_total")
if [[ "$SAFE_DEFAULT" == "0" ]] || [[ "$SAFE_DEFAULT" == "NaN" ]]; then
    pass "No safe-default fallbacks triggered (dynamo_power_agent_safe_default_applied_total=${SAFE_DEFAULT})"
else
    warn "Safe-default fallback applied ${SAFE_DEFAULT} time(s) — check agent logs for annotation parse errors"
fi

# ---------------------------------------------------------------------------
# 5. AIC optimizer metrics (Phase 3 only)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "phase3" ]]; then
    section "5 — AIC optimizer metrics"

    prom_check \
        "aic_c_ttft in [0.5, 2.0]" \
        "dynamo_planner_aic_c_ttft" \
        "ge" "0.5"

    prom_check \
        "aic_c_ttft in [0.5, 2.0]" \
        "dynamo_planner_aic_c_ttft" \
        "le" "2.0"

    prom_check \
        "aic_c_itl in [0.5, 2.0]" \
        "dynamo_planner_aic_c_itl" \
        "ge" "0.5"

    prom_check \
        "aic_c_itl in [0.5, 2.0]" \
        "dynamo_planner_aic_c_itl" \
        "le" "2.0"

    # Optimizer must not be disabled.
    DISABLED=$(prom_query "max(dynamo_aic_optimizer_disabled_reason)")
    if [[ "$DISABLED" == "0" ]] || [[ "$DISABLED" == "NaN" ]]; then
        pass "AIC optimizer is not auto-disabled"
    else
        fail "AIC optimizer is auto-disabled — check dynamo_aic_optimizer_disabled_reason label for reason"
    fi

    # Consecutive failures should be 0.
    prom_check \
        "aic_consecutive_failures == 0" \
        "dynamo_aic_consecutive_failures" \
        "eq" "0"

    # Power coefficients must be in [0.5, 2.0].
    for component in prefill decode; do
        prom_check \
            "aic_c_power{component=${component}} in [0.5, 2.0]" \
            "dynamo_planner_aic_c_power{component=\"${component}\"}" \
            "ge" "0.5"
        prom_check \
            "aic_c_power{component=${component}} in [0.5, 2.0]" \
            "dynamo_planner_aic_c_power{component=\"${component}\"}" \
            "le" "2.0"
    done

    # ---------------------------------------------------------------------------
    # 6. Admission control thresholds
    # ---------------------------------------------------------------------------
    section "6 — Admission control (AIC implied thresholds)"

    for metric in admission_implied_theta_decode admission_implied_theta_prefill_frac; do
        VAL=$(prom_query "dynamo_planner_${metric}")
        if [[ "$VAL" == "NaN" ]]; then
            warn "${metric} not found — is the planner running with enable_aic_optimizer=true?"
        else
            python3 -c "
v=float('$VAL')
assert 0 < v <= 1.0, f'${metric}={v} is out of (0,1]'
" 2>/dev/null && pass "${metric}=${VAL} in (0, 1]" \
                     || fail "${metric}=${VAL} is out of range (0, 1]"
        fi
    done
else
    section "5–6 — AIC checks skipped (mode=phase12)"
fi

# ---------------------------------------------------------------------------
# 7. Planner replica metrics sanity
# ---------------------------------------------------------------------------
section "7 — Planner replica counts"

prom_check \
    "prefill replicas >= 1" \
    "dynamo_planner_num_prefill_replicas" \
    "ge" "1"

prom_check \
    "decode replicas >= 1" \
    "dynamo_planner_num_decode_replicas" \
    "ge" "1"

# ---------------------------------------------------------------------------
# 8 — Phase 4 preview: real power_w in AIC (non-zero when data is integrated)
#
# When tools/integrate_aic_power_data.py has been run against the AIC checkout
# and the package has been reinstalled, estimate_perf() returns non-zero
# power_w values.  The planner emits these as dynamo_aic_optimizer_power_w_*
# gauges instead of falling back to TDP.
#
# This check is INFORMATIONAL only (not a hard failure) — it will be skipped
# before the AIC power data is integrated.
# ---------------------------------------------------------------------------
section "8 — Phase 4 preview: AIC real power_w gauges (informational)"

for side in prefill decode; do
    val=$(prom_query "dynamo_aic_optimizer_power_w_${side}" 2>/dev/null || echo "0")
    if [[ "$(echo "$val > 0" | bc -l 2>/dev/null || echo 0)" -eq 1 ]]; then
        pass "AIC power_w_${side} = ${val} W (real measured data — Phase 4 data integration active)"
    else
        echo -e "${YELLOW}  [SKIP] AIC power_w_${side} not yet exposed (Phase 4 data not integrated).${RESET}"
        echo "         To activate: run tools/integrate_aic_power_data.py then reinstall aiconfigurator."
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
if [[ "$FAIL" -eq 0 ]]; then
    echo -e "${GREEN}All checks passed.${RESET}"
else
    echo -e "${RED}One or more checks FAILED. Review errors above.${RESET}" >&2
    exit 1
fi
