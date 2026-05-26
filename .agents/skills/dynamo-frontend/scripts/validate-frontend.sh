#!/usr/bin/env bash
# validate-frontend.sh — pre-apply Frontend + DynamoModel validator.
#
# Confirms the DGD Frontend can take the proposed config and DynamoModel CRs
# without obvious breakage. Checks:
#   1. DGD exists and is Ready
#   2. Frontend pod is Ready
#   3. DynamoModel CRD is registered
#   4. HF token Secret is present if any DynamoModel references a gated model
#   5. Any DynamoModel manifests passed via --dm-file validate against the CRD
#
# Implements PASS/FAIL/WARN per SKILL_AUTHORING.md §8.3 (A9).
#
# Usage:
#   bash scripts/validate-frontend.sh -d <dgd-name> [-n <ns>] [--dm-file <file>]

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 -d <dgd-name> [-n <ns>] [--dm-file <file>]
  -d  Target DGD name. Required.
  -n  Namespace. Default: from \$KUBECTL_NAMESPACE or 'default'.
  --dm-file  Path to a DynamoModel YAML to dry-run against the API server.
  -h  Show this help.
USAGE
}

DGD_NAME=""
NAMESPACE="${KUBECTL_NAMESPACE:-default}"
DM_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -d) DGD_NAME="$2"; shift 2 ;;
        -n) NAMESPACE="$2"; shift 2 ;;
        --dm-file) DM_FILE="$2"; shift 2 ;;
        -h) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$DGD_NAME" ]; then
    echo "Error: -d <dgd-name> is required." >&2
    usage >&2
    exit 2
fi

PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { ((PASS++)); RESULTS+=("PASS|$1|$2"); }
fail() { ((FAIL++)); RESULTS+=("FAIL|$1|$2"); }
warn() { ((WARN++)); RESULTS+=("WARN|$1|$2"); }

# 1. DGD exists and is Ready.
if kubectl get dgd "$DGD_NAME" -n "$NAMESPACE" &>/dev/null; then
    state=$(kubectl get dgd "$DGD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.state}')
    if [ "$state" = "successful" ] || [ "$state" = "Ready" ] || [ "$state" = "Deployed" ]; then
        pass "DGD ready" "state=$state"
    else
        fail "DGD ready" "state=$state (expected successful/Ready/Deployed)"
    fi
else
    fail "DGD exists" "$DGD_NAME not found in $NAMESPACE"
    echo "===== Validation Summary ====="
    for row in "${RESULTS[@]}"; do echo "$row"; done
    exit 1
fi

# 2. Frontend pod Ready.
frontend_ready=$(kubectl get pods -n "$NAMESPACE" -l "nvidia.com/dgd-name=$DGD_NAME,app.kubernetes.io/component=frontend" \
    -o jsonpath='{.items[*].status.containerStatuses[*].ready}' 2>/dev/null | tr ' ' '\n' | grep -c true || true)
if [ "$frontend_ready" -gt 0 ]; then
    pass "Frontend pod ready" "$frontend_ready container(s) ready"
else
    fail "Frontend pod ready" "no Frontend container reports ready=true"
fi

# 3. DynamoModel CRD registered (per [F2]).
if kubectl get crd dynamomodels.nvidia.com &>/dev/null; then
    served=$(kubectl get crd dynamomodels.nvidia.com -o jsonpath='{.spec.versions[?(@.name=="v1alpha1")].served}' 2>/dev/null || echo "")
    if [ "$served" = "true" ]; then
        pass "DynamoModel CRD" "v1alpha1 served"
    else
        warn "DynamoModel CRD" "CRD present but v1alpha1 not served"
    fi
else
    fail "DynamoModel CRD" "dynamomodels.nvidia.com not installed"
fi

# 4. HF token Secret present (best-effort).
if kubectl get secret hf-token-secret -n "$NAMESPACE" &>/dev/null; then
    pass "hf-token-secret present" "$NAMESPACE/hf-token-secret"
else
    warn "hf-token-secret present" "not in $NAMESPACE — gated models will 401 (per D2)"
fi

# 5. Optional DynamoModel dry-run.
if [ -n "$DM_FILE" ]; then
    if [ ! -f "$DM_FILE" ]; then
        fail "dm-file present" "$DM_FILE not found"
    else
        # First, sanity-check the YAML parses.
        if python3 -c "import yaml; list(yaml.safe_load_all(open('$DM_FILE')))" 2>/dev/null; then
            pass "dm-file YAML parses" "$DM_FILE"
        else
            fail "dm-file YAML parses" "$DM_FILE failed yaml.safe_load_all"
        fi

        # Server dry-run against the CRD.
        if dry_out=$(kubectl apply --dry-run=server -n "$NAMESPACE" -f "$DM_FILE" 2>&1); then
            pass "dm-file server dry-run" "validates against DynamoModel CRD"
        else
            snippet=$(echo "$dry_out" | head -3 | tr '\n' ' ')
            fail "dm-file server dry-run" "$snippet"
        fi

        # Reject v1beta1 manifests upfront (per [F2] DynamoModel is v1alpha1-only).
        if grep -q "apiVersion: nvidia.com/v1beta1" "$DM_FILE" && grep -q "kind: DynamoModel" "$DM_FILE"; then
            fail "dm-file API version" "DynamoModel must use nvidia.com/v1alpha1 — v1beta1 schema does not exist (per F2)"
        fi
    fi
fi

# Summary
echo
echo "===== Validation Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo
echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
