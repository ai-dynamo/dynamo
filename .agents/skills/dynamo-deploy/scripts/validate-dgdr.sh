#!/usr/bin/env bash
# Pre-apply DGDR/DGD validator for Dynamo deployments.
#
# Runs a sequence of checks against a manifest file and the target cluster:
#   1. kubectl accessible
#   2. Connected to a cluster
#   3. Target namespace exists
#   4. DGDR CRD (or DGD CRD) registered, v1beta1 served
#   5. Conversion webhook reachable (for kinds with both versions)
#   6. Manifest parses as YAML
#   7. kubectl apply --dry-run=server validates the manifest
#   8. (DGDR only) hardware.gpuSku, if set, matches at least one node label
#   9. (when model is gated) hf-token-secret exists in the target namespace
#
# Output: one PASS|FAIL|WARN row per check, summary block, exit code 0 on
# all-pass and non-zero on any FAIL.
#
# Implements the pass/fail/warn pattern from SKILL_AUTHORING.md §8.3 (A9).
#
# Usage:
#   bash scripts/validate-dgdr.sh -f <manifest.yaml> [-n <namespace>]

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 -f <manifest.yaml> [-n <namespace>]
  -f  Path to DGDR or DGD YAML manifest
  -n  Target Kubernetes namespace (default: from \$KUBECTL_NAMESPACE or 'default')
  -h  Show this help

Exit code 0 = all checks passed.
USAGE
}

MANIFEST=""
NAMESPACE="${KUBECTL_NAMESPACE:-default}"

while getopts ":f:n:h" opt; do
    case "$opt" in
        f) MANIFEST="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown flag -$OPTARG" >&2; usage >&2; exit 2 ;;
        :)  echo "-$OPTARG requires an argument" >&2; exit 2 ;;
    esac
done

if [ -z "$MANIFEST" ]; then
    echo "Error: -f <manifest> is required." >&2
    usage >&2
    exit 2
fi

if [ ! -f "$MANIFEST" ]; then
    echo "Error: manifest not found: $MANIFEST" >&2
    exit 2
fi

# pass/fail/warn helpers per SKILL_AUTHORING.md §8.3 (A9)
PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { ((PASS++)); RESULTS+=("PASS|$1|$2"); }
fail() { ((FAIL++)); RESULTS+=("FAIL|$1|$2"); }
warn() { ((WARN++)); RESULTS+=("WARN|$1|$2"); }

# 1. kubectl accessible
if kubectl version --client -o json &>/dev/null; then
    ver=$(kubectl version --client -o json 2>/dev/null \
          | python3 -c "import sys,json; v=json.load(sys.stdin)['clientVersion']; print(f\"v{v['major']}.{v['minor']}\")" \
          2>/dev/null || echo "unknown")
    pass "kubectl accessible" "$ver"
else
    fail "kubectl accessible" "kubectl not found on PATH"
    # Cannot continue without kubectl; emit summary and exit.
    echo
    echo "===== Validation Summary ====="
    for row in "${RESULTS[@]}"; do echo "$row"; done
    echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
    exit 1
fi

# 2. Cluster reachable
if kubectl version -o json &>/dev/null; then
    ctx=$(kubectl config current-context 2>/dev/null || echo "unknown")
    pass "cluster reachable" "context=$ctx"
else
    fail "cluster reachable" "no cluster connection (check KUBECONFIG)"
    echo
    echo "===== Validation Summary ====="
    for row in "${RESULTS[@]}"; do echo "$row"; done
    echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
    exit 1
fi

# 3. Namespace exists
if kubectl get ns "$NAMESPACE" &>/dev/null; then
    pass "namespace exists" "$NAMESPACE"
else
    fail "namespace exists" "$NAMESPACE not found"
fi

# Detect the manifest kind (DGDR, DGD, or other)
KIND=$(python3 -c "
import yaml, sys
d = yaml.safe_load(open('$MANIFEST'))
print((d or {}).get('kind', ''))
" 2>/dev/null || echo "")

if [ -z "$KIND" ]; then
    fail "manifest parses as yaml" "could not load $MANIFEST"
    KIND="unknown"
else
    pass "manifest parses as yaml" "kind=$KIND"
fi

# 4. CRD registered, v1beta1 served
case "$KIND" in
    DynamoGraphDeploymentRequest)
        CRD_NAME="dynamographdeploymentrequests.nvidia.com"
        ;;
    DynamoGraphDeployment)
        CRD_NAME="dynamographdeployments.nvidia.com"
        ;;
    *)
        CRD_NAME=""
        ;;
esac

if [ -n "$CRD_NAME" ]; then
    if kubectl get crd "$CRD_NAME" &>/dev/null; then
        served=$(kubectl get crd "$CRD_NAME" -o jsonpath='{.spec.versions[?(@.name=="v1beta1")].served}' 2>/dev/null || echo "")
        if [ "$served" = "true" ]; then
            pass "$KIND CRD installed" "$CRD_NAME (v1beta1 served)"
        else
            warn "$KIND CRD installed" "$CRD_NAME present but v1beta1 not served"
        fi
    else
        fail "$KIND CRD installed" "$CRD_NAME not found"
    fi
fi

# 5. Conversion webhook reachable (best-effort)
if [ -n "$CRD_NAME" ]; then
    cwh=$(kubectl get crd "$CRD_NAME" -o jsonpath='{.spec.conversion.strategy}' 2>/dev/null || echo "")
    if [ "$cwh" = "Webhook" ]; then
        pass "conversion webhook configured" "$CRD_NAME uses Webhook strategy"
    elif [ -n "$cwh" ]; then
        warn "conversion webhook configured" "$CRD_NAME strategy=$cwh"
    fi
fi

# 6. (already done above: manifest parses as YAML)

# 7. kubectl apply --dry-run=server validates the manifest (per D6).
# Use 2>&1 to capture stderr; --validate=true is default but explicit.
if dry_out=$(kubectl apply --dry-run=server -n "$NAMESPACE" -f "$MANIFEST" 2>&1); then
    pass "server dry-run" "manifest validates against CRD schema"
else
    err_snippet=$(echo "$dry_out" | head -3 | tr '\n' ' ')
    fail "server dry-run" "${err_snippet}"
fi

# 8. (DGDR only) hardware.gpuSku matches a cluster node label
if [ "$KIND" = "DynamoGraphDeploymentRequest" ]; then
    gpu_sku=$(python3 -c "
import yaml
d = yaml.safe_load(open('$MANIFEST'))
spec = (d or {}).get('spec', {})
print((spec.get('hardware') or {}).get('gpuSku') or '')
" 2>/dev/null || echo "")

    if [ -n "$gpu_sku" ]; then
        # Cluster node labels use vendor names (e.g. NVIDIA-H200); we map
        # the DGDR enum to a substring expected on the node label.
        sku_lower=$(echo "$gpu_sku" | tr '[:upper:]' '[:lower:]')
        case "$sku_lower" in
            h200_sxm)   needle="H200" ;;
            h100_sxm)   needle="H100" ;;
            h100_pcie)  needle="H100" ;;
            a100_sxm)   needle="A100" ;;
            a100_pcie)  needle="A100" ;;
            l40s)       needle="L40S" ;;
            l40)        needle="L40" ;;
            l4)         needle="L4" ;;
            b200_sxm)   needle="B200" ;;
            gb200_sxm)  needle="GB200" ;;
            *)          needle="" ;;
        esac
        if [ -n "$needle" ]; then
            node_label_match=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.nvidia\.com/gpu\.product}{"\n"}{end}' 2>/dev/null | grep -c "$needle" || true)
            if [ "$node_label_match" -gt 0 ]; then
                pass "GPU SKU on cluster" "gpuSku=$gpu_sku matches $node_label_match node(s)"
            else
                warn "GPU SKU on cluster" "gpuSku=$gpu_sku not visible on any node (verify GFD labels or test environment)"
            fi
        else
            warn "GPU SKU on cluster" "gpuSku=$gpu_sku not in mapped enum; manual verification recommended"
        fi
    else
        pass "GPU SKU on cluster" "hardware.gpuSku unset (will auto-detect)"
    fi
fi

# 9. HF token Secret check (best-effort): if envFromSecret references
# hf-token-secret anywhere in the manifest, verify it exists.
hf_ref=$(grep -c "hf-token-secret" "$MANIFEST" 2>/dev/null || echo "0")
if [ "$hf_ref" -gt 0 ]; then
    if kubectl get secret hf-token-secret -n "$NAMESPACE" &>/dev/null; then
        pass "hf-token-secret present" "Secret/hf-token-secret in $NAMESPACE"
    else
        fail "hf-token-secret present" "manifest references hf-token-secret but Secret not in $NAMESPACE (per D2)"
    fi
fi

# Summary
echo
echo "===== Validation Summary ====="
for row in "${RESULTS[@]}"; do
    echo "$row"
done
echo
echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
