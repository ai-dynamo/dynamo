#!/usr/bin/env bash
# Post-install verifier for the dynamo-platform Helm chart.
#
# Confirms the operator, CRDs, conversion webhooks, etcd, and NATS are
# healthy. Optional checks for Grove and KAI sub-charts.
#
# Output: one PASS/FAIL line per check, summary block, exit code 0 on
# all-pass and non-zero on any FAIL.
#
# Implements the check() pattern from SKILL_AUTHORING.md §8.4 (A10).
#
# Usage:
#   bash scripts/verify-platform.sh [-n <namespace>]

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 [-n <namespace>]
  -n  Namespace where dynamo-platform is installed (default: from
      \$KUBECTL_NAMESPACE or 'dynamo-system')
  -h  Show this help

Exit code 0 = all required checks passed.
USAGE
}

NAMESPACE="${KUBECTL_NAMESPACE:-dynamo-system}"

while getopts ":n:h" opt; do
    case "$opt" in
        n) NAMESPACE="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown flag -$OPTARG" >&2; usage >&2; exit 2 ;;
        :)  echo "-$OPTARG requires an argument" >&2; exit 2 ;;
    esac
done

PASS=0
FAIL=0

# check() per SKILL_AUTHORING.md §8.4 (A10):
#   check "<description>" "<command>" "<expected pattern>"
check() {
    local desc="$1" cmd="$2" pattern="$3"
    local out
    if out=$(bash -c "$cmd" 2>&1) && echo "$out" | grep -q "$pattern"; then
        ((PASS++))
        echo "PASS: $desc"
    else
        ((FAIL++))
        echo "FAIL: $desc"
    fi
}

# kubectl present
if ! kubectl version --client -o json &>/dev/null; then
    echo "FAIL: kubectl not found on PATH"
    exit 1
fi

# 1. Operator pod ready
check "operator deployment available" \
      "kubectl get deploy -n $NAMESPACE -l app.kubernetes.io/name=dynamo-operator -o jsonpath='{.items[0].status.conditions[?(@.type==\"Available\")].status}'" \
      "True"

# 2. All seven CRDs registered
for crd in \
    dynamographdeploymentrequests.nvidia.com \
    dynamographdeployments.nvidia.com \
    dynamocomponentdeployments.nvidia.com \
    dynamographdeploymentscalingadapters.nvidia.com \
    dynamomodels.nvidia.com \
    dynamocheckpoints.nvidia.com \
    dynamoworkermetadatas.nvidia.com
do
    check "CRD $crd registered" \
          "kubectl get crd $crd -o jsonpath='{.metadata.name}'" \
          "$crd"
done

# 3. v1beta1 served for the four CRDs with both versions
for crd in \
    dynamographdeploymentrequests.nvidia.com \
    dynamographdeployments.nvidia.com \
    dynamocomponentdeployments.nvidia.com \
    dynamographdeploymentscalingadapters.nvidia.com
do
    check "$crd v1beta1 served" \
          "kubectl get crd $crd -o jsonpath='{.spec.versions[?(@.name==\"v1beta1\")].served}'" \
          "true"
done

# 4. Conversion webhook configured for CRDs with both versions
for crd in \
    dynamographdeploymentrequests.nvidia.com \
    dynamographdeployments.nvidia.com \
    dynamocomponentdeployments.nvidia.com \
    dynamographdeploymentscalingadapters.nvidia.com
do
    check "$crd conversion webhook" \
          "kubectl get crd $crd -o jsonpath='{.spec.conversion.strategy}'" \
          "Webhook"
done

# 5. etcd quorum (sub-chart in dynamo-platform)
check "etcd pods running" \
      "kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=etcd -o jsonpath='{.items[*].status.phase}'" \
      "Running"

# 6. NATS running
check "NATS pods running" \
      "kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=nats -o jsonpath='{.items[*].status.phase}'" \
      "Running"

# 7. Optional: Grove sub-chart (advisory only; not a hard failure if absent)
if kubectl get deploy -n "$NAMESPACE" -l app.kubernetes.io/name=grove &>/dev/null; then
    check "Grove sub-chart present" \
          "kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=grove -o jsonpath='{.items[*].status.phase}'" \
          "Running"
else
    echo "INFO: Grove sub-chart not installed (optional; required for gang scheduling)"
fi

# 8. Optional: KAI scheduler sub-chart
if kubectl get deploy -n "$NAMESPACE" -l app.kubernetes.io/name=kai-scheduler &>/dev/null; then
    check "KAI scheduler sub-chart present" \
          "kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=kai-scheduler -o jsonpath='{.items[*].status.phase}'" \
          "Running"
else
    echo "INFO: KAI scheduler sub-chart not installed (optional; required for GPU-aware scheduling)"
fi

# 9. The dynamo-operator Helm release exists
check "dynamo-platform Helm release deployed" \
      "helm status dynamo-platform -n $NAMESPACE -o json" \
      "\"status\":\"deployed\""

echo
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
