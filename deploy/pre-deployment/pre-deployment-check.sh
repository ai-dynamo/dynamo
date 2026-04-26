#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROFILE="baseline"
OUTPUT="table"
REQUIRE=""
DETAIL=""
CHECK_NAMES=()
CHECK_STATUSES=()
CHECK_DETAILS=()

usage() {
    cat <<'EOF'
Usage: pre-deployment-check.sh [options]

Options:
  --profile baseline|production  Check Dynamo's baseline prerequisites or the full production profile.
  --output table|json            Print a human-readable table or machine-readable JSON.
  --require name[,name...]       Run specific checks only. Use --help to see names.
  --help                         Show this help.

Check names:
  kubectl
  helm
  default-storage-class
  gpu-resources
  gpu-operator
  argocd
  prometheus-operator
  prometheus-instance
  dcgm-servicemonitor
  loki
  fluentd
  falco
  trivy
  velero
  external-secrets
  grove-kai
  lws-volcano
  keda
  opentelemetry
  network-policies
  dynamo-crds
  dynamo-webhooks
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --output)
                OUTPUT="$2"
                shift 2
                ;;
            --require)
                REQUIRE="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 2
                ;;
        esac
    done

    case "$PROFILE" in
        baseline|production) ;;
        *)
            echo "Unsupported profile: $PROFILE" >&2
            exit 2
            ;;
    esac

    case "$OUTPUT" in
        table|json) ;;
        *)
            echo "Unsupported output: $OUTPUT" >&2
            exit 2
            ;;
    esac
}

print_status() {
    if [[ "$OUTPUT" == "json" ]]; then
        return
    fi

    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    if [[ "$OUTPUT" == "json" ]]; then
        return
    fi

    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  Dynamo Pre-Deployment Check Script  ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Profile: ${PROFILE}${NC}\n"
}

print_section() {
    if [[ "$OUTPUT" != "json" ]]; then
        echo -e "\n${BLUE}--- $1 ---${NC}"
    fi
}

record_check_result() {
    CHECK_NAMES+=("$1")
    CHECK_STATUSES+=("$2")
    CHECK_DETAILS+=("$3")
}

run_check() {
    local name=$1
    local fn=$2

    DETAIL=""
    if "$fn"; then
        record_check_result "$name" "PASS" "$DETAIL"
    else
        record_check_result "$name" "FAIL" "$DETAIL"
        return 1
    fi
}

kubectl_get() {
    kubectl get "$@" 2>/dev/null
}

kubectl_has_crd() {
    local crd=$1
    kubectl_get crd "$crd" >/dev/null
}

kubectl_has_pods() {
    local selector=$1
    kubectl_get pods -A -l "$selector" --no-headers | grep -q .
}

check_kubectl() {
    print_section "Checking kubectl connectivity"

    if ! command -v kubectl >/dev/null 2>&1; then
        DETAIL="kubectl is not installed or not in PATH"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    if ! kubectl cluster-info >/dev/null 2>&1; then
        DETAIL="kubectl cannot connect to the configured Kubernetes cluster"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="kubectl is installed and cluster-info succeeds"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_helm() {
    print_section "Checking Helm"

    if ! command -v helm >/dev/null 2>&1; then
        DETAIL="helm is not installed or not in PATH"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="$(helm version --short 2>/dev/null || echo "helm is installed")"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_default_storage_class() {
    print_section "Checking for default StorageClass"

    local default_storage_classes
    default_storage_classes=$(kubectl_get storageclass -o jsonpath='{range .items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")]}{.metadata.name}{"\n"}{end}' || true)

    if [[ -z "$default_storage_classes" ]]; then
        DETAIL="no StorageClass has storageclass.kubernetes.io/is-default-class=true"
        print_status "$RED" "FAIL: $DETAIL"

        if [[ "$OUTPUT" != "json" ]]; then
            print_status "$BLUE" "Available StorageClasses:"
            kubectl_get storageclass || true
            print_status "$YELLOW" "Set a default with: kubectl patch storageclass <name> -p '{\"metadata\":{\"annotations\":{\"storageclass.kubernetes.io/is-default-class\":\"true\"}}}'"
        fi
        return 1
    fi

    local default_count
    default_count=$(echo "$default_storage_classes" | grep -c . || true)
    DETAIL="default StorageClass: $(echo "$default_storage_classes" | tr '\n' ' ' | sed 's/[[:space:]]$//')"
    print_status "$GREEN" "PASS: $DETAIL"

    if [[ "$default_count" -gt 1 ]]; then
        print_status "$YELLOW" "WARN: multiple default StorageClasses can make PVC binding unpredictable"
    fi
    return 0
}

check_cluster_resources() {
    print_section "Checking cluster GPU resources"

    local node_count
    node_count=$(kubectl_get nodes -l nvidia.com/gpu.present=true -o name | wc -l | tr -d ' ' || true)

    if [[ "$node_count" == "0" || -z "$node_count" ]]; then
        DETAIL="no nodes have label nvidia.com/gpu.present=true"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="found ${node_count} GPU node(s)"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_gpu_operator() {
    print_section "Checking NVIDIA GPU Operator"

    if kubectl_has_crd clusterpolicies.nvidia.com || kubectl_has_pods app=gpu-operator; then
        DETAIL="GPU Operator CRDs or pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="GPU Operator was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_argocd() {
    print_section "Checking Argo CD"

    if kubectl_has_crd applications.argoproj.io && kubectl_has_pods app.kubernetes.io/part-of=argocd; then
        DETAIL="Argo CD Application CRD and controller pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="Argo CD Application CRD or controller pods were not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_prometheus_operator() {
    print_section "Checking Prometheus Operator CRDs"

    local missing=()
    for crd in prometheuses.monitoring.coreos.com servicemonitors.monitoring.coreos.com prometheusrules.monitoring.coreos.com; do
        if ! kubectl_has_crd "$crd"; then
            missing+=("$crd")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        DETAIL="missing CRDs: ${missing[*]}"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="Prometheus, ServiceMonitor, and PrometheusRule CRDs are present"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_prometheus_instance() {
    print_section "Checking Prometheus instances"

    if kubectl_get prometheus -A --no-headers | grep -q .; then
        DETAIL="at least one Prometheus custom resource exists"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="no Prometheus custom resources were found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_dcgm_servicemonitor() {
    print_section "Checking DCGM ServiceMonitor"

    if kubectl_get servicemonitor -A --no-headers | grep -i 'dcgm' >/dev/null; then
        DETAIL="DCGM ServiceMonitor is present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="no DCGM ServiceMonitor was found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_loki() {
    print_section "Checking Loki"

    if kubectl_has_pods app.kubernetes.io/name=loki || kubectl_get svc -A --no-headers | grep -i 'loki' >/dev/null; then
        DETAIL="Loki pods or services are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="Loki was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_fluentd() {
    print_section "Checking Fluentd"

    if kubectl_has_pods app.kubernetes.io/name=fluentd || kubectl_has_pods app=fluentd; then
        DETAIL="Fluentd pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="Fluentd was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_falco() {
    print_section "Checking Falco"

    if kubectl_has_pods app.kubernetes.io/name=falco || kubectl_get daemonset -A --no-headers | grep -i 'falco' >/dev/null; then
        DETAIL="Falco pods or DaemonSets are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="Falco was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_trivy() {
    print_section "Checking Trivy"

    if command -v trivy >/dev/null 2>&1; then
        DETAIL="$(trivy --version 2>/dev/null | head -n1)"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    if kubectl_has_crd vulnerabilityreports.aquasecurity.github.io || kubectl_has_pods app.kubernetes.io/name=trivy-operator; then
        DETAIL="Trivy Operator CRDs or pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="neither Trivy CLI nor Trivy Operator was found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_velero() {
    print_section "Checking Velero"

    if kubectl_has_crd backups.velero.io && { kubectl_has_pods deploy=velero || kubectl_has_pods app.kubernetes.io/name=velero; }; then
        DETAIL="Velero backup CRD and server pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="Velero backup CRD or server pods were not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_external_secrets() {
    print_section "Checking External Secrets Operator"

    if kubectl_has_crd externalsecrets.external-secrets.io && kubectl_has_pods app.kubernetes.io/name=external-secrets; then
        DETAIL="ExternalSecret CRD and controller pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="External Secrets Operator was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_grove_kai() {
    print_section "Checking Grove and KAI Scheduler"

    local missing=()
    for crd in podcliquesets.grove.io podcliques.grove.io podcliquescalinggroups.grove.io queues.scheduling.run.ai; do
        if ! kubectl_has_crd "$crd"; then
            missing+=("$crd")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        DETAIL="missing CRDs: ${missing[*]}"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    if ! kubectl_get queues dynamo >/dev/null; then
        DETAIL="Grove and KAI CRDs are present, but KAI queue 'dynamo' was not found"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="Grove CRDs, KAI Queue CRD, and queue 'dynamo' are present"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_lws_volcano() {
    print_section "Checking LWS and Volcano"

    local missing=()
    for crd in leaderworkersets.leaderworkerset.x-k8s.io queues.scheduling.volcano.sh podgroups.scheduling.volcano.sh; do
        if ! kubectl_has_crd "$crd"; then
            missing+=("$crd")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        DETAIL="missing CRDs: ${missing[*]}"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="LWS and Volcano CRDs are present"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_keda() {
    print_section "Checking KEDA"

    if kubectl_has_crd scaledobjects.keda.sh && kubectl_has_pods app.kubernetes.io/name=keda-operator; then
        DETAIL="KEDA ScaledObject CRD and operator pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="KEDA was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_opentelemetry() {
    print_section "Checking OpenTelemetry Operator"

    if kubectl_has_crd opentelemetrycollectors.opentelemetry.io && kubectl_has_pods app.kubernetes.io/name=opentelemetry-operator; then
        DETAIL="OpenTelemetryCollector CRD and operator pods are present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="OpenTelemetry Operator was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_network_policies() {
    print_section "Checking NetworkPolicies"

    local policy_count
    policy_count=$(kubectl_get networkpolicy -A --no-headers | wc -l | tr -d ' ' || true)

    if [[ "$policy_count" == "0" || -z "$policy_count" ]]; then
        DETAIL="no Kubernetes NetworkPolicy resources were found"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="found ${policy_count} NetworkPolicy resource(s)"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_dynamo_crds() {
    print_section "Checking Dynamo CRDs"

    local missing=()
    for crd in dynamographdeployments.nvidia.com dynamographdeploymentrequests.nvidia.com dynamomodels.nvidia.com dynamographdeploymentscalingadapters.nvidia.com; do
        if ! kubectl_has_crd "$crd"; then
            missing+=("$crd")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        DETAIL="missing CRDs: ${missing[*]}"
        print_status "$RED" "FAIL: $DETAIL"
        return 1
    fi

    DETAIL="Dynamo serving, request, model, and scaling adapter CRDs are present"
    print_status "$GREEN" "PASS: $DETAIL"
    return 0
}

check_dynamo_webhooks() {
    print_section "Checking Dynamo webhooks"

    if kubectl_get validatingwebhookconfiguration -o name | grep -i 'dynamo' >/dev/null; then
        DETAIL="Dynamo validating webhook configuration is present"
        print_status "$GREEN" "PASS: $DETAIL"
        return 0
    fi

    DETAIL="Dynamo validating webhook configuration was not found"
    print_status "$RED" "FAIL: $DETAIL"
    return 1
}

check_for_name() {
    case "$1" in
        kubectl) run_check "kubectl Connectivity" check_kubectl ;;
        helm) run_check "Helm" check_helm ;;
        default-storage-class) run_check "Default StorageClass" check_default_storage_class ;;
        gpu-resources) run_check "Cluster GPU Resources" check_cluster_resources ;;
        gpu-operator) run_check "GPU Operator" check_gpu_operator ;;
        argocd) run_check "Argo CD" check_argocd ;;
        prometheus-operator) run_check "Prometheus Operator" check_prometheus_operator ;;
        prometheus-instance) run_check "Prometheus Instance" check_prometheus_instance ;;
        dcgm-servicemonitor) run_check "DCGM ServiceMonitor" check_dcgm_servicemonitor ;;
        loki) run_check "Loki" check_loki ;;
        fluentd) run_check "Fluentd" check_fluentd ;;
        falco) run_check "Falco" check_falco ;;
        trivy) run_check "Trivy" check_trivy ;;
        velero) run_check "Velero" check_velero ;;
        external-secrets) run_check "External Secrets" check_external_secrets ;;
        grove-kai) run_check "Grove and KAI Scheduler" check_grove_kai ;;
        lws-volcano) run_check "LWS and Volcano" check_lws_volcano ;;
        keda) run_check "KEDA" check_keda ;;
        opentelemetry) run_check "OpenTelemetry Operator" check_opentelemetry ;;
        network-policies) run_check "NetworkPolicies" check_network_policies ;;
        dynamo-crds) run_check "Dynamo CRDs" check_dynamo_crds ;;
        dynamo-webhooks) run_check "Dynamo Webhooks" check_dynamo_webhooks ;;
        *)
            echo "Unknown check name: $1" >&2
            exit 2
            ;;
    esac
}

run_checks() {
    local overall_exit_code=0
    local checks=()

    if [[ -n "$REQUIRE" ]]; then
        IFS=',' read -ra checks <<< "$REQUIRE"
    elif [[ "$PROFILE" == "production" ]]; then
        checks=(
            kubectl
            helm
            default-storage-class
            gpu-resources
            gpu-operator
            argocd
            prometheus-operator
            prometheus-instance
            dcgm-servicemonitor
            loki
            fluentd
            falco
            trivy
            velero
            external-secrets
            grove-kai
            network-policies
        )
    else
        checks=(
            kubectl
            default-storage-class
            gpu-resources
            gpu-operator
        )
    fi

    for check in "${checks[@]}"; do
        if ! check_for_name "$check"; then
            overall_exit_code=1
        fi
    done

    return "$overall_exit_code"
}

json_escape() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    value=${value//$'\r'/}
    printf '%s' "$value"
}

display_table_summary() {
    print_section "Pre-Deployment Check Summary"

    local passed=0
    local failed=0
    local i

    for ((i = 0; i < ${#CHECK_NAMES[@]}; i++)); do
        if [[ "${CHECK_STATUSES[$i]}" == "PASS" ]]; then
            print_status "$GREEN" "PASS: ${CHECK_NAMES[$i]} - ${CHECK_DETAILS[$i]}"
            passed=$((passed + 1))
        else
            print_status "$RED" "FAIL: ${CHECK_NAMES[$i]} - ${CHECK_DETAILS[$i]}"
            failed=$((failed + 1))
        fi
    done

    echo ""
    print_status "$BLUE" "Summary: $passed passed, $failed failed"

    if [[ "$failed" -eq 0 ]]; then
        print_status "$GREEN" "All pre-deployment checks passed."
    else
        print_status "$RED" "$failed pre-deployment check(s) failed."
    fi
}

display_json_summary() {
    local passed=0
    local failed=0
    local i

    for ((i = 0; i < ${#CHECK_NAMES[@]}; i++)); do
        if [[ "${CHECK_STATUSES[$i]}" == "PASS" ]]; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done

    printf '{\n'
    printf '  "profile": "%s",\n' "$(json_escape "$PROFILE")"
    printf '  "checks": [\n'
    for ((i = 0; i < ${#CHECK_NAMES[@]}; i++)); do
        printf '    {"name": "%s", "status": "%s", "detail": "%s"}' \
            "$(json_escape "${CHECK_NAMES[$i]}")" \
            "$(json_escape "${CHECK_STATUSES[$i]}")" \
            "$(json_escape "${CHECK_DETAILS[$i]}")"
        if [[ "$i" -lt $((${#CHECK_NAMES[@]} - 1)) ]]; then
            printf ','
        fi
        printf '\n'
    done
    printf '  ],\n'
    printf '  "summary": {"passed": %d, "failed": %d}\n' "$passed" "$failed"
    printf '}\n'
}

main() {
    parse_args "$@"
    print_header

    local overall_exit_code=0
    if ! run_checks; then
        overall_exit_code=1
    fi

    if [[ "$OUTPUT" == "json" ]]; then
        display_json_summary
    else
        display_table_summary
    fi

    exit "$overall_exit_code"
}

main "$@"
