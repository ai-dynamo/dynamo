#!/bin/bash
# discovery.sh - Local development helper for Kubernetes discovery
#
# Creates mock EndpointSlices that point to localhost, enabling local workers
# to be discovered through a real Kubernetes cluster.
#
# Usage:
#   ./discovery.sh create <worker-name> [port] [ready]    # Create an EndpointSlice
#   ./discovery.sh ready <worker-name> <true|false>       # Set readiness of a worker
#   ./discovery.sh ready-all <true|false>                 # Set readiness of all workers
#   ./discovery.sh delete <worker-name>                   # Delete a specific EndpointSlice
#   ./discovery.sh delete-all                             # Delete all test EndpointSlices
#   ./discovery.sh list                                   # List all test EndpointSlices
#
# Examples:
#   ./discovery.sh create vllm-decode                     # Create on port 9001, not ready
#   ./discovery.sh create vllm-decode 9002                # Create on port 9002, not ready
#   ./discovery.sh create vllm-decode 9001 true           # Create on port 9001, ready
#   ./discovery.sh ready vllm-decode true                 # Make worker ready
#   ./discovery.sh ready-all true                         # Make all workers ready
#
# The created EndpointSlice will have:
#   - Labels for Dynamo discovery (nvidia.com/dynamo-discovery-backend=kubernetes, etc.)
#   - Localhost test annotation (nvidia.com/dynamo-discovery-localhost=true)
#   - The system port set to your local port for metadata fetching
#   - A dummy placeholder IP (K8s doesn't allow loopback IPs in EndpointSlices)
#     The Rust code ignores this IP when the localhost annotation is present

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-discovery-3}"
LABEL_PREFIX="nvidia.com/dynamo"
TEST_LABEL="${LABEL_PREFIX}-discovery-test=true"
DEFAULT_PORT=9001

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate kubectl is available and connected
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    
    log_info "Connected to cluster: $(kubectl config current-context)"
}

# Create an EndpointSlice for a local worker
# Args: worker_name [port] [ready]
create_endpoint_slice() {
    local worker_name="$1"
    local local_port="${2:-$DEFAULT_PORT}"
    local ready="${3:-false}"
    
    # Normalize ready to boolean string
    if [[ "$ready" == "true" || "$ready" == "1" || "$ready" == "yes" ]]; then
        ready="true"
    else
        ready="false"
    fi
    
    local slice_name="dynamo-local-${worker_name}"
    local pod_name="local-${worker_name}-$(whoami)"
    
    # Use a dummy placeholder IP - Kubernetes doesn't allow loopback addresses in EndpointSlices.
    # Our Rust code ignores this IP when the localhost annotation is present and uses 127.0.0.1 instead.
    local placeholder_ip="10.255.255.1"
    
    log_info "Creating EndpointSlice: ${slice_name}"
    log_info "  Worker name: ${worker_name}"
    log_info "  Local port: ${local_port}"
    log_info "  Ready: ${ready}"
    log_info "  Pod name (for instance_id): ${pod_name}"
    
    # Create the EndpointSlice
    cat <<EOF | kubectl apply -f -
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: ${slice_name}
  namespace: ${NAMESPACE}
  labels:
    ${LABEL_PREFIX}-discovery-backend: kubernetes
    ${LABEL_PREFIX}-discovery-enabled: "true"
    ${LABEL_PREFIX}-discovery-test: "true"
    ${LABEL_PREFIX}-worker-name: "${worker_name}"
  annotations:
    ${LABEL_PREFIX}-discovery-localhost: "true"
    ${LABEL_PREFIX}-local-port: "${local_port}"
    description: "Local development EndpointSlice for worker ${worker_name}"
addressType: IPv4
ports:
  - name: system
    port: ${local_port}
    protocol: TCP
endpoints:
  - addresses:
      - "${placeholder_ip}"
    conditions:
      ready: ${ready}
      serving: ${ready}
      terminating: false
    targetRef:
      kind: Pod
      name: "${pod_name}"
      namespace: "${NAMESPACE}"
EOF

    log_success "EndpointSlice '${slice_name}' created (ready=${ready})"
    echo ""
    log_info "To verify: kubectl get endpointslice ${slice_name} -n ${NAMESPACE} -o yaml"
    log_info "Instance ID will be hash of: ${pod_name}"
}

# Set readiness of a specific worker
# Args: worker_name, ready (true|false)
set_ready() {
    local worker_name="$1"
    local ready="$2"
    local slice_name="dynamo-local-${worker_name}"
    
    # Normalize ready to boolean string
    if [[ "$ready" == "true" || "$ready" == "1" || "$ready" == "yes" ]]; then
        ready="true"
    else
        ready="false"
    fi
    
    log_info "Setting readiness of '${slice_name}' to ${ready}"
    
    if kubectl patch endpointslice "${slice_name}" -n "${NAMESPACE}" --type=json \
        -p="[{\"op\": \"replace\", \"path\": \"/endpoints/0/conditions/ready\", \"value\": ${ready}}, {\"op\": \"replace\", \"path\": \"/endpoints/0/conditions/serving\", \"value\": ${ready}}]" 2>/dev/null; then
        log_success "Worker '${worker_name}' is now ready=${ready}"
    else
        log_error "Failed to update '${slice_name}' - does it exist?"
        exit 1
    fi
}

# Set readiness of all test workers
# Args: ready (true|false)
set_ready_all() {
    local ready="$1"
    
    # Normalize ready to boolean string
    if [[ "$ready" == "true" || "$ready" == "1" || "$ready" == "yes" ]]; then
        ready="true"
    else
        ready="false"
    fi
    
    log_info "Setting readiness of all test workers to ${ready}"
    
    local slices=$(kubectl get endpointslice -n "${NAMESPACE}" -l "${TEST_LABEL}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
    
    if [[ -z "$slices" ]]; then
        log_warn "No test EndpointSlices found"
        return
    fi
    
    local count=0
    for slice_name in $slices; do
        if kubectl patch endpointslice "${slice_name}" -n "${NAMESPACE}" --type=json \
            -p="[{\"op\": \"replace\", \"path\": \"/endpoints/0/conditions/ready\", \"value\": ${ready}}, {\"op\": \"replace\", \"path\": \"/endpoints/0/conditions/serving\", \"value\": ${ready}}]" 2>/dev/null; then
            log_info "  Updated: ${slice_name}"
            count=$((count + 1))
        else
            log_warn "  Failed to update: ${slice_name}"
        fi
    done
    
    log_success "Updated ${count} EndpointSlice(s) to ready=${ready}"
}

# Delete a specific EndpointSlice
delete_endpoint_slice() {
    local worker_name="$1"
    local slice_name="dynamo-local-${worker_name}"
    
    log_info "Deleting EndpointSlice: ${slice_name}"
    
    if kubectl delete endpointslice "${slice_name}" -n "${NAMESPACE}" 2>/dev/null; then
        log_success "EndpointSlice '${slice_name}' deleted"
    else
        log_warn "EndpointSlice '${slice_name}' not found or already deleted"
    fi
}

# Delete all test EndpointSlices
delete_all() {
    log_info "Deleting all discovery test EndpointSlices in namespace '${NAMESPACE}'..."
    
    local count=$(kubectl get endpointslice -n "${NAMESPACE}" -l "${TEST_LABEL}" --no-headers 2>/dev/null | wc -l)
    
    if [[ "$count" -eq 0 ]]; then
        log_info "No test EndpointSlices found"
        return
    fi
    
    kubectl delete endpointslice -n "${NAMESPACE}" -l "${TEST_LABEL}"
    log_success "Deleted ${count} EndpointSlice(s)"
}

# List all test EndpointSlices
list_endpoint_slices() {
    log_info "Discovery test EndpointSlices in namespace '${NAMESPACE}':"
    echo ""
    
    kubectl get endpointslice -n "${NAMESPACE}" -l "${TEST_LABEL}" \
        -o custom-columns=\
'NAME:.metadata.name,'\
'WORKER:.metadata.labels.nvidia\.com/dynamo-worker-name,'\
'PORT:.ports[0].port,'\
'READY:.endpoints[0].conditions.ready,'\
'AGE:.metadata.creationTimestamp' \
        2>/dev/null || log_info "No test EndpointSlices found"
}

# Print usage
usage() {
    cat <<EOF
Usage: $0 <command> [args]

Commands:
  create <worker-name> [port] [ready]
      Create an EndpointSlice for a local worker.
      - worker-name: Name identifier for the worker (e.g., vllm-decode)
      - port: Local port where the worker's system server is running (default: ${DEFAULT_PORT})
      - ready: Whether the endpoint starts ready (default: false)

  ready <worker-name> <true|false>
      Set the readiness of a specific worker.

  ready-all <true|false>
      Set the readiness of all test workers.

  delete <worker-name>
      Delete the EndpointSlice for a specific worker.

  delete-all
      Delete all discovery test EndpointSlices in the namespace.

  list
      List all discovery test EndpointSlices.

Environment variables:
  NAMESPACE    Kubernetes namespace to use (default: discovery-3)

Examples:
  # Create a worker (not ready by default, port 9001)
  $0 create vllm-decode

  # Create a worker on a specific port
  $0 create vllm-decode 9002

  # Create a worker that starts ready
  $0 create vllm-decode 9001 true

  # Make a worker ready
  $0 ready vllm-decode true

  # Make a worker not ready
  $0 ready vllm-decode false

  # Make all workers ready
  $0 ready-all true

  # Make all workers not ready
  $0 ready-all false

  # Delete a worker
  $0 delete vllm-decode

  # Delete all test workers
  $0 delete-all

  # List all test workers
  $0 list

Workflow:
  1. Start your local worker with the appropriate DYN_SYSTEM_PORT
     POD_NAME=local-vllm-decode-\$(whoami) DYN_SYSTEM_PORT=9001 cargo run --bin ...

  2. Create the discovery EndpointSlice (not ready initially)
     $0 create vllm-decode

  3. When your worker is ready, mark it as ready
     $0 ready vllm-decode true

  4. Other workers in the cluster (or locally) will discover your local worker

EOF
}

# Main
main() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    check_kubectl
    
    case "$command" in
        create)
            if [[ $# -lt 1 ]]; then
                log_error "create requires at least 1 argument: <worker-name> [port] [ready]"
                echo ""
                usage
                exit 1
            fi
            create_endpoint_slice "$1" "${2:-}" "${3:-}"
            ;;
        ready)
            if [[ $# -ne 2 ]]; then
                log_error "ready requires 2 arguments: <worker-name> <true|false>"
                echo ""
                usage
                exit 1
            fi
            set_ready "$1" "$2"
            ;;
        ready-all)
            if [[ $# -ne 1 ]]; then
                log_error "ready-all requires 1 argument: <true|false>"
                echo ""
                usage
                exit 1
            fi
            set_ready_all "$1"
            ;;
        delete)
            if [[ $# -ne 1 ]]; then
                log_error "delete requires 1 argument: <worker-name>"
                echo ""
                usage
                exit 1
            fi
            delete_endpoint_slice "$1"
            ;;
        delete-all)
            delete_all
            ;;
        list)
            list_endpoint_slices
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            usage
            exit 1
            ;;
    esac
}

main "$@"
