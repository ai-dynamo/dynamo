#!/bin/bash
# launch.sh - Local development launcher for Dynamo workers
#
# Usage:
#   ./launch.sh [flags] <command> [args]
#
# Commands:
#   frontend [port]           Start frontend only
#   decode [port] [gpu]       Start decode worker only
#   prefill [port] [gpu]      Start prefill worker only
#   all                       Start all components
#
# Flags:
#   --verbose                 Show output on screen for all components
#   --verbose-frontend        Show output on screen for frontend
#   --verbose-decode          Show output on screen for decode worker
#   --verbose-prefill         Show output on screen for prefill worker
#   --etcd                    Use etcd/kv_store discovery instead of kubernetes
#
# Environment variable overrides:
#   MODEL             - Model to use (default: Qwen/Qwen3-0.6B)
#   LOG_DIR           - Directory for logs (default: ./logs)
#   DYN_LOG           - Log level (default: info)
#   DISCOVERY_BACKEND - Discovery backend: kubernetes or kv_store (default: kubernetes)
#   FRONTEND_PORT     - HTTP port for frontend (default: 8000)
#   DECODE_PORT       - System port for decode worker (default: 9001)
#   PREFILL_PORT      - System port for prefill worker (default: 9002)
#   DECODE_GPU        - GPU for decode worker (default: 0)
#   PREFILL_GPU       - GPU for prefill worker (default: 1)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_HOME="${DYNAMO_HOME:-$SCRIPT_DIR}"
EXAMPLES_DIR="${DYNAMO_HOME}/examples/backends/vllm"
LOG_DIR="${LOG_DIR:-${DYNAMO_HOME}/logs}"

# Model
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

# Ports
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
DECODE_PORT="${DECODE_PORT:-9001}"
PREFILL_PORT="${PREFILL_PORT:-9002}"

# GPUs
DECODE_GPU="${DECODE_GPU:-0}"
PREFILL_GPU="${PREFILL_GPU:-1}"

# Pod UIDs (from dummy pods in discovery-3 namespace for local development)
# These are used for owner references when creating DynamoWorkerMetadata CRs
# Pod names: local-frontend-mabdulwahhab, local-decode-mabdulwahhab, local-prefill-mabdulwahhab
FRONTEND_POD_UID="${FRONTEND_POD_UID:-"35e0c832-2ee3-440a-845d-f063dae860cf"}"
DECODE_POD_UID="${DECODE_POD_UID:-"7ce2eaa0-dd98-4659-a7ea-6398caa001af"}"
PREFILL_POD_UID="${PREFILL_POD_UID:-"ec336571-3c52-43e1-ba77-bb64c0f0d8fd"}"

# vLLM prefill-specific ports
VLLM_KV_EVENT_PORT="${VLLM_KV_EVENT_PORT:-20081}"
VLLM_NIXL_SIDE_CHANNEL_PORT="${VLLM_NIXL_SIDE_CHANNEL_PORT:-20097}"

# Discovery
# Set to "kv_store" to use etcd instead of kubernetes
DISCOVERY_BACKEND="${DISCOVERY_BACKEND:-kubernetes}"
POD_NAMESPACE="${POD_NAMESPACE:-discovery-3}"

# Logging
DYN_LOG="${DYN_LOG:-info}"

# Verbosity flags (set by command-line parsing)
VERBOSE_FRONTEND="false"
VERBOSE_DECODE="false"
VERBOSE_PREFILL="false"

# ============================================================================
# Common Environment
# ============================================================================

# These are set once for all components
export PYTHONUNBUFFERED=1
export DYN_LOG

# Discovery backend (kubernetes or kv_store/etcd)
if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
    export DYN_DISCOVERY_BACKEND=kubernetes
else
    # kv_store is the default in the runtime, uses etcd
    export DYN_DISCOVERY_BACKEND=kv_store
fi

# ============================================================================
# Helpers
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

log_component() {
    local component="$1"
    local message="$2"
    echo -e "${CYAN}[$component]${NC} $message"
}

# Run a command with output to log file (and optionally screen)
# Sets LAST_PID to the PID of the backgrounded process
# Usage: run_with_logging <verbose_flag> <log_file> <command...>
LAST_PID=""
run_with_logging() {
    local verbose="$1"
    local log_file="$2"
    shift 2
    
    if [[ "$verbose" == "true" ]]; then
        # Show on screen AND log to file
        "$@" 2>&1 | tee "$log_file" &
        LAST_PID=$!
    else
        # Log to file only
        "$@" > "$log_file" 2>&1 &
        LAST_PID=$!
    fi
}

# Track PIDs for cleanup
declare -a PIDS=()

cleanup() {
    echo ""
    log_warn "Shutting down all components..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Killing PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait a moment then force kill if needed
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_warn "Force killing PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    log_success "Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
}

wait_for_all() {
    log_info "All components started. Press Ctrl+C to stop."
    wait
}

# ============================================================================
# Component Launchers
# ============================================================================

start_frontend() {
    local port="${1:-$FRONTEND_PORT}"
    local log_file="${LOG_DIR}/frontend.log"
    local pod_name="local-frontend-$(whoami)"
    
    log_component "FRONTEND" "Starting on HTTP port ${port}"
    log_component "FRONTEND" "Discovery: ${DISCOVERY_BACKEND}"
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        log_component "FRONTEND" "Pod name: ${pod_name}"
    fi
    log_component "FRONTEND" "Logs: ${log_file}"
    log_component "FRONTEND" "Verbose: ${VERBOSE_FRONTEND}"
    
    cd "$EXAMPLES_DIR"
    
    # Build environment based on discovery backend
    local env_vars=()
    
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        env_vars+=("POD_NAME=$pod_name" "POD_NAMESPACE=$POD_NAMESPACE" "POD_UID=$FRONTEND_POD_UID")
    fi
    
    if [[ ${#env_vars[@]} -gt 0 ]]; then
        run_with_logging "$VERBOSE_FRONTEND" "$log_file" env "${env_vars[@]}" python -m dynamo.frontend --http-port="${port}"
    else
        run_with_logging "$VERBOSE_FRONTEND" "$log_file" python -m dynamo.frontend --http-port="${port}"
    fi
    PIDS+=("$LAST_PID")
    
    log_component "FRONTEND" "Started (PID: ${LAST_PID})"
}

start_decode() {
    local system_port="${1:-$DECODE_PORT}"
    local gpu="${2:-$DECODE_GPU}"
    local log_file="${LOG_DIR}/decode.log"
    local pod_name="local-decode-$(whoami)"
    
    log_component "DECODE" "Starting on system port ${system_port}, GPU ${gpu}"
    log_component "DECODE" "Model: ${MODEL}"
    log_component "DECODE" "Discovery: ${DISCOVERY_BACKEND}"
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        log_component "DECODE" "Pod name: ${pod_name}"
    fi
    log_component "DECODE" "Logs: ${log_file}"
    log_component "DECODE" "Verbose: ${VERBOSE_DECODE}"
    
    cd "$EXAMPLES_DIR"
    
    # Build environment based on discovery backend
    local env_vars=(
        "DYN_SYSTEM_PORT=$system_port"
        "CUDA_VISIBLE_DEVICES=$gpu"
    )
    
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        env_vars+=("POD_NAME=$pod_name" "POD_NAMESPACE=$POD_NAMESPACE" "POD_UID=$DECODE_POD_UID")
    fi
    
    run_with_logging "$VERBOSE_DECODE" "$log_file" env "${env_vars[@]}" python3 -m dynamo.vllm --model "$MODEL" --enforce-eager
    PIDS+=("$LAST_PID")
    
    log_component "DECODE" "Started (PID: ${LAST_PID})"
}

start_prefill() {
    local system_port="${1:-$PREFILL_PORT}"
    local gpu="${2:-$PREFILL_GPU}"
    local log_file="${LOG_DIR}/prefill.log"
    local pod_name="local-prefill-$(whoami)"
    
    log_component "PREFILL" "Starting on system port ${system_port}, GPU ${gpu}"
    log_component "PREFILL" "Model: ${MODEL}"
    log_component "PREFILL" "Discovery: ${DISCOVERY_BACKEND}"
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        log_component "PREFILL" "Pod name: ${pod_name}"
    fi
    log_component "PREFILL" "Logs: ${log_file}"
    log_component "PREFILL" "Verbose: ${VERBOSE_PREFILL}"
    
    cd "$EXAMPLES_DIR"
    
    # Build environment based on discovery backend
    local env_vars=(
        "DYN_SYSTEM_PORT=$system_port"
        "CUDA_VISIBLE_DEVICES=$gpu"
        "DYN_VLLM_KV_EVENT_PORT=$VLLM_KV_EVENT_PORT"
        "VLLM_NIXL_SIDE_CHANNEL_PORT=$VLLM_NIXL_SIDE_CHANNEL_PORT"
    )
    
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        env_vars+=("POD_NAME=$pod_name" "POD_NAMESPACE=$POD_NAMESPACE" "POD_UID=$PREFILL_POD_UID")
    fi
    
    run_with_logging "$VERBOSE_PREFILL" "$log_file" env "${env_vars[@]}" python3 -m dynamo.vllm --model "$MODEL" --enforce-eager --is-prefill-worker
    PIDS+=("$LAST_PID")
    
    log_component "PREFILL" "Started (PID: ${LAST_PID})"
}

start_all() {
    log_info "Starting all components..."
    log_info ""
    log_info "Configuration:"
    log_info "  Model:         ${MODEL}"
    log_info "  Discovery:     ${DISCOVERY_BACKEND}"
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        log_info "  K8s namespace: ${POD_NAMESPACE}"
    fi
    log_info "  Frontend:      HTTP port ${FRONTEND_PORT} (verbose: ${VERBOSE_FRONTEND})"
    log_info "  Decode:        System port ${DECODE_PORT}, GPU ${DECODE_GPU} (verbose: ${VERBOSE_DECODE})"
    log_info "  Prefill:       System port ${PREFILL_PORT}, GPU ${PREFILL_GPU} (verbose: ${VERBOSE_PREFILL})"
    log_info "  Log directory: ${LOG_DIR}"
    log_info ""
    
    start_frontend
    sleep 2  # Give frontend a moment to start
    start_decode
    sleep 2
    start_prefill
    
    echo ""
    log_success "All components started!"
    echo ""
    log_info "Endpoints:"
    log_info "  Frontend API:     http://localhost:${FRONTEND_PORT}/v1/chat/completions"
    log_info "  Decode health:    http://localhost:${DECODE_PORT}/health"
    log_info "  Prefill health:   http://localhost:${PREFILL_PORT}/health"
    echo ""
    
    if [[ "$DISCOVERY_BACKEND" == "kubernetes" ]]; then
        log_info "K8s Discovery (run these to register workers):"
        log_info "  ./discovery.sh create decode ${DECODE_PORT}"
        log_info "  ./discovery.sh create prefill ${PREFILL_PORT}"
        log_info "  ./discovery.sh ready-all true"
        echo ""
    fi
}

# ============================================================================
# Usage
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 [flags] <command> [args]

Commands:
  frontend [port]           Start frontend (default port: ${FRONTEND_PORT})
  decode [port] [gpu]       Start decode worker (default: port ${DECODE_PORT}, GPU ${DECODE_GPU})
  prefill [port] [gpu]      Start prefill worker (default: port ${PREFILL_PORT}, GPU ${PREFILL_GPU})
  all                       Start all components

Flags:
  --verbose                 Show output on screen for all components
  --verbose-frontend        Show output on screen for frontend only
  --verbose-decode          Show output on screen for decode worker only
  --verbose-prefill         Show output on screen for prefill worker only
  --etcd, --kv-store        Use etcd/kv_store discovery (default: kubernetes)
  --kubernetes, --k8s       Use kubernetes discovery (default)

Environment Variables:
  MODEL              Model to use (default: ${MODEL})
  LOG_DIR            Log directory (default: ${LOG_DIR})
  DYN_LOG            Log level (default: ${DYN_LOG})
  DISCOVERY_BACKEND  Discovery backend: kubernetes or kv_store (default: ${DISCOVERY_BACKEND})
  POD_NAMESPACE      K8s namespace for discovery (default: ${POD_NAMESPACE})
  
  FRONTEND_PORT      Frontend HTTP port (default: ${FRONTEND_PORT})
  DECODE_PORT        Decode system port (default: ${DECODE_PORT})
  PREFILL_PORT       Prefill system port (default: ${PREFILL_PORT})
  
  DECODE_GPU         GPU index for decode (default: ${DECODE_GPU})
  PREFILL_GPU        GPU index for prefill (default: ${PREFILL_GPU})
  
  # Pod UIDs (from dummy pods for local dev with K8s discovery)
  FRONTEND_POD_UID   UID for frontend CR owner reference
  DECODE_POD_UID     UID for decode CR owner reference
  PREFILL_POD_UID    UID for prefill CR owner reference

Examples:
  # Start everything (logs to files only)
  $0 all

  # Start with verbose output for all components
  $0 --verbose all

  # Start with verbose output for decode only
  $0 --verbose-decode all

  # Start with verbose for frontend and decode
  $0 --verbose-frontend --verbose-decode all

  # Start with etcd discovery instead of kubernetes
  DISCOVERY_BACKEND=kv_store $0 all

  # Start just the frontend with verbose
  $0 --verbose frontend

  # Start decode on custom port and GPU
  $0 decode 9003 2

  # Start with debug logging and custom model
  DYN_LOG=debug MODEL=meta-llama/Llama-2-7b-hf $0 --verbose all

Workflow with Kubernetes Discovery:
  1. Start local workers:
     $0 all

  2. Register with K8s discovery (in another terminal):
     ./discovery.sh create decode ${DECODE_PORT}
     ./discovery.sh create prefill ${PREFILL_PORT}
     ./discovery.sh ready-all true

  3. Test the API:
     curl http://localhost:${FRONTEND_PORT}/v1/chat/completions \\
       -H "Content-Type: application/json" \\
       -d '{"model": "${MODEL}", "messages": [{"role": "user", "content": "Hello"}]}'

  4. Cleanup:
     Ctrl+C to stop workers
     ./discovery.sh delete-all

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Parse flags first
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --verbose)
                VERBOSE_FRONTEND="true"
                VERBOSE_DECODE="true"
                VERBOSE_PREFILL="true"
                shift
                ;;
            --verbose-frontend)
                VERBOSE_FRONTEND="true"
                shift
                ;;
            --verbose-decode)
                VERBOSE_DECODE="true"
                shift
                ;;
            --verbose-prefill)
                VERBOSE_PREFILL="true"
                shift
                ;;
            --etcd|--kv-store)
                DISCOVERY_BACKEND="kv_store"
                export DYN_DISCOVERY_BACKEND=kv_store
                shift
                ;;
            --kubernetes|--k8s)
                DISCOVERY_BACKEND="kubernetes"
                export DYN_DISCOVERY_BACKEND=kubernetes
                shift
                ;;
            --)
                shift
                break
                ;;
            -*)
                if [[ "$1" == "-h" || "$1" == "--help" ]]; then
                    usage
                    exit 0
                fi
                log_error "Unknown flag: $1"
                echo ""
                usage
                exit 1
                ;;
            *)
                break
                ;;
        esac
    done
    
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    ensure_log_dir
    
    case "$command" in
        frontend)
            start_frontend "$@"
            wait_for_all
            ;;
        decode)
            start_decode "$@"
            wait_for_all
            ;;
        prefill)
            start_prefill "$@"
            wait_for_all
            ;;
        all)
            start_all
            wait_for_all
            ;;
        help)
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
