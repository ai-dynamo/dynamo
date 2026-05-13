#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


################################################################################
# Prefill-Decode Disaggregation Test Script
#
# This script automates the complete testing of SGLang prefill-decode
# disaggregation with NIXL transfer backend on Intel XPU.
#
# Usage:
#   ./run_pd_disaggregation_test.sh [start|stop|test|full]
#
# Commands:
#   start  - Start prefill, decode servers and router
#   stop   - Stop all running services
#   test   - Run curl test (requires services to be running)
#   full   - Run complete test: start services, test, and display results
#
################################################################################

set -e

# Configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
PREFILL_PORT=30000
DECODE_PORT=30001
ROUTER_PORT=8001
BOOTSTRAP_PORT=12335
PROMETHEUS_PORT=29999

# Directories
LOG_DIR="/tmp/sglang_pd_test"
PREFILL_LOG="${LOG_DIR}/prefill.log"
DECODE_LOG="${LOG_DIR}/decode.log"
ROUTER_LOG="${LOG_DIR}/router.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_port() {
    local port=$1
    if netstat -tuln 2>/dev/null | grep -q ":${port} "; then
        return 0
    else
        return 1
    fi
}

wait_for_port() {
    local port=$1
    local service=$2
    local max_wait=60
    local count=0

    log_info "Waiting for $service to start on port $port..."
    while [ $count -lt $max_wait ]; do
        if check_port $port; then
            log_success "$service is ready on port $port"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    log_error "$service failed to start on port $port after ${max_wait}s"
    return 1
}

wait_for_server_ready() {
    local log_file=$1
    local service=$2
    local max_wait=120
    local count=0

    log_info "Waiting for $service to be ready..."
    while [ $count -lt $max_wait ]; do
        if grep -q "The server is fired up and ready to roll" "$log_file" 2>/dev/null; then
            log_success "$service is ready!"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    log_error "$service failed to become ready after ${max_wait}s"
    return 1
}

wait_for_router_ready() {
    local log_file=$1
    local max_wait=60
    local count=0

    log_info "Waiting for router to register workers..."
    while [ $count -lt $max_wait ]; do
        if grep -q "Workflow completed" "$log_file" 2>/dev/null; then
            log_success "Router is ready with workers registered!"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    log_error "Router failed to become ready after ${max_wait}s"
    return 1
}

################################################################################
# Stop Function
################################################################################

stop_services() {
    log_info "Stopping all services..."

    # Kill all SGLang and router processes
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true

    # Free ports
    for port in $ROUTER_PORT $PREFILL_PORT $DECODE_PORT $BOOTSTRAP_PORT; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done

    sleep 2

    # Verify all stopped
    if ps aux | grep -E "(sglang_router|sglang.launch_server)" | grep -v grep > /dev/null 2>&1; then
        log_warning "Some processes may still be running"
        ps aux | grep -E "(sglang_router|sglang.launch_server)" | grep -v grep
    else
        log_success "All services stopped successfully"
    fi
}

################################################################################
# Start Functions
################################################################################

start_prefill_server() {
    log_info "Starting Prefill Server (XPU Device 0, Port $PREFILL_PORT)..."

    # Check if already running
    if check_port $PREFILL_PORT; then
        log_warning "Prefill server already running on port $PREFILL_PORT"
        return 0
    fi

    # Start prefill server
    ZE_AFFINITY_MASK=0 UCX_POSIX_USE_PROC_LINK=n python -m sglang.launch_server \
        --model-path "$MODEL" \
        --trust-remote-code \
        --device xpu \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend nixl \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --host 0.0.0.0 \
        --port $PREFILL_PORT \
        --grammar-backend none \
        > "$PREFILL_LOG" 2>&1 &

    local prefill_pid=$!
    echo $prefill_pid > "${LOG_DIR}/prefill.pid"

    # Wait for server to be ready
    if wait_for_port $PREFILL_PORT "Prefill Server"; then
        if wait_for_server_ready "$PREFILL_LOG" "Prefill Server"; then
            log_success "Prefill Server started successfully (PID: $prefill_pid)"
            return 0
        fi
    fi

    log_error "Failed to start Prefill Server"
    log_error "Check logs at: $PREFILL_LOG"
    return 1
}

start_decode_server() {
    log_info "Starting Decode Server (XPU Device 1, Port $DECODE_PORT)..."

    # Check if already running
    if check_port $DECODE_PORT; then
        log_warning "Decode server already running on port $DECODE_PORT"
        return 0
    fi

    # Start decode server
    ZE_AFFINITY_MASK=1 UCX_POSIX_USE_PROC_LINK=n python -m sglang.launch_server \
        --model-path "$MODEL" \
        --trust-remote-code \
        --device xpu \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend nixl \
        --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
        --host 0.0.0.0 \
        --port $DECODE_PORT \
        --grammar-backend none \
        > "$DECODE_LOG" 2>&1 &

    local decode_pid=$!
    echo $decode_pid > "${LOG_DIR}/decode.pid"

    # Wait for server to be ready
    if wait_for_port $DECODE_PORT "Decode Server"; then
        if wait_for_server_ready "$DECODE_LOG" "Decode Server"; then
            log_success "Decode Server started successfully (PID: $decode_pid)"
            return 0
        fi
    fi

    log_error "Failed to start Decode Server"
    log_error "Check logs at: $DECODE_LOG"
    return 1
}

start_router() {
    log_info "Starting Router (Port $ROUTER_PORT)..."

    # Check if already running
    if check_port $ROUTER_PORT; then
        log_warning "Router already running on port $ROUTER_PORT"
        return 0
    fi

    # Start router
    python -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "http://127.0.0.1:$PREFILL_PORT" \
        --decode "http://127.0.0.1:$DECODE_PORT" \
        --host 0.0.0.0 \
        --port $ROUTER_PORT \
        --prometheus-port $PROMETHEUS_PORT \
        > "$ROUTER_LOG" 2>&1 &

    local router_pid=$!
    echo $router_pid > "${LOG_DIR}/router.pid"

    # Wait for router to be ready
    if wait_for_port $ROUTER_PORT "Router"; then
        if wait_for_router_ready "$ROUTER_LOG"; then
            log_success "Router started successfully (PID: $router_pid)"
            return 0
        fi
    fi

    log_error "Failed to start Router"
    log_error "Check logs at: $ROUTER_LOG"
    return 1
}

start_all_services() {
    log_info "Starting all services..."
    echo

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Start services in order
    if ! start_prefill_server; then
        log_error "Failed to start prefill server, aborting"
        return 1
    fi
    echo

    if ! start_decode_server; then
        log_error "Failed to start decode server, aborting"
        return 1
    fi
    echo

    if ! start_router; then
        log_error "Failed to start router, aborting"
        return 1
    fi
    echo

    log_success "All services started successfully!"
    echo
    show_status
}

################################################################################
# Test Function
################################################################################

run_test() {
    log_info "Running inference test..."
    echo

    # Check if router is running
    if ! check_port $ROUTER_PORT; then
        log_error "Router is not running on port $ROUTER_PORT"
        log_error "Start services first with: $0 start"
        return 1
    fi

    local test_prompt="The capital of France is"
    local max_tokens=32

    log_info "Test prompt: \"$test_prompt\""
    log_info "Max tokens: $max_tokens"
    echo

    # Run curl test
    local response
    if ! response=$(curl --silent --show-error --fail \
        --connect-timeout 5 \
        --max-time 60 \
        "http://127.0.0.1:$ROUTER_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"prompt\": \"$test_prompt\",
            \"max_tokens\": $max_tokens
        }"); then
        log_error "Inference request failed"
        return 1
    fi

    # Parse and display response
    echo -e "${GREEN}=== Response ===${NC}"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    echo

    # Extract text if possible
    local generated_text=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "")

    if [ -n "$generated_text" ]; then
        log_success "Generated text: $generated_text"
        echo
        log_success "Test completed successfully!"
        return 0
    else
        log_warning "Could not parse generated text from response"
        return 1
    fi
}

################################################################################
# Status Function
################################################################################

show_status() {
    echo -e "${BLUE}=== Service Status ===${NC}"
    echo

    # Check prefill server
    if check_port $PREFILL_PORT; then
        echo -e "${GREEN}✓${NC} Prefill Server: Running on port $PREFILL_PORT"
        if [ -f "${LOG_DIR}/prefill.pid" ]; then
            echo "  PID: $(cat ${LOG_DIR}/prefill.pid)"
        fi
        echo "  Log: $PREFILL_LOG"
    else
        echo -e "${RED}✗${NC} Prefill Server: Not running"
    fi
    echo

    # Check decode server
    if check_port $DECODE_PORT; then
        echo -e "${GREEN}✓${NC} Decode Server: Running on port $DECODE_PORT"
        if [ -f "${LOG_DIR}/decode.pid" ]; then
            echo "  PID: $(cat ${LOG_DIR}/decode.pid)"
        fi
        echo "  Log: $DECODE_LOG"
    else
        echo -e "${RED}✗${NC} Decode Server: Not running"
    fi
    echo

    # Check router
    if check_port $ROUTER_PORT; then
        echo -e "${GREEN}✓${NC} Router: Running on port $ROUTER_PORT"
        if [ -f "${LOG_DIR}/router.pid" ]; then
            echo "  PID: $(cat ${LOG_DIR}/router.pid)"
        fi
        echo "  Log: $ROUTER_LOG"
    else
        echo -e "${RED}✗${NC} Router: Not running"
    fi
    echo

    # Check bootstrap server
    if check_port $BOOTSTRAP_PORT; then
        echo -e "${GREEN}✓${NC} Bootstrap Server: Running on port $BOOTSTRAP_PORT"
    else
        echo -e "${RED}✗${NC} Bootstrap Server: Not running"
    fi
    echo
}

################################################################################
# Full Test Function
################################################################################

run_full_test() {
    log_info "Running full test sequence..."
    echo

    # Stop any existing services
    stop_services
    echo

    # Start all services
    if ! start_all_services; then
        log_error "Failed to start services"
        return 1
    fi

    # Wait a bit for everything to stabilize
    log_info "Waiting for services to stabilize..."
    sleep 5
    echo

    # Run test
    if run_test; then
        echo
        log_success "Full test completed successfully!"
        return 0
    else
        log_error "Test failed"
        return 1
    fi
}

################################################################################
# Main Function
################################################################################

show_usage() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start   - Start all services (prefill, decode, router)"
    echo "  stop    - Stop all services"
    echo "  test    - Run inference test"
    echo "  status  - Show service status"
    echo "  full    - Run complete test (stop, start, test)"
    echo "  logs    - Tail all service logs"
    echo
    echo "Examples:"
    echo "  $0 start       # Start all services"
    echo "  $0 test        # Run test"
    echo "  $0 full        # Full automated test"
    echo "  $0 stop        # Stop all services"
    echo
}

tail_logs() {
    log_info "Tailing logs (Ctrl+C to exit)..."
    echo

    if [ ! -f "$PREFILL_LOG" ] || [ ! -f "$DECODE_LOG" ] || [ ! -f "$ROUTER_LOG" ]; then
        log_error "Log files not found. Start services first."
        return 1
    fi

    tail -f "$PREFILL_LOG" "$DECODE_LOG" "$ROUTER_LOG"
}

main() {
    local command="${1:-}"

    if [ -z "$command" ]; then
        show_usage
        exit 1
    fi

    case "$command" in
        start)
            start_all_services
            ;;
        stop)
            stop_services
            ;;
        test)
            run_test
            ;;
        status)
            show_status
            ;;
        full)
            run_full_test
            ;;
        logs)
            tail_logs
            ;;
        -h|--help|help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
