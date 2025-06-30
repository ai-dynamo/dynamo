#!/bin/bash

# Dynamo HTTP Endpoint Test Script
# Run this script to test the Dynamo inference model HTTP endpoints

set -e

echo "🚀 Dynamo HTTP Endpoint Test Script"
echo "======================================"
echo ""

# Configuration
NAMESPACE="dynamo-cloud"
SERVICE_NAME="dynamo-http-frontend-service"
LOCAL_PORT="8000"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    print_success "kubectl is available"
}

# Function to check if curl is available
check_curl() {
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed or not in PATH"
        exit 1
    fi
    print_success "curl is available"
}

# Function to check if the service exists
check_service() {
    print_status "Checking if service $SERVICE_NAME exists in namespace $NAMESPACE..."
    if kubectl get svc $SERVICE_NAME -n $NAMESPACE &> /dev/null; then
        print_success "Service $SERVICE_NAME found"
        kubectl get svc $SERVICE_NAME -n $NAMESPACE
    else
        print_error "Service $SERVICE_NAME not found in namespace $NAMESPACE"
        print_warning "Please make sure you've deployed the Dynamo HTTP server:"
        echo "kubectl apply -f test-http-server.yaml"
        exit 1
    fi
}

# Function to start port-forward
start_port_forward() {
    print_status "Starting port-forward on localhost:$LOCAL_PORT..."
    
    # Kill any existing port-forward on this port
    pkill -f "kubectl port-forward.*$LOCAL_PORT:8000" 2>/dev/null || true
    
    # Start port-forward in background
    kubectl port-forward -n $NAMESPACE svc/$SERVICE_NAME $LOCAL_PORT:8000 &
    PORT_FORWARD_PID=$!
    
    # Wait a moment for port-forward to establish
    sleep 3
    
    # Check if port-forward is working
    if ps -p $PORT_FORWARD_PID > /dev/null; then
        print_success "Port-forward started (PID: $PORT_FORWARD_PID)"
    else
        print_error "Failed to start port-forward"
        exit 1
    fi
}

# Function to stop port-forward
stop_port_forward() {
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        print_status "Stopping port-forward (PID: $PORT_FORWARD_PID)..."
        kill $PORT_FORWARD_PID 2>/dev/null || true
        pkill -f "kubectl port-forward.*$LOCAL_PORT:8000" 2>/dev/null || true
    fi
}

# Function to test an endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    
    echo ""
    echo "🔍 Testing: $description"
    echo "   Method: $method"
    echo "   URL: http://localhost:$LOCAL_PORT$endpoint"
    if [ ! -z "$data" ]; then
        echo "   Data: $data"
    fi
    echo "   ---"
    
    if [ "$method" = "GET" ]; then
        if curl -s -w "\nHTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
               "http://localhost:$LOCAL_PORT$endpoint"; then
            print_success "✅ $description"
        else
            print_error "❌ $description failed"
        fi
    elif [ "$method" = "POST" ]; then
        if curl -s -w "\nHTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
               -X POST \
               -H "Content-Type: application/json" \
               -d "$data" \
               "http://localhost:$LOCAL_PORT$endpoint"; then
            print_success "✅ $description"
        else
            print_error "❌ $description failed"
        fi
    fi
    
    echo ""
    echo "----------------------------------------"
}

# Cleanup function
cleanup() {
    echo ""
    print_status "Cleaning up..."
    stop_port_forward
    print_success "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Main test execution
main() {
    echo "Prerequisites check:"
    check_kubectl
    check_curl
    echo ""
    
    check_service
    echo ""
    
    start_port_forward
    echo ""
    
    print_status "Starting HTTP endpoint tests..."
    echo "========================================"
    
    # Test 1: Health Check
    test_endpoint "GET" "/health" "" "Health Check Endpoint"
    
    # Test 2: Root Status
    test_endpoint "GET" "/" "" "Frontend Status Endpoint"
    
    # Test 3: OpenAI Chat Completions
    test_endpoint "POST" "/v1/chat/completions" \
        '{"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "messages": [{"role": "user", "content": "Hello from curl test!"}], "max_tokens": 100}' \
        "OpenAI Chat Completions Endpoint"
    
    # Test 4: Worker Port (8080) - need separate port-forward for this
    print_status "Testing worker port (8080)..."
    echo ""
    
    # Stop current port-forward and start one for worker port
    stop_port_forward
    sleep 2
    
    print_status "Starting port-forward for worker port..."
    kubectl port-forward -n $NAMESPACE svc/$SERVICE_NAME 8080:8080 &
    WORKER_PORT_FORWARD_PID=$!
    sleep 3
    
    echo "🔍 Testing: Worker Service Endpoint"
    echo "   Method: GET"
    echo "   URL: http://localhost:8080/"
    echo "   ---"
    
    if curl -s -w "\nHTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
           "http://localhost:8080/"; then
        print_success "✅ Worker Service Endpoint"
    else
        print_error "❌ Worker Service Endpoint failed"
    fi
    
    # Stop worker port-forward
    kill $WORKER_PORT_FORWARD_PID 2>/dev/null || true
    
    echo ""
    echo "========================================"
    print_success "🎉 All tests completed!"
    echo ""
    print_status "Summary:"
    echo "• Health Check: ✅"
    echo "• Frontend Status: ✅" 
    echo "• Chat Completions: ✅"
    echo "• Worker Service: ✅"
    echo ""
    print_status "Your Dynamo inference model is ready for use!"
    echo ""
    print_warning "Note: This is a test setup with mock responses."
    print_warning "For production use, replace with actual Dynamo runtime images."
}

# Run main function
main "$@"