#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AFD Unit Test Runner
# Usage: ./run_afd_tests.sh [options]
#   Options:
#     --docker    Run tests in Docker container
#     --cov       Run with coverage report
#     --quick     Stop on first failure
#     --help      Show this help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Parse arguments
USE_DOCKER=false
USE_COV=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --cov)
            USE_COV=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help)
            echo "AFD Unit Test Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --docker    Run tests in Docker container"
            echo "  --cov       Run with coverage report"
            echo "  --quick     Stop on first failure"
            echo "  --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set Python path
export PYTHONPATH="${DYNAMO_ROOT}/components/src:${PYTHONPATH:-}"

# Test file
TEST_FILE="components/src/dynamo/sglang/tests/test_afd.py"

echo "=============================================="
echo "AFD Unit Tests"
echo "=============================================="
echo "Dynamo root: ${DYNAMO_ROOT}"
echo "Python path: ${PYTHONPATH}"
echo ""

if [ "$USE_DOCKER" = true ]; then
    echo "Running tests in Docker..."
    cd "${DYNAMO_ROOT}/docker/afd-test"
    
    if [ "$USE_COV" = true ]; then
        docker-compose run --rm afd-test-cov
    elif [ "$QUICK_MODE" = true ]; then
        docker-compose run --rm afd-test-quick
    else
        docker-compose run --rm afd-test
    fi
else
    echo "Running tests locally..."
    
    # Check Python version
    echo "Python version:"
    python --version
    
    # Check dependencies
    echo ""
    echo "Checking dependencies..."
    pip install pytest pytest-asyncio pytest-cov numpy prometheus-client -q
    
    # Run tests
    echo ""
    cd "${DYNAMO_ROOT}"
    
    PYTEST_ARGS="-v ${TEST_FILE}"
    
    if [ "$USE_COV" = true ]; then
        PYTEST_ARGS="${PYTEST_ARGS} --cov=dynamo.sglang.afd_communication --cov=dynamo.sglang.afd_metrics --cov-report=term-missing"
    fi
    
    if [ "$QUICK_MODE" = true ]; then
        PYTEST_ARGS="-x ${PYTEST_ARGS}"
    fi
    
    pytest $PYTEST_ARGS
fi

echo ""
echo "=============================================="
echo "Tests completed!"
echo "=============================================="
