#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration - all set via command line arguments
NAMESPACE=""
MODEL="Qwen/Qwen3-0.6B"
ISL=2000
STD=10
OSL=256
OUTPUT_DIR="./benchmarks/results"

# Input configurations stored as associative arrays
declare -A INPUT_LABELS
declare -A INPUT_VALUES

# Flags
VERBOSE=false

show_help() {
    cat << EOF
Dynamo Benchmark Runner

This script is a wrapper around genai-perf that benchmarks Dynamo LLM deployments and
plots the results in an easy-to-use way. It supports comparing multiple DynamoGraphDeployments
or endpoints with custom labels defined by you.

The client runs locally and connects to your deployments/endpoints for benchmarking.

USAGE:
    $0 --namespace NAMESPACE --input <label>=<manifest_or_endpoint> [--input <label>=<manifest_or_endpoint>]... [OPTIONS]

REQUIRED:
    -n, --namespace NAMESPACE           Kubernetes namespace
    --input <label>=<manifest_path_or_endpoint>  Benchmark input with custom label
                                          - <label>: becomes the name/label in plots
                                          - <manifest_path_or_endpoint>: either a DynamoGraphDeployment manifest or HTTP endpoint URL
                                          Can be specified multiple times for comparisons

OPTIONS:
    -h, --help                    Show this help message
    -m, --model MODEL             Model name for GenAI-Perf configuration and logging (default: Qwen/Qwen3-0.6B)
                                  NOTE: This must match the model configured in your deployment manifests and the model deployed in any endpoints.
    -i, --isl LENGTH              Input sequence length (default: $ISL)
    -s, --std STDDEV              Input sequence standard deviation (default: $STD)
    -o, --osl LENGTH              Output sequence length (default: $OSL)
    -d, --output-dir DIR          Output directory (default: $OUTPUT_DIR)
    --verbose                     Enable verbose output

EXAMPLES:
    # Compare Dynamo deployments of a single backend
    $0 --namespace \$NAMESPACE \\
       --input agg=components/backends/vllm/deploy/agg.yaml \\
       --input disagg=components/backends/vllm/deploy/disagg.yaml

    # Compare different backend types (vLLM vs TensorRT-LLM)
    $0 --namespace \$NAMESPACE \\
       --input vllm-agg=components/backends/vllm/deploy/agg.yaml \\
       --input trtllm-agg=components/backends/trtllm/deploy/agg.yaml

    # Compare Dynamo deployment vs external endpoint
    $0 --namespace \$NAMESPACE \\
       --input dynamo=components/backends/vllm/deploy/disagg.yaml \\
       --input external=http://localhost:8000

    # Compare multiple different configurations (vLLM, TensorRT-LLM, SGLang)
    $0 --namespace \$NAMESPACE \\
       --input vllm-agg=components/backends/vllm/deploy/agg.yaml \\
       --input trtllm-disagg=components/backends/trtllm/deploy/disagg.yaml \\
       --input existing-sglang=http://localhost:8000

    # Benchmark a single Dynamo deployment
    $0 --namespace \$NAMESPACE \\
       --input my-setup=components/backends/vllm/deploy/disagg.yaml

    # Benchmark single external endpoint
    $0 --namespace \$NAMESPACE \\
       --input production=http://localhost:8000

DEPLOYMENT TYPES:
    - DynamoGraphDeployment: Supports various Dynamo deployment configurations including:
      * Aggregated deployments (prefill and decode together)
      * Disaggregated deployments (prefill and decode separate)
      * Router deployments
      * Planner deployments
      * And other Dynamo configurations
    - External Endpoints: For comparing against non-Dynamo backends

NOTE:
    - Only DynamoGraphDeployment manifests are supported for automatic deployment.
    - To benchmark non-Dynamo backends (vLLM, TensorRT-LLM, SGLang, etc.), deploy them
      manually following their Kubernetes deployment guides, expose a port (i.e. via port-forward),
      and use the endpoint option.
    - For Dynamo deployment setup, follow the main installation guide at docs/guides/dynamo_deploy/installation_guide.md
      to install the platform, then use setup_benchmarking_resources.sh for benchmarking resources.
    - The --model flag configures GenAI-Perf and should match what's configured in your deployment manifests and endpoints.
    - Only one model can be benchmarked at a time across all inputs.

EOF
}

parse_input() {
    local input_arg="$1"

    # Basic format validation: must contain exactly one '=' character
    if [[ ! "$input_arg" =~ ^[^=]+=[^=]+$ ]]; then
        echo "ERROR: Invalid input format. Expected: <label>=<manifest_path_or_endpoint>" >&2
        echo "Got: $input_arg" >&2
        echo "Format must be: key=value with exactly one '=' character" >&2
        exit 1
    fi

    # Split on the first '=' character
    local label="${input_arg%%=*}"
    local value="${input_arg#*=}"

    # Basic validation - detailed validation will be done in Python
    if [[ -z "$label" ]]; then
        echo "ERROR: Label cannot be empty in input: $input_arg" >&2
        exit 1
    fi

    if [[ -z "$value" ]]; then
        echo "ERROR: Value cannot be empty in input: $input_arg" >&2
        exit 1
    fi

    # Check for duplicate labels
    if [[ -n "${INPUT_LABELS[$label]:-}" ]]; then
        echo "ERROR: Duplicate label '$label' found. Each label must be unique." >&2
        exit 1
    fi

    # Store the input
    INPUT_LABELS["$label"]=1
    INPUT_VALUES["$label"]="$value"

    echo "Added input: $label -> $value"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -m|--model)
                MODEL="$2"
                shift 2
                ;;
            -i|--isl)
                ISL="$2"
                shift 2
                ;;
            -s|--std)
                STD="$2"
                shift 2
                ;;
            -o|--osl)
                OSL="$2"
                shift 2
                ;;
            -d|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --input)
                parse_input "$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Use --help for usage information." >&2
                exit 1
                ;;
        esac
    done
}

validate_config() {
    local errors=()

    if [[ -z "$NAMESPACE" ]]; then
        errors+=("--namespace is required")
    fi

    # Check that at least one input is specified
    if [[ ${#INPUT_LABELS[@]} -eq 0 ]]; then
        errors+=("At least one --input must be specified")
    fi

    if [[ ${#errors[@]} -gt 0 ]]; then
        echo "ERROR: Missing required arguments:" >&2
        for error in "${errors[@]}"; do
            echo "  $error" >&2
        done
        echo "Use --help for usage information." >&2
        exit 1
    fi

    # Validate that specified files exist and endpoints are valid URLs
    for label in "${!INPUT_VALUES[@]}"; do
        local value="${INPUT_VALUES[$label]}"

        # Check if it's a URL (starts with http:// or https://)
        if [[ "$value" =~ ^https?:// ]]; then
            echo "Input '$label': endpoint $value"
        else
            # It should be a file path - validate it exists
            if [[ ! -f "$value" ]]; then
                echo "ERROR: Manifest file not found for input '$label': $value" >&2
                exit 1
            fi
            echo "Input '$label': manifest $value"
        fi
    done

    if [[ ! "$ISL" =~ ^[0-9]+$ ]] || [[ "$ISL" -le 0 ]]; then
        echo "ERROR: ISL must be a positive integer, got: $ISL" >&2
        exit 1
    fi

    if [[ ! "$OSL" =~ ^[0-9]+$ ]] || [[ "$OSL" -le 0 ]]; then
        echo "ERROR: OSL must be a positive integer, got: $OSL" >&2
        exit 1
    fi

    if [[ ! "$STD" =~ ^[0-9]+$ ]] || [[ "$STD" -lt 0 ]]; then
        echo "ERROR: STD must be a non-negative integer, got: $STD" >&2
        exit 1
    fi
}

print_config() {
    echo "=== Benchmark Configuration ==="
    echo "Namespace:              $NAMESPACE"
    echo "Model:                  $MODEL"
    echo "Input Sequence Length:  $ISL tokens"
    echo "Output Sequence Length: $OSL tokens"
    echo "Sequence Std Dev:       $STD tokens"
    echo "Output Directory:       $OUTPUT_DIR"
    echo ""
    echo "Benchmark Inputs:"

    for label in "${!INPUT_VALUES[@]}"; do
        local value="${INPUT_VALUES[$label]}"
        if [[ "$value" =~ ^https?:// ]]; then
            echo "  $label: endpoint $value"
        else
            echo "  $label: manifest $value"
        fi
    done

    echo "==============================="
    echo
}

run_benchmark() {
    echo "🚀 Starting benchmark workflow..."

    # Change to dynamo root directory
    cd "$DYNAMO_ROOT"

    local cmd=(
        python3 -u -m benchmarks.utils.benchmark
        --namespace "$NAMESPACE"
        --model "$MODEL"
        --isl "$ISL"
        --std "$STD"
        --osl "$OSL"
        --output-dir "$OUTPUT_DIR"
    )

    # Add all input arguments
    for label in "${!INPUT_VALUES[@]}"; do
        local value="${INPUT_VALUES[$label]}"
        cmd+=(--input "$label=$value")
    done

    if [[ "$VERBOSE" == "true" ]]; then
        echo "Executing: ${cmd[*]}"
    fi

    if ! "${cmd[@]}"; then
        echo "❌ Benchmark failed!" >&2
        exit 1
    fi

    echo "✅ Benchmark completed successfully!"
}

generate_plots() {
    echo "📊 Generating performance plots..."

    cd "$DYNAMO_ROOT"

    local plot_cmd=(
        python3 -m benchmarks.utils.plot
        --data-dir "$OUTPUT_DIR"
    )

    if [[ "$VERBOSE" == "true" ]]; then
        echo "Executing: ${plot_cmd[*]}"
    fi

    if ! "${plot_cmd[@]}"; then
        echo "⚠️  Plot generation failed, but benchmark data is still available" >&2
        return 1
    fi

    echo "✅ Plots generated successfully!"
    echo "📁 Results available at: $OUTPUT_DIR"
    echo "📈 Plots available at: $OUTPUT_DIR/plots"
}

main() {
    trap cleanup EXIT

    parse_args "$@"
    validate_config
    print_config
    if [[ "$VERBOSE" == "true" ]]; then
        export DYNAMO_VERBOSE=true
    fi

    local start_time
    start_time=$(date +%s)

    run_benchmark
    generate_plots

    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))

    echo
    echo "🎉 All done!"
    echo "⏱️  Total time: ${duration}s"
    echo "📁 Results: $OUTPUT_DIR"
    echo "📊 Plots: $OUTPUT_DIR/plots"
}

cleanup() {
    if [[ $? -ne 0 ]]; then
        echo "❌ Script failed. Check logs above for details." >&2
    fi
}

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'cleanup $?' EXIT
    main "$@"
fi
