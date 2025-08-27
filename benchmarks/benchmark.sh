#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration - all set via command line arguments
NAMESPACE=""
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ISL=200
STD=10
OSL=200
OUTPUT_DIR="./benchmarks/results"
AGG_CONFIG=""
DISAGG_CONFIG=""
VANILLA_CONFIG=""
ENDPOINT=""

# Flags
VERBOSE=false

show_help() {
    cat << EOF
Dynamo Benchmark Runner

This script runs complete LLM performance benchmarks across aggregated, disaggregated,
and vanilla deployments, then generates performance plots.

USAGE:
    $0 --namespace NAMESPACE [--endpoint URL | --agg CONFIG] [--disagg CONFIG] [--vanilla CONFIG] [OPTIONS]

REQUIRED:
    -n, --namespace NAMESPACE     Kubernetes namespace

    Either:
    --endpoint URL                Existing endpoint URL to benchmark

    Or at least one of:
    --agg CONFIG                  Aggregated deployment manifest
    --disagg CONFIG               Disaggregated deployment manifest
    --vanilla CONFIG              Vanilla backend deployment manifest

OPTIONS:
    -h, --help                    Show this help message
    -m, --model MODEL             Model name (default: $MODEL)
    -i, --isl LENGTH              Input sequence length (default: $ISL)
    -s, --std STDDEV              Input sequence standard deviation (default: $STD)
    -o, --osl LENGTH              Output sequence length (default: $OSL)
    -d, --output-dir DIR          Output directory (default: $OUTPUT_DIR)
    --verbose                     Enable verbose output

EXAMPLES:
    # Basic benchmark with all deployment types
    $0 --namespace my-namespace \\
       --agg components/backends/vllm/deploy/agg.yaml \\
       --disagg components/backends/vllm/deploy/disagg.yaml \\
       --vanilla benchmarks/utils/templates/vanilla-vllm.yaml

    # Benchmark only aggregated deployment
    $0 --namespace my-namespace \\
       --agg components/backends/vllm/deploy/agg.yaml

    # Benchmark existing endpoint
    $0 --namespace my-namespace \\
       --endpoint "http://localhost:8000"

    # Custom model and sequence lengths
    $0 --namespace my-namespace \\
       --agg my-agg.yaml --disagg my-disagg.yaml --vanilla my-vanilla.yaml \\
       --model "meta-llama/Meta-Llama-3-8B" --isl 512 --osl 512

NOTE: Use deployment manifests configured for your desired model. The manifests
      determine which model is actually deployed and benchmarked.

EOF
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
            --agg)
                AGG_CONFIG="$2"
                shift 2
                ;;
            --disagg)
                DISAGG_CONFIG="$2"
                shift 2
                ;;
            --vanilla)
                VANILLA_CONFIG="$2"
                shift 2
                ;;
            --endpoint)
                ENDPOINT="$2"
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

    # Check mutual exclusivity between endpoint and deployment manifests
    has_endpoint=false
    has_deployments=false

    if [[ -n "$ENDPOINT" ]]; then
        has_endpoint=true
    fi

    if [[ -n "$AGG_CONFIG" || -n "$DISAGG_CONFIG" || -n "$VANILLA_CONFIG" ]]; then
        has_deployments=true
    fi

    if [[ "$has_endpoint" == "true" && "$has_deployments" == "true" ]]; then
        errors+=("--endpoint cannot be used together with --agg, --disagg, or --vanilla")
    fi

    if [[ "$has_endpoint" == "false" && "$has_deployments" == "false" ]]; then
        errors+=("Must specify either --endpoint OR at least one deployment type (--agg, --disagg, or --vanilla)")
    fi

    if [[ ${#errors[@]} -gt 0 ]]; then
        echo "ERROR: Missing required arguments:" >&2
        for error in "${errors[@]}"; do
            echo "  $error" >&2
        done
        echo "Use --help for usage information." >&2
        exit 1
    fi

    # Validate that specified configuration files exist
    if [[ -n "$AGG_CONFIG" && ! -f "$AGG_CONFIG" ]]; then
        echo "ERROR: Configuration file not found: $AGG_CONFIG" >&2
        exit 1
    fi

    if [[ -n "$DISAGG_CONFIG" && ! -f "$DISAGG_CONFIG" ]]; then
        echo "ERROR: Configuration file not found: $DISAGG_CONFIG" >&2
        exit 1
    fi

    if [[ -n "$VANILLA_CONFIG" && ! -f "$VANILLA_CONFIG" ]]; then
        echo "ERROR: Configuration file not found: $VANILLA_CONFIG" >&2
        exit 1
    fi

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

    if [[ -n "$ENDPOINT" ]]; then
        echo "Endpoint URL:           $ENDPOINT"
    else
        echo "Aggregated Config:      ${AGG_CONFIG:-"(not specified)"}"
        echo "Disaggregated Config:   ${DISAGG_CONFIG:-"(not specified)"}"
        echo "Vanilla Config:         ${VANILLA_CONFIG:-"(not specified)"}"
    fi

    echo "==============================="
    echo
}

run_benchmark() {
    echo "🚀 Starting benchmark workflow..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

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

    # Add optional deployment arguments only if they are specified
    if [[ -n "$AGG_CONFIG" ]]; then
        cmd+=(--agg "$AGG_CONFIG")
    fi

    if [[ -n "$DISAGG_CONFIG" ]]; then
        cmd+=(--disagg "$DISAGG_CONFIG")
    fi

    if [[ -n "$VANILLA_CONFIG" ]]; then
        cmd+=(--vanilla "$VANILLA_CONFIG")
    fi

    if [[ -n "$ENDPOINT" ]]; then
        cmd+=(--endpoint "$ENDPOINT")
    fi

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

cleanup() {
    if [[ $? -ne 0 ]]; then
        echo "❌ Script failed. Check logs above for details." >&2
    fi
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
    local exit_code="$1"
    if [[ "$exit_code" -ne 0 ]]; then
        echo "❌ Script failed. Check logs above for details." >&2
    fi
}

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'cleanup $?' EXIT
    main "$@"
fi
