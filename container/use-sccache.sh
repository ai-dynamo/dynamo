#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# sccache management script
# This script handles sccache installation, environment setup, and statistics display

SCCACHE_VERSION="v0.8.2"


usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    install                 Install sccache binary (requires ARCH_ALT environment variable)
    show-stats              Display sccache statistics with optional build name
    export-final-metrics    Export final sccache metrics to files for host consumption
    help                    Show this help message

Environment variables:
    USE_SCCACHE             Set to 'true' to enable sccache
    SCCACHE_BUCKET          S3 bucket name (fallback if not passed as parameter)
    SCCACHE_REGION          S3 region (fallback if not passed as parameter)
    ARCH                    Architecture for S3 key prefix (fallback if not passed as parameter)
    ARCH_ALT                Alternative architecture name for downloads (e.g., x86_64, aarch64)
    SCCACHE_ENV_FILE        File path to write structured metrics (for export-final-metrics)
    SCCACHE_METRICS_FILE    File path to write raw sccache output (for export-final-metrics)

Examples:
    # Install sccache (requires ARCH_ALT to be set)
    ARCH_ALT=x86_64 $0 install
    # Show stats with build name
    $0 show-stats "UCX"
    # Export final metrics to files
    SCCACHE_ENV_FILE=/tmp/sccache.env SCCACHE_METRICS_FILE=/tmp/sccache.txt $0 export-final-metrics
EOF
}

install_sccache() {
    if [ -z "${ARCH_ALT:-}" ]; then
        echo "Error: ARCH_ALT environment variable is required for sccache installation"
        exit 1
    fi
    echo "Installing sccache ${SCCACHE_VERSION} for architecture ${ARCH_ALT}..."
    # Download and install sccache
    wget --tries=3 --waitretry=5 \
        "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz"
    tar -xzf "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl.tar.gz"
    mv "sccache-${SCCACHE_VERSION}-${ARCH_ALT}-unknown-linux-musl/sccache" /usr/local/bin/
    # Cleanup
    rm -rf sccache*
    echo "sccache installed successfully"
}

show_stats() {
    if command -v sccache >/dev/null 2>&1; then
        echo "=== sccache statistics AFTER $1 ==="
        sccache --show-stats
        
        # Export metrics to a file that can be read by the host
        if [ -n "${SCCACHE_METRICS_FILE:-}" ]; then
            echo "📊 Exporting sccache metrics to ${SCCACHE_METRICS_FILE}"
            sccache --show-stats > "${SCCACHE_METRICS_FILE}" 2>&1
        fi
    else
        echo "sccache is not available"
    fi
}

# Function to export final sccache metrics in a structured format
export_final_metrics() {
    if command -v sccache >/dev/null 2>&1; then
        echo "📊 Collecting final sccache metrics..."
        
        # Get the raw stats
        STATS_OUTPUT=$(sccache --show-stats 2>&1)
        
        # Parse key metrics using grep and awk
        CACHE_HITS=$(echo "$STATS_OUTPUT" | grep -E "Cache hits" | grep -oE '[0-9]+' | head -1 || echo "0")
        CACHE_MISSES=$(echo "$STATS_OUTPUT" | grep -E "Cache misses" | grep -oE '[0-9]+' | head -1 || echo "0")
        COMPILE_REQUESTS=$(echo "$STATS_OUTPUT" | grep -E "Compile requests[^a-z]" | grep -oE '[0-9]+' | head -1 || echo "0")
        
        # Look for S3-specific metrics (may vary by sccache version)
        S3_READS=$(echo "$STATS_OUTPUT" | grep -iE "(S3.*read|Remote.*read)" | grep -oE '[0-9]+' | head -1 || echo "0")
        S3_WRITES=$(echo "$STATS_OUTPUT" | grep -iE "(S3.*write|Remote.*write)" | grep -oE '[0-9]+' | head -1 || echo "0")
        
        # Calculate hit rate
        TOTAL_CACHEABLE=$((CACHE_HITS + CACHE_MISSES))
        if [ "$TOTAL_CACHEABLE" -gt 0 ]; then
            CACHE_HIT_RATE=$((CACHE_HITS * 100 / TOTAL_CACHEABLE))
        else
            CACHE_HIT_RATE=0
        fi
        
        # Export to environment file if specified
        if [ -n "${SCCACHE_ENV_FILE:-}" ]; then
            echo "📁 Writing sccache metrics to ${SCCACHE_ENV_FILE}"
            {
                echo "SCCACHE_CACHE_HITS=${CACHE_HITS}"
                echo "SCCACHE_CACHE_MISSES=${CACHE_MISSES}"
                echo "SCCACHE_COMPILE_REQUESTS=${COMPILE_REQUESTS}"
                echo "SCCACHE_S3_READS=${S3_READS}"
                echo "SCCACHE_S3_WRITES=${S3_WRITES}"
                echo "SCCACHE_HIT_RATE=${CACHE_HIT_RATE}"
                echo "SCCACHE_TOTAL_CACHEABLE=${TOTAL_CACHEABLE}"
            } > "${SCCACHE_ENV_FILE}"
        fi
        
        # Also export raw stats if requested
        if [ -n "${SCCACHE_METRICS_FILE:-}" ]; then
            echo "📊 Exporting raw sccache stats to ${SCCACHE_METRICS_FILE}"
            echo "$STATS_OUTPUT" > "${SCCACHE_METRICS_FILE}"
        fi
        
        # Print summary
        echo "🎯 Final sccache metrics:"
        echo "   Cache hits: ${CACHE_HITS}"
        echo "   Cache misses: ${CACHE_MISSES}"
        echo "   Total requests: ${COMPILE_REQUESTS}"
        echo "   S3 reads: ${S3_READS}"
        echo "   S3 writes: ${S3_WRITES}"
        echo "   Hit rate: ${CACHE_HIT_RATE}%"
        
        return 0
    else
        echo "⚠️  sccache is not available for final metrics collection"
        return 1
    fi
}

main() {
    case "${1:-help}" in
        install)
            install_sccache
            ;;
        generate-env)
            shift  # Remove the command from arguments
            generate_env_file "$@"  # Pass all remaining arguments
            ;;
        show-stats)
            shift  # Remove the command from arguments
            show_stats "$@"  # Pass all remaining arguments
            ;;
        export-final-metrics)
            export_final_metrics
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
