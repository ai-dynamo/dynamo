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
    install             Install sccache binary (requires ARCH_ALT environment variable)
    show-stats          Display sccache statistics with optional build name
    export-stats-json   Export sccache statistics as JSON to a file
    help                Show this help message

Environment variables:
    USE_SCCACHE             Set to 'true' to enable sccache
    SCCACHE_BUCKET          S3 bucket name (fallback if not passed as parameter)
    SCCACHE_REGION          S3 region (fallback if not passed as parameter)
    ARCH                    Architecture for S3 key prefix (fallback if not passed as parameter)
    ARCH_ALT                Alternative architecture name for downloads (e.g., x86_64, aarch64)

Examples:
    # Install sccache (requires ARCH_ALT to be set)
    ARCH_ALT=x86_64 $0 install
    # Show stats with build name
    $0 show-stats "UCX"
    # Export stats to JSON file
    $0 export-stats-json /path/to/output.json "BuildName"
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
    else
        echo "sccache is not available"
    fi
}

export_stats_json() {
    local output_file="${1:-/tmp/sccache-stats.json}"
    local build_name="${2:-unknown}"

    if ! command -v sccache >/dev/null 2>&1; then
        echo '{"error": "sccache is not available"}' > "$output_file"
        return 1
    fi

    # Get raw stats output
    local stats_output
    stats_output=$(sccache --show-stats 2>&1)

    # Parse the output and convert to JSON using awk
    echo "$stats_output" | awk -v build_name="$build_name" '
    BEGIN {
        print "{"
        print "  \"build_name\": \"" build_name "\","
        first = 1
    }

    # Match lines with format: "Key    Value" or "Key    Value %"
    /^[A-Za-z]/ {
        # Extract key and value
        key = ""
        value = ""
        unit = ""

        # Find the position where multiple spaces start (separator between key and value)
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^[0-9.]+$/) {
                # Found the start of the value
                value = $i
                # Check if there'\''s a unit after the value
                if (i + 1 <= NF && ($( i+1) == "%" || $(i+1) == "s")) {
                    unit = $(i+1)
                }
                # Everything before this is the key
                for (j = 1; j < i; j++) {
                    if (key == "") {
                        key = $j
                    } else {
                        key = key " " $j
                    }
                }
                break
            }
        }

        if (key != "" && value != "") {
            # Convert key to snake_case
            gsub(/[()]/, "", key)  # Remove parentheses
            gsub(/\//, "_", key)   # Replace / with _
            gsub(/ /, "_", key)    # Replace spaces with _
            key = tolower(key)

            # Add unit suffix if applicable
            if (unit == "%") {
                key = key "_percent"
            } else if (unit == "s") {
                key = key "_seconds"
            }

            # Check if value is a valid JSON number (integer or single decimal)
            # Version strings like "0.8.2" need to be quoted
            is_number = (value ~ /^[0-9]+$/ || value ~ /^[0-9]+\.[0-9]+$/)

            # Print JSON field
            if (!first) print ","
            if (is_number) {
                printf "  \"%s\": %s", key, value
            } else {
                printf "  \"%s\": \"%s\"", key, value
            }
            first = 0
        }
    }

    END {
        print ""
        print "}"
    }
    ' > "$output_file"

    echo "✅ sccache stats exported to: $output_file"
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
        export-stats-json)
            shift  # Remove the command from arguments
            export_stats_json "$@"  # Pass all remaining arguments
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
