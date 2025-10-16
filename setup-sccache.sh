#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Local sccache setup script for Dynamo project
# This script helps configure and manage sccache for faster local Rust compilation

set -euo pipefail

SCCACHE_BIN="/usr/bin/sccache"
SCCACHE_CACHE_DIR="$HOME/.cache/sccache"
SCCACHE_CACHE_SIZE="10G"

usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    setup           Set up sccache environment and cache directory
    start           Start sccache server
    stop            Stop sccache server
    stats           Show sccache statistics
    clear           Clear sccache cache
    status          Show sccache server status
    env             Show environment variables to export
    help            Show this help message
    install          Install sccache via apt and configure environment

Environment Configuration:
    SCCACHE_DIR     Cache directory (default: $SCCACHE_CACHE_DIR)
    SCCACHE_CACHE_SIZE  Maximum cache size (default: $SCCACHE_CACHE_SIZE)

Examples:
    # Initial setup
    $0 setup

    # Start sccache and build project
    $0 start
    cargo build --release
    $0 stats

    # Clear cache if needed
    $0 clear
EOF
}

setup_sccache() {
    echo "Setting up sccache for local development..."

    # Create cache directory
    mkdir -p "$SCCACHE_CACHE_DIR"

    # Add ~/.local/bin to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "Adding ~/.local/bin to PATH..."
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Set up environment variables
    echo "Configuring sccache environment..."
    cat >> ~/.bashrc << 'EOF'

# sccache configuration
export SCCACHE_DIR="$HOME/.cache/sccache"
export SCCACHE_CACHE_SIZE="10G"
export RUSTC_WRAPPER="/usr/bin/sccache"
EOF

    # Export for current session
    export SCCACHE_DIR="$SCCACHE_CACHE_DIR"
    export SCCACHE_CACHE_SIZE="$SCCACHE_CACHE_SIZE"
    export RUSTC_WRAPPER="$SCCACHE_BIN"

    echo "✓ sccache setup complete!"
    echo "✓ Cache directory: $SCCACHE_CACHE_DIR"
    echo "✓ Cache size limit: $SCCACHE_CACHE_SIZE"
    echo ""
    echo "To apply environment changes, run: source ~/.bashrc"
    echo "Or start a new shell session."
}

start_sccache() {
    if ! command -v sccache >/dev/null 2>&1; then
        echo "Error: sccache not found in PATH. Run '$0 setup' first."
        exit 1
    fi

    echo "Starting sccache server..."
    sccache --start-server
    echo "✓ sccache server started"
}

stop_sccache() {
    if command -v sccache >/dev/null 2>&1; then
        echo "Stopping sccache server..."
        sccache --stop-server || true
        echo "✓ sccache server stopped"
    else
        echo "sccache not found in PATH"
    fi
}

show_stats() {
    if command -v sccache >/dev/null 2>&1; then
        echo "=== sccache Statistics ==="
        sccache --show-stats
    else
        echo "Error: sccache not found in PATH"
        exit 1
    fi
}

clear_cache() {
    if command -v sccache >/dev/null 2>&1; then
        echo "Clearing sccache cache..."
        sccache --zero-stats
        echo "✓ sccache cache cleared"
    else
        echo "Error: sccache not found in PATH"
        exit 1
    fi
}

show_status() {
    if command -v sccache >/dev/null 2>&1; then
        echo "=== sccache Status ==="
        echo "Binary: $(which sccache)"
        echo "Version: $(sccache --version)"
        echo "Cache directory: ${SCCACHE_DIR:-$SCCACHE_CACHE_DIR}"
        echo "Cache size limit: ${SCCACHE_CACHE_SIZE:-$SCCACHE_CACHE_SIZE}"
        echo ""
        if sccache --show-stats >/dev/null 2>&1; then
            echo "Server status: Running"
        else
            echo "Server status: Not running"
        fi
    else
        echo "sccache not found in PATH"
        echo "Run '$0 setup' to install and configure sccache"
    fi
}

show_env() {
    echo "# Environment variables for sccache"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "export SCCACHE_DIR=\"$SCCACHE_CACHE_DIR\""
    echo "export SCCACHE_CACHE_SIZE=\"$SCCACHE_CACHE_SIZE\""
    echo "export RUSTC_WRAPPER=\"/usr/bin/sccache\""
    echo ""
    echo "# To apply these settings:"
    echo "# source <(./setup-sccache.sh env)"
}

install_sccache_apt() {
    echo "Installing sccache via apt..."
    apt-get update
    apt-get install -y sccache
    echo "✓ sccache installed"
}

main() {
    case "${1:-help}" in
        setup)
            setup_sccache
            ;;
        start)
            start_sccache
            ;;
        stop)
            stop_sccache
            ;;
        stats)
            show_stats
            ;;
        clear)
            clear_cache
            ;;
        status)
            show_status
            ;;
        env)
            show_env
            ;;
        help|--help|-h)
            usage
            ;;
        install)
            install_sccache_apt
            ;;
        *)
            echo "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
