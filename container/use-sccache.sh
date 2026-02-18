#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# sccache management script
# This script handles sccache installation, environment setup, and statistics display

SCCACHE_VERSION="v0.14.0"


usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    install         Install sccache binary (requires ARCH_ALT environment variable)
    setup-env       Print export statements to configure sccache for compilation
    show-stats      Display sccache statistics with optional build name
    help            Show this help message

setup-env modes:
    setup-env           Wraps CC/CXX with sccache + sets RUSTC_WRAPPER.
                        Works for autotools (make) and Meson builds.
    setup-env cmake     Also sets CMAKE_C/CXX/CUDA_COMPILER_LAUNCHER.
                        Use for CMake-based builds.

    Usage: eval \$($0 setup-env [cmake])

Environment variables:
    USE_SCCACHE             Set to 'true' to enable sccache
    SCCACHE_BUCKET          S3 bucket name (fallback if not passed as parameter)
    SCCACHE_REGION          S3 region (fallback if not passed as parameter)
    ARCH                    Architecture for S3 key prefix (fallback if not passed as parameter)
    ARCH_ALT                Alternative architecture name for downloads (e.g., x86_64, aarch64)

Examples:
    ARCH_ALT=x86_64 $0 install
    eval \$($0 setup-env)          # autotools / Meson
    eval \$($0 setup-env cmake)    # CMake builds
    $0 show-stats "UCX"
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

    # Create compiler wrapper scripts for autotools/Meson compatibility.
    # Autoconf breaks with CC="sccache gcc" (multi-word value), so we provide
    # single-binary wrappers that autoconf sees as a regular compiler.
    # The real compiler path is passed at runtime via SCCACHE_CC_REAL / SCCACHE_CXX_REAL.
    cat > /usr/local/bin/sccache-cc <<'WRAPPER'
#!/bin/sh
# Only use sccache for pure compilations (-c flag).
# Autoconf tests the compiler with combined compile+link (no -c), and sccache
# can interfere with those tests. sccache only caches compilations anyway,
# so routing linking/other invocations directly to gcc loses nothing.
case " $* " in
    *" -c "*) exec sccache "${SCCACHE_CC_REAL:-gcc}" "$@" ;;
    *)        exec "${SCCACHE_CC_REAL:-gcc}" "$@" ;;
esac
WRAPPER
    chmod +x /usr/local/bin/sccache-cc

    cat > /usr/local/bin/sccache-cxx <<'WRAPPER'
#!/bin/sh
case " $* " in
    *" -c "*) exec sccache "${SCCACHE_CXX_REAL:-g++}" "$@" ;;
    *)        exec "${SCCACHE_CXX_REAL:-g++}" "$@" ;;
esac
WRAPPER
    chmod +x /usr/local/bin/sccache-cxx

    echo "sccache installed successfully"
}

setup_env() {
    local mode="${1:-default}"

    echo "export RUSTC_WRAPPER=\"sccache\";"

    if [ "$mode" = "cmake" ]; then
        # CMake has native launcher support for sccache.
        echo "export CMAKE_C_COMPILER_LAUNCHER=\"sccache\";"
        echo "export CMAKE_CXX_COMPILER_LAUNCHER=\"sccache\";"
        echo "export CMAKE_CUDA_COMPILER_LAUNCHER=\"sccache\";"
    fi
    # Autotools/Meson C/C++ compilation is not wrapped with sccache.
    # sccache's server daemon has compatibility issues with gcc-toolset-14
    # inside Docker's secret-mounted environment, and these builds are fast
    # enough (~30-60s) that the caching ROI is low. The major wins come from
    # RUSTC_WRAPPER (Dynamo Rust compilation) and CMAKE_*_COMPILER_LAUNCHER.
}

show_stats() {
    if command -v sccache >/dev/null 2>&1; then
        # Generate timestamp in ISO 8601 format
        local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

        # Output human-readable text format first
        echo "=== sccache statistics AFTER $1 ==="
        sccache --show-stats
        echo ""

        # Output JSON markers for deterministic parsing
        echo "=== SCCACHE_JSON_BEGIN ==="

        # Create JSON wrapper with section metadata
        cat <<EOF
{
  "section": "$1",
  "timestamp": "$timestamp",
  "sccache_stats": $(sccache --show-stats --stats-format json)
}
EOF

        echo "=== SCCACHE_JSON_END ==="
    else
        echo "sccache is not available"
    fi
}

main() {
    case "${1:-help}" in
        install)
            install_sccache
            ;;
        setup-env)
            shift
            setup_env "$@"
            ;;
        show-stats)
            shift  # Remove the command from arguments
            show_stats "$@"  # Pass all remaining arguments
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
