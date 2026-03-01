#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# tokio-console helper for dynamo frontend debugging.
# Connects to a running tokio-console-enabled process.
#
# Prerequisites:
#   - cargo install tokio-console
#   - Process built with tokio_unstable cfg and console-subscriber
#
# Usage:
#   ./tokio_console.sh                    # connect to default http://localhost:6669
#   ./tokio_console.sh --addr host:port   # connect to specific address

set -euo pipefail

ADDR="${TOKIO_CONSOLE_ADDR:-http://localhost:6669}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --addr|-a)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --addr requires an address argument"
                exit 1
            fi
            ADDR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --addr ADDR   tokio-console address (default: http://localhost:6669)"
            echo ""
            echo "Environment:"
            echo "  TOKIO_CONSOLE_ADDR   Alternative to --addr"
            echo ""
            echo "Prerequisites:"
            echo "  1. Install: cargo install tokio-console"
            echo "  2. Build dynamo with tokio_unstable:"
            echo "     RUSTFLAGS='--cfg tokio_unstable' cargo build"
            echo "  3. Enable console-subscriber in the runtime"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! command -v tokio-console &>/dev/null; then
    echo "ERROR: tokio-console not found"
    echo "Install: cargo install tokio-console"
    exit 1
fi

echo "Connecting to tokio-console at $ADDR..."
echo "(Ensure the target process was built with tokio_unstable and console-subscriber)"
echo ""

exec tokio-console "$ADDR"
