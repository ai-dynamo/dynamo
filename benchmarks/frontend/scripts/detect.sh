#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capability detection for dynamo frontend performance profiling tools.
# Checks available tools and features, outputs matrix + recommendations.
#
# Usage: ./detect.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { printf "${GREEN}OK${NC}   %-40s %s\n" "$1" "$2"; }
warn() { printf "${YELLOW}WARN${NC} %-40s %s\n" "$1" "$2"; }
fail() { printf "${RED}FAIL${NC} %-40s %s\n" "$1" "$2"; }

echo "=== Dynamo Frontend Perf Tool Detection ==="
echo ""

# --- System ---
echo "--- System ---"
echo "Kernel: $(uname -r)"
echo "Arch:   $(uname -m)"
echo ""

# --- BPF ---
echo "--- BPF Tracing ---"
if command -v bpftrace &>/dev/null; then
    ok "bpftrace" "$(bpftrace --version 2>/dev/null | head -1)"
else
    fail "bpftrace" "not found (apt install bpftrace)"
fi

if [[ $(id -u) -eq 0 ]]; then
    ok "root access" "available"
else
    if command -v capsh &>/dev/null; then
        if capsh --print 2>/dev/null | grep -q cap_bpf; then
            ok "CAP_BPF" "available"
        else
            warn "CAP_BPF" "not available (BPF scripts need root or CAP_BPF)"
        fi
    else
        warn "root access" "not root; BPF scripts may need sudo"
    fi
fi

# --- Perf ---
echo ""
echo "--- CPU Profiling ---"
if command -v perf &>/dev/null; then
    ok "perf" "$(perf version 2>/dev/null | head -1)"
else
    fail "perf" "not found (apt install linux-tools-$(uname -r))"
fi

if command -v flamegraph &>/dev/null; then
    ok "cargo-flamegraph" "available"
else
    warn "cargo-flamegraph" "not found (cargo install flamegraph)"
fi

if command -v samply &>/dev/null; then
    ok "samply" "available"
else
    warn "samply" "not found (cargo install samply)"
fi

# Flamegraph rendering
if command -v flamegraph.pl &>/dev/null; then
    ok "flamegraph.pl" "available"
elif command -v inferno-flamegraph &>/dev/null; then
    ok "inferno" "available (flamegraph alternative)"
else
    warn "flamegraph renderer" "not found (cargo install inferno)"
fi

# --- Nsight ---
echo ""
echo "--- NVIDIA Nsight ---"
if command -v nsys &>/dev/null; then
    ok "nsys" "$(nsys --version 2>/dev/null | head -1)"
else
    warn "nsys" "not found (install NVIDIA Nsight Systems)"
fi

# --- Dynamo features ---
echo ""
echo "--- Dynamo Build Features ---"

# Check for nvtx feature in Cargo.toml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../../.."

if [[ -f "$REPO_ROOT/lib/runtime/Cargo.toml" ]]; then
    if grep -q 'nvtx' "$REPO_ROOT/lib/runtime/Cargo.toml" 2>/dev/null; then
        ok "nvtx feature" "defined in lib/runtime/Cargo.toml"
    else
        fail "nvtx feature" "not found in lib/runtime/Cargo.toml"
    fi
fi

if [[ -f "$REPO_ROOT/Cargo.toml" ]]; then
    if grep -q 'profile.profiling' "$REPO_ROOT/Cargo.toml" 2>/dev/null; then
        ok "profiling profile" "defined in Cargo.toml"
    else
        warn "profiling profile" "not found in Cargo.toml"
    fi
fi

# --- Runtime environment ---
echo ""
echo "--- Runtime Environment ---"

if [[ -n "${DYN_ENABLE_NVTX:-}" ]]; then
    ok "DYN_ENABLE_NVTX" "set to '$DYN_ENABLE_NVTX'"
else
    warn "DYN_ENABLE_NVTX" "not set (NVTX annotations disabled at runtime)"
fi

if [[ -n "${DYN_ENABLE_POLL_HISTOGRAM:-}" ]]; then
    ok "DYN_ENABLE_POLL_HISTOGRAM" "set to '$DYN_ENABLE_POLL_HISTOGRAM'"
else
    warn "DYN_ENABLE_POLL_HISTOGRAM" "not set (tokio poll histogram disabled)"
fi

if [[ -n "${DYN_PERF_DIAG:-}" ]]; then
    ok "DYN_PERF_DIAG" "set to '$DYN_PERF_DIAG'"
else
    warn "DYN_PERF_DIAG" "not set"
fi

# --- Load generation ---
echo ""
echo "--- Load Generation ---"
if command -v aiperf &>/dev/null || python -c "import aiperf" 2>/dev/null; then
    ok "aiperf" "available"
else
    warn "aiperf" "not found (pip install git+https://github.com/ai-dynamo/aiperf.git)"
fi

# --- Prometheus ---
echo ""
echo "--- Observability ---"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
if curl -s "http://localhost:$FRONTEND_PORT/metrics" >/dev/null 2>&1; then
    ok "Prometheus /metrics" "reachable at localhost:$FRONTEND_PORT"
else
    warn "Prometheus /metrics" "not reachable at localhost:$FRONTEND_PORT"
fi

# --- Python analysis ---
echo ""
echo "--- Analysis ---"
for pkg in numpy pandas matplotlib scipy; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ok "python3 $pkg" "available"
    else
        warn "python3 $pkg" "not found (pip install $pkg)"
    fi
done

echo ""
echo "=== Detection Complete ==="
