#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BPF tracing setup script.
#
# Installs bpftrace, configures kernel permissions, and optionally grants
# capabilities so bpftrace can run without sudo.
#
# Usage:
#   sudo bash setup.sh              # full setup (install + kernel + capabilities)
#   sudo bash setup.sh --check      # check current state only
#   sudo bash setup.sh --install    # install bpftrace only
#   sudo bash setup.sh --kernel     # configure kernel permissions only
#   sudo bash setup.sh --caps       # grant bpftrace capabilities only
#   sudo bash setup.sh --reset      # remove capabilities and restore kernel defaults

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
info() { echo -e "  ${CYAN}→${NC} $*"; }

# ─── Check functions ────────────────────────────────────────────────────────

check_root() {
    if [[ $EUID -ne 0 ]]; then
        fail "This script must be run as root (sudo bash setup.sh)"
        exit 1
    fi
}

check_bpftrace_installed() {
    if command -v bpftrace &>/dev/null; then
        local ver
        ver=$(bpftrace --version 2>/dev/null | head -1)
        ok "bpftrace installed: $ver"
        return 0
    else
        fail "bpftrace not installed"
        return 1
    fi
}

check_kernel_permissions() {
    local paranoid kptr issues=0

    paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
    kptr=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null || echo "unknown")

    if [[ "$paranoid" == "-1" ]]; then
        ok "perf_event_paranoid = $paranoid (unrestricted)"
    elif [[ "$paranoid" -le 1 ]] 2>/dev/null; then
        warn "perf_event_paranoid = $paranoid (limited — set to -1 for full access)"
        issues=1
    else
        fail "perf_event_paranoid = $paranoid (restricted — BPF probes may fail)"
        issues=1
    fi

    if [[ "$kptr" == "0" ]]; then
        ok "kptr_restrict = $kptr (kernel symbols visible)"
    else
        warn "kptr_restrict = $kptr (kernel symbols hidden — set to 0 for stack symbolization)"
        issues=1
    fi

    return $issues
}

check_capabilities() {
    local bpftrace_path
    bpftrace_path=$(command -v bpftrace 2>/dev/null || true)

    if [[ -z "$bpftrace_path" ]]; then
        fail "bpftrace not found — cannot check capabilities"
        return 1
    fi

    local caps
    caps=$(getcap "$bpftrace_path" 2>/dev/null || true)

    if [[ -n "$caps" ]]; then
        ok "Capabilities set: $caps"
        return 0
    else
        warn "No capabilities set on $bpftrace_path (requires sudo to run)"
        return 1
    fi
}

check_debugfs() {
    if mountpoint -q /sys/kernel/debug 2>/dev/null; then
        ok "debugfs mounted at /sys/kernel/debug"
        return 0
    else
        warn "debugfs not mounted (some tracepoints may be unavailable)"
        return 1
    fi
}

check_tracefs() {
    if [[ -d /sys/kernel/tracing ]] || [[ -d /sys/kernel/debug/tracing ]]; then
        ok "tracefs available"
        return 0
    else
        warn "tracefs not found (tracepoint-based scripts may fail)"
        return 1
    fi
}

run_check() {
    echo ""
    echo "BPF Tracing Environment Check"
    echo "=============================="
    echo ""

    echo "Installation:"
    check_bpftrace_installed || true
    echo ""

    echo "Kernel permissions:"
    check_kernel_permissions || true
    echo ""

    echo "Capabilities:"
    check_capabilities || true
    echo ""

    echo "Kernel interfaces:"
    check_debugfs || true
    check_tracefs || true
    echo ""

    # Quick smoke test
    echo "Smoke test:"
    if command -v bpftrace &>/dev/null; then
        if bpftrace -e 'BEGIN { printf("ok\n"); exit(); }' &>/dev/null; then
            ok "bpftrace can execute probes"
        else
            fail "bpftrace probe execution failed (check permissions)"
        fi
    else
        fail "Cannot run smoke test — bpftrace not installed"
    fi
    echo ""
}

# ─── Install ────────────────────────────────────────────────────────────────

install_bpftrace() {
    echo ""
    echo "Installing bpftrace"
    echo "==================="
    echo ""

    if command -v bpftrace &>/dev/null; then
        ok "bpftrace already installed ($(bpftrace --version 2>/dev/null | head -1))"
        return 0
    fi

    # Detect package manager
    if command -v apt-get &>/dev/null; then
        info "Using apt (Debian/Ubuntu)"
        apt-get update -qq
        apt-get install -y -qq bpftrace linux-headers-"$(uname -r)" 2>/dev/null || \
            apt-get install -y -qq bpftrace
        ok "bpftrace installed via apt"
    elif command -v dnf &>/dev/null; then
        info "Using dnf (Fedora/RHEL)"
        dnf install -y bpftrace kernel-devel-"$(uname -r)" 2>/dev/null || \
            dnf install -y bpftrace
        ok "bpftrace installed via dnf"
    elif command -v yum &>/dev/null; then
        info "Using yum (CentOS/RHEL)"
        yum install -y bpftrace kernel-devel-"$(uname -r)" 2>/dev/null || \
            yum install -y bpftrace
        ok "bpftrace installed via yum"
    elif command -v pacman &>/dev/null; then
        info "Using pacman (Arch)"
        pacman -S --noconfirm bpftrace
        ok "bpftrace installed via pacman"
    else
        fail "No supported package manager found. Install bpftrace manually:"
        echo "    https://github.com/bpftrace/bpftrace/blob/master/INSTALL.md"
        return 1
    fi
}

# ─── Kernel permissions ─────────────────────────────────────────────────────

configure_kernel() {
    echo ""
    echo "Configuring kernel permissions"
    echo "=============================="
    echo ""

    info "These settings are temporary and revert on reboot."
    echo ""

    # perf_event_paranoid
    local current_paranoid
    current_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
    if [[ "$current_paranoid" != "-1" ]]; then
        echo -1 > /proc/sys/kernel/perf_event_paranoid
        ok "perf_event_paranoid: $current_paranoid → -1"
    else
        ok "perf_event_paranoid already -1"
    fi

    # kptr_restrict
    local current_kptr
    current_kptr=$(cat /proc/sys/kernel/kptr_restrict)
    if [[ "$current_kptr" != "0" ]]; then
        echo 0 > /proc/sys/kernel/kptr_restrict
        ok "kptr_restrict: $current_kptr → 0"
    else
        ok "kptr_restrict already 0"
    fi

    # Mount debugfs if needed
    if ! mountpoint -q /sys/kernel/debug 2>/dev/null; then
        mount -t debugfs none /sys/kernel/debug 2>/dev/null && \
            ok "Mounted debugfs" || \
            warn "Could not mount debugfs"
    fi

    echo ""
    info "To make persistent across reboots, add to /etc/sysctl.conf:"
    echo "    kernel.perf_event_paranoid = -1"
    echo "    kernel.kptr_restrict = 0"
}

# ─── Capabilities ───────────────────────────────────────────────────────────

grant_capabilities() {
    echo ""
    echo "Granting bpftrace capabilities"
    echo "==============================="
    echo ""

    local bpftrace_path
    bpftrace_path=$(command -v bpftrace 2>/dev/null || true)

    if [[ -z "$bpftrace_path" ]]; then
        fail "bpftrace not found — install it first"
        return 1
    fi

    # Resolve symlinks to get the real binary
    bpftrace_path=$(readlink -f "$bpftrace_path")
    info "Binary: $bpftrace_path"

    # Required capabilities for BPF tracing
    local caps="cap_bpf,cap_perfmon,cap_net_admin,cap_sys_ptrace+ep"

    setcap "$caps" "$bpftrace_path"
    ok "Capabilities granted: $caps"
    echo ""
    info "bpftrace can now run WITHOUT sudo."
    info "Capabilities persist until the binary is updated/reinstalled."
    echo ""
    info "To verify:  getcap $bpftrace_path"
    info "To remove:  sudo setcap -r $bpftrace_path"
}

# ─── Reset ──────────────────────────────────────────────────────────────────

reset_all() {
    echo ""
    echo "Resetting BPF configuration"
    echo "==========================="
    echo ""

    # Remove capabilities
    local bpftrace_path
    bpftrace_path=$(command -v bpftrace 2>/dev/null || true)
    if [[ -n "$bpftrace_path" ]]; then
        bpftrace_path=$(readlink -f "$bpftrace_path")
        setcap -r "$bpftrace_path" 2>/dev/null && \
            ok "Capabilities removed from $bpftrace_path" || \
            warn "No capabilities to remove"
    fi

    # Restore kernel defaults
    echo 4 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null && \
        ok "perf_event_paranoid → 4 (default)" || \
        warn "Could not restore perf_event_paranoid"

    echo 1 > /proc/sys/kernel/kptr_restrict 2>/dev/null && \
        ok "kptr_restrict → 1 (default)" || \
        warn "Could not restore kptr_restrict"

    echo ""
    ok "Reset complete. bpftrace now requires sudo again."
}

# ─── Full setup ─────────────────────────────────────────────────────────────

full_setup() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║       BPF Tracing Setup                         ║"
    echo "╚══════════════════════════════════════════════════╝"

    install_bpftrace
    configure_kernel
    grant_capabilities

    echo ""
    echo "Setup complete. Running verification..."
    run_check
}

# ─── Usage ──────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
BPF Tracing Setup

Usage:
  sudo bash setup.sh              Full setup (install + kernel + capabilities)
  sudo bash setup.sh --check      Check current environment
  sudo bash setup.sh --install    Install bpftrace only
  sudo bash setup.sh --kernel     Configure kernel permissions (temporary)
  sudo bash setup.sh --caps       Grant bpftrace capabilities (run without sudo)
  sudo bash setup.sh --reset      Remove capabilities and restore kernel defaults

After setup, run BPF scripts without sudo:
  bpftrace -p <PID> scripts/bpf/offcputime.bt
  bpftrace -p <PID> scripts/bpf/syscall_latency.bt
  ./scripts/bpf/run.sh --pid <PID>
EOF
}

# ─── Main ───────────────────────────────────────────────────────────────────

main() {
    case "${1:-}" in
        --check)
            run_check
            ;;
        --install)
            check_root
            install_bpftrace
            ;;
        --kernel)
            check_root
            configure_kernel
            ;;
        --caps)
            check_root
            grant_capabilities
            ;;
        --reset)
            check_root
            reset_all
            ;;
        -h|--help)
            usage
            ;;
        "")
            check_root
            full_setup
            ;;
        *)
            fail "Unknown option: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
}

main "$@"
