#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
output_dir="$1"
mkdir -p "$output_dir"
for log_name in vcluster-port-forward.log vcluster-port-forward-watchdog.log; do
    if [ -f "${GITHUB_WORKSPACE}/.${log_name}" ]; then
        # Non-hidden names ensure upload-artifact includes the tunnel logs.
        cp "${GITHUB_WORKSPACE}/.${log_name}" "${output_dir}/${log_name}"
    fi
done
{
    echo "captured_at=$(date -u +%FT%TZ)"
    curl -skS --connect-timeout 1 --max-time 2 \
        -o /dev/null -w 'healthz_http_code=%{http_code}\n' \
        https://127.0.0.1:8443/healthz || echo "healthz_unreachable=true"
    for pid_name in vcluster-port-forward.pid vcluster-port-forward-watchdog.pid; do
        if [ -f "${GITHUB_WORKSPACE}/.${pid_name}" ]; then
            pid=$(cat "${GITHUB_WORKSPACE}/.${pid_name}")
            echo "${pid_name}=${pid}"
            ps -p "$pid" -o pid=,ppid=,stat=,etime=,command= || true
        else
            echo "${pid_name}=missing"
        fi
    done
} > "${output_dir}/vcluster-port-forward-status.log" 2>&1
