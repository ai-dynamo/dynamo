#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One role's work on this node, run by run_arm.sh as an srun step. Long-lived roles
# (mocks, frontend, sampler, aiperf) stay in the foreground, so the step lives as long as
# they do and ending the step stops them. RUN is the arm's node-local directory.
# usage: node.sh COMMAND RUN [ARGS...]   (EXPECT_HOST: fail unless placed on that node)
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ -n ${EXPECT_HOST:-} && $(hostname -s) != "$EXPECT_HOST" ]]; then
    echo "step placed on $(hostname -s), not $EXPECT_HOST" >&2
    exit 3
fi
cmd=$1 run=$2
shift 2
mkdir -p "$run"
ulimit -Sn "$(ulimit -Hn)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY='*' no_proxy='*'

case $cmd in
    preflight)
        # A rerun reuses its label: start from an empty run directory.
        rm -rf "$run"
        mkdir -p "$run"
        for name in global-frontend mock-pool aiperf; do
            if pgrep -x "$name" > /dev/null; then
                echo "$(hostname): $name is still running" >&2
                exit 1
            fi
        done
        echo "$(hostname): tcp_slow_start_after_idle=$(cat /proc/sys/net/ipv4/tcp_slow_start_after_idle)" \
            "nic_speed_mbps=$(cat /sys/class/net/$NIC/speed)" \
            "var_tmp_free=$(df -h --output=avail /var/tmp | tail -1 | tr -d ' ')"
        ;;
    address) # this node's IPv4 address on the NIC
        ip -4 -o addr show "$NIC" | awk '{split($4, a, "/"); print a[1]}'
        ;;
    phase) # succeeds once this node's aiperf has started its profiling phase
        grep -q 'Phase profiling (profiling) started' "$run/aiperf.log" 2> /dev/null
        ;;
    mocks) # mocks RUN MOCK_BINARY [MOCK_ARGS...]: one mock per pool, 8 cores each, half on each NUMA domain
        binary=$1
        shift
        trap 'kill $(jobs -p) 2> /dev/null' TERM INT
        for ((i = 0; i < POOLS; i++)); do
            if ((i < POOLS / 2)); then
                domain=0 first=$((i * 8))
            else
                domain=1 first=$((72 + (i - POOLS / 2) * 8))
            fi
            numactl --physcpubind=$first-$((first + 7)) --membind=$domain \
                "$binary" --port $((POOL_PORT_BASE + i)) --threads 8 "$@" > "$run/mock-$i.log" 2>&1 &
        done
        wait
        ;;
    frontend) # frontend RUN BINARY CPUS POOL_HOST [FRONTEND_ARGS...]
        binary=$1 cpus=$2 host=$3
        shift 3
        pools=()
        for ((i = 0; i < POOLS; i++)); do
            pools+=(--pool "stress/bench/pool-$(printf %02d $i)=$MODEL@http://$host:$((POOL_PORT_BASE + i))")
        done
        exec > "$run/frontend.log" 2>&1
        exec env RUST_LOG=info numactl --physcpubind="$cpus" --membind=0 \
            "$binary" --host 0.0.0.0 --port $FRONTEND_PORT --router-policy least-in-flight "${pools[@]}" "$@"
        ;;
    wait_http) # wait_http RUN URL...: up to 60 s for each to answer 200
        for url in "$@"; do
            for _ in $(seq 120); do
                curl -sf -m 2 "$url" > /dev/null && continue 2
                sleep 0.5
            done
            echo "$(hostname): $url not ready" >&2
            exit 1
        done
        ;;
    smoke) # smoke RUN BASE_URL: one streamed request must come back with exactly 7 tokens
        body='{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":7,"stream":true,"stream_options":{"include_usage":true}}'
        curl -sN -m 10 "$1/v1/chat/completions" -H 'content-type: application/json' -d "$body" > "$run/smoke.sse"
        tokens=$(grep -c '"delta":{"content"' "$run/smoke.sse" || true)
        usage=$(grep -c '"completion_tokens":7' "$run/smoke.sse" || true)
        done_=$(grep -c '^data: \[DONE\]' "$run/smoke.sse" || true)
        if [[ $tokens != 7 || $usage != 1 || $done_ != 1 ]]; then
            echo "smoke via $1 failed: $tokens tokens, $usage usage, $done_ DONE" >&2
            exit 1
        fi
        echo "smoke via $1: 7 tokens, usage, DONE"
        ;;
    sampler)
        exec python3 "$BENCH/slurm/sampler.py" "$run/telemetry.jsonl" "$NIC" \
            global-frontend mock-pool aiperf python3
        ;;
    aiperf) # aiperf RUN URLS: runs in the foreground until the profiling phase stops sending
        python3 "$BENCH/slurm/aiperf_config.py" "$1" "$run/aiperf" > "$run/aiperf-config.json"
        env HF_HOME="$AIPERF_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
            PYTHONUNBUFFERED=1 AIPERF_DATASET_MMAP_CACHE_DIR="$AIPERF_CACHE/aiperf/dataset_mmap" \
            XDG_CACHE_HOME="$AIPERF_CACHE" \
            setsid "$AIPERF" profile --config "$run/aiperf-config.json" > "$run/aiperf.log" 2>&1 < /dev/null &
        pid=$!
        trap 'kill -KILL -- -$pid 2> /dev/null' TERM INT
        # The phase's "sending complete" line carries its completed count; the records
        # aiperf would process afterwards are not needed, and they hold most of its memory.
        until grep -q 'Phase profiling (profiling) sending complete' "$run/aiperf.log" || ! kill -0 $pid 2> /dev/null; do
            sleep 1
        done
        kill -KILL -- -$pid 2> /dev/null || true
        grep -E 'Phase profiling \(profiling\) (started|sending complete)' "$run/aiperf.log" > "$run/aiperf-phase.log" || true
        ;;
    netstat) # netstat RUN TAG: kernel TCP and NIC counters, socket memory and a socket sample
        {
            echo "# at $(date +%s.%N)"
            nstat -az
            echo "# sockstat"
            cat /proc/net/sockstat
            echo "# nic"
            for key in rx_dropped tx_dropped rx_missed_errors rx_errors tx_errors rx_fifo_errors; do
                echo "$key $(cat /sys/class/net/$NIC/statistics/$key)"
            done
            echo "# sysctl"
            sysctl net.ipv4.tcp_mem net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.core.somaxconn \
                net.ipv4.tcp_max_syn_backlog net.core.netdev_max_backlog 2> /dev/null
            echo "# ip link"
            ip -s link show "$NIC"
        } > "$run/netstat-$1.txt" 2>&1
        ss -tinH state established 2> /dev/null | head -4000 > "$run/ss-$1.txt" || true
        ;;
    metrics) # metrics RUN TAG: the frontend's /metrics, headed by the scrape time and its user and system CPU ticks
        pid=$(pgrep -x global-frontend)
        { echo "# scraped_at $(date +%s.%N)"; echo "# cpu_ticks $(cut -d' ' -f14,15 /proc/$pid/stat)"; curl -sf -m 10 "http://localhost:$FRONTEND_PORT/metrics"; } \
            > "$run/metrics-$1.prom" || true
        ;;
    perf) # perf RUN SECONDS: on-CPU profile of the frontend
        pid=$(pgrep -x global-frontend)
        threads() { for task in /proc/$pid/task/*; do echo "$(cat $task/comm) $(cut -d' ' -f14,15 $task/stat)"; done; }
        threads > "$run/threads-start.txt"
        # PERF_EVENT=cycles adds kernel stacks (the default samples user space only).
        perf record -e "${PERF_EVENT:-cycles:u}" -c 2500000 -P --call-graph fp -m 1024 -o "$run/perf.data" \
            -p "$pid" -- sleep "$1" > "$run/perf.log" 2>&1
        threads > "$run/threads-end.txt"
        ;;
    collect) # collect RUN DEST: copy this node's results to DEST/<host>
        dest=$1/$(hostname -s)
        mkdir -p "$dest"
        find "$run" -maxdepth 1 -type f ! -name '*.jsonl.tmp' -exec cp {} "$dest/" \;
        ;;
    *)
        echo "unknown command $cmd" >&2
        exit 2
        ;;
esac
