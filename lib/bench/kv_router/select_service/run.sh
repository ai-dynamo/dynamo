#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run select_service_bench against fresh selection-service processes on this host.
# See ../SELECT_SERVICE_BENCH.md.
#
# Usage: run.sh OUT_DIR FLOW [select_service_bench args...]
#   FLOW is ray or select-and-reserve (HTTP selection service) or ingress
#   (Ray Serve LLM-style ingress replicas embedding dynamo.llm.SelectionService).
#
# Environment:
#   TRACE              Mooncake-format trace (required)
#   PY                 Python with Dynamo's bindings built with select-service (python3)
#   BENCH_BIN          select_service_bench binary (<repo>/target/release/select_service_bench)
#   INGRESS_N          ingress replicas for FLOW=ingress (1); they exchange peer updates
#   AFFINITY_TTL       session affinity TTL in seconds (1800, Ray's default)
#   SVC_ARGS           extra flags for every service process, e.g. "--concurrent-apply"
#   SVC_PORT           HTTP port for the ray and select-and-reserve flows (8092)
#   SVC_CPUS           taskset CPU list for the HTTP service
#   INGRESS_CPUS_EACH  pin ingress i to INGRESS_CPU_BASE + [i * EACH, (i + 1) * EACH)
#   INGRESS_CPU_BASE   first CPU for ingress pinning (0)
#   BENCH_CPUS         taskset CPU list for the load generator
#   PROFILE=1          perf record the service processes during the profile phase
#   PERF, PERF_EVENT, PROFILE_SECS   (perf, cycles, 30)
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
OUT=${1:?usage: run.sh OUT_DIR FLOW [bench args...]}
FLOW=${2:?usage: run.sh OUT_DIR FLOW [bench args...]}
shift 2
: "${TRACE:?set TRACE to a Mooncake-format trace}"
PY=${PY:-python3}
BENCH_BIN=${BENCH_BIN:-$REPO/target/release/select_service_bench}
INGRESS_N=${INGRESS_N:-1}
AFFINITY_TTL=${AFFINITY_TTL:-1800}
SVC_PORT=${SVC_PORT:-8092}
PERF=${PERF:-perf}
PERF_EVENT=${PERF_EVENT:-cycles}
read -r -a SVC_EXTRA <<< "${SVC_ARGS:-}"

if [[ -e $OUT ]]; then
    echo "$OUT already exists" >&2
    exit 1
fi
mkdir -p "$OUT"
# 1,024 workers need a SUB socket and a replay connection each in every service process.
ulimit -n "$(ulimit -Hn)"
{
    date -Is
    echo "flow=$FLOW ingress_n=$INGRESS_N affinity_ttl=$AFFINITY_TTL svc_args=${SVC_ARGS:-}"
    echo "bench args: $*"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
} > "$OUT/run.txt"

pin() {
    if [[ -n ${1:-} ]]; then
        echo taskset -c "$1"
    fi
}

SVCS=()
BENCH_EXTRA=()
SOCKS=()
if [[ $FLOW == ingress ]]; then
    # AF_UNIX paths stop at 108 bytes, so the sockets live in /tmp.
    for ((i = 0; i < INGRESS_N; i++)); do
        SOCKS+=("/tmp/select-ingress.$$.$i.sock")
    done
    for ((i = 0; i < INGRESS_N; i++)); do
        PEERS=()
        for ((j = 0; j < INGRESS_N; j++)); do
            [[ $j -ne $i ]] && PEERS+=(--peer "${SOCKS[$j]}.peer")
        done
        CPUS=
        if [[ -n ${INGRESS_CPUS_EACH:-} ]]; then
            first=$((${INGRESS_CPU_BASE:-0} + i * INGRESS_CPUS_EACH))
            CPUS=$first-$((first + INGRESS_CPUS_EACH - 1))
        fi
        echo "ingress $i cpus=${CPUS:-any}" >> "$OUT/run.txt"
        # -X perf names Python frames in perf profiles.
        $(pin "$CPUS") "$PY" -X perf "$HERE/../select_service_ray_ingress.py" \
            --socket "${SOCKS[$i]}" --peer-socket "${SOCKS[$i]}.peer" "${PEERS[@]}" \
            --session-affinity-ttl-secs "$AFFINITY_TTL" "${SVC_EXTRA[@]}" \
            > "$OUT/service.$i.log" 2>&1 &
        SVCS+=($!)
        BENCH_EXTRA+=(--ingress-socket "${SOCKS[$i]}")
    done
else
    $(pin "${SVC_CPUS:-}") "$PY" -m dynamo.select_service --port "$SVC_PORT" \
        --session-affinity-ttl-secs "$AFFINITY_TTL" "${SVC_EXTRA[@]}" \
        > "$OUT/service.0.log" 2>&1 &
    SVCS+=($!)
fi
printf '%s\n' "${SVCS[@]}" > "$OUT/service.pid"

SAMPLERS=()
BENCH=
cleanup() {
    [[ -n $BENCH ]] && kill "$BENCH" 2>/dev/null
    for pid in "${SAMPLERS[@]}"; do kill "$pid" 2>/dev/null; done
    for pid in "${SVCS[@]}"; do kill -TERM "$pid" 2>/dev/null; done
    for _ in $(seq 20); do
        alive=0
        for pid in "${SVCS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        [[ $alive == 0 ]] && break
        sleep 0.5
    done
    for pid in "${SVCS[@]}"; do kill -KILL "$pid" 2>/dev/null; done
    wait 2>/dev/null
    for sock in "${SOCKS[@]}"; do rm -f "$sock" "$sock.peer"; done
}
trap cleanup EXIT

for ((i = 0; i < ${#SVCS[@]}; i++)); do
    for _ in $(seq 240); do
        if [[ $FLOW == ingress ]]; then
            [[ -S ${SOCKS[$i]} ]] && break
        else
            curl -sf "http://127.0.0.1:$SVC_PORT/health" > /dev/null && break
        fi
        if ! kill -0 "${SVCS[$i]}" 2>/dev/null; then
            echo "service $i exited during startup; see $OUT/service.$i.log" >&2
            exit 1
        fi
        sleep 0.5
    done
done

for ((i = 0; i < ${#SVCS[@]}; i++)); do
    python3 "$HERE/cpu_sampler.py" "${SVCS[$i]}" "$OUT/service.$i.cpu.jsonl" &
    SAMPLERS+=($!)
done

$(pin "${BENCH_CPUS:-}") "$BENCH_BIN" --service-url "http://127.0.0.1:$SVC_PORT" \
    --trace "$TRACE" --flow "$FLOW" --out "$OUT/bench" "${BENCH_EXTRA[@]}" "$@" \
    > "$OUT/bench.log" 2>&1 &
BENCH=$!
# The load generator's own CPU shows whether it, rather than the service, is the limit.
python3 "$HERE/cpu_sampler.py" "$BENCH" "$OUT/bench.cpu.jsonl" &
SAMPLERS+=($!)

if [[ ${PROFILE:-0} == 1 ]]; then
    while kill -0 "$BENCH" 2>/dev/null && [[ ! -e $OUT/bench/phase.profile ]]; do sleep 1; done
    if [[ -e $OUT/bench/phase.profile ]]; then
        PIDS=$(IFS=,; echo "${SVCS[*]}")
        "$PERF" record -e "$PERF_EVENT" -F 999 -g -p "$PIDS" -o "$OUT/perf.data" \
            -- sleep "${PROFILE_SECS:-30}" > "$OUT/perf.log" 2>&1
    fi
fi

wait "$BENCH"
rc=$?
BENCH=
echo "bench rc=$rc" | tee -a "$OUT/run.txt"
for ((i = 0; i < ${#SVCS[@]}; i++)); do
    # Ingress replicas print their peer-update counters as one JSON line per interval.
    grep '^{"t"' "$OUT/service.$i.log" > "$OUT/service.$i.stats.jsonl" 2>/dev/null
done
python3 "$HERE/cpu_sampler.py" summarize "$OUT"
exit $rc
