#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One arm of the campaign, driven from the login node with srun steps in the keeper's
# allocation. The keeper's nodes, in order: A runs the Global Frontend, B the mock pools,
# C and D one aiperf each.
#
# usage: run_arm.sh LABEL direct
#        run_arm.sh LABEL frontend BUILD CPUS [FRONTEND_ARGS...]
# BUILD names $CAMPAIGN/builds/BUILD/global-frontend; CPUS is a numactl list on numa0.
# PROFILE=SECONDS also records an on-CPU perf profile of the frontend once the settle period ends.
# MOCK_ARGS are extra mock-pool flags, e.g. MOCK_ARGS="--itl-ms 2" to pace tokens.
set -euo pipefail
source "$(dirname "$0")/env.sh"

label=$1 kind=$2
shift 2
job=$(cat "$CAMPAIGN/control/job")
mapfile -t nodes < "$CAMPAIGN/control/nodes.txt"
A=${nodes[0]} B=${nodes[1]} C=${nodes[2]} D=${nodes[3]}
run=/var/tmp/gfstress/$label
out=$CAMPAIGN/results/$label
if [[ -e $out ]]; then
    echo "$out exists" >&2
    exit 1
fi
mkdir -p "$out"

# One srun step on HOST running node.sh, in the foreground.
step() {
    local host=$1 name=$2
    shift 2
    srun --jobid="$job" --overlap -N1 -n1 -w "$host" --cpus-per-task=144 --mem=0 --cpu-bind=none \
        -J "gfs-$label-$name" env EXPECT_HOST="$host" "$BENCH/slurm/node.sh" "$@"
}
# The same, left running; cleanup stops it.
background() {
    step "$@" &
    echo $! >> "$out/steps"
}
log() { echo "$(date -Is) $label: $*" | tee -a "$out/run.log"; }
cleanup() {
    if [[ -e $out/steps ]]; then
        while read -r pid; do kill -TERM "$pid" 2> /dev/null || true; done < "$out/steps"
        sleep 5
    fi
    squeue -h -s -j "$job" -o '%i %j' | awk -v prefix="gfs-$label-" 'index($2, prefix) == 1 && $2 !~ /collect$/ {print $1}' |
        xargs -r scancel 2> /dev/null || true
    for host in "$A" "$B" "$C" "$D"; do step "$host" collect collect "$run" "$out" || true; done
    log "collected"
}

build=- cpus=-
if [[ $kind == frontend ]]; then
    build=$1 cpus=$2
    shift 2
fi
python3 -c 'import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], sys.argv[2::2]))))' \
    label "$label" kind "$kind" build "$build" cpus "$cpus" args "$*" mock_args "${MOCK_ARGS:-}" \
    frontend "$A" mocks "$B" client_c "$C" client_d "$D" > "$out/roles.json"

log "preflight on $A $B $C $D"
for host in "$A" "$B" "$C" "$D"; do step "$host" preflight preflight "$run" | tee -a "$out/run.log"; done
trap cleanup EXIT

address() { step "$1" address address "$run"; }
A_IP=$(address "$A")
B_IP=$(address "$B")
# shellcheck disable=SC2086 # MOCK_ARGS splits into flags
background "$B" mocks mocks "$run" "$CAMPAIGN/builds/mock/mock-pool" ${MOCK_ARGS:-}
health=()
for ((i = 0; i < POOLS; i++)); do health+=("http://$B_IP:$((POOL_PORT_BASE + i))/health"); done
step "$C" wait wait_http "$run" "${health[@]}"
log "$POOLS mocks up on $B ($B_IP, args: ${MOCK_ARGS:-none})"
if [[ $kind == frontend ]]; then
    background "$A" frontend frontend "$run" "$CAMPAIGN/builds/$build/global-frontend" "$cpus" "$B_IP" "$@"
    step "$C" wait wait_http "$run" "http://$A_IP:$FRONTEND_PORT/health"
    urls=http://$A_IP:$FRONTEND_PORT
    log "frontend $build up on $A ($A_IP, cpus $cpus, args: $*)"
else
    urls=$(for ((i = 0; i < POOLS; i++)); do printf 'http://%s:%d,' "$B_IP" $((POOL_PORT_BASE + i)); done)
    urls=${urls%,}
fi
step "$C" smoke smoke "$run" "${urls%%,*}" | tee -a "$out/run.log"
for host in "$A" "$B" "$C" "$D"; do background "$host" sampler sampler "$run"; done

log "aiperf starting on $C and $D"
step "$C" aiperf aiperf "$run" "$urls" > "$out/aiperf-$C.out" 2>&1 &
client_c=$!
step "$D" aiperf aiperf "$run" "$urls" > "$out/aiperf-$D.out" 2>&1 &
client_d=$!

until step "$C" phase phase "$run"; do sleep 2; done
started=$SECONDS
log "profiling started"
sleep "$SETTLE_S"
date +%s.%N > "$out/window-start"
[[ $kind == frontend ]] && step "$A" metrics metrics "$run" start
for host in "$A" "$B" "$C" "$D"; do step "$host" netstat netstat "$run" start; done
if [[ $kind == frontend && -n ${PROFILE:-} ]]; then
    log "perf for ${PROFILE}s"
    step "$A" perf perf "$run" "$PROFILE"
fi
remaining=$((SETTLE_S + MEASURE_S - (SECONDS - started)))
if ((remaining > 0)); then sleep "$remaining"; fi
date +%s.%N > "$out/window-end"
[[ $kind == frontend ]] && step "$A" metrics metrics "$run" end
for host in "$A" "$B" "$C" "$D"; do step "$host" netstat netstat "$run" end; done
wait "$client_c"
wait "$client_d"
log "aiperf done"
