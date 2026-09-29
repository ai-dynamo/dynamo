#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Diagnostic only: this run's vCluster and verified local pytest are the entire scope.
set -uo pipefail
[[ ${1:-} == --bounded ]] || exec timeout --signal=USR1 --kill-after=5s 420s bash "$0" --bounded
diag_dir=checkpoint-diagnostics
pytest_pid='' pytest_start='' agent=''
kube() { kubectl --request-timeout=5s -n default "$@"; }
start_time() { local stat; stat=$(<"/proc/$1/stat") || return 1; stat=${stat##*) }; awk '{print $20}' <<< "$stat"; }
stop_test() {
    printf '%s\n' "$1" | tee "$sample/fatal.txt"
    if [[ -n $agent && ${2:-} != deadline ]]; then
        timeout --signal=KILL 10s kubectl --request-timeout=5s -n default exec -i "$agent" -c agent -- bash -s \
            < .github/scripts/checkpoint-process-probe.sh > "$sample/$agent-processes.log" 2>&1 || true
    fi
    local -a argv=()
    if [[ -n $pytest_pid && $(start_time "$pytest_pid" 2>/dev/null) == "$pytest_start" ]]; then
        mapfile -d '' -t argv < "/proc/$pytest_pid/cmdline" 2>/dev/null || exit 0
        if [[ ${argv[1]:-} == -m && ${argv[2]:-} == pytest && ${argv[3]:-} == tests/deploy/test_dynamocheckpoint.py ]]; then
            kill -TERM "$pytest_pid"
        fi
    fi
    exit 0
}
trap 'stop_test "Overall diagnostic deadline reached (420 seconds)" deadline' USR1
while :; do
    if [[ -z $pytest_pid && -f $diag_dir/pytest.pid ]]; then
        read -r pytest_pid < "$diag_dir/pytest.pid"
        [[ $pytest_pid =~ ^[1-9][0-9]*$ ]] || exit 1
        pytest_start=$(start_time "$pytest_pid" 2>/dev/null) || exit 0
    fi
    [[ -z $pytest_pid || $(start_time "$pytest_pid" 2>/dev/null) == "$pytest_start" ]] || exit 0
    sample="$diag_dir/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$sample"
    kube get pods -o json > "$sample/pods.json" || { sleep 5; continue; }
    kube get podsnapshotcontents,snapshotjobs,podsnapshots -o json > "$sample/checkpoint-status.json" 2> "$sample/status-errors.log" || true
    # Successful CRIU capture kills its source with SIGKILL; bind that exception to the captured source UID.
    captured_uids=$(jq -c '[.items[] | select(.kind == "SnapshotJob" and any(.status.conditions[]?; .type == "Captured" and .status == "True")) | .status.podSnapshotUID] as $snapshots | [.items[] | select(.kind == "PodSnapshot" and (.metadata.uid as $uid | $snapshots | index($uid)) != null) | .spec.source.podRef.uid | select(type == "string" and length > 0)]' "$sample/checkpoint-status.json" 2>/dev/null) || captured_uids='[]'
    while IFS=$'\t' read -r target node failure; do
        [[ -n $target ]] || continue
        agent=$(jq -r --arg node "$node" '[.items[] | select(.spec.nodeName == $node and .metadata.labels["app.kubernetes.io/component"] == "snapshot-agent" and .status.phase == "Running") | .metadata.name][0] // empty' "$sample/pods.json")
        kube logs "$target" -c main --tail=100 > "$sample/$target.log" 2>&1 || true
        [[ -z $failure ]] || stop_test "$target: $failure"
        # This container has its own PID namespace; collect only driver paths.
        # shellcheck disable=SC2016
        timeout --kill-after=1s 5s kubectl --request-timeout=5s -n default exec "$target" -c main -- sh -c \
            'awk '\''/\/libcuda\.so/ {print FILENAME, $NF}'\'' /proc/[0-9]*/maps 2>/dev/null | sort -u' \
            > "$sample/$target-driver-maps.log" 2>&1 || true
        fatal=$(grep -E '^(\[[^]]+\][[:space:]]*)*(RuntimeError|torch\.[A-Za-z.]*Error):.*(CUDA|cuda|driver|GPU|PTX|TensorRT)' "$sample/$target.log" | head -1 || true)
        [[ -z $fatal ]] || stop_test "$target: $fatal"
        [[ -n $agent ]] || continue
        kube logs "$agent" -c agent --tail=100 > "$sample/$agent.log" 2>&1 || true
        timeout --kill-after=1s 5s kubectl --request-timeout=5s -n default exec -i "$agent" -c agent -- bash -s > "$sample/$agent-helpers.log" 2>&1 <<'PROBE'
set -u
own_cgroup=$(</proc/self/cgroup)
[[ -n $own_cgroup ]] || exit 1
hz=$(getconf CLK_TCK)
read -r uptime _ < /proc/uptime
for path in /proc/[0-9]*; do
    [[ -r $path/cgroup && $(<"$path/cgroup") == "$own_cgroup" ]] || continue
    executable=$(readlink "$path/exe" 2>/dev/null) || continue
    [[ ${executable##*/} == cuda-checkpoint-helper ]] || continue
    stat=$(<"$path/stat") || continue
    read -r -a fields <<< "${stat##*) }"
    (( ${#fields[@]} >= 20 )) || continue
    age=$(( ${uptime%%.*} - fields[19] / hz ))
    argv=(); mapfile -d '' -t argv < "$path/cmdline" 2>/dev/null || continue
    [[ ${argv[1]:-} == --action && ${argv[2]:-} =~ ^(lock|checkpoint|restore|unlock)$ ]] || continue
    current=$(<"$path/stat") || continue
    read -r -a current_fields <<< "${current##*) }"
    [[ ${current_fields[19]:-} == "${fields[19]}" && $(<"$path/cgroup") == "$own_cgroup" ]] || continue
    printf 'helper pid=%s start=%s action=%s age=%ss\n' "${path##*/}" "${fields[19]}" "${argv[2]}" "$age"
    (( age > 20 )) && printf 'HELPER_TIMEOUT pid=%s action=%s age=%ss\n' "${path##*/}" "${argv[2]}" "$age"
done
PROBE
        timed_out=$(grep '^HELPER_TIMEOUT ' "$sample/$agent-helpers.log" | head -1 || true)
        [[ -z $timed_out ]] || stop_test "$agent: $timed_out"
    done < <(jq -r --argjson captured "$captured_uids" '.items[] | select(.metadata.deletionTimestamp == null and ((.metadata.name | startswith("checkpoint-")) or .metadata.annotations["nvidia.com/restore-from"] != null))
        | select((.metadata.annotations["nvidia.com/restore-from"] == null and (.metadata.uid as $uid | $captured | index($uid)) != null and any(.status.containerStatuses[]?; .name == "main" and (.restartCount // 0) == 0 and .state.terminated.exitCode == 137 and .state.terminated.reason == "Error")) | not)
        | ([.status.containerStatuses[]? | select(.name == "main") | if (.restartCount // 0) > 0 then "main restarted" elif (.state.terminated.exitCode // 0) != 0 then "main terminated: \(.state.terminated.reason) exit=\(.state.terminated.exitCode)" elif ((.state.waiting.reason // "") | test("^(CrashLoopBackOff|CreateContainerConfigError|CreateContainerError|RunContainerError|InvalidImageName)$")) then .state.waiting.reason else empty end][0] // (if .status.phase == "Failed" then "pod failed: \(.status.reason // "unknown")" else "" end)) as $failure | [.metadata.name, (.spec.nodeName // "-"), $failure] | @tsv' "$sample/pods.json")
    sleep 5
done
