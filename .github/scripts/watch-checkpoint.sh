#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Bounded observer inside this run's isolated vCluster; stop only this test on helper timeout.
set -uo pipefail
diag_dir=checkpoint-diagnostics
deadline=$((SECONDS + 1800))
agent=
restore_progress=
restore_progress_at=0
while ((SECONDS < deadline)); do
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    sample="$diag_dir/$stamp"
    mkdir -p "$sample"
    kubectl --request-timeout=10s -n default get pods -o json > "$sample/pods.json" || true
    kubectl --request-timeout=10s -n default get podsnapshotcontents,snapshotjobs,podsnapshots -o json > "$sample/checkpoint-status.json" 2> "$sample/status-errors.log" || true
    while IFS=$'\t' read -r capture node; do
        [[ -n "$capture" && -n "$node" ]] || continue
        kubectl --request-timeout=10s -n default logs "$capture" -c main --tail=100 > "$sample/worker.log" 2>&1 || true
        kubectl --request-timeout=10s get node "$node" -o json > "$sample/node.json" || true
        agent=$(jq -r --arg node "$node" '.items[] | select(.spec.nodeName == $node and .metadata.labels["app.kubernetes.io/component"] == "snapshot-agent" and .status.phase == "Running") | .metadata.name' "$sample/pods.json" | head -1)
        [[ -n "$agent" ]] || continue
        kubectl --request-timeout=10s -n default logs "$agent" -c agent --tail=300 > "$sample/$agent.log" 2>&1 || true
        timeout 25s kubectl -n default exec -i "$agent" -c agent -- bash -s < .github/scripts/checkpoint-process-probe.sh > "$sample/$agent-processes.log" 2>&1 || true
        # This PVC belongs exclusively to this diagnostic vCluster; never inspect host stores.
        timeout 15s kubectl -n default exec "$agent" -c agent -- bash -c 'find /checkpoints -maxdepth 5 -type f -name "dump.log" -print -exec tail -c 32768 {} \;' > "$sample/capture-files.log" 2>&1 || true
        if jq -e --arg name "$capture" '.items[] | select(.metadata.name == $name) | .metadata.annotations["nvidia.com/restore-from"]' "$sample/pods.json" >/dev/null; then
            timeout 10s kubectl -n default exec "$capture" -c main -- tail -c 32768 /var/criu-work/restore.log > "$sample/restore.log" 2>&1 || true
            progress=$(awk -F '\t' '/Starting external restore|Executing go-criu Restore call|CRIU pre-restore|CRIU post-restore|Restored process table|Resolved manifest CUDA PIDs|cuda-checkpoint-helper command succeeded/ {print $5}' "$sample/$agent.log" | tail -1)
            if ((restore_progress_at == 0)) || [[ -n "$progress" && "$progress" != "$restore_progress" ]]; then
                restore_progress=$progress
                restore_progress_at=$SECONDS
                printf 'Restore progress: %s\n' "${progress:-waiting for restore agent}"
            fi
        fi
    done < <(jq -r '.items[] | select((.metadata.name | startswith("checkpoint-")) or .metadata.annotations["nvidia.com/restore-from"] != null) | [.metadata.name, (.spec.nodeName // "")] | @tsv' "$sample/pods.json" 2>/dev/null)
    if [[ -n "$agent" ]]; then
        timeout 10s kubectl -n default exec "$agent" -c agent -- tail -100 /tmp/checkpoint-helper-actions.log > "$sample/helper-actions.log" 2>&1 || true
        helper_failed=$(timeout 10s kubectl -n default exec "$agent" -c agent -- bash -c 'if [ -f /tmp/checkpoint-diagnostic-helper-failed ]; then printf yes; fi' 2>/dev/null)
        restore_ready=$(jq -r '[.items[] | select(.metadata.annotations["nvidia.com/restore-from"] != null) | .status.conditions[]? | select(.type == "Ready" and .status == "True")] | length > 0' "$sample/pods.json")
        if { [[ "$helper_failed" == yes ]] || { ((restore_progress_at > 0 && SECONDS - restore_progress_at >= 120)) && [[ "$restore_ready" != true ]]; }; } && [[ -f "$diag_dir/pytest.pid" ]]; then
            read -r pytest_pid < "$diag_dir/pytest.pid"
            if [[ "$pytest_pid" =~ ^[1-9][0-9]*$ ]] && tr '\0' ' ' < "/proc/$pytest_pid/cmdline" | grep -q 'python -m pytest tests/deploy/test_dynamocheckpoint.py'; then
                printf 'Helper failed or restore made no progress for 120 seconds; diagnostics saved; stopping pytest PID %s\n' "$pytest_pid"
                kill -TERM "$pytest_pid"
                exit 0
            fi
        fi
    fi
    sleep 15
done
