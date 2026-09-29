#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read-only, bounded observer inside this run's isolated vCluster.
set -uo pipefail
diag_dir=checkpoint-diagnostics
deadline=$((SECONDS + 1800))
while ((SECONDS < deadline)); do
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    sample="$diag_dir/$stamp"
    mkdir -p "$sample"
    kubectl --request-timeout=10s -n default get pods -o json > "$sample/pods.json" || true
    kubectl --request-timeout=10s -n default get podsnapshotcontents,snapshotjobs,dynamocheckpoints -o json > "$sample/checkpoint-status.json" 2> "$sample/status-errors.log" || true
    while IFS=$'\t' read -r capture node; do
        [[ -n "$capture" && -n "$node" ]] || continue
        kubectl --request-timeout=10s -n default logs "$capture" -c main --tail=100 > "$sample/worker.log" 2>&1 || true
        kubectl --request-timeout=10s get node "$node" -o json > "$sample/node.json" || true
        agent=$(jq -r --arg node "$node" '.items[] | select(.spec.nodeName == $node and .metadata.labels["app.kubernetes.io/component"] == "snapshot-agent" and .status.phase == "Running") | .metadata.name' "$sample/pods.json" | head -1)
        [[ -n "$agent" ]] || continue
        kubectl --request-timeout=10s -n default logs "$agent" -c agent --tail=300 > "$sample/agent.log" 2>&1 || true
        timeout 25s kubectl -n default exec -i "$agent" -c agent -- bash -s < .github/scripts/checkpoint-process-probe.sh > "$sample/processes.log" 2>&1 || true
        # This PVC belongs exclusively to this diagnostic vCluster; never inspect host stores.
        timeout 15s kubectl -n default exec "$agent" -c agent -- bash -c 'find /checkpoints -maxdepth 5 -type f -name "dump.log" -print -exec tail -c 32768 {} \;' > "$sample/capture-files.log" 2>&1 || true
    done < <(jq -r '.items[] | select(.metadata.name | startswith("checkpoint-")) | [.metadata.name, (.spec.nodeName // "")] | @tsv' "$sample/pods.json" 2>/dev/null)
    sleep 30
done
