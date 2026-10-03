#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Mirror the small, reproducibility-relevant campaign records into this branch.
# Bulky artifacts (traces, replay outputs, caches, PDFs) stay in the campaign root;
# their identities are kept here through the manifests and SHA-256 files.
set -euo pipefail

CR="${CR:?set CR to the campaign root}"
# The plan document, if it lives outside the campaign root (optional).
PLAN_SRC="${PLAN_SRC:-}"
DEST="$(cd "$(dirname "$0")" && pwd)/campaign"
MAX_BYTES=$((5 * 1024 * 1024))
BUDGET_BYTES=$((25 * 1024 * 1024))

copy() {
    local src="$1" rel="$2"
    [[ -f "$src" ]] || return 0
    local size
    size=$(stat -c %s "$src")
    if (( size > MAX_BYTES )); then
        echo "skip (>5 MiB): $src ($size bytes)" >&2
        return 0
    fi
    mkdir -p "$DEST/$(dirname "$rel")"
    cp -p "$src" "$DEST/$rel"
}

[[ -z "$PLAN_SRC" ]] || copy "$PLAN_SRC" PLAN.md
copy "$CR/CONTRACT.md" CONTRACT.md
copy "$CR/config/engine.json" config/engine.json
for f in "$CR"/facts/*.json "$CR"/facts/*.md; do copy "$f" "facts/$(basename "$f")"; done
for f in "$CR"/cells/*.json "$CR"/cells/*.jsonl; do copy "$f" "cells/$(basename "$f")"; done
copy "$CR/traces/MANIFEST.json" traces/MANIFEST.json
copy "$CR/traces/agentx_lowered/base_cap300/MANIFEST.json" traces/agentx_lowered_base_cap300_MANIFEST.json
for f in "$CR"/audits/*/*.md; do
    copy "$f" "audits/$(basename "$(dirname "$f")")/$(basename "$f")"
done
for d in "$CR"/traces/agentx_lowered/base_cap300_*; do
    [[ -d "$d" ]] || continue
    copy "$d/MANIFEST.json" "traces/agentx_lowered_$(basename "$d")_MANIFEST.json"
done
copy "$CR/runs/setup/scripts/write_engine_config.py" scripts/setup/write_engine_config.py
while IFS= read -r -d '' f; do
    copy "$f" "runs/${f#"$CR"/runs/}"
done < <(find "$CR/runs" -path "$CR/runs/cache" -prune -o -name cells_validation.json -type f -print0)
for d in "$CR"/runs/calibrate*/scripts; do
    [[ -d "$d" ]] || continue
    for f in "$d"/*; do
        [[ -f "$f" ]] && copy "$f" "scripts/$(basename "$(dirname "$d")")/$(basename "$f")"
    done
done
copy "$CR/literature/LESSONS.md" literature/LESSONS.md
copy "$CR/literature/BIBLIOGRAPHY.md" literature/BIBLIOGRAPHY.md
for f in "$CR"/literature/notes/*.md; do copy "$f" "literature/notes/$(basename "$f")"; done
for f in "$CR"/report/*.md; do copy "$f" "report/$(basename "$f")"; done
for f in "$CR"/report/fig/*.png "$CR"/report/fig/*.svg; do copy "$f" "report/fig/$(basename "$f")"; done

# Calibration decision tables and load curves, the inputs to every frozen load level and SLA.
for d in "$CR"/runs/calibrate "$CR"/runs/calibrate-fix-*; do
    [[ -d "$d" ]] || continue
    for f in "$d"/decisions*.json "$d"/curves*.json; do
        copy "$f" "runs/$(basename "$d")/$(basename "$f")"
    done
done

# Policy specs: every evaluated policy YAML, tarred so thousands of tiny files become one blob.
if [[ -d "$CR/runs/policies" ]]; then
    mkdir -p "$DEST/runs"
    tar -C "$CR/runs" -czf "$DEST/runs/policies.tar.gz" --sort=name --mtime='2026-01-01' policies
fi

# Training and evaluation outputs: best parameters, CMA-ES history, and per-cell results.
# Results JSONL is gzipped because per-replicate rows add up.
while IFS= read -r -d '' f; do
    rel="runs/${f#"$CR"/runs/}"
    case "$f" in
        */remote/returned/*/results.jsonl | */train/*/results.jsonl)
            # Duplicates of cache-ingested records; best.json and history.jsonl summarize training.
            ;;
        *.jsonl)
            mkdir -p "$DEST/$(dirname "$rel")"
            gzip -n -c "$f" > "$DEST/$rel.gz"
            if (( $(stat -c %s "$DEST/$rel.gz") > MAX_BYTES )); then
                echo "skip (>5 MiB gzipped): $f" >&2
                rm -f "$DEST/$rel.gz"
            fi
            ;;
        *) copy "$f" "$rel" ;;
    esac
done < <(find "$CR/runs" -path "$CR/runs/cache" -prune -o -path "$CR/runs/replicates" -prune -o \
    -path "$CR/runs/audit*" -prune -o -path '*/sweep*' -prune -o -path '*/hist_sweep*' -prune -o \
    -path '*_check' -prune -o -path "$CR/runs/agentx_lowered" -prune -o \
    \( -name 'best.json' -o -name 'best_policy.yaml' -o -name 'history.jsonl' -o -name 'results.jsonl' \
       -o -name 'space.yaml' -o -name 'gate*.json' -o -name 'pilot*.json' \) -type f -print0)

total=$(du -sb "$DEST" | cut -f1)
if (( total > BUDGET_BYTES )); then
    echo "WARNING: mirror is $((total / 1024 / 1024)) MiB, over the ${BUDGET_BYTES} byte budget" >&2
fi
du -sh "$DEST"
