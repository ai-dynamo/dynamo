#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stage the pinned Qwen3-32B checkpoint onto this node's local disk and apply the YaRN config.
# Usage: stage_weights.sh PLAN_DIR OUT_DIR   (on each allocated node's host, not in the image;
#        writes OUT_DIR/<hostname>.json)
#
# Source order: the verified shared-filesystem master (sizes checked first, every byte hashed after
# the copy); if it is missing or wrong and LR_ALLOW_HF_DOWNLOAD=1, a parallel Hugging Face download
# of the pinned revision straight to node-local disk. The token, if any, is read from
# LR_HF_TOKEN_FILE and never printed or logged (Qwen3-32B is public; a token only lifts limits).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 2 ]] || die "usage: stage_weights.sh PLAN_DIR OUT_DIR"
plan_dir=$1
out_json="$2/$(hostname).json"
py="${LR_HOST_PYTHON:-python3}"
weights=("$py" "$LR_DEPLOY_DIR/weights.py")
rope="$("$py" -c 'import json,sys; r=json.load(open(sys.argv[1]))["rope_scaling"]; print(json.dumps(r) if r else "")' "$plan_dir/engine_plan.json")"
mkdir -p "$(dirname "$out_json")"
report="$(mktemp "$LR_NODE_ROOT/stage-weights.XXXXXX")"

log "node=$(hostname) master=$LR_MODEL_MASTER dest=$LR_MODEL_DIR"
df -h "$LR_NODE_ROOT" | tail -1
if [[ -f "$LR_MODEL_DIR/config.json.orig" ]]; then
  log "node-local copy already present; re-verifying"
elif "${weights[@]}" verify --manifest "$LR_MODEL_MANIFEST" --sizes-only "$LR_MODEL_MASTER" >> "$report"; then
  "${weights[@]}" copy --manifest "$LR_MODEL_MANIFEST" --jobs "${LR_COPY_JOBS:-8}" \
    "$LR_MODEL_MASTER" "$LR_MODEL_DIR" >> "$report"
elif [[ "${LR_ALLOW_HF_DOWNLOAD:-0}" == 1 ]]; then
  log "shared-filesystem master unusable; downloading $LR_MODEL_ID@$LR_MODEL_REVISION from Hugging Face"
  source "$LR_TOOLCHAIN_ENV"
  token_env=()
  if [[ -n "${LR_HF_TOKEN_FILE:-}" ]]; then
    token_env=(HF_TOKEN="$(<"$LR_HF_TOKEN_FILE")")
  fi
  env "${token_env[@]}" UV_CACHE_DIR="$LR_CACHE_ROOT/uv" HF_HUB_ENABLE_HF_TRANSFER=0 \
    uvx --from 'huggingface_hub[hf_xet]==0.36.0' hf download "$LR_MODEL_ID" \
    --revision "$LR_MODEL_REVISION" --local-dir "$LR_MODEL_DIR" --max-workers 16 \
    --exclude '*.md' '.gitattributes' > "$LR_NODE_ROOT/hf-download.log" 2>&1
else
  die "shared-filesystem master failed the size check; set LR_ALLOW_HF_DOWNLOAD=1 to download instead"
fi

if [[ -f "$LR_MODEL_DIR/config.json.orig" ]]; then
  "${weights[@]}" verify --manifest "$LR_MODEL_MANIFEST" --jobs 16 --patched-config "$LR_MODEL_DIR" >> "$report"
else
  "${weights[@]}" verify --manifest "$LR_MODEL_MANIFEST" --jobs 16 "$LR_MODEL_DIR" >> "$report"
fi
if [[ -n "$rope" ]]; then
  "${weights[@]}" patch-config --manifest "$LR_MODEL_MANIFEST" --rope-scaling "$rope" "$LR_MODEL_DIR" >> "$report"
fi
"$py" - "$report" "$out_json" "$(hostname)" <<'PY'
import json, sys
steps = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
json.dump({"host": sys.argv[3], "steps": steps}, open(sys.argv[2], "w"), indent=1, sort_keys=True)
PY
log "staged weights on $(hostname): $out_json"
