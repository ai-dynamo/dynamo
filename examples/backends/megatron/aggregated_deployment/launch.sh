#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a salloc allocation}"
: "${DMG_SQSH:?DMG_SQSH must be set (path to the phase0 sqsh on lustre)}"
: "${STAGE:?STAGE must be set (lustre staging dir)}"

[[ -f "$DMG_SQSH" ]] || { echo "DMG_SQSH not found: $DMG_SQSH" >&2; exit 1; }
[[ -d "$STAGE"   ]] || { echo "STAGE not found: $STAGE"        >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MOUNTS="$STAGE:$STAGE,$DYNAMO_ROOT:/workspace"
if [[ -n "${MEGATRON_LOCAL_DEV:-}" ]]; then
    [[ -d "$MEGATRON_LOCAL_DEV" ]] || { echo "MEGATRON_LOCAL_DEV not a dir: $MEGATRON_LOCAL_DEV" >&2; exit 1; }
    MOUNTS="$MOUNTS,$MEGATRON_LOCAL_DEV:/opt/megatron-lm"
    echo "[launch] live Megatron mount: $MEGATRON_LOCAL_DEV -> /opt/megatron-lm"
fi

EXPORT_VARS="STAGE,HF_HOME,HF_TOKEN,MODEL_CHECKPOINT,MODEL_DIR,TOKENIZER_MODEL,SERVED_MODEL_NAME"
EXPORT_VARS="$EXPORT_VARS,CONTEXT_LENGTH,TP,HTTP_PORT,COORD_PORT,MASTER_PORT"

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] expect 'READY' on stdout when ready"
echo

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    --export="ALL,$EXPORT_VARS" \
    bash /workspace/examples/backends/megatron/aggregated_deployment/orchestrate.sh
