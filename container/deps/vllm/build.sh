#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_REF="v0.8.4"
FORK_REPO="https://github.com/WenhaoHe02/vllm_0_8_4"
FORK_REF="master"
OUTPUT_PATCH_ALL="vllm_v0.8.4-dynamo-kv-disagg-patch.patch"
OUTPUT_PATCH_CORE="vllm_v0.8.4-dynamo-kv-core-only.patch"

rm -rf tmp_original_repo tmp_fork_repo
git clone --depth 1 --branch "$ORIGINAL_REF" https://github.com/vllm-project/vllm.git tmp_original_repo

git clone --depth 1 --branch "$FORK_REF" "$FORK_REPO" tmp_fork_repo

echo "生成完整 patch 到 $OUTPUT_PATCH_ALL"
diff -ruN tmp_original_repo tmp_fork_repo > "$OUTPUT_PATCH_ALL" || true
echo "提取 core patch 到 $OUTPUT_PATCH_CORE"
awk '
  /^diff --git/ { include = ($3 ~ /^a\/vllm\//); }
  include { print }
' "$OUTPUT_PATCH_ALL" > "$OUTPUT_PATCH_CORE"

echo "  - 全量 patch: $OUTPUT_PATCH_ALL"
echo "  - 核心 patch: $OUTPUT_PATCH_CORE"
