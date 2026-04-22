#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-artifact CycloneDX SBOM orchestrator.
#
#   generate_sbom.sh ((--scan-target <syft-uri>)... | --image <ref>) \
#                    --framework <dynamo|vllm|sglang|trtllm> \
#                    --output-dir <dir> --name <component-name> \
#                    [--no-inject]
#
# Scan target selection (one of):
#   --scan-target <syft-uri>   Pass-through to `syft scan`. REPEATABLE — pass
#                              the flag once per location to scan and the
#                              resulting components are merged (de-duplicated
#                              by purl, falling back to bom-ref) into one BOM.
#                              Examples:
#                                dir:/                          (in-Dockerfile FS scan)
#                                docker:nvcr.io/nvidia/dynamo:0.9.1
#                                oci-archive:./image.tar
#                                registry:nvcr.io/nvidia/dynamo:0.9.1
#   --image <ref>              Convenience shortcut. Equivalent to a single
#                                --scan-target docker:<ref>
#
# Output: $OUTPUT_DIR/<name>.cdx.json (CycloneDX 1.6, with source-binary
# injection unless --no-inject is given) plus a sibling .sha256 checksum.
#
# This script does not render ATTRIBUTIONS-*.md. Run render_attributions.py
# against the SBOM for that step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAN_TARGETS=()
IMAGE=""
FRAMEWORK=""
OUTPUT_DIR=""
COMPONENT_NAME=""
NO_INJECT=0

usage() {
  sed -n '3,28p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scan-target)      SCAN_TARGETS+=("$2"); shift 2 ;;
    --image)            IMAGE="$2"; shift 2 ;;
    --framework)        FRAMEWORK="$2"; shift 2 ;;
    --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
    --name)             COMPONENT_NAME="$2"; shift 2 ;;
    --no-inject)        NO_INJECT=1; shift ;;
    -h|--help)          usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

if [[ ${#SCAN_TARGETS[@]} -eq 0 && -z "$IMAGE" ]]; then
  echo "ERROR: at least one --scan-target or one --image is required" >&2
  usage
fi
if [[ -z "$FRAMEWORK" || -z "$OUTPUT_DIR" || -z "$COMPONENT_NAME" ]]; then
  echo "ERROR: --framework, --output-dir, and --name are required" >&2
  usage
fi

if [[ -n "$IMAGE" ]]; then
  SCAN_TARGETS+=("docker:${IMAGE}")
fi

if ! command -v syft >/dev/null 2>&1; then
  echo "ERROR: syft not found on PATH. Install from https://github.com/anchore/syft" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
SBOM_PATH="$OUTPUT_DIR/${COMPONENT_NAME}.cdx.json"

if [[ ${#SCAN_TARGETS[@]} -eq 1 ]]; then
  echo ">> syft scan ${SCAN_TARGETS[0]} -> $SBOM_PATH"
  syft scan "${SCAN_TARGETS[0]}" \
    --output "cyclonedx-json=${SBOM_PATH}" \
    --source-name "$COMPONENT_NAME"
else
  TEMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TEMP_DIR"' EXIT
  PARTIAL_FILES=()
  for i in "${!SCAN_TARGETS[@]}"; do
    TARGET="${SCAN_TARGETS[$i]}"
    PARTIAL="${TEMP_DIR}/partial_${i}.cdx.json"
    echo ">> syft scan [$((i+1))/${#SCAN_TARGETS[@]}] $TARGET -> $PARTIAL"
    syft scan "$TARGET" \
      --output "cyclonedx-json=${PARTIAL}" \
      --source-name "$COMPONENT_NAME"
    PARTIAL_FILES+=("$PARTIAL")
  done
  echo ">> merging ${#PARTIAL_FILES[@]} partial SBOMs -> $SBOM_PATH"
  python3 - "${SBOM_PATH}" "${PARTIAL_FILES[@]}" <<'PYTHON_MERGE'
import json
import sys
from datetime import datetime, timezone

output_file, *partial_files = sys.argv[1:]
seen: set[str] = set()
merged: list[dict] = []
base = None
for pf in partial_files:
    with open(pf, "r", encoding="utf-8") as f:
        sbom = json.load(f)
    if base is None:
        base = sbom
    for comp in sbom.get("components", []) or []:
        key = comp.get("purl") or comp.get("bom-ref") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(comp)
if base is None:
    print("ERROR: no partial SBOMs to merge", file=sys.stderr)
    sys.exit(1)
base["components"] = merged
base.setdefault("metadata", {})["timestamp"] = datetime.now(timezone.utc).isoformat()
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(base, f, indent=2)
    f.write("\n")
print(f"merged {len(merged)} unique components -> {output_file}", file=sys.stderr)
PYTHON_MERGE
fi

# Emit SHA256 checksum (use shasum on macOS, sha256sum on Linux)
echo ">> computing SHA256 checksum"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$SBOM_PATH" | awk '{print $1}' > "${SBOM_PATH}.sha256"
else
  shasum -a 256 "$SBOM_PATH" | awk '{print $1}' > "${SBOM_PATH}.sha256"
fi
echo ">> wrote ${SBOM_PATH}.sha256"

if [[ "$NO_INJECT" -eq 0 ]]; then
  echo ">> injecting source-compiled binaries ($FRAMEWORK)"
  python3 "$SCRIPT_DIR/inject_source_binaries.py" \
    --bom "$SBOM_PATH" \
    --framework "$FRAMEWORK"
else
  echo ">> skipping source-binary injection (--no-inject)"
fi

echo ">> done: $SBOM_PATH"
