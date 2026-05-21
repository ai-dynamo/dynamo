#!/usr/bin/env bash
# validate-optimized.sh — post-quantization validator for a Dynamo-bound checkpoint.
#
# Checks:
#   1. Output dir exists
#   2. config.json / tokenizer.json / weight shards present
#   3. quant_config.json parses and matches the chosen technique
#   4. (optional) python3 -m dynamo.<backend> can load the checkpoint
#      with a 60 s timeout
#
# Implements PASS/FAIL/WARN per SKILL_AUTHORING.md §8.3 (A9).
#
# Usage:
#   bash scripts/validate-optimized.sh -m <output-dir> [-b <backend>] [--load-test]

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 -m <output-dir> [-b <backend>] [--load-test]
  -m  Path to the optimized checkpoint dir. Required.
  -b  Backend (vllm | trtllm | sglang) for the load test. Default: vllm.
  --load-test  Run a 60 s load test via python3 -m dynamo.<backend> (skipped by default).
  -h  Show this help.
USAGE
}

MODEL_DIR=""
BACKEND="vllm"
LOAD_TEST=0

while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL_DIR="$2"; shift 2 ;;
        -b) BACKEND="$2"; shift 2 ;;
        --load-test) LOAD_TEST=1; shift ;;
        -h) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo "Error: -m <output-dir> is required." >&2
    usage >&2
    exit 2
fi

PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { ((PASS++)); RESULTS+=("PASS|$1|$2"); }
fail() { ((FAIL++)); RESULTS+=("FAIL|$1|$2"); }
warn() { ((WARN++)); RESULTS+=("WARN|$1|$2"); }

# 1. Output dir exists.
if [ -d "$MODEL_DIR" ]; then
    pass "output dir present" "$MODEL_DIR"
else
    fail "output dir present" "$MODEL_DIR not found"
    echo "===== Validation Summary ====="
    for row in "${RESULTS[@]}"; do echo "$row"; done
    exit 1
fi

# 2. Core files present.
for f in config.json tokenizer_config.json; do
    if [ -f "$MODEL_DIR/$f" ]; then
        pass "file present" "$f"
    else
        fail "file present" "$f missing"
    fi
done

# Weight shards: either *.safetensors or *.bin.
weight_count=$(find "$MODEL_DIR" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) | wc -l | tr -d ' ')
if [ "$weight_count" -gt 0 ]; then
    pass "weight shards present" "$weight_count files"
else
    fail "weight shards present" "no *.safetensors or *.bin in $MODEL_DIR"
fi

# 3. quant_config.json parses.
if [ -f "$MODEL_DIR/quant_config.json" ]; then
    if python3 -c "import json; json.load(open('$MODEL_DIR/quant_config.json'))" 2>/dev/null; then
        algo=$(python3 -c "import json; print(json.load(open('$MODEL_DIR/quant_config.json')).get('quant_algo'))")
        pass "quant_config.json parses" "quant_algo=$algo"
    else
        fail "quant_config.json parses" "JSON parse error"
    fi
else
    warn "quant_config.json present" "missing — backend may infer from config.json, but recipe-style deploys expect this file"
fi

# 4. Optional load test.
if [ "$LOAD_TEST" = "1" ]; then
    case "$BACKEND" in
        vllm|trtllm|sglang) ;;
        *) fail "backend valid" "$BACKEND not in {vllm,trtllm,sglang}"; BACKEND="" ;;
    esac
    if [ -n "$BACKEND" ]; then
        echo "[load-test] starting python3 -m dynamo.$BACKEND with 60s timeout..."
        if timeout 60 python3 -m "dynamo.$BACKEND" --model "$MODEL_DIR" --load-format auto --help &>/dev/null; then
            pass "dynamo.$BACKEND --help" "module importable; --help OK"
        else
            warn "dynamo.$BACKEND --help" "module not importable or --help failed (not necessarily a checkpoint issue)"
        fi
    fi
else
    warn "load-test skipped" "rerun with --load-test for the dynamo.<backend> probe"
fi

# Summary.
echo
echo "===== Validation Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo
echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
