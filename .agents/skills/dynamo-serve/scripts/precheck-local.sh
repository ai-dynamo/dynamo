#!/usr/bin/env bash
# precheck-local.sh — workstation readiness check for python3 -m dynamo.<backend>.
#
# Confirms: Python 3.10+, ai-dynamo + backend extras installed, GPU
# visible, model config loadable, HF token configured if needed.
#
# Implements PASS/FAIL/WARN per SKILL_AUTHORING.md §8.3 (A9).
#
# Usage:
#   bash scripts/precheck-local.sh -b <backend> -m <model> [--gated]

set -uo pipefail

usage() {
    cat <<USAGE
Usage: $0 -b <backend> -m <model> [--gated]
  -b  Backend: vllm | trtllm | sglang | mocker. Required.
  -m  Model: HF ID or local path. Required.
  --gated  Probe HF token presence (skip for ungated models).
  -h  Show this help.
USAGE
}

BACKEND=""
MODEL=""
GATED=0

while [ $# -gt 0 ]; do
    case "$1" in
        -b) BACKEND="$2"; shift 2 ;;
        -m) MODEL="$2"; shift 2 ;;
        --gated) GATED=1; shift ;;
        -h) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$BACKEND" ] || [ -z "$MODEL" ]; then
    echo "Error: -b <backend> and -m <model> are required." >&2
    usage >&2
    exit 2
fi

case "$BACKEND" in
    vllm|trtllm|sglang|mocker) ;;
    *) echo "Error: -b must be vllm|trtllm|sglang|mocker" >&2; exit 2 ;;
esac

PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { ((PASS++)); RESULTS+=("PASS|$1|$2"); }
fail() { ((FAIL++)); RESULTS+=("FAIL|$1|$2"); }
warn() { ((WARN++)); RESULTS+=("WARN|$1|$2"); }

# 1. Python 3.10+
if py_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null); then
    py_major=$(echo "$py_ver" | cut -d. -f1)
    py_minor=$(echo "$py_ver" | cut -d. -f2)
    if [ "$py_major" -eq 3 ] && [ "$py_minor" -ge 10 ]; then
        pass "Python 3.10+" "Python $py_ver"
    else
        fail "Python 3.10+" "Python $py_ver — requires 3.10 or newer"
    fi
else
    fail "Python 3.10+" "python3 not on PATH"
fi

# 2. ai-dynamo wheel
if pip show ai-dynamo &>/dev/null; then
    ver=$(pip show ai-dynamo 2>/dev/null | awk '/^Version:/ {print $2}')
    pass "ai-dynamo wheel" "ai-dynamo $ver"
else
    fail "ai-dynamo wheel" "ai-dynamo not installed; run: pip install 'ai-dynamo[$BACKEND]==<release>'"
fi

# 3. Backend module importable
if python3 -c "import dynamo.$BACKEND" 2>/dev/null; then
    pass "dynamo.$BACKEND importable" "module loads"
else
    fail "dynamo.$BACKEND importable" "python3 -c 'import dynamo.$BACKEND' failed; the [$BACKEND] extra may be missing"
fi

# 4. GPU visible (skip for mocker)
if [ "$BACKEND" != "mocker" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
        if [ "$gpu_count" -gt 0 ]; then
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
            pass "GPU visible" "$gpu_count× $gpu_name (free=${free_mb} MiB on GPU 0)"
        else
            fail "GPU visible" "nvidia-smi returned no GPUs"
        fi
    else
        fail "GPU visible" "nvidia-smi not on PATH (or no GPU driver installed)"
    fi
else
    warn "GPU visible" "skipped for mocker backend"
fi

# 5. Model config loadable
if python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$MODEL')" 2>/dev/null; then
    pass "model config loadable" "$MODEL"
else
    fail "model config loadable" "could not load '$MODEL' (gated? typo? offline?)"
fi

# 6. HF token (gated only)
if [ "$GATED" = "1" ]; then
    if [ -n "${HF_TOKEN:-}" ] || [ -f "$HOME/.cache/huggingface/token" ]; then
        if python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami())" 2>/dev/null | grep -q "name"; then
            pass "HF auth" "token resolves; whoami succeeds"
        else
            fail "HF auth" "token present but whoami failed — token expired or revoked"
        fi
    else
        fail "HF auth" "no HF_TOKEN env var and no ~/.cache/huggingface/token; gated model will 401 (per D2 / local-dev variant)"
    fi
fi

# 7. Port 8000 availability (best-effort)
if command -v lsof >/dev/null 2>&1; then
    if lsof -i:8000 -P -n 2>/dev/null | grep -q LISTEN; then
        occupant=$(lsof -i:8000 -P -n 2>/dev/null | grep LISTEN | head -1 | awk '{print $1, "PID", $2}')
        warn "port 8000 free" "in use by $occupant — pass --port <N> to dynamo.<backend>"
    else
        pass "port 8000 free" "no LISTEN on 8000"
    fi
fi

# Summary
echo
echo "===== Pre-Check Summary ====="
for row in "${RESULTS[@]}"; do echo "$row"; done
echo
echo "Passed: $PASS   Failed: $FAIL   Warned: $WARN"
echo
if [ "$FAIL" -gt 0 ]; then
    echo "Resolve the FAILs above before attempting Phase 3 (Run)."
    exit 1
fi
echo "Ready for Phase 2 (Configure) and Phase 3 (Run):"
echo "  python3 -m dynamo.$BACKEND --model $MODEL [flags...]"
exit 0
