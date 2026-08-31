#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
INNER_SMOKE="${SCRIPT_DIR}/conditional_prefill_v1_smoke_test.sh"

IMAGE="${IMAGE:-}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-0.6B}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
PROBE_WARMUP_SECONDS="${PROBE_WARMUP_SECONDS:-180}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-600}"
EFF_ISL_THRESHOLD="${EFF_ISL_THRESHOLD:-2048}"
EFF_ISL_RATIO_THRESHOLD="${EFF_ISL_RATIO_THRESHOLD:-0.7}"
CONTAINER_NAME="${CONTAINER_NAME:-dynamo-condp-trtllm-smoke-$$}"
LOG_FILE="${LOG_FILE:-/tmp/${CONTAINER_NAME}.log}"

if [[ -z "$IMAGE" ]]; then
    echo "Set IMAGE to the Dynamo TensorRT-LLM runtime image." >&2
    exit 1
fi
if [[ ! -r "$INNER_SMOKE" ]]; then
    echo "Missing or unreadable inner smoke test: $INNER_SMOKE" >&2
    exit 1
fi
if [[ ! -d "$HF_CACHE_DIR" ]]; then
    echo "Missing Hugging Face cache directory: $HF_CACHE_DIR" >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker pull "$IMAGE"
fi

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Launching $IMAGE as $CONTAINER_NAME"
echo "Logs: $LOG_FILE"

docker run --detach \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc host \
    --volume "$INNER_SMOKE:/opt/dynamo/conditional_prefill_v1_smoke_test.sh:ro" \
    --volume "$HF_CACHE_DIR:/tmp/hf_cache" \
    --env MODEL_PATH="$MODEL_PATH" \
    --env SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    --env PROBE_WARMUP_SECONDS="$PROBE_WARMUP_SECONDS" \
    --env PROBE_INTER_REQUEST_SLEEP="${PROBE_INTER_REQUEST_SLEEP:-0.25}" \
    --env PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-24}" \
    --env EFF_ISL_THRESHOLD="$EFF_ISL_THRESHOLD" \
    --env EFF_ISL_RATIO_THRESHOLD="$EFF_ISL_RATIO_THRESHOLD" \
    "$IMAGE" \
    bash -lc '
        set -e
        export ETCD_ENDPOINTS=http://127.0.0.1:2379
        export NATS_SERVER=nats://127.0.0.1:4222
        /usr/local/bin/etcd/etcd \
            --data-dir /tmp/dynamo-smoke-etcd \
            --name dynamo-smoke \
            --listen-client-urls http://127.0.0.1:2379 \
            --advertise-client-urls http://127.0.0.1:2379 \
            --listen-peer-urls http://127.0.0.1:2380 \
            --initial-advertise-peer-urls http://127.0.0.1:2380 \
            --initial-cluster dynamo-smoke=http://127.0.0.1:2380 \
            >/tmp/etcd.log 2>&1 &
        nats-server -p 4222 >/tmp/nats.log 2>&1 &
        for _ in $(seq 1 30); do
            if curl -sf http://127.0.0.1:2379/health >/dev/null; then
                break
            fi
            sleep 1
        done
        curl -sf http://127.0.0.1:2379/health >/dev/null
        exec bash /opt/dynamo/conditional_prefill_v1_smoke_test.sh
    ' >/dev/null

deadline=$((SECONDS + SMOKE_TIMEOUT_SECONDS))
completed=false
while (( SECONDS < deadline )); do
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -q '\[probe\] done'; then
        completed=true
        break
    fi
    if ! docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true; then
        echo "Smoke container exited before the probe completed" >&2
        break
    fi
    sleep 5
done

docker logs "$CONTAINER_NAME" >"$LOG_FILE" 2>&1 || true
if [[ "$completed" != true ]]; then
    tail -120 "$LOG_FILE" >&2
    exit 1
fi

docker stop --time 10 "$CONTAINER_NAME" >/dev/null
docker logs "$CONTAINER_NAME" >"$LOG_FILE" 2>&1 || true

python3 - "$LOG_FILE" "$EFF_ISL_THRESHOLD" "$EFF_ISL_RATIO_THRESHOLD" <<'PYEOF'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
absolute_threshold = int(sys.argv[2])
ratio_threshold = float(sys.argv[3])
text = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(errors="replace"))
lines = text.splitlines()

request_ends = [line for line in lines if "[probe]" in line and re.search(r"\bEND\s+\(", line)]
request_failures = [line for line in lines if "[probe]" in line and "FAILED:" in line]
decision_lines = [line for line in lines if "Conditional disagg decision" in line]


def field(line: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}=([^\s]+)", line)
    if not match:
        raise ValueError(f"missing {name}: {line}")
    return match.group(1).strip('"')


mismatches = []
bypass_count = 0
for index, line in enumerate(decision_lines, 1):
    prompt_tokens = int(field(line, "prompt_tokens"))
    net_new_tokens = int(field(line, "net_new_tokens"))
    actual = field(line, "bypass").lower() == "true"
    expected = (
        net_new_tokens < absolute_threshold
        and net_new_tokens / max(prompt_tokens, 1) < ratio_threshold
    )
    bypass_count += int(actual)
    if actual != expected:
        mismatches.append(
            f"decision {index}: prompt={prompt_tokens} net_new={net_new_tokens} "
            f"expected={expected} actual={actual}"
        )

routed_count = text.count("Conditional disagg routing to decode worker")
errors = []
if request_failures:
    errors.append(f"{len(request_failures)} probe requests failed")
if len(request_ends) != 15:
    errors.append(f"expected 15 completed requests, found {len(request_ends)}")
if len(decision_lines) != 15:
    errors.append(f"expected 15 policy decisions, found {len(decision_lines)}")
if mismatches:
    errors.extend(mismatches)
if not 0 < bypass_count < 15:
    errors.append(f"expected both bypass and remote-prefill decisions, bypasses={bypass_count}")
if routed_count != bypass_count:
    errors.append(
        f"bypass decisions ({bypass_count}) != local decode dispatches ({routed_count})"
    )

print(
    f"requests={len(request_ends)}/15 decisions={len(decision_lines)}/15 "
    f"bypasses={bypass_count} remote_prefills={len(decision_lines) - bypass_count} "
    f"policy_mismatches={len(mismatches)}"
)
if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)
PYEOF

echo "PASS: conditional-disagg TensorRT-LLM container smoke test"
echo "Log: $LOG_FILE"
