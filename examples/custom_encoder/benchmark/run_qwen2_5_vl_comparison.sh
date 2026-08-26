#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Audited one-H100 comparison: synchronous control vs dynamo.vllm CustomEncoder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKLOAD_ROOT="${DYN_BENCH_WORKLOAD_ROOT:-/dynamo-tmp/logs/08-05/qwen25-user-ensemble-textisl644-osl7-mixed1000/workload}"
OUTPUT_ROOT="${DYN_BENCH_OUTPUT_ROOT:?set DYN_BENCH_OUTPUT_ROOT to a new directory under /workspace/logs}"
REPETITIONS="${DYN_BENCH_REPETITIONS:-3}"
CONCURRENCY="${DYN_BENCH_CONCURRENCY:-64}"
CONTAINER_IMAGE="${DYN_BENCH_CONTAINER_IMAGE:-unknown}"
SOURCE_COMMIT="${DYN_BENCH_SOURCE_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
KV_EVENT_BASE_PORT="${DYN_BENCH_KV_EVENT_BASE_PORT:-20080}"
EXPECTED_INPUT_SHA256="743e859f895ee0e22df2476f74e5d3fa4d48db059273f5fe517634f31d9ef7cc"

MEASURED_INPUT="$WORKLOAD_ROOT/measured/image_custom_1000_textisl644.jsonl"
WARMUP_INPUT="$WORKLOAD_ROOT/warmup/image_custom_20_textisl644.jsonl"
MANIFEST="$WORKLOAD_ROOT/workload_manifest.json"
for path in "$MEASURED_INPUT" "$WARMUP_INPUT" "$MANIFEST"; do
    test -f "$path"
done
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"

actual_input_sha256="$(sha256sum "$MEASURED_INPUT" | awk '{print $1}')"
if [[ "$actual_input_sha256" != "$EXPECTED_INPUT_SHA256" ]]; then
    echo >&2 "Measured workload SHA mismatch: $actual_input_sha256"
    exit 1
fi

python -m examples.custom_encoder.benchmark.fixed_text_image_workload validate \
    "$WORKLOAD_ROOT" \
    --image-size-count 300x300:500 \
    --image-size-count 500x500:500 \
    > "$OUTPUT_ROOT/workload_audit.txt"

jq -e '
    (.decoder_model == "Qwen/Qwen2.5-1.5B-Instruct")
    and (.encoder_model == "Qwen/Qwen2.5-VL-3B-Instruct")
    and (.concurrency == 64)
    and (.requests_per_concurrency == 1000)
    and (.warmup_requests == 20)
    and (.text_isl == 644)
    and (.target_osl == 7)
    and (.observed_decoder_isl_by_image_size == {"300x300":773,"500x500":976})
    and ([.image_size_counts[] | [.width, .height, .unique_images, .requests]]
         == [[300,300,500,500],[500,500,500,500]])
' "$MANIFEST" >/dev/null

cp "$MANIFEST" "$OUTPUT_ROOT/workload_manifest.json"
printf '%s\n' "$SOURCE_COMMIT" > "$OUTPUT_ROOT/source_commit.txt"
printf '%s\n' "$CONTAINER_IMAGE" > "$OUTPUT_ROOT/container_image.txt"
sha256sum "$MEASURED_INPUT" "$WARMUP_INPUT" > "$OUTPUT_ROOT/workload_sha256.txt"
sha256sum \
    components/src/dynamo/vllm/multimodal_utils/custom_encoder/batcher.py \
    examples/custom_encoder/benchmark/batched_control_worker.py \
    examples/custom_encoder/benchmark/fixed_text_image_workload.py \
    examples/custom_encoder/benchmark/run_qwen2_5_vl_comparison.sh \
    examples/custom_encoder/launch/agg_qwen2_5_vl_benchmark.sh \
    examples/custom_encoder/launch/agg_qwen2_5_vl_control.sh \
    examples/custom_encoder/qwen2_5_vl_benchmark_encoder.py \
    > "$OUTPUT_ROOT/source_sha256.txt"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DYN_HTTP_PORT=8000
export DYN_SYSTEM_PORT=8081
export DYN_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export DYN_ENCODER_CLASS=examples.custom_encoder.qwen2_5_vl_benchmark_encoder.Qwen2_5VLBenchmarkEncoder
export DYN_QWEN2_VL_ENCODER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE=1536
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY=64
export DYN_QWEN2_VL_MAX_BATCH_PATCHES=10368
export DYN_QWEN2_VL_MAX_BATCH_ITEMS=8
export DYN_QWEN2_VL_MAX_QUEUE_DELAY_US=1000
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS=1,2,4,8
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES=300x300,500x500
export DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE=0
export DYN_CUSTOM_ENCODER_DISPATCH_LOG=1
export DYN_CONTROL_MAX_BATCH_ITEMS=8
export DYN_CONTROL_MAX_QUEUE_DELAY_US=1000
export DYN_MAX_MODEL_LEN=2048
export DYN_MAX_NUM_SEQS=64
export DYN_VLLM_GPU_MEMORY_UTILIZATION=0.4
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

export VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen25-control-comparison
export XDG_CACHE_HOME=/tmp/xdg-cache-qwen25-control-comparison
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,power.limit,clocks.max.sm,memory.total \
    --format=csv,noheader,nounits > "$OUTPUT_ROOT/gpu_all.txt"
sed -n '1p' "$OUTPUT_ROOT/gpu_all.txt" > "$OUTPUT_ROOT/gpu.txt"
python -c 'import torch; assert torch.cuda.device_count() == 1'

python - "$MEASURED_INPUT" "$OUTPUT_ROOT/smoke_request.json" <<'PY'
import base64
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
row = json.loads(input_path.read_text(encoding="utf-8").splitlines()[0])
image_path = Path(row["image"])
image_url = "data:image/jpeg;base64," + base64.b64encode(
    image_path.read_bytes()
).decode("ascii")
payload = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": row["text"]},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    "temperature": 0,
    "min_tokens": 7,
    "max_tokens": 7,
    "ignore_eos": True,
    "stream": False,
    "nvext": {"extra_fields": ["completion_token_ids"]},
}
output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
PY

COMMON_AIPERF_ARGS=(
    --model Qwen/Qwen2.5-1.5B-Instruct
    --url http://127.0.0.1:8000
    --endpoint-type chat
    --endpoint /v1/chat/completions
    --custom-dataset-type single_turn
    --extra-inputs max_tokens:7
    --extra-inputs min_tokens:7
    --extra-inputs ignore_eos:true
    --extra-inputs temperature:0
    --extra-inputs stream:false
    --random-seed 42
    --workers-max 20
    --record-processors 32
    --request-timeout-seconds 300
    --ui none
    --no-server-metrics
    --use-server-token-count
)

server_pid=
cleanup_server() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=
}
trap cleanup_server EXIT

wait_for_server() {
    local log_path=$1
    local models_path=$2
    for _ in $(seq 1 900); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            sed -n '1,240p' "$log_path" >&2
            return 1
        fi
        if curl -fsS http://127.0.0.1:8000/v1/models \
            > "$models_path" 2>/dev/null \
            && jq -e '.data | length > 0' "$models_path" >/dev/null; then
            return 0
        fi
        sleep 1
    done
    sed -n '1,240p' "$log_path" >&2
    return 1
}

verify_server_stopped() {
    for _ in $(seq 1 30); do
        if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo >&2 "Server still listening after cleanup"
    return 1
}

run_index=0
run_arm() {
    local repetition=$1
    local arm=$2
    local arm_dir="$OUTPUT_ROOT/rep-$repetition/$arm"
    local launcher
    local kv_event_port
    local kv_events_config
    local benchmark_log_first_line
    local result
    local zmq_prefix

    case "$arm" in
        custom-worker-control)
            launcher=./examples/custom_encoder/launch/agg_qwen2_5_vl_control.sh
            ;;
        dynamo-vllm-custom)
            launcher=./examples/custom_encoder/launch/agg_qwen2_5_vl_benchmark.sh
            ;;
        *)
            echo >&2 "Unknown benchmark arm: $arm"
            return 1
            ;;
    esac

    run_index=$((run_index + 1))
    kv_event_port=$((KV_EVENT_BASE_PORT + run_index))
    kv_events_config="$(printf \
        '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:%s","enable_kv_cache_events":true}' \
        "$kv_event_port")"

    mkdir -p "$arm_dir/warmup" "$arm_dir/measured"
    rm -rf "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
    mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
    jq -n \
        --arg arm "$arm" \
        --argjson repetition "$repetition" \
        --argjson kv_event_port "$kv_event_port" \
        '{
            arm: $arm,
            repetition: $repetition,
            max_model_len: 2048,
            max_num_seqs: 64,
            gpu_memory_utilization: 0.4,
            prefix_caching: true,
            kv_event_publishing: true,
            kv_event_port: $kv_event_port,
            max_batch_items: 8,
            queue_delay_us: 1000,
            graph_buckets: [1,2,4,8],
            graph_image_sizes: ["300x300","500x500"]
        }' > "$arm_dir/run_config.json"

    printf 'SERVER_START=%s\n' "$(date -u +%FT%TZ)" \
        | tee "$arm_dir/timestamps.txt"
    setsid bash "$launcher" \
        --enable-prefix-caching \
        --kv-events-config "$kv_events_config" \
        > "$arm_dir/server.log" 2>&1 &
    server_pid=$!
    wait_for_server "$arm_dir/server.log" "$arm_dir/models.json"
    printf 'SERVER_READY=%s\n' "$(date -u +%FT%TZ)" \
        | tee -a "$arm_dir/timestamps.txt"

    curl -fsS http://127.0.0.1:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        --data-binary "@$OUTPUT_ROOT/smoke_request.json" \
        > "$arm_dir/smoke_response.json"
    jq -e '.nvext.completion_token_ids | length == 7' \
        "$arm_dir/smoke_response.json" >/dev/null
    jq '.nvext.completion_token_ids' "$arm_dir/smoke_response.json" \
        > "$arm_dir/smoke_token_ids.json"

    benchmark_log_first_line=$(( $(wc -l < "$arm_dir/server.log") + 1 ))
    zmq_prefix="/tmp/aiperf-qwen25-control-${repetition}-${arm}"
    rm -f "${zmq_prefix}-warmup"* "${zmq_prefix}-measured"* || true

    aiperf profile "${COMMON_AIPERF_ARGS[@]}" \
        --input-file "$WARMUP_INPUT" \
        --concurrency 20 \
        --conversation-num 20 \
        --artifact-dir "$arm_dir/warmup" \
        --zmq-ipc-path "${zmq_prefix}-warmup" \
        > "$arm_dir/warmup/client.log" 2>&1

    printf 'MEASURED_START=%s\n' "$(date -u +%FT%TZ)" \
        | tee -a "$arm_dir/timestamps.txt"
    TIMEFORMAT='%R'
    { time aiperf profile "${COMMON_AIPERF_ARGS[@]}" \
        --input-file "$MEASURED_INPUT" \
        --concurrency "$CONCURRENCY" \
        --conversation-num 1000 \
        --artifact-dir "$arm_dir/measured" \
        --zmq-ipc-path "${zmq_prefix}-measured" \
        > "$arm_dir/measured/client.log" 2>&1; } \
        2> "$arm_dir/full_wall_seconds.txt"
    printf 'MEASURED_END=%s\n' "$(date -u +%FT%TZ)" \
        | tee -a "$arm_dir/timestamps.txt"

    sed -n "${benchmark_log_first_line},\$p" "$arm_dir/server.log" \
        > "$arm_dir/benchmark_server.log"
    result="$arm_dir/measured/profile_export_aiperf.json"
    jq -e '
        (.request_count.avg == 1000)
        and ((.error_summary | length) == 0)
        and (.was_cancelled == false)
        and (.input_sequence_length.avg == 874.5)
        and (.output_sequence_length.avg == 7)
    ' "$result" >/dev/null

    cleanup_server
    verify_server_stopped
    printf 'PASS arm=%s repetition=%s\n' "$arm" "$repetition"
}

for repetition in $(seq 1 "$REPETITIONS"); do
    mkdir -p "$OUTPUT_ROOT/rep-$repetition"
    if (( repetition % 2 == 1 )); then
        arm_order=(custom-worker-control dynamo-vllm-custom)
    else
        arm_order=(dynamo-vllm-custom custom-worker-control)
    fi
    printf '%s\n' "${arm_order[*]}" > "$OUTPUT_ROOT/rep-$repetition/arm_order.txt"
    for arm in "${arm_order[@]}"; do
        run_arm "$repetition" "$arm"
    done
done

python - "$OUTPUT_ROOT" "$REPETITIONS" "$CONTAINER_IMAGE" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

root = Path(sys.argv[1])
repetitions = int(sys.argv[2])
container_image = sys.argv[3]
arms = ("custom-worker-control", "dynamo-vllm-custom")
dispatch_re = re.compile(r"custom_encoder_dispatch .*?patch_cost=(\d+)")


def sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_stdev": sample_stdev(values),
    }


runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
for repetition in range(1, repetitions + 1):
    smoke_tokens: dict[str, list[int]] = {}
    for arm in arms:
        arm_dir = root / f"rep-{repetition}" / arm
        profile = json.loads(
            (arm_dir / "measured/profile_export_aiperf.json").read_text()
        )
        wall_seconds = float((arm_dir / "full_wall_seconds.txt").read_text())
        server_log = (arm_dir / "server.log").read_text(errors="replace")
        benchmark_log = (arm_dir / "benchmark_server.log").read_text(
            errors="replace"
        )
        dispatch_costs = [int(value) for value in dispatch_re.findall(benchmark_log)]
        graph_captures = server_log.count("captured CUDA graph:")
        kv_events_ready = (
            "Worker reading KV events for dp_rank=" in server_log
            or "kv_events_enabled=True" in server_log
        )
        prefix_caching_ready = (
            "enable_prefix_caching=True" in server_log
            or "prefix_caching=True" in server_log
        )
        smoke_tokens[arm] = json.loads(
            (arm_dir / "smoke_token_ids.json").read_text()
        )

        if graph_captures != 8:
            raise SystemExit(
                f"REP_{repetition}_{arm}_GRAPH_CAPTURE_COUNT={graph_captures}"
            )
        if sum(dispatch_costs) != 907800:
            raise SystemExit(
                f"REP_{repetition}_{arm}_PATCH_AUDIT={sum(dispatch_costs)}"
            )
        if not kv_events_ready:
            raise SystemExit(f"REP_{repetition}_{arm}_KV_EVENTS_NOT_READY")
        if not prefix_caching_ready:
            raise SystemExit(f"REP_{repetition}_{arm}_PREFIX_CACHE_NOT_ENABLED")

        runs[arm].append(
            {
                "repetition": repetition,
                "full_client_process_wall_s": wall_seconds,
                "full_client_process_req_s": 1000.0 / wall_seconds,
                "request_window_req_s": profile["request_throughput"]["avg"],
                "output_token_throughput": profile[
                    "output_token_throughput"
                ]["avg"],
                "latency_ms": {
                    key: profile["request_latency"][key]
                    for key in ("avg", "p50", "p95", "p99", "max")
                },
                "request_count": profile["request_count"]["avg"],
                "input_sequence_length": profile["input_sequence_length"]["avg"],
                "output_sequence_length": profile["output_sequence_length"]["avg"],
                "errors": profile["error_summary"],
                "encoder_dispatch_calls": len(dispatch_costs),
                "encoder_patch_cost": sum(dispatch_costs),
                "graph_captures": graph_captures,
                "kv_events_ready": kv_events_ready,
                "prefix_caching_ready": prefix_caching_ready,
            }
        )

    if smoke_tokens[arms[0]] != smoke_tokens[arms[1]]:
        raise SystemExit(f"REP_{repetition}_SMOKE_TOKEN_PARITY_FAILED")

aggregates: dict[str, dict[str, Any]] = {}
for arm in arms:
    arm_runs = runs[arm]
    aggregates[arm] = {
        "full_client_process_wall_s": aggregate(
            [run["full_client_process_wall_s"] for run in arm_runs]
        ),
        "full_client_process_req_s": aggregate(
            [run["full_client_process_req_s"] for run in arm_runs]
        ),
        "request_window_req_s": aggregate(
            [run["request_window_req_s"] for run in arm_runs]
        ),
        "output_token_throughput": aggregate(
            [run["output_token_throughput"] for run in arm_runs]
        ),
        "latency_ms": {
            key: aggregate([run["latency_ms"][key] for run in arm_runs])
            for key in ("avg", "p50", "p95", "p99", "max")
        },
    }

control = aggregates["custom-worker-control"]
test = aggregates["dynamo-vllm-custom"]
per_repetition_ratios = []
for index in range(repetitions):
    control_run = runs["custom-worker-control"][index]
    test_run = runs["dynamo-vllm-custom"][index]
    per_repetition_ratios.append(
        {
            "repetition": index + 1,
            "full_client_process_req_s_test_over_control": (
                test_run["full_client_process_req_s"]
                / control_run["full_client_process_req_s"]
            ),
            "request_window_req_s_test_over_control": (
                test_run["request_window_req_s"]
                / control_run["request_window_req_s"]
            ),
            "latency_avg_test_over_control": (
                test_run["latency_ms"]["avg"]
                / control_run["latency_ms"]["avg"]
            ),
            "latency_p99_test_over_control": (
                test_run["latency_ms"]["p99"]
                / control_run["latency_ms"]["p99"]
            ),
        }
    )

summary = {
    "source_commit": (root / "source_commit.txt").read_text().strip(),
    "container": container_image,
    "gpu": (root / "gpu.txt").read_text().strip(),
    "workload": json.loads((root / "workload_manifest.json").read_text()),
    "config": {
        "repetitions": repetitions,
        "warmups_per_arm_repetition": 20,
        "measured_requests_per_arm_repetition": 1000,
        "concurrency": 64,
        "raw_text_tokens": 644,
        "decoder_isl_by_image_size": {"300x300": 773, "500x500": 976},
        "average_decoder_isl": 874.5,
        "min_tokens": 7,
        "max_tokens": 7,
        "ignore_eos": True,
        "stream": False,
        "max_model_len": 2048,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.4,
        "queue_delay_us": 1000,
        "max_batch_items": 8,
        "max_batch_patches": 10368,
        "graph_buckets": [1, 2, 4, 8],
        "graph_image_sizes": ["300x300", "500x500"],
        "expected_graph_captures": 8,
        "prefix_caching": True,
        "kv_event_publishing": True,
    },
    "runs": runs,
    "aggregate": aggregates,
    "ratios": {
        "meaning": "throughput >1 favors dynamo-vllm-custom; latency <1 favors it",
        "per_repetition": per_repetition_ratios,
        "full_client_process_req_s_test_over_control": (
            test["full_client_process_req_s"]["mean"]
            / control["full_client_process_req_s"]["mean"]
        ),
        "request_window_req_s_test_over_control": (
            test["request_window_req_s"]["mean"]
            / control["request_window_req_s"]["mean"]
        ),
        "latency_avg_test_over_control": (
            test["latency_ms"]["avg"]["mean"]
            / control["latency_ms"]["avg"]["mean"]
        ),
        "latency_p99_test_over_control": (
            test["latency_ms"]["p99"]["mean"]
            / control["latency_ms"]["p99"]["mean"]
        ),
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# Qwen2.5 CustomEncoder benchmark",
    "",
    f"- Source commit: `{summary['source_commit']}`",
    f"- Container: `{summary['container']}`",
    f"- GPU: `{summary['gpu']}`",
    "- Workload: 1000 measured requests, 20 excluded warmups, closed-loop C64, "
    "644 shared raw text tokens, OSL 7, balanced unique 300x300/500x500 JPEGs.",
    "",
    "## Runs",
    "",
    "| Arm | Rep | Full req/s | Window req/s | Avg E2E ms | P99 E2E ms | Dispatches |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for arm in arms:
    for run in runs[arm]:
        lines.append(
            f"| {arm} | {run['repetition']} | "
            f"{run['full_client_process_req_s']:.3f} | "
            f"{run['request_window_req_s']:.3f} | "
            f"{run['latency_ms']['avg']:.3f} | "
            f"{run['latency_ms']['p99']:.3f} | "
            f"{run['encoder_dispatch_calls']} |"
        )
lines.extend(
    [
        "",
        "## Aggregate",
        "",
        "| Arm | Full req/s mean +/- s | Window req/s mean +/- s | Avg E2E ms mean +/- s | P99 E2E ms mean +/- s |",
        "|---|---:|---:|---:|---:|",
    ]
)
for arm in arms:
    values = aggregates[arm]
    lines.append(
        f"| {arm} | {values['full_client_process_req_s']['mean']:.3f} +/- "
        f"{values['full_client_process_req_s']['sample_stdev']:.3f} | "
        f"{values['request_window_req_s']['mean']:.3f} +/- "
        f"{values['request_window_req_s']['sample_stdev']:.3f} | "
        f"{values['latency_ms']['avg']['mean']:.3f} +/- "
        f"{values['latency_ms']['avg']['sample_stdev']:.3f} | "
        f"{values['latency_ms']['p99']['mean']:.3f} +/- "
        f"{values['latency_ms']['p99']['sample_stdev']:.3f} |"
    )
lines.extend(
    [
        "",
        "## Ratios",
        "",
        "`dynamo-vllm-custom / custom-worker-control`: throughput above 1 favors the test arm; latency below 1 favors it.",
        "",
        f"- Full client-process throughput: {summary['ratios']['full_client_process_req_s_test_over_control']:.4f}",
        f"- Request-window throughput: {summary['ratios']['request_window_req_s_test_over_control']:.4f}",
        f"- Mean E2E latency: {summary['ratios']['latency_avg_test_over_control']:.4f}",
        f"- P99 E2E latency: {summary['ratios']['latency_p99_test_over_control']:.4f}",
        "",
        "All arms passed the workload SHA, zero-error, ISL/OSL, 907800-patch, eight-graph, KV-event, prefix-cache, and greedy smoke-token parity audits.",
    ]
)
(root / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
print("QWEN2_5_CUSTOM_ENCODER_COMPARISON_PASS")
PY
