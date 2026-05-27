#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_xdc_apply_hardware_profile model-only

URL=${URL:?URL is required}
ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR is required}

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"

mkdir -p "$ARTIFACT_DIR"

AIPERF_BIN=${AIPERF_BIN:-}
AIPERF_UVX=${AIPERF_UVX:-uvx}
AIPERF_CONCURRENCY=8
AIPERF_REQUEST_COUNT=200
AIPERF_INPUT_TOKENS=1101
AIPERF_OUTPUT_TOKENS=256
AIPERF_ENDPOINT_TYPE=chat
AIPERF_STREAMING=true
AIPERF_EXTRA_INPUTS=ignore_eos:true

kvbm_xdc_resolve_aiperf_cmd() {
  if [ -n "$AIPERF_BIN" ]; then
    if command -v "$AIPERF_BIN" >/dev/null 2>&1 || [ -x "$AIPERF_BIN" ]; then
      AIPERF_CMD=("$AIPERF_BIN")
      return 0
    fi
    echo "AIPERF_BIN is set but not executable or on PATH: $AIPERF_BIN" >&2
    return 2
  fi

  if command -v aiperf >/dev/null 2>&1; then
    AIPERF_CMD=(aiperf)
    return 0
  fi

  if command -v "$AIPERF_UVX" >/dev/null 2>&1; then
    AIPERF_CMD=("$AIPERF_UVX" aiperf)
    return 0
  fi

  echo "AIPerf was not found. Install aiperf, set AIPERF_BIN, or make uvx available." >&2
  return 2
}

kvbm_xdc_configure_aiperf_cache() {
  local root=${AIPERF_CACHE_ROOT:-}
  [ -n "$root" ] || return 0
  mkdir -p "$root"
  export UV_CACHE_DIR=$root/uv
  export PIP_CACHE_DIR=$root/pip
  export HF_HOME=$root/hf
  export TRANSFORMERS_CACHE=$HF_HOME
}

kvbm_xdc_write_env() {
  local file=$1
  local key=$2
  local value=${3:-}
  printf '%s=%q\n' "$key" "$value" >>"$file"
}

kvbm_xdc_url_scheme() {
  case "$URL" in
    *://*) printf '%s\n' "${URL%%://*}" ;;
    *) printf 'unknown\n' ;;
  esac
}

kvbm_xdc_url_port() {
  local endpoint=${URL#*://}
  endpoint=${endpoint%%/*}
  case "$endpoint" in
    *:*) printf '%s\n' "${endpoint##*:}" ;;
    *) printf '\n' ;;
  esac
}

kvbm_xdc_find_profile_json() {
  if [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ]; then
    printf '%s\n' "$ARTIFACT_DIR/profile_export_aiperf.json"
    return
  fi
  find "$ARTIFACT_DIR" -type f -name profile_export_aiperf.json | sort | head -n 1
}

kvbm_xdc_jq_value() {
  local json=$1
  local query=$2
  jq -r "$query // empty" "$json" 2>/dev/null || true
}

kvbm_xdc_delta() {
  local current=$1
  local baseline=$2
  awk -v current="$current" -v baseline="$baseline" '
    BEGIN {
      if (current == "" || baseline == "") {
        exit 1
      }
      printf "%.6f", current - baseline
    }' 2>/dev/null || true
}

kvbm_xdc_count_pattern() {
  local pattern=$1
  shift
  if [ "$#" -eq 0 ]; then
    printf '0\n'
    return
  fi
  local count
  set +e
  count=$(grep -h -E "$pattern" "$@" 2>/dev/null | wc -l | tr -d ' ')
  set -e
  printf '%s\n' "${count:-0}"
}

kvbm_xdc_count_failure_pattern() {
  local pattern=$1
  local exclude_zero_summary=$2
  shift 2
  if [ "$#" -eq 0 ]; then
    printf '0\n'
    return
  fi
  local count
  set +e
  count=$(grep -h -E "$pattern" "$@" 2>/dev/null | grep -v -E "$exclude_zero_summary" | wc -l | tr -d ' ')
  set -e
  printf '%s\n' "${count:-0}"
}

kvbm_xdc_env_value() {
  local file=$1
  local key=$2
  if [ ! -f "$file" ]; then
    return
  fi
  grep -E "^$key=" "$file" 2>/dev/null | head -n 1 | cut -d= -f2- || true
}

kvbm_xdc_metrics_env_value() {
  local file=$1
  local key=$2
  local value
  value=$(kvbm_xdc_env_value "$file" "$key")
  value=${value#\"}
  value=${value%\"}
  value=${value#\'}
  value=${value%\'}
  printf '%s\n' "$value"
}

kvbm_xdc_write_contract() {
  local contract_file=$ARTIFACT_DIR/aiperf-contract.env
  : >"$contract_file"
  kvbm_xdc_write_env "$contract_file" generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  kvbm_xdc_write_env "$contract_file" model "$MODEL"
  kvbm_xdc_write_env "$contract_file" framework dynamo-vllm
  kvbm_xdc_write_env "$contract_file" endpoint_type "$AIPERF_ENDPOINT_TYPE"
  kvbm_xdc_write_env "$contract_file" streaming "$AIPERF_STREAMING"
  kvbm_xdc_write_env "$contract_file" request_count "$AIPERF_REQUEST_COUNT"
  kvbm_xdc_write_env "$contract_file" concurrency "$AIPERF_CONCURRENCY"
  kvbm_xdc_write_env "$contract_file" synthetic_input_tokens_mean "$AIPERF_INPUT_TOKENS"
  kvbm_xdc_write_env "$contract_file" synthetic_input_tokens_stddev 0
  kvbm_xdc_write_env "$contract_file" output_tokens_mean "$AIPERF_OUTPUT_TOKENS"
  kvbm_xdc_write_env "$contract_file" output_tokens_stddev 0
  kvbm_xdc_write_env "$contract_file" extra_inputs "$AIPERF_EXTRA_INPUTS"
  kvbm_xdc_write_env "$contract_file" hardware_profile "$KVBM_HARDWARE_PROFILE"
  kvbm_xdc_write_env "$contract_file" gpu_class "$GPU_CLASS"
  kvbm_xdc_write_env "$contract_file" endpoint_url_redacted true
  kvbm_xdc_write_env "$contract_file" endpoint_url_scheme "$(kvbm_xdc_url_scheme)"
  kvbm_xdc_write_env "$contract_file" endpoint_url_port "$(kvbm_xdc_url_port)"
}

kvbm_xdc_write_postcheck() {
  local result_file=$1
  local log_dir=${POSTCHECK_LOG_DIR:-${ROLE_LOG_DIR:-}}
  if [ -z "$log_dir" ] || [ ! -d "$log_dir" ]; then
    kvbm_xdc_write_env "$result_file" postcheck_written false
    kvbm_xdc_write_env "$result_file" postcheck_reason log_dir_not_provided
    return
  fi

  local logs=()
  while IFS= read -r log_file; do
    logs+=("$log_file")
  done < <(find "$log_dir" -maxdepth 2 -type f -name '*.log' | sort)

  local postcheck_file=$ARTIFACT_DIR/aiperf-postcheck.env
  : >"$postcheck_file"
  kvbm_xdc_write_env "$postcheck_file" log_dir "$log_dir"
  kvbm_xdc_write_env "$postcheck_file" log_file_count "${#logs[@]}"
  kvbm_xdc_write_env "$postcheck_file" kvbm_audit_events "$(kvbm_xdc_count_pattern 'kvbm_audit' "${logs[@]}")"
  kvbm_xdc_write_env "$postcheck_file" session_pull_rdma_done "$(kvbm_xdc_count_pattern 'session_pull_rdma_done' "${logs[@]}")"
  kvbm_xdc_write_env "$postcheck_file" worker_session_pull_returned "$(kvbm_xdc_count_pattern 'worker_session_pull_returned' "${logs[@]}")"
  kvbm_xdc_write_env "$postcheck_file" worker_g2_to_g1_done "$(kvbm_xdc_count_pattern 'worker_g2_to_g1_done' "${logs[@]}")"
  kvbm_xdc_write_env "$postcheck_file" offload_register_complete "$(kvbm_xdc_count_pattern 'offload_register_complete' "${logs[@]}")"
  kvbm_xdc_write_env "$postcheck_file" external_cache_hit_lines "$(kvbm_xdc_count_pattern '[Ee]xternal prefix cache hit rate|external_cache_hit|external_cache' "${logs[@]}")"
  kvbm_xdc_write_env "$postcheck_file" kv_load_failure_events "$(kvbm_xdc_count_failure_pattern 'KV load failure|Onboarding failed|Failed to start onboarding|kv_load_failure' 'KV load failure events:[[:space:]]*0|kv_load_failure_events=0' "${logs[@]}")"
  kvbm_xdc_write_env "$postcheck_file" nixl_create_xfer_req_failures "$(kvbm_xdc_count_failure_pattern 'createXferReq: no potential backend found|nixl_create_xfer_req_failures' 'createXferReq failures:[[:space:]]*0|nixl_create_xfer_req_failures=0' "${logs[@]}")"

  local trace_gate_file=$log_dir/trace-gate.env
  if [ -f "$trace_gate_file" ]; then
    kvbm_xdc_write_env "$postcheck_file" trace_gate_file "$trace_gate_file"
    kvbm_xdc_write_env "$postcheck_file" trace_useful "$(kvbm_xdc_env_value "$trace_gate_file" trace_useful)"
    kvbm_xdc_write_env "$postcheck_file" trace_external_cache_hit_events "$(kvbm_xdc_env_value "$trace_gate_file" external_cache_hit_events)"
    kvbm_xdc_write_env "$postcheck_file" trace_kv_load_failure_events "$(kvbm_xdc_env_value "$trace_gate_file" kv_load_failure_events)"
    kvbm_xdc_write_env "$postcheck_file" trace_nixl_create_xfer_req_failures "$(kvbm_xdc_env_value "$trace_gate_file" nixl_create_xfer_req_failures)"
  fi
  kvbm_xdc_write_env "$result_file" postcheck_written true
  kvbm_xdc_write_env "$result_file" postcheck_file "$postcheck_file"
}

kvbm_xdc_write_comparison() {
  local result_file=$1
  local current_json=$2
  local baseline_json=${BASELINE_AIPERF_JSON:-}
  local baseline_metrics_env=${BASELINE_METRICS_ENV:-}
  if [ -z "$baseline_json" ] && [ -z "$baseline_metrics_env" ]; then
    kvbm_xdc_write_env "$result_file" comparison_written false
    kvbm_xdc_write_env "$result_file" comparison_reason baseline_not_provided
    return
  fi
  if [ -n "$baseline_json" ] && [ ! -f "$baseline_json" ]; then
    kvbm_xdc_write_env "$result_file" comparison_written false
    kvbm_xdc_write_env "$result_file" comparison_reason baseline_missing
    return
  fi
  if [ -z "$baseline_json" ] && [ ! -f "$baseline_metrics_env" ]; then
    kvbm_xdc_write_env "$result_file" comparison_written false
    kvbm_xdc_write_env "$result_file" comparison_reason baseline_metrics_env_missing
    return
  fi
  if ! command -v jq >/dev/null 2>&1; then
    kvbm_xdc_write_env "$result_file" comparison_written false
    kvbm_xdc_write_env "$result_file" comparison_reason jq_missing
    return
  fi

  local comparison_file=$ARTIFACT_DIR/experiment-comparison.env
  : >"$comparison_file"
  kvbm_xdc_write_env "$comparison_file" current_aiperf_json "$current_json"
  if [ -n "$baseline_json" ]; then
    kvbm_xdc_write_env "$comparison_file" baseline_source aiperf_json
    kvbm_xdc_write_env "$comparison_file" comparison_input_canonical_aiperf_json true
    kvbm_xdc_write_env "$comparison_file" baseline_aiperf_json "$baseline_json"
  else
    kvbm_xdc_write_env "$comparison_file" baseline_source metrics_env
    kvbm_xdc_write_env "$comparison_file" comparison_input_canonical_aiperf_json false
    kvbm_xdc_write_env "$comparison_file" baseline_metrics_env "$baseline_metrics_env"
    kvbm_xdc_write_env "$comparison_file" baseline_source_type "$(kvbm_xdc_metrics_env_value "$baseline_metrics_env" baseline_source_type)"
    kvbm_xdc_write_env "$comparison_file" baseline_derived_from_log "$(kvbm_xdc_metrics_env_value "$baseline_metrics_env" derived_from_log)"
    kvbm_xdc_write_env "$comparison_file" baseline_source_log "$(kvbm_xdc_metrics_env_value "$baseline_metrics_env" source_log)"
    kvbm_xdc_write_env "$comparison_file" baseline_source_log_sha256 "$(kvbm_xdc_metrics_env_value "$baseline_metrics_env" source_log_sha256)"
    kvbm_xdc_write_env "$comparison_file" baseline_source_excerpt_line_hint "$(kvbm_xdc_metrics_env_value "$baseline_metrics_env" source_excerpt_line_hint)"
    kvbm_xdc_write_env "$comparison_file" baseline_benchmark_id "$(kvbm_xdc_metrics_env_value "$baseline_metrics_env" benchmark_id)"
  fi

  local metric
  local current
  local baseline
  local baseline_env_key
  local delta
  for metric in \
    request_count.avg:request_count \
    request_throughput.avg:request_throughput \
    output_token_throughput.avg:output_token_throughput \
    time_to_first_token.avg:ttft_avg_ms \
    time_to_first_token.p50:ttft_p50_ms \
    time_to_first_token.p90:ttft_p90_ms \
    time_to_first_token.p99:ttft_p99_ms \
    request_latency.p50:request_latency_p50_ms \
    request_latency.p99:request_latency_p99_ms \
    inter_token_latency.p50:itl_p50_ms \
    inter_token_latency.p99:itl_p99_ms \
    output_sequence_length.avg:output_sequence_length_avg \
    osl_mismatch_count.avg:osl_mismatch_count; do
    baseline_env_key=${metric#*:}
    metric=${metric%%:*}
    local key=${metric//./_}
    current=$(kvbm_xdc_jq_value "$current_json" ".$metric")
    if [ -n "$baseline_json" ]; then
      baseline=$(kvbm_xdc_jq_value "$baseline_json" ".$metric")
    else
      baseline=$(kvbm_xdc_metrics_env_value "$baseline_metrics_env" "$baseline_env_key")
    fi
    delta=$(kvbm_xdc_delta "$current" "$baseline")
    kvbm_xdc_write_env "$comparison_file" "current_$key" "$current"
    kvbm_xdc_write_env "$comparison_file" "baseline_$key" "$baseline"
    kvbm_xdc_write_env "$comparison_file" "delta_$key" "$delta"
  done

  kvbm_xdc_write_env "$result_file" comparison_written true
  kvbm_xdc_write_env "$result_file" comparison_file "$comparison_file"
}

kvbm_xdc_configure_aiperf_cache
kvbm_xdc_resolve_aiperf_cmd
kvbm_xdc_write_contract

set +e
"${AIPERF_CMD[@]}" profile \
  --url "$URL" \
  --model "$MODEL" \
  --endpoint-type "$AIPERF_ENDPOINT_TYPE" \
  --streaming \
  --num-requests "$AIPERF_REQUEST_COUNT" \
  --concurrency "$AIPERF_CONCURRENCY" \
  --request-timeout-seconds 3600 \
  --isl "$AIPERF_INPUT_TOKENS" \
  --osl "$AIPERF_OUTPUT_TOKENS" \
  --extra-inputs "$AIPERF_EXTRA_INPUTS" \
  --output-artifact-dir "$ARTIFACT_DIR" \
  --ui none
aiperf_status=$?
set -e

result_file=$ARTIFACT_DIR/aiperf-result.env
: >"$result_file"
kvbm_xdc_write_env "$result_file" aiperf_exit_code "$aiperf_status"
if [ "$aiperf_status" -eq 0 ]; then
  kvbm_xdc_write_env "$result_file" aiperf_success true
else
  kvbm_xdc_write_env "$result_file" aiperf_success false
fi

profile_json=
if [ "$aiperf_status" -eq 0 ]; then
  profile_json=$(kvbm_xdc_find_profile_json)
else
  kvbm_xdc_write_env "$result_file" profile_json_found false
  kvbm_xdc_write_env "$result_file" metrics_extracted false
  kvbm_xdc_write_env "$result_file" metrics_reason aiperf_failed
  kvbm_xdc_write_env "$result_file" comparison_written false
  kvbm_xdc_write_env "$result_file" comparison_reason aiperf_failed
fi

if [ "$aiperf_status" -eq 0 ] && [ -n "$profile_json" ]; then
  kvbm_xdc_write_env "$result_file" profile_json_found true
  kvbm_xdc_write_env "$result_file" profile_json "$profile_json"
  if command -v jq >/dev/null 2>&1; then
    kvbm_xdc_write_env "$result_file" metrics_extracted true
    kvbm_xdc_write_env "$result_file" request_count "$(kvbm_xdc_jq_value "$profile_json" '.request_count.avg')"
    kvbm_xdc_write_env "$result_file" request_throughput "$(kvbm_xdc_jq_value "$profile_json" '.request_throughput.avg')"
    kvbm_xdc_write_env "$result_file" output_token_throughput "$(kvbm_xdc_jq_value "$profile_json" '.output_token_throughput.avg')"
    kvbm_xdc_write_env "$result_file" ttft_avg_ms "$(kvbm_xdc_jq_value "$profile_json" '.time_to_first_token.avg')"
    kvbm_xdc_write_env "$result_file" ttft_p50_ms "$(kvbm_xdc_jq_value "$profile_json" '.time_to_first_token.p50')"
    kvbm_xdc_write_env "$result_file" ttft_p90_ms "$(kvbm_xdc_jq_value "$profile_json" '.time_to_first_token.p90')"
    kvbm_xdc_write_env "$result_file" ttft_p99_ms "$(kvbm_xdc_jq_value "$profile_json" '.time_to_first_token.p99')"
    kvbm_xdc_write_env "$result_file" request_latency_p50_ms "$(kvbm_xdc_jq_value "$profile_json" '.request_latency.p50')"
    kvbm_xdc_write_env "$result_file" request_latency_p99_ms "$(kvbm_xdc_jq_value "$profile_json" '.request_latency.p99')"
    kvbm_xdc_write_env "$result_file" itl_p50_ms "$(kvbm_xdc_jq_value "$profile_json" '.inter_token_latency.p50')"
    kvbm_xdc_write_env "$result_file" itl_p99_ms "$(kvbm_xdc_jq_value "$profile_json" '.inter_token_latency.p99')"
    kvbm_xdc_write_env "$result_file" output_sequence_length_avg "$(kvbm_xdc_jq_value "$profile_json" '.output_sequence_length.avg')"
    kvbm_xdc_write_env "$result_file" osl_mismatch_count "$(kvbm_xdc_jq_value "$profile_json" '.osl_mismatch_count.avg')"
    kvbm_xdc_write_comparison "$result_file" "$profile_json"
  else
    kvbm_xdc_write_env "$result_file" metrics_extracted false
    kvbm_xdc_write_env "$result_file" metrics_reason jq_missing
    kvbm_xdc_write_env "$result_file" comparison_written false
    kvbm_xdc_write_env "$result_file" comparison_reason jq_missing
  fi
elif [ "$aiperf_status" -eq 0 ]; then
  kvbm_xdc_write_env "$result_file" profile_json_found false
  kvbm_xdc_write_env "$result_file" metrics_extracted false
  kvbm_xdc_write_env "$result_file" metrics_reason profile_json_missing
  kvbm_xdc_write_env "$result_file" comparison_written false
  kvbm_xdc_write_env "$result_file" comparison_reason profile_json_missing
fi

kvbm_xdc_write_postcheck "$result_file"

exit "$aiperf_status"
