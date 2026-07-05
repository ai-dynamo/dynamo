#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${CAMPAIGN_DIR}/campaign.env"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
usage: run.sh {smoke|full} --api-base URL --label LABEL [options]

Required:
  --api-base URL          OpenAI-compatible base URL including /v1
  --label LABEL           Stack label, e.g. dynamo-vllm or vllm-serve
  --phase PHASE           Campaign phase: validation, ab, or ba

Options:
  --model MODEL           LiteLLM model name (default: openai/<served-model>)
  --served-model MODEL    Model id returned by GET /models
  --runtime-binding PATH  Validated serving-runtime binding JSON
  --n-concurrent N        Concurrent Harbor trials (default: 4)
  --temperature FLOAT     Terminus-2 sampling temperature (default: 1.0)
  --top-p FLOAT           Nucleus sampling probability (default: 1.0)
  --max-turns N           Maximum Terminus-2 episodes (default: 500)
  --max-context N         Model context length advertised to LiteLLM
  --max-output N          Per-call output token limit advertised to LiteLLM
  --timeout-multiplier F  Harbor task timeout multiplier (default: 16 = 4h)
  --jobs-dir DIR          Parent directory for raw Harbor jobs
  --summary-dir DIR       Compact summary destination (default: JOB/summary)
  --job-name NAME         Stable job name (required with --resume)
  --resume                Resume an interrupted job with identical arguments
  --dry-run               Print the resolved Harbor config; run no tasks
  -h, --help              Show this help

Authentication is read only from OPENAI_API_KEY and is never written to
metadata. It defaults to "EMPTY" for unauthenticated local endpoints.

The full mode is fixed to terminal-bench/terminal-bench-2-1@6, all 89 tasks,
and 5 attempts per task (445 trials). Smoke mode runs the first 3 pinned tasks
once each.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

mode="$1"
shift
case "${mode}" in
  smoke)
    expected_tasks="${TERMINALBENCH_SMOKE_TASK_COUNT}"
    attempts=1
    ;;
  full)
    expected_tasks="${TERMINALBENCH_TASK_COUNT}"
    attempts="${TERMINALBENCH_OFFICIAL_ATTEMPTS}"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    usage >&2
    exit 2
    ;;
esac

api_base=""
label=""
campaign_phase=""
served_model="${SERVED_MODEL_NAME:-zai-org/GLM-5.2}"
model=""
n_concurrent="${TERMINALBENCH_CONCURRENCY:-4}"
temperature="${TERMINALBENCH_TEMPERATURE:-1.0}"
top_p="${TERMINALBENCH_TOP_P:-1.0}"
max_turns="${TERMINALBENCH_MAX_TURNS:-500}"
max_context="${TERMINALBENCH_MAX_CONTEXT_TOKENS:-262144}"
max_output="${TERMINALBENCH_MAX_OUTPUT_TOKENS:-48000}"
timeout_multiplier="${TERMINALBENCH_TIMEOUT_MULTIPLIER:-16}"
jobs_dir="${TERMINALBENCH_JOBS_DIR:-${SCRIPT_DIR}/runs}"
summary_dir=""
job_name=""
runtime_binding=""
resume=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-base)
      api_base="${2:?--api-base requires a value}"
      shift 2
      ;;
    --label)
      label="${2:?--label requires a value}"
      shift 2
      ;;
    --phase)
      campaign_phase="${2:?--phase requires a value}"
      shift 2
      ;;
    --model)
      model="${2:?--model requires a value}"
      shift 2
      ;;
    --served-model)
      served_model="${2:?--served-model requires a value}"
      shift 2
      ;;
    --runtime-binding)
      runtime_binding="${2:?--runtime-binding requires a value}"
      shift 2
      ;;
    --n-concurrent)
      n_concurrent="${2:?--n-concurrent requires a value}"
      shift 2
      ;;
    --temperature)
      temperature="${2:?--temperature requires a value}"
      shift 2
      ;;
    --top-p)
      top_p="${2:?--top-p requires a value}"
      shift 2
      ;;
    --max-turns)
      max_turns="${2:?--max-turns requires a value}"
      shift 2
      ;;
    --max-context)
      max_context="${2:?--max-context requires a value}"
      shift 2
      ;;
    --max-output)
      max_output="${2:?--max-output requires a value}"
      shift 2
      ;;
    --timeout-multiplier)
      timeout_multiplier="${2:?--timeout-multiplier requires a value}"
      shift 2
      ;;
    --jobs-dir)
      jobs_dir="${2:?--jobs-dir requires a value}"
      shift 2
      ;;
    --summary-dir)
      summary_dir="${2:?--summary-dir requires a value}"
      shift 2
      ;;
    --job-name)
      job_name="${2:?--job-name requires a value}"
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${api_base}" || -z "${label}" || -z "${campaign_phase}" ]]; then
  echo "--api-base, --label, and --phase are required" >&2
  usage >&2
  exit 2
fi
case "${label}" in
  dynamo-vllm|vllm-serve|dynamo-sglang|sglang-serve) ;;
  *)
    echo "--label must be one of the four campaign variants" >&2
    exit 2
    ;;
esac
case "${campaign_phase}" in
  validation|ab|ba) ;;
  *) echo "--phase must be validation, ab, or ba" >&2; exit 2 ;;
esac
if [[ "${mode}" == full && "${campaign_phase}" == validation ]]; then
  echo "Full runs require --phase ab or ba" >&2
  exit 2
fi
if [[ ! "${label}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--label must contain only letters, digits, dot, underscore, or dash" >&2
  exit 2
fi
if [[ ! "${n_concurrent}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--n-concurrent must be a positive integer" >&2
  exit 2
fi
if [[ "${mode}" == full && "${n_concurrent}" != "${TERMINALBENCH_CONCURRENCY}" ]]; then
  echo "Full runs require --n-concurrent ${TERMINALBENCH_CONCURRENCY}" >&2
  exit 2
fi
if [[ ! "${max_turns}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-turns must be a positive integer" >&2
  exit 2
fi
if [[ ! "${max_context}" =~ ^[1-9][0-9]*$ || ! "${max_output}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-context and --max-output must be positive integers" >&2
  exit 2
fi
if (( max_output >= max_context )); then
  echo "--max-output must be smaller than --max-context" >&2
  exit 2
fi
if [[ ! "${temperature}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--temperature must be a non-negative number" >&2
  exit 2
fi
if [[ ! "${top_p}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
  echo "--top-p must be between 0 and 1" >&2
  exit 2
fi
if [[ ! "${timeout_multiplier}" =~ ^([1-9][0-9]*([.][0-9]+)?|0+[.][0-9]*[1-9][0-9]*)$ ]]; then
  echo "--timeout-multiplier must be positive" >&2
  exit 2
fi
if (( resume == 1 )) && [[ -z "${job_name}" ]]; then
  echo "--resume requires an explicit --job-name" >&2
  exit 2
fi

api_base="${api_base%/}"
model="${model:-openai/${served_model}}"
if [[ "${mode}" == full ]]; then
  require_full_value() {
    local name="$1"
    local actual="$2"
    local expected="$3"
    if [[ "${actual}" != "${expected}" ]]; then
      echo "Full runs require ${name}=${expected}; found ${actual}" >&2
      exit 2
    fi
  }
  require_full_value served-model "${served_model}" "${SERVED_MODEL_NAME}"
  require_full_value model "${model}" "openai/${SERVED_MODEL_NAME}"
  require_full_value temperature "${temperature}" "${TERMINALBENCH_TEMPERATURE}"
  require_full_value top-p "${top_p}" "${TERMINALBENCH_TOP_P}"
  require_full_value max-turns "${max_turns}" "${TERMINALBENCH_MAX_TURNS}"
  require_full_value max-context "${max_context}" "${TERMINALBENCH_MAX_CONTEXT_TOKENS}"
  require_full_value max-output "${max_output}" "${TERMINALBENCH_MAX_OUTPUT_TOKENS}"
  require_full_value timeout-multiplier \
    "${timeout_multiplier}" "${TERMINALBENCH_TIMEOUT_MULTIPLIER}"
fi
max_input=$((max_context - max_output))
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
job_name="${job_name:-${label}-${campaign_phase}-terminalbench21-${mode}-${timestamp}}"
if [[ ! "${job_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--job-name must contain only letters, digits, dot, underscore, or dash" >&2
  exit 2
fi
if [[ "-${job_name}-" != *"-${campaign_phase}-"* ]]; then
  echo "--job-name must contain the campaign phase token ${campaign_phase}" >&2
  exit 2
fi
job_dir="${jobs_dir}/${job_name}"
summary_dir="${summary_dir:-${job_dir}/summary}"

if [[ -e "${job_dir}" && ${resume} -eq 0 && ${dry_run} -eq 0 ]]; then
  echo "Job directory already exists; use a new --job-name or pass --resume: ${job_dir}" >&2
  exit 1
fi

require_harbor
require_docker

model_info="{\"max_tokens\":${max_context},\"max_input_tokens\":${max_input},\"max_output_tokens\":${max_output},\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}"
harbor_command=(
  "${HARBOR_BIN}" run
  --dataset "${TERMINALBENCH_DATASET}"
  --agent "${TERMINUS_AGENT}"
  --model "${model}"
  --agent-kwarg "api_base=${api_base}"
  --agent-kwarg "parser_name=json"
  --agent-kwarg "max_turns=${max_turns}"
  --agent-kwarg "temperature=${temperature}"
  --agent-kwarg "model_info=${model_info}"
  --agent-kwarg "llm_call_kwargs={\"max_tokens\":${max_output},\"top_p\":${top_p}}"
  --n-attempts "${attempts}"
  --n-concurrent "${n_concurrent}"
  --max-retries 0
  --timeout-multiplier "${timeout_multiplier}"
  --jobs-dir "${jobs_dir}"
  --job-name "${job_name}"
  --env docker
  --delete
  --yes
)
if [[ "${mode}" == "smoke" ]]; then
  harbor_command+=(--n-tasks "${TERMINALBENCH_SMOKE_TASK_COUNT}")
fi

if (( dry_run == 1 )); then
  "${harbor_command[@]}" --print-config
  exit 0
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
runtime_binding="${runtime_binding:-/artifacts/glm52-nscale/runtime-bindings/${label}/active.json}"
if [[ ! -s "${runtime_binding}" ]]; then
  echo "Runtime binding is missing: ${runtime_binding}" >&2
  exit 1
fi
metadata_path="${job_dir}/run-metadata.json"
dataset_metadata_path="${job_dir}/dataset-metadata.json"
task_images_path="${job_dir}/task-images.json"
"${HARBOR_SOURCE_DIR}/.venv/bin/python" "${SCRIPT_DIR}/verify_dataset.py" \
  --dataset "${TERMINALBENCH_DATASET}" \
  --expected-content-hash "${TERMINALBENCH_DATASET_CONTENT_HASH}" \
  --expected-version-id "${TERMINALBENCH_DATASET_VERSION_ID}" \
  --expected-tasks "${TERMINALBENCH_TASK_COUNT}" \
  --output "${dataset_metadata_path}"
metadata_command=(
  python3 "${SCRIPT_DIR}/capture_metadata.py" start
  --output "${metadata_path}"
  --repo-root "${CAMPAIGN_DIR}/../.."
  --source-metadata /workspace/source-provenance.json
  --source-root /workspace
  --harbor-source "${HARBOR_SOURCE_DIR}"
  --harbor-bin "${HARBOR_BIN}"
  --harbor-version "${HARBOR_VERSION}"
  --harbor-commit "${HARBOR_COMMIT}"
  --dataset "${TERMINALBENCH_DATASET}"
  --dataset-revision "${TERMINALBENCH_DATASET_REVISION}"
  --dataset-content-hash "${TERMINALBENCH_DATASET_CONTENT_HASH}"
  --dataset-version-id "${TERMINALBENCH_DATASET_VERSION_ID}"
  --dataset-metadata "${dataset_metadata_path}"
  --mode "${mode}"
  --label "${label}"
  --campaign-phase "${campaign_phase}"
  --api-base "${api_base}"
  --runtime-binding "${runtime_binding}"
  --serving-context "${MAX_MODEL_LEN}"
  --served-model "${served_model}"
  --model "${model}"
  --agent "${TERMINUS_AGENT}"
  --expected-tasks "${expected_tasks}"
  --attempts "${attempts}"
  --n-concurrent "${n_concurrent}"
  --temperature "${temperature}"
  --top-p "${top_p}"
  --max-turns "${max_turns}"
  --max-context "${max_context}"
  --max-output "${max_output}"
  --timeout-multiplier "${timeout_multiplier}"
  --job-name "${job_name}"
  --job-dir "${job_dir}"
)
if (( resume == 1 )); then
  metadata_command+=(--resume)
fi
metadata_command+=(--command "${harbor_command[@]}")
"${metadata_command[@]}"

started_epoch="$(date +%s)"
set +e
"${harbor_command[@]}"
harbor_rc=$?
set -e
finished_epoch="$(date +%s)"

python3 "${SCRIPT_DIR}/capture_metadata.py" finish \
  --metadata "${metadata_path}" \
  --job-dir "${job_dir}" \
  --exit-code "${harbor_rc}" \
  --elapsed-seconds "$((finished_epoch - started_epoch))"

if (( harbor_rc == 0 )); then
  package_cache_dir="$(
    "${HARBOR_SOURCE_DIR}/.venv/bin/python" -c \
      'from harbor.constants import PACKAGE_CACHE_DIR; print(PACKAGE_CACHE_DIR)'
  )"
  "${HARBOR_SOURCE_DIR}/.venv/bin/python" \
    "${SCRIPT_DIR}/capture_terminal_task_images.py" \
    --job-dir "${job_dir}" \
    --dataset-metadata "${dataset_metadata_path}" \
    --package-cache-dir "${package_cache_dir}" \
    --expected-tasks "${expected_tasks}" \
    --expected-attempts "${attempts}" \
    --output "${task_images_path}"
fi

summary_rc=0
if [[ -f "${job_dir}/result.json" ]]; then
  set +e
  python3 "${SCRIPT_DIR}/summarize.py" \
    --job-dir "${job_dir}" \
    --output-dir "${summary_dir}" \
    --metadata "${metadata_path}" \
    --task-images "${task_images_path}" \
    --expected-tasks "${expected_tasks}" \
    --expected-attempts "${attempts}" \
    --strict
  summary_rc=$?
  set -e
else
  echo "Harbor did not produce ${job_dir}/result.json" >&2
  summary_rc=1
fi

if (( harbor_rc != 0 )); then
  echo "Harbor exited with status ${harbor_rc}; resume with --job-name ${job_name} --resume" >&2
  exit "${harbor_rc}"
fi
if (( summary_rc != 0 )); then
  echo "Summary validation failed with status ${summary_rc}: ${summary_dir}/summary.json" >&2
  exit "${summary_rc}"
fi

echo "Terminal-Bench ${mode} PASS"
echo "  job:     ${job_dir}"
echo "  summary: ${summary_dir}/summary.json"
