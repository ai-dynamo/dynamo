#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=../../campaign.env
source "${SWEBENCH_EVAL_DIR}/../campaign.env"
export PYTHONDONTWRITEBYTECODE=1

usage() {
  printf 'usage: %s <verified|multilingual|pro> <run-name> [generate|evaluate|all] --phase <validation|ab|ba>\n' "$0" >&2
}

if [[ $# -lt 4 ]]; then
  usage
  exit 2
fi

SUITE="$1"
RUN_NAME="$2"
shift 2
PHASE=all
if [[ "${1:-}" != --phase ]]; then
  PHASE="$1"
  shift
fi
if [[ $# -ne 2 || "$1" != --phase ]]; then
  usage
  exit 2
fi
CAMPAIGN_PHASE="$2"
EXPECTED="$(suite_expected_count "${SUITE}")"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'run-name may contain only letters, digits, dot, underscore, and dash\n' >&2
  exit 2
fi
if [[ "${PHASE}" != generate && "${PHASE}" != evaluate && "${PHASE}" != all ]]; then
  usage
  exit 2
fi
case "${CAMPAIGN_PHASE}" in
  validation|ab|ba) ;;
  *) usage; exit 2 ;;
esac
if [[ ! "${RUN_NAME}" =~ (^|[-._])${CAMPAIGN_PHASE}($|[-._]) ]]; then
  printf 'run-name must contain the delimited campaign phase token %s\n' "${CAMPAIGN_PHASE}" >&2
  exit 2
fi

require_bootstrap

RUN_DIR="${SWEBENCH_RESULTS_ROOT}/${RUN_NAME}/${SUITE}"
AGENT_DIR="${RUN_DIR}/agent"
EVALUATION_DIR="${RUN_DIR}/evaluation"
DATASET_DIR="$(suite_agent_dataset "${SUITE}")"
EVALUATOR_DATASET="$(suite_evaluator_dataset "${SUITE}")"
SCOPE_FILE="${RUN_DIR}/run-scope.json"
RUN_METADATA="${RUN_DIR}/run-metadata.json"
RUNTIME_BINDING="${RUN_DIR}/runtime-binding.json"
EFFECTIVE_CONFIG="${RUN_DIR}/effective-config.json"
ENDPOINT_EVIDENCE="${RUN_DIR}/endpoint-models.json"
TASK_IMAGES="${RUN_DIR}/task-images.json"
RUN_PROVENANCE="${RUN_DIR}/dataset-provenance.json"
RUN_ENVIRONMENT_FREEZE="${RUN_DIR}/environment.freeze.txt"
RUN_NORMALIZED_ENVIRONMENT_FREEZE="${RUN_DIR}/environment.normalized.freeze.txt"
CAMPAIGN_SOURCE_METADATA="${CAMPAIGN_SOURCE_METADATA:-/workspace/source-provenance.json}"
CAMPAIGN_SOURCE_ROOT="${CAMPAIGN_SOURCE_ROOT:-/workspace}"
BASE_CONFIG="${MINI_SWE_AGENT_REPO}/src/minisweagent/config/benchmarks/swebench.yaml"
EFFECTIVE_MODEL_NAME="${MODEL_NAME:-zai-org/GLM-5.2}"
mkdir -p "${RUN_DIR}"

infer_stack_variant() {
  local candidate
  for candidate in dynamo-vllm vllm-serve dynamo-sglang sglang-serve; do
    case "${RUN_NAME}" in
      "${candidate}"|"${candidate}"[-._]*) printf '%s\n' "${candidate}"; return ;;
    esac
  done
  printf 'cannot infer stack variant from run name %s; set STACK_VARIANT\n' "${RUN_NAME}" >&2
  return 2
}

STACK_VARIANT="${STACK_VARIANT:-$(infer_stack_variant)}"
case "${STACK_VARIANT}" in
  dynamo-vllm) RUNTIME_FAMILY=vllm; RUNTIME_IMAGE="${VLLM_IMAGE}"; DYNAMO_ENABLED=true ;;
  vllm-serve) RUNTIME_FAMILY=vllm; RUNTIME_IMAGE="${VLLM_IMAGE}"; DYNAMO_ENABLED=false ;;
  dynamo-sglang) RUNTIME_FAMILY=sglang; RUNTIME_IMAGE="${SGLANG_IMAGE}"; DYNAMO_ENABLED=true ;;
  sglang-serve) RUNTIME_FAMILY=sglang; RUNTIME_IMAGE="${SGLANG_IMAGE}"; DYNAMO_ENABLED=false ;;
  *) printf 'invalid STACK_VARIANT: %s\n' "${STACK_VARIANT}" >&2; exit 2 ;;
esac
DEPLOYMENT_BINDING="/artifacts/glm52-nscale/runtime-bindings/${STACK_VARIANT}/active.json"

scope_command=(
  "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/manage_scope.py" prepare
  --dataset "${EVALUATOR_DATASET}"
  --expected "${EXPECTED}"
  --output "${SCOPE_FILE}"
  --filter "${INSTANCE_FILTER:-}"
  --slice "${INSTANCE_SLICE:-}"
  --reuse-existing-if-unselected
)
"${scope_command[@]}"
scope_full_run="$("${SWEBENCH_VENV}/bin/python" - "${SCOPE_FILE}" <<'PY'
import json
import sys

print("true" if json.load(open(sys.argv[1]))["full_run"] else "false")
PY
)"
if [[ "${scope_full_run}" == true && "${CAMPAIGN_PHASE}" == validation ]]; then
  printf 'full runs require --phase ab or --phase ba\n' >&2
  exit 2
fi
if [[ "${scope_full_run}" == false && "${CAMPAIGN_PHASE}" != validation ]]; then
  printf 'filtered/sliced smoke runs require --phase validation\n' >&2
  exit 2
fi

GENERATION_WORKERS="${AGENT_WORKERS:-16}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
EVALUATION_WORKERS="${EVALUATOR_WORKERS:-8}"
EVALUATION_TIMEOUT="${EVALUATOR_TIMEOUT:-3600}"
PRO_EVALUATION_BACKEND="${PRO_EVAL_BACKEND:-local}"
if [[ "${SUITE}" == pro ]]; then
  EVALUATION_DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
else
  EVALUATION_DOCKER_PLATFORM=""
fi
for numeric_value in \
  "${GENERATION_WORKERS}" \
  "${GENERATION_BATCH_SIZE}" \
  "${EVALUATION_WORKERS}" \
  "${EVALUATION_TIMEOUT}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'generation/evaluation worker and timeout values must be positive integers\n' >&2
    exit 2
  fi
done
if [[ "${PRO_EVALUATION_BACKEND}" != local && "${PRO_EVALUATION_BACKEND}" != modal ]]; then
  printf 'PRO_EVAL_BACKEND must be local or modal\n' >&2
  exit 2
fi
if [[ "${scope_full_run}" == true ]]; then
  if [[ "${GENERATION_WORKERS}" != 16 || "${GENERATION_BATCH_SIZE}" != 8 ]]; then
    printf 'full runs require AGENT_WORKERS=16 and GENERATION_BATCH_SIZE=8\n' >&2
    exit 2
  fi
  if [[ "${EVALUATION_WORKERS}" != 8 || "${EVALUATION_TIMEOUT}" != 3600 ]]; then
    printf 'full runs require EVALUATOR_WORKERS=8 and EVALUATOR_TIMEOUT=3600\n' >&2
    exit 2
  fi
  if [[ "${SUITE}" == pro && "${PRO_EVALUATION_BACKEND}" != local ]]; then
    printf 'full SWE-bench Pro runs require PRO_EVAL_BACKEND=local\n' >&2
    exit 2
  fi
  if [[ "${SUITE}" == pro && "${EVALUATION_DOCKER_PLATFORM}" != linux/amd64 ]]; then
    printf 'full SWE-bench Pro runs require DOCKER_PLATFORM=linux/amd64\n' >&2
    exit 2
  fi
fi
readonly GENERATION_WORKERS GENERATION_BATCH_SIZE
readonly EVALUATION_WORKERS EVALUATION_TIMEOUT
readonly PRO_EVALUATION_BACKEND EVALUATION_DOCKER_PLATFORM

if [[ -z "${INSTANCE_FILTER:-}" && -z "${INSTANCE_SLICE:-}" ]]; then
  INSTANCE_FILTER="$(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/manage_scope.py" selector \
      --scope "${SCOPE_FILE}" \
      --dataset "${EVALUATOR_DATASET}" \
      --expected "${EXPECTED}" \
      --field instance_filter
  )"
  INSTANCE_SLICE="$(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/manage_scope.py" selector \
      --scope "${SCOPE_FILE}" \
      --dataset "${EVALUATOR_DATASET}" \
      --expected "${EXPECTED}" \
      --field instance_slice
  )"
  export INSTANCE_FILTER INSTANCE_SLICE
fi

metadata_endpoint="${OPENAI_BASE_URL:-}"
metadata_model="${EFFECTIVE_MODEL_NAME}"
if [[ "${PHASE}" == evaluate && -z "${MODEL_NAME+x}" ]]; then
  metadata_model=""
fi
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL to the OpenAI-compatible /v1 endpoint}"
"${SWEBENCH_VENV}/bin/python" "${SWEBENCH_EVAL_DIR}/runtime_binding.py" \
  "${DEPLOYMENT_BINDING}" \
  --variant "${STACK_VARIANT}" \
  --phase "${CAMPAIGN_PHASE}" \
  --endpoint "${OPENAI_BASE_URL%/}" >/dev/null
if [[ "${PHASE}" != evaluate ]]; then
  : "${OPENAI_BASE_URL:?set OPENAI_BASE_URL to the OpenAI-compatible /v1 endpoint}"
  probe_api_base="${OPENAI_BASE_URL%/}"
  probe_curl_args=(-fsS --connect-timeout 10 --max-time 120)
  if [[ "${OPENAI_API_KEY:-EMPTY}" != EMPTY ]]; then
    probe_curl_args+=(-H "Authorization: Bearer ${OPENAI_API_KEY}")
  fi
  probe_response="$(mktemp "${RUN_DIR}/.endpoint-models.XXXXXX")"
  trap 'rm -f "${probe_response:-}"' EXIT
  curl "${probe_curl_args[@]}" "${probe_api_base}/models" > "${probe_response}"
  "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/endpoint_preflight.py" \
    --input "${probe_response}" \
    --output "${ENDPOINT_EVIDENCE}" \
    --model "${EFFECTIVE_MODEL_NAME}" \
    --context-window 409600
  rm -f "${probe_response}"
  trap - EXIT

  effective_config_command=(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/resolve_effective_config.py"
    --config swebench.yaml
    --config "${SCRIPT_DIR}/config/glm52.yaml"
    --config "model.model_kwargs.api_base=${probe_api_base}"
    --model "openai/${EFFECTIVE_MODEL_NAME}"
    --output "${EFFECTIVE_CONFIG}"
  )
  if [[ "${SUITE}" == pro ]]; then
    effective_config_command+=(--config "${SCRIPT_DIR}/config/pro.yaml")
  fi
  "${effective_config_command[@]}"
else
  for evidence in "${ENDPOINT_EVIDENCE}" "${EFFECTIVE_CONFIG}"; do
    if [[ ! -s "${evidence}" ]]; then
      printf 'missing immutable generation evidence: %s\n' "${evidence}" >&2
      exit 1
    fi
  done
fi

capture_python_environment() {
  local freeze_tmp normalized_tmp
  freeze_tmp="$(mktemp "${RUN_DIR}/.environment.freeze.XXXXXX")"
  normalized_tmp="$(mktemp "${RUN_DIR}/.environment.normalized.XXXXXX")"
  if ! uv pip freeze --python "${SWEBENCH_VENV}/bin/python" > "${freeze_tmp}"; then
    rm -f "${freeze_tmp}" "${normalized_tmp}"
    return 1
  fi
  if ! "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/verify_environment_lock.py" \
      --lock "${SCRIPT_DIR}/constraints.lock" \
      --freeze "${freeze_tmp}" \
      --output "${normalized_tmp}"; then
    rm -f "${freeze_tmp}" "${normalized_tmp}"
    return 1
  fi
  if [[ -e "${RUN_ENVIRONMENT_FREEZE}" || -e "${RUN_NORMALIZED_ENVIRONMENT_FREEZE}" ]]; then
    if [[ ! -f "${RUN_ENVIRONMENT_FREEZE}" || ! -f "${RUN_NORMALIZED_ENVIRONMENT_FREEZE}" ]]; then
      echo "Run Python environment evidence is incomplete" >&2
      rm -f "${freeze_tmp}" "${normalized_tmp}"
      return 1
    fi
    cmp -s "${freeze_tmp}" "${RUN_ENVIRONMENT_FREEZE}" || {
      echo "Live Python environment differs from immutable run evidence" >&2
      rm -f "${freeze_tmp}" "${normalized_tmp}"
      return 1
    }
    cmp -s "${normalized_tmp}" "${RUN_NORMALIZED_ENVIRONMENT_FREEZE}" || {
      echo "Normalized Python environment differs from immutable run evidence" >&2
      rm -f "${freeze_tmp}" "${normalized_tmp}"
      return 1
    }
  else
    cp "${freeze_tmp}" "${RUN_ENVIRONMENT_FREEZE}"
    cp "${normalized_tmp}" "${RUN_NORMALIZED_ENVIRONMENT_FREEZE}"
    chmod 0444 "${RUN_ENVIRONMENT_FREEZE}" "${RUN_NORMALIZED_ENVIRONMENT_FREEZE}"
  fi
  rm -f "${freeze_tmp}" "${normalized_tmp}"
}
capture_python_environment

metadata_command=(
  "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/manage_run.py" prepare
  --output "${RUN_METADATA}"
  --run-name "${RUN_NAME}"
  --suite "${SUITE}"
  --endpoint "${metadata_endpoint}"
  --model "${metadata_model}"
  --variant "${STACK_VARIANT}"
  --campaign-phase "${CAMPAIGN_PHASE}"
  --deployment-binding "${DEPLOYMENT_BINDING}"
  --campaign-source-metadata "${CAMPAIGN_SOURCE_METADATA}"
  --campaign-source-root "${CAMPAIGN_SOURCE_ROOT}"
  --model-id "${MODEL_ID}"
  --model-revision "${MODEL_REVISION}"
  --context-window "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TP_SIZE}"
  --runtime-family "${RUNTIME_FAMILY}"
  --runtime-image "${RUNTIME_IMAGE}"
  --runtime-source-revision "${DYNAMO_IMAGE_COMMIT}"
  --dynamo-enabled "${DYNAMO_ENABLED}"
  --generation-workers "${GENERATION_WORKERS}"
  --generation-batch-size "${GENERATION_BATCH_SIZE}"
  --evaluator-workers "${EVALUATION_WORKERS}"
  --evaluator-timeout "${EVALUATION_TIMEOUT}"
  --pro-eval-backend "${PRO_EVALUATION_BACKEND}"
  --docker-platform "${EVALUATION_DOCKER_PLATFORM}"
  --effective-config "${EFFECTIVE_CONFIG}"
  --endpoint-evidence "${ENDPOINT_EVIDENCE}"
  --runtime-binding-output "${RUNTIME_BINDING}"
  --config "upstream-swebench=${BASE_CONFIG}"
  --config "glm52=${SCRIPT_DIR}/config/glm52.yaml"
  --dataset "${EVALUATOR_DATASET}"
  --dataset-provenance "${SWEBENCH_DATA_ROOT}/provenance.json"
  --pins "${SWEBENCH_EVAL_DIR}/pins.env"
  --source-lock "${SWEBENCH_WORK_ROOT}/source-lock.json"
  --constraints-lock "${SCRIPT_DIR}/constraints.lock"
  --environment-freeze "${RUN_ENVIRONMENT_FREEZE}"
  --normalized-environment-freeze "${RUN_NORMALIZED_ENVIRONMENT_FREEZE}"
  --source-repo "mini_swe_agent=${MINI_SWE_AGENT_REPO}"
  --source-repo "swebench=${SWEBENCH_EVALUATOR_REPO}"
  --source-repo "swebench_pro=${SWEBENCH_PRO_REPO}"
  --scope "${SCOPE_FILE}"
  --predictions "${AGENT_DIR}/preds.json"
)
if [[ "${SUITE}" == pro ]]; then
  metadata_command+=(--config "pro=${SCRIPT_DIR}/config/pro.yaml")
fi
if [[ "${PHASE}" == evaluate ]]; then
  metadata_command+=(--require-existing)
fi
"${metadata_command[@]}"

if [[ -e "${RUN_PROVENANCE}" ]]; then
  if ! cmp -s "${SWEBENCH_DATA_ROOT}/provenance.json" "${RUN_PROVENANCE}"; then
    printf 'dataset provenance differs from immutable run artifact: %s\n' "${RUN_PROVENANCE}" >&2
    exit 1
  fi
else
  cp "${SWEBENCH_DATA_ROOT}/provenance.json" "${RUN_PROVENANCE}"
fi

record_command() {
  local output="$1"
  shift
  printf '%q ' "$@" > "${output}"
  printf '\n' >> "${output}"
}

append_command() {
  local output="$1"
  shift
  printf '%q ' "$@" >> "${output}"
  printf '\n' >> "${output}"
}

prune_swe_task_images() {
  local log="${RUN_DIR}/docker-cleanup.log"
  local image
  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    docker system df
  } >> "${log}" 2>&1
  while IFS= read -r image; do
    case "${image}" in
      */sweb.eval.*:*|sweb.eval.*:*|*/sweap-images:*)
        # No force: Docker refuses to remove any image referenced by a running
        # container. This runner may host another active benchmark concurrently.
        docker image rm "${image}" >> "${log}" 2>&1 || true
        ;;
    esac
  done < <(docker image ls --format '{{.Repository}}:{{.Tag}}')
  docker system df >> "${log}" 2>&1
}

run_generation() {
  : "${OPENAI_BASE_URL:?set OPENAI_BASE_URL to the OpenAI-compatible /v1 endpoint}"
  local api_base="${OPENAI_BASE_URL%/}"
  local model_name="${EFFECTIVE_MODEL_NAME}"
  local workers="${GENERATION_WORKERS}"
  local batch_size="${GENERATION_BATCH_SIZE}"
  local -a base_command
  local -a command
  local -a batch_filters=()
  local batch_filter
  local batch_status
  local capture_status
  local status=0

  if [[ ! "${workers}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'AGENT_WORKERS must be a positive integer\n' >&2
    return 2
  fi
  if [[ ! "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'GENERATION_BATCH_SIZE must be a positive integer\n' >&2
    return 2
  fi

  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export MSWEA_COST_TRACKING=ignore_errors
  export MSWEA_GLOBAL_CONFIG_DIR="${SWEBENCH_MSWEA_CONFIG_DIR}"
  export MSWEA_SILENT_STARTUP=1
  mkdir -p "${SWEBENCH_MSWEA_CONFIG_DIR}"

  mkdir -p "${AGENT_DIR}"

  base_command=(
    "${SWEBENCH_VENV}/bin/mini-extra" swebench
    --output "${AGENT_DIR}"
    --subset "${DATASET_DIR}"
    --split test
    --workers "${workers}"
    --model "openai/${model_name}"
    --config swebench.yaml
    --config "${SCRIPT_DIR}/config/glm52.yaml"
    --config "model.model_kwargs.api_base=${api_base}"
  )
  if [[ "${SUITE}" == pro ]]; then
    base_command+=(--config "${SCRIPT_DIR}/config/pro.yaml")
  fi
  if [[ "${REDO_EXISTING:-0}" == 1 ]]; then
    base_command+=(--redo-existing)
  fi

  while IFS= read -r batch_filter; do
    batch_filters+=("${batch_filter}")
  done < <(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/manage_scope.py" batch-filters \
      --scope "${SCOPE_FILE}" \
      --dataset "${EVALUATOR_DATASET}" \
      --expected "${EXPECTED}" \
      --batch-size "${batch_size}"
  )
  if [[ ${#batch_filters[@]} -eq 0 ]]; then
    printf 'generation scope produced no batches\n' >&2
    return 1
  fi

  : > "${RUN_DIR}/generation-command.txt"
  : > "${RUN_DIR}/generation.log"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/generation-started-at.txt"
  prune_swe_task_images
  for batch_filter in "${batch_filters[@]}"; do
    command=("${base_command[@]}" --filter "${batch_filter}")
    append_command "${RUN_DIR}/generation-command.txt" "${command[@]}"
    set +e
    "${command[@]}" 2>&1 | tee -a "${RUN_DIR}/generation.log"
    batch_status=${PIPESTATUS[0]}
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/capture_task_images.py" \
      --suite "${SUITE}" \
      --dataset "${EVALUATOR_DATASET}" \
      --expected "${EXPECTED}" \
      --scope "${SCOPE_FILE}" \
      --batch-filter "${batch_filter}" \
      --output "${TASK_IMAGES}"
    capture_status=$?
    set -e
    prune_swe_task_images
    if [[ ${batch_status} -ne 0 ]]; then
      status=${batch_status}
      break
    fi
    if [[ ${capture_status} -ne 0 ]]; then
      status=${capture_status}
      break
    fi
  done
  date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/generation-finished-at.txt"

  "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/summarize_generation.py" \
    --agent-dir "${AGENT_DIR}" \
    --dataset "${EVALUATOR_DATASET}" \
    --expected "${EXPECTED}" \
    --scope "${SCOPE_FILE}" \
    --run-metadata "${RUN_METADATA}" \
    --task-images "${TASK_IMAGES}" \
    --output "${RUN_DIR}/generation-summary.json" \
    --require-complete
  if [[ ${status} -ne 0 ]]; then
    return "${status}"
  fi
}

run_standard_evaluation() {
  local workers="${EVALUATION_WORKERS}"
  local timeout="${EVALUATION_TIMEOUT}"
  local run_id="${RUN_NAME}-${SUITE}-${PREDICTION_DIGEST:0:16}"
  local raw_report
  local status
  if [[ "${REDO_EVALUATION:-0}" == 1 ]]; then
    run_id="${run_id}-redo-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  fi
  local -a command=(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/evaluate_standard.py"
    --task-images "${TASK_IMAGES}"
    --dataset_name "${EVALUATOR_DATASET}"
    --split test
    --predictions_path "${AGENT_DIR}/preds.json"
    --max_workers "${workers}"
    --timeout "${timeout}"
    --run_id "${run_id}"
    --namespace swebench
    --cache_level env
    --clean true
  )

  command+=(--instance_ids)
  while IFS= read -r instance_id; do
    command+=("${instance_id}")
  done < <(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/manage_scope.py" list \
      --scope "${SCOPE_FILE}" \
      --dataset "${EVALUATOR_DATASET}" \
      --expected "${EXPECTED}"
  )

  record_command "${RUN_DIR}/evaluation-command.txt" "${command[@]}"
  set +e
  (
    cd "${EVALUATION_DIR}"
    "${command[@]}"
  ) 2>&1 | tee "${RUN_DIR}/evaluation.log"
  status=${PIPESTATUS[0]}
  set -e
  prune_swe_task_images

  raw_report="$(find "${EVALUATION_DIR}" -maxdepth 1 -type f -name "*.${run_id}.json" -print | head -n 1)"
  if [[ -z "${raw_report}" ]]; then
    printf 'SWE-bench evaluator did not produce a run report\n' >&2
    return 1
  fi
  cp "${raw_report}" "${EVALUATION_DIR}/raw-score.json"
  "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/summarize_score.py" \
    --kind swebench \
    --suite "${SUITE}" \
    --expected "${EXPECTED}" \
    --dataset "${EVALUATOR_DATASET}" \
    --scope "${SCOPE_FILE}" \
    --raw "${EVALUATION_DIR}/raw-score.json" \
    --output "${RUN_DIR}/score.json" \
    --require-complete
  if [[ ${status} -ne 0 ]]; then
    return "${status}"
  fi
}

run_pro_evaluation() {
  local workers="${EVALUATION_WORKERS}"
  local timeout="${EVALUATION_TIMEOUT}"
  local status
  local predictions="${EVALUATION_DIR}/pro-predictions.json"
  local output_dir="${EVALUATION_DIR}/outputs"
  local status_dir="${EVALUATION_DIR}/statuses"
  local prediction_prefix="${RUN_NAME}-${PREDICTION_DIGEST:0:16}"
  local -a command

  "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/convert_pro_predictions.py" \
    "${AGENT_DIR}/preds.json" \
    --output "${predictions}" \
    --prefix "${prediction_prefix}"

  command=(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/evaluate_pro.py"
    --pro-repo "${SWEBENCH_PRO_REPO}"
    --raw-sample-path "${EVALUATOR_DATASET}"
    --status-dir "${status_dir}"
    --evaluator-timeout "${timeout}"
    --task-images "${TASK_IMAGES}"
    --patch_path "${predictions}"
    --output_dir "${output_dir}"
    --scripts_dir "${SWEBENCH_PRO_REPO}/run_scripts"
    --num_workers "${workers}"
    --dockerhub_username jefzda
  )
  if [[ "${PRO_EVALUATION_BACKEND}" == local ]]; then
    command+=(--use_local_docker)
  fi
  if [[ -n "${EVALUATION_DOCKER_PLATFORM}" ]]; then
    command+=(--docker_platform "${EVALUATION_DOCKER_PLATFORM}")
  fi
  if [[ "${REDO_EVALUATION:-0}" == 1 ]]; then
    command+=(--redo)
  fi

  record_command "${RUN_DIR}/evaluation-command.txt" "${command[@]}"
  rm -f "${output_dir}/eval_results.json"
  set +e
  "${command[@]}" 2>&1 | tee "${RUN_DIR}/evaluation.log"
  status=${PIPESTATUS[0]}
  set -e
  prune_swe_task_images

  if [[ -f "${output_dir}/eval_results.json" ]]; then
    cp "${output_dir}/eval_results.json" "${EVALUATION_DIR}/raw-score.json"
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/summarize_score.py" \
      --kind pro \
      --suite pro \
      --expected "${EXPECTED}" \
      --dataset "${EVALUATOR_DATASET}" \
      --scope "${SCOPE_FILE}" \
      --raw "${EVALUATION_DIR}/raw-score.json" \
      --status-dir "${status_dir}" \
      --output "${RUN_DIR}/score.json" \
      --require-complete
  else
    printf 'SWE-bench Pro evaluator did not produce eval_results.json\n' >&2
    return 1
  fi
  if [[ ${status} -ne 0 ]]; then
    return "${status}"
  fi
}

run_evaluation() {
  if [[ ! -s "${AGENT_DIR}/preds.json" ]]; then
    printf 'missing predictions: %s\n' "${AGENT_DIR}/preds.json" >&2
    return 1
  fi
  "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/summarize_generation.py" \
    --agent-dir "${AGENT_DIR}" \
    --dataset "${EVALUATOR_DATASET}" \
    --expected "${EXPECTED}" \
    --scope "${SCOPE_FILE}" \
    --run-metadata "${RUN_METADATA}" \
    --task-images "${TASK_IMAGES}" \
    --output "${RUN_DIR}/generation-summary.json" \
    --require-complete
  PREDICTION_DIGEST="$(
    "${SWEBENCH_VENV}/bin/python" "${SCRIPT_DIR}/prediction_digest.py" \
      "${AGENT_DIR}/preds.json"
  )"
  readonly PREDICTION_DIGEST
  printf '%s\n' "${PREDICTION_DIGEST}" > "${RUN_DIR}/evaluation-predictions-sha256.txt"
  mkdir -p "${EVALUATION_DIR}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/evaluation-started-at.txt"
  if [[ "${SUITE}" == pro ]]; then
    run_pro_evaluation
  else
    run_standard_evaluation
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/evaluation-finished-at.txt"
}

case "${PHASE}" in
  generate) run_generation ;;
  evaluate) run_evaluation ;;
  all)
    run_generation
    run_evaluation
    ;;
esac
