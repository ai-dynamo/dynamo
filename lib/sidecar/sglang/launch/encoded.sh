#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone native SGLang encoders selected by the Dynamo frontend.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"

model="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
encoders=1
tp=1
encoder_tp=
single_gpu=false
packed=false
served_model_name=
engine_args=()
while (($#)); do
    case "$1" in
        --model|--model-path) model="${2:?missing model}"; shift 2 ;;
        --tp-size) tp="${2:?missing tensor parallel size}"; shift 2 ;;
        --encoder-tp-size) encoder_tp="${2:?missing encoder tensor parallel size}"; shift 2 ;;
        --encoders) encoders="${2:?missing number of encoders}"; shift 2 ;;
        --split-pd)
            echo "--split-pd is not supported: an external encoder needs an aggregated language worker." >&2
            exit 2 ;;
        --colocate) packed=true; shift ;;
        --single-gpu) packed=true; single_gpu=true; shift ;;
        --)
            shift
            while (($#)); do
                case "$1" in
                    --tp|--tp-size|--tensor-parallel-size) tp="${2:?missing tensor parallel size}"; shift 2 ;;
                    --tp=*|--tp-size=*|--tensor-parallel-size=*) tp="${1#*=}"; shift ;;
                    # The encoder sidecar registers the same name as the language worker.
                    --served-model-name)
                        served_model_name="${2:?missing served model name}"
                        engine_args+=("$1" "$2"); shift 2 ;;
                    --served-model-name=*)
                        served_model_name="${1#*=}"
                        engine_args+=("$1"); shift ;;
                    *) engine_args+=("$1"); shift ;;
                esac
            done
            ;;
        -h|--help)
            cat <<'HELP'
Usage: encoded.sh [--model MODEL] [--encoders N] [--tp-size N] [--encoder-tp-size N] [--colocate | --single-gpu] [-- SGLANG_OPTIONS...]

Starts Dynamo, N independently registered native encoder/sidecar pairs, and
one aggregated native language engine/sidecar. Split prefill/decode with an
external encoder is not supported. --tp-size sets the TP degree for all roles.
--encoder-tp-size overrides it for E. By default each role takes its own
contiguous group from CUDA_VISIBLE_DEVICES, encoders first, then PD.
--colocate shares the first visible group across all roles, bounds token pools,
and disables CUDA graphs. --single-gpu selects colocation with TP=1.

Each encoder is SGLang's HTTP encoder (--encoder-only). The language worker is
SGLang's Rust server with SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF=1. Embeddings
travel from E to PD over ZMQ (--encoder-transfer-backend zmq_to_scheduler).

Use an SGLang installation that accepts encoder_handoff (SGLANG_PYTHON) and a
matching native sidecar (DYNAMO_SIDECAR_BIN). DYN_DISCOVERY_BACKEND=file needs
no external services. Indexed DYN_ENGINE_HTTP_PORT, DYN_ENGINE_GRPC_PORT
(language worker), DYN_NCCL_PORT, and DYN_SYSTEM_PORT override local ports,
starting at index 1. MAX_MODEL_LEN and MAX_CONCURRENT_SEQS override the 4096
context and two sequences.
HELP
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$encoders" =~ ^[1-9][0-9]*$ ]] || { echo "--encoders needs a positive integer" >&2; exit 2; }
encoder_tp=${encoder_tp:-$tp}
if ! [[ "$tp" =~ ^[1-9][0-9]*$ && "$encoder_tp" =~ ^[1-9][0-9]*$ ]]; then
    echo "--tp-size and --encoder-tp-size need positive integers" >&2
    exit 2
fi
if $single_gpu && (( tp > 1 || encoder_tp > 1 )); then
    echo "--single-gpu requires TP=1. Use --colocate to share a TP group." >&2
    exit 2
fi

host=127.0.0.1
sgpy="${SGLANG_PYTHON:-python3}"
frontend_py="${DYNAMO_PYTHON:-python3}"
sidecar=()
resolve_sidecar sglang sidecar
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
if [[ -v CUDA_VISIBLE_DEVICES && -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "CUDA_VISIBLE_DEVICES is empty. This launcher requires at least one visible GPU." >&2
    exit 2
fi
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-}"
roles=$(( encoders + 1 ))
needed_gpus=$((encoders * encoder_tp + tp))
if $packed; then needed_gpus=$((encoder_tp > tp ? encoder_tp : tp)); fi
if (( ${#devices[@]} == 0 )); then
    for ((i=0; i<needed_gpus; i++)); do devices+=("$i"); done
elif (( ${#devices[@]} < needed_gpus )); then
    echo "Need $needed_gpus visible GPUs for this placement, got ${#devices[@]}." >&2
    exit 2
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_ARGS="$(build_sglang_gpu_mem_args)"
read -r -a mem_args <<< "$GPU_MEM_ARGS"
lang_args=(--language-only --incremental-streaming-output
    --context-length "$MAX_MODEL_LEN" --max-running-requests "${MAX_CONCURRENT_SEQS:-2}")
if $packed; then
    lang_args+=(--disable-decode-cuda-graph --disable-prefill-cuda-graph --disable-flashinfer-autotune)
    if ((${#mem_args[@]} == 0)); then mem_args=(--max-total-tokens "$(( 2 * MAX_MODEL_LEN ))"); fi
fi

trap dynamo_exit_trap EXIT
print_launch_banner --multimodal "SGLang native sidecars ($encoders encoders, aggregated PD)" \
    "$model" "${DYN_HTTP_PORT:-8000}"
"$frontend_py" -m dynamo.frontend &

offset=0
for ((i=1; i<=roles; i++)); do
    role_tp=$tp
    if ((i <= encoders)); then role_tp=$encoder_tp; fi
    group=("${devices[@]:offset:role_tp}")
    # Preserve the caller's device order and identifiers (ordinals or GPU UUIDs).
    gpu=$(IFS=,; echo "${group[*]}")
    if ! $packed; then offset=$((offset + role_tp)); fi
    http_port=$(dyn_port DYN_ENGINE_HTTP_PORT "$i" "$((30000 + 3*i))")
    nccl_port=$(dyn_port DYN_NCCL_PORT "$i" "$((29500 + i))")
    system_port=$(dyn_port DYN_SYSTEM_PORT "$i" "$((8080 + i))")
    args=(--model-path "$model" --host "$host" --tp-size "$role_tp" --dp-size 1
        --encoder-transfer-backend zmq_to_scheduler --nccl-port "$nccl_port" "${engine_args[@]}")
    if ((i <= encoders)); then
        # The unmodified HTTP encoder. Its sidecar has no gRPC service to dial:
        # --grpc-endpoint names the encoder's HTTP listener.
        sidecar_args=(--disaggregation-mode encode --grpc-endpoint "$host:$http_port"
            --encoder-model-path "$model")
        if [[ -n "$served_model_name" ]]; then
            sidecar_args+=(--encoder-served-model-name "$served_model_name")
        fi
        env -u SGLANG_RUST_SERVER -u SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF \
            CUDA_VISIBLE_DEVICES="$gpu" "$sgpy" -m sglang.launch_server \
            "${args[@]}" --encoder-only --port "$http_port" &
    else
        grpc_port=$(dyn_port DYN_ENGINE_GRPC_PORT "$i" "$((30000 + 3*i + 1))")
        sidecar_args=(--grpc-endpoint "$host:$grpc_port" --route-to-encoder)
        CUDA_VISIBLE_DEVICES="$gpu" SGLANG_RUST_SERVER=1 SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF=1 \
            "$sgpy" -m sglang.launch_server \
            "${args[@]}" "${lang_args[@]}" "${mem_args[@]}" --port "$http_port" --grpc-port "$grpc_port" &
    fi
    DYN_SYSTEM_PORT="$system_port" "${sidecar[@]}" "${sidecar_args[@]}" &
done
wait_any_exit
