#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared helpers for the direct trtllm-serve OpenEngine launch examples.

warn_trtllm_serve_profile_overrides() {
    local configs="$1"
    if [[ -z ${_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS:-} &&
          -z ${_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES:-} ]]; then
        return
    fi

    # Narrow exception: build_trtllm_override_args_with_mem emits JSON for the
    # in-process Dynamo worker's --override-engine-args option. trtllm-serve has
    # no equivalent JSON merge option, and mapping max_tokens/max_gpu_total_bytes
    # to its superficially similar CLI flags would change their semantics. Keep
    # the direct-server examples honest: profile KV limits belong in the YAML.
    echo "WARNING: TRT-LLM profile overrides are not forwarded to trtllm-serve." >&2
    echo "Set kv_cache_config.max_tokens or max_gpu_total_bytes in: $configs" >&2
}

print_openengine_aggregate_banner() {
    local mode="$1" model="$2" http_port="$3"
    local trtllm_http_port="$4" openengine_port="$5" system_port="$6"
    local details=(
        "TRT-LLM HTTP: http://127.0.0.1:$trtllm_http_port"
        "OpenEngine:  http://127.0.0.1:$openengine_port"
        "System API:  http://127.0.0.1:$system_port"
    )
    case "$mode" in
        audio)
            print_launch_banner --no-curl \
                "Launching TRT-LLM OpenEngine Audio Serving (1 GPU)" \
                "$model" "$http_port" "${details[@]}" \
                "Media:       audio URL/data-URI passthrough"
            ;;
        lora)
            print_launch_banner \
                "Launching TRT-LLM OpenEngine Aggregated Serving + LoRA (1 GPU)" \
                "$model" "$http_port" "${details[@]}"
            ;;
        multimodal)
            print_launch_banner --multimodal \
                "Launching TRT-LLM OpenEngine Multimodal Serving (1 GPU)" \
                "$model" "$http_port" "${details[@]}"
            ;;
        text)
            print_launch_banner \
                "Launching TRT-LLM OpenEngine Aggregated Serving (1 GPU)" \
                "$model" "$http_port" "${details[@]}"
            ;;
        *)
            echo "Unknown OpenEngine aggregate launch mode: $mode" >&2
            return 2
            ;;
    esac
}

print_openengine_disagg_banner() {
    local mode="$1" model="$2" http_port="$3"
    local prefill_port="$4" decode_port="$5"
    local details=(
        "Prefill OpenEngine: http://127.0.0.1:$prefill_port"
        "Decode OpenEngine:  http://127.0.0.1:$decode_port"
    )
    case "$mode" in
        multimodal)
            print_launch_banner --multimodal \
                "Launching TRT-LLM OpenEngine Multimodal Context-First 1P1D (2 GPUs)" \
                "$model" "$http_port" "${details[@]}"
            ;;
        text)
            print_launch_banner \
                "Launching TRT-LLM OpenEngine Context-First 1P1D + LoRA (2 GPUs)" \
                "$model" "$http_port" "${details[@]}"
            ;;
        *)
            echo "Unknown OpenEngine disaggregated launch mode: $mode" >&2
            return 2
            ;;
    esac
}
