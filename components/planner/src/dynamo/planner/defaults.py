# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Source of truth for planner defaults
class BasePlannerDefaults:
    namespace = "dynamo"
    environment = "local"
    no_operation = False
    log_dir = None
    adjustment_interval = 180  # in seconds
    max_gpu_budget = 8
    min_endpoint = 1  # applies to both decode and prefill
    decode_engine_num_gpu = 1
    prefill_engine_num_gpu = 1


class LoadPlannerDefaults(BasePlannerDefaults):
    metric_pulling_interval = 10  # in seconds
    decode_kv_scale_up_threshold = 0.9
    decode_kv_scale_down_threshold = 0.5
    prefill_queue_scale_up_threshold = 5.0
    prefill_queue_scale_down_threshold = 0.2


class SLAPlannerDefaults(BasePlannerDefaults):
    prometheus_endpoint = "http://localhost:9090"
    profile_results_dir = "profiling_results"
    isl = 3000  # in number of tokens
    osl = 150  # in number of tokens
    ttft = 0.5  # in seconds
    itl = 0.05  # in seconds
    load_predictor = "arima"  # ["constant", "arima", "prophet"]
    load_prediction_window_size = 50  # predict load using how many recent load samples


class VllmV0ComponentName:
    prefill_worker = "PrefillWorker"
    decode_worker = "VllmWorker"


class VllmV1ComponentName:
    prefill_worker = "VllmPrefillWorker"
    decode_worker = "VllmDecodeWorker"


WORKER_COMPONENT_NAMES = {
    "vllm_v0": VllmV0ComponentName,
    "vllm_v1": VllmV1ComponentName,
}
