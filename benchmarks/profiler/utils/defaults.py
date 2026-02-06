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

from enum import Enum

DYNAMO_RUN_DEFAULT_PORT = 8000

# set a decode maximum concurrency due to limits of profiling tools
# for MoE models with attn-dp, we might hit this limit
DECODE_MAX_CONCURRENCY = 2000

# set a prefill maximum number of tokens to 32768 to avoid chunked prefill but not too large to cause activation tensor too large
PREFILL_MAX_NUM_TOKENS = 32768

# AIPerf benchmarking related defaults
AIPERF_WARMUP_REQUEST_PER_DP_RANK = 3
AIPERF_PREFILL_BENCHMARK_OSL = 5
AIPERF_PREFILL_ATTN_DP_NUM_REQ_RATIO = 4

# Cost calculation defaults
# TODO: allow user to configure this in GUI
DEFAULT_GPU_COST_PER_HOUR = 3.0  # Cost per GPU per hour in dollars


class EngineType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
