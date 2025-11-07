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

from enum import Enum

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DYNAMO_RUN_DEFAULT_PORT = 8000

# set a decode maximum concurrency due to limits of profiling tools
# for MoE models with attn-dp, we might hit this limit
DECODE_MAX_CONCURRENCY = 2000


class EngineType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
