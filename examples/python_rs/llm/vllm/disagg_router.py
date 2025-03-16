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


from vllm.logger import logger as vllm_logger


class PyDisaggregatedRouter:
    def __init__(
        self,
        runtime,
        served_model_name,
        max_local_prefill_length=1000,
    ):
        self.runtime = runtime
        self.served_model_name = served_model_name
        self.max_local_prefill_length = max_local_prefill_length

    def prefill_remote(self, prompt_length: int, prefix_hit_rate: float):
        absolute_prefill_length = int(prompt_length * (1 - prefix_hit_rate))
        vllm_logger.info(
            f"Remote prefill: {absolute_prefill_length > self.max_local_prefill_length} (prefill length: {absolute_prefill_length}/{prompt_length})"
        )
        return absolute_prefill_length > self.max_local_prefill_length
