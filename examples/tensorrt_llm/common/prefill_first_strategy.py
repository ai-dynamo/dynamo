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
import copy

from common.base_engine import BaseEngine, BaseEngineConfig
from common.protocol import TRTLLMWorkerRequest


class PrefillFirstStrategy(BaseEngine):
    def __init__(
        self,
        config: BaseEngineConfig,
        is_next_worker: bool = False,
    ):
        self._config = config
        # Determine if this worker is the next worker based on the absence of a next endpoint.
        self._is_next_worker = not bool(self._config.next_endpoint)
        if self._is_next_worker:
            if self._config.disaggregation_mode != "decode":
                raise ValueError(
                    "Prefill First Strategy needs the next worker to be in decode mode"
                )
        else:
            if self._config.disaggregation_mode != "prefill":
                raise ValueError(
                    "Prefill First Strategy needs the first worker to operate in prefill mode"
                )
        super().__init__(config)

    async def generate(self, request: TRTLLMWorkerRequest):
        if self._is_next_worker:
            async for res in self.generate_local(request):
                yield res
        else:
            # For prefill first mode,
            # run prefill stage locally and decode stage remotely

            # Local Prefill Stage
            prefill_result = None
            prefill_request = copy.deepcopy(request)
            async for generate_result in self.generate_helper(
                prefill_request, prefill_only=True
            ):
                if self.check_for_error(generate_result):
                    yield generate_result
                    return
                prefill_result = generate_result

            # Remote Decode Stage
            async for res in self.remote_generate(
                request,
                disaggregated_params=prefill_result["disaggregated_params"],
                decode_only=True,
            ):
                yield res.data()
            return
