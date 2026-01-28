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

import base64
import dataclasses

from tensorrt_llm.llmapi import DisaggregatedParams


class DisaggregatedParamsCodec:
    """
    Codec for encoding and decoding disaggregated params for network transfer.
    """

    @staticmethod
    def decode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None

        opaque_state = disaggregated_params.opaque_state
        if isinstance(opaque_state, str):
            opaque_state = base64.b64decode(opaque_state)
        return dataclasses.replace(disaggregated_params, opaque_state=opaque_state)

    @staticmethod
    def encode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None

        opaque_state = disaggregated_params.opaque_state
        if isinstance(opaque_state, (bytes, bytearray)):
            opaque_state = base64.b64encode(opaque_state).decode("utf-8")
        return dataclasses.replace(disaggregated_params, opaque_state=opaque_state)
