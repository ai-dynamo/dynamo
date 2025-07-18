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

import base64

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

        opaque_state = (
            base64.b64decode(disaggregated_params.opaque_state)
            if disaggregated_params.opaque_state is not None
            else None
        )
        return DisaggregatedParams(
            request_type=disaggregated_params.request_type,
            first_gen_tokens=disaggregated_params.first_gen_tokens,
            ctx_request_id=disaggregated_params.ctx_request_id,
            opaque_state=opaque_state,
            draft_tokens=disaggregated_params.draft_tokens,
        )

    @staticmethod
    def encode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None

        encoded_opaque_state = (
            base64.b64encode(disaggregated_params.opaque_state).decode("utf-8")
            if disaggregated_params.opaque_state is not None
            else None
        )
        return DisaggregatedParams(
            request_type=disaggregated_params.request_type,
            first_gen_tokens=disaggregated_params.first_gen_tokens,
            ctx_request_id=disaggregated_params.ctx_request_id,
            opaque_state=encoded_opaque_state,
            draft_tokens=disaggregated_params.draft_tokens,
        )
