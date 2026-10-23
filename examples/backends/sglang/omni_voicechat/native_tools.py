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

"""Publish the real native graph's tool calls/results alongside its audio."""

import json
from dataclasses import replace

from sglang_omni.models.nemotron_voicechat.realtime import VoiceChatOutput
from sglang_omni.models.nemotron_voicechat.realtime import (
    deployment as audio_deployment,
)
from sglang_omni.models.nemotron_voicechat.realtime import make_adapter
from sglang_omni.serve.realtime.output import ToolCallEvent, ToolResultEvent


class ToolOutput(VoiceChatOutput):
    def __call__(self, output):
        prefix = output.session_identity.id + "-"
        if output.format == "tool_call":
            if self.response_id is None:
                raise ValueError("Native tool call preceded its audio response")
            for call in output.payload["calls"]:
                yield ToolCallEvent(
                    self.response_id,
                    prefix + call["id"],
                    call["name"],
                    json.dumps(call["arguments"], ensure_ascii=True),
                )
        elif output.format == "tool_response":
            for response in output.payload["responses"]:
                yield ToolResultEvent(
                    prefix + response["id"],
                    json.dumps(response["response"], ensure_ascii=True),
                )
        else:
            yield from super().__call__(output)


def deployment(client, *, session_limits=None):
    base = audio_deployment(client, session_limits=session_limits)

    def adapter():
        result = make_adapter(client, base.session_limits)
        result.output_converter = ToolOutput()
        return result

    return replace(base, adapter_factory=adapter)
