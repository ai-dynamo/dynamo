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

"""Run the upstream native VoiceChat loop and its microphone playground."""

import asyncio
import os

import uvicorn
from demo_tools import registry
from huggingface_hub import snapshot_download
from native_tools import deployment
from sglang_omni.client.client import Client
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    NemotronVoiceChatToolPipelineConfig,
)
from sglang_omni.models.nemotron_voicechat.payload_types import VOICECHAT_FORMAT
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto.session import SessionLimits
from sglang_omni.serve.openai_api import create_app

from examples.run_nemotron_voicechat_duplex import (
    close_workers_on_shutdown,
    mount_example_ui,
    warmup_realtime,
)

MODEL = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
MODEL_REVISION = "443794ea956ef0065f001967ffd00e77f519cb39"
BACKBONE_REVISION = "6533e8de2c68e4536bf7c411d7a3ce5734111476"


async def main():
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION)
    snapshot_download(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        revision=BACKBONE_REVISION,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    config = NemotronVoiceChatToolPipelineConfig(
        model_path=model_path, tools=registry()
    )
    config.session_publish = {
        "voicechat": [VOICECHAT_FORMAT, "tool_call"],
        "tool": ["tool_response"],
    }
    runner = MultiProcessPipelineRunner(config)
    try:
        await runner.start(timeout=900)
        client = Client(runner.coordinator)
        realtime = deployment(client)
        await warmup_realtime(
            deployment(client, session_limits=SessionLimits(operation_timeout_s=120))
        )
        app = create_app(client, model_name=MODEL, realtime_deployment=realtime)
        mount_example_ui(app)
        close_workers_on_shutdown(app, runner.stop)
        await uvicorn.Server(
            uvicorn.Config(
                app, host="0.0.0.0", port=int(os.environ.get("OMNI_PORT", "8097"))
            )
        ).serve()
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
