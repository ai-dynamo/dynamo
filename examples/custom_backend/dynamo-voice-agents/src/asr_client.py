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

"""
Test client for the streaming ASR pipeline.

Usage:
    python asr_client.py [audio_file_path]
"""

import asyncio
import os
import sys
from pathlib import Path

import uvloop
from protocol import ASRResponse

from dynamo.runtime import DistributedRuntime, dynamo_worker

# Configure Dynamo runtime
os.environ.setdefault("DYN_STORE_KV", "file")
os.environ.setdefault("DYN_REQUEST_PLANE", "tcp")


@dynamo_worker(enable_nats=False)
async def client(runtime: DistributedRuntime):
    # Get audio path from command line or use default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data"
        wav_files = list(data_dir.glob("*.wav"))
        if wav_files:
            audio_path = str(wav_files[0])
        else:
            print("Usage: python asr_client.py <audio_file_path>")
            return

    print(f"Audio file: {audio_path}")

    # Connect to chunker
    endpoint = (
        runtime.namespace("streaming_asr").component("chunker").endpoint("transcribe")
    )
    chunker_client = await endpoint.client()

    print("Waiting for workers...")
    await chunker_client.wait_for_instances()
    print("Connected.\n")

    # Stream transcription
    async for response in await chunker_client.round_robin(audio_path):
        result = ASRResponse.model_validate_json(response.data())
        prefix = "[FINAL]  " if result.is_final else "[PARTIAL]"
        print(f"{prefix} {result.text}")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(client())
