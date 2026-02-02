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
Microphone client for real-time streaming ASR.

This client captures audio from your microphone and streams it to the
Dynamo streaming ASR service, displaying transcriptions in real-time.

Usage:
    python mic_client.py

Requirements:
    pip install sounddevice numpy

Press Ctrl+C to stop recording.
"""

import asyncio
import os
import queue
import sys
import time
import uuid
from threading import Event

import numpy as np
import uvloop
from protocol import (
    StreamingASRResponse,
    StreamingAudioChunk,
    StreamingConfig,
)

# Configure Dynamo runtime
os.environ.setdefault("DYN_STORE_KV", "file")
os.environ.setdefault("DYN_REQUEST_PLANE", "tcp")

from dynamo.runtime import DistributedRuntime, dynamo_worker

# Audio configuration
SAMPLE_RATE = 16000  # 16kHz for ASR models
CHUNK_DURATION_MS = 500  # Send audio every 500ms
CHANNELS = 1


class MicrophoneStreamer:
    """Captures audio from microphone and streams to Dynamo ASR service."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_duration_ms: int = CHUNK_DURATION_MS):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.audio_queue: queue.Queue = queue.Queue()
        self.stop_event = Event()
        self.session_id = str(uuid.uuid4())
        self.sequence_number = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice to receive audio data."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        # Convert to float32 and flatten
        audio_data = indata.copy().flatten().astype(np.float32)
        self.audio_queue.put(audio_data)

    async def stream_to_dynamo(self, client):
        """Stream audio chunks to the Dynamo ASR service."""
        config = StreamingConfig(
            sample_rate=self.sample_rate,
            chunk_duration_ms=self.chunk_duration_ms,
        )

        print(f"\n🎤 Starting streaming session: {self.session_id[:8]}...")
        print("=" * 60)

        # Send START message
        start_msg = StreamingAudioChunk.start_session(self.session_id, config)
        async for response in await client.round_robin(start_msg.model_dump_json()):
            result = StreamingASRResponse.model_validate_json(response.data())
            print(f"Session started: {result.session_id[:8]}")

        self.sequence_number = 1
        last_text = ""
        last_print_time = time.time()

        try:
            while not self.stop_event.is_set():
                try:
                    # Get audio from queue with timeout
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Create and send audio chunk
                timestamp_ms = int(time.time() * 1000)
                audio_msg = StreamingAudioChunk.audio_chunk(
                    session_id=self.session_id,
                    audio=audio_data,
                    sequence_number=self.sequence_number,
                    timestamp_ms=timestamp_ms,
                )

                # Send to Dynamo and get response
                async for response in await client.round_robin(audio_msg.model_dump_json()):
                    result = StreamingASRResponse.model_validate_json(response.data())

                    # Only update display if text changed
                    if result.text != last_text:
                        # Clear line and print new text
                        print(f"\r\033[K📝 {result.text}", end="", flush=True)
                        last_text = result.text
                        last_print_time = time.time()

                self.sequence_number += 1

        except asyncio.CancelledError:
            pass

        # Send END message
        print("\n\n🛑 Ending session...")
        end_msg = StreamingAudioChunk.end_session(self.session_id, self.sequence_number)
        async for response in await client.round_robin(end_msg.model_dump_json()):
            result = StreamingASRResponse.model_validate_json(response.data())
            print(f"\n{'=' * 60}")
            print(f"📋 Final transcription:")
            print(f"   {result.text}")
            print(f"{'=' * 60}")

    def stop(self):
        """Signal the streamer to stop."""
        self.stop_event.set()


@dynamo_worker(enable_nats=False)
async def mic_client(runtime: DistributedRuntime):
    """Main microphone client entry point."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice not installed.")
        print("Install it with: pip install sounddevice")
        return

    print("=" * 60)
    print("🎙️  Real-time Streaming ASR Client")
    print("=" * 60)

    # Connect to streaming ASR service
    endpoint = (
        runtime.namespace("streaming_asr")
        .component("realtime")
        .endpoint("transcribe_stream")
    )
    client = await endpoint.client()

    print("⏳ Waiting for ASR service...")
    await client.wait_for_instances()
    print("✅ Connected to ASR service!")

    # List available audio devices
    print("\n📢 Available audio devices:")
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = "→" if i == default_input else " "
            print(f"  {marker} [{i}] {dev['name']} (inputs: {dev['max_input_channels']})")

    print(f"\n🎤 Using default input device: [{default_input}] {devices[default_input]['name']}")
    print(f"📊 Sample rate: {SAMPLE_RATE} Hz, Chunk: {CHUNK_DURATION_MS}ms")
    print("\n🔴 Recording... Press Ctrl+C to stop.\n")

    # Create streamer
    streamer = MicrophoneStreamer(
        sample_rate=SAMPLE_RATE,
        chunk_duration_ms=CHUNK_DURATION_MS,
    )

    # Start audio capture in a separate thread
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=streamer.chunk_samples,
        callback=streamer.audio_callback,
    )

    try:
        with stream:
            await streamer.stream_to_dynamo(client)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        streamer.stop()
        # Give time for final message
        await asyncio.sleep(0.5)


def main():
    """Entry point."""
    uvloop.install()
    try:
        asyncio.run(mic_client())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()
