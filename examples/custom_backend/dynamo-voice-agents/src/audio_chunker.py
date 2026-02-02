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
Audio Chunker Worker for streaming ASR.

This worker serves as the entry point for the streaming ASR pipeline.
It loads audio files, computes chunk boundaries based on streaming config,
and forwards requests to the ASR inference worker.

Namespace: streaming_asr
Component: chunker
Endpoint: transcribe

Tracing:
    Enable tracing by setting environment variables:
        OTEL_EXPORT_ENABLED=true
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
        OTEL_SERVICE_NAME=audio-chunker
"""

import asyncio
import os

# Configure Dynamo runtime to use file-based KV store and TCP request plane
os.environ.setdefault("DYN_STORE_KV", "file")
os.environ.setdefault("DYN_REQUEST_PLANE", "tcp")
import logging
import time

import torchaudio
import uvloop
from protocol import ASRRequest, ASRResponse, ChunkMetadata
from tracing import get_tracer, setup_tracing, shutdown_tracing

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="audio_chunker")


class AudioChunkerHandler:
    """Handler for audio loading and chunk configuration."""

    def __init__(self, asr_client):
        self.asr_client = asr_client

        # Streaming configuration (from nemo_example.py defaults)
        # 10-2-2 config: good balance of quality and latency
        self.chunk_secs = 2.0
        self.left_context_secs = 10.0
        self.right_context_secs = 2.0
        self.sample_rate = 16000

        # Encoder subsampling factor for parakeet-rnnt-1.1b
        self.encoder_subsampling_factor = 8
        self.feature_stride_sec = 0.01  # 10ms window stride

        # Tracing
        self.tracer = get_tracer(__name__)

    def _make_divisible_by(self, num: int, factor: int) -> int:
        """Make num divisible by factor."""
        return (num // factor) * factor

    def _compute_context_samples(self) -> ChunkMetadata:
        """Compute context sizes in samples, aligned to encoder boundaries."""
        features_per_sec = 1.0 / self.feature_stride_sec

        features_frame2audio_samples = self._make_divisible_by(
            int(self.sample_rate * self.feature_stride_sec),
            factor=self.encoder_subsampling_factor,
        )
        encoder_frame2audio_samples = (
            features_frame2audio_samples * self.encoder_subsampling_factor
        )

        # Compute encoder frames for each context
        left_encoder_frames = int(
            self.left_context_secs * features_per_sec / self.encoder_subsampling_factor
        )
        chunk_encoder_frames = int(
            self.chunk_secs * features_per_sec / self.encoder_subsampling_factor
        )
        right_encoder_frames = int(
            self.right_context_secs * features_per_sec / self.encoder_subsampling_factor
        )

        # Convert to samples
        return ChunkMetadata(
            left_context_samples=left_encoder_frames * encoder_frame2audio_samples,
            chunk_samples=chunk_encoder_frames * encoder_frame2audio_samples,
            right_context_samples=right_encoder_frames * encoder_frame2audio_samples,
            sample_rate=self.sample_rate,
        )

    async def transcribe(self, request: str):
        """
        Transcribe an audio file.

        Args:
            request: Path to the audio file to transcribe.

        Yields:
            JSON-encoded ASRResponse objects with partial and final transcriptions.
        """
        with self.tracer.start_as_current_span("transcribe") as root_span:
            audio_path = request.strip()
            root_span.set_attribute("audio.path", audio_path)
            logger.info(f"Transcribing: {audio_path}")

            # Load audio using torchaudio
            with self.tracer.start_as_current_span("load_audio") as load_span:
                start_time = time.perf_counter()
                try:
                    audio, sr = torchaudio.load(audio_path)
                    load_time = time.perf_counter() - start_time
                    load_span.set_attribute("load_time_ms", load_time * 1000)
                    load_span.set_attribute("original_sample_rate", sr)
                except Exception as e:
                    logger.error(f"Failed to load audio file: {e}")
                    load_span.record_exception(e)
                    yield ASRResponse(
                        text=f"Error loading audio: {e}", is_final=True
                    ).model_dump_json()
                    return

                # Convert to mono if stereo
                if audio.shape[0] > 1:
                    load_span.set_attribute("channels", audio.shape[0])
                    audio = audio.mean(dim=0, keepdim=True)

                # Resample if necessary
                if sr != self.sample_rate:
                    with self.tracer.start_as_current_span("resample") as resample_span:
                        resample_span.set_attribute("from_rate", sr)
                        resample_span.set_attribute("to_rate", self.sample_rate)
                        logger.info(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
                        start_time = time.perf_counter()
                        audio = torchaudio.functional.resample(
                            audio, sr, self.sample_rate
                        )
                        resample_time = time.perf_counter() - start_time
                        resample_span.set_attribute(
                            "resample_time_ms", resample_time * 1000
                        )

            # Get audio as numpy array
            audio_np = audio.squeeze(0).numpy()
            audio_length = len(audio_np)
            audio_duration_sec = audio_length / self.sample_rate

            root_span.set_attribute("audio.length_samples", audio_length)
            root_span.set_attribute("audio.duration_sec", audio_duration_sec)
            logger.info(
                f"Audio loaded: {audio_length} samples ({audio_duration_sec:.2f}s)"
            )

            # Compute chunk metadata
            with self.tracer.start_as_current_span("compute_metadata") as meta_span:
                metadata = self._compute_context_samples()
                meta_span.set_attribute(
                    "context.left_samples", metadata.left_context_samples
                )
                meta_span.set_attribute("context.chunk_samples", metadata.chunk_samples)
                meta_span.set_attribute(
                    "context.right_samples", metadata.right_context_samples
                )

            # Build ASR request
            with self.tracer.start_as_current_span("encode_request") as enc_span:
                start_time = time.perf_counter()
                asr_request = ASRRequest.from_audio(
                    audio=audio_np, audio_length=audio_length, metadata=metadata
                )
                encode_time = time.perf_counter() - start_time
                enc_span.set_attribute("encode_time_ms", encode_time * 1000)
                enc_span.set_attribute("payload_size_bytes", len(asr_request.audio_b64))

            # Call ASR worker and stream responses
            with self.tracer.start_as_current_span("asr_rpc") as rpc_span:
                response_count = 0
                first_response_time = None
                start_time = time.perf_counter()

                try:
                    async for response in await self.asr_client.round_robin(
                        asr_request.model_dump_json()
                    ):
                        if first_response_time is None:
                            first_response_time = time.perf_counter() - start_time
                            rpc_span.set_attribute(
                                "time_to_first_response_ms", first_response_time * 1000
                            )
                        response_count += 1
                        yield response.data()

                    total_time = time.perf_counter() - start_time
                    rpc_span.set_attribute("total_rpc_time_ms", total_time * 1000)
                    rpc_span.set_attribute("response_count", response_count)
                    root_span.set_attribute(
                        "real_time_factor", total_time / audio_duration_sec
                    )

                except Exception as e:
                    logger.error(f"ASR inference error: {e}")
                    rpc_span.record_exception(e)
                    yield ASRResponse(
                        text=f"Error during inference: {e}", is_final=True
                    ).model_dump_json()


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    """Main worker entry point."""
    # Setup tracing
    setup_tracing("audio-chunker")

    namespace_name = "streaming_asr"

    # Create client to ASR inference worker
    asr_endpoint = (
        runtime.namespace(namespace_name).component("inference").endpoint("process")
    )
    asr_client = await asr_endpoint.client()

    # Wait for ASR inference worker with retries
    max_retries = 30
    retry_delay = 5  # seconds
    logger.info("Waiting for ASR inference worker...")
    
    for attempt in range(max_retries):
        try:
            await asr_client.wait_for_instances()
            logger.info("ASR inference worker connected")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"ASR worker not ready (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to ASR worker after {max_retries} attempts")
                raise

    # Create and serve chunker endpoint
    component = runtime.namespace(namespace_name).component("chunker")
    endpoint = component.endpoint("transcribe")

    handler = AudioChunkerHandler(asr_client)

    logger.info(f"Serving endpoint {namespace_name}/chunker/transcribe")

    try:
        await endpoint.serve_endpoint(handler.transcribe)
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
