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
Streaming ASR Service for real-time speech-to-text.

This service accepts streaming audio chunks from a microphone client
and returns partial transcriptions in real-time using NVIDIA Parakeet RNNT model.

Namespace: streaming_asr
Component: realtime
Endpoint: transcribe_stream

Usage:
    python streaming_asr_service.py
"""

import asyncio
import copy
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import uvloop
from omegaconf import OmegaConf
from protocol import (
    StreamingASRResponse,
    StreamingAudioChunk,
    StreamingConfig,
    StreamingMessageType,
)
from tracing import get_tracer, setup_tracing, shutdown_tracing

# Configure Dynamo runtime
os.environ.setdefault("DYN_STORE_KV", "file")
os.environ.setdefault("DYN_REQUEST_PLANE", "tcp")

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="streaming_asr_service")

setup_tracing("streaming-asr-service")
tracer = get_tracer(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "nvidia/parakeet-rnnt-1.1b"

# Streaming configuration
CHUNK_SIZE_SAMPLES = 32000  # 2 seconds at 16kHz - internal processing chunk
LEFT_CONTEXT_SAMPLES = 160000  # 10 seconds
RIGHT_CONTEXT_SAMPLES = 32000  # 2 seconds


@dataclass
class ContextSize:
    """Context size configuration for streaming ASR."""

    left: int
    chunk: int
    right: int

    def total(self) -> int:
        """Return total context size (left + chunk + right)."""
        return self.left + self.chunk + self.right

    def subsample(self, factor: int) -> "ContextSize":
        """Return a new ContextSize with each dimension divided by factor."""
        return ContextSize(
            left=self.left // factor,
            chunk=self.chunk // factor,
            right=self.right // factor,
        )


@dataclass
class StreamingSession:
    """State for an active streaming ASR session."""

    session_id: str
    config: StreamingConfig
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    decoder_state: Optional[object] = None
    batched_hyps: Optional[object] = None
    last_activity: float = field(default_factory=time.time)
    total_samples_processed: int = 0
    last_text: str = ""


def make_divisible_by(num: int, factor: int) -> int:
    """Make num divisible by factor."""
    return (num // factor) * factor


class StreamingASRService:
    """Real-time streaming ASR service with stateful decoding."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.decoding_computer = None
        self.device = None
        self.compute_dtype = None

        # Model configuration (populated after loading)
        self.audio_sample_rate = None
        self.encoder_frame2audio_samples = None
        self.features_frame2audio_samples = None

        # Active sessions
        self.sessions: Dict[str, StreamingSession] = {}

        # Context configuration
        self.context_samples = ContextSize(
            left=LEFT_CONTEXT_SAMPLES,
            chunk=CHUNK_SIZE_SAMPLES,
            right=RIGHT_CONTEXT_SAMPLES,
        )

        # Tracing
        self.tracer = get_tracer(__name__)

    async def load_model(self):
        """Load the RNNT model and configure for streaming inference."""
        with self.tracer.start_as_current_span("model_load") as span:
            span.set_attribute("model.name", self.model_name)
            logger.info(f"Loading ASR model: {self.model_name}")

            from nemo.collections.asr.models import EncDecRNNTModel
            from nemo.collections.asr.parts.submodules.rnnt_decoding import (
                RNNTDecodingConfig,
            )

            # Setup device
            if torch.cuda.is_available():
                cuda_device = os.environ.get("ASR_CUDA_DEVICE", "0")
                self.device = torch.device(f"cuda:{cuda_device}")
                can_use_bfloat16 = torch.cuda.is_bf16_supported()
                self.compute_dtype = torch.bfloat16 if can_use_bfloat16 else torch.float32
            else:
                self.device = torch.device("cpu")
                self.compute_dtype = torch.float32

            span.set_attribute("device", str(self.device))
            logger.info(f"Using device: {self.device}, dtype: {self.compute_dtype}")

            # Load model
            with self.tracer.start_as_current_span("model_download"):
                self.model = EncDecRNNTModel.from_pretrained(
                    self.model_name, map_location=self.device
                )

            # Configure model for streaming
            model_cfg = copy.deepcopy(self.model._cfg)
            OmegaConf.set_struct(model_cfg.preprocessor, False)
            model_cfg.preprocessor.dither = 0.0
            model_cfg.preprocessor.pad_to = 0
            OmegaConf.set_struct(model_cfg.preprocessor, True)

            # Configure decoding
            decoding_cfg = RNNTDecodingConfig()
            decoding_cfg.strategy = "greedy_batch"
            decoding_cfg.greedy.loop_labels = True
            decoding_cfg.greedy.use_cuda_graph_decoder = False  # Disable CUDA graphs
            decoding_cfg.preserve_alignments = False
            decoding_cfg.fused_batch_size = -1
            self.model.change_decoding_strategy(decoding_cfg)

            # Prepare model
            self.model.preprocessor.featurizer.dither = 0.0
            self.model.preprocessor.featurizer.pad_to = 0
            self.model.freeze()
            self.model = self.model.to(self.device)
            self.model.to(self.compute_dtype)
            self.model.eval()

            # Get decoding computer
            decoding_obj = self.model.decoding.decoding
            if hasattr(decoding_obj, "decoding_computer"):
                self.decoding_computer = decoding_obj.decoding_computer
            elif hasattr(decoding_obj, "_decoding_computer"):
                self.decoding_computer = decoding_obj._decoding_computer
            else:
                raise AttributeError("Cannot find decoding_computer attribute")

            # Extract audio processing parameters
            self.audio_sample_rate = model_cfg.preprocessor["sample_rate"]
            feature_stride_sec = model_cfg.preprocessor["window_stride"]
            encoder_subsampling_factor = self.model.encoder.subsampling_factor

            self.features_frame2audio_samples = make_divisible_by(
                int(self.audio_sample_rate * feature_stride_sec),
                factor=encoder_subsampling_factor,
            )
            self.encoder_frame2audio_samples = (
                self.features_frame2audio_samples * encoder_subsampling_factor
            )

            logger.info(
                f"Model loaded. Sample rate: {self.audio_sample_rate}, "
                f"encoder subsampling: {encoder_subsampling_factor}"
            )

    def _get_or_create_session(
        self, session_id: str, config: Optional[StreamingConfig] = None
    ) -> StreamingSession:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            if config is None:
                config = StreamingConfig()
            self.sessions[session_id] = StreamingSession(
                session_id=session_id,
                config=config,
            )
            logger.info(f"Created new session: {session_id}")
        return self.sessions[session_id]

    def _cleanup_session(self, session_id: str):
        """Remove a session and free resources."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")

    async def transcribe_stream(self, request: str):
        """
        Process streaming audio chunks and yield transcriptions.

        This endpoint handles the streaming protocol:
        - START: Initialize a new session
        - AUDIO: Process audio chunk and return transcription
        - END: Finalize session and return final transcription
        """
        from nemo.collections.asr.parts.utils.rnnt_utils import (
            BatchedHyps,
            batched_hyps_to_hypotheses,
        )
        from nemo.collections.asr.parts.utils.streaming_utils import (
            StreamingBatchedAudioBuffer,
        )

        start_time = time.perf_counter()

        # Parse request
        chunk = StreamingAudioChunk.model_validate_json(request)
        session_id = chunk.session_id

        with self.tracer.start_as_current_span("transcribe_stream") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("message_type", chunk.message_type.value)
            span.set_attribute("sequence_number", chunk.sequence_number)

            # Handle START message
            if chunk.message_type == StreamingMessageType.START:
                session = self._get_or_create_session(session_id, chunk.config)
                response = StreamingASRResponse(
                    session_id=session_id,
                    sequence_number=0,
                    text="",
                    is_partial=False,
                    is_final=False,
                )
                yield response.model_dump_json()
                return

            # Handle END message
            if chunk.message_type == StreamingMessageType.END:
                session = self.sessions.get(session_id)
                final_text = session.last_text if session else ""
                self._cleanup_session(session_id)
                response = StreamingASRResponse(
                    session_id=session_id,
                    sequence_number=chunk.sequence_number,
                    text=final_text,
                    is_partial=False,
                    is_final=True,
                )
                yield response.model_dump_json()
                return

            # Handle AUDIO message
            session = self._get_or_create_session(session_id)
            session.last_activity = time.time()

            # Decode and append audio
            audio_chunk = chunk.decode_audio()
            if audio_chunk is None:
                return

            # Resample if needed
            if session.config.sample_rate != self.audio_sample_rate:
                # Simple resampling (for production, use proper resampling)
                ratio = self.audio_sample_rate / session.config.sample_rate
                new_length = int(len(audio_chunk) * ratio)
                indices = np.linspace(0, len(audio_chunk) - 1, new_length)
                audio_chunk = np.interp(indices, np.arange(len(audio_chunk)), audio_chunk)

            # Append to session buffer
            session.audio_buffer = np.concatenate([session.audio_buffer, audio_chunk])

            # Check if we have enough audio to process
            min_samples = self.context_samples.chunk
            if len(session.audio_buffer) < min_samples:
                # Not enough audio yet, return empty partial
                response = StreamingASRResponse(
                    session_id=session_id,
                    sequence_number=chunk.sequence_number,
                    text=session.last_text,
                    is_partial=True,
                    is_final=False,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )
                yield response.model_dump_json()
                return

            # Process audio
            with torch.no_grad(), torch.inference_mode():
                # Prepare audio tensor
                audio_tensor = torch.from_numpy(session.audio_buffer).unsqueeze(0)
                audio_tensor = audio_tensor.to(device=self.device, dtype=self.compute_dtype)
                audio_length = torch.tensor([len(session.audio_buffer)], device=self.device)

                # Create streaming buffer
                buffer = StreamingBatchedAudioBuffer(
                    batch_size=1,
                    context_samples=self.context_samples,
                    dtype=audio_tensor.dtype,
                    device=self.device,
                )

                # Process in chunks
                batch_size = 1
                current_batched_hyps = session.batched_hyps
                state = session.decoder_state
                left_sample = 0
                right_sample = min(
                    self.context_samples.chunk + self.context_samples.right,
                    audio_tensor.shape[1],
                )
                rest_audio_lengths = audio_length.clone()

                while left_sample < audio_tensor.shape[1]:
                    chunk_length = min(right_sample, audio_tensor.shape[1]) - left_sample
                    is_last_chunk_batch = chunk_length >= rest_audio_lengths
                    is_last_chunk = right_sample >= audio_tensor.shape[1]
                    chunk_lengths_batch = torch.where(
                        is_last_chunk_batch,
                        rest_audio_lengths,
                        torch.full_like(rest_audio_lengths, fill_value=chunk_length),
                    )

                    # Add samples to buffer
                    buffer.add_audio_batch_(
                        audio_tensor[:, left_sample:right_sample],
                        audio_lengths=chunk_lengths_batch,
                        is_last_chunk=is_last_chunk,
                        is_last_chunk_batch=is_last_chunk_batch,
                    )

                    # Encoder forward pass
                    encoder_output, encoder_output_len = self.model(
                        input_signal=buffer.samples,
                        input_signal_length=buffer.context_size_batch.total(),
                    )
                    encoder_output = encoder_output.transpose(1, 2)

                    # Remove left context
                    encoder_context = buffer.context_size.subsample(
                        factor=self.encoder_frame2audio_samples
                    )
                    encoder_context_batch = buffer.context_size_batch.subsample(
                        factor=self.encoder_frame2audio_samples
                    )
                    encoder_output = encoder_output[:, encoder_context.left:]

                    # Decoder forward pass
                    chunk_batched_hyps, _, state = self.decoding_computer(
                        x=encoder_output,
                        out_len=encoder_context_batch.chunk,
                        prev_batched_state=state,
                    )

                    # Merge hypotheses
                    if current_batched_hyps is None:
                        current_batched_hyps = chunk_batched_hyps
                    else:
                        current_batched_hyps.merge_(chunk_batched_hyps)

                    # Move to next chunk
                    rest_audio_lengths -= chunk_lengths_batch
                    left_sample = right_sample
                    right_sample = min(
                        right_sample + self.context_samples.chunk, audio_tensor.shape[1]
                    )

                # Save state for next call
                session.decoder_state = state
                session.batched_hyps = current_batched_hyps

                # Convert to text
                hyps = batched_hyps_to_hypotheses(
                    current_batched_hyps, None, batch_size=batch_size
                )
                text = self.model.tokenizer.ids_to_text(hyps[0].y_sequence.tolist())
                session.last_text = text

                # Keep only recent audio (for context)
                max_buffer_samples = self.context_samples.left + self.context_samples.chunk
                if len(session.audio_buffer) > max_buffer_samples:
                    session.audio_buffer = session.audio_buffer[-max_buffer_samples:]

            latency_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("text_length", len(text))

            response = StreamingASRResponse(
                session_id=session_id,
                sequence_number=chunk.sequence_number,
                text=text,
                is_partial=True,
                is_final=False,
                latency_ms=latency_ms,
            )
            yield response.model_dump_json()


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    """Main worker entry point."""
    namespace_name = "streaming_asr"
    component_name = "realtime"
    endpoint_name = "transcribe_stream"

    component = runtime.namespace(namespace_name).component(component_name)
    endpoint = component.endpoint(endpoint_name)

    # Initialize service and load model
    service = StreamingASRService()
    await service.load_model()

    logger.info(f"Serving endpoint {namespace_name}/{component_name}/{endpoint_name}")

    try:
        await endpoint.serve_endpoint(service.transcribe_stream)
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
