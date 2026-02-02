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
ASR Inference Worker for streaming speech-to-text.

This worker loads an RNNT model (nvidia/parakeet-rnnt-1.1b by default) and
performs chunked inference with stateful decoding. It yields partial transcriptions
as chunks are processed.

Namespace: streaming_asr
Component: inference
Endpoint: process

Tracing:
    Enable tracing by setting environment variables:
        OTEL_EXPORT_ENABLED=true
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
        OTEL_SERVICE_NAME=asr-inference
"""

import asyncio
import os
from dataclasses import dataclass

# Configure Dynamo runtime to use file-based KV store and TCP request plane
os.environ.setdefault("DYN_STORE_KV", "file")
os.environ.setdefault("DYN_REQUEST_PLANE", "tcp")
import copy
import logging
import time

import torch
import uvloop
from omegaconf import OmegaConf
from protocol import ASRRequest, ASRResponse
from tracing import get_tracer, setup_tracing, shutdown_tracing


# Fallback ContextSize for newer NeMo versions where it may have been removed
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

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="asr_inference")

setup_tracing("asr-inference")
tracer = get_tracer(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "nvidia/parakeet-rnnt-1.1b"


def make_divisible_by(num: int, factor: int) -> int:
    """Make num divisible by factor."""
    return (num // factor) * factor


class ASRInferenceHandler:
    """Handler for ASR inference with chunked decoding."""

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

        # Tracing
        self.tracer = get_tracer(__name__)

    async def load_model(self):
        """Load the RNNT model and configure for streaming inference."""
        with self.tracer.start_as_current_span("model_load") as span:
            span.set_attribute("model.name", self.model_name)
            logger.info(f"Loading ASR model: {self.model_name}")

            # Import NeMo components here to avoid import errors if not installed
            from nemo.collections.asr.models import EncDecRNNTModel
            from nemo.collections.asr.parts.submodules.rnnt_decoding import (
                RNNTDecodingConfig,
            )

            # Setup device - use CUDA_VISIBLE_DEVICES or ASR_CUDA_DEVICE env var
            if torch.cuda.is_available():
                cuda_device = os.environ.get("ASR_CUDA_DEVICE", "0")
                self.device = torch.device(f"cuda:{cuda_device}")
                can_use_bfloat16 = torch.cuda.is_bf16_supported()
                self.compute_dtype = (
                    torch.bfloat16 if can_use_bfloat16 else torch.float32
                )
            else:
                self.device = torch.device("cpu")
                self.compute_dtype = torch.float32

            span.set_attribute("device", str(self.device))
            span.set_attribute("compute_dtype", str(self.compute_dtype))
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

            # Configure decoding for greedy_batch with label looping
            # Disable CUDA graphs to avoid compilation issues with some CUDA/driver versions
            decoding_cfg = RNNTDecodingConfig()
            decoding_cfg.strategy = "greedy_batch"
            decoding_cfg.greedy.loop_labels = True
            decoding_cfg.greedy.use_cuda_graph_decoder = False  # Disable CUDA graphs
            decoding_cfg.preserve_alignments = False
            decoding_cfg.fused_batch_size = -1
            self.model.change_decoding_strategy(decoding_cfg)

            # Prepare model for inference
            self.model.preprocessor.featurizer.dither = 0.0
            self.model.preprocessor.featurizer.pad_to = 0
            self.model.freeze()
            self.model = self.model.to(self.device)
            self.model.to(self.compute_dtype)
            self.model.eval()

            # Get decoding computer for stateful decoding
            # Note: In newer NeMo versions, this is a private attribute
            decoding_obj = self.model.decoding.decoding
            if hasattr(decoding_obj, 'decoding_computer'):
                self.decoding_computer = decoding_obj.decoding_computer
            elif hasattr(decoding_obj, '_decoding_computer'):
                self.decoding_computer = decoding_obj._decoding_computer
            else:
                raise AttributeError("Cannot find decoding_computer attribute on decoding object")

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

            span.set_attribute("model.sample_rate", self.audio_sample_rate)
            span.set_attribute("model.encoder_subsampling", encoder_subsampling_factor)
            logger.info(
                f"Model loaded. Sample rate: {self.audio_sample_rate}, "
                f"encoder subsampling: {encoder_subsampling_factor}"
            )

    async def process(self, request: str):
        """
        Process an ASR request with chunked inference.

        Yields partial transcriptions as each chunk is processed.
        """
        with self.tracer.start_as_current_span("asr_inference") as root_span:
            from nemo.collections.asr.parts.utils.rnnt_utils import (
                BatchedHyps,
                batched_hyps_to_hypotheses,
            )
            from nemo.collections.asr.parts.utils.streaming_utils import (
                StreamingBatchedAudioBuffer,
            )
            # Note: ContextSize is defined at module level as a fallback

            # Parse and decode request
            with self.tracer.start_as_current_span("decode_request") as span:
                req = ASRRequest.model_validate_json(request)
                metadata = req.metadata

                audio_np = req.decode_audio()
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # [1, samples]
                audio_tensor = audio_tensor.to(device=self.device)
                audio_length = torch.tensor([req.audio_length], device=self.device)

                span.set_attribute("audio.length_samples", req.audio_length)
                span.set_attribute(
                    "audio.duration_sec", req.audio_length / metadata.sample_rate
                )

            batch_size = 1

            # Build context sizes from metadata
            context_samples = ContextSize(
                left=metadata.left_context_samples,
                chunk=metadata.chunk_samples,
                right=metadata.right_context_samples,
            )

            root_span.set_attribute("context.left_samples", context_samples.left)
            root_span.set_attribute("context.chunk_samples", context_samples.chunk)
            root_span.set_attribute("context.right_samples", context_samples.right)

            logger.info(
                f"Processing audio: {req.audio_length} samples, "
                f"context: L={context_samples.left}, C={context_samples.chunk}, R={context_samples.right}"
            )

            # Initialize streaming buffer
            buffer = StreamingBatchedAudioBuffer(
                batch_size=batch_size,
                context_samples=context_samples,
                dtype=audio_tensor.dtype,
                device=self.device,
            )

            # Stateful decoding variables
            current_batched_hyps: BatchedHyps | None = None
            state = None
            left_sample = 0
            right_sample = min(
                context_samples.chunk + context_samples.right, audio_tensor.shape[1]
            )
            rest_audio_lengths = audio_length.clone()
            chunk_idx = 0
            total_encoder_time = 0.0
            total_decoder_time = 0.0

            with torch.no_grad(), torch.inference_mode():
                # Iterate over audio chunks
                while left_sample < audio_tensor.shape[1]:
                    with self.tracer.start_as_current_span(
                        f"chunk_{chunk_idx}"
                    ) as chunk_span:
                        chunk_span.set_attribute("chunk.index", chunk_idx)
                        chunk_span.set_attribute("chunk.left_sample", left_sample)
                        chunk_span.set_attribute("chunk.right_sample", right_sample)

                        # Compute chunk boundaries
                        chunk_length = (
                            min(right_sample, audio_tensor.shape[1]) - left_sample
                        )
                        is_last_chunk_batch = chunk_length >= rest_audio_lengths
                        is_last_chunk = right_sample >= audio_tensor.shape[1]
                        chunk_lengths_batch = torch.where(
                            is_last_chunk_batch,
                            rest_audio_lengths,
                            torch.full_like(
                                rest_audio_lengths, fill_value=chunk_length
                            ),
                        )

                        chunk_span.set_attribute("chunk.is_last", is_last_chunk)

                        # Add samples to buffer [left-chunk-right]
                        buffer.add_audio_batch_(
                            audio_tensor[:, left_sample:right_sample],
                            audio_lengths=chunk_lengths_batch,
                            is_last_chunk=is_last_chunk,
                            is_last_chunk_batch=is_last_chunk_batch,
                        )

                        # Encoder forward pass on full buffer
                        with self.tracer.start_as_current_span(
                            "encoder_forward"
                        ) as enc_span:
                            start_time = time.perf_counter()
                            encoder_output, encoder_output_len = self.model(
                                input_signal=buffer.samples,
                                input_signal_length=buffer.context_size_batch.total(),
                            )
                            if self.device.type == "cuda":
                                torch.cuda.synchronize()
                            encoder_time = time.perf_counter() - start_time
                            total_encoder_time += encoder_time
                            enc_span.set_attribute(
                                "encoder.time_ms", encoder_time * 1000
                            )

                            encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]

                        # Remove left context from encoder output
                        encoder_context = buffer.context_size.subsample(
                            factor=self.encoder_frame2audio_samples
                        )
                        encoder_context_batch = buffer.context_size_batch.subsample(
                            factor=self.encoder_frame2audio_samples
                        )
                        encoder_output = encoder_output[:, encoder_context.left :]

                        # Decoder forward pass (maintains state across chunks)
                        with self.tracer.start_as_current_span(
                            "decoder_forward"
                        ) as dec_span:
                            start_time = time.perf_counter()
                            chunk_batched_hyps, _, state = self.decoding_computer(
                                x=encoder_output,
                                out_len=encoder_context_batch.chunk,
                                prev_batched_state=state,
                            )
                            if self.device.type == "cuda":
                                torch.cuda.synchronize()
                            decoder_time = time.perf_counter() - start_time
                            total_decoder_time += decoder_time
                            dec_span.set_attribute(
                                "decoder.time_ms", decoder_time * 1000
                            )

                        # Merge hypotheses
                        if current_batched_hyps is None:
                            current_batched_hyps = chunk_batched_hyps
                        else:
                            current_batched_hyps.merge_(chunk_batched_hyps)

                        # Convert to text and yield partial result
                        with self.tracer.start_as_current_span(
                            "tokenize_output"
                        ) as tok_span:
                            hyps = batched_hyps_to_hypotheses(
                                current_batched_hyps, None, batch_size=batch_size
                            )
                            partial_text = self.model.tokenizer.ids_to_text(
                                hyps[0].y_sequence.tolist()
                            )
                            tok_span.set_attribute(
                                "output.text_length", len(partial_text)
                            )

                        response = ASRResponse(
                            text=partial_text, is_final=is_last_chunk
                        )
                        yield response.model_dump_json()

                        # Move to next chunk
                        rest_audio_lengths -= chunk_lengths_batch
                        left_sample = right_sample
                        right_sample = min(
                            right_sample + context_samples.chunk, audio_tensor.shape[1]
                        )
                        chunk_idx += 1

            # Record totals on root span
            root_span.set_attribute("total_chunks", chunk_idx)
            root_span.set_attribute("total_encoder_time_ms", total_encoder_time * 1000)
            root_span.set_attribute("total_decoder_time_ms", total_decoder_time * 1000)


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    """Main worker entry point."""
    # Setup tracing

    namespace_name = "streaming_asr"
    component_name = "inference"
    endpoint_name = "process"

    component = runtime.namespace(namespace_name).component(component_name)
    endpoint = component.endpoint(endpoint_name)

    # Initialize handler and load model
    handler = ASRInferenceHandler()
    await handler.load_model()

    logger.info(f"Serving endpoint {namespace_name}/{component_name}/{endpoint_name}")

    try:
        await endpoint.serve_endpoint(handler.process)
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
