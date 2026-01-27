# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.vllm.multimodal_utils.audio_loader import AudioLoader
from dynamo.vllm.multimodal_utils.chat_message_utils import extract_user_text
from dynamo.vllm.multimodal_utils.chat_processor import (
    ChatProcessor,
    CompletionsProcessor,
    ProcessMixIn,
)
from dynamo.vllm.multimodal_utils.encode_utils import (
    encode_image_embeddings,
    get_embedding_hash,
    get_encoder_components,
)
from dynamo.vllm.multimodal_utils.http_client import get_http_client
from dynamo.vllm.multimodal_utils.image_loader import ImageLoader
from dynamo.vllm.multimodal_utils.model import (
    SupportedModels,
    construct_mm_data,
    load_vision_model,
)
from dynamo.vllm.multimodal_utils.protocol import (
    MultiModalGroup,
    MultiModalInput,
    MultiModalRequest,
    MyRequestOutput,
    PatchedTokensPrompt,
    VLLMNativeEncoderRequest,
    VLLMNativeEncoderResponse,
    vLLMMultimodalRequest,
)
from dynamo.vllm.multimodal_utils.video_utils import (
    calculate_frame_sampling_indices,
    get_video_metadata,
    load_video_content,
    open_video_container,
    prepare_tensor_for_rdma,
    read_video_pyav,
    resize_video_frames,
)

__all__ = [
    "ChatProcessor",
    "CompletionsProcessor",
    "ProcessMixIn",
    "AudioLoader",
    "encode_image_embeddings",
    "extract_user_text",
    "get_encoder_components",
    "get_http_client",
    "ImageLoader",
    "calculate_frame_sampling_indices",
    "get_video_metadata",
    "load_video_content",
    "open_video_container",
    "prepare_tensor_for_rdma",
    "read_video_pyav",
    "resize_video_frames",
    "SupportedModels",
    "construct_mm_data",
    "load_vision_model",
    "MultiModalInput",
    "MultiModalGroup",
    "PatchedTokensPrompt",
    "get_embedding_hash",
    "MultiModalRequest",
    "MyRequestOutput",
    "vLLMMultimodalRequest",
    "VLLMNativeEncoderRequest",
    "VLLMNativeEncoderResponse",
]
