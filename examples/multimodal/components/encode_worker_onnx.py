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

import logging
import os
from io import BytesIO
from typing import AsyncIterator

import numpy as np
import onnxruntime as ort
import requests
from onnxruntime import OrtValue
from PIL import Image
from pydantic import BaseModel, Field
from transformers import (
    LlavaProcessor,  # Needed only during init for preprocessing params
)
from utils.protocol import EncodeRequest, EncodeResponse

from dynamo.sdk import endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


def load_image(image_path_or_url: str) -> Image.Image:
    """Loads an image from a URL or local file path."""
    try:
        if image_path_or_url.startswith("http") or image_path_or_url.startswith(
            "https"
        ):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        raise e
    return image


def get_preprocessing_params(model_path: str, device_str: str):
    """Loads the processor ONCE during init to extract image preprocessing parameters."""
    try:
        logger.info(
            f"Loading processor from {model_path} to get preprocessing parameters (init only)..."
        )
        processor = LlavaProcessor.from_pretrained(model_path)
        img_processor_config = processor.image_processor
        target_dtype = np.float16 if device_str == "cuda" else np.float32
        logger.info(f"Using target dtype for preprocessing arrays: {target_dtype}")
        params = {
            "size": img_processor_config.size["shortest_edge"],
            "crop_size": (
                img_processor_config.crop_size["height"],
                img_processor_config.crop_size["width"],
            ),
            "rescale_factor": img_processor_config.rescale_factor,
            "image_mean": np.array(img_processor_config.image_mean, dtype=target_dtype),
            "image_std": np.array(img_processor_config.image_std, dtype=target_dtype),
        }
        logger.info(f"Extracted preprocessing parameters: {params}")
        return params
    except Exception as e:
        logger.error(
            f"Failed to load processor or extract params from {model_path}: {e}"
        )
        raise RuntimeError(f"Could not initialize preprocessing parameters: {e}") from e


def preprocess_image(image: Image.Image, params: dict) -> np.ndarray:
    """Replicates the image preprocessing using PIL and NumPy based on extracted params."""
    logger.debug("Starting image preprocessing...")
    target_size = params["size"]
    crop_h, crop_w = params["crop_size"]
    rescale_factor = params["rescale_factor"]
    mean = params["image_mean"]
    std = params["image_std"]
    target_dtype = mean.dtype  # Get dtype from loaded params

    # Resize
    img_w, img_h = image.size
    if img_w < img_h:
        new_w = target_size
        new_h = int(target_size * img_h / img_w)
    else:
        new_h = target_size
        new_w = int(target_size * img_w / img_h)
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # Center Crop
    left = (new_w - crop_w) / 2
    top = (new_h - crop_h) / 2
    right = (new_w + crop_w) / 2
    bottom = (new_h + crop_h) / 2
    image = image.crop((left, top, right, bottom))

    # Convert, rescale, normalize
    img_array = np.array(image).astype(np.float32) * rescale_factor
    img_array = (img_array - mean.astype(np.float32)) / std.astype(
        np.float32
    )  # Use float32 for stability

    # Transpose & Add batch dim
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Ensure final dtype
    img_array = img_array.astype(target_dtype)
    logger.debug(
        f"Preprocessing complete. Output shape: {img_array.shape}, dtype: {img_array.dtype}"
    )
    return img_array


class FrameworkArgsConfig(BaseModel):
    """Configuration for framework-specific arguments."""

    encode_framework: str = Field(
        alias="encode-framework"
    )  # Required, must be 'onnx' or 'pytorch'
    onnx_model_path: str | None = Field(
        default=None, alias="onnx-model-path"
    )  # Optional, only needed for ONNX workers
    hf_model_path: str | None = Field(
        default=None, alias="hf-model-path"
    )  # Optional, only needed for ONNX workers


# --- Dynamo Service Definition ---
@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "25Gi"},
    workers=1,
)
class EncodeWorker:
    def __init__(self) -> None:
        config = ServiceConfig.get_instance()

        # Get FrameworkArgs specific config
        raw_framework_args_config = config.get("FrameworkArgs", {})
        framework_args_config = FrameworkArgsConfig(**raw_framework_args_config)

        if not framework_args_config.onnx_model_path:
            error_msg = "'onnx-model-path' must be provided in top-level 'FrameworkArgs' for the ONNX worker."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not framework_args_config.hf_model_path:
            error_msg = "'hf-model-path' must be provided in top-level 'FrameworkArgs' for the ONNX worker (for preprocessing parameters)."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Initializing EncodeWorkerONNX with TensorRT and IOBinding...")
        logger.info(f"ONNX Runtime version in use: {ort.__version__}")
        logger.info(f"Encode Framework: {framework_args_config.encode_framework}")

        # --- Instance attributes from config ---
        self.onnx_model_dir = framework_args_config.onnx_model_path
        self.vision_tower_onnx_path = os.path.join(
            self.onnx_model_dir, "llava_vision_tower.onnx"
        )
        self.projector_onnx_path = os.path.join(
            self.onnx_model_dir, "llava_projector.onnx"
        )
        self.trt_cache_path = os.path.join(self.onnx_model_dir, "trt_cache")
        self.original_model_path = framework_args_config.hf_model_path
        self.device = "cuda"
        self.ort_device = "cuda"
        # Maps ONNX tensor type strings (e.g., 'tensor(float16)') to NumPy dtypes (e.g., np.float16)
        # for correct buffer allocation, especially with IOBinding.
        self.onnx_type_map = {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
        }

        # --- Get Preprocessing Params (once during init) ---
        self.preprocessing_params = get_preprocessing_params(
            self.original_model_path, self.device
        )

        # --- Configure Execution Providers (TensorRT, CUDA) ---
        providers = []
        # Ensure the TensorRT cache directory exists
        if not os.path.exists(self.trt_cache_path):
            try:
                os.makedirs(self.trt_cache_path)
                logger.info(
                    f"Created TensorRT engine cache directory: {self.trt_cache_path}"
                )
            except OSError as e:
                logger.error(
                    f"Failed to create TensorRT cache directory {self.trt_cache_path}: {e}"
                )
                # Decide how to handle this - maybe fall back to CUDA? For now raise an error
                raise

        providers.extend(
            [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,  # Enable engine caching
                        "trt_engine_cache_path": self.trt_cache_path,
                    },
                ),
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                    },
                ),
            ]
        )

        # --- Load ONNX Sessions ---
        logger.info("Loading ONNX inference sessions...")
        try:
            logger.info(f"Attempting to load ONNX models with providers: {providers}")

            if not os.path.exists(self.vision_tower_onnx_path):
                raise FileNotFoundError(
                    f"Vision tower ONNX model not found at {self.vision_tower_onnx_path}"
                )
            if not os.path.exists(self.projector_onnx_path):
                raise FileNotFoundError(
                    f"Projector ONNX model not found at {self.projector_onnx_path}"
                )

            self.vision_sess = ort.InferenceSession(
                self.vision_tower_onnx_path, providers=providers
            )
            logger.info(
                f"Loaded Vision Tower. Effective providers: {self.vision_sess.get_providers()}"
            )
            self.proj_sess = ort.InferenceSession(
                self.projector_onnx_path, providers=providers
            )
            logger.info(
                f"Loaded Projector. Effective providers: {self.proj_sess.get_providers()}"
            )

            # Get input/output metadata
            self.vision_input_meta = self.vision_sess.get_inputs()[0]
            self.vision_output_meta = self.vision_sess.get_outputs()[0]
            self.proj_input_meta = self.proj_sess.get_inputs()[0]
            self.proj_output_meta = self.proj_sess.get_outputs()[0]

            self.vision_input_name = self.vision_input_meta.name
            self.vision_output_name = self.vision_output_meta.name
            self.proj_input_name = self.proj_input_meta.name
            self.proj_output_name = self.proj_output_meta.name

            self.vision_output_type = self.onnx_type_map.get(
                self.vision_output_meta.type, np.float32
            )
            self.proj_output_type = self.onnx_type_map.get(
                self.proj_output_meta.type, np.float32
            )

            logger.info(
                f"Vision Tower I/O: Input='{self.vision_input_name}' ({self.vision_input_meta.type}, {self.vision_input_meta.shape}), Output='{self.vision_output_name}' ({self.vision_output_meta.type}, {self.vision_output_meta.shape})"
            )
            logger.info(
                f"Projector I/O: Input='{self.proj_input_name}' ({self.proj_input_meta.type}, {self.proj_input_meta.shape}), Output='{self.proj_output_name}' ({self.proj_output_meta.type}, {self.proj_output_meta.shape})"
            )
            logger.info("ONNX sessions and I/O metadata initialized.")

        except Exception as e:
            logger.error(
                f"Fatal error loading ONNX models or getting metadata during initialization: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to initialize ONNX sessions: {e}") from e

    def encode_image_onnx(self, image_path_or_url: str) -> np.ndarray:
        """Loads image, preprocesses, and runs ONNX inference using IOBinding."""
        try:
            # Load Image
            image = load_image(image_path_or_url)

            # Preprocess Image (results in NumPy array on CPU)
            preprocessed_image_np = preprocess_image(image, self.preprocessing_params)

            # Wrap NumPy input with OrtValue, potentially transferring to GPU
            preprocessed_image_ortvalue = OrtValue.ortvalue_from_numpy(
                preprocessed_image_np, self.ort_device
            )
            logger.debug(
                f"Input image OrtValue created on device: {preprocessed_image_ortvalue.device_name()}"
            )

            # Allocate Output Buffer & Run Vision Tower Inference with IOBinding
            logger.debug("Preparing IOBinding for ONNX Vision Tower inference...")
            vision_output_shape = [
                1 if isinstance(d, str) else d for d in self.vision_output_meta.shape
            ]
            logger.debug(
                f"Allocating vision output buffer with shape: {vision_output_shape} and type: {self.vision_output_type} on device: {self.ort_device}"
            )

            vision_output_ortvalue = OrtValue.ortvalue_from_shape_and_type(
                vision_output_shape, self.vision_output_type, self.ort_device
            )
            logger.debug(
                f"Vision Tower output buffer allocated on device: {vision_output_ortvalue.device_name()}"
            )

            # Create IOBinding for vision tower
            io_binding_vision = self.vision_sess.io_binding()
            io_binding_vision.bind_ortvalue_input(
                self.vision_input_name, preprocessed_image_ortvalue
            )
            io_binding_vision.bind_ortvalue_output(
                self.vision_output_name, vision_output_ortvalue
            )

            logger.debug("Running Vision Tower inference with IOBinding...")
            self.vision_sess.run_with_iobinding(io_binding_vision)
            logger.debug("Vision Tower inference complete. Output is in OrtValue.")
            # vision_output_ortvalue now holds the result on ORT_DEVICE

            # Clear the input OrtValue binding explicitly (good practice)
            io_binding_vision.clear_binding_inputs()

            # Allocate Output Buffer & Run Projector Inference with IOBinding
            logger.debug("Preparing IOBinding for ONNX Projector inference...")
            proj_output_shape = [
                1 if isinstance(d, str) else d for d in self.proj_output_meta.shape
            ]
            actual_seq_len = vision_output_ortvalue.shape()[
                1
            ]  # Get dynamic sequence length
            if len(proj_output_shape) > 1 and isinstance(
                self.proj_output_meta.shape[1], str
            ):
                proj_output_shape[1] = actual_seq_len
            logger.debug(
                f"Allocating projector output buffer with shape: {proj_output_shape} and type: {self.proj_output_type} on device: {self.ort_device}"
            )

            final_embeddings_ortvalue = OrtValue.ortvalue_from_shape_and_type(
                proj_output_shape, self.proj_output_type, self.ort_device
            )
            logger.debug(
                f"Projector output buffer allocated on device: {final_embeddings_ortvalue.device_name()}"
            )

            # Create IOBinding for projector
            io_binding_proj = self.proj_sess.io_binding()
            # Bind input (the output OrtValue from the vision tower)
            io_binding_proj.bind_ortvalue_input(
                self.proj_input_name, vision_output_ortvalue
            )
            # Bind output
            io_binding_proj.bind_ortvalue_output(
                self.proj_output_name, final_embeddings_ortvalue
            )

            logger.debug("Running Projector inference with IOBinding...")
            self.proj_sess.run_with_iobinding(io_binding_proj)
            logger.debug(
                "Projector inference complete. Final embeddings are in OrtValue."
            )
            # final_embeddings_ortvalue holds the final result on ORT_DEVICE

            # Clear bindings explicitly
            io_binding_proj.clear_binding_inputs()
            io_binding_proj.clear_binding_outputs()
            io_binding_vision.clear_binding_outputs()

            # Copy final result from OrtValue (potentially GPU) to NumPy array (CPU)
            # This is the only H2D/D2H copy needed for the result (input copy happened at OrtValue creation).
            # TODO : Change after NIXL Supoort
            logger.debug(
                f"Copying final embeddings from OrtValue ({final_embeddings_ortvalue.device_name()}) to NumPy array (CPU)..."
            )
            final_embeddings_np = final_embeddings_ortvalue.numpy()

            return final_embeddings_np

        except Exception as e:
            logger.error(
                f"Error during ONNX IOBinding image encoding for '{image_path_or_url}': {e}",
                exc_info=True,
            )
            raise e

    @endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        """Dynamo endpoint to handle encoding requests using ONNX with IOBinding."""
        logger.info(f"Received IOBinding encode request for image: {request.image_url}")
        try:
            # Perform encoding using the ONNX IOBinding method
            image_embeds_np = self.encode_image_onnx(
                request.image_url
            )  # This now uses IOBinding
            logger.info(
                f"ONNX IOBinding encoding successful, embedding shape: {image_embeds_np.shape}"
            )

            # Convert NumPy array to list for JSON serialization
            yield EncodeResponse(
                image_features=image_embeds_np.tolist()
            ).model_dump_json()

        except Exception as e:
            logger.error(
                f"Failed to process encode request for {request.image_url}: {e}",
                exc_info=True,
            )
            raise e
