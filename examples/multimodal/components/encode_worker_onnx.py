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
from typing import AsyncIterator
import os

import onnxruntime as ort
from onnxruntime import OrtValue, IOBinding # Import necessary classes
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import LlavaProcessor # Needed only during init for preprocessing params

from dynamo.sdk import depends, dynamo_endpoint, service
# Assuming protocol definitions are in utils relative to components
from utils.protocol import EncodeRequest, EncodeResponse

logger = logging.getLogger(__name__)

# --- Configuration --- (Adjust paths as needed for deployment)
ONNX_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "llava_onnx_encoder") # Relative path
VISION_TOWER_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_vision_tower.onnx")
PROJECTOR_ONNX_PATH = os.path.join(ONNX_MODEL_DIR, "llava_projector.onnx")
TRT_CACHE_PATH = os.path.join(ONNX_MODEL_DIR, "trt_cache") # Cache directory

# Path to original model files (needed *only* for init to get processor params)
# In a real deployment, you might hardcode these params or load from a config file
# instead of loading the original processor here.
ORIGINAL_MODEL_PATH = "/tmp/llava-1.5-7b-hf" # Or path where original files are accessible

# Use "cuda" if GPU is available and configured, otherwise "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Assuming torch might not be present
DEVICE = "cuda" # Assume GPU is always present
ORT_DEVICE = "cuda" if DEVICE == "cuda" else "cpu" # Device string for OrtValue

# Map ONNX type strings to numpy types
ONNX_TYPE_MAP = {"tensor(float16)": np.float16, "tensor(float)": np.float32}

# --- Helper Functions (Copied/Adapted from onnx_infer.py) ---

def load_image(image_path_or_url: str) -> Image.Image:
    """Loads an image from a URL or local file path."""
    try:
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            logger.debug(f"Loaded image from URL: {image_path_or_url}")
        else:
            if not os.path.exists(image_path_or_url):
                 logger.error(f"Local image not found: {image_path_or_url}")
                 raise FileNotFoundError(f"Image not found at {image_path_or_url}")
            image = Image.open(image_path_or_url).convert("RGB")
            logger.debug(f"Loaded image from path: {image_path_or_url}")
        return image
    except Exception as e:
        logger.error(f"Error loading image '{image_path_or_url}': {e}")
        raise

def get_preprocessing_params(model_path: str):
    """Loads the processor ONCE during init to extract image preprocessing parameters."""
    try:
        logger.info(f"Loading processor from {model_path} to get preprocessing parameters (init only)...")
        processor = LlavaProcessor.from_pretrained(model_path)
        img_processor_config = processor.image_processor
        target_dtype = np.float16 if DEVICE == 'cuda' else np.float32
        logger.info(f"Using target dtype for preprocessing arrays: {target_dtype}")
        params = {
            "size": img_processor_config.size["shortest_edge"],
            "crop_size": (img_processor_config.crop_size["height"], img_processor_config.crop_size["width"]),
            "rescale_factor": img_processor_config.rescale_factor,
            "image_mean": np.array(img_processor_config.image_mean, dtype=target_dtype),
            "image_std": np.array(img_processor_config.image_std, dtype=target_dtype)
        }
        logger.info(f"Extracted preprocessing parameters: {params}")
        return params
    except Exception as e:
        logger.error(f"Failed to load processor or extract params from {model_path}: {e}")
        raise RuntimeError(f"Could not initialize preprocessing parameters: {e}")

def preprocess_image(image: Image.Image, params: dict) -> np.ndarray:
    """Replicates the image preprocessing using PIL and NumPy based on extracted params."""
    logger.debug("Starting image preprocessing...")
    target_size = params["size"]
    crop_h, crop_w = params["crop_size"]
    rescale_factor = params["rescale_factor"]
    mean = params["image_mean"]
    std = params["image_std"]
    target_dtype = mean.dtype # Get dtype from loaded params

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
    img_array = (np.array(image).astype(np.float32) * rescale_factor)
    img_array = (img_array - mean.astype(np.float32)) / std.astype(np.float32) # Use float32 for stability

    # Transpose & Add batch dim
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Ensure final dtype
    img_array = img_array.astype(target_dtype)
    logger.debug(f"Preprocessing complete. Output shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array

# --- Dynamo Service Definition ---
@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    # Adjust resources based on ONNX model needs & execution provider (consider TRT memory usage)
    resources={"gpu": 1, "cpu": "10", "memory": "25Gi"}, # Increased memory slightly
    workers=1,
)
class EncodeWorker:

    def __init__(self) -> None:
        logger.info("Initializing EncodeWorkerONNX with TensorRT and IOBinding...")
        logger.info(f"Using device for provider selection: {DEVICE}")
        logger.info(f"Using ORT device for OrtValue: {ORT_DEVICE}")
        logger.info(f"ONNX Runtime version in use: {ort.__version__}")

        # --- Get Preprocessing Params (once during init) ---
        self.preprocessing_params = get_preprocessing_params(ORIGINAL_MODEL_PATH)

        # --- Configure Execution Providers (TensorRT, CUDA, CPU) ---
        providers = []
        if DEVICE == 'cuda':
             # Ensure the TensorRT cache directory exists
            if not os.path.exists(TRT_CACHE_PATH):
                try:
                    os.makedirs(TRT_CACHE_PATH)
                    logger.info(f"Created TensorRT engine cache directory: {TRT_CACHE_PATH}")
                except OSError as e:
                    logger.error(f"Failed to create TensorRT cache directory {TRT_CACHE_PATH}: {e}")
                    # Decide how to handle this - maybe fall back to CUDA? For now, log and continue.

            providers.extend([
                ('TensorrtExecutionProvider', {
                    'device_id': 0, # Or appropriate GPU ID
                    'trt_fp16_enable': True, # Enable FP16 precision (adjust if model requires FP32)
                    'trt_engine_cache_enable': True, # Enable engine caching
                    'trt_engine_cache_path': TRT_CACHE_PATH,
                    # Add other TRT options if needed (e.g., int8, workspace size)
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0, # Or appropriate GPU ID
                    # Add other CUDA options if needed
                })
            ])
        providers.append('CPUExecutionProvider') # Fallback

        # --- Load ONNX Sessions ---
        logger.info("Loading ONNX inference sessions...")
        try:
            logger.info(f"Attempting to load ONNX models with providers: {providers}")

            if not os.path.exists(VISION_TOWER_ONNX_PATH):
                raise FileNotFoundError(f"Vision tower ONNX model not found at {VISION_TOWER_ONNX_PATH}")
            if not os.path.exists(PROJECTOR_ONNX_PATH):
                raise FileNotFoundError(f"Projector ONNX model not found at {PROJECTOR_ONNX_PATH}")

            # Set ORT log level (optional, adjust as needed)
            # ort.set_default_logger_severity(0) # Verbose logging

            self.vision_sess = ort.InferenceSession(VISION_TOWER_ONNX_PATH, providers=providers)
            logger.info(f"Loaded Vision Tower. Effective providers: {self.vision_sess.get_providers()}")
            self.proj_sess = ort.InferenceSession(PROJECTOR_ONNX_PATH, providers=providers)
            logger.info(f"Loaded Projector. Effective providers: {self.proj_sess.get_providers()}")

            # Get input/output metadata
            self.vision_input_meta = self.vision_sess.get_inputs()[0]
            self.vision_output_meta = self.vision_sess.get_outputs()[0]
            self.proj_input_meta = self.proj_sess.get_inputs()[0]
            self.proj_output_meta = self.proj_sess.get_outputs()[0]

            self.vision_input_name = self.vision_input_meta.name
            self.vision_output_name = self.vision_output_meta.name
            self.proj_input_name = self.proj_input_meta.name
            self.proj_output_name = self.proj_output_meta.name

            self.vision_output_type = ONNX_TYPE_MAP.get(self.vision_output_meta.type, np.float32)
            self.proj_output_type = ONNX_TYPE_MAP.get(self.proj_output_meta.type, np.float32)

            logger.info(f"Vision Tower I/O: Input='{self.vision_input_name}' ({self.vision_input_meta.type}, {self.vision_input_meta.shape}), Output='{self.vision_output_name}' ({self.vision_output_meta.type}, {self.vision_output_meta.shape})")
            logger.info(f"Projector I/O: Input='{self.proj_input_name}' ({self.proj_input_meta.type}, {self.proj_input_meta.shape}), Output='{self.proj_output_name}' ({self.proj_output_meta.type}, {self.proj_output_meta.shape})")
            logger.info("ONNX sessions and I/O metadata initialized.")

        except Exception as e:
            logger.error(f"Fatal error loading ONNX models or getting metadata during initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize ONNX sessions: {e}")

    def encode_image_onnx(self, image_path_or_url: str) -> np.ndarray:
        """Loads image, preprocesses, and runs ONNX inference using IOBinding."""
        try:
            # 1. Load Image
            image = load_image(image_path_or_url)

            # 2. Preprocess Image (results in NumPy array on CPU)
            preprocessed_image_np = preprocess_image(image, self.preprocessing_params)

            # Wrap NumPy input with OrtValue, potentially transferring to GPU
            preprocessed_image_ortvalue = OrtValue.ortvalue_from_numpy(preprocessed_image_np, ORT_DEVICE)
            logger.debug(f"Input image OrtValue created on device: {preprocessed_image_ortvalue.device_name()}")

            # 3. Allocate Output Buffer & Run Vision Tower Inference with IOBinding
            logger.debug("Preparing IOBinding for ONNX Vision Tower inference...")
            # Determine concrete output shape (assume batch size 1, handle symbolic dims)
            # Example for Llava: (1, 577, 1024)
            vision_output_shape = [1 if isinstance(d, str) else d for d in self.vision_output_meta.shape]
            logger.debug(f"Allocating vision output buffer with shape: {vision_output_shape} and type: {self.vision_output_type} on device: {ORT_DEVICE}")

            vision_output_ortvalue = OrtValue.ortvalue_from_shape_and_type(vision_output_shape, self.vision_output_type, ORT_DEVICE)
            logger.debug(f"Vision Tower output buffer allocated on device: {vision_output_ortvalue.device_name()}")

            # Create IOBinding for vision tower
            io_binding_vision = self.vision_sess.io_binding()
            io_binding_vision.bind_ortvalue_input(self.vision_input_name, preprocessed_image_ortvalue)
            io_binding_vision.bind_ortvalue_output(self.vision_output_name, vision_output_ortvalue)

            logger.debug("Running Vision Tower inference with IOBinding...")
            self.vision_sess.run_with_iobinding(io_binding_vision)
            logger.debug("Vision Tower inference complete. Output is in OrtValue.")
            # vision_output_ortvalue now holds the result on ORT_DEVICE

            # Clear the input OrtValue binding explicitly (good practice)
            io_binding_vision.clear_binding_inputs()


            # 4. Allocate Output Buffer & Run Projector Inference with IOBinding
            logger.debug("Preparing IOBinding for ONNX Projector inference...")
            # Determine projector output shape (e.g., [1, 577, 4096])
            # Use the actual sequence length from the vision tower output
            proj_output_shape = [1 if isinstance(d, str) else d for d in self.proj_output_meta.shape]
            actual_seq_len = vision_output_ortvalue.shape()[1] # Get dynamic sequence length
            if len(proj_output_shape) > 1 and isinstance(self.proj_output_meta.shape[1], str):
                 proj_output_shape[1] = actual_seq_len
            logger.debug(f"Allocating projector output buffer with shape: {proj_output_shape} and type: {self.proj_output_type} on device: {ORT_DEVICE}")

            final_embeddings_ortvalue = OrtValue.ortvalue_from_shape_and_type(proj_output_shape, self.proj_output_type, ORT_DEVICE)
            logger.debug(f"Projector output buffer allocated on device: {final_embeddings_ortvalue.device_name()}")

            # Create IOBinding for projector
            io_binding_proj = self.proj_sess.io_binding()
            # Bind input (the output OrtValue from the vision tower)
            io_binding_proj.bind_ortvalue_input(self.proj_input_name, vision_output_ortvalue)
            # Bind output
            io_binding_proj.bind_ortvalue_output(self.proj_output_name, final_embeddings_ortvalue)

            logger.debug("Running Projector inference with IOBinding...")
            self.proj_sess.run_with_iobinding(io_binding_proj)
            logger.debug("Projector inference complete. Final embeddings are in OrtValue.")
            # final_embeddings_ortvalue holds the final result on ORT_DEVICE

            # Clear bindings explicitly
            io_binding_proj.clear_binding_inputs()
            io_binding_proj.clear_binding_outputs()
            io_binding_vision.clear_binding_outputs() # Clear vision output binding too


            # 5. Copy final result from OrtValue (potentially GPU) to NumPy array (CPU)
            # This is the only H2D/D2H copy needed for the result (input copy happened at OrtValue creation).
            logger.debug(f"Copying final embeddings from OrtValue ({final_embeddings_ortvalue.device_name()}) to NumPy array (CPU)...")
            final_embeddings_np = final_embeddings_ortvalue.numpy()
            logger.debug(f"Final embeddings NumPy shape: {final_embeddings_np.shape}, dtype: {final_embeddings_np.dtype}")

            return final_embeddings_np

        except Exception as e:
            logger.error(f"Error during ONNX IOBinding image encoding for '{image_path_or_url}': {e}", exc_info=True)
            # Reraise to signal failure to the caller
            raise

    @dynamo_endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        """Dynamo endpoint to handle encoding requests using ONNX with IOBinding."""
        logger.info(f"Received IOBinding encode request for image: {request.image_url}")
        try:
            # Perform encoding using the ONNX IOBinding method
            image_embeds_np = self.encode_image_onnx(request.image_url) # This now uses IOBinding
            logger.info(f"ONNX IOBinding encoding successful, embedding shape: {image_embeds_np.shape}")

            # Convert NumPy array to list for JSON serialization
            yield EncodeResponse(image_features=image_embeds_np.tolist()).model_dump_json()

        except Exception as e:
            # Log error and potentially yield an error response if protocol supports it
            logger.error(f"Failed to process encode request for {request.image_url}: {e}", exc_info=True)
            # Handle error appropriately, maybe yield an error indicator if possible
            # For now, the exception will likely terminate the stream/response
            pass # Or re-raise e
