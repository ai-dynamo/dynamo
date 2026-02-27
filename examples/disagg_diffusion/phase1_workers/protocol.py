# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protocol types for disaggregated diffusion stages.

These Pydantic models define the request/response contracts between
Encoder, Denoiser, and VAE workers.  Tensor payloads are serialized
as base64-encoded torch.save bytes.  For production use, replace with
shared-memory or object-store references.
"""

import base64
import io
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Tensor serialization helpers
# ---------------------------------------------------------------------------

def tensor_to_b64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t.cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_tensor(s: str, device: str = "cuda") -> torch.Tensor:
    raw = base64.b64decode(s)
    return torch.load(io.BytesIO(raw), weights_only=True).to(device)


def tensors_to_b64(tensors: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    cpu_tensors = {k: v.cpu() for k, v in tensors.items()}
    torch.save(cpu_tensors, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_tensors(s: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(s)
    data = torch.load(io.BytesIO(raw), weights_only=True)
    return {k: v.to(device) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Stage 1: Encoder
# ---------------------------------------------------------------------------

class EncoderRequest(BaseModel):
    prompt: str
    model: str = "black-forest-labs/FLUX.1-schnell"


class EncoderResponse(BaseModel):
    """Serialized text embeddings."""
    embeddings_b64: str  # base64(torch.save({prompt_embeds, pooled_prompt_embeds, text_ids}))
    shapes: Dict[str, List[int]]


# ---------------------------------------------------------------------------
# Stage 2: Denoiser
# ---------------------------------------------------------------------------

class DenoiserRequest(BaseModel):
    embeddings_b64: str
    model: str = "black-forest-labs/FLUX.1-schnell"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    seed: int = 42


class DenoiserResponse(BaseModel):
    """Serialized denoised latents."""
    latents_b64: str  # base64(torch.save({latents, scaling_factor, shift_factor}))
    shape: List[int]


# ---------------------------------------------------------------------------
# Stage 3: VAE Decoder
# ---------------------------------------------------------------------------

class VAEDecodeRequest(BaseModel):
    latents_b64: str
    model: str = "black-forest-labs/FLUX.1-schnell"


class VAEDecodeResponse(BaseModel):
    """Final output image."""
    image_b64: Optional[str] = None  # base64 PNG
    url: Optional[str] = None


# ---------------------------------------------------------------------------
# End-to-end (for orchestrator convenience)
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "black-forest-labs/FLUX.1-schnell"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    seed: int = 42
    response_format: str = "b64_json"


class GenerateResponse(BaseModel):
    image_b64: Optional[str] = None
    url: Optional[str] = None
    timings: Optional[Dict[str, float]] = None
