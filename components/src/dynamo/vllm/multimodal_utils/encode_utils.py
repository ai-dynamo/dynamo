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

import hashlib
import inspect
import logging
from collections import deque
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def get_embedding_hash(key: str) -> str:
    """
    Generate a unique hash key for storing/retrieving image embeddings.

    Args:
        key: The base key string (e.g., image URL or identifier)
    Returns:
        A unique hash string for the given key.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def _normalize_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int):
        return torch.device("cpu" if device < 0 else f"cuda:{device}")
    return torch.device("cpu")


def _get_module_device(module: torch.nn.Module) -> torch.device:
    device = getattr(module, "device", None)
    if device is not None:
        return _normalize_device(device)

    get_device = getattr(module, "get_device", None)
    if callable(get_device):
        try:
            return _normalize_device(get_device())
        except TypeError:
            pass

    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _get_callable_parameter_names(callable_obj: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return set()


def _looks_like_direct_vision_encoder(candidate: Any) -> bool:
    if getattr(candidate, "spatial_merge_size", None) is not None:
        return True

    param_names = _get_callable_parameter_names(getattr(candidate, "forward", candidate))
    return "grid_thw" in param_names or "image_grid_thw" in param_names


def _iter_model_candidates(root: Any):
    queue = deque([root])
    seen: set[int] = set()

    while queue:
        candidate = queue.popleft()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate

        nested_model = getattr(candidate, "model", None)
        if nested_model is not None:
            queue.append(nested_model)


def _summarize_attrs(candidate: Any) -> str:
    public_attrs = sorted(attr for attr in dir(candidate) if not attr.startswith("_"))
    return ", ".join(public_attrs[:25])


def _prepare_grid_thw(image_embeds: Dict[str, Any], device: torch.device) -> Any:
    grid_thw = image_embeds.get("image_grid_thw")
    if grid_thw is None:
        return None

    if isinstance(grid_thw, torch.Tensor):
        grid_thw = grid_thw.to(device)
    else:
        grid_thw = torch.tensor(grid_thw, device=device)

    return grid_thw


def _invoke_image_extractor(
    extractor: Any,
    image_embeds: Dict[str, Any],
    device: torch.device,
) -> Any:
    pixel_values = image_embeds["pixel_values"].to(device)
    grid_thw = _prepare_grid_thw(image_embeds, device)
    param_names = _get_callable_parameter_names(extractor)

    args = [] if "pixel_values" in param_names else [pixel_values]
    kwargs = {"pixel_values": pixel_values} if "pixel_values" in param_names else {}

    if grid_thw is not None:
        if "image_grid_thw" in param_names:
            kwargs["image_grid_thw"] = grid_thw
        elif "grid_thw" in param_names:
            # Direct `grid_thw` encoders are loaded from vLLM's encoder-only path.
            kwargs["grid_thw"] = grid_thw.tolist()
        elif param_names:
            raise ValueError(
                "Processed image inputs include `image_grid_thw`, but the selected "
                f"encoder path {extractor} does not accept `image_grid_thw` or "
                "`grid_thw`."
            )

    return extractor(*args, **kwargs)


def _merge_tensor_chunks(chunks: Any) -> torch.Tensor:
    if isinstance(chunks, torch.Tensor):
        return chunks

    if isinstance(chunks, (tuple, list)) and chunks and all(
        isinstance(item, torch.Tensor) for item in chunks
    ):
        return torch.cat(tuple(chunks), dim=0)

    raise TypeError(f"Unsupported embedding chunks type: {type(chunks)}")


def _extract_tensor_output(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs

    if isinstance(outputs, dict):
        for key in ("pooler_output", "last_hidden_state"):
            if key in outputs and outputs[key] is not None:
                return _merge_tensor_chunks(outputs[key])

    for attr in ("pooler_output", "last_hidden_state"):
        value = getattr(outputs, attr, None)
        if value is not None:
            return _merge_tensor_chunks(value)

    if isinstance(outputs, (tuple, list)):
        return _merge_tensor_chunks(outputs)

    raise TypeError(f"Unsupported embedding output type: {type(outputs)}")


def encode_image_embeddings(
    model_name: str,
    image_embeds: Dict[str, Any],
    vision_encoder: torch.nn.Module,
    projector: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Encode image embeddings using the appropriate encoder path discovered from
    the loaded model structure.
    """
    del model_name

    device = _get_module_device(vision_encoder)
    with torch.no_grad():
        if projector is not None:
            pixel_values = image_embeds["pixel_values"].to(device)
            vision_outputs = vision_encoder(pixel_values)
            hidden_states = getattr(vision_outputs, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = _extract_tensor_output(vision_outputs)
            embeddings = projector(hidden_states)
        else:
            feature_extractor = getattr(vision_encoder, "get_image_features", None)
            if callable(feature_extractor):
                raw_outputs = _invoke_image_extractor(
                    feature_extractor, image_embeds, device
                )
            else:
                raw_outputs = _invoke_image_extractor(vision_encoder, image_embeds, device)
            embeddings = _extract_tensor_output(raw_outputs)

        embeddings = embeddings.unsqueeze(0) if embeddings.ndim == 2 else embeddings

    return embeddings


def get_encoder_components(
    model_name: str, vision_model: torch.nn.Module
) -> tuple[Any, Optional[Any]]:
    """
    Resolve the encoder and optional projector from the loaded model object.

    The resolution is based on the structure of the loaded module rather than
    the model name so newly supported VLMs can work without list maintenance.
    """
    for candidate in _iter_model_candidates(vision_model):
        vision_tower = getattr(candidate, "vision_tower", None)
        if vision_tower is not None:
            return vision_tower, getattr(candidate, "multi_modal_projector", None)

        if callable(getattr(candidate, "get_image_features", None)):
            return candidate, None

        visual = getattr(candidate, "visual", None)
        if visual is not None:
            return visual, None

        if _looks_like_direct_vision_encoder(candidate):
            return candidate, None

    raise NotImplementedError(
        "Unable to locate vision encoder components for "
        f"'{model_name}'. Root model attributes: {_summarize_attrs(vision_model)}"
    )


def split_image_embeddings(
    embeddings: torch.Tensor,
    image_embeds: Dict[str, Any],
    vision_encoder: torch.nn.Module,
) -> tuple[Any, Optional[list[Any]]]:
    """
    Split encoded embeddings back into per-image chunks.

    Models that expose `image_grid_thw` plus `spatial_merge_size` produce a
    concatenated sequence and must be split with the merge metadata. Other
    models already retain the batch dimension.
    """
    image_grid_thw = (
        image_embeds["image_grid_thw"].tolist()
        if "image_grid_thw" in image_embeds
        else None
    )
    spatial_merge_size = getattr(vision_encoder, "spatial_merge_size", None)

    if image_grid_thw is not None and spatial_merge_size is not None:
        sizes = (
            image_embeds["image_grid_thw"].prod(-1) // spatial_merge_size**2
        ).tolist()
        split_embeddings = embeddings.squeeze(0).split(sizes)
        logger.debug(
            "Split grid-aware embeddings into shapes: %s",
            [tensor.shape for tensor in split_embeddings],
        )
        return split_embeddings, image_grid_thw

    logger.debug("Image embedding shape: %s", embeddings.shape)
    return embeddings, image_grid_thw
