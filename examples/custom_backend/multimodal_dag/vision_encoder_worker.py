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

"""Separate custom Python worker that runs the Qwen2.5-VL vision tower."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import torch
import uvloop
from transformers import AutoProcessor

from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.multimodal_utils.encode_utils import (
    encode_image_embeddings,
    get_encoder_components,
)
from dynamo.vllm.multimodal_utils.external_qwen_artifact import ExternalQwenArtifact
from dynamo.vllm.multimodal_utils.model import load_vision_model
from examples.custom_backend.multimodal_dag.protocol import (
    DEFAULT_BACKEND_MODEL,
    VISION_ENCODER_ENDPOINT,
    validate_chat_request,
)

configure_dynamo_logging(service_name="multimodal-dag-vision-encoder")
logger = logging.getLogger(__name__)


def render_unexpanded_prompt_token_ids(
    processor: Any,
    messages: list[dict[str, Any]],
) -> list[int]:
    """Render one canonical image token without running media preprocessing."""

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Qwen processor did not return a rendered prompt")
    token_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise ValueError("Qwen tokenizer did not return token IDs")
    return token_ids


class CustomVisionEncoder:
    """Load the Qwen processor and vision-only model once per worker."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._processor = AutoProcessor.from_pretrained(model)
        self._image_loader = ImageLoader()
        vision_model = load_vision_model(model, enforce_eager=True)
        self._vision_encoder, self._projector = get_encoder_components(
            model, vision_model
        )
        logger.info("Loaded external Qwen vision encoder for %s", model)

    async def generate(
        self,
        request: Mapping[str, Any],
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        del context
        validated = validate_chat_request(request)
        logger.info("Encoding one image for model %s", self._model)

        image = await self._image_loader.load_image(validated.image_url)
        prompt_token_ids, image_inputs = await asyncio.gather(
            asyncio.to_thread(
                render_unexpanded_prompt_token_ids,
                self._processor,
                validated.processor_messages,
            ),
            asyncio.to_thread(
                self._processor.image_processor,
                images=[image],
                return_tensors="pt",
            ),
        )
        embeddings = await asyncio.to_thread(
            encode_image_embeddings,
            model_name=self._model,
            image_embeds=image_inputs,
            vision_encoder=self._vision_encoder,
            projector=self._projector,
        )
        if embeddings.dim() != 3 or embeddings.shape[0] != 1:
            raise ValueError(
                "vision encoder must return [1, visual_tokens, hidden_size]; "
                f"got {tuple(embeddings.shape)}"
            )
        projected = (
            embeddings[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        )
        grid_tensor = image_inputs.get("image_grid_thw")
        if not isinstance(grid_tensor, torch.Tensor):
            raise ValueError("Qwen image processor did not return image_grid_thw")
        grid = grid_tensor.tolist()

        artifact = ExternalQwenArtifact.create(
            model=self._model,
            prompt_token_ids=prompt_token_ids,
            image_embeds=projected,
            image_grid_thw=grid,
        )
        logger.info(
            "Encoded image to projected shape=%s grid=%s",
            tuple(projected.shape),
            grid,
        )
        yield artifact.to_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_BACKEND_MODEL)
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    args = _parse_args()
    handler = CustomVisionEncoder(args.model)
    endpoint = runtime.endpoint(VISION_ENCODER_ENDPOINT)
    await endpoint.serve_endpoint(
        handler.generate,
        graceful_shutdown=True,
        metrics_labels=[("service", "multimodal_dag_vision_encoder")],
    )


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
