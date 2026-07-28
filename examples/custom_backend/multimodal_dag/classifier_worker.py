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

"""CPU-only dummy classifier for projected Qwen image embeddings."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import torch
import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.multimodal_utils.external_qwen_artifact import ExternalQwenArtifact
from examples.custom_backend.multimodal_dag.protocol import CLASSIFIER_ENDPOINT

configure_dynamo_logging(service_name="multimodal-dag-classifier")
logger = logging.getLogger(__name__)


def classify_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Mean-pool projected rows and apply a fixed two-logit dummy head."""

    artifact = ExternalQwenArtifact.from_dict(payload)
    image_embeds = artifact.load_image_embeds()
    if image_embeds.shape[1] < 2:
        raise ValueError("dummy classifier requires embedding hidden size >= 2")

    pooled = image_embeds.float().mean(dim=0)
    probabilities = torch.softmax(pooled[:2], dim=0)
    class_index = int(torch.argmax(probabilities).item())
    return {
        "label": f"class_{class_index}",
        "score": float(probabilities[class_index].item()),
        "embedding_shape": list(image_embeds.shape),
    }


class DummyClassifier:
    """Dynamo endpoint wrapper around the deterministic dummy head."""

    async def generate(
        self,
        request: Mapping[str, Any],
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        del context
        logger.info("Starting dummy classification")
        result = classify_artifact(request)
        logger.info("Completed dummy classification")
        yield result


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    endpoint = runtime.endpoint(CLASSIFIER_ENDPOINT)
    await endpoint.serve_endpoint(
        DummyClassifier().generate,
        graceful_shutdown=True,
        metrics_labels=[("service", "multimodal_dag_classifier")],
    )


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
