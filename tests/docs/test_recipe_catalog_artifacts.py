# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

CATALOG = Path("docs/fern/pages/recipes/_catalog")
EXPECTED_RECIPE_IMAGES = {
    "glm-5-2": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-glm-5.2-dev.1",
    "inkling": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0-inkling-dev.1",
    "kimi-k2-6": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-kimi-k2.6-dev.1",
    "kimi-k3": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-kimi-k3-dev.1",
    "nemotron-3-super": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-super-dev.1",
    "nemotron-3-ultra": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-ultra-dev.1",
}


def test_recipe_specific_images_are_catalog_owned() -> None:
    for recipe_id, image in EXPECTED_RECIPE_IMAGES.items():
        document = yaml.safe_load(
            (CATALOG / "recipes" / f"{recipe_id}.yaml").read_text()
        )
        assert document["artifacts"]["recipe_specific_images"] == [image]
