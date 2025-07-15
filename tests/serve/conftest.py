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

import pytest

# List of models used in the serve tests
SERVE_TEST_MODELS = [
    "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llava-hf/llava-1.5-7b-hf",
]

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def predownload_models():
    # Check for HF_TOKEN in environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN found in environment")
    else:
        logger.warning(
            "HF_TOKEN not found in environment. "
            "Some models may fail to download or you may encounter rate limits. "
            "Get a token from https://huggingface.co/settings/tokens"
        )

    try:
        from huggingface_hub import snapshot_download

        for model_id in SERVE_TEST_MODELS:
            logger.info(f"Pre-downloading model: {model_id}")

            try:
                # Download the full model snapshot (includes all files)
                # HuggingFace will handle caching automatically
                snapshot_download(
                    repo_id=model_id,
                    token=hf_token,
                )
                logger.info(f"Successfully pre-downloaded: {model_id}")

            except Exception as e:
                logger.error(f"Failed to pre-download {model_id}: {e}")
                # Don't fail the fixture - let individual tests handle missing models

    except ImportError:
        logger.warning(
            "huggingface_hub not installed. "
            "Models will be downloaded during test execution."
        )

    yield


# Automatically use the predownload fixture for all serve tests
def pytest_collection_modifyitems(config, items):
    for item in items:
        # Skip items that don't have fixturenames (like MypyFileItem)
        if not hasattr(item, "fixturenames"):
            continue

        # Only apply to tests in the serve directory
        if "serve" in str(item.path):
            # Check if the test already uses the fixture
            if "predownload_models" not in item.fixturenames:
                # Don't add if test explicitly marks to skip model download
                if not item.get_closest_marker("skip_model_download"):
                    item.fixturenames = list(item.fixturenames)
                    item.fixturenames.append("predownload_models")
