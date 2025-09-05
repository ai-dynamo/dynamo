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
import shutil
import tempfile

import pytest

from tests.utils.managed_process import ManagedProcess

LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,  # ISO 8601 UTC format
)

# List of models used in tests
TEST_MODELS = [
    "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llava-hf/llava-1.5-7b-hf",
]


def download_models(model_list=None, ignore_weights=False):
    """Download models - can be called directly or via fixture

    Args:
        model_list: List of model IDs to download. If None, downloads TEST_MODELS.
        ignore_weights: If True, skips downloading model weight files. Default is False.
    """
    if model_list is None:
        model_list = TEST_MODELS

    # Check for HF_TOKEN in environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logging.info("HF_TOKEN found in environment")
    else:
        logging.warning(
            "HF_TOKEN not found in environment. "
            "Some models may fail to download or you may encounter rate limits. "
            "Get a token from https://huggingface.co/settings/tokens"
        )

    try:
        from huggingface_hub import snapshot_download

        for model_id in model_list:
            logging.info(
                f"Pre-downloading {'model (no weights)' if ignore_weights else 'model'}: {model_id}"
            )

            try:
                if ignore_weights:
                    # Weight file patterns to exclude (based on hub.rs implementation)
                    weight_patterns = [
                        "*.bin",
                        "*.safetensors",
                        "*.h5",
                        "*.msgpack",
                        "*.ckpt.index",
                    ]

                    # Download everything except weight files
                    snapshot_download(
                        repo_id=model_id,
                        token=hf_token,
                        ignore_patterns=weight_patterns,
                    )
                else:
                    # Download the full model snapshot (includes all files)
                    snapshot_download(
                        repo_id=model_id,
                        token=hf_token,
                    )
                logging.info(f"Successfully pre-downloaded: {model_id}")

            except Exception as e:
                logging.error(f"Failed to pre-download {model_id}: {e}")
                # Don't fail the fixture - let individual tests handle missing models

    except ImportError:
        logging.warning(
            "huggingface_hub not installed. "
            "Models will be downloaded during test execution."
        )


@pytest.fixture(scope="session")
def predownload_models():
    """Fixture wrapper around download_models for all TEST_MODELS"""
    download_models()
    yield


@pytest.fixture(scope="session")
def predownload_tokenizers():
    """Fixture wrapper around download_models for all TEST_MODELS"""
    download_models(ignore_weights=True)
    yield


@pytest.fixture(autouse=True)
def logger(request):
    log_path = os.path.join(request.node.name, "test.log.txt")
    logger = logging.getLogger()
    shutil.rmtree(request.node.name, ignore_errors=True)
    os.makedirs(request.node.name, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    yield
    handler.close()
    logger.removeHandler(handler)


def pytest_collection_modifyitems(config, items):
    """
    This function is called to modify the list of tests to run.
    It is used to skip tests that are not supported on all environments.
    """

    # Tests marked with trtllm requires specific environment with tensorrtllm
    # installed. Hence, we skip them if the user did not explicitly ask for them.
    if config.getoption("-m") and "trtllm_marker" in config.getoption("-m"):
        return
    skip_trtllm = pytest.mark.skip(reason="need -m trtllm_marker to run")
    for item in items:
        if "trtllm_marker" in item.keywords:
            item.add_marker(skip_trtllm)

        # Auto-inject predownload_models fixture for serve tests only (not router tests)
        # Skip items that don't have fixturenames (like MypyFileItem)
        if hasattr(item, "fixturenames"):
            # Guard clause: skip if already has the fixtures
            if (
                "predownload_models" in item.fixturenames
                or "predownload_tokenizers" in item.fixturenames
            ):
                continue

            # Guard clause: skip if marked with skip_model_download
            if item.get_closest_marker("skip_model_download"):
                continue

            # Add appropriate fixture based on test path
            if "serve" in str(item.path):
                item.fixturenames = list(item.fixturenames)
                item.fixturenames.append("predownload_models")
            elif "router" in str(item.path):
                item.fixturenames = list(item.fixturenames)
                item.fixturenames.append("predownload_tokenizers")


class EtcdServer(ManagedProcess):
    def __init__(self, request, port=2379, timeout=300):
        port_string = str(port)
        etcd_env = os.environ.copy()
        etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
        data_dir = tempfile.mkdtemp(prefix="etcd_")
        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--advertise-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--data-dir",
            data_dir,
        ]
        super().__init__(
            env=etcd_env,
            command=command,
            timeout=timeout,
            display_output=False,
            health_check_ports=[port],
            data_dir=data_dir,
            log_dir=request.node.name,
        )


class NatsServer(ManagedProcess):
    def __init__(self, request, port=4222, timeout=300):
        data_dir = tempfile.mkdtemp(prefix="nats_")
        command = ["nats-server", "-js", "--trace", "--store_dir", data_dir]
        super().__init__(
            command=command,
            timeout=timeout,
            display_output=False,
            data_dir=data_dir,
            health_check_ports=[port],
            log_dir=request.node.name,
        )


@pytest.fixture()
def runtime_services(request):
    with NatsServer(request) as nats_process:
        with EtcdServer(request) as etcd_process:
            yield nats_process, etcd_process
