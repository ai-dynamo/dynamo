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

from tests.utils.constants import TEST_MODELS
from tests.utils.managed_process import ManagedProcess


def pytest_configure(config):
    # Defining model morker to avoid `'model' not found in `markers` configuration option`
    # error when pyproject.toml is not available in the container
    config.addinivalue_line("markers", "model: model id used by a test or parameter")


LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,  # ISO 8601 UTC format
)


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
def predownload_models(pytestconfig):
    """Fixture wrapper around download_models for models used in collected tests"""
    # Get models from pytest config if available, otherwise fall back to TEST_MODELS
    models = getattr(pytestconfig, "models_to_download", None)
    if models:
        logging.info(
            f"Downloading {len(models)} models needed for collected tests\nModels: {models}"
        )
        download_models(model_list=list(models))
    else:
        # Fallback to original behavior if extraction failed
        download_models()
    yield


@pytest.fixture(scope="session")
def predownload_tokenizers(pytestconfig):
    """Fixture wrapper around download_models for tokenizers used in collected tests"""
    # Get models from pytest config if available, otherwise fall back to TEST_MODELS
    models = getattr(pytestconfig, "models_to_download", None)
    if models:
        logging.info(
            f"Downloading tokenizers for {len(models)} models needed for collected tests\nModels: {models}"
        )
        download_models(model_list=list(models), ignore_weights=True)
    else:
        # Fallback to original behavior if extraction failed
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


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """
    This function is called to modify the list of tests to run.
    """
    # Collect models via explicit pytest mark from final filtered items only
    models_to_download = set()
    for item in items:
        # Only collect from items that are not skipped
        if any(
            getattr(m, "name", "") == "skip" for m in getattr(item, "own_markers", [])
        ):
            continue
        model_mark = item.get_closest_marker("model")
        if model_mark and model_mark.args:
            models_to_download.add(model_mark.args[0])

    # Store models to download in pytest config for fixtures to access
    if models_to_download:
        config.models_to_download = models_to_download


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
