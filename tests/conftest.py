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
import tempfile

import pytest

from tests.utils.managed_process import ManagedProcess

# Custom format inspired by your example
LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",  # ISO 8601 UTC format
)


def pytest_collection_modifyitems(config, items):
    """
    This function is called to modify the list of tests to run.
    It is used to skip tests that are not supported on all environments.
    """

    # Tests marked with tensorrtllm requires specific environment with tensorrtllm
    # installed. Hence, we skip them if the user did not explicitly ask for them.
    if config.getoption("-m") and "tensorrtllm" in config.getoption("-m"):
        return
    skip_tensorrtllm = pytest.mark.skip(reason="need -m tensorrtllm to run")
    for item in items:
        if "tensorrtllm" in item.keywords:
            item.add_marker(skip_tensorrtllm)


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
            data_dir=tempfile.mkdtemp(prefix="etcd_"),
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
