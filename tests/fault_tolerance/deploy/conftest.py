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

import pytest


# Shared CLI options (--image, --namespace, --skip-service-restart) are defined in tests/conftest.py.
# Only fault_tolerance-specific options are defined here.
def pytest_addoption(parser):
    parser.addoption(
        "--client-type",
        type=str,
        default=None,
        choices=["aiperf", "legacy"],
        help="Client type for load generation: 'aiperf' (default) or 'legacy'",
    )
    parser.addoption(
        "--storage-class",
        type=str,
        default=None,
        help="Storage class for PVC log collection (must support RWX). "
        "If not specified, uses cluster default.",
    )


@pytest.fixture
def image(request):
    return request.config.getoption("--image")


@pytest.fixture
def namespace(request):
    """Get Kubernetes namespace from CLI option, with fault-tolerance-specific default."""
    value = request.config.getoption("--namespace")
    return value if value is not None else "fault-tolerance-test"


@pytest.fixture
def client_type(request):
    """Get client type from command line or use scenario default."""
    return request.config.getoption("--client-type")


@pytest.fixture
def skip_service_restart(request):
    """Whether to skip restarting NATS and etcd services.

    Fault tolerance tests default to RESTARTING services (for clean state).
    The --skip-service-restart flag can override this behavior.

    Returns:
        If --skip-service-restart is passed: True (skip restart)
        If flag not passed: False (FT tests restart by default)
    """
    value = request.config.getoption("--skip-service-restart")
    return value if value is not None else False  # Default: restart for FT tests


@pytest.fixture
def storage_class(request):
    """Get storage class for PVC log collection from command line."""
    return request.config.getoption("--storage-class")
