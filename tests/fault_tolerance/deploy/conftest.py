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

import pytest


def pytest_addoption(parser):
    parser.addoption("--image", type=str, default=None)
    parser.addoption("--namespace", type=str, default="fault-tolerance-test")
    parser.addoption(
        "--client-type",
        type=str,
        default=None,
        choices=["aiperf", "legacy"],
        help="Client type for load generation: 'aiperf' (default) or 'legacy'",
    )


@pytest.fixture
def image(request):
    return request.config.getoption("--image")


@pytest.fixture
def namespace(request):
    return request.config.getoption("--namespace")


@pytest.fixture
def client_type(request):
    """Get client type from command line or use scenario default."""
    return request.config.getoption("--client-type")
