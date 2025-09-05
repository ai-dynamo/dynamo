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
    parser.addoption("--requests-per-client", type=int, default=100)
    parser.addoption("--clients", type=int, default=10)
    parser.addoption("--input-token-length", type=int, default=100)
    parser.addoption("--output-token-length", type=int, default=100)
    parser.addoption("--max-retries", type=int, default=1)
    parser.addoption("--model", type=str, default=None)
    parser.addoption("--max-request-rate", type=int, default=1)
    parser.addoption("--sla", type=float, default=None)


@pytest.fixture(scope="session")
def sla(request):
    return request.config.getoption("--sla")


@pytest.fixture
def max_request_rate(request):
    return request.config.getoption("--max-request-rate")


@pytest.fixture
def image(request):
    return request.config.getoption("--image")


@pytest.fixture
def model(request):
    return request.config.getoption("--model")


@pytest.fixture
def namespace(request):
    return request.config.getoption("--namespace")


@pytest.fixture
def max_retries(request):
    return request.config.getoption("--max-retries")


@pytest.fixture
def num_clients(request):
    return request.config.getoption("--clients")


@pytest.fixture
def input_token_length(request):
    return request.config.getoption("--input-token-length")


@pytest.fixture
def output_token_length(request):
    return request.config.getoption("--output-token-length")


@pytest.fixture
def requests_per_client(request):
    return request.config.getoption("--requests-per-client")
