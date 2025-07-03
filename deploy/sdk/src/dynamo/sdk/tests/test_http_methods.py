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

from dynamo.sdk import api, endpoint
from dynamo.sdk.core.decorators.endpoint import DynamoEndpoint
from dynamo.sdk.core.protocol.interface import DynamoTransport
from dynamo.sdk.core.runner.dynamo import LocalEndpoint


def test_dynamo_endpoint_http_method():
    """Test DynamoEndpoint with HTTP method."""
    # Test with explicit HTTP method
    endpoint = DynamoEndpoint(
        lambda: None, transports=[DynamoTransport.HTTP], http_method="GET"
    )
    assert endpoint.http_method == "GET"

    # Test with no HTTP method (should be None)
    endpoint = DynamoEndpoint(lambda: None)
    assert endpoint.http_method is None

    # Test method normalization to uppercase
    endpoint = DynamoEndpoint(
        lambda: None, transports=[DynamoTransport.HTTP], http_method="post"
    )
    assert endpoint.http_method == "POST"

    # Test mixed case methods
    endpoint = DynamoEndpoint(
        lambda: None, transports=[DynamoTransport.HTTP], http_method="DeLeTe"
    )
    assert endpoint.http_method == "DELETE"


def test_transport_http_method_interaction():
    """Test interaction between transports and HTTP methods."""
    # Test HTTP method with non-HTTP transport
    with pytest.raises(ValueError):
        DynamoEndpoint(
            lambda: None, http_method="GET", transports=[DynamoTransport.DEFAULT]
        )

    # Test multiple transports with HTTP method
    endpoint = DynamoEndpoint(
        lambda: None,
        http_method="GET",
        transports=[DynamoTransport.HTTP, DynamoTransport.DEFAULT],
    )
    assert endpoint.http_method == "GET"


def test_endpoint_decorator_validation():
    """Test endpoint decorator validation of HTTP methods."""
    # Should raise ValueError when http_method is specified without HTTP transport
    with pytest.raises(
        ValueError,
        match="http_method can only be specified for endpoints with HTTP transport",
    ):

        @endpoint(http_method="GET")
        async def invalid_endpoint():
            pass

    # Should work when HTTP transport is specified
    @endpoint(http_method="GET", transports=[DynamoTransport.HTTP])
    async def valid_endpoint():
        pass

    assert valid_endpoint.http_method == "GET"


def test_api_decorator():
    """Test api decorator with HTTP methods."""

    # Test default behavior (None -> POST)
    @api()
    async def default_endpoint():
        pass

    assert default_endpoint.http_method is None

    # Test with explicit HTTP method
    @api(http_method="GET")
    async def get_endpoint():
        pass

    assert get_endpoint.http_method == "GET"


def test_local_endpoint():
    """Test LocalEndpoint HTTP method handling."""
    # Test with HTTP method
    local_endpoint = LocalEndpoint(
        "test", None, [DynamoTransport.HTTP], http_method="GET"
    )
    assert local_endpoint.http_method == "GET"

    # Test without HTTP method
    local_endpoint = LocalEndpoint("test", None, [DynamoTransport.HTTP])
    assert local_endpoint.http_method is None
