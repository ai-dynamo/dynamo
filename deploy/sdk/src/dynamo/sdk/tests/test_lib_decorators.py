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

from dynamo.sdk.lib.decorators import DynamoEndpoint, endpoint


def test_dynamo_endpoint_http_method():
    """Test DynamoEndpoint with HTTP method."""

    async def test_func(request: dict) -> dict:
        return {}

    # Test with explicit HTTP method
    endpoint = DynamoEndpoint(test_func, http_method="GET")
    assert endpoint.http_method == "GET"

    # Test with no HTTP method (should be None)
    endpoint = DynamoEndpoint(test_func)
    assert endpoint.http_method is None

    # Test method normalization to uppercase
    endpoint = DynamoEndpoint(test_func, http_method="post")
    assert endpoint.http_method == "POST"


def test_endpoint_decorator_validation():
    """Test endpoint decorator validation of HTTP methods."""
    # Should raise ValueError when http_method is specified but is_api=False
    with pytest.raises(
        ValueError, match="http_method can only be specified for API endpoints"
    ):

        @endpoint(http_method="GET", is_api=False)
        async def invalid_endpoint(request: dict) -> dict:
            return {}

    # Should work when is_api=True
    @endpoint(http_method="GET", is_api=True)
    async def valid_endpoint(request: dict) -> dict:
        return {}

    assert valid_endpoint.http_method == "GET"


def test_endpoint_decorator_defaults():
    """Test endpoint decorator default behavior."""

    # Test default behavior (no HTTP method)
    @endpoint()
    async def default_endpoint(request: dict) -> dict:
        return {}

    assert default_endpoint.http_method is None

    # Test with is_api=True but no http_method
    @endpoint(is_api=True)
    async def api_endpoint(request: dict) -> dict:
        return {}

    assert api_endpoint.http_method is None


def test_endpoint_decorator_http_method_propagation():
    """Test that HTTP method is properly propagated through the decorator."""

    # Test with explicit HTTP method
    @endpoint(is_api=True, http_method="PUT")
    async def put_endpoint(request: dict) -> dict:
        return {}

    assert put_endpoint.http_method == "PUT"

    # Test method normalization
    @endpoint(is_api=True, http_method="delete")
    async def delete_endpoint(request: dict) -> dict:
        return {}

    assert delete_endpoint.http_method == "DELETE"


def test_endpoint_decorator_transport():
    """Test that transport is set correctly based on is_api."""
    from dynamo.sdk.core.protocol.interface import DynamoTransport

    # Test with is_api=True
    @endpoint(is_api=True)
    async def api_endpoint(request: dict) -> dict:
        return {}

    assert api_endpoint._transports == [DynamoTransport.HTTP]

    # Test with is_api=False
    @endpoint(is_api=False)
    async def non_api_endpoint(request: dict) -> dict:
        return {}

    assert non_api_endpoint._transports == [DynamoTransport.DEFAULT]

    # Test with explicit HTTP transport via is_api
    @endpoint(is_api=True)
    async def custom_transport(request: dict) -> dict:
        return {}

    assert custom_transport._transports == [DynamoTransport.HTTP]
