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

from typing import Optional
from unittest.mock import patch

import pytest

from dynamo.sdk.request_tracing import (
    RequestTracingMixin,
    clear_request_context,
    get_current_request_id,
    set_request_context,
    with_request_id,
)

pytestmark = pytest.mark.pre_merge


@pytest.fixture(autouse=True)
def clear_context():
    """Clear request context before and after each test."""
    clear_request_context()
    yield
    clear_request_context()


class MockProcessor(RequestTracingMixin):
    """Mock processor class for testing the with_request_id decorator."""

    def __init__(self):
        self.received_args = None
        self.received_kwargs = None
        self.received_request_id = None

    @with_request_id()
    async def process_first_param(
        self, request_id: Optional[str] = None, data: str = "test"
    ):
        """Method with request_id as the first parameter after self."""
        self.received_args = (request_id, data)
        self.received_kwargs = {"request_id": request_id, "data": data}
        self.received_request_id = request_id
        return f"processed_{data}"

    @with_request_id()
    async def process_middle_param(
        self, data1: str, request_id: Optional[str] = None, data2: str = "test2"
    ):
        """Method with request_id in the middle of parameters."""
        self.received_args = (data1, request_id, data2)
        self.received_kwargs = {
            "data1": data1,
            "request_id": request_id,
            "data2": data2,
        }
        self.received_request_id = request_id
        return f"processed_{data1}_{data2}"

    @with_request_id()
    async def process_last_param(
        self, data1: str, data2: str, request_id: Optional[str] = None
    ):
        """Method with request_id as the last parameter."""
        self.received_args = (data1, data2, request_id)
        self.received_kwargs = {
            "data1": data1,
            "data2": data2,
            "request_id": request_id,
        }
        self.received_request_id = request_id
        return f"processed_{data1}_{data2}"

    @with_request_id()
    async def process_no_request_id_param(self, data: str = "test"):
        """Method without a request_id parameter to test auto-generation."""
        self.received_args = (data,)
        self.received_kwargs = {"data": data}
        self.received_request_id = get_current_request_id()
        return f"processed_{data}"

    @with_request_id("req_id")
    async def process_custom_param_name(
        self, req_id: Optional[str] = None, data: str = "test"
    ):
        """Method with custom parameter name for request ID."""
        self.received_args = (req_id, data)
        self.received_kwargs = {"req_id": req_id, "data": data}
        self.received_request_id = req_id
        return f"processed_{data}"


class TestWithRequestIdDecorator:
    """Test cases for the with_request_id decorator's positional argument handling."""

    async def test_request_id_first_param_positional(self):
        """Test passing request_id as first positional argument."""
        processor = MockProcessor()
        test_request_id = "test-request-123"

        result = await processor.process_first_param(test_request_id, "data1")

        # Verify the request_id was correctly handled
        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1"

    async def test_request_id_first_param_keyword(self):
        """Test passing request_id as keyword argument when it's the first param."""
        processor = MockProcessor()
        test_request_id = "test-request-456"

        result = await processor.process_first_param(
            request_id=test_request_id, data="data1"
        )

        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1"

    async def test_request_id_first_param_omitted(self):
        """Test omitting request_id when it's the first param - should auto-generate."""
        processor = MockProcessor()

        result = await processor.process_first_param(data="data1")

        # Should have auto-generated a request ID
        assert processor.received_request_id is not None
        assert isinstance(processor.received_request_id, str)
        assert len(processor.received_request_id) == 36  # UUID length
        assert get_current_request_id() == processor.received_request_id
        assert result == "processed_data1"

    async def test_request_id_middle_param_positional(self):
        """Test passing request_id as positional argument in the middle."""
        processor = MockProcessor()
        test_request_id = "test-request-middle"

        result = await processor.process_middle_param("data1", test_request_id, "data2")

        # Verify the positional argument was correctly handled and args reconstructed
        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1_data2"

    async def test_request_id_middle_param_keyword(self):
        """Test passing request_id as keyword argument when it's in the middle."""
        processor = MockProcessor()
        test_request_id = "test-request-middle-kw"

        result = await processor.process_middle_param(
            "data1", request_id=test_request_id, data2="data2"
        )

        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1_data2"

    async def test_request_id_middle_param_omitted(self):
        """Test omitting request_id when it's in the middle - should auto-generate."""
        processor = MockProcessor()

        result = await processor.process_middle_param("data1", data2="data2")

        assert processor.received_request_id is not None
        assert isinstance(processor.received_request_id, str)
        assert len(processor.received_request_id) == 36  # UUID length
        assert get_current_request_id() == processor.received_request_id
        assert result == "processed_data1_data2"

    async def test_request_id_last_param_positional(self):
        """Test passing request_id as the last positional argument."""
        processor = MockProcessor()
        test_request_id = "test-request-last"

        result = await processor.process_last_param("data1", "data2", test_request_id)

        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1_data2"

    async def test_request_id_last_param_keyword(self):
        """Test passing request_id as keyword argument when it's the last param."""
        processor = MockProcessor()
        test_request_id = "test-request-last-kw"

        result = await processor.process_last_param(
            "data1", "data2", request_id=test_request_id
        )

        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1_data2"

    async def test_request_id_last_param_omitted(self):
        """Test omitting request_id when it's the last param - should auto-generate."""
        processor = MockProcessor()

        result = await processor.process_last_param("data1", "data2")

        assert processor.received_request_id is not None
        assert isinstance(processor.received_request_id, str)
        assert len(processor.received_request_id) == 36  # UUID length
        assert get_current_request_id() == processor.received_request_id
        assert result == "processed_data1_data2"

    async def test_no_request_id_param_auto_generation(self):
        """Test auto-generation when method has no request_id parameter."""
        processor = MockProcessor()

        result = await processor.process_no_request_id_param("data1")

        # Should have auto-generated and set in context
        assert processor.received_request_id is not None
        assert isinstance(processor.received_request_id, str)
        assert len(processor.received_request_id) == 36  # UUID length
        assert get_current_request_id() == processor.received_request_id
        assert result == "processed_data1"

    async def test_custom_param_name(self):
        """Test decorator with custom parameter name."""
        processor = MockProcessor()
        test_request_id = "test-custom-param"

        result = await processor.process_custom_param_name(test_request_id, "data1")

        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1"

    async def test_custom_param_name_keyword(self):
        """Test decorator with custom parameter name using keyword argument."""
        processor = MockProcessor()
        test_request_id = "test-custom-param-kw"

        result = await processor.process_custom_param_name(
            req_id=test_request_id, data="data1"
        )

        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id
        assert result == "processed_data1"

    async def test_args_tuple_reconstruction(self):
        """Test that args tuple is correctly reconstructed when request_id is positional."""
        processor = MockProcessor()
        test_request_id = "test-reconstruction"

        # Use a method where request_id is in the middle to test tuple reconstruction
        await processor.process_middle_param("arg1", test_request_id, "arg3")

        # The args should have been reconstructed with None in place of the request_id
        # since it gets moved to kwargs
        assert processor.received_request_id == test_request_id
        assert get_current_request_id() == test_request_id

    async def test_adj_idx_calculation(self):
        """Test the adj_idx = param_idx - 1 calculation for skipping self."""
        processor = MockProcessor()

        # This test verifies that the decorator correctly accounts for 'self'
        # when calculating the adjusted index for positional arguments

        # Test with request_id as first param (param_idx=1, adj_idx=0)
        await processor.process_first_param("test-id-1")
        assert processor.received_request_id == "test-id-1"

        # Test with request_id as middle param (param_idx=2, adj_idx=1)
        await processor.process_middle_param("data1", "test-id-2", "data2")
        assert processor.received_request_id == "test-id-2"

        # Test with request_id as last param (param_idx=3, adj_idx=2)
        await processor.process_last_param("data1", "data2", "test-id-3")
        assert processor.received_request_id == "test-id-3"

    async def test_existing_context_preserved(self):
        """Test that existing request context is used when no request_id is provided."""
        existing_request_id = "existing-context-id"
        set_request_context(existing_request_id)

        processor = MockProcessor()

        # Don't pass request_id - should use existing context
        await processor.process_first_param(data="test_data")

        # Should use the existing context request_id
        assert processor.received_request_id == existing_request_id
        assert get_current_request_id() == existing_request_id

    async def test_explicit_none_triggers_generation(self):
        """Test that explicitly passing None triggers request_id generation."""
        processor = MockProcessor()

        await processor.process_first_param(None, "test_data")

        # Should have generated a new request_id even though None was explicitly passed
        assert processor.received_request_id is not None
        assert processor.received_request_id != "None"
        assert isinstance(processor.received_request_id, str)
        assert len(processor.received_request_id) == 36  # UUID length

    async def test_multiple_calls_context_isolation(self):
        """Test that multiple calls maintain proper context isolation."""
        processor = MockProcessor()

        # First call
        await processor.process_first_param("request-1")
        first_id = processor.received_request_id

        # Second call
        await processor.process_first_param("request-2")
        second_id = processor.received_request_id

        # Third call with auto-generation
        await processor.process_first_param()
        third_id = processor.received_request_id

        # All should be different
        assert first_id == "request-1"
        assert second_id == "request-2"
        assert third_id != first_id
        assert third_id != second_id
        assert isinstance(third_id, str)
        assert len(third_id) == 36  # UUID length

    @patch("dynamo.sdk.request_tracing.uuid.uuid4")
    async def test_uuid_generation_called(self, mock_uuid):
        """Test that uuid.uuid4() is called when generating request IDs."""
        mock_uuid.return_value.side_effect = lambda: type(
            "MockUUID", (), {"__str__": lambda self: "mocked-uuid-1234"}
        )()
        mock_uuid.return_value.__str__ = lambda: "mocked-uuid-1234"

        processor = MockProcessor()

        await processor.process_first_param(data="test")

        # Verify UUID generation was called
        mock_uuid.assert_called()
        # Note: The actual implementation calls str(uuid.uuid4()), so we verify the behavior
        assert processor.received_request_id is not None

    async def test_edge_case_empty_args(self):
        """Test edge case with minimal arguments."""
        processor = MockProcessor()

        # Call method with only keyword arguments
        result = await processor.process_first_param(data="minimal")

        assert processor.received_request_id is not None
        assert result == "processed_minimal"
        assert get_current_request_id() == processor.received_request_id


if __name__ == "__main__":
    pytest.main([__file__])
