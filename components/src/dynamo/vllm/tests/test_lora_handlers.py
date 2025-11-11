# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LoRA handler functionality in vLLM workers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.llm import lora_name_to_hash_id


class TestLoraHandlers:
    """Test suite for LoRA handler methods."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler with necessary attributes."""
        handler = MagicMock()
        handler.lora_name_to_id = {}
        handler.lora_name_to_path = {}
        handler.engine_client = AsyncMock()
        handler.config = MagicMock()
        handler.config.model = "base_model"
        handler.generate_endpoint = MagicMock()
        handler.generate_endpoint.inner = MagicMock()
        return handler

    @pytest.mark.asyncio
    async def test_load_lora_success(self, mock_handler):
        """Test successful LoRA loading."""
        from dynamo.vllm.handlers import BaseWorkerHandler

        # Import the load_lora method
        request = {"lora_name": "test_lora", "lora_path": "/path/to/lora"}

        # Calculate expected ID
        expected_id = lora_name_to_hash_id("test_lora")

        # Create a generator
        async def run_load_lora():
            async for result in BaseWorkerHandler.load_lora(mock_handler, request):
                return result

        result = await run_load_lora()

        # Verify the result
        assert result["status"] == "success"
        assert result["lora_name"] == "test_lora"
        assert result["lora_path"] == "/path/to/lora"
        assert result["lora_id"] == expected_id

        # Verify engine_client.add_lora was called
        mock_handler.engine_client.add_lora.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_lora_missing_parameters(self, mock_handler):
        """Test LoRA loading with missing parameters."""
        from dynamo.vllm.handlers import BaseWorkerHandler

        # Test missing lora_name
        request = {"lora_path": "/path/to/lora"}

        async def run_load_lora():
            async for result in BaseWorkerHandler.load_lora(mock_handler, request):
                return result

        result = await run_load_lora()
        assert result["status"] == "error"
        assert "required" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_unload_lora_success(self, mock_handler):
        """Test successful LoRA unloading."""
        from dynamo.vllm.handlers import BaseWorkerHandler

        # Pre-populate the handler with a loaded LoRA
        lora_name = "test_lora"
        lora_id = lora_name_to_hash_id(lora_name)
        mock_handler.lora_name_to_id[lora_name] = lora_id
        mock_handler.lora_name_to_path[lora_name] = "/path/to/lora"

        request = {"lora_name": lora_name}

        async def run_unload_lora():
            async for result in BaseWorkerHandler.unload_lora(mock_handler, request):
                return result

        result = await run_unload_lora()

        # Verify the result
        assert result["status"] == "success"
        assert result["lora_name"] == lora_name
        assert result["lora_id"] == lora_id

        # Verify engine_client.remove_lora was called
        mock_handler.engine_client.remove_lora.assert_called_once_with(lora_id)

    @pytest.mark.asyncio
    async def test_unload_lora_not_found(self, mock_handler):
        """Test unloading a LoRA that doesn't exist."""
        from dynamo.vllm.handlers import BaseWorkerHandler

        request = {"lora_name": "nonexistent_lora"}

        async def run_unload_lora():
            async for result in BaseWorkerHandler.unload_lora(mock_handler, request):
                return result

        result = await run_unload_lora()
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_list_loras(self, mock_handler):
        """Test listing loaded LoRAs."""
        from dynamo.vllm.handlers import BaseWorkerHandler

        # Pre-populate with some LoRAs
        mock_handler.lora_name_to_id = {
            "lora1": lora_name_to_hash_id("lora1"),
            "lora2": lora_name_to_hash_id("lora2"),
        }

        async def run_list_loras():
            async for result in BaseWorkerHandler.list_loras(mock_handler, None):
                return result

        result = await run_list_loras()

        # Verify the result
        assert result["status"] == "success"
        assert result["count"] == 2
        assert "lora1" in result["loras"]
        assert "lora2" in result["loras"]
        assert result["loras"]["lora1"] == lora_name_to_hash_id("lora1")
        assert result["loras"]["lora2"] == lora_name_to_hash_id("lora2")

    def test_lora_id_consistency(self):
        """Test that LoRA IDs are consistent across multiple calls."""
        lora_names = ["adapter1", "adapter2", "adapter3"]

        # Generate IDs multiple times
        ids_first = [lora_name_to_hash_id(name) for name in lora_names]
        ids_second = [lora_name_to_hash_id(name) for name in lora_names]

        # Should be identical
        assert ids_first == ids_second

    def test_lora_id_uniqueness(self):
        """Test that different LoRA names produce different IDs."""
        lora_names = [f"lora_{i}" for i in range(100)]
        ids = [lora_name_to_hash_id(name) for name in lora_names]

        # All IDs should be unique
        assert len(ids) == len(set(ids))

    def test_lora_id_range(self):
        """Test that LoRA IDs are in the valid range."""
        max_int32 = 0x7FFFFFFF
        lora_names = [f"test_lora_{i}" for i in range(50)]

        for name in lora_names:
            lora_id = lora_name_to_hash_id(name)
            assert 1 <= lora_id <= max_int32, f"ID {lora_id} out of range for {name}"


class TestLoraIntegration:
    """Integration tests for LoRA functionality."""

    @pytest.mark.asyncio
    async def test_load_multiple_loras(self, mock_handler):
        """Test loading multiple LoRAs sequentially."""
        from dynamo.vllm.handlers import BaseWorkerHandler

        loras = [
            {"lora_name": "lora1", "lora_path": "/path/to/lora1"},
            {"lora_name": "lora2", "lora_path": "/path/to/lora2"},
            {"lora_name": "lora3", "lora_path": "/path/to/lora3"},
        ]

        for request in loras:
            async for result in BaseWorkerHandler.load_lora(mock_handler, request):
                assert result["status"] == "success"
                assert result["lora_name"] == request["lora_name"]

        # Verify all LoRAs are tracked
        assert len(mock_handler.lora_name_to_id) == 3
        assert "lora1" in mock_handler.lora_name_to_id
        assert "lora2" in mock_handler.lora_name_to_id
        assert "lora3" in mock_handler.lora_name_to_id

    @pytest.mark.asyncio
    async def test_load_unload_load_cycle(self, mock_handler):
        """Test loading, unloading, and reloading a LoRA."""
        from dynamo.vllm.handlers import BaseWorkerHandler

        lora_name = "cycle_lora"
        request_load = {"lora_name": lora_name, "lora_path": "/path/to/lora"}
        request_unload = {"lora_name": lora_name}

        # Load
        async for result in BaseWorkerHandler.load_lora(mock_handler, request_load):
            assert result["status"] == "success"
            first_id = result["lora_id"]

        # Unload
        async for result in BaseWorkerHandler.unload_lora(mock_handler, request_unload):
            assert result["status"] == "success"

        # Reload
        async for result in BaseWorkerHandler.load_lora(mock_handler, request_load):
            assert result["status"] == "success"
            second_id = result["lora_id"]

        # IDs should be the same (deterministic hashing)
        assert first_id == second_id
