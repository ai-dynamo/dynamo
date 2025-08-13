# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for port allocation and management utilities."""

import json
import socket
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from dynamo.vllm.ports import (
    DynamoPortRange,
    EtcdContext,
    PortAllocationRequest,
    PortMetadata,
    allocate_and_reserve_port,
    allocate_and_reserve_port_block,
    check_port_available,
    get_host_ip,
    hold_ports,
    reserve_port_in_etcd,
)


class TestDynamoPortRange:
    """Test DynamoPortRange validation."""

    def test_valid_port_range(self):
        """Test creating a valid port range."""
        port_range = DynamoPortRange(min=2000, max=3000)
        assert port_range.min == 2000
        assert port_range.max == 3000

    def test_port_range_outside_registered_range(self):
        """Test that port ranges outside 1024-49151 are rejected."""
        with pytest.raises(ValueError, match="outside registered ports range"):
            DynamoPortRange(min=500, max=2000)

        with pytest.raises(ValueError, match="outside registered ports range"):
            DynamoPortRange(min=2000, max=50000)

    def test_invalid_port_range_min_greater_than_max(self):
        """Test that min >= max is rejected."""
        with pytest.raises(ValueError, match="min .* must be less than max"):
            DynamoPortRange(min=3000, max=2000)

        with pytest.raises(ValueError, match="min .* must be less than max"):
            DynamoPortRange(min=3000, max=3000)


class TestPortMetadata:
    """Test port metadata functionality."""

    def test_to_etcd_value_with_block_info(self):
        """Test converting metadata to ETCD value with block info."""
        metadata = PortMetadata(
            worker_id="test-worker",
            reason="test-reason",
            block_info={"block_index": 0, "block_size": 4, "block_start": 8080},
        )

        value = metadata.to_etcd_value()
        assert value["block_index"] == 0
        assert value["block_size"] == 4
        assert value["block_start"] == 8080


class TestHoldPorts:
    """Test hold_ports context manager."""

    def test_hold_single_port(self):
        """Test holding a single port."""
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        with hold_ports(port):
            assert not check_port_available(port)

        # Port should be released after context exit
        assert check_port_available(port)

    def test_hold_multiple_ports(self):
        """Test holding multiple ports."""
        ports = []
        for _ in range(2):
            with socket.socket() as s:
                s.bind(("", 0))
                ports.append(s.getsockname()[1])

        with hold_ports(ports):
            for port in ports:
                assert not check_port_available(port)

        # All ports should be released after context exit
        for port in ports:
            assert check_port_available(port)


class TestReservePortInEtcd:
    """Test ETCD port reservation."""

    @pytest.mark.asyncio
    async def test_reserve_port_success(self):
        """Test successful port reservation in ETCD."""
        mock_client = AsyncMock()
        mock_client.primary_lease_id = Mock(return_value="test-lease-123")

        context = EtcdContext(client=mock_client, namespace="test-ns")
        metadata = PortMetadata(worker_id="test-worker", reason="test")

        host_ip = get_host_ip()
        await reserve_port_in_etcd(context, 8080, metadata)

        mock_client.kv_create.assert_called_once()
        call_args = mock_client.kv_create.call_args

        assert call_args.kwargs["key"] == f"dyn://test-ns/ports/{host_ip}/8080"
        assert call_args.kwargs["lease_id"] == "test-lease-123"

        # Check the value is valid JSON
        value_bytes = call_args.kwargs["value"]
        value_dict = json.loads(value_bytes.decode())
        assert value_dict["worker_id"] == "test-worker"
        assert value_dict["reason"] == "test"


class TestAllocateAndReservePort:
    """Test single port allocation."""

    @pytest.mark.asyncio
    async def test_allocate_single_port_success(self):
        """Test successful single port allocation."""
        mock_client = AsyncMock()
        mock_client.primary_lease_id = Mock(return_value="test-lease")

        context = EtcdContext(client=mock_client, namespace="test-ns")
        metadata = PortMetadata(worker_id="test-worker", reason="test")
        port_range = DynamoPortRange(min=20000, max=20010)

        # Mock that all ports are available
        with patch("dynamo.vllm.ports.check_port_available", return_value=True):
            with patch("dynamo.vllm.ports.hold_ports") as mock_hold:
                # Set up the context manager mock
                mock_hold.return_value.__enter__ = Mock()
                mock_hold.return_value.__exit__ = Mock(return_value=None)

                port = await allocate_and_reserve_port(
                    context, metadata, port_range, max_attempts=5
                )

        assert 20000 <= port <= 20010
        mock_client.kv_create.assert_called_once()


class TestAllocateAndReservePortBlock:
    """Test port block allocation."""

    @pytest.mark.asyncio
    async def test_allocate_block_success(self):
        """Test successful port block allocation."""
        mock_client = AsyncMock()
        mock_client.primary_lease_id = Mock(return_value="test-lease")

        context = EtcdContext(client=mock_client, namespace="test-ns")
        metadata = PortMetadata(worker_id="test-worker", reason="test")
        port_range = DynamoPortRange(min=20000, max=20010)

        request = PortAllocationRequest(
            etcd_context=context,
            metadata=metadata,
            port_range=port_range,
            block_size=3,
            max_attempts=5,
        )

        with patch("dynamo.vllm.ports.hold_ports") as mock_hold:
            # Set up the context manager mock
            mock_hold.return_value.__enter__ = Mock()
            mock_hold.return_value.__exit__ = Mock(return_value=None)

            ports = await allocate_and_reserve_port_block(request)

        assert len(ports) == 3
        assert all(20000 <= p <= 20010 for p in ports)
        assert ports == list(range(ports[0], ports[0] + 3))

        # Should have reserved 3 ports
        assert mock_client.kv_create.call_count == 3

    @pytest.mark.asyncio
    async def test_allocate_block_port_range_too_small(self):
        """Test error when port range is too small for block."""
        context = EtcdContext(client=Mock(), namespace="test-ns")
        metadata = PortMetadata(worker_id="test-worker", reason="test")
        port_range = DynamoPortRange(min=20000, max=20002)

        request = PortAllocationRequest(
            etcd_context=context,
            metadata=metadata,
            port_range=port_range,
            block_size=5,  # Needs 5 ports but range only has 3
        )

        with pytest.raises(
            ValueError, match="Port range .* is too small for block size"
        ):
            await allocate_and_reserve_port_block(request)


class TestGetHostIP:
    """Test get_host_ip function."""

    def test_get_host_ip_success(self):
        """Test successful host IP retrieval."""
        with patch("socket.gethostname", return_value="test-host"):
            with patch("socket.gethostbyname", return_value="192.168.1.100"):
                with patch("socket.socket") as mock_socket_class:
                    # Mock successful bind
                    mock_socket = MagicMock()
                    mock_socket_class.return_value.__enter__.return_value = mock_socket

                    ip = get_host_ip()
                    assert ip == "192.168.1.100"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
