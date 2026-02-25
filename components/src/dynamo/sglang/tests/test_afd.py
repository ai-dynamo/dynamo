# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for AFD (Attention-FFN Disaggregation) implementation.

Tests cover:
- DisaggregationMode enum extension
- Configuration parsing
- Communication protocol (serialization/deserialization)
- Metrics collection
- Performance analyzer
"""

import asyncio
import json
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import asdict

# Import AFD modules
from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.afd_communication import (
    AFDActivationBatch,
    AFDFFNResult,
    AFDCommunicationManager,
    AFDMicrobatchPipeline,
    AFDMessageType,
)
from dynamo.sglang.afd_metrics import (
    AFDWorkerMetrics,
    AFDMetricsCollector,
    AFDPerformanceAnalyzer,
)


class TestDisaggregationMode:
    """Test DisaggregationMode enum extension for AFD."""

    def test_attention_mode_exists(self):
        """Test ATTENTION mode is available."""
        assert hasattr(DisaggregationMode, 'ATTENTION')
        assert DisaggregationMode.ATTENTION.value == "attention"

    def test_ffn_mode_exists(self):
        """Test FFN mode is available."""
        assert hasattr(DisaggregationMode, 'FFN')
        assert DisaggregationMode.FFN.value == "ffn"

    def test_all_modes(self):
        """Test all disaggregation modes are available."""
        expected_modes = ['AGGREGATED', 'PREFILL', 'DECODE', 'ATTENTION', 'FFN']
        actual_modes = [m.name for m in DisaggregationMode]
        for mode in expected_modes:
            assert mode in actual_modes, f"Missing mode: {mode}"


class TestAFDActivationBatch:
    """Test AFD activation batch serialization/deserialization."""

    def test_create_activation_batch(self):
        """Test creating an activation batch."""
        batch = AFDActivationBatch(
            request_id="test-request-123",
            layer_idx=5,
            activations=np.random.randn(2, 128, 4096).astype(np.float32),
            metadata={"temperature": 0.7, "top_p": 0.9},
        )
        
        assert batch.request_id == "test-request-123"
        assert batch.layer_idx == 5
        assert batch.activations.shape == (2, 128, 4096)
        assert batch.metadata["temperature"] == 0.7

    def test_serialize_deserialize_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original_batch = AFDActivationBatch(
            request_id="roundtrip-test",
            layer_idx=10,
            activations=np.random.randn(1, 64, 512).astype(np.float32),
            attention_mask=np.ones((1, 64), dtype=np.int64),
            position_ids=np.arange(64).reshape(1, 64).astype(np.int32),
            metadata={"key": "value", "number": 42},
        )
        
        # Serialize
        serialized = original_batch.serialize()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize
        deserialized = AFDActivationBatch.deserialize(serialized)
        
        # Verify
        assert deserialized.request_id == original_batch.request_id
        assert deserialized.layer_idx == original_batch.layer_idx
        np.testing.assert_array_almost_equal(
            deserialized.activations, original_batch.activations
        )
        np.testing.assert_array_equal(
            deserialized.attention_mask, original_batch.attention_mask
        )
        np.testing.assert_array_equal(
            deserialized.position_ids, original_batch.position_ids
        )
        assert deserialized.metadata == original_batch.metadata

    def test_serialize_without_optional_fields(self):
        """Test serialization without optional fields."""
        batch = AFDActivationBatch(
            request_id="minimal",
            layer_idx=0,
            activations=np.zeros((1, 1, 256), dtype=np.float32),
        )
        
        serialized = batch.serialize()
        deserialized = AFDActivationBatch.deserialize(serialized)
        
        assert deserialized.attention_mask is None
        assert deserialized.position_ids is None
        assert deserialized.metadata == {}


class TestAFDFFNResult:
    """Test AFD FFN result serialization/deserialization."""

    def test_create_ffn_result(self):
        """Test creating an FFN result."""
        result = AFDFFNResult(
            request_id="result-123",
            output=np.random.randn(2, 128, 4096).astype(np.float32),
            finish_reason="stop",
        )
        
        assert result.request_id == "result-123"
        assert result.output.shape == (2, 128, 4096)
        assert result.finish_reason == "stop"

    def test_serialize_deserialize_result(self):
        """Test FFN result roundtrip."""
        original = AFDFFNResult(
            request_id="result-roundtrip",
            output=np.random.randn(1, 32, 1024).astype(np.float32),
            finish_reason="length",
        )
        
        serialized = original.serialize()
        deserialized = AFDFFNResult.deserialize(serialized)
        
        assert deserialized.request_id == original.request_id
        np.testing.assert_array_almost_equal(deserialized.output, original.output)
        assert deserialized.finish_reason == original.finish_reason

    def test_result_without_finish_reason(self):
        """Test result without finish reason."""
        result = AFDFFNResult(
            request_id="no-finish",
            output=np.zeros((1, 1, 512), dtype=np.float32),
        )
        
        serialized = result.serialize()
        deserialized = AFDFFNResult.deserialize(serialized)
        
        assert deserialized.finish_reason is None


class TestAFDCommunicationManager:
    """Test AFD communication manager."""

    @pytest.mark.asyncio
    async def test_create_manager(self):
        """Test creating communication manager."""
        manager = AFDCommunicationManager(
            ffn_endpoint="dynamo.ffn.generate",
            attention_ratio=8,
            microbatch_size=256,
            sync_timeout_ms=1000,
        )
        
        assert manager.ffn_endpoint == "dynamo.ffn.generate"
        assert manager.attention_ratio == 8
        assert manager.microbatch_size == 256

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        manager = AFDCommunicationManager(ffn_endpoint="test.endpoint")
        
        await manager.connect()
        assert manager._running is True
        
        await manager.disconnect()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_send_activation_without_connection(self):
        """Test sending activation without connection raises error."""
        manager = AFDCommunicationManager()
        
        batch = AFDActivationBatch(
            request_id="test",
            layer_idx=0,
            activations=np.zeros((1, 1, 256), dtype=np.float32),
        )
        
        with pytest.raises(RuntimeError, match="not connected"):
            await manager.send_activation_batch(batch)


class TestAFDMicrobatchPipeline:
    """Test AFD microbatch pipeline."""

    @pytest.mark.asyncio
    async def test_create_pipeline(self):
        """Test creating microbatch pipeline."""
        comm_manager = AFDCommunicationManager()
        pipeline = AFDMicrobatchPipeline(
            communication_manager=comm_manager,
            num_stages=4,
            batch_size=256,
        )
        
        assert pipeline.num_stages == 4
        assert pipeline.batch_size == 256

    @pytest.mark.asyncio
    async def test_pipeline_lifecycle(self):
        """Test pipeline start/stop."""
        comm_manager = AFDCommunicationManager()
        pipeline = AFDMicrobatchPipeline(comm_manager)
        
        await pipeline.start()
        assert pipeline._running is True
        
        await pipeline.stop()
        assert pipeline._running is False


class TestAFDWorkerMetrics:
    """Test AFD worker metrics."""

    def test_create_worker_metrics(self):
        """Test creating worker metrics."""
        metrics = AFDWorkerMetrics(
            worker_type="attention",
            worker_id="attn-0",
        )
        
        assert metrics.worker_type == "attention"
        assert metrics.worker_id == "attn-0"
        assert metrics.total_requests == 0

    def test_record_requests(self):
        """Test recording request metrics."""
        metrics = AFDWorkerMetrics(worker_type="ffn", worker_id="ffn-0")
        
        # Record multiple requests
        for i in range(5):
            metrics.record_request_start()
            metrics.record_request_end(compute_time_ms=10.0 + i)
        
        assert metrics.total_requests == 5
        assert metrics.active_requests == 0
        assert metrics.total_compute_time_ms == 60.0  # 10+11+12+13+14

    def test_record_transfers(self):
        """Test recording transfer metrics."""
        metrics = AFDWorkerMetrics(worker_type="attention", worker_id="attn-0")
        
        metrics.record_transfer(
            bytes_transferred=1024 * 1024,  # 1MB
            latency_ms=5.0,
            transfer_type="activation",
        )
        
        assert metrics.total_bytes_sent == 1024 * 1024
        assert metrics.total_transfer_time_ms == 5.0
        assert len(metrics.latency_samples) == 1

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        metrics = AFDWorkerMetrics(worker_type="attention", worker_id="attn-0")
        
        # Add samples from 1 to 100
        for i in range(1, 101):
            metrics.latency_samples.append(float(i))
        
        assert metrics.avg_latency_ms == 50.5
        assert metrics.p99_latency_ms == 99.0

    def test_queue_tracking(self):
        """Test queue length tracking."""
        metrics = AFDWorkerMetrics(worker_type="attention", worker_id="attn-0")
        
        metrics.update_queue_length(10)
        assert metrics.current_queue_length == 10
        assert metrics.max_queue_length == 10
        
        metrics.update_queue_length(5)
        assert metrics.current_queue_length == 5
        assert metrics.max_queue_length == 10  # Max stays at 10

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        metrics = AFDWorkerMetrics(worker_type="attention", worker_id="attn-0")
        
        metrics.update_memory_usage(
            kv_cache=1024 * 1024 * 1024,  # 1GB
            activation=512 * 1024 * 1024,  # 512MB
        )
        
        assert metrics.kv_cache_memory == 1024 * 1024 * 1024
        assert metrics.activation_memory == 512 * 1024 * 1024

    def test_export_to_dict(self):
        """Test exporting metrics to dictionary."""
        metrics = AFDWorkerMetrics(worker_type="ffn", worker_id="ffn-0")
        metrics.record_request_start()
        metrics.record_request_end(10.0)
        metrics.record_transfer(1024, 5.0)
        
        exported = metrics.to_dict()
        
        assert exported["worker_type"] == "ffn"
        assert exported["worker_id"] == "ffn-0"
        assert exported["total_requests"] == 1
        assert exported["total_bytes_received"] == 1024  # FFN receives


class TestAFDMetricsCollector:
    """Test AFD metrics collector."""

    def test_create_collector(self):
        """Test creating metrics collector."""
        collector = AFDMetricsCollector(attention_ratio=8)
        
        assert collector.attention_ratio == 8
        assert len(collector._attention_workers) == 0
        assert len(collector._ffn_workers) == 0

    def test_register_workers(self):
        """Test registering workers."""
        collector = AFDMetricsCollector()
        
        attn_metrics = collector.register_attention_worker("attn-0")
        ffn_metrics = collector.register_ffn_worker("ffn-0")
        
        assert len(collector._attention_workers) == 1
        assert len(collector._ffn_workers) == 1
        assert attn_metrics.worker_type == "attention"
        assert ffn_metrics.worker_type == "ffn"

    def test_aggregate_metrics(self):
        """Test aggregate metrics calculation."""
        collector = AFDMetricsCollector(attention_ratio=4)
        
        # Register workers
        for i in range(4):
            attn = collector.register_attention_worker(f"attn-{i}")
            attn.record_request_start()
            attn.record_request_end(10.0)
            attn.record_transfer(1024, 5.0)
        
        ffn = collector.register_ffn_worker("ffn-0")
        ffn.record_request_start()
        ffn.record_request_end(20.0)
        
        agg = collector.get_aggregate_metrics()
        
        assert agg["attention_ratio"] == 4
        assert agg["num_attention_workers"] == 4
        assert agg["num_ffn_workers"] == 1
        assert agg["total_attention_requests"] == 4
        assert agg["total_ffn_requests"] == 1


class TestAFDPerformanceAnalyzer:
    """Test AFD performance analyzer."""

    def test_detect_bottleneck_attention_bound(self):
        """Test detecting attention-bound workload."""
        collector = AFDMetricsCollector()
        
        # Create mock metrics
        attn = collector.register_attention_worker("attn-0")
        attn.total_compute_time_ms = 100.0
        attn.total_transfer_time_ms = 10.0
        
        ffn = collector.register_ffn_worker("ffn-0")
        ffn.total_compute_time_ms = 10.0
        
        analyzer = AFDPerformanceAnalyzer(collector)
        result = analyzer.detect_bottleneck()
        
        assert result["bottleneck"] == "attention"
        assert result["time_ratio"] > 1.5

    def test_detect_bottleneck_ffn_bound(self):
        """Test detecting FFN-bound workload."""
        collector = AFDMetricsCollector()
        
        attn = collector.register_attention_worker("attn-0")
        attn.total_compute_time_ms = 10.0
        attn.total_transfer_time_ms = 5.0
        
        ffn = collector.register_ffn_worker("ffn-0")
        ffn.total_compute_time_ms = 100.0
        
        analyzer = AFDPerformanceAnalyzer(collector)
        result = analyzer.detect_bottleneck()
        
        assert result["bottleneck"] == "ffn"
        assert result["time_ratio"] < 0.67

    def test_detect_balanced(self):
        """Test detecting balanced workload."""
        collector = AFDMetricsCollector()
        
        attn = collector.register_attention_worker("attn-0")
        attn.total_compute_time_ms = 50.0
        attn.total_transfer_time_ms = 5.0
        
        ffn = collector.register_ffn_worker("ffn-0")
        ffn.total_compute_time_ms = 50.0
        
        analyzer = AFDPerformanceAnalyzer(collector)
        result = analyzer.detect_bottleneck()
        
        assert result["bottleneck"] == "balanced"

    def test_calculate_optimal_ratio(self):
        """Test optimal ratio calculation."""
        collector = AFDMetricsCollector(attention_ratio=1)
        
        attn = collector.register_attention_worker("attn-0")
        attn.total_compute_time_ms = 100.0
        
        ffn = collector.register_ffn_worker("ffn-0")
        ffn.total_compute_time_ms = 25.0
        
        analyzer = AFDPerformanceAnalyzer(collector)
        optimal = analyzer.calculate_optimal_ratio()
        
        # Optimal should be around 4 (100/25)
        assert 3 <= optimal <= 5

    def test_resource_utilization(self):
        """Test resource utilization analysis."""
        collector = AFDMetricsCollector(attention_ratio=4)
        
        for i in range(4):
            attn = collector.register_attention_worker(f"attn-{i}")
            attn.update_queue_length(i + 1)
        
        ffn = collector.register_ffn_worker("ffn-0")
        ffn.update_queue_length(10)
        
        analyzer = AFDPerformanceAnalyzer(collector)
        util = analyzer.get_resource_utilization()
        
        assert util["attention"]["num_workers"] == 4
        assert util["ffn"]["num_workers"] == 1
        assert util["ratio_actual"] == 4.0


class TestAFDIntegration:
    """Integration tests for AFD components."""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test end-to-end AFD flow (simulated)."""
        # Create communication manager
        comm_manager = AFDCommunicationManager(
            ffn_endpoint="test.ffn",
            attention_ratio=2,
        )
        
        # Create metrics collector
        metrics_collector = AFDMetricsCollector(attention_ratio=2)
        attn_metrics = metrics_collector.register_attention_worker("attn-0")
        ffn_metrics = metrics_collector.register_ffn_worker("ffn-0")
        
        # Simulate attention worker
        attn_metrics.record_request_start()
        attn_metrics.update_memory_usage(kv_cache=1024 * 1024 * 1024)
        
        # Create activation batch
        batch = AFDActivationBatch(
            request_id="test-req-1",
            layer_idx=0,
            activations=np.random.randn(1, 128, 4096).astype(np.float32),
            metadata={"temperature": 0.7},
        )
        
        # Serialize
        serialized = batch.serialize()
        assert len(serialized) > 0
        
        # Simulate transfer
        attn_metrics.record_transfer(
            bytes_transferred=len(serialized),
            latency_ms=5.0,
        )
        
        # Simulate FFN worker
        ffn_metrics.record_request_start()
        ffn_metrics.update_memory_usage(activation=512 * 1024 * 1024)
        
        # Deserialize
        deserialized = AFDActivationBatch.deserialize(serialized)
        assert deserialized.request_id == "test-req-1"
        
        # Create FFN result
        result = AFDFFNResult(
            request_id="test-req-1",
            output=np.random.randn(1, 128, 4096).astype(np.float32),
        )
        
        # Complete requests
        attn_metrics.record_request_end(10.0)
        ffn_metrics.record_request_end(15.0)
        
        # Analyze
        analyzer = AFDPerformanceAnalyzer(metrics_collector)
        bottleneck = analyzer.detect_bottleneck()
        
        assert bottleneck["bottleneck"] in ["attention", "ffn", "balanced"]
        assert attn_metrics.total_requests == 1
        assert ffn_metrics.total_requests == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
