# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Python MetricsRegistry bindings.

This test suite verifies that Python can create, introspect, and use Prometheus
metrics through the Dynamo MetricsRegistry interface.
"""

import pytest


async def get_metrics_runtime(runtime, endpoint_name):
    """Helper to create a unique metrics runtime for each test."""
    namespace = runtime.namespace("test_metrics_ns")
    component = namespace.component("test_metrics_comp")
    endpoint = component.endpoint(endpoint_name)
    return endpoint.metrics


pytestmark = pytest.mark.pre_merge


@pytest.mark.asyncio
@pytest.mark.forked
async def test_counter_introspection(runtime):
    """Test Counter metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_counter_introspection")

    counter = metrics_runtime.create_counter(
        "test_counter", "A test counter", [("env", "test")]  # constant labels
    )

    # Test name() method
    name = counter.name()
    assert isinstance(name, str)
    assert "test_counter" in name
    assert name.endswith("test_counter")

    # Test const_labels() method
    labels = counter.const_labels()
    assert isinstance(labels, dict)
    assert "env" in labels
    assert labels["env"] == "test"
    assert "dynamo_namespace" in labels
    assert labels["dynamo_namespace"] == "test_metrics_ns"


@pytest.mark.asyncio
@pytest.mark.forked
async def test_intcounter_introspection(runtime):
    """Test IntCounter metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_intcounter_introspection")
    counter = metrics_runtime.create_intcounter(
        "test_int_counter", "A test int counter", [("type", "integer")]
    )

    name = counter.name()
    assert isinstance(name, str)
    assert "test_int_counter" in name

    labels = counter.const_labels()
    assert isinstance(labels, dict)
    assert labels["type"] == "integer"


@pytest.mark.asyncio
@pytest.mark.forked
async def test_gauge_introspection(runtime):
    """Test Gauge metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_gauge_introspection")
    gauge = metrics_runtime.create_gauge(
        "test_gauge", "A test gauge", [("unit", "bytes")]
    )

    name = gauge.name()
    assert isinstance(name, str)
    assert "test_gauge" in name

    labels = gauge.const_labels()
    assert isinstance(labels, dict)
    assert labels["unit"] == "bytes"


@pytest.mark.asyncio
@pytest.mark.forked
async def test_intgauge_introspection(runtime):
    """Test IntGauge metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_intgauge_introspection")
    gauge = metrics_runtime.create_intgauge(
        "test_int_gauge", "A test int gauge", []  # no constant labels
    )

    name = gauge.name()
    assert isinstance(name, str)
    assert "test_int_gauge" in name

    labels = gauge.const_labels()
    assert isinstance(labels, dict)
    # Should still have hierarchy labels
    assert "dynamo_namespace" in labels


@pytest.mark.asyncio
@pytest.mark.forked
async def test_histogram_introspection(runtime):
    """Test Histogram metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_histogram_introspection")
    histogram = metrics_runtime.create_histogram(
        "test_histogram", "A test histogram", [("method", "POST")]
    )

    name = histogram.name()
    assert isinstance(name, str)
    assert "test_histogram" in name

    labels = histogram.const_labels()
    assert isinstance(labels, dict)
    assert labels["method"] == "POST"


@pytest.mark.asyncio
@pytest.mark.forked
async def test_countervec_introspection(runtime):
    """Test CounterVec metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_countervec_introspection")
    counter_vec = metrics_runtime.create_countervec(
        "test_counter_vec",
        "A test counter vec",
        ["worker_id", "status"],  # variable labels
        [("cluster", "prod")],  # constant labels
    )

    # Test name()
    name = counter_vec.name()
    assert isinstance(name, str)
    assert "test_counter_vec" in name

    # Test const_labels()
    const_labels = counter_vec.const_labels()
    assert isinstance(const_labels, dict)
    assert const_labels["cluster"] == "prod"
    assert "dynamo_namespace" in const_labels

    # Test variable_labels()
    var_labels = counter_vec.variable_labels()
    assert isinstance(var_labels, list)
    assert len(var_labels) == 2
    assert "worker_id" in var_labels
    assert "status" in var_labels


@pytest.mark.asyncio
@pytest.mark.forked
async def test_intcountervec_introspection(runtime):
    """Test IntCounterVec metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(
        runtime, "ep_intcountervec_introspection"
    )
    counter_vec = metrics_runtime.create_intcountervec(
        "test_int_counter_vec",
        "A test int counter vec",
        ["region", "zone"],
        [],  # no constant labels
    )

    name = counter_vec.name()
    assert "test_int_counter_vec" in name

    const_labels = counter_vec.const_labels()
    assert isinstance(const_labels, dict)

    var_labels = counter_vec.variable_labels()
    assert len(var_labels) == 2
    assert "region" in var_labels
    assert "zone" in var_labels


@pytest.mark.asyncio
@pytest.mark.forked
async def test_gaugevec_introspection(runtime):
    """Test GaugeVec metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_gaugevec_introspection")
    gauge_vec = metrics_runtime.create_gaugevec(
        "test_gauge_vec", "A test gauge vec", ["instance", "job"], [("env", "staging")]
    )

    name = gauge_vec.name()
    assert "test_gauge_vec" in name

    const_labels = gauge_vec.const_labels()
    assert const_labels["env"] == "staging"

    var_labels = gauge_vec.variable_labels()
    assert len(var_labels) == 2
    assert "instance" in var_labels
    assert "job" in var_labels


@pytest.mark.asyncio
@pytest.mark.forked
async def test_intgaugevec_introspection(runtime):
    """Test IntGaugeVec metric introspection methods."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_intgaugevec_introspection")
    gauge_vec = metrics_runtime.create_intgaugevec(
        "test_int_gauge_vec",
        "A test int gauge vec",
        ["device", "partition"],
        [("datacenter", "us-west")],
    )

    name = gauge_vec.name()
    assert "test_int_gauge_vec" in name

    const_labels = gauge_vec.const_labels()
    assert const_labels["datacenter"] == "us-west"

    var_labels = gauge_vec.variable_labels()
    assert len(var_labels) == 2
    assert "device" in var_labels
    assert "partition" in var_labels


@pytest.mark.asyncio
@pytest.mark.forked
async def test_metric_operations(runtime):
    """Test that metrics can be used after introspection."""
    metrics_runtime = await get_metrics_runtime(runtime, "ep_metric_operations")
    # Counter operations
    counter = metrics_runtime.create_intcounter("ops_counter", "Operations counter", [])
    counter.inc()
    counter.inc_by(5)
    assert counter.get() == 6

    # Gauge operations
    gauge = metrics_runtime.create_intgauge(
        "connections_gauge", "Connections gauge", []
    )
    gauge.set(10)
    assert gauge.get() == 10
    gauge.inc()
    assert gauge.get() == 11
    gauge.dec()
    assert gauge.get() == 10

    # Vec operations
    gauge_vec = metrics_runtime.create_intgaugevec(
        "worker_gauge_vec", "Worker gauge vec", ["worker_id"], []
    )
    gauge_vec.set(5, {"worker_id": "w1"})
    assert gauge_vec.get({"worker_id": "w1"}) == 5
    gauge_vec.inc({"worker_id": "w1"})
    assert gauge_vec.get({"worker_id": "w1"}) == 6


@pytest.mark.asyncio
@pytest.mark.forked
async def test_multiple_metrics_same_runtime(runtime):
    """Test creating multiple metrics in the same runtime."""
    metrics_runtime = await get_metrics_runtime(
        runtime, "ep_multiple_metrics_same_runtime"
    )
    counter1 = metrics_runtime.create_intcounter("counter1", "Counter 1", [])
    counter2 = metrics_runtime.create_intcounter("counter2", "Counter 2", [])
    gauge1 = metrics_runtime.create_gauge("gauge1", "Gauge 1", [])

    # All should have unique names
    names = {counter1.name(), counter2.name(), gauge1.name()}
    assert len(names) == 3

    # All should share the same hierarchy labels
    for metric in [counter1, counter2, gauge1]:
        labels = metric.const_labels()
        assert labels["dynamo_namespace"] == "test_metrics_ns"
        assert "dynamo_component" in labels  # Component name is test-specific
        assert labels["dynamo_endpoint"] == "ep_multiple_metrics_same_runtime"
