# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filtering semantics of the typed engine-metrics contract.

Whether the typed form carries what the engines actually emit is asserted
end to end against live deployments -- see ``MetricsPayload`` in
``tests/utils/payloads.py``, which runs against real vLLM, SGLang and
TensorRT-LLM. Constructed registries cannot answer that question honestly,
since they only contain shapes someone thought to write down.

What is left here is the part a real engine cannot exercise: which filter
inputs select which families.
"""

import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram

from dynamo.common.utils.prometheus import get_prometheus_typed


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_filtering_is_family_level_in_the_typed_contract():
    """The two contracts filter on different things, and this pins which.

    The text form matches each rendered *line*, so its names carry the
    ``_total`` / ``_created`` suffixes. The typed form matches the *family*
    name from ``collect()``, which does not. For the prefix filters actually
    used (``vllm:``, ``sglang:``, ``trtllm_``, ``python_``, ``process_``) both
    agree, because family and sample names share the prefix.

    They diverge only for a filter naming a suffixed sample. Family-level is
    the intended behaviour here: filtering part of a histogram would leave a
    family whose buckets and ``_count`` disagree.
    """
    registry = CollectorRegistry()
    Counter("vllm:request_success", "Successful requests", registry=registry).inc(3)

    families = [m.name for m in registry.collect()]
    samples = [sample.name for m in registry.collect() for sample in m.samples]
    assert families == ["vllm:request_success"]
    assert "vllm:request_success_total" in samples

    # The prefix both share: agreement.
    assert len(get_prometheus_typed(registry, metric_prefix_filters=["vllm:"])) == 1

    # A suffixed sample name is not the family name, so it selects nothing.
    # Documented as intentional rather than discovered later as a surprise.
    assert (
        get_prometheus_typed(
            registry, metric_prefix_filters=["vllm:request_success_total"]
        )
        == []
    )

    # A whole family is kept or dropped together: never a partial histogram.
    Histogram(
        "vllm:latency_seconds", "Latency", buckets=[0.1, 1.0], registry=registry
    ).observe(0.5)
    typed = get_prometheus_typed(registry, metric_prefix_filters=["vllm:latency"])
    assert len(typed) == 1
    sample_names = {s[0] for s in typed[0][3]}
    assert {
        "vllm:latency_seconds_bucket",
        "vllm:latency_seconds_sum",
        "vllm:latency_seconds_count",
    } <= sample_names
