# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom Prometheus collector that injects additional labels into metrics.

Wraps a source registry and clones metrics during collection, injecting user-specified
labels without modifying the original metrics. Preserves all metric types, reserved
labels (histogram 'le', summary 'quantile'), timestamps, and exemplars.
"""

from typing import TYPE_CHECKING, Iterator

from prometheus_client.registry import Collector

# Import prometheus_client types for type hints only
# Actual prometheus_client imports happen inside methods to respect initialization order
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry
    from prometheus_client.metrics_core import Metric as MetricFamily


class LabelInjectingCollector(Collector):
    """
    Prometheus collector that injects labels into metrics during collection.
    Preserves all metric types, reserved labels, timestamps, and exemplars.
    """

    def __init__(
        self,
        source_registry: "CollectorRegistry",
        labels_to_inject: dict[str, str],
    ):
        """
        Args:
            source_registry: Source registry to collect from
            labels_to_inject: Labels to inject (e.g. {"dynamo_namespace": "prod"})

        Raises:
            ValueError: If labels_to_inject is empty or contains reserved labels (le, quantile)
        """
        if not labels_to_inject:
            raise ValueError("labels_to_inject cannot be empty")

        # Check for reserved label names that should not be overwritten
        reserved_labels = {"le", "quantile"}
        conflicting_labels = reserved_labels.intersection(labels_to_inject.keys())
        if conflicting_labels:
            raise ValueError(
                f"Cannot inject reserved label names: {conflicting_labels}. "
                f"Reserved labels: {reserved_labels}"
            )

        self.source_registry = source_registry
        self.labels_to_inject = labels_to_inject

    def collect(self) -> Iterator["MetricFamily"]:
        """
        Collect metrics from source registry with injected labels.
        Preserves all metric types, reserved labels, timestamps, and exemplars.
        """
        # Delayed import here to respect prometheus_client multiprocess initialization order
        from prometheus_client.metrics_core import Metric as MetricFamily
        from prometheus_client.samples import Sample

        for metric_family in self.source_registry.collect():
            # Clone the metric family with injected labels
            new_samples = []
            for sample in metric_family.samples:
                # Merge existing labels with injected labels
                # Existing labels take precedence (don't overwrite if already present)
                merged_labels = {**self.labels_to_inject, **sample.labels}

                # Create new sample with merged labels
                new_sample = Sample(
                    name=sample.name,
                    labels=merged_labels,
                    value=sample.value,
                    timestamp=sample.timestamp,
                    exemplar=sample.exemplar,
                )
                new_samples.append(new_sample)

            # Create new metric family with updated samples
            new_metric_family = MetricFamily(
                name=metric_family.name,
                documentation=metric_family.documentation,
                typ=metric_family.type,
                unit=metric_family.unit,
            )
            new_metric_family.samples = new_samples

            yield new_metric_family

    def describe(self) -> Iterator["MetricFamily"]:
        """Describe metrics from source registry (forwards to source)."""
        return iter(self.source_registry.collect())
