// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared registry and metric handles for a KVBM runtime.

use std::sync::Arc;

use prometheus::Registry;

use crate::{CompatMetrics, MetricsAggregator, TransferMetrics, start_metrics_server};

/// Shared observability surface for a runtime or embedded component.
#[derive(Clone)]
pub struct KvbmObservability {
    registry: Registry,
    logical_aggregator: MetricsAggregator,
    compat_metrics: CompatMetrics,
    transfer_metrics: TransferMetrics,
}

impl KvbmObservability {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        let logical_aggregator = MetricsAggregator::new();
        logical_aggregator.register_with(&registry)?;

        let compat_metrics = CompatMetrics::new();
        compat_metrics.register(&registry)?;

        let transfer_metrics = TransferMetrics::new();
        transfer_metrics.register(&registry)?;

        Ok(Self {
            registry,
            logical_aggregator,
            compat_metrics,
            transfer_metrics,
        })
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn logical_aggregator(&self) -> MetricsAggregator {
        self.logical_aggregator.clone()
    }

    pub fn compat_metrics(&self) -> &CompatMetrics {
        &self.compat_metrics
    }

    pub fn transfer_metrics(&self) -> &TransferMetrics {
        &self.transfer_metrics
    }

    pub fn start_server(&self, enabled: bool, port: u16) {
        if enabled {
            let _handle = start_metrics_server(self.registry.clone(), port);
        }
    }

    pub fn set_external_labels(&self, labels: Vec<(String, String)>) {
        self.logical_aggregator.set_external_labels(labels);
    }
}

impl Default for KvbmObservability {
    fn default() -> Self {
        Self::new().expect("valid observability registry")
    }
}

pub type SharedKvbmObservability = Arc<KvbmObservability>;
