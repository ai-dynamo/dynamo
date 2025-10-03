// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use prometheus::{
    IntCounterVec, IntGaugeVec, Opts, Registry,
    core::{AtomicI64, AtomicU64, GenericCounter, GenericGauge},
    register_int_counter_vec_with_registry, register_int_gauge_vec_with_registry,
};
use std::sync::Arc;
pub struct BlockManagerMetrics {
    gauges: IntGaugeVec,
    counters: IntCounterVec,
}

impl BlockManagerMetrics {
    pub fn new(metrics_registry: &Arc<Registry>) -> Result<Arc<Self>> {
        let gauge_opts = Opts::new("gauges", "Gauges for the pools")
            .namespace("dynamo")
            .subsystem("kvbm");

        let counter_opts = Opts::new("pools", "Counters for the pools")
            .namespace("dynamo")
            .subsystem("kvbm");

        let gauges = register_int_gauge_vec_with_registry!(
            gauge_opts,
            &["pool", "metric_type"],
            metrics_registry
        )?;

        let counters = register_int_counter_vec_with_registry!(
            counter_opts,
            &["pool", "metric_type"],
            metrics_registry
        )?;

        Ok(Arc::new(Self { gauges, counters }))
    }

    pub fn pool(self: &Arc<Self>, group: &str) -> Arc<PoolMetrics> {
        PoolMetrics::new(self, group)
    }
}

pub struct PoolMetrics {
    block_manager_metrics: Arc<BlockManagerMetrics>,
    group: String,
}

impl PoolMetrics {
    pub fn new(block_manager_metrics: &Arc<BlockManagerMetrics>, group: &str) -> Arc<Self> {
        Arc::new(Self {
            block_manager_metrics: block_manager_metrics.clone(),
            group: group.to_string(),
        })
    }

    pub fn gauge(&self, metric_type: &str) -> GenericGauge<AtomicI64> {
        self.block_manager_metrics
            .gauges
            .with_label_values(&[&self.group, &metric_type.to_string()])
    }

    pub fn counter(&self, metric_type: &str) -> GenericCounter<AtomicU64> {
        self.block_manager_metrics
            .counters
            .with_label_values(&[&self.group, &metric_type.to_string()])
    }
}
