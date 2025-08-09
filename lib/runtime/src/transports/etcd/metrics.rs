// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::metrics::MetricsRegistry;
use prometheus::IntGauge;
use std::sync::Arc;

// Metrics configuration for etcd transport
#[derive(Clone, Debug)]
pub struct EtcdMetrics {
    pub etcd_block_total: IntGauge,
    pub etcd_block_bytes_total: IntGauge,
}

impl EtcdMetrics {
    pub fn new(etcd_block_total: IntGauge, etcd_block_bytes_total: IntGauge) -> Self {
        Self {
            etcd_block_total,
            etcd_block_bytes_total,
        }
    }

    pub fn from_distributed_runtime(drt: &crate::DistributedRuntime) -> anyhow::Result<Self> {
        let etcd_block_total = drt.create_intgauge(
            "etcd_block_total",
            "Total number of etcd key-value pairs processed",
            &[],
        )?;

        let etcd_block_bytes_total = drt.create_intgauge(
            "etcd_block_bytes_total",
            "Total number of bytes processed in etcd blocks",
            &[],
        )?;

        Ok(Self::new(
            Arc::into_inner(etcd_block_total).unwrap(),
            Arc::into_inner(etcd_block_bytes_total).unwrap(),
        ))
    }
}
