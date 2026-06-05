// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the service shell.
//!
//! Metrics are registered into a local [`prometheus::Registry`] so the HTTP
//! sidecar can serve a focused `/metrics` endpoint without touching the
//! global default registry (which other crates may pollute).

use std::sync::Arc;

use prometheus::{IntCounter, IntGauge, IntGaugeVec, Opts, Registry};

/// Prometheus metrics handle. Cheap to clone — all inner counters are `Arc`-shared
/// inside `prometheus` itself.
#[derive(Clone)]
pub struct ServiceMetrics {
    pub registry: Arc<Registry>,
    pub capacity_slots: IntGauge,
    pub used_slots: IntGauge,
    pub registered_clients: IntGauge,
    pub register_total: IntCounter,
    pub register_rejected_total: IntCounter,
    pub unregister_total: IntCounter,
    pub reset_total: IntCounter,
    /// Bytes allocated to the host-memory pool per `{node, tier}` label
    /// pair. `tier` is the [`dynamo_memory::HugepageTier`] the slab landed
    /// on (`Explicit`, `Thp`, or `None`).
    pub pool_bytes_total: IntGaugeVec,
    /// Number of slabs the host-memory pool created per `{tier}`.
    pub pool_slabs_total: IntGaugeVec,
}

impl ServiceMetrics {
    /// Build a fresh registry with all metrics declared and registered.
    pub fn new() -> Self {
        let registry = Arc::new(Registry::new());

        let capacity_slots = IntGauge::new(
            "kvbm_service_capacity_slots",
            "Total slot capacity (num GPUs)",
        )
        .unwrap();
        let used_slots = IntGauge::new(
            "kvbm_service_used_slots",
            "Slots reserved by currently registered clients",
        )
        .unwrap();
        let registered_clients = IntGauge::new(
            "kvbm_service_registered_clients",
            "Number of clients currently holding registrations",
        )
        .unwrap();
        let register_total = IntCounter::new(
            "kvbm_service_register_total",
            "Total successful Register RPCs",
        )
        .unwrap();
        let register_rejected_total = IntCounter::new(
            "kvbm_service_register_rejected_total",
            "Total Register RPCs rejected (validation or capacity)",
        )
        .unwrap();
        let unregister_total = IntCounter::new(
            "kvbm_service_unregister_total",
            "Total registrations ended (graceful or stream drop)",
        )
        .unwrap();
        let reset_total = IntCounter::new(
            "kvbm_service_reset_total",
            "Total times the service returned to the empty state",
        )
        .unwrap();

        let pool_bytes_total = IntGaugeVec::new(
            Opts::new(
                "kvbm_service_pool_bytes_total",
                "Bytes allocated to the host-memory pool, labeled by NUMA node id \
                 and the hugepage tier the slab landed on",
            ),
            &["node", "tier"],
        )
        .unwrap();
        let pool_slabs_total = IntGaugeVec::new(
            Opts::new(
                "kvbm_service_pool_slabs_total",
                "Number of host-memory pool slabs by hugepage tier",
            ),
            &["tier"],
        )
        .unwrap();

        registry.register(Box::new(capacity_slots.clone())).unwrap();
        registry.register(Box::new(used_slots.clone())).unwrap();
        registry
            .register(Box::new(registered_clients.clone()))
            .unwrap();
        registry.register(Box::new(register_total.clone())).unwrap();
        registry
            .register(Box::new(register_rejected_total.clone()))
            .unwrap();
        registry
            .register(Box::new(unregister_total.clone()))
            .unwrap();
        registry.register(Box::new(reset_total.clone())).unwrap();
        registry
            .register(Box::new(pool_bytes_total.clone()))
            .unwrap();
        registry
            .register(Box::new(pool_slabs_total.clone()))
            .unwrap();

        Self {
            registry,
            capacity_slots,
            used_slots,
            registered_clients,
            register_total,
            register_rejected_total,
            unregister_total,
            reset_total,
            pool_bytes_total,
            pool_slabs_total,
        }
    }

    /// Render the registry as Prometheus text format.
    pub fn encode_text(&self) -> Result<String, prometheus::Error> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let mut buf = Vec::new();
        encoder.encode(&self.registry.gather(), &mut buf)?;
        Ok(String::from_utf8(buf).unwrap_or_default())
    }
}

impl Default for ServiceMetrics {
    fn default() -> Self {
        Self::new()
    }
}
