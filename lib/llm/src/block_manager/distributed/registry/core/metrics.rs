// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional metrics hooks for registry hub operations.

use std::time::Duration;

/// Observer interface for hub operation metrics.
///
/// This keeps core registry logic decoupled from any metrics backend.
pub trait RegistryMetricsSink: Send + Sync {
    /// Called when a query payload cannot be decoded.
    fn on_query_decode_failure(&self) {}

    /// Called when a query request is processed.
    fn on_query_processed(&self, _query_type: &'static str, _keys: usize, _latency: Duration) {}

    /// Called after handling a `match` query.
    fn on_match_result(&self, _hits: usize, _misses: usize) {}

    /// Called when record count changes outside registration (for example remove).
    fn on_record_count_change(&self, _records_current: usize) {}

    /// Called after processing a register batch.
    fn on_register_batch(&self, _entries: usize, _latency: Duration, _records_current: usize) {}

    /// Called after a `can_offload` response is computed.
    fn on_can_offload_result(
        &self,
        _granted: usize,
        _already_stored: usize,
        _leased: usize,
        _active_leases: usize,
    ) {
    }

    /// Called when leases are released during successful registration.
    fn on_leases_released(&self, _released: usize, _active_leases: usize) {}

    /// Called when background cleanup expires leases.
    fn on_leases_expired(&self, _expired: usize, _active_leases: usize) {}
}

/// Default sink that performs no metric collection.
#[derive(Debug, Default)]
pub struct NoopRegistryMetricsSink;

impl RegistryMetricsSink for NoopRegistryMetricsSink {}
