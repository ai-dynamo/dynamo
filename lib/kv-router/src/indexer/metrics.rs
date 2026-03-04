// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "metrics")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "metrics")]
use dynamo_runtime::{component::Component, metrics::MetricsHierarchy, prometheus_names::kvrouter};
use prometheus::{IntCounterVec, Opts};

use crate::protocols::{KvCacheEventData, KvCacheEventError};

/// Metrics for the KV Indexer.
#[derive(Clone)]
pub struct KvIndexerMetrics {
    /// Counter of events applied.
    pub kv_cache_events_applied: IntCounterVec,
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

/// Metric name for KV cache events applied counter.
const KV_CACHE_EVENTS_APPLIED_NAME: &str = "dynamo_kvrouter_kv_cache_events_applied";

#[cfg(feature = "metrics")]
static KV_INDEXER_METRICS: OnceLock<Arc<KvIndexerMetrics>> = OnceLock::new();

impl KvIndexerMetrics {
    #[cfg(feature = "metrics")]
    fn new(kv_cache_events_applied: IntCounterVec) -> Self {
        Self {
            kv_cache_events_applied,
        }
    }

    /// Creates a new KvIndexerMetrics from a Component, memoizing the result in
    /// KV_INDEXER_METRICS to avoid duplicate registration issues.
    #[cfg(feature = "metrics")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        KV_INDEXER_METRICS.get_or_init(|| {
            match component.metrics().create_intcountervec(
                kvrouter::KV_CACHE_EVENTS_APPLIED,
                "Total number of KV cache events applied to index",
                &["event_type", "status"],
                &[],
            ) {
                Ok(kv_cache_events_applied) => Arc::new(Self::new(kv_cache_events_applied)),
                Err(e) => {
                    tracing::warn!("Failed to create kv indexer metrics from component: {}. Using unregistered metrics as fallback.", e);
                    Arc::new(Self::new_unregistered())
                }
            }
        }).clone()
    }

    /// Creates a new KvIndexerMetrics which is not registered with a MetricsRegistry.
    /// This may be used for tests or as a fallback for when a MetricsRegistry is not available / has errored.
    pub fn new_unregistered() -> Self {
        Self {
            kv_cache_events_applied: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_EVENTS_APPLIED_NAME,
                    "Total number of KV cache events applied to index",
                ),
                &["event_type", "status"],
            )
            .unwrap(),
        }
    }

    pub fn get_event_type(event_data: &KvCacheEventData) -> &'static str {
        match event_data {
            KvCacheEventData::Stored(_) => METRIC_EVENT_STORED,
            KvCacheEventData::Removed(_) => METRIC_EVENT_REMOVED,
            KvCacheEventData::Cleared => METRIC_EVENT_CLEARED,
        }
    }

    pub fn increment_event_applied(
        &self,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) {
        match result {
            Ok(_) => {
                self.kv_cache_events_applied
                    .with_label_values(&[event_type, METRIC_STATUS_OK])
                    .inc_by(1);
            }
            Err(e) => {
                let error_label = match e {
                    KvCacheEventError::ParentBlockNotFound => METRIC_STATUS_PARENT_NOT_FOUND,
                    KvCacheEventError::BlockNotFound => METRIC_STATUS_BLOCK_NOT_FOUND,
                    KvCacheEventError::InvalidBlockSequence => METRIC_STATUS_INVALID_BLOCK,
                };
                self.kv_cache_events_applied
                    .with_label_values(&[event_type, error_label])
                    .inc_by(1);
            }
        }
    }
}
