// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "runtime-protocols")]
use std::sync::Arc;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use std::sync::OnceLock;

#[cfg(feature = "runtime-protocols")]
use dynamo_runtime::component::Component;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use dynamo_runtime::metrics::MetricsHierarchy;
#[cfg(feature = "metrics")]
use prometheus::{IntCounterVec, Opts};

use crate::protocols::{KvCacheEventData, KvCacheEventError};

/// Lightweight, `Copy` discriminant for [`KvCacheEventData`].
///
/// Extracted before the event is moved into `apply_event()`, then passed to
/// [`PreBoundEventCounters::inc`] so the compiler enforces exhaustiveness
/// without requiring a clone of the full event payload.
///
/// `Display` produces the Prometheus label value (`"stored"`, `"removed"`,
/// `"cleared"`), so this enum is also the single source of truth for the
/// `event_type` label — replacing the former `get_event_type()` helper.
#[derive(Debug, Clone, Copy)]
pub enum EventKind {
    Stored,
    Removed,
    Cleared,
}

impl EventKind {
    pub fn of(data: &KvCacheEventData) -> Self {
        match data {
            KvCacheEventData::Stored(_) => Self::Stored,
            KvCacheEventData::Removed(_) => Self::Removed,
            KvCacheEventData::Cleared => Self::Cleared,
        }
    }
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stored => f.write_str(METRIC_EVENT_STORED),
            Self::Removed => f.write_str(METRIC_EVENT_REMOVED),
            Self::Cleared => f.write_str(METRIC_EVENT_CLEARED),
        }
    }
}

/// Metrics for the KV Indexer.
#[derive(Clone)]
#[cfg_attr(not(feature = "metrics"), derive(Default))]
pub struct KvIndexerMetrics {
    /// Counter of events applied.
    #[cfg(feature = "metrics")]
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
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_SUFFIX: &str = "kv_cache_events_applied";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_NAME: &str = "dynamo_kvrouter_kv_cache_events_applied";

#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
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
    #[cfg(feature = "runtime-protocols")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        #[cfg(feature = "metrics")]
        {
            KV_INDEXER_METRICS
                .get_or_init(|| {
                    match component.metrics().create_intcountervec(
                        KV_CACHE_EVENTS_APPLIED_SUFFIX,
                        "Total number of KV cache events applied to index",
                        &["event_type", "status"],
                        &[],
                    ) {
                        Ok(kv_cache_events_applied) => {
                            Arc::new(Self::new(kv_cache_events_applied))
                        }
                        Err(e) => {
                            tracing::warn!("Failed to create kv indexer metrics from component: {}. Using unregistered metrics as fallback.", e);
                            Arc::new(Self::new_unregistered())
                        }
                    }
                })
                .clone()
        }

        #[cfg(not(feature = "metrics"))]
        {
            let _ = component;
            Arc::new(Self::new_unregistered())
        }
    }

    /// Creates a new KvIndexerMetrics which is not registered with a MetricsRegistry.
    /// This may be used for tests or as a fallback for when a MetricsRegistry is not available / has errored.
    #[cfg(feature = "metrics")]
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

    /// Creates a no-op metrics instance when Prometheus support is disabled.
    #[cfg(not(feature = "metrics"))]
    pub fn new_unregistered() -> Self {
        Self::default()
    }

    pub fn increment_event_applied(
        &self,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) {
        #[cfg(feature = "metrics")]
        {
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
        #[cfg(not(feature = "metrics"))]
        let _ = (self, event_type, result);
    }

    /// Pre-resolve all `IntCounter` handles for the finite (event_type, status) label space.
    /// Call this once per worker thread at startup, then use
    /// [`PreBoundEventCounters::inc`] in the hot loop to avoid the
    /// `with_label_values` hashmap lookup on every event.
    pub fn prebind(&self) -> PreBoundEventCounters {
        PreBoundEventCounters::new(self)
    }
}

/// Pre-resolved `IntCounter` handles for every (event_type, status) combination.
///
/// Created once per worker thread via [`KvIndexerMetrics::prebind`], then used in
/// the event processing loop with a direct `.inc()` call instead of the
/// `IntCounterVec::with_label_values()` hashmap lookup.
pub struct PreBoundEventCounters {
    #[cfg(feature = "metrics")]
    stored_ok: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    stored_parent_not_found: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    stored_block_not_found: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    stored_invalid_block: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    removed_ok: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    removed_parent_not_found: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    removed_block_not_found: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    removed_invalid_block: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    cleared_ok: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    cleared_parent_not_found: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    cleared_block_not_found: prometheus::IntCounter,
    #[cfg(feature = "metrics")]
    cleared_invalid_block: prometheus::IntCounter,
}

impl PreBoundEventCounters {
    fn new(metrics: &KvIndexerMetrics) -> Self {
        #[cfg(feature = "metrics")]
        {
            let cv = &metrics.kv_cache_events_applied;
            Self {
                stored_ok: cv.with_label_values(&[METRIC_EVENT_STORED, METRIC_STATUS_OK]),
                stored_parent_not_found: cv
                    .with_label_values(&[METRIC_EVENT_STORED, METRIC_STATUS_PARENT_NOT_FOUND]),
                stored_block_not_found: cv
                    .with_label_values(&[METRIC_EVENT_STORED, METRIC_STATUS_BLOCK_NOT_FOUND]),
                stored_invalid_block: cv
                    .with_label_values(&[METRIC_EVENT_STORED, METRIC_STATUS_INVALID_BLOCK]),
                removed_ok: cv.with_label_values(&[METRIC_EVENT_REMOVED, METRIC_STATUS_OK]),
                removed_parent_not_found: cv
                    .with_label_values(&[METRIC_EVENT_REMOVED, METRIC_STATUS_PARENT_NOT_FOUND]),
                removed_block_not_found: cv
                    .with_label_values(&[METRIC_EVENT_REMOVED, METRIC_STATUS_BLOCK_NOT_FOUND]),
                removed_invalid_block: cv
                    .with_label_values(&[METRIC_EVENT_REMOVED, METRIC_STATUS_INVALID_BLOCK]),
                cleared_ok: cv.with_label_values(&[METRIC_EVENT_CLEARED, METRIC_STATUS_OK]),
                cleared_parent_not_found: cv
                    .with_label_values(&[METRIC_EVENT_CLEARED, METRIC_STATUS_PARENT_NOT_FOUND]),
                cleared_block_not_found: cv
                    .with_label_values(&[METRIC_EVENT_CLEARED, METRIC_STATUS_BLOCK_NOT_FOUND]),
                cleared_invalid_block: cv
                    .with_label_values(&[METRIC_EVENT_CLEARED, METRIC_STATUS_INVALID_BLOCK]),
            }
        }
        #[cfg(not(feature = "metrics"))]
        {
            let _ = metrics;
            Self {}
        }
    }

    /// Increment the pre-resolved counter for the given event kind and result.
    ///
    /// Takes [`EventKind`] (a `Copy` discriminant) instead of a string label,
    /// so the compiler enforces exhaustiveness — a new [`EventKind`] or
    /// [`KvCacheEventError`] variant will produce a compile error here.
    pub fn inc(&self, kind: EventKind, result: Result<(), KvCacheEventError>) {
        #[cfg(feature = "metrics")]
        {
            let counter = match (kind, result) {
                (EventKind::Stored, Ok(())) => &self.stored_ok,
                (EventKind::Stored, Err(KvCacheEventError::ParentBlockNotFound)) => {
                    &self.stored_parent_not_found
                }
                (EventKind::Stored, Err(KvCacheEventError::BlockNotFound)) => {
                    &self.stored_block_not_found
                }
                (EventKind::Stored, Err(KvCacheEventError::InvalidBlockSequence)) => {
                    &self.stored_invalid_block
                }
                (EventKind::Removed, Ok(())) => &self.removed_ok,
                (EventKind::Removed, Err(KvCacheEventError::ParentBlockNotFound)) => {
                    &self.removed_parent_not_found
                }
                (EventKind::Removed, Err(KvCacheEventError::BlockNotFound)) => {
                    &self.removed_block_not_found
                }
                (EventKind::Removed, Err(KvCacheEventError::InvalidBlockSequence)) => {
                    &self.removed_invalid_block
                }
                (EventKind::Cleared, Ok(())) => &self.cleared_ok,
                (EventKind::Cleared, Err(KvCacheEventError::ParentBlockNotFound)) => {
                    &self.cleared_parent_not_found
                }
                (EventKind::Cleared, Err(KvCacheEventError::BlockNotFound)) => {
                    &self.cleared_block_not_found
                }
                (EventKind::Cleared, Err(KvCacheEventError::InvalidBlockSequence)) => {
                    &self.cleared_invalid_block
                }
            };
            counter.inc();
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, kind, result);
    }
}
