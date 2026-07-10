// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(any(feature = "metrics", feature = "runtime-protocols"))]
use std::sync::Arc;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use std::sync::OnceLock;
#[cfg(feature = "bench")]
use std::sync::atomic::AtomicU64;

#[cfg(feature = "runtime-protocols")]
use dynamo_runtime::component::Component;
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
use dynamo_runtime::metrics::MetricsHierarchy;
#[cfg(feature = "metrics")]
use prometheus::{IntCounter, IntCounterVec, Opts};

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

/// Lightweight, `Copy` discriminant for KV event warnings.
#[derive(Debug, Clone, Copy)]
pub enum EventWarningKind {
    DuplicateStore,
}

#[derive(Debug, Clone, Copy)]
pub enum CkfMutationKind {
    UnknownRemove,
    CapacityExhausted,
}

impl std::fmt::Display for CkfMutationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRemove => f.write_str(METRIC_CKF_MUTATION_UNKNOWN_REMOVE),
            Self::CapacityExhausted => f.write_str(METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED),
        }
    }
}

impl std::fmt::Display for EventWarningKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStore => f.write_str(METRIC_WARNING_DUPLICATE_STORE),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerCounters {
    pub(crate) find_match_dispatches: AtomicU64,
    pub(crate) find_match_early_returns: AtomicU64,
    pub(crate) anchor_installs: AtomicU64,
    pub(crate) anchor_reuses: AtomicU64,
    pub(crate) remove_broadcasts: AtomicU64,
}

#[cfg(feature = "bench")]
impl ShardedIndexerCounters {
    fn new() -> Self {
        Self {
            find_match_dispatches: AtomicU64::new(0),
            find_match_early_returns: AtomicU64::new(0),
            anchor_installs: AtomicU64::new(0),
            anchor_reuses: AtomicU64::new(0),
            remove_broadcasts: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerTiming {
    pub(crate) calls: AtomicU64,
    pub(crate) routing_ns: AtomicU64,
    pub(crate) shard_ns: AtomicU64,
}

#[cfg(feature = "bench")]
impl ShardedIndexerTiming {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            routing_ns: AtomicU64::new(0),
            shard_ns: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "bench")]
pub(crate) struct ShardedIndexerMetrics {
    pub(crate) counters: ShardedIndexerCounters,
    pub(crate) timing: ShardedIndexerTiming,
}

#[cfg(feature = "bench")]
impl ShardedIndexerMetrics {
    pub(crate) fn new() -> Self {
        Self {
            counters: ShardedIndexerCounters::new(),
            timing: ShardedIndexerTiming::new(),
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
    /// Counter of suspicious-but-valid KV events.
    #[cfg(feature = "metrics")]
    pub kv_cache_event_warnings: IntCounterVec,
    /// Counters for CKF provenance-fallback behavior.
    #[cfg(feature = "metrics")]
    pub ckf_search_fallback: IntCounterVec,
    /// Counters for CKF mutation outcomes that are finer-grained than event status.
    #[cfg(feature = "metrics")]
    pub ckf_mutation: IntCounterVec,
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";
pub const METRIC_STATUS_CAPACITY_EXHAUSTED: &str = "capacity_exhausted";
pub const METRIC_STATUS_INDEXER_INVARIANT_VIOLATION: &str = "indexer_invariant_violation";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

/// Metric warning labels.
pub const METRIC_WARNING_DUPLICATE_STORE: &str = "duplicate_store";

/// CKF search-fallback metric labels.
pub const METRIC_CKF_FALLBACK_LEFT_EDGE_LANES: &str = "left_edge_lanes";
pub const METRIC_CKF_FALLBACK_ACTIVATED_LANES: &str = "activated_lanes";
pub const METRIC_CKF_FALLBACK_PROBE_CALLS: &str = "probe_calls";
pub const METRIC_CKF_FALLBACK_LANE_PROBES: &str = "lane_probes";
pub const METRIC_CKF_FALLBACK_PROVENANCE_SKIPS: &str = "provenance_skips";

/// CKF mutation metric labels.
pub const METRIC_CKF_MUTATION_UNKNOWN_REMOVE: &str = "unknown_remove";
pub const METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED: &str = "capacity_exhausted";

/// Metric name for KV cache events applied counter.
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const KV_CACHE_EVENTS_APPLIED_SUFFIX: &str = "kv_cache_events_applied";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_NAME: &str = "dynamo_kvrouter_kv_cache_events_applied";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_HELP: &str = "Total number of KV cache events applied to index";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENTS_APPLIED_LABELS: &[&str] = &["event_type", "status"];
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const KV_CACHE_EVENT_WARNINGS_SUFFIX: &str = "kv_cache_event_warnings";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENT_WARNINGS_NAME: &str = "dynamo_kvrouter_kv_cache_event_warnings";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENT_WARNINGS_HELP: &str =
    "Total number of suspicious KV cache events seen by the router indexer";
#[cfg(feature = "metrics")]
const KV_CACHE_EVENT_WARNINGS_LABELS: &[&str] = &["warning_kind"];
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const CKF_SEARCH_FALLBACK_SUFFIX: &str = "ckf_search_fallback";
#[cfg(feature = "metrics")]
const CKF_SEARCH_FALLBACK_NAME: &str = "dynamo_kvrouter_ckf_search_fallback";
#[cfg(feature = "metrics")]
const CKF_SEARCH_FALLBACK_HELP: &str = "CKF provenance-fallback activity and grouped probe cost";
#[cfg(feature = "metrics")]
const CKF_SEARCH_FALLBACK_LABELS: &[&str] = &["kind"];
#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
const CKF_MUTATION_SUFFIX: &str = "ckf_mutation_total";
#[cfg(feature = "metrics")]
const CKF_MUTATION_NAME: &str = "dynamo_kvrouter_ckf_mutation_total";
#[cfg(feature = "metrics")]
const CKF_MUTATION_HELP: &str = "Total number of CKF block-level mutation outcomes";
#[cfg(feature = "metrics")]
const CKF_MUTATION_LABELS: &[&str] = &["outcome"];

#[cfg(all(feature = "metrics", feature = "runtime-protocols"))]
static KV_INDEXER_METRICS: OnceLock<Arc<KvIndexerMetrics>> = OnceLock::new();

impl KvIndexerMetrics {
    #[cfg(feature = "metrics")]
    fn new(
        kv_cache_events_applied: IntCounterVec,
        kv_cache_event_warnings: IntCounterVec,
        ckf_search_fallback: IntCounterVec,
        ckf_mutation: IntCounterVec,
    ) -> Self {
        Self {
            kv_cache_events_applied,
            kv_cache_event_warnings,
            ckf_search_fallback,
            ckf_mutation,
        }
    }

    #[cfg(feature = "metrics")]
    fn new_prometheus() -> Result<Self, prometheus::Error> {
        Ok(Self::new(
            IntCounterVec::new(
                Opts::new(KV_CACHE_EVENTS_APPLIED_NAME, KV_CACHE_EVENTS_APPLIED_HELP),
                KV_CACHE_EVENTS_APPLIED_LABELS,
            )?,
            IntCounterVec::new(
                Opts::new(KV_CACHE_EVENT_WARNINGS_NAME, KV_CACHE_EVENT_WARNINGS_HELP),
                KV_CACHE_EVENT_WARNINGS_LABELS,
            )?,
            IntCounterVec::new(
                Opts::new(CKF_SEARCH_FALLBACK_NAME, CKF_SEARCH_FALLBACK_HELP),
                CKF_SEARCH_FALLBACK_LABELS,
            )?,
            IntCounterVec::new(
                Opts::new(CKF_MUTATION_NAME, CKF_MUTATION_HELP),
                CKF_MUTATION_LABELS,
            )?,
        ))
    }

    /// Creates and registers a shared metrics instance in `registry`.
    #[cfg(feature = "metrics")]
    pub fn new_registered(registry: &prometheus::Registry) -> Result<Arc<Self>, prometheus::Error> {
        let metrics = Arc::new(Self::new_prometheus()?);
        registry.register(Box::new(metrics.kv_cache_events_applied.clone()))?;
        registry.register(Box::new(metrics.kv_cache_event_warnings.clone()))?;
        registry.register(Box::new(metrics.ckf_search_fallback.clone()))?;
        registry.register(Box::new(metrics.ckf_mutation.clone()))?;
        Ok(metrics)
    }

    /// Creates a new KvIndexerMetrics from a Component, memoizing the result in
    /// KV_INDEXER_METRICS to avoid duplicate registration issues.
    #[cfg(feature = "runtime-protocols")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        #[cfg(feature = "metrics")]
        {
            KV_INDEXER_METRICS
                .get_or_init(|| {
                    match (
                        component.metrics().create_intcountervec(
                            KV_CACHE_EVENTS_APPLIED_SUFFIX,
                            KV_CACHE_EVENTS_APPLIED_HELP,
                            KV_CACHE_EVENTS_APPLIED_LABELS,
                            &[],
                        ),
                        component.metrics().create_intcountervec(
                            KV_CACHE_EVENT_WARNINGS_SUFFIX,
                            KV_CACHE_EVENT_WARNINGS_HELP,
                            KV_CACHE_EVENT_WARNINGS_LABELS,
                            &[],
                        ),
                        component.metrics().create_intcountervec(
                            CKF_SEARCH_FALLBACK_SUFFIX,
                            CKF_SEARCH_FALLBACK_HELP,
                            CKF_SEARCH_FALLBACK_LABELS,
                            &[],
                        ),
                        component.metrics().create_intcountervec(
                            CKF_MUTATION_SUFFIX,
                            CKF_MUTATION_HELP,
                            CKF_MUTATION_LABELS,
                            &[],
                        ),
                    ) {
                        (
                            Ok(kv_cache_events_applied),
                            Ok(kv_cache_event_warnings),
                            Ok(ckf_search_fallback),
                            Ok(ckf_mutation),
                        ) => Arc::new(Self::new(
                            kv_cache_events_applied,
                            kv_cache_event_warnings,
                            ckf_search_fallback,
                            ckf_mutation,
                        )),
                        (Err(e), _, _, _)
                        | (_, Err(e), _, _)
                        | (_, _, Err(e), _)
                        | (_, _, _, Err(e)) => {
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
        Self::new_prometheus().expect("valid KV indexer metric definitions")
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
                        KvCacheEventError::CapacityExhausted => METRIC_STATUS_CAPACITY_EXHAUSTED,
                        KvCacheEventError::IndexerInvariantViolation => {
                            METRIC_STATUS_INDEXER_INVARIANT_VIOLATION
                        }
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

    pub fn increment_event_warning(&self, warning_kind: &'static str) {
        #[cfg(feature = "metrics")]
        {
            self.kv_cache_event_warnings
                .with_label_values(&[warning_kind])
                .inc_by(1);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, warning_kind);
    }

    /// Pre-resolve all `IntCounter` handles for the finite (event_type, status) label space.
    /// Call this once per worker thread at startup, then use
    /// [`PreBoundEventCounters::inc`] in the hot loop to avoid the
    /// `with_label_values` hashmap lookup on every event.
    pub fn prebind(&self) -> PreBoundEventCounters {
        PreBoundEventCounters::new(self)
    }

    #[cfg(feature = "metrics")]
    pub(crate) fn prebind_ckf_search(&self) -> PreBoundCkfSearchCounters {
        PreBoundCkfSearchCounters::new(&self.ckf_search_fallback)
    }
}

#[cfg(feature = "metrics")]
#[derive(Clone, Debug)]
pub(crate) struct PreBoundCkfSearchCounters {
    left_edge_lanes: IntCounter,
    activated_lanes: IntCounter,
    probe_calls: IntCounter,
    lane_probes: IntCounter,
    provenance_skips: IntCounter,
}

#[cfg(feature = "metrics")]
impl PreBoundCkfSearchCounters {
    fn new(counters: &IntCounterVec) -> Self {
        Self {
            left_edge_lanes: counters.with_label_values(&[METRIC_CKF_FALLBACK_LEFT_EDGE_LANES]),
            activated_lanes: counters.with_label_values(&[METRIC_CKF_FALLBACK_ACTIVATED_LANES]),
            probe_calls: counters.with_label_values(&[METRIC_CKF_FALLBACK_PROBE_CALLS]),
            lane_probes: counters.with_label_values(&[METRIC_CKF_FALLBACK_LANE_PROBES]),
            provenance_skips: counters.with_label_values(&[METRIC_CKF_FALLBACK_PROVENANCE_SKIPS]),
        }
    }

    pub(crate) fn record(
        &self,
        left_edge_lanes: u64,
        activated_lanes: u64,
        probe_calls: u64,
        lane_probes: u64,
        provenance_skips: u64,
    ) {
        if left_edge_lanes | activated_lanes | probe_calls | lane_probes | provenance_skips == 0 {
            return;
        }
        if left_edge_lanes != 0 {
            self.left_edge_lanes.inc_by(left_edge_lanes);
        }
        if activated_lanes != 0 {
            self.activated_lanes.inc_by(activated_lanes);
        }
        if probe_calls != 0 {
            self.probe_calls.inc_by(probe_calls);
        }
        if lane_probes != 0 {
            self.lane_probes.inc_by(lane_probes);
        }
        if provenance_skips != 0 {
            self.provenance_skips.inc_by(provenance_skips);
        }
    }
}

/// Pre-resolved `IntCounter` handles for every (event_type, status) combination.
///
/// Created once per worker thread via [`KvIndexerMetrics::prebind`], then used in
/// the event processing loop with a direct `.inc()` call instead of the
/// `IntCounterVec::with_label_values()` hashmap lookup.
pub struct PreBoundEventCounters {
    #[cfg(feature = "metrics")]
    inner: PreBoundMetricCounters,
}

#[cfg(feature = "metrics")]
struct PreBoundMetricCounters {
    stored: ResultCounters,
    removed: ResultCounters,
    cleared: ResultCounters,
    duplicate_store_warning: IntCounter,
    ckf_mutation: CkfMutationCounters,
}

#[cfg(feature = "metrics")]
struct CkfMutationCounters {
    unknown_remove: IntCounter,
    capacity_exhausted: IntCounter,
}

#[cfg(feature = "metrics")]
struct ResultCounters {
    ok: IntCounter,
    parent_not_found: IntCounter,
    block_not_found: IntCounter,
    invalid_block: IntCounter,
    capacity_exhausted: IntCounter,
    indexer_invariant_violation: IntCounter,
}

#[cfg(feature = "metrics")]
impl ResultCounters {
    fn new(counters: &IntCounterVec, event_type: &'static str) -> Self {
        Self {
            ok: counters.with_label_values(&[event_type, METRIC_STATUS_OK]),
            parent_not_found: counters
                .with_label_values(&[event_type, METRIC_STATUS_PARENT_NOT_FOUND]),
            block_not_found: counters
                .with_label_values(&[event_type, METRIC_STATUS_BLOCK_NOT_FOUND]),
            invalid_block: counters.with_label_values(&[event_type, METRIC_STATUS_INVALID_BLOCK]),
            capacity_exhausted: counters
                .with_label_values(&[event_type, METRIC_STATUS_CAPACITY_EXHAUSTED]),
            indexer_invariant_violation: counters
                .with_label_values(&[event_type, METRIC_STATUS_INDEXER_INVARIANT_VIOLATION]),
        }
    }

    fn for_result(&self, result: Result<(), KvCacheEventError>) -> &IntCounter {
        match result {
            Ok(()) => &self.ok,
            Err(KvCacheEventError::ParentBlockNotFound) => &self.parent_not_found,
            Err(KvCacheEventError::BlockNotFound) => &self.block_not_found,
            Err(KvCacheEventError::InvalidBlockSequence) => &self.invalid_block,
            Err(KvCacheEventError::CapacityExhausted) => &self.capacity_exhausted,
            Err(KvCacheEventError::IndexerInvariantViolation) => &self.indexer_invariant_violation,
        }
    }
}

impl PreBoundEventCounters {
    fn new(metrics: &KvIndexerMetrics) -> Self {
        #[cfg(feature = "metrics")]
        {
            let cv = &metrics.kv_cache_events_applied;
            let warnings = &metrics.kv_cache_event_warnings;
            Self {
                inner: PreBoundMetricCounters {
                    stored: ResultCounters::new(cv, METRIC_EVENT_STORED),
                    removed: ResultCounters::new(cv, METRIC_EVENT_REMOVED),
                    cleared: ResultCounters::new(cv, METRIC_EVENT_CLEARED),
                    duplicate_store_warning: warnings
                        .with_label_values(&[METRIC_WARNING_DUPLICATE_STORE]),
                    ckf_mutation: CkfMutationCounters {
                        unknown_remove: metrics
                            .ckf_mutation
                            .with_label_values(&[METRIC_CKF_MUTATION_UNKNOWN_REMOVE]),
                        capacity_exhausted: metrics
                            .ckf_mutation
                            .with_label_values(&[METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED]),
                    },
                },
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
            let counters = match kind {
                EventKind::Stored => &self.inner.stored,
                EventKind::Removed => &self.inner.removed,
                EventKind::Cleared => &self.inner.cleared,
            };
            counters.for_result(result).inc();
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, kind, result);
    }

    pub fn inc_warning(&self, kind: EventWarningKind) {
        #[cfg(feature = "metrics")]
        {
            let counter = match kind {
                EventWarningKind::DuplicateStore => &self.inner.duplicate_store_warning,
            };
            counter.inc();
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, kind);
    }

    pub fn inc_ckf_mutation(&self, kind: CkfMutationKind, count: u64) {
        #[cfg(feature = "metrics")]
        {
            if count == 0 {
                return;
            }
            let counter = match kind {
                CkfMutationKind::UnknownRemove => &self.inner.ckf_mutation.unknown_remove,
                CkfMutationKind::CapacityExhausted => &self.inner.ckf_mutation.capacity_exhausted,
            };
            counter.inc_by(count);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = (self, kind, count);
    }
}
