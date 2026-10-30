// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "export")]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "export")]
use prometheus::{IntCounter, IntCounterVec, Opts};

use dynamo_kv_events::event::{KvCacheEventData, KvCacheEventError, RouterEvent};

static IGNORED_RESIDENCY_EVENT_COUNT: AtomicU64 = AtomicU64::new(0);

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

    pub fn label(self) -> &'static str {
        match self {
            Self::Stored => METRIC_EVENT_STORED,
            Self::Removed => METRIC_EVENT_REMOVED,
            Self::Cleared => METRIC_EVENT_CLEARED,
        }
    }
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Record and rate-limit diagnostics for a wire event that cannot enter the
/// canonical residency model. The metric label has fixed cardinality.
pub fn record_unsupported_residency_event(metrics: Option<&KvIndexerMetrics>, event: &RouterEvent) {
    if let Some(metrics) = metrics {
        metrics.increment_event_applied(
            EventKind::of(&event.event.data).label(),
            Err(KvCacheEventError::UnsupportedResidencyDomain),
        );
    }

    let ignored_count = IGNORED_RESIDENCY_EVENT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if ignored_count.is_power_of_two() {
        tracing::warn!(
            ignored_count,
            worker_id = event.worker_id,
            dp_rank = event.event.dp_rank,
            storage_tier = ?event.storage_tier,
            "Ignoring KV event with unsupported residency domain"
        );
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
pub struct ShardedIndexerCounters {
    pub find_match_dispatches: AtomicU64,
    pub find_match_early_returns: AtomicU64,
    pub anchor_installs: AtomicU64,
    pub anchor_reuses: AtomicU64,
    pub remove_broadcasts: AtomicU64,
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
pub struct ShardedIndexerTiming {
    pub calls: AtomicU64,
    pub routing_ns: AtomicU64,
    pub shard_ns: AtomicU64,
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
pub struct ShardedIndexerMetrics {
    pub counters: ShardedIndexerCounters,
    pub timing: ShardedIndexerTiming,
}

#[cfg(feature = "bench")]
impl Default for ShardedIndexerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bench")]
impl ShardedIndexerMetrics {
    pub fn new() -> Self {
        Self {
            counters: ShardedIndexerCounters::new(),
            timing: ShardedIndexerTiming::new(),
        }
    }
}

/// Metrics for the KV Indexer.
#[derive(Clone)]
#[cfg_attr(not(feature = "export"), derive(Default))]
pub struct KvIndexerMetrics {
    /// Counter of events applied.
    #[cfg(feature = "export")]
    pub kv_cache_events_applied: IntCounterVec,
    /// Counter of suspicious-but-valid KV events.
    #[cfg(feature = "export")]
    pub kv_cache_event_warnings: IntCounterVec,
    /// Counters for CKF mutation outcomes that are finer-grained than event status.
    #[cfg(feature = "export")]
    pub ckf_mutation: IntCounterVec,
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";
pub const METRIC_STATUS_CAPACITY_EXHAUSTED: &str = "capacity_exhausted";
pub const METRIC_STATUS_ALLOCATION_FAILED: &str = "allocation_failed";
pub const METRIC_STATUS_OWNERSHIP_DEGREE_OVERFLOW: &str = "ownership_degree_overflow";
pub const METRIC_STATUS_INDEXER_INVARIANT_VIOLATION: &str = "indexer_invariant_violation";
pub const METRIC_STATUS_UNSUPPORTED_RESIDENCY_DOMAIN: &str = "unsupported_residency_domain";
/// An error kind this crate does not know: `KvCacheEventError` is non-exhaustive and lives in
/// `dynamo-kv-events`, so a newer event crate can report kinds this indexer predates.
pub const METRIC_STATUS_UNKNOWN: &str = "unknown";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

/// Metric warning labels.
pub const METRIC_WARNING_DUPLICATE_STORE: &str = "duplicate_store";

/// CKF mutation metric labels.
pub const METRIC_CKF_MUTATION_UNKNOWN_REMOVE: &str = "unknown_remove";
pub const METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED: &str = "capacity_exhausted";

/// Metric name for KV cache events applied counter.
#[cfg(feature = "export")]
const KV_CACHE_EVENTS_APPLIED_NAME: &str = "dynamo_kvrouter_kv_cache_events_applied";
#[cfg(feature = "export")]
pub const KV_CACHE_EVENTS_APPLIED_HELP: &str = "Total number of KV cache events applied to index";
#[cfg(feature = "export")]
pub const KV_CACHE_EVENTS_APPLIED_LABELS: &[&str] = &["event_type", "status"];
#[cfg(feature = "export")]
const KV_CACHE_EVENT_WARNINGS_NAME: &str = "dynamo_kvrouter_kv_cache_event_warnings";
#[cfg(feature = "export")]
pub const KV_CACHE_EVENT_WARNINGS_HELP: &str =
    "Total number of suspicious KV cache events seen by the router indexer";
#[cfg(feature = "export")]
pub const KV_CACHE_EVENT_WARNINGS_LABELS: &[&str] = &["warning_kind"];
#[cfg(feature = "export")]
const CKF_MUTATION_NAME: &str = "dynamo_kvrouter_ckf_mutation_total";
#[cfg(feature = "export")]
pub const CKF_MUTATION_HELP: &str = "Total number of CKF block-level mutation outcomes";
#[cfg(feature = "export")]
pub const CKF_MUTATION_LABELS: &[&str] = &["outcome"];

impl KvIndexerMetrics {
    /// Build from counter vectors a host created and registered itself.
    #[cfg(feature = "export")]
    pub fn from_counters(
        kv_cache_events_applied: IntCounterVec,
        kv_cache_event_warnings: IntCounterVec,
        ckf_mutation: IntCounterVec,
    ) -> Self {
        Self {
            kv_cache_events_applied,
            kv_cache_event_warnings,
            ckf_mutation,
        }
    }

    #[cfg(feature = "export")]
    fn new_prometheus() -> Result<Self, prometheus::Error> {
        Ok(Self::from_counters(
            IntCounterVec::new(
                Opts::new(KV_CACHE_EVENTS_APPLIED_NAME, KV_CACHE_EVENTS_APPLIED_HELP),
                KV_CACHE_EVENTS_APPLIED_LABELS,
            )?,
            IntCounterVec::new(
                Opts::new(KV_CACHE_EVENT_WARNINGS_NAME, KV_CACHE_EVENT_WARNINGS_HELP),
                KV_CACHE_EVENT_WARNINGS_LABELS,
            )?,
            IntCounterVec::new(
                Opts::new(CKF_MUTATION_NAME, CKF_MUTATION_HELP),
                CKF_MUTATION_LABELS,
            )?,
        ))
    }

    /// Creates and registers a shared metrics instance in `registry`.
    #[cfg(feature = "export")]
    pub fn new_registered(registry: &prometheus::Registry) -> Result<Arc<Self>, prometheus::Error> {
        let metrics = Arc::new(Self::new_prometheus()?);
        registry.register(Box::new(metrics.kv_cache_events_applied.clone()))?;
        registry.register(Box::new(metrics.kv_cache_event_warnings.clone()))?;
        registry.register(Box::new(metrics.ckf_mutation.clone()))?;
        Ok(metrics)
    }

    /// Creates a new KvIndexerMetrics which is not registered with a MetricsRegistry.
    /// This may be used for tests or as a fallback for when a MetricsRegistry is not available / has errored.
    #[cfg(feature = "export")]
    pub fn new_unregistered() -> Self {
        Self::new_prometheus().expect("valid KV indexer metric definitions")
    }

    /// Creates a no-op metrics instance when Prometheus support is disabled.
    #[cfg(not(feature = "export"))]
    pub fn new_unregistered() -> Self {
        Self::default()
    }

    pub fn increment_event_applied(
        &self,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) {
        #[cfg(feature = "export")]
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
                        KvCacheEventError::AllocationFailed => METRIC_STATUS_ALLOCATION_FAILED,
                        KvCacheEventError::OwnershipDegreeOverflow => {
                            METRIC_STATUS_OWNERSHIP_DEGREE_OVERFLOW
                        }
                        KvCacheEventError::IndexerInvariantViolation => {
                            METRIC_STATUS_INDEXER_INVARIANT_VIOLATION
                        }
                        KvCacheEventError::UnsupportedResidencyDomain => {
                            METRIC_STATUS_UNSUPPORTED_RESIDENCY_DOMAIN
                        }
                        _ => METRIC_STATUS_UNKNOWN,
                    };
                    self.kv_cache_events_applied
                        .with_label_values(&[event_type, error_label])
                        .inc_by(1);
                }
            }
        }
        #[cfg(not(feature = "export"))]
        let _ = (self, event_type, result);
    }

    pub fn increment_event_warning(&self, warning_kind: &'static str) {
        #[cfg(feature = "export")]
        {
            self.kv_cache_event_warnings
                .with_label_values(&[warning_kind])
                .inc_by(1);
        }
        #[cfg(not(feature = "export"))]
        let _ = (self, warning_kind);
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
    #[cfg(feature = "export")]
    inner: PreBoundMetricCounters,
}

#[cfg(feature = "export")]
struct PreBoundMetricCounters {
    stored: ResultCounters,
    removed: ResultCounters,
    cleared: ResultCounters,
    duplicate_store_warning: IntCounter,
    ckf_mutation: CkfMutationCounters,
}

#[cfg(feature = "export")]
struct CkfMutationCounters {
    unknown_remove: IntCounter,
    capacity_exhausted: IntCounter,
}

#[cfg(feature = "export")]
struct ResultCounters {
    ok: IntCounter,
    parent_not_found: IntCounter,
    block_not_found: IntCounter,
    invalid_block: IntCounter,
    capacity_exhausted: IntCounter,
    allocation_failed: IntCounter,
    ownership_degree_overflow: IntCounter,
    indexer_invariant_violation: IntCounter,
    unsupported_residency_domain: IntCounter,
    unknown: IntCounter,
}

#[cfg(feature = "export")]
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
            allocation_failed: counters
                .with_label_values(&[event_type, METRIC_STATUS_ALLOCATION_FAILED]),
            ownership_degree_overflow: counters
                .with_label_values(&[event_type, METRIC_STATUS_OWNERSHIP_DEGREE_OVERFLOW]),
            indexer_invariant_violation: counters
                .with_label_values(&[event_type, METRIC_STATUS_INDEXER_INVARIANT_VIOLATION]),
            unsupported_residency_domain: counters
                .with_label_values(&[event_type, METRIC_STATUS_UNSUPPORTED_RESIDENCY_DOMAIN]),
            unknown: counters.with_label_values(&[event_type, METRIC_STATUS_UNKNOWN]),
        }
    }

    fn for_result(&self, result: Result<(), KvCacheEventError>) -> &IntCounter {
        match result {
            Ok(()) => &self.ok,
            Err(KvCacheEventError::ParentBlockNotFound) => &self.parent_not_found,
            Err(KvCacheEventError::BlockNotFound) => &self.block_not_found,
            Err(KvCacheEventError::InvalidBlockSequence) => &self.invalid_block,
            Err(KvCacheEventError::CapacityExhausted) => &self.capacity_exhausted,
            Err(KvCacheEventError::AllocationFailed) => &self.allocation_failed,
            Err(KvCacheEventError::OwnershipDegreeOverflow) => &self.ownership_degree_overflow,
            Err(KvCacheEventError::IndexerInvariantViolation) => &self.indexer_invariant_violation,
            Err(KvCacheEventError::UnsupportedResidencyDomain) => {
                &self.unsupported_residency_domain
            }
            Err(_) => &self.unknown,
        }
    }
}

impl PreBoundEventCounters {
    fn new(metrics: &KvIndexerMetrics) -> Self {
        #[cfg(feature = "export")]
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
        #[cfg(not(feature = "export"))]
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
        #[cfg(feature = "export")]
        {
            let counters = match kind {
                EventKind::Stored => &self.inner.stored,
                EventKind::Removed => &self.inner.removed,
                EventKind::Cleared => &self.inner.cleared,
            };
            counters.for_result(result).inc();
        }
        #[cfg(not(feature = "export"))]
        let _ = (self, kind, result);
    }

    pub fn inc_warning(&self, kind: EventWarningKind) {
        #[cfg(feature = "export")]
        {
            let counter = match kind {
                EventWarningKind::DuplicateStore => &self.inner.duplicate_store_warning,
            };
            counter.inc();
        }
        #[cfg(not(feature = "export"))]
        let _ = (self, kind);
    }

    pub fn inc_ckf_mutation(&self, kind: CkfMutationKind, count: u64) {
        #[cfg(feature = "export")]
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
        #[cfg(not(feature = "export"))]
        let _ = (self, kind, count);
    }
}
