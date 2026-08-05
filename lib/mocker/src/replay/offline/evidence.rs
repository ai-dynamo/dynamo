// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, ensure};
use blake3::Hasher;
use serde::Serialize;
use uuid::Uuid;

use crate::replay::{ReplayCaptureOptions, TraceCollector};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPool {
    Agg,
    Prefill,
    Decode,
}

impl WorkerPool {
    fn tag(self) -> u8 {
        match self {
            Self::Agg => 0,
            Self::Prefill => 1,
            Self::Decode => 2,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Agg => "agg",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLifecycleTransitionKind {
    WorkerStarting,
    WorkerReady,
    WorkerDraining,
    WorkerRemoved,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WorkerLifecycleTransition {
    pub worker_id: usize,
    pub transition: WorkerLifecycleTransitionKind,
    pub prior_state: Option<&'static str>,
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin_operation_ordinal: Option<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct WorkerPoolState {
    pub active: Vec<usize>,
    pub starting: Vec<usize>,
    pub draining: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LifecycleOperation {
    pub operation_ordinal: u64,
    pub at_ms: f64,
    pub pool: WorkerPool,
    pub cause: &'static str,
    pub planner_tick_ordinal: Option<u64>,
    pub origin_operation_ordinal: Option<u64>,
    pub transitions: Vec<WorkerLifecycleTransition>,
    pub state_after_batch: WorkerPoolState,
    pub topology_released_request_uuids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureKind {
    VllmPreemption,
    SglangRetraction,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct EnginePressureState {
    pub running_requests: usize,
    pub waiting_requests: Option<usize>,
    pub active_blocks: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PressureRecord {
    pub pressure_ordinal: u64,
    pub at_ms: f64,
    pub pool: WorkerPool,
    pub worker_id: u64,
    pub dp_rank: u32,
    pub kind: PressureKind,
    pub request_uuid: String,
    pub state_before: EnginePressureState,
    pub state_after: EnginePressureState,
    pub request_active_blocks_before: usize,
    pub logical_available_blocks_before: Option<usize>,
    pub required_blocks_before: Option<usize>,
    pub readmitted_at_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PressureEvidence {
    pub records: Vec<PressureRecord>,
    pub vllm_preemptions_total: u64,
    pub sglang_retractions_total: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvIngestBoundary {
    PassStart,
    PassEnd,
    SchedulerCommand,
    OffloadTick,
    WorkerLifecycle,
}

impl KvIngestBoundary {
    fn tag(self) -> u8 {
        match self {
            Self::PassStart => 0,
            Self::PassEnd => 1,
            Self::SchedulerCommand => 2,
            Self::OffloadTick => 3,
            Self::WorkerLifecycle => 4,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::PassStart => "pass_start",
            Self::PassEnd => "pass_end",
            Self::SchedulerCommand => "scheduler_command",
            Self::OffloadTick => "offload_tick",
            Self::WorkerLifecycle => "worker_lifecycle",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct KvIngestBoundaryStats {
    pub batches: u64,
    pub events: u64,
    pub first_at_ms: f64,
    pub last_at_ms: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct KvIngestEvidence {
    pub encoding: String,
    pub blake3_256: String,
    pub batches: u64,
    pub events: u64,
    pub blocks: u64,
    pub kind_counts: BTreeMap<String, u64>,
    pub pool_counts: BTreeMap<String, u64>,
    pub tier_counts: BTreeMap<String, u64>,
    pub boundaries: BTreeMap<String, KvIngestBoundaryStats>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OfflineRuntimeEvidence {
    pub lifecycle_operations: Vec<LifecycleOperation>,
    pub pressure: Option<PressureEvidence>,
    pub kv_ingest: Option<KvIngestEvidence>,
}

#[derive(Debug)]
struct EvidenceCollector {
    options: ReplayCaptureOptions,
    evidence: OfflineRuntimeEvidence,
    pressure_records: Vec<PressureRecord>,
    outstanding_pressure: BTreeMap<(Uuid, WorkerPool), Vec<u64>>,
    pending_pressure_references: Vec<(Uuid, u64)>,
    kv_ingest: Option<KvIngestAccumulator>,
    startup_origins: BTreeMap<(WorkerPool, usize), u64>,
    drain_origins: BTreeMap<(WorkerPool, usize), u64>,
}

impl EvidenceCollector {
    fn new(options: ReplayCaptureOptions) -> Self {
        Self {
            options,
            evidence: OfflineRuntimeEvidence::default(),
            pressure_records: Vec::new(),
            outstanding_pressure: BTreeMap::new(),
            pending_pressure_references: Vec::new(),
            kv_ingest: options
                .capture_canonical_evidence
                .then(KvIngestAccumulator::new),
            startup_origins: BTreeMap::new(),
            drain_origins: BTreeMap::new(),
        }
    }

    fn finish(mut self) -> OfflineRuntimeEvidence {
        if self.options.capture_canonical_evidence {
            let vllm_preemptions_total = self
                .pressure_records
                .iter()
                .filter(|record| record.kind == PressureKind::VllmPreemption)
                .count() as u64;
            let sglang_retractions_total = self
                .pressure_records
                .iter()
                .filter(|record| record.kind == PressureKind::SglangRetraction)
                .count() as u64;
            self.evidence.pressure = Some(PressureEvidence {
                records: self.pressure_records,
                vllm_preemptions_total,
                sglang_retractions_total,
            });
            self.evidence.kv_ingest = self.kv_ingest.map(KvIngestAccumulator::finish);
        }
        self.evidence
    }
}

thread_local! {
    static ACTIVE_EVIDENCE: RefCell<Option<EvidenceCollector>> = const { RefCell::new(None) };
    static ENGINE_EVIDENCE_CONTEXT: RefCell<Option<EngineEvidenceContext>> =
        const { RefCell::new(None) };
}

pub fn with_runtime_evidence<T>(
    options: ReplayCaptureOptions,
    run: impl FnOnce() -> T,
) -> (T, OfflineRuntimeEvidence) {
    struct Restore {
        previous: Option<EvidenceCollector>,
        restored: bool,
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            if self.restored {
                return;
            }
            ACTIVE_EVIDENCE.with(|active| {
                active.replace(self.previous.take());
            });
        }
    }

    let previous =
        ACTIVE_EVIDENCE.with(|active| active.replace(Some(EvidenceCollector::new(options))));
    let mut restore = Restore {
        previous,
        restored: false,
    };
    let result = run();
    let collector = ACTIVE_EVIDENCE.with(|active| {
        active
            .replace(restore.previous.take())
            .expect("runtime evidence scope disappeared before replay completed")
    });
    restore.restored = true;
    (result, collector.finish())
}

#[derive(Clone, Copy, Debug)]
struct EngineEvidenceContext {
    at_ms: f64,
    pool: WorkerPool,
    worker_id: u64,
    dp_rank: u32,
}

pub(crate) fn with_engine_evidence_context<T>(
    at_ms: f64,
    pool: WorkerPool,
    worker_id: u64,
    dp_rank: u32,
    run: impl FnOnce() -> T,
) -> T {
    if !canonical_evidence_capture_active() {
        return run();
    }

    struct Restore(Option<EngineEvidenceContext>);

    impl Drop for Restore {
        fn drop(&mut self) {
            ENGINE_EVIDENCE_CONTEXT.with(|context| {
                context.replace(self.0.take());
            });
        }
    }

    let previous = ENGINE_EVIDENCE_CONTEXT.with(|context| {
        context.replace(Some(EngineEvidenceContext {
            at_ms,
            pool,
            worker_id,
            dp_rank,
        }))
    });
    let _restore = Restore(previous);
    run()
}

#[inline]
pub(crate) fn with_engine_evidence_timestamp<T>(at_ms: f64, run: impl FnOnce() -> T) -> T {
    if !canonical_evidence_capture_active() {
        return run();
    }
    let Some(context) = ENGINE_EVIDENCE_CONTEXT.with(|context| *context.borrow()) else {
        return run();
    };
    with_engine_evidence_context(at_ms, context.pool, context.worker_id, context.dp_rank, run)
}

pub(crate) fn lifecycle_capture_active() -> bool {
    ACTIVE_EVIDENCE.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(|collector| collector.options.capture_planner_details)
    })
}

#[inline]
pub(crate) fn canonical_evidence_capture_active() -> bool {
    #[cfg(not(feature = "replay-bench"))]
    return false;

    #[cfg(feature = "replay-bench")]
    ACTIVE_EVIDENCE.with(|active| {
        active
            .borrow()
            .as_ref()
            .is_some_and(|collector| collector.options.capture_canonical_evidence)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_pressure(
    kind: PressureKind,
    request_uuid: Uuid,
    state_before: EnginePressureState,
    state_after: EnginePressureState,
    request_active_blocks_before: usize,
    logical_available_blocks_before: Option<usize>,
    required_blocks_before: Option<usize>,
) -> Option<u64> {
    if !canonical_evidence_capture_active() {
        return None;
    }
    let context = ENGINE_EVIDENCE_CONTEXT.with(|context| *context.borrow())?;
    ACTIVE_EVIDENCE.with(|active| {
        let mut active = active.borrow_mut();
        let collector = active.as_mut()?;
        let pressure_ordinal = u64::try_from(collector.pressure_records.len()).ok()?;
        collector.pressure_records.push(PressureRecord {
            pressure_ordinal,
            at_ms: context.at_ms,
            pool: context.pool,
            worker_id: context.worker_id,
            dp_rank: context.dp_rank,
            kind,
            request_uuid: request_uuid.to_string(),
            state_before,
            state_after,
            request_active_blocks_before,
            logical_available_blocks_before,
            required_blocks_before,
            readmitted_at_ms: None,
        });
        collector
            .outstanding_pressure
            .entry((request_uuid, context.pool))
            .or_default()
            .push(pressure_ordinal);
        collector
            .pending_pressure_references
            .push((request_uuid, pressure_ordinal));
        Some(pressure_ordinal)
    })
}

pub(crate) fn attach_pressure_references(collector: &mut TraceCollector) {
    if !canonical_evidence_capture_active() {
        return;
    }
    let references = ACTIVE_EVIDENCE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(active) = active.as_mut() else {
            return Vec::new();
        };
        std::mem::take(&mut active.pending_pressure_references)
    });
    for (uuid, ordinal) in references {
        collector.on_pressure_reference(uuid, ordinal);
    }
}

#[inline]
pub(crate) fn record_pressure_readmission(uuid: Uuid, at_ms: f64) {
    if !canonical_evidence_capture_active() {
        return;
    }
    let Some(context) = ENGINE_EVIDENCE_CONTEXT.with(|context| *context.borrow()) else {
        return;
    };
    ACTIVE_EVIDENCE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(collector) = active.as_mut() else {
            return;
        };
        let key = (uuid, context.pool);
        let (pressure_ordinal, remove_key) = {
            let Some(ordinals) = collector.outstanding_pressure.get_mut(&key) else {
                return;
            };
            let pressure_ordinal = ordinals
                .pop()
                .expect("outstanding pressure index must not contain an empty stack");
            (pressure_ordinal, ordinals.is_empty())
        };
        if remove_key {
            collector.outstanding_pressure.remove(&key);
        }
        let index = usize::try_from(pressure_ordinal)
            .expect("pressure ordinal must fit the evidence record index");
        let record = collector
            .pressure_records
            .get_mut(index)
            .expect("pressure ordinal must reference an evidence record");
        record.readmitted_at_ms = Some(at_ms);
    });
}

pub(crate) fn record_kv_ingest(
    pool: WorkerPool,
    boundary: KvIngestBoundary,
    at_ms: f64,
    event_count: usize,
    encode_events: impl FnOnce(&mut KvIngestEventEncoder<'_>) -> Result<()>,
) -> Result<()> {
    if !canonical_evidence_capture_active() {
        return Ok(());
    }
    ensure!(
        at_ms.is_finite(),
        "canonical KV ingestion rejects non-finite timestamp {at_ms}"
    );
    ACTIVE_EVIDENCE.with(|active| {
        let mut active = active.borrow_mut();
        let collector = active
            .as_mut()
            .expect("canonical evidence capture disappeared during KV ingestion");
        collector
            .kv_ingest
            .as_mut()
            .expect("canonical KV accumulator was not initialized")
            .record_batch(pool, boundary, at_ms, event_count, encode_events)
    })
}

pub(crate) fn startup_origin(pool: WorkerPool, worker_id: usize) -> Option<u64> {
    ACTIVE_EVIDENCE.with(|active| {
        active
            .borrow()
            .as_ref()?
            .startup_origins
            .get(&(pool, worker_id))
            .copied()
    })
}

pub(crate) fn drain_origin(pool: WorkerPool, worker_id: usize) -> Option<u64> {
    ACTIVE_EVIDENCE.with(|active| {
        active
            .borrow()
            .as_ref()?
            .drain_origins
            .get(&(pool, worker_id))
            .copied()
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_lifecycle_operation(
    at_ms: f64,
    pool: WorkerPool,
    cause: &'static str,
    planner_tick_ordinal: Option<u64>,
    origin_operation_ordinal: Option<u64>,
    mut transitions: Vec<WorkerLifecycleTransition>,
    state_after_batch: WorkerPoolState,
    topology_released_request_uuids: Vec<Uuid>,
) -> Option<u64> {
    ACTIVE_EVIDENCE.with(|active| {
        let mut active = active.borrow_mut();
        let collector = active.as_mut()?;
        if !collector.options.capture_planner_details
            || (transitions.is_empty() && topology_released_request_uuids.is_empty())
        {
            return None;
        }
        let operation_ordinal =
            u64::try_from(collector.evidence.lifecycle_operations.len()).ok()?;
        for transition in &mut transitions {
            if transition.origin_operation_ordinal.is_none() {
                transition.origin_operation_ordinal = Some(operation_ordinal);
            }
            let key = (pool, transition.worker_id);
            match transition.transition {
                WorkerLifecycleTransitionKind::WorkerStarting => {
                    collector.startup_origins.insert(key, operation_ordinal);
                }
                WorkerLifecycleTransitionKind::WorkerDraining => {
                    collector.drain_origins.insert(key, operation_ordinal);
                }
                WorkerLifecycleTransitionKind::WorkerReady => {
                    collector.startup_origins.remove(&key);
                }
                WorkerLifecycleTransitionKind::WorkerRemoved => {
                    collector.startup_origins.remove(&key);
                    collector.drain_origins.remove(&key);
                }
            }
        }
        collector
            .evidence
            .lifecycle_operations
            .push(LifecycleOperation {
                operation_ordinal,
                at_ms,
                pool,
                cause,
                planner_tick_ordinal,
                origin_operation_ordinal,
                transitions,
                state_after_batch,
                topology_released_request_uuids: {
                    let mut seen = BTreeSet::new();
                    topology_released_request_uuids
                        .into_iter()
                        .filter(|uuid| seen.insert(*uuid))
                        .map(|uuid| uuid.to_string())
                        .collect()
                },
            });
        Some(operation_ordinal)
    })
}

#[derive(Debug)]
struct KvIngestAccumulator {
    hasher: Hasher,
    evidence: KvIngestEvidence,
}

impl KvIngestAccumulator {
    const ENCODING: &'static str = "dynamo.offline-kv-ingest.v1";

    fn new() -> Self {
        let mut hasher = Hasher::new();
        put_bytes(&mut hasher, b"dynamo.offline-kv-ingest");
        put_u32(&mut hasher, 1);
        Self {
            hasher,
            evidence: KvIngestEvidence {
                encoding: Self::ENCODING.to_string(),
                ..KvIngestEvidence::default()
            },
        }
    }

    fn finish(mut self) -> KvIngestEvidence {
        self.evidence.blake3_256 = self.hasher.finalize().to_hex().to_string();
        self.evidence
    }

    fn record_batch(
        &mut self,
        pool: WorkerPool,
        boundary: KvIngestBoundary,
        at_ms: f64,
        event_count: usize,
        encode_events: impl FnOnce(&mut KvIngestEventEncoder<'_>) -> Result<()>,
    ) -> Result<()> {
        let batch_ordinal = self.evidence.batches;
        self.evidence.batches = self
            .evidence
            .batches
            .checked_add(1)
            .expect("KV ingestion batch count overflow");
        let event_count = to_u64(event_count, "KV event count")?;
        put_u8(&mut self.hasher, pool.tag());
        put_u8(&mut self.hasher, boundary.tag());
        put_u64(&mut self.hasher, batch_ordinal);
        put_f64(&mut self.hasher, at_ms);
        put_u64(&mut self.hasher, event_count);

        increment(&mut self.evidence.pool_counts, pool.as_str(), event_count);
        let boundary_stats = self
            .evidence
            .boundaries
            .entry(boundary.as_str().to_string())
            .or_insert_with(|| KvIngestBoundaryStats {
                first_at_ms: normalize_zero(at_ms),
                ..KvIngestBoundaryStats::default()
            });
        boundary_stats.batches += 1;
        boundary_stats.events += event_count;
        boundary_stats.last_at_ms = normalize_zero(at_ms);

        let mut encoder = KvIngestEventEncoder {
            hasher: &mut self.hasher,
            evidence: &mut self.evidence,
        };
        encode_events(&mut encoder)
    }
}

fn increment(counts: &mut BTreeMap<String, u64>, key: &str, amount: u64) {
    *counts.entry(key.to_string()).or_default() += amount;
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| anyhow::anyhow!("{context} exceeds u64"))
}

fn put_u8(hasher: &mut Hasher, value: u8) {
    hasher.update(&[value]);
}

fn put_u32(hasher: &mut Hasher, value: u32) {
    hasher.update(&value.to_be_bytes());
}

fn put_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

fn put_f64(hasher: &mut Hasher, value: f64) {
    put_u64(hasher, normalize_zero(value).to_bits());
}

fn put_bytes(hasher: &mut Hasher, bytes: &[u8]) {
    put_u64(
        hasher,
        u64::try_from(bytes.len()).expect("static KV digest domain length exceeds u64"),
    );
    hasher.update(bytes);
}

fn put_optional_u32(hasher: &mut Hasher, value: Option<u32>) {
    match value {
        Some(value) => {
            put_u8(hasher, 1);
            put_u32(hasher, value);
        }
        None => put_u8(hasher, 0),
    }
}

fn put_optional_u64(hasher: &mut Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            put_u8(hasher, 1);
            put_u64(hasher, value);
        }
        None => put_u8(hasher, 0),
    }
}

pub(crate) struct KvIngestEventEncoder<'a> {
    hasher: &'a mut Hasher,
    evidence: &'a mut KvIngestEvidence,
}

impl KvIngestEventEncoder<'_> {
    pub(crate) fn begin_event(
        &mut self,
        worker_id: u64,
        dp_rank: u32,
        storage_tier_tag: u8,
        storage_tier_name: &'static str,
        event_id: u64,
    ) {
        self.evidence.events = self
            .evidence
            .events
            .checked_add(1)
            .expect("KV ingestion event count overflow");
        put_u64(self.hasher, worker_id);
        put_u32(self.hasher, dp_rank);
        put_u8(self.hasher, storage_tier_tag);
        put_u64(self.hasher, event_id);
        increment(&mut self.evidence.tier_counts, storage_tier_name, 1);
    }

    pub(crate) fn begin_kind(&mut self, tag: u8, name: &'static str) {
        put_u8(self.hasher, tag);
        increment(&mut self.evidence.kind_counts, name, 1);
    }

    pub(crate) fn add_blocks(&mut self, count: usize, context: &str) -> Result<()> {
        self.evidence.blocks = self
            .evidence
            .blocks
            .checked_add(to_u64(count, context)?)
            .expect("KV ingestion block count overflow");
        Ok(())
    }

    pub(crate) fn put_len(&mut self, value: usize, context: &str) -> Result<()> {
        put_u64(self.hasher, to_u64(value, context)?);
        Ok(())
    }

    pub(crate) fn put_u8(&mut self, value: u8) {
        put_u8(self.hasher, value);
    }

    pub(crate) fn put_u64(&mut self, value: u64) {
        put_u64(self.hasher, value);
    }

    pub(crate) fn put_optional_u32(&mut self, value: Option<u32>) {
        put_optional_u32(self.hasher, value);
    }

    pub(crate) fn put_optional_u64(&mut self, value: Option<u64>) {
        put_optional_u64(self.hasher, value);
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::replay::ReplayCaptureOptions;

    #[cfg(feature = "replay-bench")]
    use super::{
        EnginePressureState, PressureKind, record_pressure, record_pressure_readmission,
        with_engine_evidence_context,
    };
    use super::{
        WorkerLifecycleTransition, WorkerLifecycleTransitionKind, WorkerPool, WorkerPoolState,
        record_lifecycle_operation, startup_origin, with_runtime_evidence,
    };

    fn starting(worker_id: usize) -> WorkerLifecycleTransition {
        WorkerLifecycleTransition {
            worker_id,
            transition: WorkerLifecycleTransitionKind::WorkerStarting,
            prior_state: None,
            state: "starting",
            reason: None,
            origin_operation_ordinal: None,
        }
    }

    #[test]
    fn lifecycle_preserves_per_worker_origins_and_release_only_operations() {
        let released_uuid = Uuid::from_u128(42);
        let ((), evidence) = with_runtime_evidence(
            ReplayCaptureOptions {
                capture_planner_details: true,
                ..Default::default()
            },
            || {
                record_lifecycle_operation(
                    1.0,
                    WorkerPool::Prefill,
                    "planner_scale",
                    Some(0),
                    None,
                    vec![starting(1)],
                    WorkerPoolState {
                        starting: vec![1],
                        ..Default::default()
                    },
                    Vec::new(),
                );
                record_lifecycle_operation(
                    2.0,
                    WorkerPool::Prefill,
                    "planner_scale",
                    Some(1),
                    None,
                    vec![starting(2)],
                    WorkerPoolState {
                        starting: vec![1, 2],
                        ..Default::default()
                    },
                    Vec::new(),
                );
                let first_origin = startup_origin(WorkerPool::Prefill, 1);
                let second_origin = startup_origin(WorkerPool::Prefill, 2);
                record_lifecycle_operation(
                    3.0,
                    WorkerPool::Prefill,
                    "worker_ready_event",
                    None,
                    None,
                    vec![
                        WorkerLifecycleTransition {
                            worker_id: 1,
                            transition: WorkerLifecycleTransitionKind::WorkerReady,
                            prior_state: Some("starting"),
                            state: "active",
                            reason: None,
                            origin_operation_ordinal: first_origin,
                        },
                        WorkerLifecycleTransition {
                            worker_id: 2,
                            transition: WorkerLifecycleTransitionKind::WorkerReady,
                            prior_state: Some("starting"),
                            state: "active",
                            reason: None,
                            origin_operation_ordinal: second_origin,
                        },
                    ],
                    WorkerPoolState {
                        active: vec![1, 2],
                        ..Default::default()
                    },
                    Vec::new(),
                );
                record_lifecycle_operation(
                    4.0,
                    WorkerPool::Decode,
                    "planner_scale",
                    Some(3),
                    None,
                    Vec::new(),
                    WorkerPoolState::default(),
                    vec![released_uuid],
                );
            },
        );

        assert_eq!(evidence.lifecycle_operations.len(), 4);
        assert_eq!(
            evidence.lifecycle_operations[2]
                .transitions
                .iter()
                .map(|transition| transition.origin_operation_ordinal)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1)]
        );
        let release_only = &evidence.lifecycle_operations[3];
        assert!(release_only.transitions.is_empty());
        assert_eq!(release_only.planner_tick_ordinal, Some(3));
        assert_eq!(
            release_only.topology_released_request_uuids,
            vec![released_uuid.to_string()]
        );
    }

    #[cfg(feature = "replay-bench")]
    #[test]
    fn pressure_readmission_updates_latest_outstanding_record() {
        let uuid = Uuid::from_u128(42);
        let ((), evidence) = with_runtime_evidence(
            ReplayCaptureOptions {
                capture_canonical_evidence: true,
                ..Default::default()
            },
            || {
                for at_ms in [1.0, 2.0] {
                    with_engine_evidence_context(at_ms, WorkerPool::Agg, 0, 0, || {
                        record_pressure(
                            PressureKind::VllmPreemption,
                            uuid,
                            EnginePressureState::default(),
                            EnginePressureState::default(),
                            0,
                            None,
                            None,
                        )
                    });
                }
                with_engine_evidence_context(3.0, WorkerPool::Agg, 0, 0, || {
                    record_pressure_readmission(uuid, 3.0);
                });
                with_engine_evidence_context(4.0, WorkerPool::Agg, 0, 0, || {
                    record_pressure_readmission(uuid, 4.0);
                });
            },
        );

        let records = evidence.pressure.unwrap().records;
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].readmitted_at_ms, Some(4.0));
        assert_eq!(records[1].readmitted_at_ms, Some(3.0));
    }

    #[cfg(feature = "replay-bench")]
    #[test]
    fn same_pass_pressure_readmission_waits_for_the_next_admission() {
        let uuid = Uuid::from_u128(43);
        let ((), evidence) = with_runtime_evidence(
            ReplayCaptureOptions {
                capture_canonical_evidence: true,
                ..Default::default()
            },
            || {
                with_engine_evidence_context(1.0, WorkerPool::Agg, 0, 0, || {
                    record_pressure_readmission(uuid, 1.0);
                    record_pressure(
                        PressureKind::SglangRetraction,
                        uuid,
                        EnginePressureState::default(),
                        EnginePressureState::default(),
                        0,
                        None,
                        None,
                    );
                });
                with_engine_evidence_context(2.0, WorkerPool::Agg, 0, 0, || {
                    record_pressure_readmission(uuid, 2.0);
                });
            },
        );

        let records = evidence.pressure.unwrap().records;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].readmitted_at_ms, Some(2.0));
    }
}
