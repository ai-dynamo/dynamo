// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, ensure};
use blake3::Hasher;
use serde::Serialize;
use uuid::Uuid;

use crate::engine::{
    PressureEvent as EnginePressureEvent, PressureKind as EnginePressureKind,
    PressureState as SchedulerPressureState,
};

use crate::replay::{ReplayCaptureOptions, TraceCollector};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPool {
    Agg,
    Prefill,
    Decode,
}

impl WorkerPool {
    const fn tag(self) -> u8 {
        match self {
            Self::Agg => 0,
            Self::Prefill => 1,
            Self::Decode => 2,
        }
    }

    const fn as_str(self) -> &'static str {
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

/// Replay boundary at which one batch of native KV observations was ingested.
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
    const fn tag(self) -> u8 {
        match self {
            Self::PassStart => 0,
            Self::PassEnd => 1,
            Self::SchedulerCommand => 2,
            Self::OffloadTick => 3,
            Self::WorkerLifecycle => 4,
        }
    }

    const fn as_str(self) -> &'static str {
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

/// Evidence owned and returned by one replay execution.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OfflineRuntimeEvidence {
    pub lifecycle_operations: Vec<LifecycleOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pressure: Option<PressureEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_ingest: Option<KvIngestEvidence>,
}

/// Explicit execution-local evidence state. It deliberately is not stored in
/// a thread-local: two replays on one thread cannot observe or overwrite each
/// other's capture state.
#[derive(Debug)]
pub(crate) struct ReplayEvidenceCollector {
    options: ReplayCaptureOptions,
    lifecycle_operations: Vec<LifecycleOperation>,
    pressure_records: Vec<PressureRecord>,
    kv_ingest: Option<KvIngestAccumulator>,
    outstanding_pressure: BTreeMap<(Uuid, WorkerPool), Vec<u64>>,
    startup_origins: BTreeMap<(WorkerPool, usize), u64>,
    drain_origins: BTreeMap<(WorkerPool, usize), u64>,
}

impl ReplayEvidenceCollector {
    pub(crate) fn new(options: ReplayCaptureOptions) -> Self {
        Self {
            options,
            lifecycle_operations: Vec::new(),
            pressure_records: Vec::new(),
            kv_ingest: options
                .capture_canonical_evidence
                .then(KvIngestAccumulator::new),
            outstanding_pressure: BTreeMap::new(),
            startup_origins: BTreeMap::new(),
            drain_origins: BTreeMap::new(),
        }
    }

    pub(crate) fn options(&self) -> ReplayCaptureOptions {
        self.options
    }

    pub(crate) fn startup_origin(&self, pool: WorkerPool, worker_id: usize) -> Option<u64> {
        self.startup_origins.get(&(pool, worker_id)).copied()
    }

    pub(crate) fn drain_origin(&self, pool: WorkerPool, worker_id: usize) -> Option<u64> {
        self.drain_origins.get(&(pool, worker_id)).copied()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_lifecycle_operation(
        &mut self,
        at_ms: f64,
        pool: WorkerPool,
        cause: &'static str,
        planner_tick_ordinal: Option<u64>,
        origin_operation_ordinal: Option<u64>,
        mut transitions: Vec<WorkerLifecycleTransition>,
        state_after_batch: WorkerPoolState,
        topology_released_request_uuids: Vec<Uuid>,
    ) -> Option<u64> {
        if !self.options.capture_lifecycle_evidence
            || (transitions.is_empty() && topology_released_request_uuids.is_empty())
        {
            return None;
        }

        let operation_ordinal = u64::try_from(self.lifecycle_operations.len()).ok()?;
        for transition in &mut transitions {
            if transition.origin_operation_ordinal.is_none() {
                transition.origin_operation_ordinal = Some(operation_ordinal);
            }
            let key = (pool, transition.worker_id);
            match transition.transition {
                WorkerLifecycleTransitionKind::WorkerStarting => {
                    self.startup_origins.insert(key, operation_ordinal);
                }
                WorkerLifecycleTransitionKind::WorkerDraining => {
                    self.drain_origins.insert(key, operation_ordinal);
                }
                WorkerLifecycleTransitionKind::WorkerReady => {
                    self.startup_origins.remove(&key);
                }
                WorkerLifecycleTransitionKind::WorkerRemoved => {
                    self.startup_origins.remove(&key);
                    self.drain_origins.remove(&key);
                }
            }
        }
        let mut seen = BTreeSet::new();
        self.lifecycle_operations.push(LifecycleOperation {
            operation_ordinal,
            at_ms,
            pool,
            cause,
            planner_tick_ordinal,
            origin_operation_ordinal,
            transitions,
            state_after_batch,
            topology_released_request_uuids: topology_released_request_uuids
                .into_iter()
                .filter(|uuid| seen.insert(*uuid))
                .map(|uuid| uuid.to_string())
                .collect(),
        });
        Some(operation_ordinal)
    }

    /// Record one engine pressure transition and attach its stable ordinal to
    /// the matching per-request report record.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pressure(
        &mut self,
        collector: &mut TraceCollector,
        at_ms: f64,
        pool: WorkerPool,
        worker_id: u64,
        dp_rank: u32,
        kind: PressureKind,
        request_uuid: Uuid,
        state_before: EnginePressureState,
        state_after: EnginePressureState,
        request_active_blocks_before: usize,
        logical_available_blocks_before: Option<usize>,
        required_blocks_before: Option<usize>,
    ) -> Option<u64> {
        if !self.options.capture_canonical_evidence {
            return None;
        }
        let pressure_ordinal = u64::try_from(self.pressure_records.len()).ok()?;
        self.pressure_records.push(PressureRecord {
            pressure_ordinal,
            at_ms,
            pool,
            worker_id,
            dp_rank,
            kind,
            request_uuid: request_uuid.to_string(),
            state_before,
            state_after,
            request_active_blocks_before,
            logical_available_blocks_before,
            required_blocks_before,
            readmitted_at_ms: None,
        });
        self.outstanding_pressure
            .entry((request_uuid, pool))
            .or_default()
            .push(pressure_ordinal);
        collector.on_pressure_reference(request_uuid, pressure_ordinal);
        Some(pressure_ordinal)
    }

    pub(crate) fn record_native_pressure(
        &mut self,
        collector: &mut TraceCollector,
        pool: WorkerPool,
        worker_id: u64,
        dp_rank: u32,
        event: EnginePressureEvent,
    ) -> Option<u64> {
        let kind = match event.kind {
            EnginePressureKind::VllmPreemption => PressureKind::VllmPreemption,
            EnginePressureKind::SglangRetraction => PressureKind::SglangRetraction,
        };
        self.record_pressure(
            collector,
            event.at_ms,
            pool,
            worker_id,
            dp_rank,
            kind,
            event.request_id,
            lower_pressure_state(event.state_before),
            lower_pressure_state(event.state_after),
            event.request_active_blocks_before,
            event.logical_available_blocks_before,
            event.required_blocks_before,
        )
    }

    pub(crate) fn record_pressure_readmission(
        &mut self,
        request_uuid: Uuid,
        pool: WorkerPool,
        at_ms: f64,
    ) {
        if !self.options.capture_canonical_evidence {
            return;
        }
        let key = (request_uuid, pool);
        let Some(ordinals) = self.outstanding_pressure.get_mut(&key) else {
            return;
        };
        let Some(pressure_ordinal) = ordinals.pop() else {
            return;
        };
        let remove_key = ordinals.is_empty();
        if remove_key {
            self.outstanding_pressure.remove(&key);
        }
        if let Some(record) = self
            .pressure_records
            .get_mut(usize::try_from(pressure_ordinal).expect("pressure ordinal must fit usize"))
        {
            record.readmitted_at_ms = Some(at_ms);
        }
    }

    pub(crate) fn record_kv_ingest(
        &mut self,
        pool: WorkerPool,
        boundary: KvIngestBoundary,
        at_ms: f64,
        event_count: usize,
        encode_events: impl FnOnce(&mut KvIngestEventEncoder<'_>) -> Result<()>,
    ) -> Result<()> {
        if !self.options.capture_canonical_evidence {
            return Ok(());
        }
        ensure!(
            at_ms.is_finite(),
            "canonical KV ingestion rejects non-finite timestamp {at_ms}"
        );
        self.kv_ingest
            .as_mut()
            .expect("canonical KV accumulator was not initialized")
            .record_batch(pool, boundary, at_ms, event_count, encode_events)
    }

    pub(crate) fn finish(self) -> OfflineRuntimeEvidence {
        let Self {
            options,
            lifecycle_operations,
            pressure_records,
            kv_ingest,
            ..
        } = self;
        let pressure = options.capture_canonical_evidence.then(|| {
            let vllm_preemptions_total = pressure_records
                .iter()
                .filter(|record| record.kind == PressureKind::VllmPreemption)
                .count() as u64;
            let sglang_retractions_total = pressure_records
                .iter()
                .filter(|record| record.kind == PressureKind::SglangRetraction)
                .count() as u64;
            PressureEvidence {
                records: pressure_records,
                vllm_preemptions_total,
                sglang_retractions_total,
            }
        });
        let kv_ingest = kv_ingest.map(KvIngestAccumulator::finish);
        OfflineRuntimeEvidence {
            lifecycle_operations,
            pressure,
            kv_ingest,
        }
    }
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

/// Neutral encoder used by observation adapters to contribute KV events to
/// canonical replay evidence without exposing Router types to Replay.
pub struct KvIngestEventEncoder<'a> {
    hasher: &'a mut Hasher,
    evidence: &'a mut KvIngestEvidence,
}

impl KvIngestEventEncoder<'_> {
    pub fn begin_event(
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

    pub fn begin_kind(&mut self, tag: u8, name: &'static str) {
        put_u8(self.hasher, tag);
        increment(&mut self.evidence.kind_counts, name, 1);
    }

    pub fn add_blocks(&mut self, count: usize, context: &str) -> Result<()> {
        self.evidence.blocks = self
            .evidence
            .blocks
            .checked_add(to_u64(count, context)?)
            .expect("KV ingestion block count overflow");
        Ok(())
    }

    pub fn put_len(&mut self, value: usize, context: &str) -> Result<()> {
        put_u64(self.hasher, to_u64(value, context)?);
        Ok(())
    }

    pub fn put_u8(&mut self, value: u8) {
        put_u8(self.hasher, value);
    }

    pub fn put_u64(&mut self, value: u64) {
        put_u64(self.hasher, value);
    }

    pub fn put_optional_u32(&mut self, value: Option<u32>) {
        put_optional_u32(self.hasher, value);
    }

    pub fn put_optional_u64(&mut self, value: Option<u64>) {
        put_optional_u64(self.hasher, value);
    }
}

fn lower_pressure_state(state: SchedulerPressureState) -> EnginePressureState {
    EnginePressureState {
        running_requests: state.running_requests,
        waiting_requests: state.waiting_requests,
        active_blocks: state.active_blocks,
    }
}

impl Default for ReplayEvidenceCollector {
    fn default() -> Self {
        Self::new(ReplayCaptureOptions::default())
    }
}

pub(crate) fn common_origin(mut origins: impl Iterator<Item = u64>) -> Option<u64> {
    let first = origins.next()?;
    origins.all(|origin| origin == first).then_some(first)
}

#[cfg(test)]
mod tests {
    use crate::engine::{PressureEvent, PressureKind, PressureState};
    use uuid::Uuid;

    use crate::replay::{ReplayCaptureOptions, ReplayDeterminism, TraceCollector};

    use super::{
        KvIngestBoundary, ReplayEvidenceCollector, WorkerLifecycleTransition,
        WorkerLifecycleTransitionKind, WorkerPool, WorkerPoolState,
    };

    #[test]
    fn execution_local_pressure_ordinals_link_to_request_records() {
        let uuid = Uuid::from_u128(42);
        let mut trace = TraceCollector::default();
        trace.set_capture_per_request(true);
        trace.on_arrival(uuid, 0.0, 4, 2);
        trace.on_admit(uuid, 0.0, 0);
        let mut evidence = ReplayEvidenceCollector::new(ReplayCaptureOptions {
            capture_per_request: true,
            capture_canonical_evidence: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        });
        evidence.record_native_pressure(
            &mut trace,
            WorkerPool::Agg,
            7,
            1,
            PressureEvent {
                at_ms: 2.0,
                kind: PressureKind::VllmPreemption,
                request_id: uuid,
                state_before: PressureState {
                    running_requests: 1,
                    waiting_requests: Some(0),
                    active_blocks: 4,
                },
                state_after: PressureState {
                    running_requests: 0,
                    waiting_requests: Some(1),
                    active_blocks: 0,
                },
                request_active_blocks_before: 4,
                logical_available_blocks_before: None,
                required_blocks_before: None,
            },
        );
        evidence.record_pressure_readmission(uuid, WorkerPool::Agg, 3.0);
        trace.on_terminal(uuid, 4.0, crate::replay::ReplayTerminalStatus::Completed);
        trace.set_runtime_evidence(evidence.finish());

        let report = trace.finish();
        assert_eq!(report.per_request[0].pressure_record_ordinals, vec![0]);
        let pressure = report.runtime_evidence.pressure.unwrap();
        assert_eq!(pressure.vllm_preemptions_total, 1);
        assert_eq!(pressure.records[0].readmitted_at_ms, Some(3.0));
    }

    #[test]
    fn lifecycle_origins_are_owned_by_one_collector() {
        let mut evidence = ReplayEvidenceCollector::new(ReplayCaptureOptions {
            capture_lifecycle_evidence: true,
            ..Default::default()
        });
        evidence.record_lifecycle_operation(
            1.0,
            WorkerPool::Decode,
            "planner_scale",
            Some(0),
            None,
            vec![WorkerLifecycleTransition {
                worker_id: 3,
                transition: WorkerLifecycleTransitionKind::WorkerStarting,
                prior_state: None,
                state: "starting",
                reason: None,
                origin_operation_ordinal: None,
            }],
            WorkerPoolState {
                starting: vec![3],
                ..Default::default()
            },
            Vec::new(),
        );
        let origin = evidence.startup_origin(WorkerPool::Decode, 3);
        evidence.record_lifecycle_operation(
            2.0,
            WorkerPool::Decode,
            "worker_ready_event",
            None,
            origin,
            vec![WorkerLifecycleTransition {
                worker_id: 3,
                transition: WorkerLifecycleTransitionKind::WorkerReady,
                prior_state: Some("starting"),
                state: "active",
                reason: None,
                origin_operation_ordinal: origin,
            }],
            WorkerPoolState {
                active: vec![3],
                ..Default::default()
            },
            Vec::new(),
        );

        let evidence = evidence.finish();
        assert_eq!(evidence.lifecycle_operations.len(), 2);
        assert_eq!(
            evidence.lifecycle_operations[1].origin_operation_ordinal,
            Some(0)
        );
    }

    #[test]
    fn canonical_kv_ingest_is_execution_local_and_byte_stable() {
        let capture = || {
            let mut evidence = ReplayEvidenceCollector::new(ReplayCaptureOptions {
                capture_canonical_evidence: true,
                ..Default::default()
            });
            evidence
                .record_kv_ingest(
                    WorkerPool::Agg,
                    KvIngestBoundary::PassEnd,
                    7.0,
                    1,
                    |encoder| {
                        encoder.begin_event(3, 0, 0, "device", 9);
                        encoder.begin_kind(0, "stored");
                        encoder.add_blocks(2, "test blocks")?;
                        encoder.put_len(2, "test blocks")?;
                        encoder.put_u64(11);
                        encoder.put_u64(12);
                        Ok(())
                    },
                )
                .unwrap();
            evidence.finish().kv_ingest.unwrap()
        };

        let first = capture();
        let second = capture();
        assert_eq!(first, second);
        assert_eq!(first.batches, 1);
        assert_eq!(first.events, 1);
        assert_eq!(first.blocks, 2);
        assert_eq!(first.boundaries["pass_end"].last_at_ms, 7.0);
        assert_eq!(first.blake3_256.len(), 64);
    }
}
