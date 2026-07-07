// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{Context, ensure};
use async_trait::async_trait;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::{
    CuckooFilter, Probe, TransposedTable, apply_decoded_delta, assemble_chunks, decode_delta,
    overlap_depth_searched, probes_for,
};
use crate::indexer::{KvIndexerInterface, KvRouterError, ShardSizeSnapshot};
use crate::protocols::{
    BlockHashOptions, LocalBlockHash, OverlapScores, RouterEvent, TokensWithHashes,
    WorkerWithDpRank, compute_block_hash_for_seq, compute_seq_hash_for_block,
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CuckooIndexerMode {
    Native,
    Transposed,
}

#[derive(Clone, Debug)]
pub struct CuckooConsumerSession {
    pub dc_worker_id: u64,
    pub relay_instance_id: u64,
    pub num_buckets: usize,
    pub seed: u64,
}

pub type CuckooDcConfig = CuckooConsumerSession;

#[derive(Clone, Debug)]
pub struct CuckooIndexerConfig {
    pub mode: CuckooIndexerMode,
    pub event_threads: usize,
    pub block_size: u32,
    pub dcs: Vec<CuckooConsumerSession>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum CuckooPublication {
    Unchanged,
    Delta(Arc<[u8]>),
    Full(Vec<Arc<[u8]>>),
}

#[derive(Clone, Debug)]
pub struct CuckooFrameEnvelope {
    pub dc_worker_id: u64,
    pub relay_instance_id: u64,
    pub publication: CuckooPublication,
}

impl CuckooPublication {
    pub fn bytes(&self) -> usize {
        match self {
            Self::Unchanged => 0,
            Self::Delta(frame) => frame.len(),
            Self::Full(chunks) => chunks.iter().map(|chunk| chunk.len()).sum(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CuckooFrameMetadata {
    pub logical_event_id: u64,
    pub scheduled_at: Instant,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CuckooQueueMetrics {
    pub at_stop: u64,
    pub maximum_depth: u64,
    pub drain_ns: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CuckooPipelineErrors {
    pub decode: u64,
    pub application: u64,
    pub epoch: u64,
    pub desynchronization: u64,
}

impl CuckooPipelineErrors {
    pub fn total(self) -> u64 {
        self.decode + self.application + self.epoch + self.desynchronization
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CuckooStatsSnapshot {
    pub frames: u64,
    pub dirty_buckets: u64,
    pub bytes: u64,
    pub apply_ns: u64,
    pub full_bytes: u64,
    pub full_apply_ns: u64,
    pub delta_bytes: u64,
    pub delta_apply_ns: u64,
    pub full: u64,
    pub delta: u64,
    pub generation_conflicts: u64,
    pub native_fallbacks: u64,
    pub repeated_fallbacks: u64,
    pub maximum_queue_depth: u64,
    pub errors: CuckooPipelineErrors,
}

struct RouterFilterState {
    filter: CuckooFilter,
    epoch: u64,
    available: bool,
    desynchronized: bool,
}

struct FrameTask {
    dc: usize,
    metadata: CuckooFrameMetadata,
    enqueued_at: Instant,
    envelope: CuckooFrameEnvelope,
}

enum ConsumerTask {
    Frame(FrameTask),
    Barrier(flume::Sender<()>),
    Stop,
}

#[derive(Default)]
struct CuckooStats {
    frames: AtomicU64,
    dirty_buckets: AtomicU64,
    bytes: AtomicU64,
    apply_ns: AtomicU64,
    full_bytes: AtomicU64,
    full_apply_ns: AtomicU64,
    delta_bytes: AtomicU64,
    delta_apply_ns: AtomicU64,
    full: AtomicU64,
    delta: AtomicU64,
    generation_conflicts: AtomicU64,
    native_fallbacks: AtomicU64,
    repeated_fallbacks: AtomicU64,
    maximum_queue_depth: AtomicU64,
    decode_errors: AtomicU64,
    application_errors: AtomicU64,
    epoch_errors: AtomicU64,
    desynchronizations: AtomicU64,
    scheduled_to_enqueue_ns: Mutex<Vec<u64>>,
    enqueue_to_applied_ns: Mutex<Vec<u64>>,
    scheduled_to_applied_ns: Mutex<Vec<u64>>,
}

impl CuckooStats {
    fn reset(&self) {
        for counter in [
            &self.frames,
            &self.dirty_buckets,
            &self.bytes,
            &self.apply_ns,
            &self.full_bytes,
            &self.full_apply_ns,
            &self.delta_bytes,
            &self.delta_apply_ns,
            &self.full,
            &self.delta,
            &self.generation_conflicts,
            &self.native_fallbacks,
            &self.repeated_fallbacks,
            &self.maximum_queue_depth,
            &self.decode_errors,
            &self.application_errors,
            &self.epoch_errors,
            &self.desynchronizations,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.scheduled_to_enqueue_ns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.enqueue_to_applied_ns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.scheduled_to_applied_ns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    fn snapshot(&self) -> CuckooStatsSnapshot {
        CuckooStatsSnapshot {
            frames: self.frames.load(Ordering::Relaxed),
            dirty_buckets: self.dirty_buckets.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            apply_ns: self.apply_ns.load(Ordering::Relaxed),
            full_bytes: self.full_bytes.load(Ordering::Relaxed),
            full_apply_ns: self.full_apply_ns.load(Ordering::Relaxed),
            delta_bytes: self.delta_bytes.load(Ordering::Relaxed),
            delta_apply_ns: self.delta_apply_ns.load(Ordering::Relaxed),
            full: self.full.load(Ordering::Relaxed),
            delta: self.delta.load(Ordering::Relaxed),
            generation_conflicts: self.generation_conflicts.load(Ordering::Relaxed),
            native_fallbacks: self.native_fallbacks.load(Ordering::Relaxed),
            repeated_fallbacks: self.repeated_fallbacks.load(Ordering::Relaxed),
            maximum_queue_depth: self.maximum_queue_depth.load(Ordering::Relaxed),
            errors: CuckooPipelineErrors {
                decode: self.decode_errors.load(Ordering::Relaxed),
                application: self.application_errors.load(Ordering::Relaxed),
                epoch: self.epoch_errors.load(Ordering::Relaxed),
                desynchronization: self.desynchronizations.load(Ordering::Relaxed),
            },
        }
    }
}

pub struct CuckooFrameIndexer {
    mode: CuckooIndexerMode,
    block_size: u32,
    seed: u64,
    dcs: Vec<CuckooConsumerSession>,
    dc_indices: FxHashMap<u64, usize>,
    states: Vec<Arc<RwLock<RouterFilterState>>>,
    transposed: Option<Arc<TransposedTable>>,
    senders: Vec<flume::Sender<ConsumerTask>>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    dc_to_thread: Vec<usize>,
    stats: Arc<CuckooStats>,
}

impl CuckooFrameIndexer {
    pub fn new(config: CuckooIndexerConfig) -> anyhow::Result<Arc<Self>> {
        ensure!(
            config.event_threads > 0,
            "event_threads must be greater than zero"
        );
        ensure!(!config.dcs.is_empty(), "at least one DC is required");
        ensure!(
            config.dcs.len() <= 16,
            "transposed CKF supports at most 16 DCs"
        );
        let first = &config.dcs[0];
        let seed = first.seed;
        ensure!(
            config
                .dcs
                .iter()
                .all(|dc| { dc.num_buckets == first.num_buckets && dc.seed == first.seed }),
            "all CKF DCs must use the same bucket count and seed"
        );
        let mut dc_indices = FxHashMap::default();
        for (index, dc) in config.dcs.iter().enumerate() {
            ensure!(
                dc_indices.insert(dc.dc_worker_id, index).is_none(),
                "duplicate CKF DC worker id {}",
                dc.dc_worker_id
            );
        }
        let filters: Vec<CuckooFilter> = config
            .dcs
            .iter()
            .map(|dc| CuckooFilter::with_num_buckets(dc.num_buckets, dc.seed))
            .collect();
        let transposed = if config.mode == CuckooIndexerMode::Transposed {
            Some(Arc::new(TransposedTable::from_filters(&filters)?))
        } else {
            None
        };
        let states: Vec<_> = filters
            .into_iter()
            .map(|filter| {
                Arc::new(RwLock::new(RouterFilterState {
                    filter,
                    epoch: 0,
                    available: false,
                    desynchronized: false,
                }))
            })
            .collect();
        let dc_to_thread: Vec<usize> = (0..config.dcs.len())
            .map(|dc| dc % config.event_threads)
            .collect();
        let stats = Arc::new(CuckooStats::default());
        let mut senders = Vec::with_capacity(config.event_threads);
        let mut handles = Vec::with_capacity(config.event_threads);
        for worker in 0..config.event_threads {
            let (sender, receiver) = flume::unbounded();
            senders.push(sender);
            let worker_states = states.clone();
            let worker_table = transposed.clone();
            let worker_dcs = config.dcs.clone();
            let worker_stats = Arc::clone(&stats);
            handles.push(
                std::thread::Builder::new()
                    .name(format!("ckf-frame-consumer-{worker}"))
                    .spawn(move || {
                        run_worker(
                            receiver,
                            &worker_states,
                            worker_table.as_deref(),
                            &worker_dcs,
                            &worker_stats,
                        );
                    })?,
            );
        }
        Ok(Arc::new(Self {
            mode: config.mode,
            block_size: config.block_size,
            seed,
            dcs: config.dcs,
            dc_indices,
            states,
            transposed,
            senders,
            handles: Mutex::new(handles),
            dc_to_thread,
            stats,
        }))
    }

    pub fn install_bootstrap(&self, envelope: CuckooFrameEnvelope) -> anyhow::Result<()> {
        let dc = self.dc_index(envelope.dc_worker_id)?;
        ensure!(
            matches!(&envelope.publication, CuckooPublication::Full(_)),
            "bootstrap requires a full snapshot"
        );
        apply_publication(
            dc,
            &envelope,
            &self.states[dc],
            self.transposed.as_deref(),
            &self.dcs[dc],
        )?;
        Ok(())
    }

    pub fn submit(
        &self,
        envelope: CuckooFrameEnvelope,
        metadata: CuckooFrameMetadata,
    ) -> anyhow::Result<()> {
        match &envelope.publication {
            CuckooPublication::Unchanged => {
                anyhow::bail!("Unchanged is a publisher outcome, not a serialized CKF frame");
            }
            CuckooPublication::Delta(_) => {
                self.stats.delta.fetch_add(1, Ordering::Relaxed);
            }
            CuckooPublication::Full(_) => {
                self.stats.full.fetch_add(1, Ordering::Relaxed);
            }
        }
        let dc = self.dc_index(envelope.dc_worker_id)?;
        let worker = self.dc_to_thread[dc];
        let task = ConsumerTask::Frame(FrameTask {
            dc,
            metadata,
            enqueued_at: Instant::now(),
            envelope,
        });
        self.senders[worker]
            .send(task)
            .context("CKF frame consumer stopped")?;
        update_max(&self.stats.maximum_queue_depth, self.senders[worker].len());
        Ok(())
    }

    pub fn lookup(&self, local_hashes: &[LocalBlockHash]) -> OverlapScores {
        let sequence = compute_seq_hash_for_block(local_hashes);
        let probes = probes_for(&sequence, self.seed);
        scores_from_depths(&self.optimized_depths_for_probes(&probes), &self.dcs)
    }

    pub fn optimized_depths(&self, local_hashes: &[LocalBlockHash]) -> Vec<u32> {
        let sequence = compute_seq_hash_for_block(local_hashes);
        let probes = probes_for(&sequence, self.seed);
        self.optimized_depths_for_probes(&probes)
    }

    pub fn linear_depths(&self, local_hashes: &[LocalBlockHash]) -> Vec<u32> {
        let sequence = compute_seq_hash_for_block(local_hashes);
        self.states
            .iter()
            .map(|state| {
                let state = state
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if !state.available {
                    return 0;
                }
                sequence
                    .iter()
                    .take_while(|&&hash| state.filter.contains(hash))
                    .count() as u32
            })
            .collect()
    }

    pub fn flush_with_metrics(&self) -> anyhow::Result<CuckooQueueMetrics> {
        let at_stop = self.senders.iter().map(flume::Sender::len).sum::<usize>() as u64;
        let started = Instant::now();
        let mut waits = Vec::with_capacity(self.senders.len());
        for sender in &self.senders {
            let (ack, wait) = flume::bounded(1);
            sender
                .send(ConsumerTask::Barrier(ack))
                .context("CKF worker stopped before barrier")?;
            waits.push(wait);
        }
        for wait in waits {
            wait.recv().context("CKF worker dropped barrier")?;
        }
        Ok(CuckooQueueMetrics {
            at_stop,
            maximum_depth: self.stats.maximum_queue_depth.load(Ordering::Relaxed),
            drain_ns: started.elapsed().as_nanos() as u64,
        })
    }

    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    pub fn stats_snapshot(&self) -> CuckooStatsSnapshot {
        self.stats.snapshot()
    }

    pub fn take_update_latencies(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let scheduled = std::mem::take(
            &mut *self
                .stats
                .scheduled_to_enqueue_ns
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        let applied = std::mem::take(
            &mut *self
                .stats
                .enqueue_to_applied_ns
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        let visible = std::mem::take(
            &mut *self
                .stats
                .scheduled_to_applied_ns
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        (scheduled, applied, visible)
    }

    pub fn verify_transposed(&self) -> anyhow::Result<()> {
        let Some(table) = &self.transposed else {
            return Ok(());
        };
        let filters: Vec<CuckooFilter> = self
            .states
            .iter()
            .map(|state| {
                state
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .filter
                    .clone()
            })
            .collect();
        table.verify_filters(&filters)
    }

    #[cfg(feature = "bench")]
    pub fn touch_for_benchmark(&self) {
        for state in &self.states {
            let state = state
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for bucket in 0..state.filter.num_buckets() {
                std::hint::black_box(state.filter.bucket_slots(bucket));
            }
        }
        if let Some(table) = &self.transposed {
            table.touch_for_benchmark();
        }
    }

    pub fn shutdown_threads(&self) {
        for sender in &self.senders {
            let _ = sender.send(ConsumerTask::Stop);
        }
        let mut handles = self
            .handles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for handle in handles.drain(..) {
            let _ = handle.join();
        }
    }

    fn dc_index(&self, dc_worker_id: u64) -> anyhow::Result<usize> {
        self.dc_indices
            .get(&dc_worker_id)
            .copied()
            .with_context(|| format!("unknown CKF DC worker id {dc_worker_id}"))
    }

    fn optimized_depths_for_probes(&self, probes: &[Probe]) -> Vec<u32> {
        match self.mode {
            CuckooIndexerMode::Native => self.native_depths(probes),
            CuckooIndexerMode::Transposed => self.transposed_depths(probes),
        }
    }

    fn native_depths(&self, probes: &[Probe]) -> Vec<u32> {
        self.states
            .iter()
            .map(|state| {
                let state = state
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if state.available {
                    overlap_depth_searched(&state.filter, probes)
                } else {
                    0
                }
            })
            .collect()
    }

    fn transposed_depths(&self, probes: &[Probe]) -> Vec<u32> {
        let table = self.transposed.as_ref().expect("transposed table missing");
        let mut result = table.search(probes, self.available_mask());
        self.stats
            .generation_conflicts
            .fetch_add(result.conflict_mask.count_ones() as u64, Ordering::Relaxed);
        let mut conflicts = result.conflict_mask;
        while conflicts != 0 {
            let dc = conflicts.trailing_zeros() as usize;
            conflicts &= conflicts - 1;
            self.stats.native_fallbacks.fetch_add(1, Ordering::Relaxed);
            let state = self.states[dc]
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.available {
                result.depths[dc] = overlap_depth_searched(&state.filter, probes);
            }
        }
        result.depths
    }

    fn available_mask(&self) -> u16 {
        self.states
            .iter()
            .enumerate()
            .fold(0u16, |mask, (dc, state)| {
                let available = state
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .available;
                mask | (u16::from(available) << dc)
            })
    }
}

impl Drop for CuckooFrameIndexer {
    fn drop(&mut self) {
        for sender in &self.senders {
            let _ = sender.send(ConsumerTask::Stop);
        }
    }
}

#[async_trait]
impl KvIndexerInterface for CuckooFrameIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        Ok(self.lookup(&sequence))
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
        is_eagle: Option<bool>,
    ) -> Result<OverlapScores, KvRouterError> {
        Ok(self.lookup(&compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                lora_name,
                is_eagle,
                ..Default::default()
            },
        )))
    }

    async fn apply_event(&self, _event: RouterEvent) {
        tracing::error!("CKF indexers consume serialized Relay frames, not RouterEvents");
        self.stats
            .application_errors
            .fetch_add(1, Ordering::Relaxed);
    }

    async fn remove_worker(&self, worker_id: u64) {
        let Some(&dc) = self.dc_indices.get(&worker_id) else {
            return;
        };
        let mut state = self.states[dc]
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let generation = self.transposed.as_ref().map(|table| table.begin_update(dc));
        state.available = false;
        state.desynchronized = true;
        if let Some(table) = &self.transposed {
            table.end_update(dc, generation.expect("generation missing"));
        }
    }

    fn shutdown(&self) {
        self.shutdown_threads();
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        Err(KvRouterError::Unsupported(
            "CKF filters cannot reconstruct RouterEvents".to_string(),
        ))
    }

    async fn process_routing_decision_for_request(
        &self,
        _tokens_with_hashes: &mut TokensWithHashes,
        _worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        Err(KvRouterError::Unsupported(
            "CKF indexers are updated by serialized Relay frames".to_string(),
        ))
    }

    async fn flush(&self) -> usize {
        self.flush_with_metrics()
            .map(|metrics| metrics.at_stop as usize)
            .unwrap_or_default()
    }

    async fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        self.states
            .iter()
            .enumerate()
            .map(|(dc, state)| {
                let state = state
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                ShardSizeSnapshot {
                    shard_idx: dc,
                    worker_count: usize::from(state.available),
                    block_count: state.filter.len(),
                    node_count: 0,
                }
            })
            .collect()
    }
}

fn run_worker(
    receiver: flume::Receiver<ConsumerTask>,
    states: &[Arc<RwLock<RouterFilterState>>],
    transposed: Option<&TransposedTable>,
    dcs: &[CuckooConsumerSession],
    stats: &CuckooStats,
) {
    while let Ok(task) = receiver.recv() {
        match task {
            ConsumerTask::Frame(task) => {
                let started = Instant::now();
                let result = apply_publication(
                    task.dc,
                    &task.envelope,
                    &states[task.dc],
                    transposed,
                    &dcs[task.dc],
                );
                let applied_at = Instant::now();
                let elapsed_ns = started.elapsed().as_nanos() as u64;
                stats.apply_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
                record_apply_time(&task.envelope.publication, elapsed_ns, stats);
                stats
                    .scheduled_to_enqueue_ns
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(
                        task.enqueued_at
                            .saturating_duration_since(task.metadata.scheduled_at)
                            .as_nanos() as u64,
                    );
                match result {
                    Ok(dirty_buckets) => {
                        record_applied(&task.envelope.publication, dirty_buckets, stats);
                        stats
                            .enqueue_to_applied_ns
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(
                                applied_at
                                    .saturating_duration_since(task.enqueued_at)
                                    .as_nanos() as u64,
                            );
                        stats
                            .scheduled_to_applied_ns
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(
                                applied_at
                                    .saturating_duration_since(task.metadata.scheduled_at)
                                    .as_nanos() as u64,
                            );
                    }
                    Err(error) => {
                        classify_error(&error, stats);
                        mark_desynchronized(task.dc, &states[task.dc], transposed, stats);
                        tracing::warn!(
                            logical_event_id = task.metadata.logical_event_id,
                            dc = task.dc,
                            %error,
                            "failed to apply CKF publication"
                        );
                    }
                }
            }
            ConsumerTask::Barrier(ack) => {
                let _ = ack.send(());
            }
            ConsumerTask::Stop => break,
        }
    }
}

fn apply_publication(
    dc: usize,
    envelope: &CuckooFrameEnvelope,
    state: &RwLock<RouterFilterState>,
    transposed: Option<&TransposedTable>,
    config: &CuckooConsumerSession,
) -> anyhow::Result<usize> {
    ensure!(
        envelope.dc_worker_id == config.dc_worker_id,
        "DC worker identity mismatch"
    );
    ensure!(
        envelope.relay_instance_id == config.relay_instance_id,
        "Relay instance mismatch"
    );
    match &envelope.publication {
        CuckooPublication::Unchanged => Ok(0),
        CuckooPublication::Delta(frame) => {
            let mut state = state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ensure!(state.available, "no base snapshot for DC");
            ensure!(!state.desynchronized, "DC is desynchronized");
            let delta =
                decode_delta(&state.filter, state.epoch, frame).context("decode CKF delta")?;
            ensure!(
                delta.dc_worker_id == config.dc_worker_id,
                "delta DC mismatch"
            );
            let generation = transposed.map(|table| table.begin_update(dc));
            apply_decoded_delta(&mut state.filter, &delta);
            state.epoch = delta.new_epoch;
            if let Some(table) = transposed {
                table.apply_entries(dc, &delta.entries);
                table.end_update(dc, generation.expect("generation missing"));
            }
            Ok(delta.entries.len())
        }
        CuckooPublication::Full(chunks) => {
            let (filter, meta) = assemble_chunks(chunks).context("assemble CKF full snapshot")?;
            ensure!(
                meta.dc_worker_id == config.dc_worker_id,
                "snapshot DC mismatch"
            );
            ensure!(
                meta.num_buckets == config.num_buckets && meta.seed == config.seed,
                "snapshot shape or seed mismatch"
            );
            let mut state = state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ensure!(
                !state.available || state.desynchronized || meta.filter_epoch > state.epoch,
                "snapshot epoch did not advance"
            );
            let generation = transposed.map(|table| table.begin_update(dc));
            state.filter = filter;
            state.epoch = meta.filter_epoch;
            state.available = true;
            state.desynchronized = false;
            if let Some(table) = transposed {
                table.rebuild_dc(dc, &state.filter);
                table.end_update(dc, generation.expect("generation missing"));
            }
            Ok(0)
        }
    }
}

fn record_applied(publication: &CuckooPublication, dirty_buckets: usize, stats: &CuckooStats) {
    let frames = match publication {
        CuckooPublication::Unchanged => 0,
        CuckooPublication::Delta(_) => 1,
        CuckooPublication::Full(chunks) => chunks.len(),
    };
    stats.frames.fetch_add(frames as u64, Ordering::Relaxed);
    stats
        .dirty_buckets
        .fetch_add(dirty_buckets as u64, Ordering::Relaxed);
    stats
        .bytes
        .fetch_add(publication.bytes() as u64, Ordering::Relaxed);
    match publication {
        CuckooPublication::Full(_) => {
            stats
                .full_bytes
                .fetch_add(publication.bytes() as u64, Ordering::Relaxed);
        }
        CuckooPublication::Delta(_) => {
            stats
                .delta_bytes
                .fetch_add(publication.bytes() as u64, Ordering::Relaxed);
        }
        CuckooPublication::Unchanged => {}
    }
}

fn record_apply_time(publication: &CuckooPublication, elapsed_ns: u64, stats: &CuckooStats) {
    match publication {
        CuckooPublication::Full(_) => {
            stats.full_apply_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        CuckooPublication::Delta(_) => {
            stats
                .delta_apply_ns
                .fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        CuckooPublication::Unchanged => {}
    }
}

fn classify_error(error: &anyhow::Error, stats: &CuckooStats) {
    let message = format!("{error:#}");
    if message.contains("epoch") || message.contains("base snapshot") {
        stats.epoch_errors.fetch_add(1, Ordering::Relaxed);
    } else if message.contains("decode")
        || message.contains("checksum")
        || message.contains("snapshot")
        || message.contains("chunk")
    {
        stats.decode_errors.fetch_add(1, Ordering::Relaxed);
    } else {
        stats.application_errors.fetch_add(1, Ordering::Relaxed);
    }
}

fn mark_desynchronized(
    dc: usize,
    state: &RwLock<RouterFilterState>,
    transposed: Option<&TransposedTable>,
    stats: &CuckooStats,
) {
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let generation = transposed.map(|table| table.begin_update(dc));
    if !state.desynchronized {
        stats.desynchronizations.fetch_add(1, Ordering::Relaxed);
    }
    state.desynchronized = true;
    state.available = false;
    if let Some(table) = transposed {
        table.end_update(dc, generation.expect("generation missing"));
    }
}

fn scores_from_depths(depths: &[u32], dcs: &[CuckooConsumerSession]) -> OverlapScores {
    let mut scores = OverlapScores::new();
    for (dc, &depth) in depths.iter().enumerate() {
        if depth > 0 {
            scores.scores.insert(
                WorkerWithDpRank::from_worker_id(dcs[dc].dc_worker_id),
                depth,
            );
        }
    }
    scores
}

fn update_max(counter: &AtomicU64, value: usize) {
    let value = value as u64;
    let mut current = counter.load(Ordering::Relaxed);
    while current < value {
        match counter.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::cuckoo::{DEFAULT_FILTER_SEED, Publish, SnapshotProducer};

    fn config(mode: CuckooIndexerMode) -> CuckooIndexerConfig {
        let producer = SnapshotProducer::new(0, 64, DEFAULT_FILTER_SEED);
        CuckooIndexerConfig {
            mode,
            event_threads: 1,
            block_size: 16,
            dcs: vec![CuckooConsumerSession {
                dc_worker_id: 0,
                relay_instance_id: 10,
                num_buckets: producer.num_buckets(),
                seed: DEFAULT_FILTER_SEED,
            }],
        }
    }

    fn full_publication(producer: &mut SnapshotProducer) -> CuckooPublication {
        CuckooPublication::Full(
            producer
                .full_snapshot()
                .chunks_with(4)
                .map(Arc::from)
                .collect(),
        )
    }

    #[test]
    fn full_snapshot_bootstrap_and_delta_apply() {
        let mut producer = SnapshotProducer::new(0, 64, DEFAULT_FILTER_SEED);
        assert!(producer.insert(11));
        let bootstrap: Vec<Arc<[u8]>> = producer
            .full_snapshot()
            .chunks_with(4)
            .map(Arc::from)
            .collect();
        let indexer = CuckooFrameIndexer::new(config(CuckooIndexerMode::Transposed)).unwrap();
        indexer
            .install_bootstrap(CuckooFrameEnvelope {
                dc_worker_id: 0,
                relay_instance_id: 10,
                publication: CuckooPublication::Full(bootstrap),
            })
            .unwrap();
        assert!(producer.insert(22));
        let Publish::Delta(delta) = producer.publish() else {
            panic!("expected delta")
        };
        indexer
            .submit(
                CuckooFrameEnvelope {
                    dc_worker_id: 0,
                    relay_instance_id: 10,
                    publication: CuckooPublication::Delta(Arc::from(delta)),
                },
                CuckooFrameMetadata {
                    logical_event_id: 1,
                    scheduled_at: Instant::now(),
                },
            )
            .unwrap();
        indexer.flush_with_metrics().unwrap();
        indexer.verify_transposed().unwrap();
        assert_eq!(indexer.linear_depths(&[LocalBlockHash(11)]), vec![1]);
    }

    #[test]
    fn corrupt_delta_desynchronizes_and_full_snapshot_recovers() {
        let mut producer = SnapshotProducer::new(0, 64, DEFAULT_FILTER_SEED);
        assert!(producer.insert(11));
        let bootstrap: Vec<Arc<[u8]>> = producer
            .full_snapshot()
            .chunks_with(4)
            .map(Arc::from)
            .collect();
        let indexer = CuckooFrameIndexer::new(config(CuckooIndexerMode::Native)).unwrap();
        indexer
            .install_bootstrap(CuckooFrameEnvelope {
                dc_worker_id: 0,
                relay_instance_id: 10,
                publication: CuckooPublication::Full(bootstrap),
            })
            .unwrap();
        assert!(producer.insert(22));
        let Publish::Delta(mut delta) = producer.publish() else {
            panic!("expected delta")
        };
        *delta.last_mut().unwrap() ^= 1;
        indexer
            .submit(
                CuckooFrameEnvelope {
                    dc_worker_id: 0,
                    relay_instance_id: 10,
                    publication: CuckooPublication::Delta(Arc::from(delta)),
                },
                CuckooFrameMetadata {
                    logical_event_id: 1,
                    scheduled_at: Instant::now(),
                },
            )
            .unwrap();
        indexer.flush_with_metrics().unwrap();
        assert_eq!(indexer.lookup(&[LocalBlockHash(11)]).scores.len(), 0);
        assert_eq!(indexer.stats_snapshot().errors.desynchronization, 1);

        let recovery: Vec<Arc<[u8]>> = producer
            .full_snapshot()
            .chunks_with(4)
            .map(Arc::from)
            .collect();
        indexer
            .submit(
                CuckooFrameEnvelope {
                    dc_worker_id: 0,
                    relay_instance_id: 10,
                    publication: CuckooPublication::Full(recovery),
                },
                CuckooFrameMetadata {
                    logical_event_id: 2,
                    scheduled_at: Instant::now(),
                },
            )
            .unwrap();
        indexer.flush_with_metrics().unwrap();
        assert_eq!(
            indexer
                .lookup(&[LocalBlockHash(11)])
                .scores
                .get(&WorkerWithDpRank::from_worker_id(0)),
            Some(&1)
        );
    }

    #[test]
    fn transposed_conflict_falls_back_only_for_changed_dc() {
        let mut producer0 = SnapshotProducer::new(0, 64, DEFAULT_FILTER_SEED);
        let mut producer1 = SnapshotProducer::new(1, 64, DEFAULT_FILTER_SEED);
        assert!(producer0.insert(11));
        assert!(producer1.insert(11));
        let buckets = producer0.num_buckets();
        let indexer = CuckooFrameIndexer::new(CuckooIndexerConfig {
            mode: CuckooIndexerMode::Transposed,
            event_threads: 1,
            block_size: 16,
            dcs: vec![
                CuckooConsumerSession {
                    dc_worker_id: 0,
                    relay_instance_id: 10,
                    num_buckets: buckets,
                    seed: DEFAULT_FILTER_SEED,
                },
                CuckooConsumerSession {
                    dc_worker_id: 1,
                    relay_instance_id: 11,
                    num_buckets: buckets,
                    seed: DEFAULT_FILTER_SEED,
                },
            ],
        })
        .unwrap();
        for (dc, instance, producer) in [(0, 10, &mut producer0), (1, 11, &mut producer1)] {
            indexer
                .install_bootstrap(CuckooFrameEnvelope {
                    dc_worker_id: dc,
                    relay_instance_id: instance,
                    publication: CuckooPublication::Full(
                        producer
                            .full_snapshot()
                            .chunks_with(4)
                            .map(Arc::from)
                            .collect(),
                    ),
                })
                .unwrap();
        }
        indexer.reset_stats();
        let table = indexer.transposed.as_ref().unwrap();
        let generation = table.begin_update(1);
        let scores = indexer.lookup(&[LocalBlockHash(11)]);
        table.end_update(1, generation);
        assert_eq!(scores.scores.len(), 2);
        let stats = indexer.stats_snapshot();
        assert_eq!(stats.generation_conflicts, 1);
        assert_eq!(stats.native_fallbacks, 1);
    }

    #[test]
    fn consumer_session_rejects_wrong_outer_and_embedded_identity() {
        let mut producer = SnapshotProducer::new(0, 64, DEFAULT_FILTER_SEED);
        assert!(producer.insert(11));
        let indexer = CuckooFrameIndexer::new(config(CuckooIndexerMode::Native)).unwrap();

        let wrong_instance = indexer.install_bootstrap(CuckooFrameEnvelope {
            dc_worker_id: 0,
            relay_instance_id: 99,
            publication: full_publication(&mut producer),
        });
        assert!(
            wrong_instance
                .unwrap_err()
                .to_string()
                .contains("Relay instance")
        );

        let mut other_dc = SnapshotProducer::new(1, 64, DEFAULT_FILTER_SEED);
        assert!(other_dc.insert(11));
        let wrong_embedded_dc = indexer.install_bootstrap(CuckooFrameEnvelope {
            dc_worker_id: 0,
            relay_instance_id: 10,
            publication: full_publication(&mut other_dc),
        });
        assert!(
            wrong_embedded_dc
                .unwrap_err()
                .to_string()
                .contains("snapshot DC")
        );
        assert!(indexer.lookup(&[LocalBlockHash(11)]).scores.is_empty());
    }

    #[test]
    fn unchanged_outcome_never_enters_the_frame_queue() {
        let indexer = CuckooFrameIndexer::new(config(CuckooIndexerMode::Native)).unwrap();
        let error = indexer
            .submit(
                CuckooFrameEnvelope {
                    dc_worker_id: 0,
                    relay_instance_id: 10,
                    publication: CuckooPublication::Unchanged,
                },
                CuckooFrameMetadata {
                    logical_event_id: 1,
                    scheduled_at: Instant::now(),
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("not a serialized CKF frame"));
        assert_eq!(indexer.flush_with_metrics().unwrap().at_stop, 0);
    }

    #[test]
    fn deterministic_assignment_preserves_per_dc_fifo_epochs() {
        let mut producers: Vec<_> = (0..16)
            .map(|dc| SnapshotProducer::new(dc, 64, DEFAULT_FILTER_SEED))
            .collect();
        let buckets = producers[0].num_buckets();
        let indexer = CuckooFrameIndexer::new(CuckooIndexerConfig {
            mode: CuckooIndexerMode::Transposed,
            event_threads: 8,
            block_size: 16,
            dcs: (0..16)
                .map(|dc| CuckooConsumerSession {
                    dc_worker_id: dc,
                    relay_instance_id: 10 + dc,
                    num_buckets: buckets,
                    seed: DEFAULT_FILTER_SEED,
                })
                .collect(),
        })
        .unwrap();
        assert_eq!(
            indexer.dc_to_thread,
            (0..16).map(|dc| dc % 8).collect::<Vec<_>>()
        );
        for (dc, producer) in producers.iter_mut().enumerate() {
            indexer
                .install_bootstrap(CuckooFrameEnvelope {
                    dc_worker_id: dc as u64,
                    relay_instance_id: 10 + dc as u64,
                    publication: full_publication(producer),
                })
                .unwrap();
        }

        for (event, hash) in [(1, 22), (2, 33)] {
            assert!(producers[0].insert(hash));
            let Publish::Delta(frame) = producers[0].publish() else {
                panic!("expected delta")
            };
            indexer
                .submit(
                    CuckooFrameEnvelope {
                        dc_worker_id: 0,
                        relay_instance_id: 10,
                        publication: CuckooPublication::Delta(Arc::from(frame)),
                    },
                    CuckooFrameMetadata {
                        logical_event_id: event,
                        scheduled_at: Instant::now(),
                    },
                )
                .unwrap();
        }
        indexer.flush_with_metrics().unwrap();
        let state = indexer.states[0]
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(state.epoch, producers[0].epoch());
        assert!(state.filter.contains(22));
        assert!(state.filter.contains(33));
        assert_eq!(indexer.stats_snapshot().errors.total(), 0);
    }

    #[test]
    fn scores_use_dp_rank_zero_and_omit_zero_depth_dcs() {
        let mut first = SnapshotProducer::new(7, 64, DEFAULT_FILTER_SEED);
        let second = SnapshotProducer::new(9, 64, DEFAULT_FILTER_SEED);
        let local = [LocalBlockHash(123)];
        let sequence = compute_seq_hash_for_block(&local);
        assert!(first.insert(sequence[0]));
        let buckets = first.num_buckets();
        let indexer = CuckooFrameIndexer::new(CuckooIndexerConfig {
            mode: CuckooIndexerMode::Native,
            event_threads: 1,
            block_size: 16,
            dcs: vec![
                CuckooConsumerSession {
                    dc_worker_id: 7,
                    relay_instance_id: 17,
                    num_buckets: buckets,
                    seed: DEFAULT_FILTER_SEED,
                },
                CuckooConsumerSession {
                    dc_worker_id: 9,
                    relay_instance_id: 19,
                    num_buckets: buckets,
                    seed: DEFAULT_FILTER_SEED,
                },
            ],
        })
        .unwrap();
        for (dc, instance, mut producer) in [(7, 17, first), (9, 19, second)] {
            indexer
                .install_bootstrap(CuckooFrameEnvelope {
                    dc_worker_id: dc,
                    relay_instance_id: instance,
                    publication: full_publication(&mut producer),
                })
                .unwrap();
        }
        let scores = indexer.lookup(&local);
        assert_eq!(scores.scores.len(), 1);
        assert_eq!(scores.scores.get(&WorkerWithDpRank::new(7, 0)), Some(&1));
        assert!(!scores.scores.contains_key(&WorkerWithDpRank::new(9, 0)));
    }
}
