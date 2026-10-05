// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agentic (Weka/AgentX) corpus preparation for the Mooncake open-loop indexer benchmark.
//!
//! The Mooncake prep re-simulates each worker's share of a trace offline and replays the
//! captured lookups and KV events open loop. This module does the same for an agentic
//! workload, where each turn depends on earlier completions:
//!
//! 1. A row pool (Weka rows imported once by AISimulate's `WekaImporter`) is copied `K` times
//!    with disjoint hashes and labels, mirroring Mooncake's trace duplication factor.
//! 2. Play instances are shuffled and dealt round-robin to `W` workers.
//! 3. Each worker runs a closed-loop, `L`-lane offline replay capped at a fixed virtual time,
//!    so per-worker load is constant across `W` (weak scaling).
//! 4. The agentic driver interns hashes per driver, so independent workers emit identical
//!    local hashes for unrelated content. A chain-consistent per-worker salt removes that
//!    fake cross-worker sharing.
//! 5. Each worker's timeline is shifted by a seeded phase so lane starts do not coincide.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, bail, ensure};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash, compute_next_seq_hash,
};
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_mocker::loadgen::{
    AgenticDependency, AgenticMooncakeHeader, AgenticMooncakeRow, AgenticTrace, WekaImporter,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::progress::make_progress_bar;
use super::replay::WorkerReplayArtifacts;

pub const AGENTIC_POOL_SCHEMA: &str = "dynamo-bench.agentic-row-pool.v1";
/// Seed domain for the per-worker local-hash salt.
const SALT_DOMAIN: u64 = 0xA9E7_1C00;
/// Seed domain for the per-worker phase offsets.
const PHASE_DOMAIN: u64 = 0x5048_4153;
/// The open-loop query slab indexes with u32; stay well below it.
pub const QUERY_SLAB_GATE_BLOCKS: u64 = (u32::MAX as u64) / 10 * 9;

/// One play's rows in the pool, in source order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolPlay {
    pub play_id: String,
    pub rows: Vec<AgenticMooncakeRow>,
}

/// Imported agentic rows grouped by play, plus the import provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticPool {
    pub schema: String,
    pub source_path: String,
    pub source_sha256: String,
    pub import_files: usize,
    pub import_requests: usize,
    pub import_raw_zero_outputs: usize,
    pub nested_timestamp_basis: String,
    pub header: AgenticMooncakeHeader,
    /// Plays ordered by `source_play_ordinal`.
    pub plays: Vec<PoolPlay>,
}

impl AgenticPool {
    /// Group already-imported rows by play, ordered by source ordinal (then play ID).
    pub fn from_rows(
        header: AgenticMooncakeHeader,
        rows: Vec<AgenticMooncakeRow>,
    ) -> anyhow::Result<Self> {
        let request_count = rows.len();
        let mut by_play: HashMap<String, (Option<usize>, Vec<AgenticMooncakeRow>)> = HashMap::new();
        for row in rows {
            let entry = by_play
                .entry(row.play_id.clone())
                .or_insert_with(|| (row.source_play_ordinal, Vec::new()));
            ensure!(
                entry.0 == row.source_play_ordinal,
                "play {} has inconsistent source_play_ordinal values",
                row.play_id
            );
            entry.1.push(row);
        }
        let mut plays = by_play
            .into_iter()
            .map(|(play_id, (ordinal, rows))| (ordinal, PoolPlay { play_id, rows }))
            .collect::<Vec<_>>();
        plays.sort_by(|(left_ordinal, left), (right_ordinal, right)| {
            left_ordinal
                .cmp(right_ordinal)
                .then_with(|| left.play_id.cmp(&right.play_id))
        });
        Ok(Self {
            schema: AGENTIC_POOL_SCHEMA.to_string(),
            source_path: String::new(),
            source_sha256: String::new(),
            import_files: 0,
            import_requests: request_count,
            import_raw_zero_outputs: 0,
            nested_timestamp_basis: String::new(),
            header,
            plays: plays.into_iter().map(|(_, play)| play).collect(),
        })
    }

    pub fn request_count(&self) -> usize {
        self.plays.iter().map(|play| play.rows.len()).sum()
    }

    /// Import a Weka file or directory with AISimulate's importer (timestamp basis `auto`).
    pub fn import_weka(path: &Path) -> anyhow::Result<Self> {
        let source_sha256 = if path.is_file() {
            file_sha256(path)?
        } else {
            String::new()
        };
        let importer = WekaImporter::open(path)
            .with_context(|| format!("importing Weka corpus {}", path.display()))?;
        let (summary, rows) = importer.collect_rows()?;
        let mut pool = Self::from_rows(summary.header.clone(), rows)?;
        ensure!(
            pool.plays.len() == summary.plays,
            "Weka import grouped {} plays, importer reported {}",
            pool.plays.len(),
            summary.plays
        );
        pool.source_path = path.display().to_string();
        pool.source_sha256 = source_sha256;
        pool.import_files = summary.files;
        pool.import_requests = summary.requests;
        pool.import_raw_zero_outputs = summary.raw_zero_outputs;
        pool.nested_timestamp_basis = summary.nested_timestamp_basis.as_str().to_string();
        Ok(pool)
    }

    /// Write the pool as named-field MessagePack (rows skip absent optional fields).
    pub fn write(&self, path: &Path) -> anyhow::Result<String> {
        let bytes = rmp_serde::to_vec_named(self)?;
        std::fs::write(path, &bytes)
            .with_context(|| format!("writing agentic pool {}", path.display()))?;
        Ok(format!("{:x}", Sha256::digest(&bytes)))
    }

    /// Read a pool and return it with its file SHA-256, failing closed on a schema mismatch.
    pub fn read(path: &Path) -> anyhow::Result<(Self, String)> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading agentic pool {}", path.display()))?;
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let pool: Self = rmp_serde::from_slice(&bytes)
            .with_context(|| format!("decoding agentic pool {}", path.display()))?;
        ensure!(
            pool.schema == AGENTIC_POOL_SCHEMA,
            "agentic pool schema {:?}, expected {AGENTIC_POOL_SCHEMA:?}",
            pool.schema
        );
        ensure!(!pool.plays.is_empty(), "agentic pool has no plays");
        Ok((pool, sha256))
    }
}

pub fn file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Knobs that shape the agentic corpus. Defaults match the study design.
#[derive(Clone, Debug, Serialize)]
pub struct AgenticCorpusConfig {
    pub workers: usize,
    /// Target plays per worker; sets the copy count `K = ceil(S * W / plays)`.
    pub plays_per_worker: usize,
    pub lanes_per_worker: usize,
    /// Soft virtual-time cap of each worker's closed-loop replay.
    pub sim_ms: u64,
    /// Cap on every dependency delay (think/tool time).
    pub idle_cap_ms: f64,
    /// Worker phase offsets are drawn from `[0, phase_spread * sim_ms)`.
    pub phase_spread: f64,
    /// Hash-depth multiplier mirroring Mooncake's trace length factor.
    pub length_factor: usize,
    pub seed: u64,
    /// Allow lanes that run out of plays before the cap (plumbing smokes only).
    pub allow_exhausted_lanes: bool,
    pub collision_stats: bool,
}

impl AgenticCorpusConfig {
    pub fn copies(&self, pool_plays: usize) -> usize {
        (self.plays_per_worker * self.workers).div_ceil(pool_plays.max(1))
    }

    fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.workers > 0, "agentic corpus needs at least one worker");
        ensure!(
            self.plays_per_worker > 0,
            "--agentic-plays-per-worker must be > 0"
        );
        ensure!(
            self.lanes_per_worker > 0,
            "--agentic-lanes-per-worker must be > 0"
        );
        ensure!(self.sim_ms > 0, "--agentic-sim-ms must be > 0");
        ensure!(
            self.length_factor > 0,
            "--agentic-length-factor must be > 0"
        );
        ensure!(
            self.idle_cap_ms.is_finite() && self.idle_cap_ms >= 0.0,
            "--agentic-idle-cap-ms must be finite and >= 0"
        );
        ensure!(
            self.phase_spread.is_finite() && (0.0..1.0).contains(&self.phase_spread),
            "--agentic-phase-spread must be in [0, 1)"
        );
        Ok(())
    }
}

/// `(copy, pool play index)` for one play instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlayInstance {
    pub copy: usize,
    pub play: usize,
}

/// Shuffle all `copies * plays` instances with `seed` and deal them round-robin to workers.
pub fn deal_play_instances(
    plays: usize,
    copies: usize,
    workers: usize,
    seed: u64,
) -> Vec<Vec<PlayInstance>> {
    let mut instances = (0..copies)
        .flat_map(|copy| (0..plays).map(move |play| PlayInstance { copy, play }))
        .collect::<Vec<_>>();
    instances.shuffle(&mut StdRng::seed_from_u64(seed));
    let mut dealt = vec![Vec::with_capacity(instances.len().div_ceil(workers)); workers];
    for (index, instance) in instances.into_iter().enumerate() {
        dealt[index % workers].push(instance);
    }
    dealt
}

/// Copy `c` of hash `h`: disjoint across copies, equality-preserving within a copy.
#[inline]
pub fn copy_hash(hash: u64, copy: usize) -> u64 {
    xxh3_64_with_seed(&hash.to_le_bytes(), copy as u64 + 1)
}

fn copy_label(label: &str, copy: usize) -> String {
    format!("c{copy}:{label}")
}

/// Relabel one row into copy `copy` (the TDF analog).
pub fn relabel_row(row: &AgenticMooncakeRow, copy: usize) -> AgenticMooncakeRow {
    AgenticMooncakeRow {
        request_id: copy_label(&row.request_id, copy),
        play_id: copy_label(&row.play_id, copy),
        source_play_ordinal: row.source_play_ordinal,
        session_id: copy_label(&row.session_id, copy),
        model: row.model.clone(),
        input_length: row.input_length,
        output_length: row.output_length,
        output_token_ids: row.output_token_ids.clone(),
        hash_ids: row
            .hash_ids
            .as_ref()
            .map(|hashes| hashes.iter().map(|&hash| copy_hash(hash, copy)).collect()),
        not_before_ms: row.not_before_ms,
        recorded_api_time_ms: row.recorded_api_time_ms,
        priority: row.priority,
        strict_priority: row.strict_priority,
        policy_class: row.policy_class.clone(),
        dependencies: row
            .dependencies
            .iter()
            .map(|dependency| AgenticDependency {
                request_id: copy_label(&dependency.request_id, copy),
                ..dependency.clone()
            })
            .collect(),
    }
}

/// Stretch a row's hash depth by `factor` (the TLF analog), keeping the builder's exact
/// `ceil(input_length / block_size)` hash-count contract: each full block id becomes
/// `factor` ids and a partial tail of `r` tokens becomes `ceil(factor * r / block)` ids.
pub fn expand_row_length(
    row: &mut AgenticMooncakeRow,
    block_size: usize,
    factor: usize,
) -> anyhow::Result<()> {
    if factor == 1 {
        return Ok(());
    }
    let input = row
        .input_length
        .with_context(|| format!("row {} has no input_length", row.request_id))?;
    let hashes = row
        .hash_ids
        .take()
        .with_context(|| format!("row {} has no hash_ids", row.request_id))?;
    let full_blocks = input / block_size;
    let tail_tokens = input % block_size;
    ensure!(
        hashes.len() == full_blocks + usize::from(tail_tokens > 0),
        "row {} has {} hashes for input_length {input} at block size {block_size}",
        row.request_id,
        hashes.len()
    );
    let tail_ids = (factor * tail_tokens).div_ceil(block_size);
    let mut expanded = Vec::with_capacity(full_blocks * factor + tail_ids);
    for &hash in &hashes[..full_blocks] {
        expanded.extend((0..factor).map(|j| xxh3_64_with_seed(&hash.to_le_bytes(), j as u64)));
    }
    if tail_tokens > 0 {
        let tail = hashes[full_blocks];
        expanded.extend((0..tail_ids).map(|j| xxh3_64_with_seed(&tail.to_le_bytes(), j as u64)));
    }
    row.input_length = Some(input * factor);
    row.hash_ids = Some(expanded);
    Ok(())
}

/// Rows for one worker: relabeled copies, contiguous ordinals in dealt order, delays capped,
/// absolute schedule removed (readiness is purely causal), and optional length expansion.
pub fn worker_rows(
    pool: &AgenticPool,
    instances: &[PlayInstance],
    config: &AgenticCorpusConfig,
) -> anyhow::Result<(Vec<AgenticMooncakeRow>, usize)> {
    let mut rows = Vec::new();
    let mut capped_delays = 0usize;
    for (ordinal, instance) in instances.iter().enumerate() {
        for source in &pool.plays[instance.play].rows {
            let mut row = relabel_row(source, instance.copy);
            row.source_play_ordinal = Some(ordinal);
            row.not_before_ms = 0.0;
            for dependency in &mut row.dependencies {
                if dependency.delay_ms > config.idle_cap_ms {
                    dependency.delay_ms = config.idle_cap_ms;
                    capped_delays += 1;
                }
            }
            expand_row_length(&mut row, pool.header.block_size, config.length_factor)?;
            rows.push(row);
        }
    }
    Ok((rows, capped_delays))
}

/// Closed-loop supply and capture counts for one worker.
#[derive(Clone, Debug, Default, Serialize)]
pub struct WorkerCaptureStats {
    pub plays_assigned: usize,
    pub plays_started: usize,
    pub plays_completed: usize,
    pub lanes: usize,
    /// Lanes whose every play completed before the cap: closed-loop supply ran out.
    pub lanes_exhausted: usize,
    pub requests_captured: usize,
    pub requests_completed: usize,
    pub rejected_outputs: usize,
    pub capped_delays: usize,
    pub stored_events: usize,
    pub stored_blocks: usize,
    pub removed_blocks: usize,
    pub request_blocks: usize,
    pub phase_offset_us: u64,
}

fn chain(parent: Option<u64>, local: u64) -> u64 {
    match parent {
        None => local,
        Some(parent) => compute_next_seq_hash(parent, LocalBlockHash(local)),
    }
}

#[inline]
fn salt_local(local: u64, worker: usize) -> u64 {
    xxh3_64_with_seed(&local.to_le_bytes(), SALT_DOMAIN ^ worker as u64)
}

fn sorted_unique(mut values: Vec<u64>) -> Vec<u64> {
    values.sort_unstable();
    values.dedup();
    values
}

fn artifact_local_hashes(artifacts: &WorkerReplayArtifacts) -> Vec<u64> {
    let mut values = Vec::new();
    for request in &artifacts.requests {
        values.extend_from_slice(&request.replay_hashes.local_block_hashes);
    }
    for event in &artifacts.kv_events {
        if let KvCacheEventData::Stored(store) = &event.event.data {
            values.extend(store.blocks.iter().map(|block| block.tokens_hash.0));
        }
    }
    sorted_unique(values)
}

/// Fail closed unless every stored block follows the engine/router chain rule.
pub fn check_stored_chains(worker: usize, artifacts: &WorkerReplayArtifacts) -> anyhow::Result<()> {
    for (index, event) in artifacts.kv_events.iter().enumerate() {
        let KvCacheEventData::Stored(store) = &event.event.data else {
            continue;
        };
        let mut parent = store.parent_hash.map(|hash| hash.0);
        for block in &store.blocks {
            let expected = chain(parent, block.tokens_hash.0);
            ensure!(
                block.block_hash.0 == expected,
                "worker {worker} event {index}: stored block hash {} != chain({parent:?}, {}) = {expected}",
                block.block_hash.0,
                block.tokens_hash.0
            );
            parent = Some(block.block_hash.0);
        }
    }
    Ok(())
}

/// Salt one worker's local hashes and recompute its sequence hashes with the shared chain
/// rule, so equal content stays equal within the worker and becomes disjoint across workers.
pub fn salt_worker_artifacts(
    worker: usize,
    artifacts: &mut WorkerReplayArtifacts,
) -> anyhow::Result<()> {
    check_stored_chains(worker, artifacts)?;
    let mut sequence_map = FxHashMap::<u64, u64>::default();
    for (index, event) in artifacts.kv_events.iter_mut().enumerate() {
        match &mut event.event.data {
            KvCacheEventData::Stored(store) => {
                let mut parent = match store.parent_hash {
                    None => None,
                    Some(hash) => Some(*sequence_map.get(&hash.0).with_context(|| {
                        format!(
                            "worker {worker} event {index}: unmapped parent hash {}",
                            hash.0
                        )
                    })?),
                };
                if let Some(hash) = parent {
                    store.parent_hash = Some(ExternalSequenceBlockHash(hash));
                }
                for block in &mut store.blocks {
                    let local = salt_local(block.tokens_hash.0, worker);
                    let sequence = chain(parent, local);
                    sequence_map.insert(block.block_hash.0, sequence);
                    block.tokens_hash = LocalBlockHash(local);
                    block.block_hash = ExternalSequenceBlockHash(sequence);
                    parent = Some(sequence);
                }
            }
            KvCacheEventData::Removed(remove) => {
                for hash in &mut remove.block_hashes {
                    let mapped = *sequence_map.get(&hash.0).with_context(|| {
                        format!(
                            "worker {worker} event {index}: unmapped removed hash {}",
                            hash.0
                        )
                    })?;
                    *hash = ExternalSequenceBlockHash(mapped);
                }
            }
            KvCacheEventData::Cleared => {}
        }
    }
    for request in &mut artifacts.requests {
        let hashes = &mut request.replay_hashes;
        for local in &mut hashes.local_block_hashes {
            *local = salt_local(*local, worker);
        }
        let mut parent = None;
        hashes.sequence_hashes.clear();
        for &local in &hashes.local_block_hashes {
            let sequence = chain(parent, local);
            hashes.sequence_hashes.push(sequence);
            parent = Some(sequence);
        }
    }
    Ok(())
}

/// Number of distinct values that appear in two or more of the (sorted, deduplicated) lists.
pub fn count_shared_values(lists: &[Vec<u64>]) -> (u64, u64) {
    let mut all = Vec::with_capacity(lists.iter().map(Vec::len).sum());
    for list in lists {
        all.extend_from_slice(list);
    }
    all.sort_unstable();
    let mut distinct = 0u64;
    let mut shared = 0u64;
    let mut index = 0;
    while index < all.len() {
        let mut end = index + 1;
        while end < all.len() && all[end] == all[index] {
            end += 1;
        }
        distinct += 1;
        shared += u64::from(end - index > 1);
        index = end;
    }
    (shared, distinct)
}

fn shift_worker_timeline(artifacts: &mut WorkerReplayArtifacts, offset_us: u64) {
    for request in &mut artifacts.requests {
        request.timestamp_us += offset_us;
    }
    for event in &mut artifacts.kv_events {
        event.timestamp_us += offset_us;
    }
}

struct WorkerCapture {
    artifacts: WorkerReplayArtifacts,
    stats: WorkerCaptureStats,
    locals_before_salt: Vec<u64>,
    locals_after_salt: Vec<u64>,
}

/// Build, replay closed loop, salt, and phase-shift one worker's share.
fn capture_worker(
    pool: &AgenticPool,
    worker: usize,
    instances: &[PlayInstance],
    config: &AgenticCorpusConfig,
    engine_args: MockEngineArgs,
    phase_offset_us: u64,
) -> anyhow::Result<WorkerCapture> {
    ensure!(
        instances.len() >= config.lanes_per_worker,
        "worker {worker} has {} plays for {} lanes; raise --agentic-plays-per-worker",
        instances.len(),
        config.lanes_per_worker
    );
    let (rows, capped_delays) = worker_rows(pool, instances, config)?;
    let mut header = pool.header.clone();
    header.source.digest = format!("{}:w{worker}", header.source.digest);
    let trace = AgenticTrace::from_agentic_mooncake_rows(header, rows)
        .with_context(|| format!("building worker {worker} agentic graph"))?;

    // The driver names node i's request `Uuid(i + 1)`; nodes are sorted by request ID.
    let play_ordinal = instances
        .iter()
        .enumerate()
        .map(|(ordinal, instance)| {
            (
                copy_label(&pool.plays[instance.play].play_id, instance.copy),
                ordinal,
            )
        })
        .collect::<HashMap<_, _>>();
    let node_play = trace
        .nodes()
        .iter()
        .map(|node| {
            play_ordinal
                .get(node.play_id())
                .copied()
                .with_context(|| format!("node play {} was not dealt", node.play_id()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut nodes_per_play = vec![0usize; instances.len()];
    for &play in &node_play {
        nodes_per_play[play] += 1;
    }
    let node_of = |uuid: u128| -> anyhow::Result<usize> {
        let ordinal = usize::try_from(uuid)
            .ok()
            .and_then(|value| value.checked_sub(1))
            .filter(|&value| value < node_play.len());
        ordinal
            .with_context(|| format!("worker {worker}: request {uuid:#x} is not an agentic node"))
    };

    let mut artifacts = dynamo_mocker::replay::generate_agentic_worker_artifacts_offline(
        engine_args,
        trace,
        config.lanes_per_worker,
        Some(config.sim_ms as f64),
    )?;
    artifacts
        .output_signals
        .retain(|signal| signal.signal.completed || signal.signal.rejected);

    let lanes = config.lanes_per_worker.min(instances.len());
    let mut started = vec![false; instances.len()];
    for request in &artifacts.requests {
        started[node_play[node_of(request.uuid.as_u128())?]] = true;
    }
    let mut completed_nodes = FxHashSet::default();
    let mut rejected_outputs = 0usize;
    for signal in &artifacts.output_signals {
        if signal.signal.rejected {
            rejected_outputs += 1;
        } else {
            completed_nodes.insert(node_of(signal.signal.uuid.as_u128())?);
        }
    }
    let mut completed_per_play = vec![0usize; instances.len()];
    for &node in &completed_nodes {
        completed_per_play[node_play[node]] += 1;
    }
    let play_completed = |play: usize| completed_per_play[play] == nodes_per_play[play];
    let lanes_exhausted = (0..lanes)
        .filter(|&lane| (lane..instances.len()).step_by(lanes).all(play_completed))
        .count();

    let locals_before_salt = if config.collision_stats {
        artifact_local_hashes(&artifacts)
    } else {
        Vec::new()
    };
    salt_worker_artifacts(worker, &mut artifacts)?;
    let locals_after_salt = if config.collision_stats {
        artifact_local_hashes(&artifacts)
    } else {
        Vec::new()
    };
    shift_worker_timeline(&mut artifacts, phase_offset_us);
    artifacts.output_signals = Vec::new();

    let mut stats = WorkerCaptureStats {
        plays_assigned: instances.len(),
        plays_started: started.iter().filter(|&&value| value).count(),
        plays_completed: (0..instances.len())
            .filter(|&play| play_completed(play))
            .count(),
        lanes,
        lanes_exhausted,
        requests_captured: artifacts.requests.len(),
        requests_completed: completed_nodes.len(),
        rejected_outputs,
        capped_delays,
        phase_offset_us,
        ..WorkerCaptureStats::default()
    };
    for request in &artifacts.requests {
        stats.request_blocks += request.replay_hashes.local_block_hashes.len();
    }
    for event in &artifacts.kv_events {
        match &event.event.data {
            KvCacheEventData::Stored(store) => {
                stats.stored_events += 1;
                stats.stored_blocks += store.blocks.len();
            }
            KvCacheEventData::Removed(remove) => stats.removed_blocks += remove.block_hashes.len(),
            KvCacheEventData::Cleared => {}
        }
    }
    Ok(WorkerCapture {
        artifacts,
        stats,
        locals_before_salt,
        locals_after_salt,
    })
}

/// Min/mean/max of one per-worker quantity.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Spread {
    pub min: u64,
    pub mean: f64,
    pub max: u64,
    pub total: u64,
}

impl Spread {
    fn of(values: impl Iterator<Item = u64>) -> Self {
        let values = values.collect::<Vec<_>>();
        if values.is_empty() {
            return Self::default();
        }
        let total = values.iter().sum::<u64>();
        Self {
            min: *values.iter().min().unwrap(),
            mean: total as f64 / values.len() as f64,
            max: *values.iter().max().unwrap(),
            total,
        }
    }
}

/// Agentic corpus provenance and validity checks, written to every result file.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AgenticPrepReport {
    pub workload: &'static str,
    pub pool_path: String,
    pub pool_sha256: String,
    pub pool_source_sha256: String,
    pub pool_plays: usize,
    pub pool_requests: usize,
    pub pool_block_size: usize,
    pub pool_source_digest: String,
    pub nested_timestamp_basis: String,
    pub config: Option<AgenticCorpusConfig>,
    pub copies: usize,
    pub play_instances: usize,
    pub engine_block_size: usize,
    pub engine_num_gpu_blocks: usize,
    pub engine_speedup_ratio: f64,
    pub plays_assigned_per_worker: Spread,
    pub plays_started_per_worker: Spread,
    pub plays_completed_per_worker: Spread,
    pub lanes_total: usize,
    pub lanes_exhausted_before_cap: usize,
    pub requests_captured_per_worker: Spread,
    pub requests_completed: u64,
    pub rejected_outputs: u64,
    pub capped_delays: u64,
    pub request_blocks_per_worker: Spread,
    pub stored_blocks_per_worker: Spread,
    pub removed_blocks_per_worker: Spread,
    pub phase_offset_us: Spread,
    /// Distinct local hashes held by two or more workers before/after the salt (`None` when
    /// collision statistics are disabled).
    pub shared_local_hashes_before_salt: Option<u64>,
    pub shared_local_hashes_after_salt: Option<u64>,
    pub distinct_local_hashes_after_salt: Option<u64>,
    pub total_request_blocks: u64,
    pub query_slab_gate_blocks: u64,
    pub capture_ms: f64,
    pub collision_stats_ms: f64,
    /// xxh3 digest of the merged, rescaled corpus (filled in by the bench).
    pub merged_corpus_digest: String,
}

/// Run the closed-loop capture for every worker and return per-worker artifacts ready for
/// the shared merge and rescale.
pub async fn generate_agentic_artifacts(
    pool: std::sync::Arc<AgenticPool>,
    config: &AgenticCorpusConfig,
    engine_args: MockEngineArgs,
    report: &mut AgenticPrepReport,
) -> anyhow::Result<Vec<WorkerReplayArtifacts>> {
    config.validate()?;
    let copies = config.copies(pool.plays.len());
    let dealt = deal_play_instances(pool.plays.len(), copies, config.workers, config.seed);
    let mut phase_rng = StdRng::seed_from_u64(config.seed ^ PHASE_DOMAIN);
    let phase_window_us = config.phase_spread * config.sim_ms as f64 * 1000.0;
    let phases = (0..config.workers)
        .map(|_| (phase_rng.random::<f64>() * phase_window_us) as u64)
        .collect::<Vec<_>>();

    report.copies = copies;
    report.play_instances = copies * pool.plays.len();
    report.engine_block_size = engine_args.block_size;
    report.engine_num_gpu_blocks = engine_args.num_gpu_blocks;
    report.engine_speedup_ratio = engine_args.speedup_ratio;
    report.config = Some(config.clone());
    println!(
        "Capturing agentic corpus: {} workers, {} copies x {} plays, {} lanes/worker, cap {} ms",
        config.workers,
        copies,
        pool.plays.len(),
        config.lanes_per_worker,
        config.sim_ms
    );

    let started = std::time::Instant::now();
    let progress = make_progress_bar(Some(config.workers as u64));
    let mut tasks = Vec::with_capacity(config.workers);
    for (worker, instances) in dealt.into_iter().enumerate() {
        let pool = pool.clone();
        let config = config.clone();
        let engine_args = engine_args.clone();
        let progress = progress.clone();
        let phase = phases[worker];
        tasks.push(tokio::task::spawn_blocking(move || {
            let capture = capture_worker(&pool, worker, &instances, &config, engine_args, phase);
            progress.inc(1);
            capture
        }));
    }
    let mut artifacts = Vec::with_capacity(config.workers);
    let mut stats = Vec::with_capacity(config.workers);
    let mut before = Vec::new();
    let mut after = Vec::new();
    for task in tasks {
        let capture = task.await??;
        artifacts.push(capture.artifacts);
        stats.push(capture.stats);
        before.push(capture.locals_before_salt);
        after.push(capture.locals_after_salt);
    }
    progress.finish_and_clear();
    report.capture_ms = started.elapsed().as_secs_f64() * 1e3;

    let spread = |field: fn(&WorkerCaptureStats) -> usize| {
        Spread::of(stats.iter().map(|stats| field(stats) as u64))
    };
    report.plays_assigned_per_worker = spread(|stats| stats.plays_assigned);
    report.plays_started_per_worker = spread(|stats| stats.plays_started);
    report.plays_completed_per_worker = spread(|stats| stats.plays_completed);
    report.requests_captured_per_worker = spread(|stats| stats.requests_captured);
    report.request_blocks_per_worker = spread(|stats| stats.request_blocks);
    report.stored_blocks_per_worker = spread(|stats| stats.stored_blocks);
    report.removed_blocks_per_worker = spread(|stats| stats.removed_blocks);
    report.phase_offset_us = Spread::of(stats.iter().map(|stats| stats.phase_offset_us));
    report.lanes_total = stats.iter().map(|stats| stats.lanes).sum();
    report.lanes_exhausted_before_cap = stats.iter().map(|stats| stats.lanes_exhausted).sum();
    report.requests_completed = spread(|stats| stats.requests_completed).total;
    report.rejected_outputs = spread(|stats| stats.rejected_outputs).total;
    report.capped_delays = spread(|stats| stats.capped_delays).total;
    report.total_request_blocks = report.request_blocks_per_worker.total;
    report.query_slab_gate_blocks = QUERY_SLAB_GATE_BLOCKS;

    if config.collision_stats {
        let started = std::time::Instant::now();
        let (before_shared, _) =
            tokio::task::spawn_blocking(move || count_shared_values(&before)).await?;
        let (after_shared, after_distinct) =
            tokio::task::spawn_blocking(move || count_shared_values(&after)).await?;
        report.shared_local_hashes_before_salt = Some(before_shared);
        report.shared_local_hashes_after_salt = Some(after_shared);
        report.distinct_local_hashes_after_salt = Some(after_distinct);
        report.collision_stats_ms = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "Cross-worker shared local hashes: {before_shared} before salt, {after_shared} after"
        );
        ensure!(
            after_shared == 0,
            "{after_shared} local hashes are still shared across workers after the salt"
        );
    }

    let started_per_lane_ok = stats.iter().all(|stats| {
        stats.plays_started >= stats.lanes && stats.plays_started <= stats.plays_assigned
    });
    ensure!(
        started_per_lane_ok,
        "closed-loop supply: some worker started fewer plays than lanes"
    );
    if report.lanes_exhausted_before_cap > 0 && !config.allow_exhausted_lanes {
        bail!(
            "closed-loop supply: {} of {} lanes ran out of plays before the {} ms cap; raise --agentic-plays-per-worker or lower --agentic-sim-ms",
            report.lanes_exhausted_before_cap,
            report.lanes_total,
            config.sim_ms
        );
    }
    if report.rejected_outputs > 0 {
        bail!(
            "closed-loop capture rejected {} requests; the engine cannot hold them",
            report.rejected_outputs
        );
    }
    ensure!(
        report.total_request_blocks <= QUERY_SLAB_GATE_BLOCKS,
        "agentic corpus has {} query blocks, above the u32 query-slab gate {QUERY_SLAB_GATE_BLOCKS}; lower --agentic-sim-ms",
        report.total_request_blocks
    );
    Ok(artifacts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_mocker::loadgen::{
        AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependencyRelation,
        AgenticDependencyTrigger, AgenticHashIdScope, AgenticSourceProvenance,
    };
    use std::collections::{BTreeMap, BTreeSet};

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/agentic/weka-two-plays.jsonl"
    );

    fn header(block_size: usize) -> AgenticMooncakeHeader {
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.to_string(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "test".to_string(),
                digest: "test".to_string(),
            },
        }
    }

    fn row(play: &str, index: usize, hashes: Vec<u64>, input: usize) -> AgenticMooncakeRow {
        AgenticMooncakeRow {
            request_id: format!("{play}:request:{index}"),
            play_id: format!("{play}:play"),
            source_play_ordinal: None,
            session_id: format!("{play}:session"),
            model: "model".to_string(),
            input_length: Some(input),
            output_length: Some(4),
            hash_ids: Some(hashes),
            not_before_ms: 1_000.0 * index as f64,
            dependencies: (index > 0)
                .then(|| AgenticDependency {
                    request_id: format!("{play}:request:{}", index - 1),
                    trigger: AgenticDependencyTrigger::Completion,
                    delay_ms: 10_000_000.0,
                    relation: AgenticDependencyRelation::Sequence,
                })
                .into_iter()
                .collect(),
            ..AgenticMooncakeRow::default()
        }
    }

    fn config(workers: usize) -> AgenticCorpusConfig {
        AgenticCorpusConfig {
            workers,
            plays_per_worker: 2,
            lanes_per_worker: 1,
            sim_ms: 60_000,
            idle_cap_ms: 300_000.0,
            phase_spread: 0.05,
            length_factor: 1,
            seed: 42,
            allow_exhausted_lanes: true,
            collision_stats: true,
        }
    }

    #[test]
    fn relabel_preserves_within_copy_equality_and_separates_copies() {
        let source = row("p", 1, vec![1, 2, 3, 2], 16);
        let copies = (0..4)
            .map(|copy| relabel_row(&source, copy))
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        for (copy, relabeled) in copies.iter().enumerate() {
            let hashes = relabeled.hash_ids.as_ref().unwrap();
            assert_eq!(hashes[1], hashes[3], "equal source hashes stay equal");
            assert_ne!(hashes[0], hashes[1]);
            assert_eq!(relabeled.request_id, format!("c{copy}:p:request:1"));
            assert_eq!(
                relabeled.dependencies[0].request_id,
                format!("c{copy}:p:request:0")
            );
            for hash in hashes.iter().collect::<BTreeSet<_>>() {
                assert!(seen.insert(*hash), "copies must be hash-disjoint");
            }
        }
    }

    /// Canonical form of a row set: labels reduced to their within-play source IDs and hash
    /// IDs renamed by first appearance in a canonical traversal order.
    fn canonical(rows: &[AgenticMooncakeRow], play_key: impl Fn(&str) -> usize) -> Vec<String> {
        let source_id = |id: &str| id.rsplit(":request:").next().unwrap().to_string();
        let session = |id: &str| id.rsplit(":session:").next().unwrap().to_string();
        let mut ordered = rows
            .iter()
            .map(|row| ((play_key(&row.play_id), source_id(&row.request_id)), row))
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.0.cmp(&right.0));
        let mut renamed = BTreeMap::new();
        ordered
            .into_iter()
            .map(|((play, id), row)| {
                let hashes = row
                    .hash_ids
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|hash| {
                        let next = renamed.len();
                        *renamed.entry(*hash).or_insert(next)
                    })
                    .collect::<Vec<_>>();
                let deps = row
                    .dependencies
                    .iter()
                    .map(|dep| {
                        format!(
                            "{}/{:?}/{:?}/{}",
                            source_id(&dep.request_id),
                            dep.trigger,
                            dep.relation,
                            dep.delay_ms
                        )
                    })
                    .collect::<Vec<_>>();
                format!(
                    "{play}|{id}|{}|{:?}|{:?}|{}|{hashes:?}|{deps:?}",
                    session(&row.session_id),
                    row.input_length,
                    row.output_length,
                    row.not_before_ms
                )
            })
            .collect()
    }

    #[test]
    fn relabel_matches_native_line_duplication() {
        const COPIES: usize = 3;
        let pool = AgenticPool::import_weka(Path::new(FIXTURE)).unwrap();
        let plays = pool.plays.len();
        assert_eq!(plays, 2);
        let ours = (0..COPIES)
            .flat_map(|copy| {
                pool.plays
                    .iter()
                    .flat_map(move |play| play.rows.iter().map(move |row| relabel_row(row, copy)))
            })
            .collect::<Vec<_>>();
        let play_ids = pool
            .plays
            .iter()
            .map(|play| play.play_id.clone())
            .collect::<Vec<_>>();
        let our_key = |play_id: &str| {
            let (copy, original) = play_id[1..].split_once(':').unwrap();
            let play = play_ids.iter().position(|id| id == original).unwrap();
            copy.parse::<usize>().unwrap() * plays + play
        };

        let fixture = std::fs::read_to_string(FIXTURE).unwrap();
        let dir = std::env::temp_dir().join(format!("agentic-dup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let duplicated = dir.join("dup.jsonl");
        std::fs::write(&duplicated, fixture.repeat(COPIES)).unwrap();
        let native = AgenticPool::import_weka(&duplicated).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let native_rows = native
            .plays
            .iter()
            .flat_map(|play| play.rows.iter().cloned())
            .collect::<Vec<_>>();
        // The importer orders plays by line, and line k is copy k / plays of play k % plays.
        let native_order = native
            .plays
            .iter()
            .map(|play| play.play_id.clone())
            .collect::<Vec<_>>();
        let native_key = |play_id: &str| native_order.iter().position(|id| id == play_id).unwrap();

        assert_eq!(ours.len(), native_rows.len());
        assert_eq!(
            canonical(&ours, our_key),
            canonical(&native_rows, native_key)
        );
    }

    #[test]
    fn worker_rows_cap_delays_reset_schedule_and_build() {
        let pool = AgenticPool::from_rows(
            header(4),
            (0..3).map(|index| row("p", index, vec![1, 2], 8)).collect(),
        )
        .unwrap();
        let instances = [
            PlayInstance { copy: 0, play: 0 },
            PlayInstance { copy: 1, play: 0 },
        ];
        let (rows, capped) = worker_rows(&pool, &instances, &config(1)).unwrap();
        assert_eq!(capped, 4);
        assert!(rows.iter().all(|row| row.not_before_ms == 0.0));
        let trace = AgenticTrace::from_agentic_mooncake_rows(header(4), rows).unwrap();
        assert_eq!(trace.play_count(), 2);
        for node in trace.nodes() {
            assert_eq!(node.not_before_ms(), 0.0);
            assert!(
                node.dependencies()
                    .iter()
                    .all(|dep| dep.delay_ms == 300_000.0)
            );
        }
    }

    #[test]
    fn length_factor_keeps_the_hash_count_contract() {
        for (input, factor) in [(16usize, 2usize), (15, 2), (13, 3), (4, 4), (1, 3)] {
            let blocks = input.div_ceil(4);
            let mut source = row("p", 0, (0..blocks as u64).collect(), input);
            expand_row_length(&mut source, 4, factor).unwrap();
            assert_eq!(source.input_length, Some(input * factor));
            assert_eq!(
                source.hash_ids.as_ref().unwrap().len(),
                (input * factor).div_ceil(4)
            );
            AgenticTrace::from_agentic_mooncake_rows(header(4), vec![source]).unwrap();
        }
        // Shared full blocks stay shared after expansion.
        let mut left = row("a", 0, vec![7, 8], 8);
        let mut right = row("b", 0, vec![7, 9], 6);
        expand_row_length(&mut left, 4, 2).unwrap();
        expand_row_length(&mut right, 4, 2).unwrap();
        assert_eq!(
            left.hash_ids.as_ref().unwrap()[..2],
            right.hash_ids.as_ref().unwrap()[..2]
        );
    }

    #[test]
    fn deal_is_balanced_and_deterministic() {
        let dealt = deal_play_instances(5, 3, 4, 42);
        assert_eq!(dealt, deal_play_instances(5, 3, 4, 42));
        let sizes = dealt.iter().map(Vec::len).collect::<Vec<_>>();
        assert_eq!(sizes, vec![4, 4, 4, 3]);
        let all = dealt.into_iter().flatten().collect::<BTreeSet<_>>();
        assert_eq!(all.len(), 15);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn salt_removes_cross_worker_collisions_and_keeps_chains() {
        let pool = std::sync::Arc::new(
            AgenticPool::from_rows(
                header(4),
                (0..4)
                    .flat_map(|play| {
                        (0..3).map(move |index| {
                            row(&format!("p{play}"), index, vec![play, 100 + play], 8)
                        })
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let engine_args = MockEngineArgs::builder()
            .num_gpu_blocks(64)
            .block_size(4)
            .speedup_ratio(1.0)
            .enable_prefix_caching(true)
            .max_num_batched_tokens(None)
            .max_num_seqs(None)
            .build()
            .unwrap();
        let mut report = AgenticPrepReport::default();
        let artifacts = generate_agentic_artifacts(pool, &config(2), engine_args, &mut report)
            .await
            .unwrap();
        assert_eq!(artifacts.len(), 2);
        assert!(report.shared_local_hashes_before_salt.unwrap() > 0);
        assert_eq!(report.shared_local_hashes_after_salt, Some(0));
        for (worker, artifact) in artifacts.iter().enumerate() {
            check_stored_chains(worker, artifact).unwrap();
            assert!(!artifact.requests.is_empty());
            for request in &artifact.requests {
                let hashes = &request.replay_hashes;
                let mut parent = None;
                for (local, sequence) in hashes
                    .local_block_hashes
                    .iter()
                    .zip(&hashes.sequence_hashes)
                {
                    assert_eq!(*sequence, chain(parent, *local));
                    parent = Some(*sequence);
                }
            }
        }
    }
}
