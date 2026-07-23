// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, bail};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, WorkerWithDpRank};
use dynamo_mocker::loadgen::{SessionTrace, Trace};
use dynamo_mocker::replay::ReplayWorkerArtifacts;
use rustc_hash::{FxHashMap, FxHashSet};
use sha2::{Digest, Sha256};

pub use super::dc_ckf_metadata::{
    DcCkfCapacityMetadata, DcCkfCorpusMetadata, DcCkfCorpusSpec, DcCkfPoolCapacity,
    DcCkfPoolCorpusMetadata, DcCkfSourceTopology,
};

#[derive(Debug, Clone)]
pub struct PreparedDcCkfCorpus {
    pub worker_traces: Vec<Trace>,
    pub metadata: DcCkfCorpusMetadata,
}

impl PreparedDcCkfCorpus {
    pub fn measure_capacity(
        &self,
        artifacts: &[ReplayWorkerArtifacts],
    ) -> anyhow::Result<DcCkfCapacityMetadata> {
        measure_dc_ckf_capacity(&self.metadata, artifacts)
    }
}

pub fn prepare_dc_ckf_corpus(spec: &DcCkfCorpusSpec) -> anyhow::Result<PreparedDcCkfCorpus> {
    validate_spec(spec)?;
    let trace_sha256 = sha256_file(&spec.trace_path)?;
    if let Some(expected) = spec.expected_sha256.as_deref() {
        verify_sha256(expected, &trace_sha256)?;
    }

    let original =
        Trace::from_mooncake(&spec.trace_path, spec.trace_block_size).with_context(|| {
            format!(
                "failed to load Mooncake trace {}",
                spec.trace_path.display()
            )
        })?;
    let original_session_count = original.sessions.len();
    let prepared = checked_expand_and_duplicate(
        original,
        spec.prefix_depth_factor,
        spec.trace_duplication_factor,
    )?;
    let prepared_session_count = prepared.sessions.len();

    let source_count = spec
        .dc_count
        .checked_mul(spec.workers_per_dc)
        .context("DC CKF source count overflow")?;
    let mut source_sessions = (0..source_count)
        .map(|_| Vec::<SessionTrace>::new())
        .collect::<Vec<_>>();

    for (session_index, session) in prepared.sessions.into_iter().enumerate() {
        let dc_ordinal = session_index % spec.dc_count;
        let worker_ordinal = (session_index / spec.dc_count) % spec.workers_per_dc;
        let source_index = source_index(dc_ordinal, worker_ordinal, spec.workers_per_dc)?;
        source_sessions[source_index].push(session);
    }

    let mut global_hashes = FxHashSet::default();
    let mut sources = Vec::with_capacity(source_count);
    let mut pools = (0..spec.dc_count)
        .map(|dc_ordinal| DcCkfPoolCorpusMetadata {
            dc_ordinal,
            endpoint_ordinal: spec.endpoint_ordinal,
            session_count: 0,
            turn_count: 0,
            trace_distinct_hash_upper_bound: 0,
        })
        .collect::<Vec<_>>();
    let mut pool_hashes = (0..spec.dc_count)
        .map(|_| FxHashSet::default())
        .collect::<Vec<FxHashSet<u32>>>();
    let mut turn_count = 0_usize;
    let mut hash_reference_count = 0_usize;

    for (source_index, sessions) in source_sessions.iter().enumerate() {
        let dc_ordinal = source_index / spec.workers_per_dc;
        let worker_ordinal = source_index % spec.workers_per_dc;
        let source_turn_count = sessions.iter().try_fold(0_usize, |count, session| {
            count
                .checked_add(session.turns.len())
                .context("Mooncake source turn count overflow")
        })?;
        let mut source_hashes = FxHashSet::default();
        for hash in sessions
            .iter()
            .flat_map(|session| &session.turns)
            .flat_map(|turn| &turn.hash_ids)
            .copied()
        {
            source_hashes.insert(hash);
            pool_hashes[dc_ordinal].insert(hash);
            global_hashes.insert(hash);
            hash_reference_count = hash_reference_count
                .checked_add(1)
                .context("Mooncake hash reference count overflow")?;
        }
        turn_count = turn_count
            .checked_add(source_turn_count)
            .context("Mooncake turn count overflow")?;
        pools[dc_ordinal].session_count = pools[dc_ordinal]
            .session_count
            .checked_add(sessions.len())
            .context("Mooncake pool session count overflow")?;
        pools[dc_ordinal].turn_count = pools[dc_ordinal]
            .turn_count
            .checked_add(source_turn_count)
            .context("Mooncake pool turn count overflow")?;
        sources.push(DcCkfSourceTopology {
            source_index,
            dc_ordinal,
            endpoint_ordinal: spec.endpoint_ordinal,
            worker_ordinal,
            member: stable_member(
                spec.endpoint_ordinal,
                dc_ordinal,
                worker_ordinal,
                spec.default_dp_rank,
            )?,
            session_count: sessions.len(),
            turn_count: source_turn_count,
            trace_distinct_hash_upper_bound: source_hashes.len(),
        });
    }
    for (pool, hashes) in pools.iter_mut().zip(pool_hashes) {
        pool.trace_distinct_hash_upper_bound = hashes.len();
    }

    let worker_traces = source_sessions
        .into_iter()
        .map(|sessions| Trace {
            block_size: spec.trace_block_size,
            sessions,
        })
        .collect();

    Ok(PreparedDcCkfCorpus {
        worker_traces,
        metadata: DcCkfCorpusMetadata {
            trace_path: spec.trace_path.clone(),
            trace_sha256,
            trace_block_size: spec.trace_block_size,
            prefix_depth_factor: spec.prefix_depth_factor,
            trace_duplication_factor: spec.trace_duplication_factor,
            original_session_count,
            prepared_session_count,
            turn_count,
            hash_reference_count,
            trace_distinct_hash_upper_bound: global_hashes.len(),
            dc_count: spec.dc_count,
            workers_per_dc: spec.workers_per_dc,
            endpoint_ordinal: spec.endpoint_ordinal,
            sources,
            pools,
        },
    })
}

pub fn measure_dc_ckf_capacity(
    topology: &DcCkfCorpusMetadata,
    artifacts: &[ReplayWorkerArtifacts],
) -> anyhow::Result<DcCkfCapacityMetadata> {
    if artifacts.len() != topology.sources.len() {
        bail!(
            "artifact/source count mismatch: artifacts={}, sources={}",
            artifacts.len(),
            topology.sources.len()
        );
    }

    let mut ordered = Vec::new();
    for (source, artifact) in topology.sources.iter().zip(artifacts) {
        for (event_ordinal, timed) in artifact.kv_events.iter().enumerate() {
            ordered.push((
                timed.timestamp_us,
                source.source_index,
                event_ordinal,
                source.dc_ordinal,
                source.member.worker_id,
                &timed.event,
            ));
        }
    }
    // Same-source FIFO is authoritative. The source index only makes unrelated events with equal
    // timestamps deterministic; it does not claim a production cross-source order.
    ordered.sort_by_key(|(timestamp, source, ordinal, ..)| (*timestamp, *source, *ordinal));

    let mut trackers = (0..topology.dc_count)
        .map(|_| PoolCapacityTracker::default())
        .collect::<Vec<_>>();
    for (_, _, _, dc_ordinal, worker_id, event) in ordered {
        let tracker = trackers
            .get_mut(dc_ordinal)
            .with_context(|| format!("source references out-of-range DC ordinal {dc_ordinal}"))?;
        tracker.apply(WorkerWithDpRank::new(worker_id, event.dp_rank), event)?;
    }

    let mut pools = Vec::with_capacity(topology.dc_count);
    for (dc_ordinal, tracker) in trackers.into_iter().enumerate() {
        pools.push(DcCkfPoolCapacity {
            dc_ordinal,
            endpoint_ordinal: topology.endpoint_ordinal,
            measured_peak_active_distinct_hashes: tracker.peak_active_distinct,
            final_active_distinct_hashes: tracker.global_degrees.len(),
            recommended_distinct_hash_capacity: capacity_with_headroom(
                tracker.peak_active_distinct,
            )?,
            event_count: tracker.event_count,
        });
    }
    Ok(DcCkfCapacityMetadata {
        headroom_percent: 20,
        pools,
    })
}

pub fn capacity_with_headroom(peak_active_distinct: usize) -> anyhow::Result<usize> {
    peak_active_distinct
        .checked_add(peak_active_distinct.div_ceil(5))
        .context("DC CKF capacity plus 20% headroom overflow")
}

fn validate_spec(spec: &DcCkfCorpusSpec) -> anyhow::Result<()> {
    if spec.trace_block_size == 0 {
        bail!("trace_block_size must be positive");
    }
    if spec.prefix_depth_factor == 0 {
        bail!("prefix_depth_factor must be positive");
    }
    if spec.trace_duplication_factor == 0 {
        bail!("trace_duplication_factor must be positive");
    }
    if spec.dc_count == 0 {
        bail!("dc_count must be positive");
    }
    if spec.workers_per_dc == 0 {
        bail!("workers_per_dc must be positive");
    }
    stable_member(
        spec.endpoint_ordinal,
        spec.dc_count - 1,
        spec.workers_per_dc - 1,
        spec.default_dp_rank,
    )?;
    Ok(())
}

fn checked_expand_and_duplicate(
    mut trace: Trace,
    prefix_depth_factor: usize,
    copies: usize,
) -> anyhow::Result<Trace> {
    let factor = u32::try_from(prefix_depth_factor).context("prefix depth does not fit in u32")?;
    for session in &mut trace.sessions {
        for turn in &mut session.turns {
            turn.input_length = turn
                .input_length
                .checked_mul(prefix_depth_factor)
                .context("Mooncake input length expansion overflow")?;
            let expanded_len = turn
                .hash_ids
                .len()
                .checked_mul(prefix_depth_factor)
                .context("Mooncake hash prefix expansion length overflow")?;
            let mut expanded = Vec::with_capacity(expanded_len);
            for hash in turn.hash_ids.iter().copied() {
                let base = hash
                    .checked_mul(factor)
                    .context("Mooncake hash prefix expansion overflow")?;
                for offset in 0..factor {
                    let expanded_hash = base
                        .checked_add(offset)
                        .context("Mooncake hash prefix expansion overflow")?;
                    expanded.push(expanded_hash);
                }
            }
            turn.hash_ids = expanded;
        }
    }

    let max_hash = trace
        .sessions
        .iter()
        .flat_map(|session| &session.turns)
        .flat_map(|turn| &turn.hash_ids)
        .copied()
        .max()
        .unwrap_or(0);
    let offset_base = max_hash
        .checked_add(1)
        .context("Mooncake hash duplication offset overflow")?;
    let original_sessions = trace.sessions;
    let duplicated_count = original_sessions
        .len()
        .checked_mul(copies)
        .context("Mooncake duplicated session count overflow")?;
    let mut sessions = Vec::with_capacity(duplicated_count);

    for copy_index in 0..copies {
        let copy_index = u32::try_from(copy_index).context("copy index does not fit in u32")?;
        let offset = offset_base
            .checked_mul(copy_index)
            .context("Mooncake hash duplication offset overflow")?;
        for session in &original_sessions {
            let mut duplicated = session.clone();
            if copies > 1 {
                duplicated.session_id = format!("{}:copy_{copy_index}", session.session_id);
            }
            for turn in &mut duplicated.turns {
                for hash in &mut turn.hash_ids {
                    *hash = hash
                        .checked_add(offset)
                        .context("Mooncake duplicated hash overflow")?;
                }
            }
            sessions.push(duplicated);
        }
    }
    trace.sessions = sessions;
    Ok(trace)
}

fn stable_member(
    endpoint_ordinal: usize,
    dc_ordinal: usize,
    worker_ordinal: usize,
    dp_rank: u32,
) -> anyhow::Result<WorkerWithDpRank> {
    let endpoint = u16::try_from(endpoint_ordinal)
        .context("endpoint ordinal exceeds the stable worker-ID namespace")?;
    let dc =
        u16::try_from(dc_ordinal).context("DC ordinal exceeds the stable worker-ID namespace")?;
    let worker = u32::try_from(worker_ordinal)
        .context("worker ordinal exceeds the stable worker-ID namespace")?;
    let worker_id = (u64::from(endpoint) << 48) | (u64::from(dc) << 32) | u64::from(worker);
    Ok(WorkerWithDpRank::new(worker_id, dp_rank))
}

fn source_index(
    dc_ordinal: usize,
    worker_ordinal: usize,
    workers_per_dc: usize,
) -> anyhow::Result<usize> {
    dc_ordinal
        .checked_mul(workers_per_dc)
        .and_then(|base| base.checked_add(worker_ordinal))
        .context("DC CKF source index overflow")
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let file = File::open(path)
        .with_context(|| format!("failed to open trace for SHA-256: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to hash trace {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_sha256(expected: &str, actual: &str) -> anyhow::Result<()> {
    let normalized = expected
        .strip_prefix("sha256:")
        .unwrap_or(expected)
        .to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("expected trace SHA-256 must contain exactly 64 hexadecimal digits");
    }
    if normalized != actual {
        bail!("trace SHA-256 mismatch: expected {normalized}, got {actual}");
    }
    Ok(())
}

#[derive(Default)]
struct PoolCapacityTracker {
    members: FxHashMap<WorkerWithDpRank, FxHashSet<u64>>,
    global_degrees: FxHashMap<u64, u32>,
    peak_active_distinct: usize,
    event_count: usize,
}

impl PoolCapacityTracker {
    fn apply(&mut self, member: WorkerWithDpRank, event: &KvCacheEvent) -> anyhow::Result<()> {
        match &event.data {
            KvCacheEventData::Stored(stored) => {
                for block in &stored.blocks {
                    self.store(member, block.block_hash.0)?;
                }
            }
            KvCacheEventData::Removed(removed) => {
                for hash in &removed.block_hashes {
                    self.remove(member, hash.0)?;
                }
            }
            KvCacheEventData::Cleared => self.clear(member)?,
        }
        self.event_count = self
            .event_count
            .checked_add(1)
            .context("DC CKF event count overflow")?;
        self.peak_active_distinct = self.peak_active_distinct.max(self.global_degrees.len());
        Ok(())
    }

    fn store(&mut self, member: WorkerWithDpRank, hash: u64) -> anyhow::Result<()> {
        if !self.members.entry(member).or_default().insert(hash) {
            return Ok(());
        }
        let degree = self.global_degrees.entry(hash).or_default();
        *degree = degree
            .checked_add(1)
            .context("DC CKF ownership degree overflow while sizing corpus")?;
        Ok(())
    }

    fn remove(&mut self, member: WorkerWithDpRank, hash: u64) -> anyhow::Result<()> {
        let Some(hashes) = self.members.get_mut(&member) else {
            return Ok(());
        };
        if !hashes.remove(&hash) {
            return Ok(());
        }
        if hashes.is_empty() {
            self.members.remove(&member);
        }
        self.decrement(hash)
    }

    fn clear(&mut self, member: WorkerWithDpRank) -> anyhow::Result<()> {
        let Some(hashes) = self.members.remove(&member) else {
            return Ok(());
        };
        for hash in hashes {
            self.decrement(hash)?;
        }
        Ok(())
    }

    fn decrement(&mut self, hash: u64) -> anyhow::Result<()> {
        let Some(degree) = self.global_degrees.get_mut(&hash) else {
            bail!("capacity tracker lost ownership degree for hash {hash}");
        };
        if *degree == 1 {
            self.global_degrees.remove(&hash);
        } else {
            *degree -= 1;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use std::io::Write;
    use std::path::PathBuf;

    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash,
    };
    use dynamo_mocker::replay::ReplayTimedKvEvent;
    use tempfile::NamedTempFile;

    use super::*;

    fn write_trace(rows: &[serde_json::Value]) -> anyhow::Result<NamedTempFile> {
        let mut file = NamedTempFile::new()?;
        for row in rows {
            writeln!(file, "{row}")?;
        }
        Ok(file)
    }

    fn stored(event_id: u64, dp_rank: u32, hashes: &[u64]) -> KvCacheEvent {
        KvCacheEvent {
            event_id,
            dp_rank,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: hashes
                    .iter()
                    .map(|hash| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(*hash),
                        tokens_hash: LocalBlockHash(*hash),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
        }
    }

    fn removed(event_id: u64, dp_rank: u32, hashes: &[u64]) -> KvCacheEvent {
        KvCacheEvent {
            event_id,
            dp_rank,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: hashes
                    .iter()
                    .copied()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            }),
        }
    }

    fn timed(timestamp_us: u64, event: KvCacheEvent) -> ReplayTimedKvEvent {
        ReplayTimedKvEvent {
            event,
            storage_tier: Default::default(),
            timestamp_us,
        }
    }

    #[test]
    fn preparation_preserves_sessions_and_uses_dc_then_worker_round_robin() -> anyhow::Result<()> {
        let file = write_trace(&[
            serde_json::json!({"timestamp": 0, "session_id": "a", "input_length": 2, "hash_ids": [1, 2], "output_length": 1}),
            serde_json::json!({"timestamp": 1, "session_id": "b", "input_length": 1, "hash_ids": [3], "output_length": 1}),
            serde_json::json!({"timestamp": 2, "session_id": "c", "input_length": 1, "hash_ids": [4], "output_length": 1}),
            serde_json::json!({"timestamp": 3, "session_id": "d", "input_length": 1, "hash_ids": [5], "output_length": 1}),
            serde_json::json!({"timestamp": 4, "session_id": "a", "input_length": 3, "hash_ids": [1, 2, 6], "output_length": 1}),
        ])?;
        let mut spec = DcCkfCorpusSpec::new(file.path(), 2, 2);
        spec.trace_block_size = 1;
        spec.trace_duplication_factor = 2;
        let prepared = prepare_dc_ckf_corpus(&spec)?;

        assert_eq!(prepared.worker_traces.len(), 4);
        let session_ids = prepared
            .worker_traces
            .iter()
            .map(|trace| {
                trace
                    .sessions
                    .iter()
                    .map(|session| session.session_id.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(session_ids[0], ["a:copy_0", "a:copy_1"]);
        assert_eq!(session_ids[1], ["c:copy_0", "c:copy_1"]);
        assert_eq!(session_ids[2], ["b:copy_0", "b:copy_1"]);
        assert_eq!(session_ids[3], ["d:copy_0", "d:copy_1"]);
        assert_eq!(prepared.worker_traces[0].sessions[0].turns.len(), 2);
        assert_eq!(prepared.metadata.prepared_session_count, 8);
        assert_eq!(prepared.metadata.trace_distinct_hash_upper_bound, 12);
        serde_json::to_string(&prepared.metadata)?;
        Ok(())
    }

    #[test]
    fn preparation_canonicalizes_raw_hashes_before_duplication() -> anyhow::Result<()> {
        let file = write_trace(&[serde_json::json!({
            "timestamp": 0,
            "input_length": 1,
            "hash_ids": [u64::from(u32::MAX)],
            "output_length": 1
        })])?;
        let mut spec = DcCkfCorpusSpec::new(file.path(), 1, 1);
        spec.trace_block_size = 1;
        spec.trace_duplication_factor = 2;
        let prepared = prepare_dc_ckf_corpus(&spec)?;
        let hashes = prepared.worker_traces[0]
            .sessions
            .iter()
            .flat_map(|session| &session.turns)
            .flat_map(|turn| &turn.hash_ids)
            .copied()
            .collect::<FxHashSet<_>>();
        assert_eq!(hashes, FxHashSet::from_iter([0, 1]));
        Ok(())
    }

    #[test]
    fn sha256_verification_fails_before_preparation() -> anyhow::Result<()> {
        let file = write_trace(&[serde_json::json!({
            "timestamp": 0,
            "input_length": 1,
            "hash_ids": [1],
            "output_length": 1
        })])?;
        let mut spec = DcCkfCorpusSpec::new(file.path(), 1, 1);
        spec.expected_sha256 = Some("00".repeat(32));
        let error = prepare_dc_ckf_corpus(&spec).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
        Ok(())
    }

    #[test]
    fn event_replay_measures_distinct_peak_with_shared_ownership() -> anyhow::Result<()> {
        let topology = DcCkfCorpusMetadata {
            trace_path: PathBuf::from("trace.jsonl"),
            trace_sha256: "00".repeat(32),
            trace_block_size: 512,
            prefix_depth_factor: 1,
            trace_duplication_factor: 1,
            original_session_count: 2,
            prepared_session_count: 2,
            turn_count: 2,
            hash_reference_count: 2,
            trace_distinct_hash_upper_bound: 2,
            dc_count: 1,
            workers_per_dc: 2,
            endpoint_ordinal: 0,
            sources: vec![
                DcCkfSourceTopology {
                    source_index: 0,
                    dc_ordinal: 0,
                    endpoint_ordinal: 0,
                    worker_ordinal: 0,
                    member: stable_member(0, 0, 0, 0)?,
                    session_count: 1,
                    turn_count: 1,
                    trace_distinct_hash_upper_bound: 1,
                },
                DcCkfSourceTopology {
                    source_index: 1,
                    dc_ordinal: 0,
                    endpoint_ordinal: 0,
                    worker_ordinal: 1,
                    member: stable_member(0, 0, 1, 0)?,
                    session_count: 1,
                    turn_count: 1,
                    trace_distinct_hash_upper_bound: 1,
                },
            ],
            pools: vec![DcCkfPoolCorpusMetadata {
                dc_ordinal: 0,
                endpoint_ordinal: 0,
                session_count: 2,
                turn_count: 2,
                trace_distinct_hash_upper_bound: 2,
            }],
        };
        let artifacts = vec![
            ReplayWorkerArtifacts {
                kv_events: vec![
                    timed(1, stored(1, 0, &[10, 20])),
                    timed(3, removed(2, 0, &[10, 20])),
                ],
                ..Default::default()
            },
            ReplayWorkerArtifacts {
                kv_events: vec![
                    timed(2, stored(1, 0, &[20, 30])),
                    timed(
                        4,
                        KvCacheEvent {
                            event_id: 2,
                            dp_rank: 0,
                            data: KvCacheEventData::Cleared,
                        },
                    ),
                ],
                ..Default::default()
            },
        ];

        let measured = measure_dc_ckf_capacity(&topology, &artifacts)?;
        assert_eq!(measured.pools[0].measured_peak_active_distinct_hashes, 3);
        assert_eq!(measured.pools[0].recommended_distinct_hash_capacity, 4);
        assert_eq!(measured.pools[0].final_active_distinct_hashes, 0);
        Ok(())
    }
}
