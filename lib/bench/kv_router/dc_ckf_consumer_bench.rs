// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage-2 benchmark for endpoint-scoped global CKF ingestion and queries.
//!
//! The serialized corpus contains barrier snapshots plus a finite, contiguous sequence of
//! absolute bucket-image deltas. The finite stream ends at its starting physical state, so a
//! measured run can repeat those frozen images while assigning fresh contiguous sequence numbers.
//! This keeps the consumer continuously busy without introducing producer work into this stage.

use std::collections::BTreeSet;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, ensure};
use clap::{Parser, ValueEnum};
use dynamo_bench::kv_router_common::dc_ckf_shared::DcCkfCorpusMetadata;
use dynamo_kv_router::indexer::cuckoo::{
    CacheDomainId, CacheDomainIdentity, CkfConfig, ConsumerDrainMarker, ConsumerInstanceId,
    DcCkfDelta, DcCkfDeltaSink, DcCkfPublisher, DcCkfSnapshot, DcCkfState, DcId, EndpointId,
    GlobalCkfIndexer, GlobalCkfIngestOutcome, GlobalCkfIngestionPool, GlobalCkfIngestionPoolConfig,
    GlobalCkfLaneOwner, GlobalCkfManifest, LaneLease, PrefixSearchConfig, ProducerIdentity,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
    compute_seq_hash_for_block,
};
use serde::{Deserialize, Serialize};

const CORPUS_SCHEMA_VERSION: u32 = 2;
const RESULT_SCHEMA_VERSION: u32 = 2;
const MAX_LANES: usize = 16;
const DEFAULT_EXPECTED_BLOCKS_PER_LANE: usize = 4_096;
const DEFAULT_BASELINE_HASHES_PER_LANE: usize = 2_048;
const DEFAULT_DELTA_FRAMES_PER_LANE: usize = 4_096;
const LATENCY_SAMPLE_MASK: u64 = 63;

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum BenchmarkMode {
    /// Authoritative stage-2 measurement: replay deltas while queries run concurrently.
    Mixed,
    /// Attribute snapshot, admission, sequencing, and packed-bucket ingestion without queries.
    IngestionOnly,
    /// Attribute query cost over stable ready lanes without delta ingestion.
    StableQueryOnly,
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Benchmark endpoint-scoped global CKF ingestion and concurrent queries"
)]
struct Args {
    /// Load a frozen JSON corpus instead of generating a deterministic one.
    #[arg(long)]
    corpus_input: Option<PathBuf>,

    /// Write the normalized generated or loaded corpus as JSON before benchmarking.
    #[arg(long)]
    corpus_output: Option<PathBuf>,

    /// Optional dc_ckf_shared topology metadata used to size deterministic fallback data.
    #[arg(long)]
    topology_metadata: Option<PathBuf>,

    /// Number of endpoint lanes to activate (1, 2, 4, 8, or 16).
    #[arg(long, default_value_t = 4)]
    lanes: usize,

    /// Number of lane-sticky ingestion workers (1, 2, 4, or 8).
    #[arg(long, default_value_t = 4)]
    ingestion_workers: usize,

    /// Number of concurrent query issuer threads.
    #[arg(long, default_value_t = 2)]
    query_issuers: usize,

    /// Authoritative mixed mode, ingestion-only attribution, or stable-query attribution.
    #[arg(long, value_enum, default_value_t = BenchmarkMode::Mixed)]
    mode: BenchmarkMode,

    /// Measured steady-state duration in seconds.
    #[arg(long = "duration", default_value_t = 5.0)]
    duration_seconds: f64,

    /// Fresh-process-equivalent warmup duration in seconds.
    #[arg(long = "warmup", default_value_t = 1.0)]
    warmup_seconds: f64,

    /// Bounded queue capacity per ingestion worker.
    #[arg(long, default_value_t = 256)]
    queue_capacity: usize,

    /// Maximum dirty-to-applied age before the real ingestion pool retires a lane.
    #[arg(long, default_value_t = 10)]
    max_dirty_age_ms: u64,

    /// Synthetic fallback CKF capacity per lane.
    #[arg(long, default_value_t = DEFAULT_EXPECTED_BLOCKS_PER_LANE)]
    expected_blocks_per_lane: usize,

    /// Active hashes installed in each generated barrier snapshot.
    #[arg(long, default_value_t = DEFAULT_BASELINE_HASHES_PER_LANE)]
    baseline_hashes_per_lane: usize,

    /// Sequenced delta batches generated for each lane; must be positive and even.
    #[arg(long, default_value_t = DEFAULT_DELTA_FRAMES_PER_LANE)]
    delta_frames_per_lane: usize,

    /// JSON benchmark report.
    #[arg(long, default_value = "dc_ckf_consumer_result.json")]
    output_json: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrozenConsumerCorpus {
    schema_version: u32,
    generator: CorpusGeneratorMetadata,
    manifest: GlobalCkfManifest,
    lanes: Vec<FrozenLaneCorpus>,
    queries: Vec<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusGeneratorMetadata {
    source_topology: Option<DcCkfCorpusMetadata>,
    expected_blocks_per_lane: usize,
    baseline_hashes_per_lane: usize,
    delta_frames_per_lane: usize,
    deterministic_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrozenLaneCorpus {
    lane: usize,
    snapshot: DcCkfSnapshot,
    deltas: Vec<DcCkfDelta>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    mode: BenchmarkMode,
    corpus: CorpusReport,
    allocation: AllocationReport,
    configured_duration_ns: u64,
    configured_warmup_ns: u64,
    queue_capacity_per_worker: usize,
    max_dirty_to_applied_age_ns: u64,
    lane_to_ingestion_worker: Vec<LaneWorkerMapping>,
    elapsed_ns: u64,
    assignment_latency: LatencySummary,
    snapshot_install_latency: LatencySummary,
    delta_admission_latency: LatencySummary,
    exact_drain_latency: LatencySummary,
    query_latency: LatencySummary,
    delta_submissions: u64,
    submitted_images: u64,
    exact_drains: u64,
    max_outstanding_deltas: usize,
    max_outstanding_images: usize,
    query_operations: u64,
    query_unavailable: u64,
    query_result_checksum: u64,
    delta_throughput_per_second: f64,
    query_throughput_per_second: f64,
    expected_ready_mask: u16,
    final_ready_mask: u16,
    ingestion_faults: Vec<String>,
    terminal_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct CorpusReport {
    configured_lanes: usize,
    bucket_count: usize,
    frozen_delta_frames: usize,
    frozen_images: usize,
    queries: usize,
    repeats_frozen_images_with_fresh_sequences: bool,
}

#[derive(Debug, Serialize)]
struct AllocationReport {
    logical_cpus: usize,
    reserved_control_cpus: usize,
    service_thread_budget: usize,
    ingestion_workers: usize,
    query_issuers: usize,
}

#[derive(Debug, Serialize)]
struct LaneWorkerMapping {
    lane: usize,
    worker: usize,
}

#[derive(Debug, Default, Serialize)]
struct LatencySummary {
    operations: u64,
    samples: usize,
    min_ns: u64,
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    p999_ns: u64,
    max_ns: u64,
}

#[derive(Default)]
struct RunMetrics {
    assignment_ns: Vec<u64>,
    snapshot_ns: Vec<u64>,
    delta_admission_ns: Vec<u64>,
    delta_admission_operations: u64,
    drain_ns: Vec<u64>,
    delta_submissions: u64,
    submitted_images: u64,
    exact_drains: u64,
    max_outstanding_deltas: usize,
    max_outstanding_images: usize,
    lane_to_ingestion_worker: Vec<LaneWorkerMapping>,
    query_metrics: Vec<QueryThreadMetrics>,
    final_ready_mask: u16,
    faults: Vec<String>,
    terminal_error: Option<String>,
    elapsed: Duration,
}

#[derive(Default)]
struct QueryThreadMetrics {
    operations: u64,
    unavailable: u64,
    result_checksum: u64,
    latency_ns: Vec<u64>,
}

struct SelectedCorpus<'a> {
    manifest: GlobalCkfManifest,
    lanes: Vec<&'a FrozenLaneCorpus>,
    queries: Arc<Vec<Vec<LocalBlockHash>>>,
    query_lanes: Arc<Vec<usize>>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let source_topology = args
        .topology_metadata
        .as_deref()
        .map(read_json::<DcCkfCorpusMetadata>)
        .transpose()?;
    let corpus = match args.corpus_input.as_deref() {
        Some(path) => read_json(path)?,
        None => generate_corpus(&args, source_topology)?,
    };
    validate_corpus(&corpus)?;
    if let Some(path) = args.corpus_output.as_deref() {
        write_json(path, &corpus)?;
    }

    let selected = select_corpus(&corpus, args.lanes)?;
    let active_query_issuers = if matches!(args.mode, BenchmarkMode::IngestionOnly) {
        0
    } else {
        args.query_issuers
    };
    let allocation = validate_cpu_budget(args.ingestion_workers, active_query_issuers)?;
    let warmup = duration_from_seconds("warmup", args.warmup_seconds)?;
    let measured = duration_from_seconds("duration", args.duration_seconds)?;

    if !warmup.is_zero() {
        let warmup_result = execute_run(&args, &selected, warmup)?;
        ensure!(
            warmup_result.terminal_error.is_none() && warmup_result.faults.is_empty(),
            "consumer warmup faulted: terminal={:?}, faults={:?}",
            warmup_result.terminal_error,
            warmup_result.faults
        );
    }
    let metrics = execute_run(&args, &selected, measured)?;
    let report = build_report(&args, &corpus, &selected, allocation, warmup, metrics);
    write_json(&args.output_json, &report)?;

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(
        [1, 2, 4, 8, 16].contains(&args.lanes),
        "lanes must be one of 1, 2, 4, 8, 16"
    );
    ensure!(
        [1, 2, 4, 8].contains(&args.ingestion_workers),
        "ingestion_workers must be one of 1, 2, 4, 8"
    );
    ensure!(args.query_issuers > 0, "query_issuers must be positive");
    ensure!(args.queue_capacity > 0, "queue_capacity must be positive");
    ensure!(
        args.max_dirty_age_ms > 0,
        "max_dirty_age_ms must be positive"
    );
    ensure!(
        args.expected_blocks_per_lane > 0,
        "expected_blocks_per_lane must be positive"
    );
    ensure!(
        args.baseline_hashes_per_lane > 0
            && args.baseline_hashes_per_lane <= args.expected_blocks_per_lane,
        "baseline_hashes_per_lane must be in 1..=expected_blocks_per_lane"
    );
    ensure!(
        args.delta_frames_per_lane > 0 && args.delta_frames_per_lane.is_multiple_of(2),
        "delta_frames_per_lane must be positive and even"
    );
    duration_from_seconds("duration", args.duration_seconds)?;
    duration_from_seconds("warmup", args.warmup_seconds)?;
    Ok(())
}

fn duration_from_seconds(name: &str, seconds: f64) -> anyhow::Result<Duration> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0,
        "{name} must be finite and nonnegative"
    );
    if name == "duration" {
        ensure!(seconds > 0.0, "duration must be positive");
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn generate_corpus(
    args: &Args,
    source_topology: Option<DcCkfCorpusMetadata>,
) -> anyhow::Result<FrozenConsumerCorpus> {
    if let Some(topology) = &source_topology {
        ensure!(
            topology.dc_count >= args.lanes,
            "topology contains {} DCs but {} lanes were requested",
            topology.dc_count,
            args.lanes
        );
    }

    let expected_blocks = source_topology
        .as_ref()
        .and_then(|topology| {
            topology
                .pools
                .iter()
                .take(args.lanes)
                .map(|pool| pool.trace_distinct_hash_upper_bound)
                .max()
        })
        .map(|bound| bound.max(args.expected_blocks_per_lane))
        .unwrap_or(args.expected_blocks_per_lane);
    let baseline_hashes = args.baseline_hashes_per_lane.min(expected_blocks);
    let endpoint = EndpointId::new(
        source_topology
            .as_ref()
            .map_or(1, |topology| topology.endpoint_ordinal as u64 + 1),
    );
    let consumer = ConsumerInstanceId::new(0xC0DE_CAFE);
    let cache_domain = CacheDomainIdentity::new(CacheDomainId::new(1), 512, 1);
    let mut generated = Vec::with_capacity(args.lanes);
    let query_count = 256usize;
    let query_depth = 32usize.min(baseline_hashes);

    for lane in 0..args.lanes {
        generated.push(generate_lane_state(
            lane,
            expected_blocks,
            baseline_hashes,
            query_depth,
            args.delta_frames_per_lane,
            cache_domain,
            endpoint,
            consumer,
        )?);
    }
    let format = generated[0].snapshot.identity().format();
    let owners: [Option<GlobalCkfLaneOwner>; MAX_LANES] = std::array::from_fn(|lane| {
        (lane < args.lanes)
            .then(|| GlobalCkfLaneOwner::new(cache_domain, DcId::new(lane as u64 + 1), endpoint))
    });
    let manifest = GlobalCkfManifest::new(consumer, endpoint, format, owners)?;
    let mut lanes = Vec::with_capacity(args.lanes);
    for lane_state in generated {
        ensure!(
            lane_state.snapshot.identity().format() == format,
            "synthetic lane {} format differs from lane zero",
            lane_state.lane
        );
        lanes.push(FrozenLaneCorpus {
            lane: lane_state.lane,
            snapshot: lane_state.snapshot,
            deltas: lane_state.deltas,
        });
    }

    let mut queries = Vec::with_capacity(query_count);
    for query_index in 0..query_count {
        let lane = query_index % args.lanes;
        let start = lane_hash_base(lane)?;
        let group_count = (baseline_hashes / query_depth).max(1);
        let offset = ((query_index / args.lanes) % group_count) * query_depth;
        queries.push(
            (0..query_depth)
                .map(|position| start + (offset + position) as u64)
                .collect(),
        );
    }

    Ok(FrozenConsumerCorpus {
        schema_version: CORPUS_SCHEMA_VERSION,
        generator: CorpusGeneratorMetadata {
            source_topology,
            expected_blocks_per_lane: expected_blocks,
            baseline_hashes_per_lane: baseline_hashes,
            delta_frames_per_lane: args.delta_frames_per_lane,
            deterministic_seed: CkfConfig::default().seed,
        },
        manifest,
        lanes,
        queries,
    })
}

struct GeneratedLaneState {
    lane: usize,
    snapshot: DcCkfSnapshot,
    deltas: Vec<DcCkfDelta>,
}

#[derive(Default)]
struct CorpusDeltaSink {
    deltas: Vec<DcCkfDelta>,
}

impl DcCkfDeltaSink for CorpusDeltaSink {
    type Error = std::convert::Infallible;

    fn enqueue(&mut self, delta: DcCkfDelta) -> Result<(), Self::Error> {
        self.deltas.push(delta);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_lane_state(
    lane: usize,
    expected_blocks: usize,
    baseline_hashes: usize,
    query_depth: usize,
    delta_frames: usize,
    cache_domain: CacheDomainIdentity,
    endpoint: EndpointId,
    consumer: ConsumerInstanceId,
) -> anyhow::Result<GeneratedLaneState> {
    let mut state = DcCkfState::new(CkfConfig {
        expected_blocks_per_dc: expected_blocks,
        publish_every_n_events: 1,
        ..CkfConfig::default()
    })?;
    let member = WorkerWithDpRank::new(lane as u64 + 1, 0);
    let base = lane_hash_base(lane)?;
    let local_hashes = (0..baseline_hashes)
        .map(|offset| LocalBlockHash(base + offset as u64))
        .collect::<Vec<_>>();
    let sequence_hashes = local_hashes
        .chunks(query_depth)
        .flat_map(compute_seq_hash_for_block)
        .collect::<Vec<_>>();
    for (hash_offset, (&local_hash, &sequence_hash)) in
        local_hashes.iter().zip(&sequence_hashes).enumerate()
    {
        let outcome = state.apply_event(stored_event(
            member,
            hash_offset as u64 + 1,
            sequence_hash,
            local_hash,
        ));
        ensure!(
            outcome.first_error().is_none(),
            "synthetic lane {lane} failed while installing baseline hash {hash_offset}: {:?}",
            outcome.first_error()
        );
    }
    let identity = ProducerIdentity::new(
        cache_domain,
        DcId::new(lane as u64 + 1),
        endpoint,
        1,
        1,
        state.format(),
    );
    let lease = LaneLease::new(consumer, lane as u8, 1);
    let mut publisher = DcCkfPublisher::new(identity, 0, CorpusDeltaSink::default());
    let snapshot = publisher
        .snapshot_after_barrier(&mut state, lease)
        .map_err(|error| anyhow::anyhow!("failed to snapshot lane {lane}: {error:?}"))?;

    for frame_pair in 0..(delta_frames / 2) {
        let hash_offset = frame_pair % baseline_hashes;
        let hash = sequence_hashes[hash_offset];
        let remove = state.apply_event(removed_event(
            member,
            baseline_hashes as u64 + frame_pair as u64 * 2 + 1,
            hash,
        ));
        ensure!(
            remove.first_error().is_none(),
            "synthetic remove failed for lane {lane}"
        );
        let remove_batch = remove
            .into_publication()
            .context("publish-every-event synthetic remove produced no publication")?;
        publisher
            .publish(remove_batch)
            .map_err(|error| anyhow::anyhow!("failed to publish lane {lane} remove: {error:?}"))?;

        let store = state.apply_event(stored_event(
            member,
            baseline_hashes as u64 + frame_pair as u64 * 2 + 2,
            hash,
            local_hashes[hash_offset],
        ));
        ensure!(
            store.first_error().is_none(),
            "synthetic store failed for lane {lane}"
        );
        let store_batch = store
            .into_publication()
            .context("publish-every-event synthetic store produced no publication")?;
        publisher
            .publish(store_batch)
            .map_err(|error| anyhow::anyhow!("failed to publish lane {lane} store: {error:?}"))?;
    }
    let (_, final_buckets) = state.barrier_snapshot()?;
    ensure!(
        final_buckets.as_ref() == snapshot.buckets(),
        "synthetic lane {lane} delta cycle did not return to its barrier snapshot"
    );
    let deltas = publisher.into_sink().deltas;
    ensure!(
        deltas.len() == delta_frames,
        "synthetic lane {lane} emitted {} deltas, expected {delta_frames}",
        deltas.len()
    );

    Ok(GeneratedLaneState {
        lane,
        snapshot,
        deltas,
    })
}

fn lane_hash_base(lane: usize) -> anyhow::Result<u64> {
    (lane as u64)
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(1))
        .context("synthetic lane hash namespace overflow")
}

fn stored_event(
    member: WorkerWithDpRank,
    event_id: u64,
    sequence_hash: u64,
    local_hash: LocalBlockHash,
) -> RouterEvent {
    RouterEvent::new(
        member.worker_id,
        KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(sequence_hash),
                    tokens_hash: local_hash,
                    mm_extra_info: None,
                }],
            }),
            dp_rank: member.dp_rank,
        },
    )
}

fn removed_event(member: WorkerWithDpRank, event_id: u64, hash: u64) -> RouterEvent {
    RouterEvent::new(
        member.worker_id,
        KvCacheEvent {
            event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(hash)],
            }),
            dp_rank: member.dp_rank,
        },
    )
}

fn validate_corpus(corpus: &FrozenConsumerCorpus) -> anyhow::Result<()> {
    ensure!(
        corpus.schema_version == CORPUS_SCHEMA_VERSION,
        "unsupported consumer corpus schema version {}",
        corpus.schema_version
    );
    ensure!(
        !corpus.lanes.is_empty(),
        "consumer corpus contains no lanes"
    );
    ensure!(
        corpus.lanes.len() <= MAX_LANES,
        "consumer corpus contains more than 16 lanes"
    );
    ensure!(
        !corpus.queries.is_empty(),
        "consumer corpus contains no queries"
    );
    ensure!(
        corpus.queries.iter().all(|query| !query.is_empty()),
        "consumer corpus contains an empty query"
    );

    let mut seen_lanes = BTreeSet::new();
    for lane_corpus in &corpus.lanes {
        ensure!(
            lane_corpus.lane < MAX_LANES,
            "corpus lane {} is out of range",
            lane_corpus.lane
        );
        ensure!(
            seen_lanes.insert(lane_corpus.lane),
            "duplicate corpus lane {}",
            lane_corpus.lane
        );
        ensure!(
            corpus.manifest.owner(lane_corpus.lane).is_some(),
            "corpus lane {} is not configured in its manifest",
            lane_corpus.lane
        );
        validate_lane_corpus(&corpus.manifest, lane_corpus)?;
    }
    Ok(())
}

fn validate_lane_corpus(
    manifest: &GlobalCkfManifest,
    lane_corpus: &FrozenLaneCorpus,
) -> anyhow::Result<()> {
    let snapshot = &lane_corpus.snapshot;
    let identity = snapshot.identity();
    let lease = snapshot.lease();
    let owner = manifest
        .owner(lane_corpus.lane)
        .context("validated lane lost immutable owner")?;
    ensure!(
        lease.physical_lane() as usize == lane_corpus.lane,
        "snapshot lease targets wrong lane"
    );
    ensure!(
        lease.consumer_instance() == manifest.consumer_instance(),
        "snapshot targets wrong consumer"
    );
    ensure!(
        identity.format() == manifest.format(),
        "snapshot format differs from manifest"
    );
    ensure!(
        identity.endpoint_id() == manifest.endpoint_id(),
        "snapshot endpoint differs from manifest"
    );
    ensure!(
        identity.cache_domain() == owner.cache_domain() && identity.dc_id() == owner.dc_id(),
        "snapshot producer does not own lane {}",
        lane_corpus.lane
    );
    ensure!(
        snapshot.buckets().len() == manifest.format().bucket_count(),
        "snapshot bucket count differs from manifest"
    );
    ensure!(
        !lane_corpus.deltas.is_empty(),
        "lane {} contains no deltas",
        lane_corpus.lane
    );

    let mut sequence = snapshot.sequence();
    let mut reconstructed = snapshot.buckets().to_vec();
    for delta in &lane_corpus.deltas {
        ensure!(
            delta.identity() == identity,
            "lane {} delta identity changed",
            lane_corpus.lane
        );
        ensure!(
            delta.lease() == lease,
            "lane {} delta lease changed",
            lane_corpus.lane
        );
        ensure!(
            delta.base_sequence() == sequence,
            "lane {} delta sequence gap",
            lane_corpus.lane
        );
        let expected = sequence
            .checked_add(1)
            .context("corpus sequence overflow")?;
        ensure!(
            delta.sequence() == expected,
            "lane {} delta is not contiguous",
            lane_corpus.lane
        );
        ensure!(
            !delta.images().is_empty(),
            "lane {} contains an empty delta",
            lane_corpus.lane
        );
        let mut indices = BTreeSet::new();
        for image in delta.images() {
            ensure!(
                image.bucket() < reconstructed.len(),
                "lane {} delta bucket is out of range",
                lane_corpus.lane
            );
            ensure!(
                indices.insert(image.bucket()),
                "lane {} delta contains duplicate image index",
                lane_corpus.lane
            );
            reconstructed[image.bucket()] = image.value();
        }
        sequence = delta.sequence();
    }
    ensure!(
        reconstructed == snapshot.buckets(),
        "lane {} frozen delta stream must end at its starting snapshot for repeatable replay",
        lane_corpus.lane
    );
    Ok(())
}

fn select_corpus(
    corpus: &FrozenConsumerCorpus,
    lanes: usize,
) -> anyhow::Result<SelectedCorpus<'_>> {
    ensure!(
        corpus.lanes.len() >= lanes,
        "corpus has {} lanes, requested {lanes}",
        corpus.lanes.len()
    );
    let mut selected_lanes = corpus.lanes.iter().collect::<Vec<_>>();
    selected_lanes.sort_by_key(|lane| lane.lane);
    selected_lanes.truncate(lanes);
    let selected_set = selected_lanes
        .iter()
        .map(|lane| lane.lane)
        .collect::<BTreeSet<_>>();
    let owners = std::array::from_fn(|lane| {
        selected_set
            .contains(&lane)
            .then(|| corpus.manifest.owner(lane))
            .flatten()
    });
    let manifest = GlobalCkfManifest::new(
        corpus.manifest.consumer_instance(),
        corpus.manifest.endpoint_id(),
        corpus.manifest.format(),
        owners,
    )?;
    let mut queries = Vec::new();
    let mut query_lanes = Vec::new();
    for (query_index, query) in corpus.queries.iter().enumerate() {
        let lane = corpus.lanes[query_index % corpus.lanes.len()].lane;
        if !selected_set.contains(&lane) {
            continue;
        }
        queries.push(query.iter().copied().map(LocalBlockHash).collect());
        query_lanes.push(lane);
    }
    Ok(SelectedCorpus {
        manifest,
        lanes: selected_lanes,
        queries: Arc::new(queries),
        query_lanes: Arc::new(query_lanes),
    })
}

fn validate_cpu_budget(
    ingestion_workers: usize,
    query_issuers: usize,
) -> anyhow::Result<AllocationReport> {
    let logical_cpus = thread::available_parallelism().map_or(1, usize::from);
    let reserved_control_cpus = logical_cpus.div_ceil(10).max(2).min(logical_cpus);
    let service_thread_budget = logical_cpus.saturating_sub(reserved_control_cpus);
    let requested = ingestion_workers
        .checked_add(query_issuers)
        .context("service-thread count overflow")?;
    ensure!(
        requested <= service_thread_budget,
        "requested {requested} service threads ({ingestion_workers} ingestion + {query_issuers} query) but only {service_thread_budget} remain after reserving {reserved_control_cpus} of {logical_cpus} logical CPUs for control; use a larger dedicated allocation or scale the sweep down"
    );
    Ok(AllocationReport {
        logical_cpus,
        reserved_control_cpus,
        service_thread_budget,
        ingestion_workers,
        query_issuers,
    })
}

fn execute_run(
    args: &Args,
    corpus: &SelectedCorpus<'_>,
    duration: Duration,
) -> anyhow::Result<RunMetrics> {
    let indexer = GlobalCkfIndexer::new(corpus.manifest.clone(), PrefixSearchConfig::default())?;
    let bucket_count = corpus.manifest.format().bucket_count();
    let outstanding_limit = bucket_count
        .checked_mul(args.queue_capacity)
        .context("outstanding-image limit overflow")?;
    let pool = GlobalCkfIngestionPool::new(
        indexer.clone(),
        GlobalCkfIngestionPoolConfig {
            worker_count: args.ingestion_workers,
            queue_capacity: args.queue_capacity,
            control_timeout: Duration::from_secs(10),
            max_outstanding_images_per_lane: Some(outstanding_limit),
            max_dirty_to_applied_age: Duration::from_millis(args.max_dirty_age_ms),
        },
    )?;
    let mut metrics = RunMetrics {
        lane_to_ingestion_worker: corpus
            .lanes
            .iter()
            .map(|lane| LaneWorkerMapping {
                lane: lane.lane,
                worker: pool
                    .worker_for_lane(lane.lane)
                    .expect("selected lane must have a sticky ingestion worker"),
            })
            .collect(),
        ..RunMetrics::default()
    };
    for lane in &corpus.lanes {
        let start = Instant::now();
        pool.assign(lane.snapshot.identity(), lane.snapshot.lease())?;
        metrics.assignment_ns.push(elapsed_ns(start));

        let snapshot = lane.snapshot.clone();
        let start = Instant::now();
        let outcome = pool.install_snapshot(snapshot)?;
        metrics.snapshot_ns.push(elapsed_ns(start));
        ensure!(
            matches!(outcome, GlobalCkfIngestOutcome::SnapshotInstalled { .. }),
            "lane {} snapshot was not installed: {outcome:?}",
            lane.lane
        );
    }
    validate_expected_query_depths(&indexer, corpus)?;

    let stop_queries = Arc::new(AtomicBool::new(false));
    let active_query_issuers = if matches!(
        args.mode,
        BenchmarkMode::Mixed | BenchmarkMode::StableQueryOnly
    ) {
        args.query_issuers
    } else {
        0
    };
    let query_start = Arc::new(Barrier::new(active_query_issuers + 1));
    let query_handles = if matches!(
        args.mode,
        BenchmarkMode::Mixed | BenchmarkMode::StableQueryOnly
    ) {
        spawn_query_threads(
            &indexer,
            Arc::clone(&corpus.queries),
            Arc::clone(&stop_queries),
            Arc::clone(&query_start),
            args.query_issuers,
        )?
    } else {
        Vec::new()
    };

    let run_start = Instant::now();
    query_start.wait();
    let deadline = run_start + duration;
    if matches!(
        args.mode,
        BenchmarkMode::Mixed | BenchmarkMode::IngestionOnly
    ) {
        ingest_until(args, corpus, &pool, deadline, &mut metrics);
    } else {
        sleep_until(deadline);
    }
    stop_queries.store(true, Ordering::Release);
    metrics.query_metrics = join_query_threads(query_handles)?;
    metrics.elapsed = run_start.elapsed();
    metrics.final_ready_mask = indexer.ready_lanes();
    while let Some(fault) = pool.try_recv_fault() {
        metrics.faults.push(format!("{fault:?}"));
    }
    Ok(metrics)
}

fn ingest_until(
    args: &Args,
    corpus: &SelectedCorpus<'_>,
    pool: &GlobalCkfIngestionPool,
    deadline: Instant,
    metrics: &mut RunMetrics,
) {
    let mut next_sequence = corpus
        .lanes
        .iter()
        .map(|lane| lane.snapshot.sequence())
        .collect::<Vec<_>>();
    let mut next_frame = vec![0usize; corpus.lanes.len()];
    let lane_workers = corpus
        .lanes
        .iter()
        .map(|lane| {
            pool.worker_for_lane(lane.lane)
                .expect("selected lane must have a sticky ingestion worker")
        })
        .collect::<Vec<_>>();
    let mut outstanding_by_worker = vec![0usize; args.ingestion_workers];
    let mut total_outstanding = 0usize;
    let mut total_outstanding_images = 0usize;
    let drain_threshold_per_worker = (args.queue_capacity / 2).max(1);

    'run: while Instant::now() < deadline {
        for lane_index in 0..corpus.lanes.len() {
            if Instant::now() >= deadline {
                break 'run;
            }
            let lane = corpus.lanes[lane_index];
            let template = &lane.deltas[next_frame[lane_index]];
            let Some(sequence) = next_sequence[lane_index].checked_add(1) else {
                metrics.terminal_error = Some(format!("lane {} sequence exhausted", lane.lane));
                break 'run;
            };
            let delta = DcCkfDelta::new(
                lane.snapshot.identity(),
                lane.snapshot.lease(),
                next_sequence[lane_index],
                sequence,
                template.images().to_vec(),
            );
            let image_count = delta.images().len();
            let operation = metrics.delta_admission_operations;
            let start = Instant::now();
            let submit = pool.submit_delta(delta);
            let latency = elapsed_ns(start);
            metrics.delta_admission_operations += 1;
            if operation & LATENCY_SAMPLE_MASK == 0 {
                metrics.delta_admission_ns.push(latency);
            }
            if let Err(error) = submit {
                metrics.terminal_error = Some(error.to_string());
                break 'run;
            }
            metrics.delta_submissions += 1;
            metrics.submitted_images += image_count as u64;
            next_sequence[lane_index] = sequence;
            next_frame[lane_index] = (next_frame[lane_index] + 1) % lane.deltas.len();
            let worker = lane_workers[lane_index];
            outstanding_by_worker[worker] += 1;
            total_outstanding += 1;
            total_outstanding_images += image_count;
            metrics.max_outstanding_deltas = metrics.max_outstanding_deltas.max(total_outstanding);
            metrics.max_outstanding_images =
                metrics.max_outstanding_images.max(total_outstanding_images);

            if outstanding_by_worker[worker] >= drain_threshold_per_worker {
                if let Err(error) = drain_lanes(corpus, pool, &next_sequence, metrics) {
                    metrics.terminal_error = Some(error.to_string());
                    break 'run;
                }
                outstanding_by_worker.fill(0);
                total_outstanding = 0;
                total_outstanding_images = 0;
            }
        }
    }
    if metrics.terminal_error.is_none()
        && total_outstanding > 0
        && let Err(error) = drain_lanes(corpus, pool, &next_sequence, metrics)
    {
        metrics.terminal_error = Some(error.to_string());
    }
}

fn drain_lanes(
    corpus: &SelectedCorpus<'_>,
    pool: &GlobalCkfIngestionPool,
    next_sequence: &[u64],
    metrics: &mut RunMetrics,
) -> anyhow::Result<()> {
    for (lane_index, lane) in corpus.lanes.iter().enumerate() {
        let start = Instant::now();
        let outcome = pool.complete_drain(ConsumerDrainMarker::new(
            lane.snapshot.lease(),
            next_sequence[lane_index],
        ))?;
        metrics.drain_ns.push(elapsed_ns(start));
        metrics.exact_drains += 1;
        ensure!(
            matches!(outcome, GlobalCkfIngestOutcome::DrainAcknowledged { .. }),
            "lane {} exact drain failed: {outcome:?}",
            lane.lane
        );
    }
    Ok(())
}

fn spawn_query_threads(
    indexer: &GlobalCkfIndexer,
    queries: Arc<Vec<Vec<LocalBlockHash>>>,
    stop: Arc<AtomicBool>,
    start: Arc<Barrier>,
    issuers: usize,
) -> anyhow::Result<Vec<thread::JoinHandle<QueryThreadMetrics>>> {
    let mut handles = Vec::with_capacity(issuers);
    for issuer in 0..issuers {
        let indexer = indexer.clone();
        let queries = Arc::clone(&queries);
        let stop = Arc::clone(&stop);
        let start = Arc::clone(&start);
        handles.push(
            thread::Builder::new()
                .name(format!("dc-ckf-query-{issuer}"))
                .spawn(move || {
                    // Query counters and samples deliberately remain thread-local. The query
                    // itself reads one Acquire ready mask and never takes an ingestion lock.
                    let mut metrics = QueryThreadMetrics::default();
                    let mut query_index = issuer % queries.len();
                    start.wait();
                    while !stop.load(Ordering::Acquire) {
                        let start = Instant::now();
                        match indexer.find_prefix_matches(&queries[query_index]) {
                            Ok(result) => {
                                let mut checksum = u64::from(result.captured_ready_lanes());
                                for lane_match in result.lanes().iter().flatten() {
                                    checksum = checksum
                                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                        .wrapping_add(u64::from(lane_match.physical_lane()))
                                        .wrapping_add(u64::from(lane_match.prefix_depth()));
                                }
                                metrics.result_checksum ^= black_box(checksum);
                            }
                            Err(_) => metrics.unavailable += 1,
                        }
                        let operation = metrics.operations;
                        metrics.operations += 1;
                        if operation & LATENCY_SAMPLE_MASK == 0 {
                            metrics.latency_ns.push(elapsed_ns(start));
                        }
                        query_index += 1;
                        if query_index == queries.len() {
                            query_index = 0;
                        }
                    }
                    metrics
                })?,
        );
    }
    Ok(handles)
}

fn validate_expected_query_depths(
    indexer: &GlobalCkfIndexer,
    corpus: &SelectedCorpus<'_>,
) -> anyhow::Result<()> {
    for (query_index, query) in corpus.queries.iter().enumerate() {
        let lane = corpus.query_lanes[query_index];
        let result = indexer.find_prefix_matches(query)?;
        let depth = result.lanes()[lane]
            .as_ref()
            .context("expected query lane is not ready")?
            .prefix_depth();
        ensure!(
            depth as usize == query.len(),
            "synthetic query {query_index} matched lane {lane} to depth {depth}, expected {}",
            query.len()
        );
    }
    Ok(())
}

fn join_query_threads(
    handles: Vec<thread::JoinHandle<QueryThreadMetrics>>,
) -> anyhow::Result<Vec<QueryThreadMetrics>> {
    handles
        .into_iter()
        .map(|handle| {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("global CKF query issuer panicked"))
        })
        .collect()
}

fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        thread::sleep((deadline - now).min(Duration::from_millis(10)));
    }
}

fn build_report(
    args: &Args,
    full_corpus: &FrozenConsumerCorpus,
    selected: &SelectedCorpus<'_>,
    allocation: AllocationReport,
    warmup: Duration,
    metrics: RunMetrics,
) -> BenchmarkReport {
    let elapsed_seconds = metrics.elapsed.as_secs_f64();
    let query_operations = metrics
        .query_metrics
        .iter()
        .map(|item| item.operations)
        .sum();
    let query_unavailable = metrics
        .query_metrics
        .iter()
        .map(|item| item.unavailable)
        .sum();
    let query_result_checksum = metrics
        .query_metrics
        .iter()
        .fold(0u64, |checksum, item| checksum ^ item.result_checksum);
    let mut query_ns = metrics
        .query_metrics
        .into_iter()
        .flat_map(|item| item.latency_ns)
        .collect::<Vec<_>>();
    let query_samples = query_ns.len();
    let query_latency = summarize_latency(&mut query_ns, query_operations);
    debug_assert_eq!(query_latency.samples, query_samples);
    let frozen_delta_frames = selected.lanes.iter().map(|lane| lane.deltas.len()).sum();
    let frozen_images = selected
        .lanes
        .iter()
        .flat_map(|lane| &lane.deltas)
        .map(|delta| delta.images().len())
        .sum();
    let expected_ready_mask = selected
        .lanes
        .iter()
        .fold(0u16, |mask, lane| mask | (1u16 << lane.lane));

    let mut assignment_ns = metrics.assignment_ns;
    let assignment_operations = assignment_ns.len() as u64;
    let assignment_latency = summarize_latency(&mut assignment_ns, assignment_operations);
    let mut snapshot_ns = metrics.snapshot_ns;
    let snapshot_operations = snapshot_ns.len() as u64;
    let snapshot_install_latency = summarize_latency(&mut snapshot_ns, snapshot_operations);
    let mut delta_ns = metrics.delta_admission_ns;
    let delta_admission_latency =
        summarize_latency(&mut delta_ns, metrics.delta_admission_operations);
    let mut drain_ns = metrics.drain_ns;
    let exact_drain_latency = summarize_latency(&mut drain_ns, metrics.exact_drains);

    BenchmarkReport {
        schema_version: RESULT_SCHEMA_VERSION,
        mode: args.mode,
        corpus: CorpusReport {
            configured_lanes: selected.lanes.len(),
            bucket_count: full_corpus.manifest.format().bucket_count(),
            frozen_delta_frames,
            frozen_images,
            queries: selected.queries.len(),
            repeats_frozen_images_with_fresh_sequences: true,
        },
        allocation,
        configured_duration_ns: duration_ns(Duration::from_secs_f64(args.duration_seconds)),
        configured_warmup_ns: duration_ns(warmup),
        queue_capacity_per_worker: args.queue_capacity,
        max_dirty_to_applied_age_ns: duration_ns(Duration::from_millis(args.max_dirty_age_ms)),
        lane_to_ingestion_worker: metrics.lane_to_ingestion_worker,
        elapsed_ns: duration_ns(metrics.elapsed),
        assignment_latency,
        snapshot_install_latency,
        delta_admission_latency,
        exact_drain_latency,
        query_latency,
        delta_submissions: metrics.delta_submissions,
        submitted_images: metrics.submitted_images,
        exact_drains: metrics.exact_drains,
        max_outstanding_deltas: metrics.max_outstanding_deltas,
        max_outstanding_images: metrics.max_outstanding_images,
        query_operations,
        query_unavailable,
        query_result_checksum,
        delta_throughput_per_second: metrics.delta_submissions as f64 / elapsed_seconds,
        query_throughput_per_second: query_operations as f64 / elapsed_seconds,
        expected_ready_mask,
        final_ready_mask: metrics.final_ready_mask,
        ingestion_faults: metrics.faults,
        terminal_error: metrics.terminal_error,
    }
}

fn summarize_latency(values: &mut [u64], operations: u64) -> LatencySummary {
    if values.is_empty() {
        return LatencySummary {
            operations,
            ..LatencySummary::default()
        };
    }
    values.sort_unstable();
    LatencySummary {
        operations,
        samples: values.len(),
        min_ns: values[0],
        p50_ns: nearest_rank(values, 50, 100),
        p95_ns: nearest_rank(values, 95, 100),
        p99_ns: nearest_rank(values, 99, 100),
        p999_ns: nearest_rank(values, 999, 1_000),
        max_ns: values[values.len() - 1],
    }
}

fn nearest_rank(values: &[u64], numerator: usize, denominator: usize) -> u64 {
    let rank = values
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[rank]
}

fn elapsed_ns(start: Instant) -> u64 {
    duration_ns(start.elapsed())
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> anyhow::Result<T> {
    let file = File::open(path)
        .with_context(|| format!("failed to open JSON input {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse JSON input {}", path.display()))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create JSON output {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), value)
        .with_context(|| format!("failed to write JSON output {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            corpus_input: None,
            corpus_output: None,
            topology_metadata: None,
            lanes: 2,
            ingestion_workers: 1,
            query_issuers: 1,
            mode: BenchmarkMode::Mixed,
            duration_seconds: 0.02,
            warmup_seconds: 0.0,
            queue_capacity: 32,
            max_dirty_age_ms: 1_000,
            expected_blocks_per_lane: 128,
            baseline_hashes_per_lane: 64,
            delta_frames_per_lane: 32,
            output_json: PathBuf::from("unused.json"),
        }
    }

    #[test]
    fn generated_corpus_has_contiguous_repeatable_streams() -> anyhow::Result<()> {
        let args = test_args();
        let corpus = generate_corpus(&args, None)?;
        validate_corpus(&corpus)?;
        for lane in &corpus.lanes {
            assert_eq!(lane.deltas[0].base_sequence(), lane.snapshot.sequence());
            assert_eq!(
                lane.deltas.last().unwrap().sequence(),
                lane.snapshot.sequence() + lane.deltas.len() as u64
            );
            for pair in lane.deltas.chunks_exact(2) {
                let remove = pair[0].images();
                let store = pair[1].images();
                assert_eq!(remove.len(), 1);
                assert_eq!(store.len(), 1);
                assert_eq!(remove[0].bucket(), store[0].bucket());
                assert!((0..4).any(|slot| {
                    let shift = slot * u16::BITS as usize;
                    let removed = (remove[0].value() >> shift) as u16;
                    let restored = (store[0].value() >> shift) as u16;
                    removed == 0 && restored != 0
                }));
            }
        }
        let encoded = serde_json::to_vec(&corpus)?;
        let decoded: FrozenConsumerCorpus = serde_json::from_slice(&encoded)?;
        validate_corpus(&decoded)
    }

    #[test]
    fn validation_rejects_a_sequence_gap() -> anyhow::Result<()> {
        let args = test_args();
        let mut corpus = generate_corpus(&args, None)?;
        let lane = &mut corpus.lanes[0];
        let original = lane.deltas[1].clone();
        lane.deltas[1] = DcCkfDelta::new(
            original.identity(),
            original.lease(),
            original.base_sequence() + 1,
            original.sequence() + 1,
            original.images().to_vec(),
        );
        let error = validate_corpus(&corpus).unwrap_err();
        assert!(error.to_string().contains("sequence gap"));
        Ok(())
    }

    #[test]
    fn mixed_ingestion_and_query_smoke() -> anyhow::Result<()> {
        let args = test_args();
        let corpus = generate_corpus(&args, None)?;
        let selected = select_corpus(&corpus, args.lanes)?;
        let metrics = execute_run(&args, &selected, Duration::from_millis(20))?;
        assert!(metrics.delta_submissions > 0);
        assert!(
            metrics
                .query_metrics
                .iter()
                .map(|item| item.operations)
                .sum::<u64>()
                > 0
        );
        assert!(metrics.faults.is_empty(), "{:?}", metrics.faults);
        assert!(
            metrics.terminal_error.is_none(),
            "{:?}",
            metrics.terminal_error
        );
        assert_eq!(metrics.final_ready_mask, 0b11);
        Ok(())
    }
}
