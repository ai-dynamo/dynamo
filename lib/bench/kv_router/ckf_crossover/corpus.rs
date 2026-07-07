// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, ensure};
use dynamo_bench::kv_router_common::replay::{generate_replay_artifacts, process_mooncake_trace};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, StorageTier, compute_seq_hash_for_block,
};
use rustc_hash::FxHashSet;
use sha2::{Digest, Sha256};

use crate::types::{
    AccuracySample, BallastSpec, CORPUS_VERSION, CkfPublication, CorpusEntry, CorpusHeader,
    CorpusManifest, CorpusOperation, FilterShape, LogicalTotals, PreparedCorpus, PublisherStats,
    PublisherTiming, ResidentImage,
};
use dynamo_kv_router::indexer::cuckoo::{
    DEFAULT_FILTER_SEED, Publish, SLOTS, SnapshotProducer, apply_delta, assemble_chunks,
};

const FILTER_EXPECTED_LOAD: f64 = 0.8;

#[derive(Clone, Debug)]
pub struct PrepareOptions {
    pub trace_path: PathBuf,
    pub output_path: PathBuf,
    pub expected_trace_sha256: String,
    pub trace_duplication_factor: usize,
    pub trace_length_factor: usize,
    pub trace_partition_seed: u64,
    pub block_size: u32,
    pub num_dcs: usize,
    pub event_threads: usize,
    pub query_concurrency: usize,
    pub num_gpu_blocks: usize,
    pub trace_simulation_duration_ms: Option<u64>,
    pub ballast_memberships_per_dc: usize,
    pub ballast_depth: usize,
    pub ballast_seed: u64,
    pub git_sha: String,
    pub accuracy_request_limit: usize,
}

#[derive(Clone)]
struct SourceEntry {
    timestamp_us: u64,
    stable_order: u64,
    operation: SourceOperation,
}

#[derive(Clone)]
enum SourceOperation {
    Request(Vec<LocalBlockHash>),
    Update { dc: usize, event: RouterEvent },
}

struct ConsumerShadow {
    filter: dynamo_kv_router::indexer::cuckoo::CuckooFilter,
    epoch: u64,
}

pub async fn prepare_corpus(options: &PrepareOptions) -> anyhow::Result<CorpusManifest> {
    ensure!(options.num_dcs > 0, "num_dcs must be greater than zero");
    ensure!(
        options.event_threads > 0,
        "event_threads must be greater than zero"
    );
    ensure!(
        options.num_dcs.is_multiple_of(options.event_threads),
        "num_dcs must be divisible by event_threads"
    );
    ensure!(
        options.ballast_depth > 0,
        "ballast_depth must be greater than zero"
    );

    let trace_sha256 = sha256_file(&options.trace_path)?;
    ensure!(
        trace_sha256 == options.expected_trace_sha256,
        "trace digest mismatch: expected {}, got {}",
        options.expected_trace_sha256,
        trace_sha256
    );

    let traces = process_mooncake_trace(
        options
            .trace_path
            .to_str()
            .context("trace path is not UTF-8")?,
        options.block_size,
        options.trace_length_factor,
        options.trace_duplication_factor,
        options.num_dcs,
        options.trace_partition_seed,
    )?;
    let artifacts = generate_replay_artifacts(
        &traces,
        options.num_gpu_blocks,
        options.block_size,
        options.trace_simulation_duration_ms,
    )
    .await?;
    ensure!(
        artifacts.len() == options.num_dcs,
        "trace partition count mismatch: expected {}, got {}",
        options.num_dcs,
        artifacts.len()
    );

    let mut source = Vec::new();
    let mut stable_order = 0u64;
    let mut totals = LogicalTotals::default();
    let mut trace_hashes = FxHashSet::default();
    for (dc, artifact) in artifacts.iter().enumerate() {
        for request in &artifact.requests {
            totals.requests += 1;
            totals.request_blocks += request.replay_hashes.local_block_hashes.len() as u64;
            trace_hashes.extend(compute_seq_hash_for_block(
                &request.replay_hashes.local_block_hashes,
            ));
            source.push(SourceEntry {
                timestamp_us: request.timestamp_us,
                stable_order,
                operation: SourceOperation::Request(
                    request.replay_hashes.local_block_hashes.clone(),
                ),
            });
            stable_order += 1;
        }
        for timed_event in &artifact.kv_events {
            if !timed_event.storage_tier.is_gpu() {
                continue;
            }
            ensure!(
                !matches!(&timed_event.event.data, KvCacheEventData::Cleared),
                "the crossover corpus supports Stored/Removed transitions; Cleared would also remove synthetic ballast"
            );
            let event_blocks = event_block_count(&timed_event.event.data);
            totals.events += 1;
            totals.event_blocks += event_blocks as u64;
            collect_event_hashes(&timed_event.event.data, &mut trace_hashes);
            source.push(SourceEntry {
                timestamp_us: timed_event.timestamp_us,
                stable_order,
                operation: SourceOperation::Update {
                    dc,
                    event: RouterEvent::with_storage_tier(
                        dc as u64,
                        timed_event.event.clone(),
                        StorageTier::Device,
                    ),
                },
            });
            stable_order += 1;
        }
    }
    source.sort_by_key(|entry| {
        let kind = match entry.operation {
            SourceOperation::Request(_) => 0u8,
            SourceOperation::Update { .. } => 1u8,
        };
        (entry.timestamp_us, kind, entry.stable_order)
    });
    let source_span_us = source.last().map_or(0, |entry| entry.timestamp_us);

    let ballast = BallastSpec {
        memberships_per_dc: options.ballast_memberships_per_dc,
        prefix_depth: options.ballast_depth,
        seed: options.ballast_seed,
        namespace_tag: 0xD16C_C001_F17E_0001,
    };
    validate_ballast_namespace(&ballast, options.num_dcs, &trace_hashes)?;

    let maximum_live = maximum_live_counts(&source, options.num_dcs, &ballast)?;
    let provisioned_memberships = maximum_live.iter().copied().max().unwrap_or_default();
    let mut producers = Vec::with_capacity(options.num_dcs);
    let mut filter_shapes = Vec::with_capacity(options.num_dcs);
    let mut insertion_failures = vec![0u64; options.num_dcs];
    for (dc, &maximum) in maximum_live.iter().enumerate() {
        let mut producer =
            SnapshotProducer::new(dc as u64, provisioned_memberships, DEFAULT_FILTER_SEED);
        for_each_ballast_family(dc, &ballast, |_, sequence| {
            for &hash in sequence {
                if !producer.insert(hash) {
                    insertion_failures[dc] += 1;
                }
            }
        });
        filter_shapes.push(FilterShape {
            seed: producer.seed(),
            buckets: 0,
            slots: SLOTS,
            fingerprint_bits: 16,
            expected_load: FILTER_EXPECTED_LOAD,
            maximum_live_count: maximum,
            insertion_failures: insertion_failures[dc],
        });
        producers.push(producer);
    }

    let mut bootstrap_chunks = Vec::with_capacity(options.num_dcs);
    let mut shadows = Vec::with_capacity(options.num_dcs);
    for (dc, producer) in producers.iter_mut().enumerate() {
        let snapshot = producer.full_snapshot();
        filter_shapes[dc].buckets = snapshot.num_buckets();
        let chunks: Vec<Arc<[u8]>> = snapshot.chunks().map(Arc::from).collect();
        let (filter, meta) = assemble_chunks(&chunks)?;
        ensure!(meta.dc_worker_id == dc as u64, "bootstrap DC mismatch");
        ensure!(
            meta.filter_epoch == producer.epoch(),
            "bootstrap epoch mismatch"
        );
        shadows.push(ConsumerShadow {
            filter,
            epoch: meta.filter_epoch,
        });
        bootstrap_chunks.push(chunks);
    }
    ensure!(
        filter_shapes.iter().all(|shape| {
            shape.buckets == filter_shapes[0].buckets && shape.seed == filter_shapes[0].seed
        }),
        "all DC filters must have identical shape and seed for the transposed backend"
    );

    let mut trace_resident: Vec<FxHashSet<u64>> =
        (0..options.num_dcs).map(|_| FxHashSet::default()).collect();
    let mut entries = Vec::with_capacity(source.len());
    let mut publisher = PublisherStats::default();
    let mut publisher_timing = PublisherTiming::default();
    let mut prefix_closure_violations = 0u64;
    let mut trace_accuracy_requests = Vec::new();
    let mut logical_event_id = 0u64;

    for source_entry in source {
        let operation = match source_entry.operation {
            SourceOperation::Request(local_hashes) => {
                prefix_closure_violations +=
                    count_prefix_closure_violations(&local_hashes, &trace_resident);
                if trace_accuracy_requests.len() < options.accuracy_request_limit {
                    trace_accuracy_requests.push(local_hashes.clone());
                }
                CorpusOperation::Request { local_hashes }
            }
            SourceOperation::Update { dc, event } => {
                apply_authoritative_event(
                    &event.event.data,
                    &mut trace_resident[dc],
                    &mut producers[dc],
                    &mut insertion_failures[dc],
                    &mut publisher.missing_removals,
                    &mut publisher.filter_removal_failures,
                );
                let started = Instant::now();
                let publication = producers[dc].publish();
                let publication = match publication {
                    Publish::Full(snapshot) => {
                        let chunks: Vec<Arc<[u8]>> = snapshot.chunks().map(Arc::from).collect();
                        let generation_ns = started.elapsed().as_nanos() as u64;
                        publisher_timing.generation_ns += generation_ns;
                        publisher_timing.full_generation_ns += generation_ns;
                        let bytes = chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
                        let (filter, meta) = assemble_chunks(&chunks)?;
                        ensure!(
                            meta.dc_worker_id == dc as u64,
                            "full publication DC mismatch"
                        );
                        shadows[dc] = ConsumerShadow {
                            filter,
                            epoch: meta.filter_epoch,
                        };
                        publisher.full += 1;
                        publisher.bytes += bytes as u64;
                        publisher.full_bytes += bytes as u64;
                        CkfPublication::Full(chunks)
                    }
                    Publish::Delta(frame) => {
                        let generation_ns = started.elapsed().as_nanos() as u64;
                        publisher_timing.generation_ns += generation_ns;
                        publisher_timing.delta_generation_ns += generation_ns;
                        let current_epoch = shadows[dc].epoch;
                        let delta = apply_delta(&mut shadows[dc].filter, current_epoch, &frame)?;
                        ensure!(
                            delta.dc_worker_id == dc as u64,
                            "delta publication DC mismatch"
                        );
                        shadows[dc].epoch = delta.new_epoch;
                        publisher.delta += 1;
                        publisher.dirty_buckets += delta.entries.len() as u64;
                        publisher.bytes += frame.len() as u64;
                        publisher.delta_bytes += frame.len() as u64;
                        CkfPublication::Delta(Arc::from(frame))
                    }
                    Publish::Unchanged => {
                        let generation_ns = started.elapsed().as_nanos() as u64;
                        publisher_timing.generation_ns += generation_ns;
                        publisher_timing.unchanged_generation_ns += generation_ns;
                        publisher.unchanged += 1;
                        CkfPublication::Unchanged
                    }
                };
                logical_event_id += 1;
                CorpusOperation::Update {
                    logical_event_id,
                    dc: dc as u16,
                    event,
                    publication,
                }
            }
        };
        entries.push(CorpusEntry {
            timestamp_us: source_entry.timestamp_us,
            stable_order: source_entry.stable_order,
            operation,
        });
    }

    let mut accuracy_samples: Vec<AccuracySample> = trace_accuracy_requests
        .into_iter()
        .map(|local_hashes| AccuracySample {
            exact_depths: exact_trace_depths(&local_hashes, &trace_resident),
            local_hashes,
        })
        .collect();
    for dc in 0..options.num_dcs {
        for family in 0..ballast_family_count(&ballast).min(2) {
            let local_hashes = ballast_local_hashes(&ballast, dc, family);
            let mut exact_depths = vec![0; options.num_dcs];
            exact_depths[dc] = local_hashes.len() as u32;
            accuracy_samples.push(AccuracySample {
                local_hashes,
                exact_depths,
            });
        }
    }

    for (dc, shape) in filter_shapes.iter_mut().enumerate() {
        shape.insertion_failures = insertion_failures[dc];
    }
    let final_trace_resident = trace_resident
        .into_iter()
        .map(|set| {
            let mut hashes: Vec<u64> = set.into_iter().collect();
            hashes.sort_unstable();
            hashes
        })
        .collect();
    let header = CorpusHeader {
        version: CORPUS_VERSION,
        created_by_git_sha: options.git_sha.clone(),
        codec_id: "CKF1-v1".to_string(),
        hash_scheme_id: "dynamo-sequence-xxh3-seed-1337-v1".to_string(),
        filter_shape_sha256: sha256_serialized(&filter_shapes)?,
        trace_sha256,
        trace_duplication_factor: options.trace_duplication_factor,
        trace_length_factor: options.trace_length_factor,
        trace_partition_seed: options.trace_partition_seed,
        block_size: options.block_size,
        num_dcs: options.num_dcs,
        event_threads: options.event_threads,
        query_concurrency: options.query_concurrency,
        source_span_us,
        ballast,
        filter_shapes,
        totals,
        prefix_closure_violations,
    };
    ensure!(
        header
            .filter_shapes
            .iter()
            .all(|shape| shape.insertion_failures == 0),
        "CKF insertion failures occurred while preparing the corpus"
    );
    ensure!(
        publisher.missing_removals == 0,
        "authoritative Relay preparation observed {} missing removals",
        publisher.missing_removals
    );
    ensure!(
        publisher.filter_removal_failures == 0,
        "authoritative Relay preparation observed {} CKF removal failures",
        publisher.filter_removal_failures
    );
    let corpus = PreparedCorpus {
        header: header.clone(),
        bootstrap_chunks,
        entries,
        final_trace_resident,
        accuracy_samples,
        publisher: publisher.clone(),
    };
    write_corpus(&options.output_path, &corpus)?;
    let corpus_sha256 = sha256_file(&options.output_path)?;
    let corpus_bytes = options.output_path.metadata()?.len();
    let resident_path = options.output_path.with_extension("resident.msgpack");
    write_resident_parts(&resident_path, &header, &corpus.bootstrap_chunks)?;
    let resident_sha256 = sha256_file(&resident_path)?;
    let resident_bytes = resident_path.metadata()?.len();
    let manifest = CorpusManifest {
        corpus_path: options.output_path.display().to_string(),
        corpus_sha256,
        corpus_bytes,
        resident_path: resident_path.display().to_string(),
        resident_sha256,
        resident_bytes,
        header,
        publisher,
        publisher_timing,
    };
    let manifest_path = options.output_path.with_extension("manifest.json");
    let mut writer = BufWriter::new(File::create(&manifest_path)?);
    serde_json::to_writer_pretty(&mut writer, &manifest)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(manifest)
}

pub fn load_corpus(path: &Path, expected_sha256: Option<&str>) -> anyhow::Result<PreparedCorpus> {
    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(path)?;
        ensure!(
            actual == expected,
            "corpus digest mismatch: expected {expected}, got {actual}"
        );
    }
    let file = File::open(path).with_context(|| format!("open corpus {}", path.display()))?;
    let corpus: PreparedCorpus = rmp_serde::from_read(BufReader::new(file))?;
    ensure!(
        corpus.header.version == CORPUS_VERSION,
        "unsupported corpus version"
    );
    Ok(corpus)
}

pub fn write_corpus(path: &Path, corpus: &PreparedCorpus) -> anyhow::Result<()> {
    let file = File::create(path).with_context(|| format!("create corpus {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    rmp_serde::encode::write_named(&mut writer, corpus)?;
    writer.flush()?;
    Ok(())
}

pub fn load_resident_image(path: &Path) -> anyhow::Result<ResidentImage> {
    let file =
        File::open(path).with_context(|| format!("open resident image {}", path.display()))?;
    let image: ResidentImage = rmp_serde::from_read(BufReader::new(file))?;
    ensure!(
        image.header.version == CORPUS_VERSION,
        "unsupported resident image version"
    );
    Ok(image)
}

pub fn write_resident_image(path: &Path, image: &ResidentImage) -> anyhow::Result<()> {
    let file =
        File::create(path).with_context(|| format!("create resident image {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    rmp_serde::encode::write_named(&mut writer, image)?;
    writer.flush()?;
    Ok(())
}

fn write_resident_parts(
    path: &Path,
    header: &CorpusHeader,
    bootstrap_chunks: &[Vec<Arc<[u8]>>],
) -> anyhow::Result<()> {
    #[derive(serde::Serialize)]
    struct BorrowedResidentImage<'a> {
        header: &'a CorpusHeader,
        bootstrap_chunks: &'a [Vec<Arc<[u8]>>],
    }
    let file =
        File::create(path).with_context(|| format!("create resident image {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    rmp_serde::encode::write_named(
        &mut writer,
        &BorrowedResidentImage {
            header,
            bootstrap_chunks,
        },
    )?;
    writer.flush()?;
    Ok(())
}

pub fn d2_fixture_corpus(git_sha: String) -> anyhow::Result<PreparedCorpus> {
    let ballast = BallastSpec {
        memberships_per_dc: 32,
        prefix_depth: 8,
        seed: 17,
        namespace_tag: 0xD200_0000_0000_0001,
    };
    let mut producers: Vec<SnapshotProducer> = (0..2)
        .map(|dc| {
            let mut producer = SnapshotProducer::new(dc, 96, DEFAULT_FILTER_SEED);
            for_each_ballast_family(dc as usize, &ballast, |_, sequence| {
                for &hash in sequence {
                    assert!(producer.insert(hash));
                }
            });
            producer
        })
        .collect();
    let mut bootstrap_chunks = Vec::new();
    let mut shapes = Vec::new();
    for producer in &mut producers {
        let snapshot = producer.full_snapshot();
        shapes.push(FilterShape {
            seed: producer.seed(),
            buckets: snapshot.num_buckets(),
            slots: SLOTS,
            fingerprint_bits: 16,
            expected_load: FILTER_EXPECTED_LOAD,
            maximum_live_count: 96,
            insertion_failures: 0,
        });
        bootstrap_chunks.push(snapshot.chunks_with(4).map(Arc::from).collect());
    }
    let request = vec![
        LocalBlockHash(101),
        LocalBlockHash(202),
        LocalBlockHash(303),
    ];
    let sequence = compute_seq_hash_for_block(&request);
    let stored = RouterEvent::new(
        0,
        KvCacheEvent {
            event_id: 1,
            dp_rank: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: Some(0),
                blocks: request
                    .iter()
                    .copied()
                    .zip(sequence.iter().copied())
                    .map(|(tokens_hash, block_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block_hash),
                        tokens_hash,
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
        },
    );
    let removed = RouterEvent::new(
        0,
        KvCacheEvent {
            event_id: 2,
            dp_rank: 0,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(sequence[2])],
            }),
        },
    );
    let mut entries = vec![CorpusEntry {
        timestamp_us: 0,
        stable_order: 0,
        operation: CorpusOperation::Request {
            local_hashes: request.clone(),
        },
    }];
    let mut publisher = PublisherStats::default();
    let mut resident = FxHashSet::default();
    for (index, event) in [stored.clone(), stored, removed].into_iter().enumerate() {
        let mut failures = 0;
        apply_authoritative_event(
            &event.event.data,
            &mut resident,
            &mut producers[0],
            &mut failures,
            &mut publisher.missing_removals,
            &mut publisher.filter_removal_failures,
        );
        let publication = match producers[0].publish() {
            Publish::Delta(frame) => {
                publisher.delta += 1;
                publisher.bytes += frame.len() as u64;
                publisher.delta_bytes += frame.len() as u64;
                CkfPublication::Delta(Arc::from(frame))
            }
            Publish::Unchanged => {
                publisher.unchanged += 1;
                CkfPublication::Unchanged
            }
            Publish::Full(snapshot) => {
                let chunks: Vec<Arc<[u8]>> = snapshot.chunks_with(4).map(Arc::from).collect();
                publisher.full += 1;
                let bytes = chunks.iter().map(|chunk| chunk.len()).sum::<usize>() as u64;
                publisher.bytes += bytes;
                publisher.full_bytes += bytes;
                CkfPublication::Full(chunks)
            }
        };
        entries.push(CorpusEntry {
            timestamp_us: (index as u64 + 1) * 250,
            stable_order: index as u64 + 1,
            operation: CorpusOperation::Update {
                logical_event_id: index as u64 + 1,
                dc: 0,
                event,
                publication,
            },
        });
    }
    entries.push(CorpusEntry {
        timestamp_us: 1_000,
        stable_order: 4,
        operation: CorpusOperation::Request {
            local_hashes: request.clone(),
        },
    });
    Ok(PreparedCorpus {
        header: CorpusHeader {
            version: CORPUS_VERSION,
            created_by_git_sha: git_sha,
            codec_id: "CKF1-v1".to_string(),
            hash_scheme_id: "dynamo-sequence-xxh3-seed-1337-v1".to_string(),
            filter_shape_sha256: sha256_serialized(&shapes)?,
            trace_sha256: "fixture".to_string(),
            trace_duplication_factor: 1,
            trace_length_factor: 1,
            trace_partition_seed: 42,
            block_size: 128,
            num_dcs: 2,
            event_threads: 2,
            query_concurrency: 2,
            source_span_us: 1_000,
            ballast,
            filter_shapes: shapes,
            totals: LogicalTotals {
                requests: 2,
                events: 3,
                request_blocks: 6,
                event_blocks: 7,
            },
            prefix_closure_violations: 0,
        },
        bootstrap_chunks,
        entries,
        final_trace_resident: vec![sequence[..2].to_vec(), Vec::new()],
        accuracy_samples: vec![AccuracySample {
            local_hashes: request,
            exact_depths: vec![2, 0],
        }],
        publisher,
    })
}

pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_serialized(value: &impl serde::Serialize) -> anyhow::Result<String> {
    let bytes = rmp_serde::to_vec_named(value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub fn for_each_ballast_family(
    dc: usize,
    spec: &BallastSpec,
    mut callback: impl FnMut(usize, &[u64]),
) {
    let families = spec.memberships_per_dc.div_ceil(spec.prefix_depth);
    let mut remaining = spec.memberships_per_dc;
    for family in 0..families {
        let depth = remaining.min(spec.prefix_depth);
        let local: Vec<LocalBlockHash> = (0..depth)
            .map(|position| ballast_local_hash(spec, dc, family, position))
            .collect();
        let sequence = compute_seq_hash_for_block(&local);
        callback(family, &sequence);
        remaining -= depth;
    }
}

pub fn ballast_store_event(
    dc: usize,
    spec: &BallastSpec,
    family: usize,
    event_id: u64,
) -> RouterEvent {
    let local = ballast_local_hashes(spec, dc, family);
    let sequence = compute_seq_hash_for_block(&local);
    let blocks = local
        .into_iter()
        .zip(sequence)
        .map(|(tokens_hash, block_hash)| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(block_hash),
            tokens_hash,
            mm_extra_info: None,
        })
        .collect();
    RouterEvent::new(
        dc as u64,
        KvCacheEvent {
            event_id,
            dp_rank: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: Some(0),
                blocks,
            }),
        },
    )
}

pub fn ballast_family_count(spec: &BallastSpec) -> usize {
    spec.memberships_per_dc.div_ceil(spec.prefix_depth)
}

pub fn ballast_local_hashes(spec: &BallastSpec, dc: usize, family: usize) -> Vec<LocalBlockHash> {
    let start = family * spec.prefix_depth;
    let depth = (spec.memberships_per_dc - start).min(spec.prefix_depth);
    (0..depth)
        .map(|position| ballast_local_hash(spec, dc, family, position))
        .collect()
}

fn ballast_local_hash(
    spec: &BallastSpec,
    dc: usize,
    family: usize,
    position: usize,
) -> LocalBlockHash {
    let words = [
        spec.namespace_tag,
        dc as u64,
        family as u64,
        position as u64,
    ];
    LocalBlockHash(xxhash_rust::xxh3::xxh3_64_with_seed(
        bytemuck::cast_slice(&words),
        spec.seed,
    ))
}

fn validate_ballast_namespace(
    spec: &BallastSpec,
    num_dcs: usize,
    trace_hashes: &FxHashSet<u64>,
) -> anyhow::Result<()> {
    let mut ballast_hashes = HashSet::with_capacity(spec.memberships_per_dc * num_dcs);
    for dc in 0..num_dcs {
        let mut conflict = None;
        for_each_ballast_family(dc, spec, |_, sequence| {
            for &hash in sequence {
                if trace_hashes.contains(&hash) || !ballast_hashes.insert(hash) {
                    conflict = Some(hash);
                    break;
                }
            }
        });
        ensure!(
            conflict.is_none(),
            "ballast namespace collision at DC {dc}: hash={:#x}",
            conflict.unwrap_or_default()
        );
    }
    Ok(())
}

fn maximum_live_counts(
    source: &[SourceEntry],
    num_dcs: usize,
    ballast: &BallastSpec,
) -> anyhow::Result<Vec<usize>> {
    let mut resident: Vec<FxHashSet<u64>> = (0..num_dcs).map(|_| FxHashSet::default()).collect();
    let mut maximum = vec![ballast.memberships_per_dc; num_dcs];
    for entry in source {
        let SourceOperation::Update { dc, event } = &entry.operation else {
            continue;
        };
        apply_trace_resident(&event.event.data, &mut resident[*dc]);
        maximum[*dc] = maximum[*dc].max(ballast.memberships_per_dc + resident[*dc].len());
    }
    Ok(maximum)
}

fn count_prefix_closure_violations(
    local_hashes: &[LocalBlockHash],
    resident: &[FxHashSet<u64>],
) -> u64 {
    let sequence = compute_seq_hash_for_block(local_hashes);
    resident
        .iter()
        .map(|dc| {
            let mut missed = false;
            sequence
                .iter()
                .filter(|&&hash| {
                    let present = dc.contains(&hash);
                    let violation = missed && present;
                    missed |= !present;
                    violation
                })
                .count() as u64
        })
        .sum()
}

fn exact_trace_depths(local_hashes: &[LocalBlockHash], resident: &[FxHashSet<u64>]) -> Vec<u32> {
    let sequence = compute_seq_hash_for_block(local_hashes);
    resident
        .iter()
        .map(|dc| {
            sequence
                .iter()
                .take_while(|&&hash| dc.contains(&hash))
                .count() as u32
        })
        .collect()
}

fn apply_authoritative_event(
    data: &KvCacheEventData,
    resident: &mut FxHashSet<u64>,
    producer: &mut SnapshotProducer,
    insertion_failures: &mut u64,
    missing_removals: &mut u64,
    filter_removal_failures: &mut u64,
) {
    match data {
        KvCacheEventData::Stored(store) => {
            for block in &store.blocks {
                if resident.insert(block.block_hash.0) && !producer.insert(block.block_hash.0) {
                    *insertion_failures += 1;
                }
            }
        }
        KvCacheEventData::Removed(remove) => {
            for hash in &remove.block_hashes {
                if resident.remove(&hash.0) {
                    if !producer.remove(hash.0) {
                        *filter_removal_failures += 1;
                    }
                } else {
                    *missing_removals += 1;
                }
            }
        }
        KvCacheEventData::Cleared => {
            for hash in resident.drain() {
                if !producer.remove(hash) {
                    *filter_removal_failures += 1;
                }
            }
        }
    }
}

fn apply_trace_resident(data: &KvCacheEventData, resident: &mut FxHashSet<u64>) {
    match data {
        KvCacheEventData::Stored(store) => {
            resident.extend(store.blocks.iter().map(|block| block.block_hash.0));
        }
        KvCacheEventData::Removed(remove) => {
            for hash in &remove.block_hashes {
                resident.remove(&hash.0);
            }
        }
        KvCacheEventData::Cleared => resident.clear(),
    }
}

fn event_block_count(data: &KvCacheEventData) -> usize {
    match data {
        KvCacheEventData::Stored(store) => store.blocks.len(),
        KvCacheEventData::Removed(remove) => remove.block_hashes.len(),
        KvCacheEventData::Cleared => 0,
    }
}

fn collect_event_hashes(data: &KvCacheEventData, hashes: &mut FxHashSet<u64>) {
    match data {
        KvCacheEventData::Stored(store) => {
            hashes.extend(store.blocks.iter().map(|block| block.block_hash.0));
        }
        KvCacheEventData::Removed(remove) => {
            hashes.extend(remove.block_hashes.iter().map(|hash| hash.0));
        }
        KvCacheEventData::Cleared => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ballast_has_exact_requested_membership_count() {
        let spec = BallastSpec {
            memberships_per_dc: 257,
            prefix_depth: 128,
            seed: 7,
            namespace_tag: 9,
        };
        let mut total = 0;
        for_each_ballast_family(0, &spec, |_, sequence| total += sequence.len());
        assert_eq!(total, 257);
        assert_eq!(ballast_family_count(&spec), 3);
    }

    #[test]
    fn removed_blocks_are_counted() {
        let data = KvCacheEventData::Removed(dynamo_kv_router::protocols::KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
        });
        assert_eq!(event_block_count(&data), 2);
    }

    #[test]
    fn d2_fixture_keeps_one_publication_per_complete_event() {
        let corpus = d2_fixture_corpus("fixture".to_string()).unwrap();
        let updates: Vec<_> = corpus
            .entries
            .iter()
            .filter_map(|entry| match &entry.operation {
                CorpusOperation::Update {
                    event, publication, ..
                } => Some((event, publication)),
                CorpusOperation::Request { .. } => None,
            })
            .collect();
        assert_eq!(updates.len(), 3);
        assert_eq!(event_block_count(&updates[0].0.event.data), 3);
        assert!(matches!(updates[0].1, CkfPublication::Delta(_)));
        assert!(matches!(updates[1].1, CkfPublication::Unchanged));
        assert!(matches!(updates[2].1, CkfPublication::Delta(_)));
        assert_eq!(corpus.header.totals.event_blocks, 7);
    }

    #[test]
    fn d2_fixture_serialization_is_deterministic() {
        let first = d2_fixture_corpus("fixture".to_string()).unwrap();
        let second = d2_fixture_corpus("fixture".to_string()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.msgpack");
        let second_path = dir.path().join("second.msgpack");
        write_corpus(&first_path, &first).unwrap();
        write_corpus(&second_path, &second).unwrap();
        assert_eq!(
            sha256_file(&first_path).unwrap(),
            sha256_file(&second_path).unwrap()
        );
    }
}
