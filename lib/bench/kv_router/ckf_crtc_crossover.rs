// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "ckf_crossover/corpus.rs"]
mod corpus;
#[path = "ckf_crossover/crtc.rs"]
mod crtc;
#[path = "ckf_crossover/dispatch.rs"]
mod dispatch;
#[path = "ckf_crossover/memory.rs"]
mod memory;
#[path = "ckf_crossover/types.rs"]
mod types;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use dynamo_kv_router::indexer::cuckoo::{
    CuckooDcConfig, CuckooFrameEnvelope, CuckooFrameIndexer, CuckooIndexerConfig,
    CuckooIndexerMode, CuckooPublication, SnapshotProducer,
};
use rustc_hash::FxHashSet;
use serde::Serialize;

use corpus::{
    PrepareOptions, d2_fixture_corpus, for_each_ballast_family, load_corpus, load_resident_image,
    prepare_corpus, sha256_file, write_corpus, write_resident_image,
};
use crtc::CrtcBackend;
use dispatch::{RunOptions, run_trial};
use memory::memory_snapshot;
use types::{
    BackendKind, DEFAULT_BALLAST_DEPTH, DEFAULT_BALLAST_MEMBERSHIPS, DEFAULT_DCS,
    DEFAULT_EVENT_THREADS, DEFAULT_QUERY_CONCURRENCY, DEFAULT_TRACE_SHA256, ResidentImage,
    relay_instance_id,
};

#[derive(Parser)]
#[command(about = "D=16 production CRTC versus native/transposed CKF experiment")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Prepare {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = DEFAULT_TRACE_SHA256)]
        trace_sha256: String,
        #[arg(long, default_value_t = 20)]
        trace_duplication_factor: usize,
        #[arg(long, default_value_t = 4)]
        trace_length_factor: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 128)]
        block_size: u32,
        #[arg(long, default_value_t = DEFAULT_DCS)]
        dcs: usize,
        #[arg(long, default_value_t = DEFAULT_EVENT_THREADS)]
        event_threads: usize,
        #[arg(long, default_value_t = DEFAULT_QUERY_CONCURRENCY)]
        query_concurrency: usize,
        #[arg(long, default_value_t = 16_384)]
        num_gpu_blocks: usize,
        #[arg(long)]
        trace_simulation_duration_ms: Option<u64>,
        #[arg(long, default_value_t = DEFAULT_BALLAST_MEMBERSHIPS)]
        ballast_memberships_per_dc: usize,
        #[arg(long, default_value_t = DEFAULT_BALLAST_DEPTH)]
        ballast_depth: usize,
        #[arg(long, default_value_t = 0xB411_A57D_1600_0001)]
        ballast_seed: u64,
        #[arg(long)]
        git_sha: String,
        #[arg(long, default_value_t = 4096)]
        accuracy_requests: usize,
    },
    RunCell {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        expected_corpus_sha256: String,
        #[arg(long, value_enum)]
        backend: BackendKind,
        #[arg(long)]
        replay_window_ms: f64,
        #[arg(long)]
        repetition: usize,
        #[arg(long, default_value = "capacity")]
        phase: String,
        #[arg(long)]
        measured_code_sha: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 64)]
        warmup_queries: usize,
    },
    Fixture {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "fixture-sha")]
        git_sha: String,
    },
    Memory {
        #[arg(long)]
        resident_image: PathBuf,
        #[arg(long, value_enum)]
        mode: MemoryMode,
        #[arg(long)]
        measured_code_sha: String,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum MemoryMode {
    Crtc,
    CkfNative,
    CkfTransposed,
    RelayProducer,
}

#[derive(Serialize)]
struct MemoryResult {
    schema_version: u32,
    measured_code_sha: String,
    resident_image_sha256: String,
    mode: MemoryMode,
    rss_bytes: u64,
    pss_bytes: Option<u64>,
    uss_bytes: Option<u64>,
    num_dcs: usize,
    memberships_per_dc: usize,
    authoritative_filter_bytes: u64,
    derived_transposed_bytes: u64,
}

fn main() -> anyhow::Result<()> {
    let worker_threads = std::env::var("TOKIO_WORKER_THREADS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("TOKIO_WORKER_THREADS must be a positive integer")?
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    ensure!(worker_threads > 0, "TOKIO_WORKER_THREADS must be positive");
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()?
        .block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Prepare {
            trace,
            output,
            trace_sha256,
            trace_duplication_factor,
            trace_length_factor,
            seed,
            block_size,
            dcs,
            event_threads,
            query_concurrency,
            num_gpu_blocks,
            trace_simulation_duration_ms,
            ballast_memberships_per_dc,
            ballast_depth,
            ballast_seed,
            git_sha,
            accuracy_requests,
        } => {
            let manifest = prepare_corpus(&PrepareOptions {
                trace_path: trace,
                output_path: output,
                expected_trace_sha256: trace_sha256,
                trace_duplication_factor,
                trace_length_factor,
                trace_partition_seed: seed,
                block_size,
                num_dcs: dcs,
                event_threads,
                query_concurrency,
                num_gpu_blocks,
                trace_simulation_duration_ms,
                ballast_memberships_per_dc,
                ballast_depth,
                ballast_seed,
                git_sha,
                accuracy_request_limit: accuracy_requests,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::RunCell {
            corpus,
            expected_corpus_sha256,
            backend,
            replay_window_ms,
            repetition,
            phase,
            measured_code_sha,
            output,
            warmup_queries,
        } => {
            verify_expected_cpu_binding()?;
            ensure!(replay_window_ms > 0.0, "replay_window_ms must be positive");
            let prepared = Arc::new(load_corpus(&corpus, Some(&expected_corpus_sha256))?);
            let result = run_trial(
                prepared,
                RunOptions {
                    backend,
                    replay_window: Duration::from_secs_f64(replay_window_ms / 1000.0),
                    repetition,
                    phase,
                    measured_code_sha,
                    corpus_sha256: expected_corpus_sha256,
                    warmup_queries,
                },
            )
            .await?;
            write_json(&output, &result)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Fixture { output, git_sha } => {
            let fixture = d2_fixture_corpus(git_sha)?;
            write_corpus(&output, &fixture)?;
            write_resident_image(
                &output.with_extension("resident.msgpack"),
                &ResidentImage {
                    header: fixture.header.clone(),
                    bootstrap_chunks: fixture.bootstrap_chunks.clone(),
                },
            )?;
            println!("fixture_sha256={}", sha256_file(&output)?);
        }
        Command::Memory {
            resident_image,
            mode,
            measured_code_sha,
            output,
        } => {
            verify_expected_cpu_binding()?;
            let image_sha256 = sha256_file(&resident_image)?;
            let image = load_resident_image(&resident_image)?;
            let result = measure_memory(image, mode, measured_code_sha, image_sha256).await?;
            write_json(&output, &result)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

async fn measure_memory(
    image: ResidentImage,
    mode: MemoryMode,
    measured_code_sha: String,
    resident_image_sha256: String,
) -> anyhow::Result<MemoryResult> {
    let num_dcs = image.header.num_dcs;
    let memberships_per_dc = image.header.ballast.memberships_per_dc;
    let filter_bytes = image
        .header
        .filter_shapes
        .iter()
        .map(|shape| shape.buckets * shape.slots * 2)
        .sum::<usize>() as u64;
    match mode {
        MemoryMode::Crtc => {
            let header = image.header.clone();
            drop(image);
            trim_allocator();
            let backend = CrtcBackend::from_header(&header).await?;
            backend.touch_for_benchmark();
            trim_allocator();
            let snapshot = memory_snapshot()?;
            std::hint::black_box(&backend);
            Ok(memory_result(
                measured_code_sha,
                resident_image_sha256,
                mode,
                num_dcs,
                memberships_per_dc,
                (0, 0),
                snapshot,
            ))
        }
        MemoryMode::CkfNative | MemoryMode::CkfTransposed => {
            let core_mode = if matches!(mode, MemoryMode::CkfNative) {
                CuckooIndexerMode::Native
            } else {
                CuckooIndexerMode::Transposed
            };
            let indexer = CuckooFrameIndexer::new(CuckooIndexerConfig {
                mode: core_mode,
                event_threads: image.header.event_threads,
                block_size: image.header.block_size,
                dcs: image
                    .header
                    .filter_shapes
                    .iter()
                    .enumerate()
                    .map(|(dc, shape)| CuckooDcConfig {
                        dc_worker_id: dc as u64,
                        relay_instance_id: relay_instance_id(dc),
                        num_buckets: shape.buckets,
                        seed: shape.seed,
                    })
                    .collect(),
            })?;
            for (dc, chunks) in image.bootstrap_chunks.iter().enumerate() {
                indexer.install_bootstrap(CuckooFrameEnvelope {
                    dc_worker_id: dc as u64,
                    relay_instance_id: relay_instance_id(dc),
                    publication: CuckooPublication::Full(chunks.clone()),
                })?;
            }
            drop(image);
            trim_allocator();
            indexer.touch_for_benchmark();
            trim_allocator();
            let snapshot = memory_snapshot()?;
            std::hint::black_box(&indexer);
            Ok(memory_result(
                measured_code_sha,
                resident_image_sha256,
                mode,
                num_dcs,
                memberships_per_dc,
                (
                    filter_bytes,
                    if matches!(mode, MemoryMode::CkfTransposed) {
                        filter_bytes
                    } else {
                        0
                    },
                ),
                snapshot,
            ))
        }
        MemoryMode::RelayProducer => {
            let header = image.header.clone();
            drop(image);
            trim_allocator();
            let provisioned_memberships = header
                .filter_shapes
                .iter()
                .map(|shape| shape.maximum_live_count)
                .max()
                .unwrap_or_default();
            let mut authoritative = Vec::with_capacity(num_dcs);
            let mut producers = Vec::with_capacity(num_dcs);
            for dc in 0..num_dcs {
                let mut set = FxHashSet::default();
                let mut producer = SnapshotProducer::new(
                    dc as u64,
                    provisioned_memberships,
                    header.filter_shapes[dc].seed,
                );
                for_each_ballast_family(dc, &header.ballast, |_, sequence| {
                    for &hash in sequence {
                        set.insert(hash);
                        assert!(producer.insert(hash));
                    }
                });
                producer.full_snapshot();
                authoritative.push(set);
                producers.push(producer);
            }
            trim_allocator();
            let snapshot = memory_snapshot()?;
            std::hint::black_box((&authoritative, &producers));
            Ok(memory_result(
                measured_code_sha,
                resident_image_sha256,
                mode,
                num_dcs,
                memberships_per_dc,
                (filter_bytes, 0),
                snapshot,
            ))
        }
    }
}

fn memory_result(
    measured_code_sha: String,
    resident_image_sha256: String,
    mode: MemoryMode,
    num_dcs: usize,
    memberships_per_dc: usize,
    structure_bytes: (u64, u64),
    snapshot: memory::MemorySnapshot,
) -> MemoryResult {
    let (authoritative_filter_bytes, derived_transposed_bytes) = structure_bytes;
    MemoryResult {
        schema_version: 1,
        measured_code_sha,
        resident_image_sha256,
        mode,
        rss_bytes: snapshot.rss_bytes,
        pss_bytes: snapshot.pss_bytes,
        uss_bytes: snapshot.uss_bytes,
        num_dcs,
        memberships_per_dc,
        authoritative_filter_bytes,
        derived_transposed_bytes,
    }
}

fn write_json(path: &PathBuf, value: &impl Serialize) -> anyhow::Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn trim_allocator() {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(target_os = "linux")]
fn verify_expected_cpu_binding() -> anyhow::Result<()> {
    let Ok(expected) = std::env::var("EXPECTED_CPU_BINDING") else {
        return Ok(());
    };
    let status = std::fs::read_to_string("/proc/self/status")?;
    let actual = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(str::trim)
        .context("Cpus_allowed_list missing from /proc/self/status")?;
    ensure!(
        parse_cpu_list(actual)? == parse_cpu_list(&expected)?,
        "CPU binding mismatch: expected {expected}, got {actual}"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn verify_expected_cpu_binding() -> anyhow::Result<()> {
    ensure!(
        std::env::var_os("EXPECTED_CPU_BINDING").is_none(),
        "EXPECTED_CPU_BINDING is only supported on Linux"
    );
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn parse_cpu_list(value: &str) -> anyhow::Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.split(',').filter(|part| !part.is_empty()) {
        let Some((start, end)) = part.split_once('-') else {
            cpus.push(part.parse()?);
            continue;
        };
        let start: usize = start.parse()?;
        let end: usize = end.parse()?;
        ensure!(start <= end, "invalid CPU range {part}");
        cpus.extend(start..=end);
    }
    cpus.sort_unstable();
    cpus.dedup();
    ensure!(!cpus.is_empty(), "CPU list is empty");
    Ok(cpus)
}

#[cfg(test)]
mod cpu_binding_tests {
    use super::parse_cpu_list;

    #[test]
    fn cpu_list_parser_normalizes_ranges_and_singletons() {
        assert_eq!(parse_cpu_list("4,0-2,2").unwrap(), vec![0, 1, 2, 4]);
    }
}
