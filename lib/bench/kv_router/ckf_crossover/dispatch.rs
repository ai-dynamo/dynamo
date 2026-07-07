// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::ensure;
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::indexer::cuckoo::{
    CuckooDcConfig, CuckooFrameEnvelope, CuckooFrameIndexer, CuckooFrameMetadata,
    CuckooIndexerConfig, CuckooIndexerMode,
};
use dynamo_kv_router::protocols::OverlapScores;
use tokio::sync::mpsc;

use crate::crtc::{CrtcBackend, CrtcCompletionMetrics};
use crate::types::{
    AccuracyMetrics, AccuracySample, BackendKind, CorpusOperation, PipelineErrors, PreparedCorpus,
    QueueMetrics, TrialResult, percentile_summary, relay_instance_id,
};

#[derive(Clone)]
enum Backend {
    Crtc(Arc<CrtcBackend>),
    Ckf(Arc<CuckooFrameIndexer>),
}

impl Backend {
    fn lookup(&self, local_hashes: &[LocalBlockHash]) -> OverlapScores {
        match self {
            Self::Crtc(backend) => backend.lookup(local_hashes),
            Self::Ckf(backend) => backend.lookup(local_hashes),
        }
    }

    fn reset_stats(&self) {
        match self {
            Self::Crtc(backend) => backend.reset_stats(),
            Self::Ckf(backend) => backend.reset_stats(),
        }
    }

    fn touch_for_benchmark(&self) {
        match self {
            Self::Crtc(backend) => backend.touch_for_benchmark(),
            Self::Ckf(backend) => backend.touch_for_benchmark(),
        }
    }

    fn shutdown(&self) {
        match self {
            Self::Crtc(backend) => backend.shutdown(),
            Self::Ckf(backend) => backend.shutdown_threads(),
        }
    }
}

struct QueryTask {
    entry_index: usize,
    scheduled_at: Instant,
    enqueued_at: Instant,
}

#[derive(Default)]
struct QuerySamples {
    queue_wait_ns: Vec<u64>,
    service_ns: Vec<u64>,
    scheduled_to_completion_ns: Vec<u64>,
}

struct UpdateMetrics {
    queue: QueueMetrics,
    scheduled_to_enqueue_ns: Vec<u64>,
    enqueue_to_applied_ns: Vec<u64>,
    scheduled_to_applied_ns: Vec<u64>,
    errors: PipelineErrors,
}

pub struct RunOptions {
    pub backend: BackendKind,
    pub replay_window: Duration,
    pub repetition: usize,
    pub phase: String,
    pub measured_code_sha: String,
    pub corpus_sha256: String,
    pub warmup_queries: usize,
}

pub async fn run_trial(
    corpus: Arc<PreparedCorpus>,
    options: RunOptions,
) -> anyhow::Result<TrialResult> {
    ensure!(
        corpus.header.query_concurrency > 0,
        "query concurrency must be greater than zero"
    );
    ensure!(
        corpus.header.source_span_us > 0,
        "corpus source span must be greater than zero"
    );
    let backend = match options.backend {
        BackendKind::Crtc => Backend::Crtc(CrtcBackend::from_corpus(&corpus).await?),
        BackendKind::CkfNative | BackendKind::CkfTransposed => {
            let mode = match options.backend {
                BackendKind::CkfNative => CuckooIndexerMode::Native,
                BackendKind::CkfTransposed => CuckooIndexerMode::Transposed,
                BackendKind::Crtc => unreachable!(),
            };
            let indexer = CuckooFrameIndexer::new(CuckooIndexerConfig {
                mode,
                event_threads: corpus.header.event_threads,
                block_size: corpus.header.block_size,
                dcs: corpus
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
            for (dc, chunks) in corpus.bootstrap_chunks.iter().enumerate() {
                indexer.install_bootstrap(CuckooFrameEnvelope {
                    dc_worker_id: dc as u64,
                    relay_instance_id: relay_instance_id(dc),
                    publication: crate::types::CkfPublication::Full(chunks.clone()),
                })?;
            }
            Backend::Ckf(indexer)
        }
    };

    backend.touch_for_benchmark();
    for request in corpus
        .accuracy_samples
        .iter()
        .take(options.warmup_queries.max(1))
    {
        std::hint::black_box(backend.lookup(&request.local_hashes));
    }
    if let Backend::Ckf(consumer) = &backend {
        consumer.verify_transposed()?;
        consumer.flush_with_metrics()?;
    }
    backend.reset_stats();

    let queued_queries = Arc::new(AtomicU64::new(0));
    let maximum_query_depth = Arc::new(AtomicU64::new(0));
    let mut query_senders = Vec::with_capacity(corpus.header.query_concurrency);
    let mut query_handles = Vec::with_capacity(corpus.header.query_concurrency);
    for _ in 0..corpus.header.query_concurrency {
        let (sender, mut receiver) = mpsc::unbounded_channel::<QueryTask>();
        query_senders.push(sender);
        let worker_backend = backend.clone();
        let worker_corpus = Arc::clone(&corpus);
        let worker_queued = Arc::clone(&queued_queries);
        query_handles.push(tokio::spawn(async move {
            let mut samples = QuerySamples::default();
            while let Some(task) = receiver.recv().await {
                worker_queued.fetch_sub(1, Ordering::Relaxed);
                let started = Instant::now();
                samples.queue_wait_ns.push(
                    started
                        .saturating_duration_since(task.enqueued_at)
                        .as_nanos() as u64,
                );
                let CorpusOperation::Request { local_hashes } =
                    &worker_corpus.entries[task.entry_index].operation
                else {
                    unreachable!("query task references an update")
                };
                std::hint::black_box(worker_backend.lookup(local_hashes));
                let completed = Instant::now();
                samples
                    .service_ns
                    .push(completed.saturating_duration_since(started).as_nanos() as u64);
                samples.scheduled_to_completion_ns.push(
                    completed
                        .saturating_duration_since(task.scheduled_at)
                        .as_nanos() as u64,
                );
            }
            samples
        }));
    }

    let start = Instant::now();
    let mut issue_lag_ns = Vec::with_capacity(corpus.entries.len());
    let mut next_query_worker = 0usize;
    for (entry_index, entry) in corpus.entries.iter().enumerate() {
        let offset = scaled_offset(
            entry.timestamp_us,
            corpus.header.source_span_us,
            options.replay_window,
        );
        let target = start + offset;
        let now = Instant::now();
        if now < target {
            tokio::time::sleep_until(tokio::time::Instant::from_std(target)).await;
        }
        let issued_at = Instant::now();
        issue_lag_ns.push(issued_at.saturating_duration_since(target).as_nanos() as u64);
        match &entry.operation {
            CorpusOperation::Request { .. } => {
                let enqueued_at = Instant::now();
                let depth = queued_queries.fetch_add(1, Ordering::Relaxed) + 1;
                update_max(&maximum_query_depth, depth);
                if let Err(error) = query_senders[next_query_worker].send(QueryTask {
                    entry_index,
                    scheduled_at: target,
                    enqueued_at,
                }) {
                    queued_queries.fetch_sub(1, Ordering::Relaxed);
                    return Err(anyhow::anyhow!("query executor stopped: {error}"));
                }
                next_query_worker = (next_query_worker + 1) % query_senders.len();
            }
            CorpusOperation::Update {
                logical_event_id,
                dc,
                event,
                publication,
            } => match &backend {
                Backend::Crtc(crtc) => crtc.submit(event, *logical_event_id, target)?,
                Backend::Ckf(ckf) => {
                    if matches!(publication, crate::types::CkfPublication::Unchanged) {
                        continue;
                    }
                    ckf.submit(
                        CuckooFrameEnvelope {
                            dc_worker_id: *dc as u64,
                            relay_instance_id: relay_instance_id(*dc as usize),
                            publication: publication.clone(),
                        },
                        CuckooFrameMetadata {
                            logical_event_id: *logical_event_id,
                            scheduled_at: target,
                        },
                    )?
                }
            },
        }
    }
    let issue_elapsed = start.elapsed();
    let query_at_stop = queued_queries.load(Ordering::Relaxed);
    let query_drain_started = Instant::now();
    drop(query_senders);
    let flush_backend = backend.clone();
    let update_handle = tokio::spawn(async move { flush_updates(&flush_backend).await });
    let mut query_samples = QuerySamples::default();
    for handle in query_handles {
        let samples = handle.await?;
        query_samples.queue_wait_ns.extend(samples.queue_wait_ns);
        query_samples.service_ns.extend(samples.service_ns);
        query_samples
            .scheduled_to_completion_ns
            .extend(samples.scheduled_to_completion_ns);
    }
    let query_queue = QueueMetrics {
        at_stop: query_at_stop,
        maximum_depth: maximum_query_depth.load(Ordering::Relaxed),
        drain_ms: query_drain_started.elapsed().as_secs_f64() * 1000.0,
    };
    let update = update_handle.await??;
    let total_elapsed = start.elapsed();

    let accuracy = match &backend {
        Backend::Crtc(_) => AccuracyMetrics::default(),
        Backend::Ckf(consumer) => final_accuracy(consumer, &corpus.accuracy_samples),
    };
    let totals = &corpus.header.totals;
    let replay_seconds = options.replay_window.as_secs_f64().max(f64::EPSILON);
    let issue_seconds = issue_elapsed.as_secs_f64().max(f64::EPSILON);
    let elapsed_seconds = total_elapsed.as_secs_f64().max(f64::EPSILON);
    let nominal_offered_mixed_ops_s = totals.mixed_ops() as f64 / replay_seconds;
    let nominal_offered_block_ops_s = totals.block_ops() as f64 / replay_seconds;
    let achieved_mixed_ops_s = totals.mixed_ops() as f64 / elapsed_seconds;
    let achieved_block_ops_s = totals.block_ops() as f64 / elapsed_seconds;
    let generator_limited = issue_elapsed > options.replay_window.mul_f64(1.10);
    let pipeline_errors = update.errors;
    let kept_up = total_elapsed <= options.replay_window.mul_f64(1.10)
        && !generator_limited
        && pipeline_errors.total() == 0;
    let (
        crtc_raw_events,
        crtc_raw_blocks,
        ckf_frames,
        ckf_dirty_buckets,
        ckf_bytes,
        ckf_apply_mib_s,
        ckf_full_apply_mib_s,
        ckf_delta_apply_mib_s,
        full_publications,
        delta_publications,
        unchanged_publications,
        generation_conflicts,
        native_fallbacks,
        repeated_fallbacks,
    ) = match &backend {
        Backend::Crtc(crtc) => (
            crtc.stats.raw_events.load(Ordering::Relaxed),
            crtc.stats.raw_blocks.load(Ordering::Relaxed),
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        Backend::Ckf(ckf) => {
            let stats = ckf.stats_snapshot();
            let bytes = stats.bytes;
            let apply_ns = stats.apply_ns;
            (
                0,
                0,
                stats.frames,
                stats.dirty_buckets,
                bytes,
                bytes as f64 / (1024.0 * 1024.0) / (apply_ns as f64 / 1e9).max(f64::EPSILON),
                stats.full_bytes as f64
                    / (1024.0 * 1024.0)
                    / (stats.full_apply_ns as f64 / 1e9).max(f64::EPSILON),
                stats.delta_bytes as f64
                    / (1024.0 * 1024.0)
                    / (stats.delta_apply_ns as f64 / 1e9).max(f64::EPSILON),
                stats.full,
                stats.delta,
                corpus.publisher.unchanged,
                stats.generation_conflicts,
                stats.native_fallbacks,
                stats.repeated_fallbacks,
            )
        }
    };

    let result = TrialResult {
        schema_version: 1,
        measured_code_sha: options.measured_code_sha,
        corpus_sha256: options.corpus_sha256,
        backend: options.backend,
        repetition: options.repetition,
        phase: options.phase,
        replay_window_ms: options.replay_window.as_secs_f64() * 1000.0,
        nominal_offered_mixed_ops_s,
        nominal_offered_block_ops_s,
        actual_issue_mixed_ops_s: totals.mixed_ops() as f64 / issue_seconds,
        achieved_mixed_ops_s,
        achieved_block_ops_s,
        achieved_over_offered: achieved_block_ops_s / nominal_offered_block_ops_s,
        total_elapsed_ms: total_elapsed.as_secs_f64() * 1000.0,
        generator_limited,
        kept_up,
        issue_lag: percentile_summary(issue_lag_ns),
        query_queue_wait: percentile_summary(query_samples.queue_wait_ns),
        lookup_service: percentile_summary(query_samples.service_ns),
        scheduled_to_completion: percentile_summary(query_samples.scheduled_to_completion_ns),
        update_scheduled_to_enqueue: percentile_summary(update.scheduled_to_enqueue_ns),
        update_enqueue_to_applied: percentile_summary(update.enqueue_to_applied_ns),
        update_scheduled_to_applied: percentile_summary(update.scheduled_to_applied_ns),
        query_queue,
        update_queue: update.queue,
        crtc_raw_events,
        crtc_raw_blocks,
        ckf_frames,
        ckf_dirty_buckets,
        ckf_bytes,
        ckf_apply_mib_s,
        ckf_full_apply_mib_s,
        ckf_delta_apply_mib_s,
        full_publications,
        delta_publications,
        unchanged_publications,
        generation_conflicts,
        native_fallbacks,
        repeated_fallbacks,
        errors: pipeline_errors,
        accuracy,
        rss_bytes: 0,
        pss_bytes: None,
        uss_bytes: None,
    };
    backend.shutdown();
    Ok(result)
}

async fn flush_updates(backend: &Backend) -> anyhow::Result<UpdateMetrics> {
    match backend {
        Backend::Crtc(crtc) => {
            let (
                queue,
                CrtcCompletionMetrics {
                    scheduled_to_enqueue_ns,
                    enqueue_to_applied_ns,
                    scheduled_to_applied_ns,
                    errors,
                },
            ) = crtc.flush().await?;
            Ok(UpdateMetrics {
                queue,
                scheduled_to_enqueue_ns,
                enqueue_to_applied_ns,
                scheduled_to_applied_ns,
                errors,
            })
        }
        Backend::Ckf(ckf) => {
            let core_queue = ckf.flush_with_metrics()?;
            let queue = QueueMetrics {
                at_stop: core_queue.at_stop,
                maximum_depth: core_queue.maximum_depth,
                drain_ms: core_queue.drain_ns as f64 / 1e6,
            };
            let (scheduled_to_enqueue_ns, enqueue_to_applied_ns, scheduled_to_applied_ns) =
                ckf.take_update_latencies();
            let core_errors = ckf.stats_snapshot().errors;
            Ok(UpdateMetrics {
                queue,
                scheduled_to_enqueue_ns,
                enqueue_to_applied_ns,
                scheduled_to_applied_ns,
                errors: PipelineErrors {
                    insertion: 0,
                    removal: 0,
                    decode: core_errors.decode,
                    application: core_errors.application,
                    epoch: core_errors.epoch,
                    desynchronization: core_errors.desynchronization,
                },
            })
        }
    }
}

fn final_accuracy(indexer: &CuckooFrameIndexer, samples: &[AccuracySample]) -> AccuracyMetrics {
    let mut metrics = AccuracyMetrics::default();
    for sample in samples {
        let optimized = indexer.optimized_depths(&sample.local_hashes);
        let linear = indexer.linear_depths(&sample.local_hashes);
        metrics.checked_results += optimized.len() as u64;
        if optimized != linear {
            metrics.full_map_mismatches += 1;
        }
        for (&actual, &expected) in optimized.iter().zip(&sample.exact_depths) {
            if actual > expected {
                metrics.inflated += 1;
                metrics.maximum_inflation = metrics.maximum_inflation.max(actual - expected);
            } else if actual < expected {
                metrics.under_reported += 1;
            }
        }
        if best_dc(&optimized) != best_dc(&sample.exact_depths) {
            metrics.wrong_best_dc += 1;
        }
    }
    metrics
}

fn best_dc(depths: &[u32]) -> Option<usize> {
    depths
        .iter()
        .enumerate()
        .max_by_key(|(dc, depth)| (**depth, std::cmp::Reverse(*dc)))
        .and_then(|(dc, &depth)| (depth > 0).then_some(dc))
}

fn scaled_offset(timestamp_us: u64, source_span_us: u64, replay_window: Duration) -> Duration {
    replay_window.mul_f64(timestamp_us as f64 / source_span_us as f64)
}

fn update_max(counter: &AtomicU64, value: u64) {
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
    use crate::corpus::d2_fixture_corpus;

    #[test]
    fn fixed_corpus_window_scales_deadlines() {
        let window = Duration::from_secs(24);
        assert_eq!(scaled_offset(0, 1_000, window), Duration::ZERO);
        assert_eq!(scaled_offset(500, 1_000, window), Duration::from_secs(12));
        assert_eq!(scaled_offset(1_000, 1_000, window), window);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn d2_replay_uses_backend_specific_inputs_and_drains() {
        let corpus = Arc::new(d2_fixture_corpus("fixture".to_string()).unwrap());
        for backend in [
            BackendKind::Crtc,
            BackendKind::CkfNative,
            BackendKind::CkfTransposed,
        ] {
            let result = run_trial(
                Arc::clone(&corpus),
                RunOptions {
                    backend,
                    replay_window: Duration::from_millis(50),
                    repetition: 1,
                    phase: "smoke".to_string(),
                    measured_code_sha: "fixture".to_string(),
                    corpus_sha256: "fixture".to_string(),
                    warmup_queries: 1,
                },
            )
            .await
            .unwrap();
            assert_eq!(result.errors.total(), 0, "{backend}");
            assert!(result.total_elapsed_ms >= 50.0, "{backend}");
            match backend {
                BackendKind::Crtc => {
                    assert_eq!(result.crtc_raw_events, 3);
                    assert_eq!(result.crtc_raw_blocks, 7);
                    assert_eq!(result.ckf_frames, 0);
                }
                BackendKind::CkfNative | BackendKind::CkfTransposed => {
                    assert_eq!(result.crtc_raw_events, 0);
                    assert_eq!(result.ckf_frames, 2);
                    assert_eq!(result.delta_publications, 2);
                    assert_eq!(result.unchanged_publications, 1);
                    assert_eq!(result.accuracy.inflated, 0);
                    assert_eq!(result.accuracy.under_reported, 0);
                    assert_eq!(result.accuracy.wrong_best_dc, 0);
                }
            }
        }
    }
}
