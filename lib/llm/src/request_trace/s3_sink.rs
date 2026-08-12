// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3 destination for request trace records.
//!
//! Records are batched in-process as gzipped JSONL, and each finished batch is
//! uploaded as one object via `PutObject`. Object keys use a simple time-based
//! layout for PR 1
//! (`{prefix}/{yyyy}/{mm}/{dd}/{host}-{HHMMSS}-{run_id}-{seq}.jsonl.gz`);
//! richer partitioning ships in a follow-up.
//!
//! Credentials come from the shared `object_store` S3 client — environment
//! variables, IMDS, IRSA, and Pod Identity are supported. Shared AWS config and
//! credential profiles are not loaded; how the frontend pod is credentialed is
//! a deployment concern, not this sink's.
//!
//! # Upload concurrency
//!
//! A single worker task drains the record channel and uploads each finished
//! batch inline (same shape as the OpenTelemetry Rust `BatchLogProcessor` and
//! the local `telemetry::jsonl_gz` writer). While an upload is in flight the
//! worker is not draining, so a slow `PutObject` applies backpressure and
//! `emit` drops records once the channel fills. Drops are counted and surfaced
//! (see [`S3RequestTraceSink::emit`]). Overlapping uploads with a bounded
//! concurrency pool so the drain never stalls is a follow-up; it belongs with
//! the object-layout work where the sink is restructured to carry more context.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use flate2::{Compression, write::GzEncoder};
use object_store::{
    Attribute, Attributes, ClientConfigKey, ObjectStore, RetryConfig,
    aws::{AmazonS3Builder, AmazonS3ConfigKey},
    path::Path as ObjectPath,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

use super::RequestTraceRecord;
use super::config::RequestTracePolicy;
use super::sink::RequestTraceSink;

const CHANNEL_CAPACITY: usize = 2048;
const DEFAULT_BUFFER_INITIAL_BYTES: usize = 256 * 1024;
// Keep the age gauge current while a partially filled batch waits for the
// regular flush tick. The batch is reset before gzip/upload begins, so there
// is no active in-memory batch to report while an upload is in flight.
const METRICS_REFRESH_INTERVAL: Duration = Duration::from_secs(1);
// Bound S3 upload duration so a stalled endpoint or slow network cannot wedge
// the worker task indefinitely. `attempt_timeout` covers a single HTTP attempt;
// the outer timeout bounds the full call including retries (three total).
// After the operation timeout expires the batch is discarded with
// a warning; a persistent retry queue is deferred to a follow-up PR.
const S3_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
const S3_OPERATION_TIMEOUT: Duration = Duration::from_secs(90);

/// Prometheus metrics for the direct S3 request-trace sink.
///
/// These are deliberately process-wide because request tracing starts at most
/// one worker per process. Metric labels are limited to the fixed drop reasons;
/// bucket, object key, and error text must not become labels.
struct S3RequestTraceMetrics {
    records_accepted: prometheus::IntCounter,
    records_dropped: prometheus::IntCounterVec,
    queue_depth: prometheus::IntGauge,
    in_memory_batch_bytes: prometheus::IntGauge,
    oldest_buffered_record_age_seconds: prometheus::Gauge,
    upload_duration_seconds: prometheus::Histogram,
    uploaded_bytes: prometheus::IntCounter,
    upload_terminal_retry_attempts: prometheus::IntCounter,
    upload_terminal_failures: prometheus::IntCounter,
}

impl S3RequestTraceMetrics {
    fn new() -> Self {
        let metric = |suffix: &str| format!("dynamo_frontend_request_trace_s3_{suffix}");
        Self {
            records_accepted: prometheus::IntCounter::new(
                metric("records_accepted_total"),
                "Request-trace records accepted by the S3 sink queue",
            )
            .expect("valid S3 request-trace accepted-record metric"),
            records_dropped: prometheus::IntCounterVec::new(
                prometheus::Opts::new(
                    metric("records_dropped_total"),
                    "Request-trace records dropped before S3 upload",
                ),
                &["reason"],
            )
            .expect("valid S3 request-trace dropped-record metric"),
            queue_depth: prometheus::IntGauge::new(
                metric("queue_depth"),
                "Current number of request-trace records queued for the S3 sink",
            )
            .expect("valid S3 request-trace queue-depth metric"),
            in_memory_batch_bytes: prometheus::IntGauge::new(
                metric("in_memory_batch_uncompressed_bytes"),
                "Uncompressed request-trace bytes buffered in the active in-memory S3 batch",
            )
            .expect("valid S3 request-trace in-memory-bytes metric"),
            oldest_buffered_record_age_seconds: prometheus::Gauge::new(
                metric("oldest_buffered_record_age_seconds"),
                "Age of the oldest record in the active in-memory S3 batch",
            )
            .expect("valid S3 request-trace oldest-buffered-record metric"),
            upload_duration_seconds: prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(
                    metric("upload_duration_seconds"),
                    "End-to-end duration of a completed or terminally failed S3 upload",
                )
                .buckets(vec![
                    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 90.0,
                ]),
            )
            .expect("valid S3 request-trace upload-duration metric"),
            uploaded_bytes: prometheus::IntCounter::new(
                metric("uploaded_bytes_total"),
                "Compressed request-trace bytes successfully uploaded to S3",
            )
            .expect("valid S3 request-trace uploaded-bytes metric"),
            upload_terminal_retry_attempts: prometheus::IntCounter::new(
                metric("upload_terminal_retry_attempts_total"),
                "Object-store retry attempts reported by terminally failed S3 uploads",
            )
            .expect("valid S3 request-trace terminal-retry-attempt metric"),
            upload_terminal_failures: prometheus::IntCounter::new(
                metric("upload_terminal_failures_total"),
                "S3 upload batches discarded after the object-store retry policy terminates",
            )
            .expect("valid S3 request-trace terminal-failure metric"),
        }
    }

    fn register(&self, registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.records_accepted.clone()))?;
        registry.register(Box::new(self.records_dropped.clone()))?;
        registry.register(Box::new(self.queue_depth.clone()))?;
        registry.register(Box::new(self.in_memory_batch_bytes.clone()))?;
        registry.register(Box::new(self.oldest_buffered_record_age_seconds.clone()))?;
        registry.register(Box::new(self.upload_duration_seconds.clone()))?;
        registry.register(Box::new(self.uploaded_bytes.clone()))?;
        registry.register(Box::new(self.upload_terminal_retry_attempts.clone()))?;
        registry.register(Box::new(self.upload_terminal_failures.clone()))?;
        Ok(())
    }

    fn reset_gauges(&self) {
        self.queue_depth.set(0);
        self.in_memory_batch_bytes.set(0);
        self.oldest_buffered_record_age_seconds.set(0.0);
    }

    fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.set(depth.try_into().unwrap_or(i64::MAX));
    }

    fn set_batch_state(&self, bytes: u64, oldest: Option<Instant>) {
        self.in_memory_batch_bytes
            .set(bytes.try_into().unwrap_or(i64::MAX));
        self.oldest_buffered_record_age_seconds.set(
            oldest
                .map(|enqueued_at| enqueued_at.elapsed().as_secs_f64())
                .unwrap_or(0.0),
        );
    }

    fn record_drop(&self, reason: &'static str, count: u64) {
        self.records_dropped
            .with_label_values(&[reason])
            .inc_by(count);
    }
}

static S3_REQUEST_TRACE_METRICS: LazyLock<S3RequestTraceMetrics> =
    LazyLock::new(S3RequestTraceMetrics::new);

pub(super) fn register_metrics(registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
    S3_REQUEST_TRACE_METRICS.register(registry)
}

pub(super) fn record_bus_lagged_records(count: u64) {
    S3_REQUEST_TRACE_METRICS.record_drop("bus_lag", count);
}

#[derive(Clone)]
struct QueuedRecord {
    record: RequestTraceRecord,
    enqueued_at: Instant,
}

pub struct S3RequestTraceSink {
    tx: mpsc::Sender<QueuedRecord>,
    shutdown: CancellationToken,
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Count of records dropped by `emit` because the batcher channel was full
    /// or closed. Surfaced once on the first drop and again as a summary at
    /// shutdown, so a slow S3 does not produce one log line per dropped record.
    dropped: Arc<AtomicU64>,
}

#[derive(Clone)]
struct S3UploadOptions {
    bucket: String,
    prefix: String,
    host: String,
    /// Per-process UUID mixed into every object key so that a pod restart
    /// (which reuses hostname and can land within the same second) can't
    /// overwrite a previous batch, and two frontends sharing a hostname
    /// stay disjoint.
    run_id: String,
}

impl S3RequestTraceSink {
    pub async fn from_policy(policy: &RequestTracePolicy) -> Result<Self> {
        let bucket = policy.s3_bucket.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "{} must be set when {} includes s3",
                env_request_trace::DYN_REQUEST_TRACE_S3_BUCKET,
                env_request_trace::DYN_REQUEST_TRACE_SINKS,
            )
        })?;
        let prefix = policy.s3_prefix.clone().unwrap_or_default();
        let host = hostname_or_fallback();
        let run_id = Uuid::new_v4().simple().to_string();
        let roll_uncompressed_bytes = policy.s3_roll_uncompressed_bytes;
        let flush_interval = Duration::from_millis(policy.s3_flush_interval_ms.max(1));

        let store: Arc<dyn ObjectStore> = Arc::new(
            request_trace_s3_builder(&bucket, policy.s3_region.as_deref())
                .build()
                .context("building request trace S3 client")?,
        );

        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        S3_REQUEST_TRACE_METRICS.reset_gauges();
        let shutdown = CancellationToken::new();
        let upload_options = S3UploadOptions {
            bucket,
            prefix,
            host,
            run_id,
        };
        let worker_shutdown = shutdown.clone();
        let worker = tokio::spawn(async move {
            run_worker(
                store,
                upload_options,
                rx,
                worker_shutdown,
                roll_uncompressed_bytes,
                flush_interval,
            )
            .await;
        });

        Ok(Self {
            tx,
            shutdown,
            worker: Mutex::new(Some(worker)),
            dropped: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Record one dropped record and, only on the first drop, emit a single
    /// warning. Subsequent drops bump the counter silently; the running total
    /// is reported once at shutdown. Mirrors the OpenTelemetry Rust
    /// `BatchLogProcessor` drop-accounting pattern so a degraded S3 endpoint
    /// cannot flood the log with one line per dropped record. Returns `true`
    /// when this call emitted the warning (i.e. it was the first drop).
    fn note_dropped(&self, reason: &str) -> bool {
        let metric_reason = match reason {
            "channel_full" => "channel_full",
            "channel_closed" => "channel_closed",
            _ => "unknown",
        };
        S3_REQUEST_TRACE_METRICS.record_drop(metric_reason, 1);
        if self.dropped.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                reason,
                "request trace s3: dropping records (batcher backpressure); \
                 further drops are counted and summarized at shutdown"
            );
            true
        } else {
            false
        }
    }
}

#[async_trait]
impl RequestTraceSink for S3RequestTraceSink {
    fn name(&self) -> &'static str {
        "s3"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        if let Err(error) = self.tx.try_send(QueuedRecord {
            record: record.clone(),
            enqueued_at: Instant::now(),
        }) {
            let reason = match error {
                mpsc::error::TrySendError::Full(_) => "channel_full",
                mpsc::error::TrySendError::Closed(_) => "channel_closed",
            };
            self.note_dropped(reason);
        } else {
            S3_REQUEST_TRACE_METRICS.records_accepted.inc();
            S3_REQUEST_TRACE_METRICS.set_queue_depth(CHANNEL_CAPACITY - self.tx.capacity());
        }
    }

    async fn shutdown(&self) {
        self.shutdown.cancel();
        // Recover the guard even if a prior panic poisoned the lock; it only
        // guards an `Option<JoinHandle>`, so a poisoned lock must not turn
        // teardown into a second panic.
        let worker = self
            .worker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(worker) = worker
            && let Err(error) = worker.await
        {
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                error = %error,
                "request trace s3: batcher task join failed during shutdown"
            );
        }
        let dropped = self.dropped.load(Ordering::Relaxed);
        if dropped > 0 {
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                dropped,
                "request trace s3: dropped records during the run (batcher backpressure)"
            );
        }
    }
}

async fn run_worker(
    store: Arc<dyn ObjectStore>,
    options: S3UploadOptions,
    mut rx: mpsc::Receiver<QueuedRecord>,
    shutdown: CancellationToken,
    roll_uncompressed_bytes: u64,
    flush_interval: Duration,
) {
    let uploader = Arc::new(S3Uploader { store, options });
    let mut batch = JsonlBatch::new();
    let mut seq: u64 = 0;
    let mut flush_tick = tokio::time::interval(flush_interval);
    flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut metrics_tick = tokio::time::interval(METRICS_REFRESH_INTERVAL);
    metrics_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Skip the immediate tick that `interval` fires at t=0.
    flush_tick.tick().await;
    metrics_tick.tick().await;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                // Close the receiver first so an in-flight `emit()` cannot
                // land a record after we start draining. Then `recv()` yields
                // every already-enqueued record and returns `None` once empty.
                rx.close();
                while let Some(queued) = rx.recv().await {
                    S3_REQUEST_TRACE_METRICS.set_queue_depth(rx.len());
                    if let Err(error) = batch.push(&queued.record, queued.enqueued_at) {
                        tracing::warn!(
                            target: "dynamo_llm::request_trace",
                            %error,
                            "request trace s3: serialize failed during shutdown"
                        );
                        S3_REQUEST_TRACE_METRICS.record_drop("serialization_failed", 1);
                    } else if batch.uncompressed_bytes() >= roll_uncompressed_bytes {
                        // Enforce the roll threshold during shutdown too, so a
                        // full channel can't collapse into one oversized PUT
                        // that loses everything on a single upload failure.
                        upload_ready_batch(&uploader, &mut batch, &mut seq).await;
                    }
                }
                if !batch.is_empty() {
                    upload_ready_batch(&uploader, &mut batch, &mut seq).await;
                }
                return;
            }
            _ = flush_tick.tick() => {
                if !batch.is_empty() {
                    upload_ready_batch(&uploader, &mut batch, &mut seq).await;
                }
            }
            _ = metrics_tick.tick() => {
                if !batch.is_empty() {
                    batch.refresh_metrics();
                }
            }
            message = rx.recv() => {
                match message {
                    Some(record) => {
                        S3_REQUEST_TRACE_METRICS.set_queue_depth(rx.len());
                        if let Err(error) = batch.push(&record.record, record.enqueued_at) {
                            tracing::warn!(
                                target: "dynamo_llm::request_trace",
                                %error,
                                "request trace s3: serialize failed; dropping record"
                            );
                            S3_REQUEST_TRACE_METRICS.record_drop("serialization_failed", 1);
                        } else if batch.uncompressed_bytes() >= roll_uncompressed_bytes {
                            upload_ready_batch(&uploader, &mut batch, &mut seq).await;
                        }
                    }
                    None => {
                        if !batch.is_empty() {
                            upload_ready_batch(&uploader, &mut batch, &mut seq).await;
                        }
                        return;
                    }
                }
            }
        }
    }
}

async fn upload_ready_batch(uploader: &Arc<S3Uploader>, batch: &mut JsonlBatch, seq: &mut u64) {
    let batch_records = batch.lines();
    let ready = match batch.take_finished().await {
        Ok(bytes) => bytes,
        Err(error) => {
            S3_REQUEST_TRACE_METRICS.record_drop("gzip_finalize_failed", batch_records);
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                %error,
                "request trace s3: finalize gzip batch failed; discarding"
            );
            return;
        }
    };
    let this_seq = *seq;
    *seq = seq.saturating_add(1);
    let key = uploader.object_key(SystemTime::now(), this_seq);
    let batch_bytes = ready.bytes.len();
    let upload_started = Instant::now();
    match uploader.put_object(key.clone(), ready.bytes).await {
        Ok(()) => {
            S3_REQUEST_TRACE_METRICS
                .upload_duration_seconds
                .observe(upload_started.elapsed().as_secs_f64());
            S3_REQUEST_TRACE_METRICS
                .uploaded_bytes
                .inc_by(batch_bytes as u64);
        }
        Err(error) => {
            S3_REQUEST_TRACE_METRICS
                .upload_duration_seconds
                .observe(upload_started.elapsed().as_secs_f64());
            S3_REQUEST_TRACE_METRICS.upload_terminal_failures.inc();
            S3_REQUEST_TRACE_METRICS.record_drop("terminal_upload_failure", ready.records);
            S3_REQUEST_TRACE_METRICS
                .upload_terminal_retry_attempts
                .inc_by(terminal_retry_attempts(&error));
            // The client exhausted its retries (three total attempts, bounded by
            // the operation timeout). The batch is dropped here rather
            // than requeued; a persistent retry buffer is a follow-up concern
            // tracked in the S3 layout PR.
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                key = %key,
                batch_bytes,
                %error,
                "request trace s3: put_object failed after retries; batch discarded"
            );
        }
    }
}

/// `object_store` exposes retry counts only in the terminal `RetryError`
/// display text. Successful uploads do not surface their internal attempts.
/// Keep this parser narrow so changes to the dependency's message fail closed
/// to zero rather than inventing retry attempts.
fn terminal_retry_attempts(error: &anyhow::Error) -> u64 {
    const PREFIX: &str = "after ";
    const SUFFIX: &str = " retries";

    error
        .chain()
        .find_map(|cause| {
            let message = cause.to_string();
            let start = message.find(PREFIX)? + PREFIX.len();
            let end = message[start..].find(SUFFIX)? + start;
            message[start..end].parse().ok()
        })
        .unwrap_or(0)
}

struct S3Uploader {
    store: Arc<dyn ObjectStore>,
    options: S3UploadOptions,
}

impl S3Uploader {
    fn object_key(&self, at: SystemTime, seq: u64) -> String {
        // Simple time-based layout for PR 1. Richer partitioning
        // (model=/date=/hour= Hive style) lands in the follow-up.
        //
        // `run_id` guarantees per-process uniqueness so that a pod restart
        // within the same second (or two frontends sharing a hostname)
        // cannot overwrite each other's objects.
        let secs = at
            .duration_since(UNIX_EPOCH)
            .map(|dur| dur.as_secs())
            .unwrap_or_default();
        let (yyyy, mm, dd, hh, mi, ss) = utc_date_parts(secs);
        let mut key = String::new();
        let prefix = self.options.prefix.trim_matches('/');
        if !prefix.is_empty() {
            key.push_str(prefix);
            key.push('/');
        }
        key.push_str(&format!(
            "{yyyy:04}/{mm:02}/{dd:02}/{host}-{hh:02}{mi:02}{ss:02}-{run_id}-{seq:06}.jsonl.gz",
            host = self.options.host,
            run_id = self.options.run_id,
        ));
        key
    }

    async fn put_object(&self, key: String, body: Vec<u8>) -> Result<()> {
        let location = ObjectPath::parse(&key)
            .with_context(|| format!("invalid request trace S3 object key {key:?}"))?;
        let attributes = Attributes::from_iter([(Attribute::ContentType, "application/gzip")]);
        tokio::time::timeout(
            S3_OPERATION_TIMEOUT,
            self.store
                .put_opts(&location, body.into(), attributes.into()),
        )
        .await
        .with_context(|| {
            format!(
                "s3 put_object into bucket {} timed out",
                self.options.bucket
            )
        })?
        .with_context(|| format!("s3 put_object into bucket {}", self.options.bucket))?;
        Ok(())
    }
}

fn request_trace_s3_builder(bucket: &str, region: Option<&str>) -> AmazonS3Builder {
    let retry_config = RetryConfig {
        max_retries: 2,
        retry_timeout: S3_OPERATION_TIMEOUT,
        ..Default::default()
    };
    let mut builder = AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .with_config(
            AmazonS3ConfigKey::Client(ClientConfigKey::Timeout),
            format!("{}s", S3_ATTEMPT_TIMEOUT.as_secs()),
        )
        .with_retry(retry_config);
    if let Some(region) = region {
        builder = builder.with_region(region);
    } else if let Ok(region) = std::env::var("AWS_REGION") {
        builder = builder.with_region(region);
    }
    builder
}

/// Buffers raw JSONL bytes until the batch is ready to flush; gzip
/// compression happens in [`take_finished`], which offloads the CPU work
/// to a blocking pool (matching `telemetry::jsonl_gz`).
struct JsonlBatch {
    raw: Vec<u8>,
    lines: u64,
    oldest_enqueued_at: Option<Instant>,
}

struct FinishedBatch {
    bytes: Vec<u8>,
    records: u64,
}

impl JsonlBatch {
    fn new() -> Self {
        Self {
            raw: Vec::with_capacity(DEFAULT_BUFFER_INITIAL_BYTES),
            lines: 0,
            oldest_enqueued_at: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.lines == 0
    }

    fn uncompressed_bytes(&self) -> u64 {
        self.raw.len() as u64
    }

    fn lines(&self) -> u64 {
        self.lines
    }

    fn push(&mut self, record: &RequestTraceRecord, enqueued_at: Instant) -> Result<()> {
        let mut line = serde_json::to_vec(record).context("serializing request trace record")?;
        line.push(b'\n');
        self.raw.extend_from_slice(&line);
        self.lines = self.lines.saturating_add(1);
        self.oldest_enqueued_at.get_or_insert(enqueued_at);
        self.refresh_metrics();
        Ok(())
    }

    fn refresh_metrics(&self) {
        self.refresh_metrics_with(&S3_REQUEST_TRACE_METRICS);
    }

    fn refresh_metrics_with(&self, metrics: &S3RequestTraceMetrics) {
        metrics.set_batch_state(self.uncompressed_bytes(), self.oldest_enqueued_at);
    }

    /// Consume the accumulated JSONL, gzip it on a blocking worker, and
    /// leave the batch empty so it can accept the next record.
    async fn take_finished(&mut self) -> Result<FinishedBatch> {
        let raw = std::mem::replace(
            &mut self.raw,
            Vec::with_capacity(DEFAULT_BUFFER_INITIAL_BYTES),
        );
        let records = self.lines;
        self.lines = 0;
        self.oldest_enqueued_at = None;
        S3_REQUEST_TRACE_METRICS.set_batch_state(0, None);
        let bytes = tokio::task::spawn_blocking(move || {
            let mut encoder = GzEncoder::new(
                Vec::with_capacity(raw.len() / 4 + DEFAULT_BUFFER_INITIAL_BYTES),
                Compression::default(),
            );
            encoder
                .write_all(&raw)
                .context("writing request trace batch to gzip encoder")?;
            encoder
                .finish()
                .context("finalizing gzip batch for s3 upload")
        })
        .await
        .context("gzip encoder task panicked")??;
        Ok(FinishedBatch { bytes, records })
    }
}

fn hostname_or_fallback() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Convert unix seconds to UTC (year, month, day, hour, minute, second).
/// Implemented inline to avoid pulling in chrono for one call site.
fn utc_date_parts(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    // 1970-01-01 is a Thursday. Compute days since epoch, then split.
    const SECS_PER_DAY: u64 = 86_400;
    let days = (secs / SECS_PER_DAY) as i64;
    let time_of_day = secs % SECS_PER_DAY;
    let hh = (time_of_day / 3600) as u32;
    let mi = ((time_of_day % 3600) / 60) as u32;
    let ss = (time_of_day % 60) as u32;

    // Howard Hinnant civil_from_days
    let z = days + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32, hh, mi, ss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_trace::{RequestTraceEventType, RequestTraceSchema};
    use object_store::memory::InMemory;

    #[test]
    fn utc_date_parts_epoch() {
        let (y, m, d, h, mi, s) = utc_date_parts(0);
        assert_eq!((y, m, d, h, mi, s), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn utc_date_parts_known_moment() {
        // 2026-07-15T12:34:56Z = 1_784_118_896 (`date -u -d ...`)
        let (y, m, d, h, mi, s) = utc_date_parts(1_784_118_896);
        assert_eq!((y, m, d, h, mi, s), (2026, 7, 15, 12, 34, 56));
    }

    #[test]
    fn object_key_includes_prefix_date_run_and_seq() {
        let uploader = test_uploader("traces/");
        let at = UNIX_EPOCH + Duration::from_secs(1_784_118_896);
        let key = uploader.object_key(at, 42);
        assert_eq!(
            key,
            "traces/2026/07/15/frontend-0-123456-cafebabe-000042.jsonl.gz"
        );
    }

    #[test]
    fn object_key_omits_leading_slash_when_prefix_empty() {
        let uploader = test_uploader("");
        let key = uploader.object_key(UNIX_EPOCH, 0);
        assert!(!key.starts_with('/'));
        assert!(key.starts_with("1970/01/01/frontend-0-"));
    }

    #[tokio::test]
    async fn put_object_writes_body_and_content_type() {
        let store = Arc::new(InMemory::new());
        let uploader = S3Uploader {
            store: store.clone(),
            options: test_options(""),
        };
        let key = "traces/batch.jsonl.gz";
        let body = vec![1, 2, 3, 4];

        uploader
            .put_object(key.to_string(), body.clone())
            .await
            .unwrap();

        let result = store.get(&ObjectPath::from(key)).await.unwrap();
        assert_eq!(
            result
                .attributes
                .get(&Attribute::ContentType)
                .map(AsRef::as_ref),
            Some("application/gzip")
        );
        assert_eq!(result.bytes().await.unwrap().as_ref(), body.as_slice());
    }

    #[tokio::test]
    async fn put_object_preserves_percent_encoded_key() {
        let store = Arc::new(InMemory::new());
        let uploader = S3Uploader {
            store: store.clone(),
            options: test_options(""),
        };
        let key = "traces/%2F/batch.jsonl.gz";
        let body = vec![1, 2, 3, 4];

        uploader
            .put_object(key.to_string(), body.clone())
            .await
            .unwrap();

        let location = ObjectPath::parse(key).unwrap();
        let result = store.get(&location).await.unwrap();
        assert_eq!(result.bytes().await.unwrap().as_ref(), body.as_slice());
    }

    #[tokio::test]
    async fn put_object_rejects_key_with_empty_segment() {
        let uploader = test_uploader("");

        let error = uploader
            .put_object("traces//batch.jsonl.gz".to_string(), vec![1, 2, 3, 4])
            .await
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("invalid request trace S3 object key")
        );
    }

    #[test]
    fn s3_builder_preserves_environment_client_options() {
        temp_env::with_vars(
            [
                ("AWS_ALLOW_HTTP", Some("true")),
                ("AWS_PROXY_URL", Some("http://proxy.example:8080")),
            ],
            || {
                let builder = request_trace_s3_builder("b", Some("us-west-2"));
                assert_eq!(
                    builder
                        .get_config_value(&AmazonS3ConfigKey::Client(ClientConfigKey::AllowHttp))
                        .as_deref(),
                    Some("true")
                );
                assert_eq!(
                    builder
                        .get_config_value(&AmazonS3ConfigKey::Client(ClientConfigKey::ProxyUrl))
                        .as_deref(),
                    Some("http://proxy.example:8080")
                );
            },
        );
    }

    #[test]
    fn s3_metrics_register_all_metric_families() {
        let registry = prometheus::Registry::new();
        S3RequestTraceMetrics::new().register(&registry).unwrap();
        let names: Vec<_> = registry
            .gather()
            .into_iter()
            .map(|family| family.name().to_string())
            .collect();

        assert_eq!(
            names,
            vec![
                "dynamo_frontend_request_trace_s3_in_memory_batch_uncompressed_bytes",
                "dynamo_frontend_request_trace_s3_oldest_buffered_record_age_seconds",
                "dynamo_frontend_request_trace_s3_queue_depth",
                "dynamo_frontend_request_trace_s3_records_accepted_total",
                "dynamo_frontend_request_trace_s3_records_dropped_total",
                "dynamo_frontend_request_trace_s3_upload_duration_seconds",
                "dynamo_frontend_request_trace_s3_upload_terminal_failures_total",
                "dynamo_frontend_request_trace_s3_upload_terminal_retry_attempts_total",
                "dynamo_frontend_request_trace_s3_uploaded_bytes_total",
            ]
        );
    }

    #[test]
    fn batch_metrics_refresh_advances_oldest_record_age() {
        let metrics = S3RequestTraceMetrics::new();
        let mut batch = JsonlBatch::new();
        batch.raw.extend_from_slice(b"record\n");
        batch.lines = 1;
        batch.oldest_enqueued_at = Some(Instant::now() - Duration::from_secs(1));

        batch.refresh_metrics_with(&metrics);
        let initial_age = metrics.oldest_buffered_record_age_seconds.get();
        std::thread::sleep(Duration::from_millis(10));
        batch.refresh_metrics_with(&metrics);

        assert!(metrics.oldest_buffered_record_age_seconds.get() > initial_age);
    }

    #[test]
    fn terminal_retry_attempts_extracts_object_store_retry_count() {
        let error = anyhow::anyhow!(
            "Generic S3 error: Error performing PUT https://bucket.example/key in 90s, after 2 retries, max_retries: 2, retry_timeout: 90s - timeout"
        );
        assert_eq!(terminal_retry_attempts(&error), 2);
        assert_eq!(
            terminal_retry_attempts(&anyhow::anyhow!("permission denied")),
            0
        );
    }

    fn test_uploader(prefix: &str) -> S3Uploader {
        S3Uploader {
            store: Arc::new(InMemory::new()),
            options: test_options(prefix),
        }
    }

    fn test_options(prefix: &str) -> S3UploadOptions {
        S3UploadOptions {
            bucket: "b".to_string(),
            prefix: prefix.to_string(),
            host: "frontend-0".to_string(),
            run_id: "cafebabe".to_string(),
        }
    }

    /// Build a sink whose bounded channel is never drained, so `emit` hits the
    /// same backpressure path a stalled uploader would cause. The receiver is
    /// returned to the caller and held so the channel stays open (full), not
    /// closed. No S3 client or worker task is created.
    fn stalled_sink(capacity: usize) -> (S3RequestTraceSink, mpsc::Receiver<QueuedRecord>) {
        let (tx, rx) = mpsc::channel(capacity);
        let sink = S3RequestTraceSink {
            tx,
            shutdown: CancellationToken::new(),
            worker: Mutex::new(None),
            dropped: Arc::new(AtomicU64::new(0)),
        };
        (sink, rx)
    }

    fn sample_record() -> RequestTraceRecord {
        RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::RequestEnd,
            event_time_unix_ms: 0,
            event_source: None,
            agent_context: None,
            request: None,
            tool: None,
            payload: None,
        }
    }

    #[tokio::test]
    async fn emit_drops_and_counts_when_channel_is_full() {
        let capacity = 4;
        // Hold the receiver so the channel stays open; never drain it.
        let (sink, _rx) = stalled_sink(capacity);

        // The channel accepts `capacity` records, then every further emit drops.
        let total = capacity + 20;
        for _ in 0..total {
            sink.emit(&sample_record()).await;
        }

        let dropped = sink.dropped.load(Ordering::Relaxed);
        assert_eq!(dropped, (total - capacity) as u64);
    }

    #[test]
    fn note_dropped_warns_only_on_first_drop() {
        let (sink, _rx) = stalled_sink(1);
        // First drop emits the warning; every subsequent drop is silent.
        assert!(sink.note_dropped("channel_full"));
        for _ in 0..1000 {
            assert!(!sink.note_dropped("channel_full"));
        }
        assert_eq!(sink.dropped.load(Ordering::Relaxed), 1001);
    }
}
