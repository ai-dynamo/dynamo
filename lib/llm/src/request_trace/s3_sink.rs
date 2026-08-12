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
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use flate2::{Compression, write::GzEncoder};
use object_store::{
    Attribute, Attributes, BackoffConfig, ClientConfigKey, ObjectStore, RetryConfig,
    aws::{AmazonS3Builder, AmazonS3ConfigKey},
    path::Path as ObjectPath,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

use super::RequestTraceRecord;
use super::config::{RequestTracePolicy, s3_upload_policy};
use super::sink::RequestTraceSink;

const CHANNEL_CAPACITY: usize = 2048;
const DEFAULT_BUFFER_INITIAL_BYTES: usize = 256 * 1024;
// Bound S3 upload duration so a stalled endpoint or slow network cannot wedge
// the worker task indefinitely. `attempt_timeout` covers a single HTTP attempt;
// the outer timeout bounds the full call including retries. After the operation
// timeout expires the batch is discarded with
// a warning; a persistent retry queue is deferred to a follow-up PR.

pub struct S3RequestTraceSink {
    tx: mpsc::Sender<RequestTraceRecord>,
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
    attempt_timeout: Duration,
    operation_timeout: Duration,
    max_retries: usize,
    retry_initial_backoff: Duration,
    retry_max_backoff: Duration,
    retry_backoff_base: f64,
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
        let s3_upload = s3_upload_policy();
        let upload_options = S3UploadOptions {
            bucket,
            prefix,
            host,
            run_id,
            attempt_timeout: Duration::from_millis(s3_upload.attempt_timeout_ms),
            operation_timeout: Duration::from_millis(s3_upload.operation_timeout_ms),
            max_retries: s3_upload.max_retries,
            retry_initial_backoff: Duration::from_millis(s3_upload.retry_initial_backoff_ms),
            retry_max_backoff: Duration::from_millis(s3_upload.retry_max_backoff_ms),
            retry_backoff_base: s3_upload.retry_backoff_base,
        };

        let store: Arc<dyn ObjectStore> = Arc::new(
            request_trace_s3_builder(&upload_options, policy.s3_region.as_deref())
                .build()
                .context("building request trace S3 client")?,
        );

        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        let shutdown = CancellationToken::new();
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
        if let Err(error) = self.tx.try_send(record.clone()) {
            let reason = match error {
                mpsc::error::TrySendError::Full(_) => "channel_full",
                mpsc::error::TrySendError::Closed(_) => "channel_closed",
            };
            self.note_dropped(reason);
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
    mut rx: mpsc::Receiver<RequestTraceRecord>,
    shutdown: CancellationToken,
    roll_uncompressed_bytes: u64,
    flush_interval: Duration,
) {
    let uploader = Arc::new(S3Uploader { store, options });
    let mut batch = JsonlBatch::new();
    let mut seq: u64 = 0;
    let mut flush_tick = tokio::time::interval(flush_interval);
    flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Skip the immediate tick that `interval` fires at t=0.
    flush_tick.tick().await;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                // Close the receiver first so an in-flight `emit()` cannot
                // land a record after we start draining. Then `recv()` yields
                // every already-enqueued record and returns `None` once empty.
                rx.close();
                while let Some(record) = rx.recv().await {
                    if let Err(error) = batch.push(&record) {
                        tracing::warn!(
                            target: "dynamo_llm::request_trace",
                            %error,
                            "request trace s3: serialize failed during shutdown"
                        );
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
            message = rx.recv() => {
                match message {
                    Some(record) => {
                        if let Err(error) = batch.push(&record) {
                            tracing::warn!(
                                target: "dynamo_llm::request_trace",
                                %error,
                                "request trace s3: serialize failed; dropping record"
                            );
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
    let ready = match batch.take_finished().await {
        Ok(bytes) => bytes,
        Err(error) => {
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
    let batch_bytes = ready.len();
    if let Err(error) = uploader.put_object(key.clone(), ready).await {
        // The client exhausted its configured retry limit (bounded by the
        // operation timeout). The batch is dropped here rather
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
            self.options.operation_timeout,
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

fn request_trace_s3_builder(options: &S3UploadOptions, region: Option<&str>) -> AmazonS3Builder {
    let mut builder = AmazonS3Builder::from_env()
        .with_bucket_name(&options.bucket)
        .with_config(
            AmazonS3ConfigKey::Client(ClientConfigKey::Timeout),
            format!("{}ms", options.attempt_timeout.as_millis()),
        )
        .with_retry(request_trace_s3_retry_config(options));
    if let Some(region) = region {
        builder = builder.with_region(region);
    } else if let Ok(region) = std::env::var("AWS_REGION") {
        builder = builder.with_region(region);
    }
    builder
}

fn request_trace_s3_retry_config(options: &S3UploadOptions) -> RetryConfig {
    RetryConfig {
        backoff: BackoffConfig {
            init_backoff: options.retry_initial_backoff,
            max_backoff: options.retry_max_backoff,
            base: options.retry_backoff_base,
        },
        max_retries: options.max_retries,
        retry_timeout: options.operation_timeout,
    }
}

/// Buffers raw JSONL bytes until the batch is ready to flush; gzip
/// compression happens in [`take_finished`], which offloads the CPU work
/// to a blocking pool (matching `telemetry::jsonl_gz`).
struct JsonlBatch {
    raw: Vec<u8>,
    lines: u64,
}

impl JsonlBatch {
    fn new() -> Self {
        Self {
            raw: Vec::with_capacity(DEFAULT_BUFFER_INITIAL_BYTES),
            lines: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.lines == 0
    }

    fn uncompressed_bytes(&self) -> u64 {
        self.raw.len() as u64
    }

    fn push(&mut self, record: &RequestTraceRecord) -> Result<()> {
        let mut line = serde_json::to_vec(record).context("serializing request trace record")?;
        line.push(b'\n');
        self.raw.extend_from_slice(&line);
        self.lines = self.lines.saturating_add(1);
        Ok(())
    }

    /// Consume the accumulated JSONL, gzip it on a blocking worker, and
    /// leave the batch empty so it can accept the next record.
    async fn take_finished(&mut self) -> Result<Vec<u8>> {
        let raw = std::mem::replace(
            &mut self.raw,
            Vec::with_capacity(DEFAULT_BUFFER_INITIAL_BYTES),
        );
        self.lines = 0;
        tokio::task::spawn_blocking(move || {
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
        .context("gzip encoder task panicked")?
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
                let builder = request_trace_s3_builder(&test_options(""), Some("us-west-2"));
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
    fn s3_builder_applies_upload_policy() {
        let mut options = test_options("");
        options.attempt_timeout = Duration::from_millis(2_500);
        options.operation_timeout = Duration::from_secs(12);
        options.max_retries = 4;
        options.retry_initial_backoff = Duration::from_millis(250);
        options.retry_max_backoff = Duration::from_secs(5);
        options.retry_backoff_base = 3.5;

        let builder = request_trace_s3_builder(&options, Some("us-west-2"));
        assert_eq!(
            builder
                .get_config_value(&AmazonS3ConfigKey::Client(ClientConfigKey::Timeout))
                .as_deref(),
            Some("2500ms")
        );

        let retry = request_trace_s3_retry_config(&options);
        assert_eq!(retry.max_retries, 4);
        assert_eq!(retry.retry_timeout, Duration::from_secs(12));
        assert_eq!(retry.backoff.init_backoff, Duration::from_millis(250));
        assert_eq!(retry.backoff.max_backoff, Duration::from_secs(5));
        assert_eq!(retry.backoff.base, 3.5);
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
            attempt_timeout: Duration::from_secs(30),
            operation_timeout: Duration::from_secs(90),
            max_retries: 2,
            retry_initial_backoff: Duration::from_millis(100),
            retry_max_backoff: Duration::from_secs(15),
            retry_backoff_base: 2.0,
        }
    }

    /// Build a sink whose bounded channel is never drained, so `emit` hits the
    /// same backpressure path a stalled uploader would cause. The receiver is
    /// returned to the caller and held so the channel stays open (full), not
    /// closed. No S3 client or worker task is created.
    fn stalled_sink(capacity: usize) -> (S3RequestTraceSink, mpsc::Receiver<RequestTraceRecord>) {
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
