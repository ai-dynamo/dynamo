// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3 destination for request trace records.
//!
//! Records are batched in-process as gzipped JSONL, and each finished batch is
//! uploaded as one object via `PutObject`. Object keys use Hive-style time
//! partitions
//! (`{prefix}/date=YYYY-MM-DD/hour=HH/{MMSS}-{seq}-{pod}-{run_id}.jsonl.gz`)
//! so Athena and Glue can prune by day and hour. Time is the only partition
//! column; every row carries its own `model` field, so consumers filter on
//! that column rather than on the key.
//!
//! Credentials come from the shared `object_store` S3 client — environment
//! variables, IMDS, IRSA, and Pod Identity are supported. Shared AWS config and
//! credential profiles are not loaded; how the frontend pod is credentialed is
//! a deployment concern, not this sink's.
//!
//! # Upload concurrency
//!
//! One worker task drains the record channel and batches records; finished
//! batches are uploaded on separate tasks, so a slow `PutObject` never stops
//! the worker from draining. Two semaphores bound the work: at most
//! `DYN_REQUEST_TRACE_S3_MAX_CONCURRENT_UPLOADS` `PutObject` calls are in
//! flight, and behind them a queue holds up to that many times
//! [`UPLOAD_QUEUE_DEPTH_MULTIPLIER`] finalized batches waiting their turn.
//!
//! Loss is bounded and always counted, never silent. If S3 stays slow long
//! enough to fill both the in-flight set and the queue, further batches are
//! discarded; if the record channel itself fills first, `emit` drops records.
//! Both paths increment one counter, warn on the first drop, and report the
//! running total at shutdown (the OpenTelemetry Rust `BatchLogProcessor`
//! drop-accounting pattern), so a degraded endpoint cannot flood the log.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use flate2::{Compression, write::GzEncoder};
use object_store::{
    Attribute, Attributes, ClientConfigKey, ObjectStore, RetryConfig,
    aws::{AmazonS3Builder, AmazonS3ConfigKey},
    path::Path as ObjectPath,
};
use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use dynamo_runtime::config::environment_names::kubernetes as env_kubernetes;
use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

use super::RequestTraceRecord;
use super::config::RequestTracePolicy;
use super::sink::RequestTraceSink;

const CHANNEL_CAPACITY: usize = 2048;
const DEFAULT_BUFFER_INITIAL_BYTES: usize = 256 * 1024;
// Bound S3 upload duration so a stalled endpoint or slow network cannot wedge
// the worker task indefinitely. `attempt_timeout` covers a single HTTP attempt;
// the outer timeout bounds the full call including retries (three total).
// After the operation timeout expires the batch is discarded with
// a warning; a persistent retry queue is deferred to a follow-up PR.
const S3_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
const S3_OPERATION_TIMEOUT: Duration = Duration::from_secs(90);
/// Ready batches queued behind the in-flight uploads, as a multiple of the
/// concurrency limit. Batches roll about once per flush interval per pod, so a
/// small queue rides out a multi-minute S3 slowdown before anything is dropped.
const UPLOAD_QUEUE_DEPTH_MULTIPLIER: usize = 2;

pub struct S3RequestTraceSink {
    tx: mpsc::Sender<RequestTraceRecord>,
    shutdown: CancellationToken,
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Records lost, from either path: `emit` finding the batcher channel full
    /// or closed, or the worker discarding a batch because the upload queue was
    /// saturated. Shared with the worker task. Surfaced once on the first drop
    /// and again as a summary at shutdown, so a slow S3 does not produce one
    /// log line per dropped record.
    dropped: Arc<AtomicU64>,
}

#[derive(Clone)]
struct S3UploadOptions {
    bucket: String,
    prefix: String,
    host: String,
    /// Per-process UUID mixed into every object key so that a pod restart
    /// (which reuses the pod name and can land within the same second) can't
    /// overwrite a previous batch, and two pods sharing a name stay disjoint.
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
        let prefix = resolve_prefix(policy.s3_prefix.as_deref());
        let host = pod_identity();
        let run_id = Uuid::new_v4().simple().to_string();
        let roll_uncompressed_bytes = policy.s3_roll_uncompressed_bytes;
        let flush_interval = Duration::from_millis(policy.s3_flush_interval_ms.max(1));
        let max_concurrent_uploads = policy.s3_max_concurrent_uploads.max(1);

        let store: Arc<dyn ObjectStore> = Arc::new(
            request_trace_s3_builder(&bucket, policy.s3_region.as_deref())
                .build()
                .context("building request trace S3 client")?,
        );

        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        let shutdown = CancellationToken::new();
        let upload_options = S3UploadOptions {
            bucket,
            prefix,
            host,
            run_id,
        };
        let worker_shutdown = shutdown.clone();
        let dropped = Arc::new(AtomicU64::new(0));
        let worker_dropped = dropped.clone();
        let worker = tokio::spawn(async move {
            run_worker(
                store,
                upload_options,
                rx,
                worker_shutdown,
                roll_uncompressed_bytes,
                flush_interval,
                max_concurrent_uploads,
                worker_dropped,
            )
            .await;
        });

        Ok(Self {
            tx,
            shutdown,
            worker: Mutex::new(Some(worker)),
            dropped,
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
                "request trace s3: records lost during the run (batcher backpressure or saturated upload queue)"
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_worker(
    store: Arc<dyn ObjectStore>,
    options: S3UploadOptions,
    mut rx: mpsc::Receiver<RequestTraceRecord>,
    shutdown: CancellationToken,
    roll_uncompressed_bytes: u64,
    flush_interval: Duration,
    max_concurrent_uploads: usize,
    dropped: Arc<AtomicU64>,
) {
    let uploader = Arc::new(S3Uploader::new(store, options));
    // Two tiers. `admission_slots` caps how much work is outstanding at all:
    // the uploads in flight plus the bounded queue of ready batches behind
    // them. Handing a batch to a spawned task means a slow `PutObject` never
    // stalls the drain loop below.
    let admission_slots = max_concurrent_uploads * (1 + UPLOAD_QUEUE_DEPTH_MULTIPLIER);
    let upload_slots = Arc::new(Semaphore::new(admission_slots));
    // `in_flight_slots` is what actually bounds concurrent `PutObject` calls to
    // `max_concurrent_uploads`. Without this second gate every admitted batch
    // would upload immediately and the extra admission permits would multiply
    // real concurrency instead of forming a queue.
    let in_flight_slots = Arc::new(Semaphore::new(max_concurrent_uploads));
    let mut batch = JsonlBatch::new();
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
                        upload_ready_batch(&uploader, &mut batch).await;
                    }
                }
                if !batch.is_empty() {
                    upload_ready_batch(&uploader, &mut batch).await;
                }
                drain_uploads(&upload_slots, admission_slots).await;
                return;
            }
            _ = flush_tick.tick() => {
                if !batch.is_empty() {
                    spawn_upload(&uploader, &mut batch, &upload_slots, &in_flight_slots, &dropped).await;
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
                            spawn_upload(&uploader, &mut batch, &upload_slots, &in_flight_slots, &dropped).await;
                        }
                    }
                    None => {
                        if !batch.is_empty() {
                            upload_ready_batch(&uploader, &mut batch).await;
                        }
                        drain_uploads(&upload_slots, admission_slots).await;
                        return;
                    }
                }
            }
        }
    }
}

/// Wait for every spawned upload to finish before the worker returns.
///
/// Acquiring the semaphore's *total* permit count is only possible once no
/// upload task holds one, which drains the in-flight set without tracking task
/// handles. The count must be the fixed total: `available_permits()` reports
/// only the currently free permits, so acquiring that many would succeed
/// immediately and abandon the in-flight uploads.
async fn drain_uploads(upload_slots: &Arc<Semaphore>, admission_slots: usize) {
    let _ = upload_slots.acquire_many(admission_slots as u32).await;
}

/// Finalize the batch and upload it on its own task, so a slow `PutObject`
/// never stalls the drain loop.
///
/// Admission is bounded in two tiers: `upload_slots` caps the total outstanding
/// work (uploads in flight plus the queue behind them), while `in_flight_slots`
/// caps how many spawned tasks may call `PutObject` at once. When every
/// admission permit is taken — S3 is degraded and the queue has filled — the
/// batch is dropped and counted rather than blocking ingestion, which is the
/// whole point of moving uploads off this task.
async fn spawn_upload(
    uploader: &Arc<S3Uploader>,
    batch: &mut JsonlBatch,
    upload_slots: &Arc<Semaphore>,
    in_flight_slots: &Arc<Semaphore>,
    dropped: &Arc<AtomicU64>,
) {
    let lines = batch.lines();
    let Ok(permit) = upload_slots.clone().try_acquire_owned() else {
        // Discard the batch so the buffer resets; keeping it would only grow
        // unboundedly while S3 stays unavailable.
        let discarded = batch.discard();
        let total = dropped.fetch_add(discarded, Ordering::Relaxed) + discarded;
        tracing::warn!(
            target: "dynamo_llm::request_trace",
            records = discarded,
            total_dropped = total,
            "request trace s3: upload queue full (S3 slow or unavailable); batch discarded"
        );
        return;
    };

    let ready = match batch.take_finished().await {
        Ok(bytes) => bytes,
        Err(error) => {
            let total = dropped.fetch_add(lines, Ordering::Relaxed) + lines;
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                %error,
                records = lines,
                total_dropped = total,
                "request trace s3: finalize gzip batch failed; discarding"
            );
            return;
        }
    };

    // Stamp the key from when the batch was flushed, not from when its upload
    // eventually starts. A queued batch can wait behind the in-flight limit for
    // an unbounded time while S3 is slow, and stamping later would file those
    // records under a `date=`/`hour=` partition after the one they belong to —
    // so a time-pruned Athena query over the correct partition would miss them.
    let flushed_at = SystemTime::now();
    let uploader = uploader.clone();
    let dropped = dropped.clone();
    let in_flight_slots = in_flight_slots.clone();
    tokio::spawn(async move {
        // Hold the admission permit for the whole upload; dropping it on
        // completion frees a slot for the next queued batch.
        let _permit = permit;
        // Queue here until an in-flight slot frees up, so at most
        // `max_concurrent_uploads` `PutObject` calls are ever outstanding. This
        // await is what turns the surplus admission permits into a real queue.
        let _in_flight = in_flight_slots.acquire().await;
        let key = uploader.object_key(flushed_at);
        let batch_bytes = ready.len();
        if let Err(error) = uploader.put_object(key.clone(), ready).await {
            let total = dropped.fetch_add(lines, Ordering::Relaxed) + lines;
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                key = %key,
                batch_bytes,
                records = lines,
                total_dropped = total,
                %error,
                "request trace s3: put_object failed after SDK retries; batch discarded"
            );
        }
    });
}

async fn upload_ready_batch(uploader: &Arc<S3Uploader>, batch: &mut JsonlBatch) {
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
    let key = uploader.object_key(SystemTime::now());
    let batch_bytes = ready.len();
    if let Err(error) = uploader.put_object(key.clone(), ready).await {
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

struct S3Uploader {
    store: Arc<dyn ObjectStore>,
    options: S3UploadOptions,
    /// Monotonic per-process batch counter, mixed into every key. `MMSS` is only
    /// second-granular, so a size roll under load or several rolls during
    /// shutdown can flush twice within one second; without this the second
    /// object would overwrite the first.
    sequence: AtomicU64,
}

impl S3Uploader {
    fn new(store: Arc<dyn ObjectStore>, options: S3UploadOptions) -> Self {
        Self {
            store,
            options,
            sequence: AtomicU64::new(0),
        }
    }

    fn object_key(&self, at: SystemTime) -> String {
        // Hive-style time partitions (`date=`/`hour=`) so Athena and Glue can
        // prune by day and hour instead of scanning the whole prefix. Time is
        // the only partition column: a batch can hold records for several
        // models, and every row already carries its own `model` field, so
        // consumers filter on that column rather than the key.
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
        // `MMSS` orders objects within the hour partition, `seq` separates two
        // batches that flush in the same second, `pod` identifies the writing
        // pod without opening the object, and the per-process `run_id` keeps two
        // pods (or a pod and its restart) that flush in the same second from
        // overwriting each other.
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        key.push_str(&format!(
            "date={yyyy:04}-{mm:02}-{dd:02}/hour={hh:02}/{mi:02}{ss:02}-{seq:04}-{pod}-{run_id}.jsonl.gz",
            pod = self.options.host,
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

    /// Records currently buffered, used to attribute drops to a record count.
    fn lines(&self) -> u64 {
        self.lines
    }

    /// Throw the buffered records away and reset, returning how many were lost.
    fn discard(&mut self) -> u64 {
        let lost = self.lines;
        self.raw.clear();
        self.lines = 0;
        lost
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

/// Resolve the leading key segment.
///
/// An explicitly configured `DYN_REQUEST_TRACE_S3_PREFIX` always wins: setting
/// it is a deliberate choice to control the bucket layout (nesting under
/// `org/team/env`, or reserving a prefix in a bucket shared with other data).
/// Otherwise the parent `DynamoGraphDeployment` is used, so each deployment
/// lands under its own prefix without any configuration. DGD names are only
/// unique within a namespace, so the namespace is included as
/// `{namespace}/{name}` to keep same-named deployments in different namespaces
/// from writing into one prefix. The operator injects
/// `DYN_PARENT_DGD_K8S_NAME` and `DYN_PARENT_DGD_K8S_NAMESPACE` on every
/// component container, so on Kubernetes both are available; the name alone is
/// used if the namespace is missing, and off-cluster runs with neither set write
/// to the bucket root.
fn resolve_prefix(configured: Option<&str>) -> String {
    if let Some(prefix) = configured.map(str::trim).filter(|value| !value.is_empty()) {
        return prefix.to_string();
    }
    let Some(name) = env_non_empty(env_kubernetes::DYN_PARENT_DGD_K8S_NAME) else {
        return String::new();
    };
    match env_non_empty(env_kubernetes::DYN_PARENT_DGD_K8S_NAMESPACE) {
        Some(namespace) => format!("{namespace}/{name}"),
        None => name,
    }
}

/// Identify the writing pod, following the same precedence the runtime's
/// Kubernetes discovery uses (`POD_NAME` first; see
/// `runtime::discovery::kube::utils`). The operator injects `POD_NAME` from
/// `metadata.name`, so it is the authoritative source on-cluster; the
/// remaining steps cover non-operator and bare-metal runs.
fn pod_identity() -> String {
    env_non_empty(env_kubernetes::POD_NAME)
        .or_else(|| env_non_empty("HOSTNAME"))
        .or_else(|| {
            std::fs::read_to_string("/etc/hostname")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
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
    fn object_key_includes_prefix_partitions_time_pod_and_run_id() {
        let uploader = test_uploader("traces/");
        let at = UNIX_EPOCH + Duration::from_secs(1_784_118_896);
        let key = uploader.object_key(at);
        assert_eq!(
            key,
            "traces/date=2026-07-15/hour=12/3456-0000-frontend-0-cafebabe.jsonl.gz"
        );
    }

    #[test]
    fn object_keys_stay_distinct_within_the_same_second() {
        // A size roll under load, or several rolls during shutdown, can flush
        // twice inside one wall-clock second. `MMSS` alone would collide and the
        // second upload would overwrite the first, so the sequence must advance.
        let uploader = test_uploader("");
        let at = UNIX_EPOCH + Duration::from_secs(1_784_118_896);
        let keys: Vec<String> = (0..3).map(|_| uploader.object_key(at)).collect();
        assert!(keys[0].contains("3456-0000-"), "{}", keys[0]);
        assert!(keys[1].contains("3456-0001-"), "{}", keys[1]);
        assert!(keys[2].contains("3456-0002-"), "{}", keys[2]);
        let unique: std::collections::HashSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), 3, "keys collided: {keys:?}");
    }

    #[test]
    fn object_key_omits_leading_slash_when_prefix_empty() {
        let uploader = test_uploader("");
        let key = uploader.object_key(UNIX_EPOCH);
        assert!(!key.starts_with('/'));
        assert!(key.starts_with("date=1970-01-01/hour=00/0000-0000-frontend-0-"));
    }

    #[test]
    fn object_key_partitions_by_the_supplied_flush_time() {
        // A queued batch can wait behind the in-flight limit for an unbounded
        // time while S3 is slow. The key must come from when the batch was
        // flushed, not from when its upload finally starts, or the records land
        // in a later `date=`/`hour=` partition than they belong to and a
        // time-pruned query over the correct partition misses them.
        let uploader = test_uploader("");
        // 2026-07-15 12:34:56 UTC, one second before the hour partition rolls
        // would be misleading, so use a moment whose hour is unambiguous.
        let flushed_at = UNIX_EPOCH + Duration::from_secs(1_784_118_896);
        // An upload that starts an hour later must still be keyed by `flushed_at`.
        let started_later = flushed_at + Duration::from_secs(3600);
        assert!(uploader.object_key(flushed_at).contains("hour=12/3456-"));
        assert!(uploader.object_key(started_later).contains("hour=13/3456-"));
    }

    #[tokio::test]
    async fn drain_uploads_waits_for_an_in_flight_upload_to_finish() {
        // Regression test: draining must acquire the semaphore's *total* permit
        // count. Using `available_permits()` returns the free count, so the
        // acquire would succeed instantly and abandon in-flight uploads.
        let admission_slots = 4;
        let slots = Arc::new(Semaphore::new(admission_slots));
        let held = slots.clone().acquire_owned().await.unwrap();

        // With one permit held, the drain must not complete.
        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                drain_uploads(&slots, admission_slots)
            )
            .await
            .is_err(),
            "drain returned while an upload still held a permit"
        );

        // Releasing it (as a finished upload does) lets the drain complete.
        drop(held);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                drain_uploads(&slots, admission_slots)
            )
            .await
            .is_ok(),
            "drain did not complete after every permit was released"
        );
    }

    #[tokio::test]
    async fn admitted_batches_queue_behind_the_in_flight_limit() {
        // The admission semaphore is deliberately larger than the in-flight
        // limit, so admission alone must not permit an upload to start;
        // otherwise the configured concurrency would be exceeded.
        let max_concurrent_uploads = 1;
        let admission_slots = max_concurrent_uploads * (1 + UPLOAD_QUEUE_DEPTH_MULTIPLIER);
        assert!(admission_slots > max_concurrent_uploads);

        let in_flight = Arc::new(Semaphore::new(max_concurrent_uploads));
        let busy = in_flight.clone().acquire_owned().await.unwrap();
        // Every in-flight slot is taken, so a newly admitted batch waits here
        // rather than issuing a concurrent `PutObject`.
        assert!(
            in_flight.clone().try_acquire_owned().is_err(),
            "an extra upload started while the in-flight limit was saturated"
        );
        drop(busy);
        assert!(in_flight.try_acquire_owned().is_ok());
    }

    #[tokio::test]
    async fn put_object_writes_body_and_content_type() {
        let store = Arc::new(InMemory::new());
        let uploader = S3Uploader::new(store.clone(), test_options(""));
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
        let uploader = S3Uploader::new(store.clone(), test_options(""));
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

    fn test_uploader(prefix: &str) -> S3Uploader {
        S3Uploader::new(Arc::new(InMemory::new()), test_options(prefix))
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
    #[serial_test::serial]
    fn prefix_falls_back_to_the_parent_dgd_name() {
        temp_env::with_vars(
            [(env_kubernetes::DYN_PARENT_DGD_K8S_NAME, Some("my-dgd"))],
            || {
                assert_eq!(resolve_prefix(None), "my-dgd");
                assert_eq!(resolve_prefix(Some("   ")), "my-dgd");
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn prefix_qualifies_the_dgd_name_with_its_namespace() {
        // DGD names are unique only within a namespace, so two same-named
        // deployments in different namespaces must not share one prefix.
        temp_env::with_vars(
            [
                (env_kubernetes::DYN_PARENT_DGD_K8S_NAME, Some("my-dgd")),
                (env_kubernetes::DYN_PARENT_DGD_K8S_NAMESPACE, Some("team-a")),
            ],
            || assert_eq!(resolve_prefix(None), "team-a/my-dgd"),
        );
        temp_env::with_vars(
            [
                (env_kubernetes::DYN_PARENT_DGD_K8S_NAME, Some("my-dgd")),
                (env_kubernetes::DYN_PARENT_DGD_K8S_NAMESPACE, Some("team-b")),
            ],
            || assert_eq!(resolve_prefix(None), "team-b/my-dgd"),
        );
    }

    #[test]
    #[serial_test::serial]
    fn explicit_prefix_wins_over_the_parent_dgd_name() {
        temp_env::with_vars(
            [(env_kubernetes::DYN_PARENT_DGD_K8S_NAME, Some("my-dgd"))],
            || assert_eq!(resolve_prefix(Some("traces/prod")), "traces/prod"),
        );
    }

    #[test]
    #[serial_test::serial]
    fn prefix_is_empty_off_cluster_without_configuration() {
        temp_env::with_vars(
            [(env_kubernetes::DYN_PARENT_DGD_K8S_NAME, None::<&str>)],
            || assert_eq!(resolve_prefix(None), ""),
        );
    }

    #[test]
    #[serial_test::serial]
    fn pod_identity_prefers_pod_name_over_hostname() {
        temp_env::with_vars(
            [
                (env_kubernetes::POD_NAME, Some("frontend-abc123")),
                ("HOSTNAME", Some("some-host")),
            ],
            || assert_eq!(pod_identity(), "frontend-abc123"),
        );
    }

    #[test]
    #[serial_test::serial]
    fn pod_identity_falls_back_to_hostname() {
        temp_env::with_vars(
            [
                (env_kubernetes::POD_NAME, None::<&str>),
                ("HOSTNAME", Some("some-host")),
            ],
            || assert_eq!(pod_identity(), "some-host"),
        );
    }

    #[tokio::test]
    async fn spawn_upload_discards_and_counts_when_queue_is_saturated() {
        let uploader = Arc::new(test_uploader(""));
        // A semaphore with no permits available stands in for "every upload
        // slot and every queue slot is taken".
        let slots = Arc::new(Semaphore::new(1));
        let held = slots.clone().acquire_owned().await.unwrap();
        let in_flight = Arc::new(Semaphore::new(1));
        let dropped = Arc::new(AtomicU64::new(0));

        let mut batch = JsonlBatch::new();
        for _ in 0..3 {
            batch.push(&sample_record()).unwrap();
        }
        assert_eq!(batch.lines(), 3);

        spawn_upload(&uploader, &mut batch, &slots, &in_flight, &dropped).await;

        // The batch is discarded, its records counted, and the buffer reset so
        // it does not grow while S3 stays unavailable.
        assert_eq!(dropped.load(Ordering::Relaxed), 3);
        assert!(batch.is_empty());
        assert_eq!(batch.uncompressed_bytes(), 0);
        drop(held);
    }

    #[tokio::test]
    async fn spawn_upload_takes_a_slot_when_capacity_is_available() {
        let uploader = Arc::new(test_uploader(""));
        let slots = Arc::new(Semaphore::new(2));
        let in_flight = Arc::new(Semaphore::new(1));
        let dropped = Arc::new(AtomicU64::new(0));

        let mut batch = JsonlBatch::new();
        batch.push(&sample_record()).unwrap();
        spawn_upload(&uploader, &mut batch, &slots, &in_flight, &dropped).await;

        // The batch was handed off (buffer reset) rather than discarded. The
        // spawned upload fails against the no-network client, which counts the
        // records, so only assert the hand-off drained the batch here.
        assert!(batch.is_empty());
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
