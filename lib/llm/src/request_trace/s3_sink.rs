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
//! Credentials come from the AWS SDK default provider chain — env vars, IMDS,
//! IRSA, Pod Identity, and shared profiles are all handled by the SDK. How the
//! frontend pod is credentialed is a deployment concern, not this sink's.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region, timeout::TimeoutConfig};
use aws_sdk_s3::primitives::ByteStream;
use flate2::{Compression, write::GzEncoder};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

use super::RequestTraceRecord;
use super::config::RequestTracePolicy;
use super::sink::RequestTraceSink;

const CHANNEL_CAPACITY: usize = 2048;
const DEFAULT_BUFFER_INITIAL_BYTES: usize = 256 * 1024;
// Bound S3 upload duration so a stalled endpoint or slow network cannot wedge
// the worker task indefinitely. `attempt_timeout` covers a single HTTP attempt;
// `operation_timeout` bounds the full call including SDK retries (three total
// by default). After the operation timeout expires the batch is discarded with
// a warning; a persistent retry queue is deferred to a follow-up PR.
const S3_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
const S3_OPERATION_TIMEOUT: Duration = Duration::from_secs(90);

pub struct S3RequestTraceSink {
    tx: mpsc::Sender<RequestTraceRecord>,
    shutdown: CancellationToken,
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
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

        let timeout_config = TimeoutConfig::builder()
            .operation_attempt_timeout(S3_ATTEMPT_TIMEOUT)
            .operation_timeout(S3_OPERATION_TIMEOUT)
            .build();
        let mut loader =
            aws_config::defaults(BehaviorVersion::latest()).timeout_config(timeout_config);
        if let Some(region) = policy.s3_region.clone() {
            loader = loader.region(Region::new(region));
        }
        let sdk_config = loader.load().await;
        let client = aws_sdk_s3::Client::new(&sdk_config);

        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
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
                client,
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
        })
    }
}

#[async_trait]
impl RequestTraceSink for S3RequestTraceSink {
    fn name(&self) -> &'static str {
        "s3"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        if let Err(error) = self.tx.try_send(record.clone()) {
            match error {
                mpsc::error::TrySendError::Full(_) => tracing::warn!(
                    target: "dynamo_llm::request_trace",
                    "request trace s3: batcher backpressure; dropping record"
                ),
                mpsc::error::TrySendError::Closed(_) => tracing::warn!(
                    target: "dynamo_llm::request_trace",
                    "request trace s3: batcher closed; dropping record"
                ),
            }
        }
    }

    async fn shutdown(&self) {
        self.shutdown.cancel();
        let worker = self
            .worker
            .lock()
            .expect("s3 sink shutdown mutex poisoned")
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
    }
}

async fn run_worker(
    client: aws_sdk_s3::Client,
    options: S3UploadOptions,
    mut rx: mpsc::Receiver<RequestTraceRecord>,
    shutdown: CancellationToken,
    roll_uncompressed_bytes: u64,
    flush_interval: Duration,
) {
    let uploader = Arc::new(S3Uploader { client, options });
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
        // The SDK exhausted its retries (three total attempts by default,
        // bounded by the operation timeout). The batch is dropped here rather
        // than requeued; a persistent retry buffer is a follow-up concern
        // tracked in the S3 layout PR.
        tracing::warn!(
            target: "dynamo_llm::request_trace",
            key = %key,
            batch_bytes,
            %error,
            "request trace s3: put_object failed after SDK retries; batch discarded"
        );
    }
}

struct S3Uploader {
    client: aws_sdk_s3::Client,
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
        self.client
            .put_object()
            .bucket(&self.options.bucket)
            .key(key)
            .content_type("application/gzip")
            .body(ByteStream::from(body))
            .send()
            .await
            .with_context(|| format!("s3 put_object into bucket {}", self.options.bucket))?;
        Ok(())
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
        let uploader = S3Uploader {
            client: dummy_client(),
            options: S3UploadOptions {
                bucket: "b".to_string(),
                prefix: "traces/".to_string(),
                host: "frontend-0".to_string(),
                run_id: "cafebabe".to_string(),
            },
        };
        let at = UNIX_EPOCH + Duration::from_secs(1_784_118_896);
        let key = uploader.object_key(at, 42);
        assert_eq!(
            key,
            "traces/2026/07/15/frontend-0-123456-cafebabe-000042.jsonl.gz"
        );
    }

    #[test]
    fn object_key_omits_leading_slash_when_prefix_empty() {
        let uploader = S3Uploader {
            client: dummy_client(),
            options: S3UploadOptions {
                bucket: "b".to_string(),
                prefix: String::new(),
                host: "h".to_string(),
                run_id: "abc".to_string(),
            },
        };
        let key = uploader.object_key(UNIX_EPOCH, 0);
        assert!(!key.starts_with('/'));
        assert!(key.starts_with("1970/01/01/h-"));
    }

    fn dummy_client() -> aws_sdk_s3::Client {
        // A no-network client. Any subsequent `send()` call would fail, but
        // pure-Rust key generation does not exercise the transport.
        let config = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .build();
        aws_sdk_s3::Client::from_conf(config)
    }
}
