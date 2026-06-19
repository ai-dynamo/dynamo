// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, anyhow};
use async_nats::jetstream;
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::llm::audit as env_audit;
use dynamo_runtime::transports::nats;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::telemetry::jsonl::{JsonlSinkOptions, JsonlWriter};
use crate::telemetry::jsonl_gz::{JsonlGzipSinkOptions, JsonlGzipWriter, SegmentSink};
use crate::telemetry::s3_segment_sink::{
    S3SegmentIdentity, S3SegmentSink, S3SegmentSinkConfig, S3SinkEncryption,
};

use super::{
    bus,
    config::{self, AuditPolicy},
    handle::AuditRecord,
};

#[async_trait]
pub trait AuditSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, rec: &AuditRecord);
}

pub struct StderrSink;
#[async_trait]
impl AuditSink for StderrSink {
    fn name(&self) -> &'static str {
        "stderr"
    }
    async fn emit(&self, rec: &AuditRecord) {
        match serde_json::to_string(rec) {
            Ok(js) => {
                tracing::info!(target="dynamo_llm::audit", log_type="audit", record=%js, "audit")
            }
            Err(e) => tracing::warn!("audit: serialize failed: {e}"),
        }
    }
}

pub struct NatsSink {
    js: jetstream::Context,
    subject: String,
}

impl NatsSink {
    pub fn new(nats_client: dynamo_runtime::transports::nats::Client) -> Self {
        let subject = std::env::var(env_audit::DYN_AUDIT_NATS_SUBJECT)
            .unwrap_or_else(|_| "dynamo.audit.v1".to_string());
        Self {
            js: nats_client.jetstream().clone(),
            subject,
        }
    }
}

#[async_trait]
impl AuditSink for NatsSink {
    fn name(&self) -> &'static str {
        "nats"
    }

    async fn emit(&self, rec: &AuditRecord) {
        match serde_json::to_vec(rec) {
            Ok(bytes) => {
                if let Err(e) = self.js.publish(self.subject.clone(), bytes.into()).await {
                    tracing::warn!("nats: publish failed: {e}");
                }
            }
            Err(e) => tracing::warn!("nats: serialize failed: {e}"),
        }
    }
}

pub struct JsonlAuditSink {
    writer: JsonlWriter<AuditRecord>,
}

impl JsonlAuditSink {
    pub async fn new(path: String, options: JsonlSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening jsonl audit sink at {path}"))?;
        Ok(Self { writer })
    }

    async fn from_policy(policy: &AuditPolicy) -> anyhow::Result<Self> {
        let path = policy.output_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes jsonl",
                env_audit::DYN_AUDIT_OUTPUT_PATH,
                env_audit::DYN_AUDIT_SINKS
            )
        })?;
        Self::new(
            path,
            JsonlSinkOptions {
                buffer_bytes: policy.jsonl_buffer_bytes,
                flush_interval: Duration::from_millis(policy.jsonl_flush_interval_ms.max(1)),
            },
        )
        .await
    }
}

#[async_trait]
impl AuditSink for JsonlAuditSink {
    fn name(&self) -> &'static str {
        "jsonl"
    }

    async fn emit(&self, rec: &AuditRecord) {
        if self.writer.send(rec.clone()).await.is_err() {
            tracing::warn!("audit jsonl sink closed; dropping record");
        }
    }
}

pub struct JsonlGzipAuditSink {
    writer: JsonlGzipWriter<AuditRecord>,
}

impl JsonlGzipAuditSink {
    pub async fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlGzipWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening gzip jsonl audit sink at {path}"))?;
        Ok(Self { writer })
    }

    async fn from_policy(policy: &AuditPolicy) -> anyhow::Result<Self> {
        let path = policy.output_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes jsonl_gz",
                env_audit::DYN_AUDIT_OUTPUT_PATH,
                env_audit::DYN_AUDIT_SINKS
            )
        })?;
        Self::new(
            path,
            JsonlGzipSinkOptions {
                buffer_bytes: policy.jsonl_buffer_bytes,
                flush_interval: Duration::from_millis(policy.jsonl_flush_interval_ms.max(1)),
                roll_uncompressed_bytes: policy.jsonl_gz_roll_bytes,
                roll_lines: policy.jsonl_gz_roll_lines,
                roll_interval: None,
                channel_capacity: 2048,
            },
        )
        .await
    }
}

#[async_trait]
impl AuditSink for JsonlGzipAuditSink {
    fn name(&self) -> &'static str {
        "jsonl_gz"
    }

    async fn emit(&self, rec: &AuditRecord) {
        if self.writer.send(rec.clone()).await.is_err() {
            tracing::warn!("audit jsonl_gz sink closed; dropping record");
        }
    }
}

/// S3-backed audit sink. Buffers records into rotated, gzip-compressed
/// segments in memory and uploads each completed segment as one S3
/// object. Backed by the shared `JsonlGzipWriter` rotation engine — the
/// only thing different from `JsonlGzipAuditSink` is the destination.
pub struct S3AuditSink {
    writer: JsonlGzipWriter<AuditRecord>,
}

impl S3AuditSink {
    async fn from_policy(policy: &AuditPolicy) -> anyhow::Result<Self> {
        let bucket = policy.s3_bucket.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes s3",
                env_audit::DYN_AUDIT_S3_BUCKET,
                env_audit::DYN_AUDIT_SINKS
            )
        })?;

        let identity =
            S3SegmentIdentity::resolve(policy.s3_instance_id.clone(), policy.deployment.clone());
        let encryption = S3SinkEncryption {
            sse: policy.s3_sse.clone(),
            kms_key_id: policy.s3_kms_key_id.clone(),
        };
        let client =
            S3SegmentSink::build_client(policy.s3_region.clone(), policy.s3_endpoint_url.clone())
                .await
                .context("audit s3: building AWS SDK client")?;
        let segment_sink: Arc<dyn SegmentSink> = Arc::new(S3SegmentSink::new(
            client,
            S3SegmentSinkConfig {
                bucket,
                prefix: policy.s3_prefix.clone(),
                identity,
                encryption,
            },
        ));

        let writer = JsonlGzipWriter::with_segment_sink(
            segment_sink,
            JsonlGzipSinkOptions {
                buffer_bytes: policy.s3_batch_bytes,
                flush_interval: Duration::from_millis(policy.jsonl_flush_interval_ms.max(1)),
                roll_uncompressed_bytes: policy.s3_roll_bytes,
                roll_lines: policy.s3_roll_lines,
                roll_interval: Some(Duration::from_millis(policy.s3_roll_interval_ms.max(1))),
                channel_capacity: policy.s3_channel_capacity,
            },
        )
        .context("audit s3: starting rotation engine")?;
        Ok(Self { writer })
    }
}

#[async_trait]
impl AuditSink for S3AuditSink {
    fn name(&self) -> &'static str {
        "s3"
    }

    async fn emit(&self, rec: &AuditRecord) {
        if self.writer.send(rec.clone()).await.is_err() {
            tracing::warn!("audit s3 sink closed; dropping record");
        }
    }
}

async fn parse_sinks_from_env() -> anyhow::Result<Vec<Arc<dyn AuditSink>>> {
    let policy = config::policy();
    let mut out: Vec<Arc<dyn AuditSink>> = Vec::new();
    for name in &policy.sinks {
        match name.as_str() {
            "stderr" => out.push(Arc::new(StderrSink)),
            "nats" => {
                let nats_client = nats::ClientOptions::default()
                    .connect()
                    .await
                    .with_context(|| {
                        format!(
                            "Attempting to connect NATS sink from env var {}",
                            env_audit::DYN_AUDIT_SINKS
                        )
                    })?;
                out.push(Arc::new(NatsSink::new(nats_client)));
            }
            "jsonl" => {
                let sink: Arc<dyn AuditSink> = Arc::new(JsonlAuditSink::from_policy(policy).await?);
                out.push(sink);
            }
            "jsonl_gz" => {
                let sink: Arc<dyn AuditSink> =
                    Arc::new(JsonlGzipAuditSink::from_policy(policy).await?);
                out.push(sink);
            }
            "s3" => {
                let sink: Arc<dyn AuditSink> = Arc::new(S3AuditSink::from_policy(policy).await?);
                out.push(sink);
            }
            other => tracing::warn!(%other, "audit: unknown sink ignored"),
        }
    }
    Ok(out)
}

/// Spawn one worker per sink; each subscribes to the bus (off the hot path).
/// Workers drain remaining records and exit when `shutdown` is cancelled.
pub async fn spawn_workers_from_env(shutdown: CancellationToken) -> anyhow::Result<()> {
    let sinks = parse_sinks_from_env().await?;
    let sink_count = sinks.len();
    for sink in sinks {
        let name = sink.name();
        let mut rx: broadcast::Receiver<AuditRecord> = bus::subscribe();
        let worker_shutdown = shutdown.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = worker_shutdown.cancelled() => {
                        loop {
                            match rx.try_recv() {
                                Ok(rec) => sink.emit(&rec).await,
                                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                                    super::metrics::AUDIT_RECORDS_DROPPED_TOTAL
                                        .with_label_values(&["bus_lag"])
                                        .inc_by(n);
                                    tracing::warn!(
                                        sink = name,
                                        dropped = n,
                                        "audit bus lagged during shutdown; dropped records"
                                    );
                                }
                                Err(
                                    broadcast::error::TryRecvError::Empty
                                    | broadcast::error::TryRecvError::Closed,
                                ) => break,
                            }
                        }
                        return;
                    }
                    msg = rx.recv() => {
                        match msg {
                            Ok(rec) => sink.emit(&rec).await,
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                super::metrics::AUDIT_RECORDS_DROPPED_TOTAL
                                    .with_label_values(&["bus_lag"])
                                    .inc_by(n);
                                tracing::warn!(
                                    sink = name,
                                    dropped = n,
                                    "audit bus lagged; dropped records"
                                );
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
        });
    }
    tracing::info!(sinks = sink_count, "Audit sinks ready");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::MultiGzDecoder;
    use tempfile::tempdir;

    use crate::telemetry::jsonl_gz::segment_path;

    use super::*;

    fn sample_record() -> AuditRecord {
        AuditRecord {
            schema_version: 1,
            request_id: "req-abc".to_string(),
            requested_streaming: false,
            model: "test-model".to_string(),
            deployment: None,
            emitted_at_unix_ms: 0,
            request: None,
            response: None,
        }
    }

    fn read_gzip_jsonl(path: &std::path::Path) -> String {
        let bytes = std::fs::read(path).expect("gzip audit segment readable");
        let mut decoder = MultiGzDecoder::new(bytes.as_slice());
        let mut content = String::new();
        decoder
            .read_to_string(&mut content)
            .expect("gzip audit segment decompresses");
        content
    }

    #[tokio::test]
    async fn jsonl_gz_audit_sink_appends_gzip_members() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("audit");
        let sink = JsonlGzipAuditSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                roll_interval: None,
                channel_capacity: 2048,
            },
        )
        .await
        .expect("sink should start");

        sink.emit(&sample_record()).await;
        sink.emit(&sample_record()).await;

        let segment = segment_path(&path, 0);
        let mut content = String::new();
        for _ in 0..100 {
            if segment.exists() {
                content = read_gzip_jsonl(&segment);
                if content.matches("\"request_id\":\"req-abc\"").count() == 2 {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(content.matches("\"request_id\":\"req-abc\"").count(), 2);
    }
}
