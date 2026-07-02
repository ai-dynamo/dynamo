// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::{Context as _, anyhow};
use async_nats::jetstream;
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;
use dynamo_runtime::transports::nats;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::telemetry::jsonl::{JsonlSinkOptions, JsonlWriter};
use crate::telemetry::jsonl_gz::{JsonlGzipSinkOptions, JsonlGzipWriter};

use super::{
    RequestTraceDestination, RequestTraceFileCompression, RequestTraceFileFormat,
    RequestTracePolicy, RequestTraceRecord, config, otel_sink::OtelRequestTraceSink,
};

static WORKERS_STARTED: AtomicBool = AtomicBool::new(false);

#[async_trait]
pub trait RequestTraceSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, record: &RequestTraceRecord);
    async fn shutdown(&self) {}
}

pub struct StderrRequestTraceSink;

#[async_trait]
impl RequestTraceSink for StderrRequestTraceSink {
    fn name(&self) -> &'static str {
        "stderr"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        match serde_json::to_string(record) {
            Ok(json) => tracing::info!(
                target = "dynamo_llm::request_trace",
                log_type = "request_trace",
                record = %json,
                "request_trace"
            ),
            Err(error) => tracing::warn!("request trace serialization failed: {error}"),
        }
    }
}

pub struct NatsRequestTraceSink {
    js: jetstream::Context,
    subject: String,
}

impl NatsRequestTraceSink {
    async fn from_policy(policy: &RequestTracePolicy) -> anyhow::Result<Self> {
        let nats_client = nats::ClientOptions::default()
            .connect()
            .await
            .with_context(|| {
                format!(
                    "Attempting to connect NATS request trace sink from env var {}",
                    env_request_trace::DYN_REQUEST_TRACE_DESTINATIONS
                )
            })?;
        Ok(Self {
            js: nats_client.jetstream().clone(),
            subject: policy.nats_subject.clone(),
        })
    }
}

#[async_trait]
impl RequestTraceSink for NatsRequestTraceSink {
    fn name(&self) -> &'static str {
        "nats"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        match serde_json::to_vec(record) {
            Ok(bytes) => {
                if let Err(error) = self.js.publish(self.subject.clone(), bytes.into()).await {
                    tracing::warn!("request trace nats: publish failed: {error}");
                }
            }
            Err(error) => tracing::warn!("request trace nats: serialize failed: {error}"),
        }
    }
}

pub struct JsonlRequestTraceSink {
    writer: JsonlWriter<RequestTraceRecord>,
}

impl JsonlRequestTraceSink {
    pub async fn new(path: String, options: JsonlSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening jsonl request trace sink at {path}"))?;
        Ok(Self { writer })
    }

    async fn from_policy(policy: &RequestTracePolicy) -> anyhow::Result<Self> {
        let path = policy.file_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes file",
                env_request_trace::DYN_REQUEST_TRACE_FILE_PATH,
                env_request_trace::DYN_REQUEST_TRACE_DESTINATIONS
            )
        })?;
        Self::new(
            path,
            JsonlSinkOptions {
                buffer_bytes: policy.file_buffer_bytes,
                flush_interval: Duration::from_millis(policy.file_flush_interval_ms.max(1)),
            },
        )
        .await
    }
}

#[async_trait]
impl RequestTraceSink for JsonlRequestTraceSink {
    fn name(&self) -> &'static str {
        "file"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        if self.writer.send(record.clone()).await.is_err() {
            tracing::warn!("request trace file sink closed; dropping record");
        }
    }
}

pub struct JsonlGzipRequestTraceSink {
    writer: JsonlGzipWriter<RequestTraceRecord>,
}

impl JsonlGzipRequestTraceSink {
    pub async fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlGzipWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening gzip jsonl request trace sink at {path}"))?;
        Ok(Self { writer })
    }

    async fn from_policy(policy: &RequestTracePolicy) -> anyhow::Result<Self> {
        let path = policy.file_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes file",
                env_request_trace::DYN_REQUEST_TRACE_FILE_PATH,
                env_request_trace::DYN_REQUEST_TRACE_DESTINATIONS
            )
        })?;
        Self::new(
            path,
            JsonlGzipSinkOptions {
                buffer_bytes: policy.file_buffer_bytes,
                flush_interval: Duration::from_millis(policy.file_flush_interval_ms.max(1)),
                roll_uncompressed_bytes: policy.file_roll_bytes,
                roll_lines: policy.file_roll_lines,
                max_segments: None,
            },
        )
        .await
    }
}

#[async_trait]
impl RequestTraceSink for JsonlGzipRequestTraceSink {
    fn name(&self) -> &'static str {
        "file"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        if self.writer.send(record.clone()).await.is_err() {
            tracing::warn!("request trace file sink closed; dropping record");
        }
    }
}

async fn parse_destinations_from_env() -> anyhow::Result<Vec<Arc<dyn RequestTraceSink>>> {
    let policy = config::policy();
    let mut sinks: Vec<Arc<dyn RequestTraceSink>> = Vec::new();
    for destination in &policy.destinations {
        match destination {
            RequestTraceDestination::Stderr => sinks.push(Arc::new(StderrRequestTraceSink)),
            RequestTraceDestination::Nats => {
                sinks.push(Arc::new(NatsRequestTraceSink::from_policy(policy).await?))
            }
            RequestTraceDestination::Otel => {
                sinks.push(Arc::new(OtelRequestTraceSink::from_policy(policy).await?))
            }
            RequestTraceDestination::File => match policy.file_format {
                RequestTraceFileFormat::Jsonl => match policy.file_compression {
                    RequestTraceFileCompression::None => {
                        sinks.push(Arc::new(JsonlRequestTraceSink::from_policy(policy).await?))
                    }
                    RequestTraceFileCompression::Gzip => sinks.push(Arc::new(
                        JsonlGzipRequestTraceSink::from_policy(policy).await?,
                    )),
                },
            },
        }
    }
    Ok(sinks)
}

pub async fn spawn_workers_from_env(shutdown: CancellationToken) -> anyhow::Result<()> {
    if WORKERS_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(());
    }

    if let Err(error) = spawn_workers(shutdown).await {
        WORKERS_STARTED.store(false, Ordering::Release);
        return Err(error);
    }
    Ok(())
}

async fn spawn_workers(shutdown: CancellationToken) -> anyhow::Result<()> {
    let sinks = parse_destinations_from_env().await?;
    let sink_count = sinks.len();
    for sink in sinks {
        let name = sink.name();
        let mut receiver: broadcast::Receiver<RequestTraceRecord> = super::subscribe();
        let worker_shutdown = shutdown.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = worker_shutdown.cancelled() => {
                        loop {
                            match receiver.try_recv() {
                                Ok(record) => sink.emit(&record).await,
                                Err(broadcast::error::TryRecvError::Lagged(count)) => tracing::warn!(
                                    sink = name,
                                    dropped = count,
                                    "request trace bus lagged during shutdown; dropped records"
                                ),
                                Err(
                                    broadcast::error::TryRecvError::Empty
                                    | broadcast::error::TryRecvError::Closed
                                ) => break,
                            }
                        }
                        break;
                    }
                    message = receiver.recv() => {
                        match message {
                            Ok(record) => sink.emit(&record).await,
                            Err(broadcast::error::RecvError::Lagged(count)) => tracing::warn!(
                                sink = name,
                                dropped = count,
                                "request trace bus lagged; dropped records"
                            ),
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
            sink.shutdown().await;
        });
    }

    if sink_count == 0 {
        tracing::warn!(
            "request trace is enabled but no valid request trace destinations were configured"
        );
    }
    tracing::info!(
        destinations = sink_count,
        "Request trace destinations ready"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::MultiGzDecoder;
    use tempfile::tempdir;

    use crate::request_trace::RequestReplayMetrics;
    use crate::telemetry::jsonl_gz::segment_path;

    use super::*;
    use crate::request_trace::RequestTraceEventType;
    use crate::request_trace::RequestTraceMetrics;
    use crate::request_trace::RequestTraceSchema;

    fn sample_record() -> RequestTraceRecord {
        RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::RequestEnd,
            event_time_unix_ms: 1_100,
            event_source: None,
            agent_context: None,
            request: Some(RequestTraceMetrics {
                request_id: "req-123".to_string(),
                x_request_id: None,
                model: None,
                input_tokens: None,
                output_tokens: Some(7),
                cached_tokens: None,
                request_received_ms: Some(1_000),
                prefill_wait_time_ms: None,
                prefill_time_ms: None,
                ttft_ms: None,
                total_time_ms: None,
                avg_itl_ms: None,
                kv_hit_rate: None,
                kv_transfer_estimated_latency_ms: None,
                queue_depth: None,
                worker: None,
                replay: Some(RequestReplayMetrics {
                    trace_block_size: 2,
                    input_length: 3,
                    input_sequence_hashes: vec![11, 22],
                }),
                finish_reason_metadata: None,
            }),
            tool: None,
            payload: None,
        }
    }

    #[tokio::test]
    async fn jsonl_sink_writes_request_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace.jsonl");
        let sink = JsonlRequestTraceSink::new(
            path.display().to_string(),
            JsonlSinkOptions {
                buffer_bytes: 128,
                flush_interval: Duration::from_millis(10),
            },
        )
        .await
        .unwrap();

        sink.emit(&sample_record()).await;

        let mut content = String::new();
        for _ in 0..100 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.contains("\"request_id\":\"req-123\"") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(content.contains("\"schema\":\"dynamo.request.trace.v1\""));
        assert!(!content.contains("agent_context"));
        assert!(!content.contains("\"tool\""));
    }

    #[tokio::test]
    async fn gzip_sink_writes_and_rolls_request_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace");
        let sink = JsonlGzipRequestTraceSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: Some(1),
                max_segments: None,
            },
        )
        .await
        .unwrap();

        sink.emit(&sample_record()).await;
        sink.emit(&sample_record()).await;

        for index in 0..2 {
            let segment = segment_path(&path, index);
            let mut content = String::new();
            for _ in 0..100 {
                if segment.exists() {
                    let bytes = std::fs::read(&segment).unwrap();
                    let mut decoder = MultiGzDecoder::new(bytes.as_slice());
                    decoder.read_to_string(&mut content).unwrap();
                    if content.contains("\"request_id\":\"req-123\"") {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(content.contains("\"request_id\":\"req-123\""));
        }
    }
}
