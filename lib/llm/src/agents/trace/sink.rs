// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::{Context as _, anyhow};
use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::recorder::{Recorder, RecorderOptions};
use crate::telemetry::jsonl::JsonlSinkOptions;

use super::{AgentTracePolicy, AgentTraceRecord, config};

static WORKERS_STARTED: AtomicBool = AtomicBool::new(false);

#[async_trait]
pub trait TraceSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, rec: &AgentTraceRecord);
}

pub struct StderrTraceSink;

#[async_trait]
impl TraceSink for StderrTraceSink {
    fn name(&self) -> &'static str {
        "stderr"
    }

    async fn emit(&self, rec: &AgentTraceRecord) {
        match serde_json::to_string(rec) {
            Ok(js) => tracing::info!(
                target = "dynamo_llm::agents::trace",
                log_type = "agent_trace",
                record = %js,
                "agent_trace"
            ),
            Err(err) => tracing::warn!("agent trace: serialize failed: {err}"),
        }
    }
}

pub struct JsonlTraceSink {
    tx: mpsc::Sender<AgentTraceRecord>,
    recorder: Recorder<AgentTraceRecord>,
}

impl JsonlTraceSink {
    pub async fn new(path: String, options: JsonlSinkOptions) -> anyhow::Result<Self> {
        let recorder_shutdown = CancellationToken::new();
        let recorder = Recorder::new_with_options(
            recorder_shutdown,
            &path,
            RecorderOptions {
                buffer_bytes: options.buffer_bytes.max(1),
                flush_interval: Some(options.flush_interval.max(Duration::from_millis(1))),
                append: true,
                ..Default::default()
            },
        )
        .await
        .with_context(|| format!("opening jsonl agent trace sink at {path}"))?;
        let tx = recorder.event_sender();
        Ok(Self { tx, recorder })
    }

    async fn from_policy(policy: &AgentTracePolicy) -> anyhow::Result<Self> {
        let path = policy.jsonl_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes jsonl",
                dynamo_runtime::config::environment_names::llm::agent_trace::DYN_AGENT_TRACE_JSONL_PATH,
                dynamo_runtime::config::environment_names::llm::agent_trace::DYN_AGENT_TRACE_SINKS
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

impl Drop for JsonlTraceSink {
    fn drop(&mut self) {
        self.recorder.shutdown();
    }
}

#[async_trait]
impl TraceSink for JsonlTraceSink {
    fn name(&self) -> &'static str {
        "jsonl"
    }

    async fn emit(&self, rec: &AgentTraceRecord) {
        if self.tx.send(rec.clone()).await.is_err() {
            tracing::warn!("agent trace jsonl sink closed; dropping record");
        }
    }
}

async fn parse_sinks_from_env() -> anyhow::Result<Vec<Arc<dyn TraceSink>>> {
    let policy = config::policy();
    let mut out: Vec<Arc<dyn TraceSink>> = Vec::new();
    for name in &policy.sinks {
        match name.as_str() {
            "stderr" => out.push(Arc::new(StderrTraceSink)),
            "jsonl" => {
                let sink: Arc<dyn TraceSink> = Arc::new(JsonlTraceSink::from_policy(policy).await?);
                out.push(sink);
            }
            other => tracing::warn!(%other, "agent trace: unknown sink ignored"),
        }
    }
    Ok(out)
}

/// Spawn one worker per trace sink; each subscribes to the trace bus off the hot path.
pub async fn spawn_workers_from_env(shutdown: CancellationToken) -> anyhow::Result<()> {
    if WORKERS_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(());
    }

    if let Err(err) = spawn_workers(shutdown).await {
        WORKERS_STARTED.store(false, Ordering::Release);
        return Err(err);
    }

    Ok(())
}

async fn spawn_workers(shutdown: CancellationToken) -> anyhow::Result<()> {
    let sinks = parse_sinks_from_env().await?;
    let sink_count = sinks.len();
    for sink in sinks {
        let name = sink.name();
        let mut rx: broadcast::Receiver<AgentTraceRecord> = super::subscribe();
        let worker_shutdown = shutdown.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = worker_shutdown.cancelled() => {
                        loop {
                            match rx.try_recv() {
                                Ok(rec) => sink.emit(&rec).await,
                                Err(broadcast::error::TryRecvError::Lagged(n)) => tracing::warn!(
                                    sink = name,
                                    dropped = n,
                                    "agent trace bus lagged during shutdown; dropped records"
                                ),
                                Err(
                                    broadcast::error::TryRecvError::Empty
                                    | broadcast::error::TryRecvError::Closed
                                ) => break,
                            }
                        }
                        return;
                    }
                    msg = rx.recv() => {
                        match msg {
                            Ok(rec) => sink.emit(&rec).await,
                            Err(broadcast::error::RecvError::Lagged(n)) => tracing::warn!(
                                sink = name,
                                dropped = n,
                                "agent trace bus lagged; dropped records"
                            ),
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
        });
    }

    tracing::info!(sinks = sink_count, "Agent trace sinks ready");
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use crate::agents::context::AgentContext;

    use super::*;
    use crate::agents::trace::{
        AgentRequestMetrics, TraceEventSource, TraceEventType, TraceSchema,
    };

    fn sample_record() -> AgentTraceRecord {
        AgentTraceRecord {
            schema: TraceSchema::V1,
            event_type: TraceEventType::RequestEnd,
            event_time_unix_ms: 1000,
            event_source: TraceEventSource::Dynamo,
            agent_context: AgentContext {
                workflow_type_id: "ms_agent".to_string(),
                workflow_id: "run-1".to_string(),
                program_id: "run-1:agent".to_string(),
                parent_program_id: None,
            },
            request: Some(AgentRequestMetrics {
                request_id: "req-123".to_string(),
                x_request_id: Some("llm-call-1".to_string()),
                model: "test-model".to_string(),
                input_tokens: Some(42),
                output_tokens: Some(7),
                cached_tokens: Some(5),
                request_received_ms: Some(1000),
                prefill_wait_time_ms: None,
                prefill_time_ms: None,
                ttft_ms: None,
                total_time_ms: Some(25.0),
                kv_hit_rate: None,
                kv_transfer_estimated_latency_ms: None,
                queue_depth: None,
                worker: None,
            }),
        }
    }

    #[tokio::test]
    async fn jsonl_trace_sink_writes_request_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_trace.jsonl");
        let sink = JsonlTraceSink::new(
            path.display().to_string(),
            JsonlSinkOptions {
                buffer_bytes: 128,
                flush_interval: Duration::from_millis(10),
            },
        )
        .await
        .expect("sink should start");

        sink.emit(&sample_record()).await;

        let mut content = String::new();
        for _ in 0..100 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.contains("\"event_type\":\"request_end\"")
                && content.contains("\"request_id\":\"req-123\"")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(content.contains("\"event_type\":\"request_end\""));
        assert!(content.contains("\"request_id\":\"req-123\""));
        assert!(content.contains("\"workflow_id\":\"run-1\""));
    }

    #[tokio::test]
    async fn stderr_trace_sink_accepts_request_record() {
        StderrTraceSink.emit(&sample_record()).await;
    }
}
