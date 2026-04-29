// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context as _, anyhow};
use async_trait::async_trait;
use flate2::{Compression, write::GzEncoder};
use serde::Serialize;
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
        let path = policy.output_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes jsonl",
                dynamo_runtime::config::environment_names::llm::agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH,
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

#[derive(Debug, Clone)]
pub struct JsonlGzipSinkOptions {
    pub buffer_bytes: usize,
    pub flush_interval: Duration,
    pub roll_uncompressed_bytes: u64,
    pub roll_lines: Option<u64>,
}

pub struct JsonlGzipTraceSink {
    tx: mpsc::Sender<AgentTraceRecord>,
    shutdown: CancellationToken,
}

#[derive(Serialize)]
struct AgentTraceRecordEntry<'a> {
    timestamp: u64,
    event: &'a AgentTraceRecord,
}

struct GzipBatchWriter {
    base_path: PathBuf,
    current_index: u64,
    start_time: Instant,
    batch: Vec<u8>,
    segment_uncompressed_bytes: u64,
    segment_lines: u64,
    options: JsonlGzipSinkOptions,
}

impl GzipBatchWriter {
    fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let base_path = PathBuf::from(path);
        if let Some(parent) = base_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating agent trace directory {}", parent.display()))?;
        }

        let current_index = next_segment_index(&base_path)?;
        Ok(Self {
            base_path,
            current_index,
            start_time: Instant::now(),
            batch: Vec::with_capacity(options.buffer_bytes.max(1)),
            segment_uncompressed_bytes: 0,
            segment_lines: 0,
            options,
        })
    }

    async fn push(&mut self, rec: &AgentTraceRecord) -> anyhow::Result<()> {
        let entry = AgentTraceRecordEntry {
            timestamp: self.start_time.elapsed().as_millis() as u64,
            event: rec,
        };
        let mut line = serde_json::to_vec(&entry).context("serializing agent trace record")?;
        line.push(b'\n');

        let line_len = line.len() as u64;
        if self.should_roll_before(line_len) {
            self.flush_batch().await?;
            self.roll_segment();
        }

        self.batch.extend_from_slice(&line);
        self.segment_uncompressed_bytes = self.segment_uncompressed_bytes.saturating_add(line_len);
        self.segment_lines = self.segment_lines.saturating_add(1);

        if self.batch.len() >= self.options.buffer_bytes.max(1) {
            self.flush_batch().await?;
        }

        Ok(())
    }

    fn should_roll_before(&self, next_line_len: u64) -> bool {
        if self.segment_lines == 0 {
            return false;
        }

        if self
            .segment_uncompressed_bytes
            .saturating_add(next_line_len)
            > self.options.roll_uncompressed_bytes.max(1)
        {
            return true;
        }

        self.options
            .roll_lines
            .is_some_and(|limit| self.segment_lines >= limit)
    }

    fn roll_segment(&mut self) {
        self.current_index = self.current_index.saturating_add(1);
        self.segment_uncompressed_bytes = 0;
        self.segment_lines = 0;
    }

    async fn flush_batch(&mut self) -> anyhow::Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }

        let path = segment_path(&self.base_path, self.current_index);
        let batch = self.batch.clone();

        tokio::task::spawn_blocking(move || write_gzip_member(path, batch))
            .await
            .context("gzip agent trace writer task panicked")??;

        self.batch.clear();
        Ok(())
    }
}

impl JsonlGzipTraceSink {
    pub async fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let shutdown = CancellationToken::new();
        let (tx, rx) = mpsc::channel::<AgentTraceRecord>(2048);
        let mut writer = GzipBatchWriter::new(path.clone(), options)
            .with_context(|| format!("opening gzip jsonl agent trace sink at {path}"))?;
        let worker_shutdown = shutdown.clone();

        tokio::spawn(async move {
            run_gzip_writer(rx, &mut writer, worker_shutdown).await;
        });

        Ok(Self { tx, shutdown })
    }

    async fn from_policy(policy: &AgentTracePolicy) -> anyhow::Result<Self> {
        let path = policy.output_path.clone().ok_or_else(|| {
            anyhow!(
                "{} must be set when {} includes jsonl_gz",
                dynamo_runtime::config::environment_names::llm::agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH,
                dynamo_runtime::config::environment_names::llm::agent_trace::DYN_AGENT_TRACE_SINKS
            )
        })?;
        Self::new(
            path,
            JsonlGzipSinkOptions {
                buffer_bytes: policy.jsonl_buffer_bytes,
                flush_interval: Duration::from_millis(policy.jsonl_flush_interval_ms.max(1)),
                roll_uncompressed_bytes: policy.jsonl_gz_roll_bytes,
                roll_lines: policy.jsonl_gz_roll_lines,
            },
        )
        .await
    }
}

impl Drop for JsonlGzipTraceSink {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

#[async_trait]
impl TraceSink for JsonlGzipTraceSink {
    fn name(&self) -> &'static str {
        "jsonl_gz"
    }

    async fn emit(&self, rec: &AgentTraceRecord) {
        if self.tx.send(rec.clone()).await.is_err() {
            tracing::warn!("agent trace jsonl_gz sink closed; dropping record");
        }
    }
}

async fn run_gzip_writer(
    mut rx: mpsc::Receiver<AgentTraceRecord>,
    writer: &mut GzipBatchWriter,
    shutdown: CancellationToken,
) {
    let mut flush_tick =
        tokio::time::interval(writer.options.flush_interval.max(Duration::from_millis(1)));
    flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                while let Ok(rec) = rx.try_recv() {
                    if let Err(err) = writer.push(&rec).await {
                        tracing::warn!("agent trace jsonl_gz sink dropped record during shutdown: {err}");
                    }
                }
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("agent trace jsonl_gz sink failed final flush: {err}");
                }
                return;
            }
            _ = flush_tick.tick() => {
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("agent trace jsonl_gz sink failed flush: {err}");
                }
            }
            msg = rx.recv() => {
                match msg {
                    Some(rec) => {
                        if let Err(err) = writer.push(&rec).await {
                            tracing::warn!("agent trace jsonl_gz sink dropped record: {err}");
                        }
                    }
                    None => {
                        if let Err(err) = writer.flush_batch().await {
                            tracing::warn!("agent trace jsonl_gz sink failed final flush: {err}");
                        }
                        return;
                    }
                }
            }
        }
    }
}

fn write_gzip_member(path: PathBuf, batch: Vec<u8>) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating agent trace directory {}", parent.display()))?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening gzip agent trace segment {}", path.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = GzEncoder::new(writer, Compression::default());
    encoder
        .write_all(&batch)
        .with_context(|| format!("compressing gzip agent trace segment {}", path.display()))?;
    let mut writer = encoder
        .finish()
        .with_context(|| format!("finishing gzip agent trace segment {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing gzip agent trace segment {}", path.display()))?;
    Ok(())
}

fn segment_path(base_path: &Path, index: u64) -> PathBuf {
    let raw = base_path.to_string_lossy();
    let prefix = raw
        .strip_suffix(".jsonl.gz")
        .or_else(|| raw.strip_suffix(".jsonl"))
        .unwrap_or(&raw);
    PathBuf::from(format!("{prefix}.{index:06}.jsonl.gz"))
}

fn next_segment_index(base_path: &Path) -> anyhow::Result<u64> {
    for index in 0..u64::MAX {
        if !segment_path(base_path, index).try_exists()? {
            return Ok(index);
        }
    }
    anyhow::bail!(
        "no available gzip agent trace segment index for {}",
        base_path.display()
    )
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
            "jsonl_gz" => {
                let sink: Arc<dyn TraceSink> =
                    Arc::new(JsonlGzipTraceSink::from_policy(policy).await?);
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
    use std::io::Read;

    use flate2::read::MultiGzDecoder;
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
                avg_itl_ms: None,
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

    fn read_gzip_jsonl(path: &std::path::Path) -> String {
        let bytes = std::fs::read(path).expect("gzip trace segment should be readable");
        let mut decoder = MultiGzDecoder::new(bytes.as_slice());
        let mut content = String::new();
        decoder
            .read_to_string(&mut content)
            .expect("gzip trace segment should decompress");
        content
    }

    #[tokio::test]
    async fn jsonl_gz_trace_sink_appends_gzip_members() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_trace");
        let sink = JsonlGzipTraceSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
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
                if content.matches("\"event_type\":\"request_end\"").count() == 2 {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        assert_eq!(content.matches("\"event_type\":\"request_end\"").count(), 2);
        assert!(content.contains("\"workflow_id\":\"run-1\""));
    }

    #[tokio::test]
    async fn jsonl_gz_trace_sink_rolls_segments() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_trace");
        let sink = JsonlGzipTraceSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: Some(1),
            },
        )
        .await
        .expect("sink should start");

        sink.emit(&sample_record()).await;
        sink.emit(&sample_record()).await;

        let first = segment_path(&path, 0);
        let second = segment_path(&path, 1);
        let mut first_content = String::new();
        let mut second_content = String::new();
        for _ in 0..100 {
            if first.exists() && second.exists() {
                first_content = read_gzip_jsonl(&first);
                second_content = read_gzip_jsonl(&second);
                if first_content.contains("\"request_id\":\"req-123\"")
                    && second_content.contains("\"request_id\":\"req-123\"")
                {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        assert!(first_content.contains("\"event_type\":\"request_end\""));
        assert!(second_content.contains("\"event_type\":\"request_end\""));
    }
}
