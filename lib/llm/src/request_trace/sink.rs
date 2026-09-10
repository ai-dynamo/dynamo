// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::sync::{
    Arc, Mutex,
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
    RequestTraceFileFormat, RequestTracePolicy, RequestTraceRecord, RequestTraceSinkKind, config,
    otel_sink::OtelRequestTraceSink,
};

static WORKERS_STARTED: AtomicBool = AtomicBool::new(false);
static WORKERS: Mutex<Option<SinkWorkers>> = Mutex::new(None);

/// Upper bound on how long process teardown waits for the sink workers to
/// drain. Chosen to fit inside a default Kubernetes
/// `terminationGracePeriodSeconds` of 30 with room for the rest of teardown, so
/// a wedged sink endpoint cannot turn a rollout into a `SIGKILL`.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

#[async_trait]
pub trait RequestTraceSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, record: &RequestTraceRecord);
    async fn shutdown(&self) {}
    /// Records this sink dropped. Read by the shutdown joiner so counts are
    /// still reported when a sink does not finish draining in time.
    fn dropped_records(&self) -> u64 {
        0
    }
}

pub struct StderrRequestTraceSink;

#[async_trait]
impl RequestTraceSink for StderrRequestTraceSink {
    fn name(&self) -> &'static str {
        "stderr"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        match serde_json::to_string(record) {
            Ok(json) => {
                if let Err(error) = writeln!(std::io::stderr(), "{json}") {
                    tracing::warn!(%error, "request trace stderr write failed");
                }
            }
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
                    env_request_trace::DYN_REQUEST_TRACE_SINKS
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
                env_request_trace::DYN_REQUEST_TRACE_SINKS
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
                env_request_trace::DYN_REQUEST_TRACE_SINKS
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

async fn parse_sinks_from_env() -> anyhow::Result<Vec<Arc<dyn RequestTraceSink>>> {
    let policy = config::policy();
    let mut sinks: Vec<Arc<dyn RequestTraceSink>> = Vec::new();
    for sink_kind in &policy.sinks {
        match sink_kind {
            RequestTraceSinkKind::Stderr => sinks.push(Arc::new(StderrRequestTraceSink)),
            RequestTraceSinkKind::Nats => {
                sinks.push(Arc::new(NatsRequestTraceSink::from_policy(policy).await?))
            }
            RequestTraceSinkKind::Otel => {
                sinks.push(Arc::new(OtelRequestTraceSink::from_policy(policy).await?))
            }
            RequestTraceSinkKind::File => match policy.file_format {
                RequestTraceFileFormat::Jsonl => {
                    sinks.push(Arc::new(JsonlRequestTraceSink::from_policy(policy).await?))
                }
                RequestTraceFileFormat::JsonlGz => sinks.push(Arc::new(
                    JsonlGzipRequestTraceSink::from_policy(policy).await?,
                )),
            },
            RequestTraceSinkKind::S3 => {
                #[cfg(feature = "request-trace-s3")]
                {
                    use super::s3_sink::S3RequestTraceSink;
                    sinks.push(Arc::new(S3RequestTraceSink::from_policy(policy).await?));
                }
                #[cfg(not(feature = "request-trace-s3"))]
                {
                    return Err(anyhow!(
                        "request trace s3 sink requested but dynamo-llm was built without the \"request-trace-s3\" feature",
                    ));
                }
            }
        }
    }
    Ok(sinks)
}

/// The sink workers, retained so that teardown can wait for them.
pub struct SinkWorkers {
    /// Cancelled by [`SinkWorkers::shutdown`]. A child of the token passed to
    /// [`spawn_workers`], so a runtime-wide cancellation still stops the
    /// workers, but teardown does not have to wait for one to arrive.
    token: CancellationToken,
    handles: Vec<tokio::task::JoinHandle<()>>,
    /// Same order as `handles`, so a handle that is still running can be paired
    /// with the sink it belongs to.
    sinks: Vec<Arc<dyn RequestTraceSink>>,
}

/// What the bounded shutdown observed.
#[derive(Debug)]
pub struct TraceShutdownReport {
    pub timed_out: bool,
    /// (sink name, records dropped so far) for sinks that did not finish draining.
    pub pending: Vec<(&'static str, u64)>,
}

impl SinkWorkers {
    /// Cancel the workers and wait for them to finish draining, giving up after
    /// `timeout`. On timeout the sink tasks are abandoned and the process is
    /// about to exit, so no sink can report for itself; the counts are read
    /// here instead and both logged and returned.
    pub async fn shutdown(self, timeout: Duration) -> TraceShutdownReport {
        self.token.cancel();
        let Self {
            mut handles, sinks, ..
        } = self;

        if tokio::time::timeout(timeout, futures::future::join_all(handles.iter_mut()))
            .await
            .is_ok()
        {
            return TraceShutdownReport {
                timed_out: false,
                pending: Vec::new(),
            };
        }

        let pending: Vec<(&'static str, u64)> = handles
            .iter()
            .zip(sinks.iter())
            .filter(|(handle, _)| !handle.is_finished())
            .map(|(_, sink)| (sink.name(), sink.dropped_records()))
            .collect();
        tracing::warn!(
            timeout_ms = timeout.as_millis() as u64,
            pending = ?pending,
            "request trace sinks did not finish draining before the shutdown timeout"
        );
        TraceShutdownReport {
            timed_out: true,
            pending,
        }
    }
}

pub async fn spawn_workers_from_env(shutdown: CancellationToken) -> anyhow::Result<()> {
    if WORKERS_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(());
    }

    let sinks = match parse_sinks_from_env().await {
        Ok(sinks) => sinks,
        Err(error) => {
            WORKERS_STARTED.store(false, Ordering::Release);
            return Err(error);
        }
    };
    *WORKERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(spawn_workers(sinks, shutdown));
    Ok(())
}

/// Cancel the retained workers and wait for them to drain, bounded by
/// [`SHUTDOWN_TIMEOUT`]. Returns `None` when no workers were started, which is
/// the case whenever request tracing is disabled.
pub async fn shutdown_workers() -> Option<TraceShutdownReport> {
    let workers = WORKERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()?;
    Some(workers.shutdown(SHUTDOWN_TIMEOUT).await)
}

fn spawn_workers(
    sinks: Vec<Arc<dyn RequestTraceSink>>,
    shutdown: CancellationToken,
) -> SinkWorkers {
    let sink_count = sinks.len();
    let token = shutdown.child_token();
    let mut handles = Vec::with_capacity(sink_count);
    for sink in &sinks {
        let sink = sink.clone();
        let name = sink.name();
        let mut receiver: broadcast::Receiver<RequestTraceRecord> = super::subscribe();
        let worker_shutdown = token.clone();
        handles.push(tokio::spawn(async move {
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
        }));
    }

    if sink_count == 0 {
        tracing::warn!("request trace is enabled but no valid request trace sinks were configured");
    }
    tracing::info!(sinks = sink_count, "Request trace sinks ready");
    SinkWorkers {
        token,
        handles,
        sinks,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::sync::atomic::AtomicUsize;

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

    /// Sink whose teardown is observable from the outside: `shutdown` only sets
    /// `shutdown_done` after an await point, so a caller that does not wait for
    /// the worker sees `false`.
    struct FakeSink {
        emitted: Arc<AtomicUsize>,
        shutdown_done: Arc<AtomicBool>,
        dropped: u64,
        /// When set, `shutdown` never returns — a sink whose endpoint is wedged.
        hang_on_shutdown: bool,
    }

    #[async_trait]
    impl RequestTraceSink for FakeSink {
        fn name(&self) -> &'static str {
            "fake"
        }

        async fn emit(&self, record: &RequestTraceRecord) {
            if record
                .request
                .as_ref()
                .is_some_and(|request| request.request_id == "req-123")
            {
                self.emitted.fetch_add(1, Ordering::SeqCst);
            }
        }

        async fn shutdown(&self) {
            if self.hang_on_shutdown {
                std::future::pending::<()>().await;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            self.shutdown_done.store(true, Ordering::SeqCst);
        }

        fn dropped_records(&self) -> u64 {
            self.dropped
        }
    }

    #[tokio::test]
    async fn shutdown_drains_the_backlog_before_returning() {
        crate::request_trace::init_bus_for_test(64);
        let emitted = Arc::new(AtomicUsize::new(0));
        let shutdown_done = Arc::new(AtomicBool::new(false));
        let sink: Arc<dyn RequestTraceSink> = Arc::new(FakeSink {
            emitted: emitted.clone(),
            shutdown_done: shutdown_done.clone(),
            dropped: 0,
            hang_on_shutdown: false,
        });
        let workers = spawn_workers(vec![sink], CancellationToken::new());

        crate::request_trace::publish(sample_record());

        let report = workers.shutdown(Duration::from_secs(5)).await;

        assert!(!report.timed_out);
        assert_eq!(
            emitted.load(Ordering::SeqCst),
            1,
            "the record published before shutdown should reach the sink"
        );
        assert!(
            shutdown_done.load(Ordering::SeqCst),
            "shutdown returned before the sink worker finished"
        );
    }

    #[tokio::test]
    async fn shutdown_timeout_reports_pending_sinks_and_their_drops() {
        crate::request_trace::init_bus_for_test(64);
        let sink: Arc<dyn RequestTraceSink> = Arc::new(FakeSink {
            emitted: Arc::new(AtomicUsize::new(0)),
            shutdown_done: Arc::new(AtomicBool::new(false)),
            dropped: 1132,
            hang_on_shutdown: true,
        });
        let workers = spawn_workers(vec![sink], CancellationToken::new());

        let report = workers.shutdown(Duration::from_millis(100)).await;

        assert!(report.timed_out);
        assert_eq!(report.pending, vec![("fake", 1132)]);
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
