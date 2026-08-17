// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::io::Write;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use anyhow::{Context as _, anyhow};
use async_nats::jetstream;
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;
use dynamo_runtime::transports::nats;
use tokio::sync::{Mutex, broadcast};
use tokio_util::sync::CancellationToken;

use crate::telemetry::jsonl::{JsonlSinkOptions, JsonlWriter};
use crate::telemetry::jsonl_gz::{JsonlGzipSinkOptions, JsonlGzipWriter};

use super::{
    RequestTraceFileFormat, RequestTracePolicy, RequestTraceRecord, RequestTraceSinkKind, config,
    otel_sink::OtelRequestTraceSink,
};

/// One live set of sink workers. Every worker of the generation holds a clone of
/// the `Arc`, so the generation is alive exactly while at least one worker is
/// still running: the last worker to finish drops the final strong reference and
/// `stopped` fires. The registry below keeps only a `Weak`, so a generation that
/// has wound down cannot block the next one from starting.
struct WorkerGeneration {
    stopped: CancellationToken,
}

impl Drop for WorkerGeneration {
    fn drop(&mut self) {
        self.stopped.cancel();
    }
}

static GENERATION: OnceLock<Mutex<Weak<WorkerGeneration>>> = OnceLock::new();

fn generation() -> &'static Mutex<Weak<WorkerGeneration>> {
    GENERATION.get_or_init(|| Mutex::new(Weak::new()))
}

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

pub async fn spawn_workers_from_env(shutdown: CancellationToken) -> anyhow::Result<()> {
    spawn_generation(shutdown, parse_sinks_from_env).await
}

/// Start one generation of sink workers unless a generation is already live.
///
/// The lock is held across sink construction so that an initialization
/// overlapping a slow sink connect waits for, and reports, the real outcome
/// instead of being told success while the first one is still connecting.
async fn spawn_generation<F, Fut>(shutdown: CancellationToken, make_sinks: F) -> anyhow::Result<()>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = anyhow::Result<Vec<Arc<dyn RequestTraceSink>>>>,
{
    let mut live = generation().lock().await;
    if live.upgrade().is_some() {
        return Ok(());
    }

    let generation = Arc::new(WorkerGeneration {
        stopped: CancellationToken::new(),
    });
    // On error nothing is stored, so the next initialization retries.
    let sinks = make_sinks().await?;
    spawn_workers(shutdown, sinks, &generation);
    *live = Arc::downgrade(&generation);
    Ok(())
}

fn spawn_workers(
    shutdown: CancellationToken,
    sinks: Vec<Arc<dyn RequestTraceSink>>,
    generation: &Arc<WorkerGeneration>,
) {
    let sink_count = sinks.len();
    for sink in sinks {
        let name = sink.name();
        let mut receiver: broadcast::Receiver<RequestTraceRecord> = super::subscribe();
        let worker_shutdown = shutdown.clone();
        let generation = Arc::clone(generation);
        tokio::spawn(async move {
            // Named binding on purpose: a bare `_` would drop the handle here
            // and release the generation while this worker is still draining.
            let _generation = generation;
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
        tracing::warn!("request trace is enabled but no valid request trace sinks were configured");
    }
    tracing::info!(sinks = sink_count, "Request trace sinks ready");
}

/// Clone of the live generation's `stopped` token, for tests that need to await
/// a generation winding down. It must never hand out the `Arc` itself: an
/// awaiting caller holding one would keep the generation alive forever.
#[cfg(test)]
async fn live_generation_stopped() -> Option<CancellationToken> {
    generation()
        .lock()
        .await
        .upgrade()
        .map(|generation| generation.stopped.clone())
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use flate2::read::MultiGzDecoder;
    use tempfile::tempdir;
    use tokio::sync::mpsc;

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

    fn record_with_request_id(request_id: &str) -> RequestTraceRecord {
        let mut record = sample_record();
        if let Some(request) = record.request.as_mut() {
            request.request_id = request_id.to_string();
        }
        record
    }

    /// The generation registry is process-global, so the tests that drive it end
    /// to end must not overlap with each other.
    static GENERATION_TEST_LOCK: Mutex<()> = Mutex::const_new(());

    struct RecordingSink {
        emitted: mpsc::UnboundedSender<RequestTraceRecord>,
        shutdowns: Arc<AtomicUsize>,
    }

    impl RecordingSink {
        fn new() -> (
            Arc<Self>,
            mpsc::UnboundedReceiver<RequestTraceRecord>,
            Arc<AtomicUsize>,
        ) {
            let (emitted, records) = mpsc::unbounded_channel();
            let shutdowns = Arc::new(AtomicUsize::new(0));
            let sink = Arc::new(Self {
                emitted,
                shutdowns: Arc::clone(&shutdowns),
            });
            (sink, records, shutdowns)
        }
    }

    #[async_trait]
    impl RequestTraceSink for RecordingSink {
        fn name(&self) -> &'static str {
            "recording"
        }

        async fn emit(&self, record: &RequestTraceRecord) {
            let _ = self.emitted.send(record.clone());
        }

        async fn shutdown(&self) {
            self.shutdowns.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Wait for one specific record. The trace bus is process-global and other
    /// tests in this binary publish to it, so a sink cannot assume the first
    /// record it sees is the one its own test published.
    async fn await_record(
        records: &mut mpsc::UnboundedReceiver<RequestTraceRecord>,
        request_id: &str,
    ) -> bool {
        tokio::time::timeout(Duration::from_secs(5), async {
            while let Some(record) = records.recv().await {
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == request_id)
                {
                    return true;
                }
            }
            false
        })
        .await
        .unwrap_or(false)
    }

    async fn shutdown_generation(shutdown: CancellationToken) {
        shutdown.cancel();
        if let Some(stopped) = live_generation_stopped().await {
            tokio::time::timeout(Duration::from_secs(5), stopped.cancelled())
                .await
                .expect("generation did not release after its shutdown token fired");
        }
    }

    #[tokio::test]
    async fn sink_workers_restart_after_generation_shutdown() {
        let _serialized = GENERATION_TEST_LOCK.lock().await;
        crate::request_trace::init_bus_for_test(64);

        let (sink_one, mut records_one, shutdowns_one) = RecordingSink::new();
        let token_one = CancellationToken::new();
        spawn_generation(token_one.clone(), || async move {
            let sinks: Vec<Arc<dyn RequestTraceSink>> = vec![sink_one];
            Ok(sinks)
        })
        .await
        .unwrap();

        crate::request_trace::publish(record_with_request_id("restart-generation-one"));
        assert!(
            await_record(&mut records_one, "restart-generation-one").await,
            "the first generation's sink never received its record"
        );

        let stopped = live_generation_stopped()
            .await
            .expect("the first generation is live");
        token_one.cancel();
        tokio::time::timeout(Duration::from_secs(5), stopped.cancelled())
            .await
            .expect("the first generation did not release after its shutdown token fired");
        assert_eq!(
            shutdowns_one.load(Ordering::Acquire),
            1,
            "the first generation released before its sink was shut down"
        );

        let (sink_two, mut records_two, _shutdowns_two) = RecordingSink::new();
        let token_two = CancellationToken::new();
        spawn_generation(token_two.clone(), || async move {
            let sinks: Vec<Arc<dyn RequestTraceSink>> = vec![sink_two];
            Ok(sinks)
        })
        .await
        .unwrap();

        crate::request_trace::publish(record_with_request_id("restart-generation-two"));
        assert!(
            await_record(&mut records_two, "restart-generation-two").await,
            "the second generation's sink never received its record"
        );

        shutdown_generation(token_two).await;
    }

    #[tokio::test]
    async fn concurrent_initializers_share_one_generation() {
        let _serialized = GENERATION_TEST_LOCK.lock().await;
        crate::request_trace::init_bus_for_test(64);

        let (sink_one, mut records_one, _shutdowns_one) = RecordingSink::new();
        let token = CancellationToken::new();
        spawn_generation(token.clone(), || async move {
            let sinks: Vec<Arc<dyn RequestTraceSink>> = vec![sink_one];
            Ok(sinks)
        })
        .await
        .unwrap();

        let (sink_two, mut records_two, _shutdowns_two) = RecordingSink::new();
        let second_generation_builds = Arc::new(AtomicUsize::new(0));
        let builds = Arc::clone(&second_generation_builds);
        spawn_generation(CancellationToken::new(), move || async move {
            builds.fetch_add(1, Ordering::AcqRel);
            let sinks: Vec<Arc<dyn RequestTraceSink>> = vec![sink_two];
            Ok(sinks)
        })
        .await
        .unwrap();

        assert_eq!(
            second_generation_builds.load(Ordering::Acquire),
            0,
            "an initialization overlapping a live generation built a second set of sinks"
        );

        crate::request_trace::publish(record_with_request_id("shared-generation"));
        assert!(
            await_record(&mut records_one, "shared-generation").await,
            "the live generation's sink stopped receiving records"
        );
        // No worker was ever spawned for the second sink, so this cannot race
        // with a delivery still in flight.
        assert!(
            records_two.try_recv().is_err(),
            "a duplicate worker emitted into the overlapping initializer's sink"
        );

        shutdown_generation(token).await;
    }

    #[tokio::test]
    async fn generation_with_no_sinks_does_not_latch() {
        let _serialized = GENERATION_TEST_LOCK.lock().await;
        crate::request_trace::init_bus_for_test(64);

        spawn_generation(CancellationToken::new(), || async { Ok(Vec::new()) })
            .await
            .unwrap();
        assert!(
            live_generation_stopped().await.is_none(),
            "a generation with no workers stayed live"
        );

        let (sink, mut records, _shutdowns) = RecordingSink::new();
        let token = CancellationToken::new();
        spawn_generation(token.clone(), || async move {
            let sinks: Vec<Arc<dyn RequestTraceSink>> = vec![sink];
            Ok(sinks)
        })
        .await
        .unwrap();

        crate::request_trace::publish(record_with_request_id("after-empty-generation"));
        assert!(
            await_record(&mut records, "after-empty-generation").await,
            "a generation configured with no sinks latched the guard"
        );

        shutdown_generation(token).await;
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
