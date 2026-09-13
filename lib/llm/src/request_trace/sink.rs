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
use tokio::sync::{Mutex, broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::telemetry::jsonl::{JsonlSinkOptions, JsonlWriter};
use crate::telemetry::jsonl_gz::{JsonlGzipSinkOptions, JsonlGzipWriter};

use super::{
    RequestTraceFileFormat, RequestTracePolicy, RequestTraceRecord, RequestTraceSinkKind, config,
    otel_sink::OtelRequestTraceSink,
};

// Workers own the generation; the registry holds only a weak reference so shutdown permits restart.
struct WorkerGeneration {
    shutdown: CancellationToken,
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
    /// `None` once the sink has been shut down; further records are dropped.
    writer: tokio::sync::Mutex<Option<JsonlWriter<RequestTraceRecord>>>,
}

impl JsonlRequestTraceSink {
    pub async fn new(path: String, options: JsonlSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening jsonl request trace sink at {path}"))?;
        Ok(Self {
            writer: tokio::sync::Mutex::new(Some(writer)),
        })
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
        let guard = self.writer.lock().await;
        match guard.as_ref() {
            Some(writer) => {
                if writer.send(record.clone()).await.is_err() {
                    tracing::warn!("request trace file writer channel closed; dropping record");
                }
            }
            None => tracing::warn!("request trace file sink shut down; dropping record"),
        }
    }

    async fn shutdown(&self) {
        // Serialize callers until the drain finishes, including concurrent shutdowns.
        let mut guard = self.writer.lock().await;
        if let Some(writer) = guard.as_mut() {
            if let Err(error) = writer.shutdown().await {
                tracing::warn!(%error, "request trace file sink shutdown failed");
            }
            guard.take();
        }
    }
}

pub struct JsonlGzipRequestTraceSink {
    // Cloned input channel used by emit, so concurrent emits never contend on the
    // writer lock. Sending fails once the writer closes admission for shutdown.
    sender: mpsc::Sender<RequestTraceRecord>,
    // shutdown consumes the writer; None means it has already closed.
    writer: Mutex<Option<JsonlGzipWriter<RequestTraceRecord>>>,
}

impl JsonlGzipRequestTraceSink {
    pub async fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let writer = JsonlGzipWriter::new(path.clone(), options)
            .await
            .with_context(|| format!("opening gzip jsonl request trace sink at {path}"))?;
        // A freshly constructed writer always has its sender, so this is Some.
        let sender = writer
            .sender()
            .expect("newly constructed JsonlGzipWriter always has a sender");
        Ok(Self {
            sender,
            writer: Mutex::new(Some(writer)),
        })
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
        // Lock-free: send straight to the writer task's channel. After shutdown the
        // receiver is gone, so this errors and the record is dropped.
        if self.sender.send(record.clone()).await.is_err() {
            tracing::warn!("request trace file sink closed; dropping record");
        }
    }

    async fn shutdown(&self) {
        // Serialize shutdown callers until the final flush completes. Keep the
        // writer available if this caller is cancelled while awaiting it.
        let mut writer = self.writer.lock().await;
        if let Some(writer) = writer.as_mut()
            && let Err(error) = writer.shutdown().await
        {
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                error = %error,
                "request trace file sink: gzip writer close failed during shutdown"
            );
        }
        writer.take();
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

async fn spawn_generation<F, Fut>(shutdown: CancellationToken, make_sinks: F) -> anyhow::Result<()>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = anyhow::Result<Vec<Arc<dyn RequestTraceSink>>>>,
{
    loop {
        let mut live = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return Err(anyhow!("request trace initialization cancelled")),
            live = generation().lock() => live,
        };
        if let Some(existing) = live.upgrade() {
            if !existing.shutdown.is_cancelled() {
                return Ok(());
            }
            // Release the strong reference so the last worker can signal its drop.
            let stopped = existing.stopped.clone();
            drop(existing);
            drop(live);
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Err(anyhow!("request trace initialization cancelled")),
                _ = stopped.cancelled() => {},
            }
            continue;
        }

        let generation = Arc::new(WorkerGeneration {
            shutdown: shutdown.clone(),
            stopped: CancellationToken::new(),
        });
        let sinks = make_sinks().await?;
        anyhow::ensure!(
            !shutdown.is_cancelled(),
            "request trace initialization cancelled"
        );
        spawn_workers(shutdown, sinks, &generation);
        *live = Arc::downgrade(&generation);
        return Ok(());
    }
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
            let _generation_guard = generation;
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
    use tokio::sync::{mpsc, oneshot};

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

    // Tests that drive the process-global generation registry must not overlap.
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

    // Other tests publish to the process-global bus, so filter by request ID.
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

    struct DrainingSink {
        entered: CancellationToken,
        release: CancellationToken,
    }

    #[async_trait]
    impl RequestTraceSink for DrainingSink {
        fn name(&self) -> &'static str {
            "draining"
        }

        async fn emit(&self, _record: &RequestTraceRecord) {}

        async fn shutdown(&self) {
            self.entered.cancel();
            self.release.cancelled().await;
        }
    }

    #[tokio::test]
    async fn successor_waits_for_draining_generation_and_receives_records() {
        let _serialized = GENERATION_TEST_LOCK.lock().await;
        crate::request_trace::init_bus_for_test(64);
        let entered = CancellationToken::new();
        let release = CancellationToken::new();
        let sink = Arc::new(DrainingSink {
            entered: entered.clone(),
            release: release.clone(),
        });
        let old_token = CancellationToken::new();
        spawn_generation(old_token.clone(), || async move {
            Ok(vec![sink as Arc<dyn RequestTraceSink>])
        })
        .await
        .unwrap();
        old_token.cancel();
        entered.cancelled().await;

        // Cancellation of a waiting runtime must not stop or replace the old drain.
        let cancelled_token = CancellationToken::new();
        let mut cancelled = Box::pin(spawn_generation(cancelled_token.clone(), || async {
            panic!("cancelled waiter must never construct sinks");
        }));
        assert!(futures::poll!(cancelled.as_mut()).is_pending());
        cancelled_token.cancel();
        assert!(cancelled.await.is_err());

        let (sink, mut records, _) = RecordingSink::new();
        let token = CancellationToken::new();
        let builds = AtomicUsize::new(0);
        let mut successor = Box::pin(spawn_generation(token.clone(), || async {
            builds.fetch_add(1, Ordering::AcqRel);
            Ok(vec![sink as Arc<dyn RequestTraceSink>])
        }));
        assert!(futures::poll!(successor.as_mut()).is_pending());
        assert_eq!(builds.load(Ordering::Acquire), 0);

        release.cancel();
        tokio::time::timeout(Duration::from_secs(5), successor)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(builds.load(Ordering::Acquire), 1);
        crate::request_trace::publish(record_with_request_id("successor-after-drain"));
        assert!(await_record(&mut records, "successor-after-drain").await);
        shutdown_generation(token).await;
    }

    #[tokio::test]
    async fn second_initializer_reuses_live_generation() {
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
        assert!(
            records_two.try_recv().is_err(),
            "a duplicate worker emitted into the overlapping initializer's sink"
        );

        shutdown_generation(token).await;
    }

    #[tokio::test]
    async fn overlapping_initializer_waits_for_slow_sink_construction() {
        let _serialized = GENERATION_TEST_LOCK.lock().await;
        crate::request_trace::init_bus_for_test(64);

        let (sink_one, mut records_one, _shutdowns_one) = RecordingSink::new();
        let (construction_started, started) = oneshot::channel();
        let (release, wait_for_release) = oneshot::channel();
        let token = CancellationToken::new();
        let first = tokio::spawn(spawn_generation(token.clone(), move || async move {
            let _ = construction_started.send(());
            wait_for_release
                .await
                .expect("the test releases the blocked construction");
            let sinks: Vec<Arc<dyn RequestTraceSink>> = vec![sink_one];
            Ok(sinks)
        }));
        started
            .await
            .expect("the first initializer reached sink construction");

        let (sink_two, _records_two, _shutdowns_two) = RecordingSink::new();
        let second_generation_builds = Arc::new(AtomicUsize::new(0));
        let builds = Arc::clone(&second_generation_builds);
        let mut second = tokio::spawn(spawn_generation(
            CancellationToken::new(),
            move || async move {
                builds.fetch_add(1, Ordering::AcqRel);
                let sinks: Vec<Arc<dyn RequestTraceSink>> = vec![sink_two];
                Ok(sinks)
            },
        ));

        assert!(
            tokio::time::timeout(Duration::from_millis(250), &mut second)
                .await
                .is_err(),
            "an initializer overlapping an unfinished sink construction returned before it completed"
        );
        assert_eq!(
            second_generation_builds.load(Ordering::Acquire),
            0,
            "the overlapping initializer built sinks while the first construction held the lock"
        );

        release
            .send(())
            .expect("the first initializer is still waiting on construction");
        first
            .await
            .expect("the first initializer task did not panic")
            .expect("the first initializer started its generation");
        tokio::time::timeout(Duration::from_secs(5), &mut second)
            .await
            .expect("the overlapping initializer never returned after construction completed")
            .expect("the overlapping initializer task did not panic")
            .expect("the overlapping initializer reported success");

        assert_eq!(
            second_generation_builds.load(Ordering::Acquire),
            0,
            "the overlapping initializer built a second set of sinks once it acquired the lock"
        );

        crate::request_trace::publish(record_with_request_id("slow-construction"));
        assert!(
            await_record(&mut records_one, "slow-construction").await,
            "the generation built under contention never received its record"
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

    #[tokio::test]
    async fn gzip_sink_shutdown_flushes_buffered_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace_shutdown");
        let sink = JsonlGzipRequestTraceSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024 * 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: None,
            },
        )
        .await
        .unwrap();

        sink.emit(&sample_record()).await;

        RequestTraceSink::shutdown(&sink).await;
        RequestTraceSink::shutdown(&sink).await;

        let segment = segment_path(&path, 0);
        assert!(
            segment.exists(),
            "shutdown returned without flushing the gzip segment at {}",
            segment.display()
        );
        let bytes = std::fs::read(&segment).unwrap();
        let mut content = String::new();
        MultiGzDecoder::new(bytes.as_slice())
            .read_to_string(&mut content)
            .unwrap();
        assert!(content.contains("\"request_id\":\"req-123\""));
    }

    #[tokio::test]
    async fn gzip_sink_concurrent_shutdown_waits_for_reserved_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace_concurrent_shutdown");
        let sink = JsonlGzipRequestTraceSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions::default(),
        )
        .await
        .unwrap();
        let permit = sink.sender.clone().reserve_owned().await.unwrap();
        let first = RequestTraceSink::shutdown(&sink);
        let second = RequestTraceSink::shutdown(&sink);
        tokio::pin!(first, second);
        tokio::select! {
            _ = &mut first => panic!("shutdown abandoned an outstanding permit"),
            _ = tokio::time::timeout(Duration::from_secs(5), sink.sender.closed()) => {
                assert!(sink.sender.is_closed(), "shutdown must close admission");
            }
        }
        assert!(futures::poll!(&mut second).is_pending());
        permit.send(sample_record());
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(first, second);
        })
        .await
        .unwrap();

        let bytes = std::fs::read(segment_path(&path, 0)).unwrap();
        let mut content = String::new();
        MultiGzDecoder::new(bytes.as_slice())
            .read_to_string(&mut content)
            .unwrap();
        assert!(content.contains("\"request_id\":\"req-123\""));
    }

    #[tokio::test]
    async fn gzip_sink_emit_after_shutdown_drops_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace_after_shutdown");
        let sink = JsonlGzipRequestTraceSink::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024 * 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: None,
            },
        )
        .await
        .unwrap();

        // After shutdown the writer is gone, so emit() must hit the closed-writer
        // branch: the record is dropped (with a warning) rather than written, and
        // nothing panics.
        RequestTraceSink::shutdown(&sink).await;
        sink.emit(&sample_record()).await;

        let segment = segment_path(&path, 0);
        let written = if segment.exists() {
            let bytes = std::fs::read(&segment).unwrap();
            let mut content = String::new();
            let _ = MultiGzDecoder::new(bytes.as_slice()).read_to_string(&mut content);
            content.contains("\"request_id\":\"req-123\"")
        } else {
            false
        };
        assert!(
            !written,
            "record emitted after shutdown must be dropped, not written to {}",
            segment.display()
        );
    }

    #[tokio::test]
    async fn jsonl_sink_shutdown_drains_accepted_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace_shutdown.jsonl");
        // Nothing but shutdown can flush this record: the buffer dwarfs one
        // record and the flush tick is a minute away.
        let sink = JsonlRequestTraceSink::new(
            path.display().to_string(),
            JsonlSinkOptions {
                buffer_bytes: 1024 * 1024,
                flush_interval: Duration::from_secs(60),
            },
        )
        .await
        .unwrap();

        sink.emit(&sample_record()).await;

        RequestTraceSink::shutdown(&sink).await;
        // A second shutdown must return normally rather than panic.
        RequestTraceSink::shutdown(&sink).await;

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(
            content.contains("\"request_id\":\"req-123\""),
            "shutdown returned without flushing the accepted record to {}",
            path.display()
        );
    }

    #[tokio::test]
    async fn jsonl_sink_shutdown_flushes_buffered_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("request_trace_jsonl_shutdown.jsonl");
        let sink = JsonlRequestTraceSink::new(
            path.display().to_string(),
            JsonlSinkOptions {
                // Large buffer + long interval: nothing reaches disk until shutdown.
                buffer_bytes: 1024 * 1024,
                flush_interval: Duration::from_secs(60),
            },
        )
        .await
        .unwrap();

        sink.emit(&sample_record()).await;
        RequestTraceSink::shutdown(&sink).await;

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("\"request_id\":\"req-123\""),
            "shutdown must flush the buffered record; file was: {content:?}"
        );
    }
}
