// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use anyhow::{Context as _, anyhow};
use async_nats::jetstream;
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;
use dynamo_runtime::transports::nats;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::telemetry::jsonl::{JsonlSinkOptions, JsonlWriter};
use crate::telemetry::jsonl_gz::{JsonlGzipSinkOptions, JsonlGzipWriter};

use super::{
    RequestTraceFileFormat, RequestTracePolicy, RequestTraceRecord, RequestTraceSinkKind, config,
    otel_sink::OtelRequestTraceSink,
};

static WORKERS_STARTED: AtomicBool = AtomicBool::new(false);
static WORKERS: Mutex<Option<SinkWorkers>> = Mutex::new(None);
// Serializes worker creation against shutdown. The worker slot alone cannot do
// that because shutdown removes a generation before awaiting its final drain.
static WORKER_LIFECYCLE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static ACTIVE_INPUTS: AtomicUsize = AtomicUsize::new(0);

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
    writer: tokio::sync::Mutex<Option<JsonlGzipWriter<RequestTraceRecord>>>,
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
    /// `timeout`. On timeout the sink tasks are aborted after their counts are
    /// captured, so the next worker generation cannot overlap them.
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
        for handle in &handles {
            handle.abort();
        }
        // Wait for cancellation to complete before reopening the start gate.
        // Dropping a JoinHandle would detach its task and let it consume new
        // broadcast records alongside the next worker generation.
        let _ = futures::future::join_all(handles.iter_mut()).await;
        TraceShutdownReport {
            timed_out: true,
            pending,
        }
    }
}

pub async fn spawn_workers_from_env(shutdown: CancellationToken) -> anyhow::Result<()> {
    let _lifecycle = WORKER_LIFECYCLE.lock().await;
    if WORKERS_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        let workers = {
            let mut slot = WORKERS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !slot
                .as_ref()
                .is_some_and(|workers| workers.token.is_cancelled())
            {
                return Ok(());
            }
            slot.take().expect("cancelled workers must be retained")
        };
        // The last input can only cancel from `Drop`, where it cannot await a
        // drain. A later lifecycle owns the join: retire that cancelled
        // generation before it installs a replacement.
        workers.shutdown(SHUTDOWN_TIMEOUT).await;
        WORKERS_STARTED.store(false, Ordering::Release);
        WORKERS_STARTED.store(true, Ordering::Release);
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

/// Registers one running input for as long as it is held.
///
/// The sink workers are one process-wide set, but an input is not: the mocker
/// starts one endpoint input per worker and awaits them together, and the HTTP
/// frontend runs nested inside another input. Draining when the first of those
/// returns would leave the rest publishing into a bus with no worker behind it,
/// so the drain waits for the last registration to be released.
///
/// Only [`ActiveInput::release_and_drain`] gives the bounded drain. Dropping the
/// last guard instead — a cancelled or panicking input — cancels the workers so
/// they start draining, but cannot wait for them, and warns.
pub struct ActiveInput(());

impl ActiveInput {
    /// Count this input as running until the guard is released or dropped.
    pub fn register() -> Self {
        ACTIVE_INPUTS.fetch_add(1, Ordering::AcqRel);
        Self(())
    }

    /// Release this registration and, when it was the last one, cancel the sink
    /// workers and wait for them to drain. Returns `None` while another input
    /// is still running, and when no workers were started.
    pub async fn release_and_drain(self) -> Option<TraceShutdownReport> {
        // Serialize the final decrement with worker creation. Otherwise a new
        // input could reuse the old workers immediately before this last input
        // takes them out of the slot to drain.
        let _lifecycle = WORKER_LIFECYCLE.lock().await;
        if !self.release() {
            return None;
        }
        shutdown_workers_locked().await
    }

    /// Release the registration, reporting whether it was the last one. `Drop`
    /// releases it too, so the guard is forgotten here rather than released
    /// twice.
    fn release(self) -> bool {
        let was_last = ACTIVE_INPUTS.fetch_sub(1, Ordering::AcqRel) == 1;
        std::mem::forget(self);
        was_last
    }
}

impl Drop for ActiveInput {
    fn drop(&mut self) {
        // Take the worker slot before releasing the registration. A new input
        // either increments the count first (so this is not the last guard),
        // or waits for this cancellation and then retires the cancelled
        // generation before it reuses the slot.
        let slot = WORKERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if ACTIVE_INPUTS.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        // The last input ended without calling `release_and_drain`, so it was
        // cancelled or it panicked. `Drop` cannot await, and spawning a
        // detached drain would only look like one: nothing would be left to
        // wait on it, which is the timing guess this change exists to remove.
        // Cancelling is the part that can be done here, and it is worth doing:
        // each worker leaves its loop and runs the sink's own `shutdown`, so a
        // caller that keeps the runtime alive past the cancellation still gets
        // the backlog written. What cannot be promised is that it finishes
        // before the process exits, so say so rather than fail quietly. There
        // is nothing to say when no workers were running, which is every run
        // with request tracing switched off.
        if let Some(workers) = slot.as_ref() {
            workers.token.cancel();
            tracing::warn!(
                "request trace sinks were cancelled without a bounded drain because the last \
                 input ended early; records still queued may be lost if the process exits \
                 immediately"
            );
        }
    }
}

/// Cancel the retained workers and wait for them to drain, bounded by
/// [`SHUTDOWN_TIMEOUT`]. Returns `None` when no workers were started, which is
/// the case whenever request tracing is disabled.
pub async fn shutdown_workers() -> Option<TraceShutdownReport> {
    let _lifecycle = WORKER_LIFECYCLE.lock().await;
    shutdown_workers_locked().await
}

async fn shutdown_workers_locked() -> Option<TraceShutdownReport> {
    let workers = {
        let mut slot = WORKERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slot.take()?
    };
    let report = workers.shutdown(SHUTDOWN_TIMEOUT).await;
    // Keep the gate closed until the retired generation is fully stopped. This
    // prevents a new subscription from receiving records that the old workers
    // are still draining.
    WORKERS_STARTED.store(false, Ordering::Release);
    Some(report)
}

fn spawn_workers(
    sinks: Vec<Arc<dyn RequestTraceSink>>,
    shutdown: CancellationToken,
) -> SinkWorkers {
    let sink_count = sinks.len();
    // The workers are process-wide, so they cannot hang off the runtime of
    // whichever input happened to initialize tracing first. The mocker gives
    // each of its workers its own runtime, and one of those ending — cleanly or
    // not — would otherwise cancel the shared sinks while the other inputs are
    // still publishing into them. When an input is registered, teardown belongs
    // to `ActiveInput`: the last one out cancels these workers whether it
    // returns, is cancelled, or panics.
    let token = CancellationToken::new();
    if ACTIVE_INPUTS.load(Ordering::Acquire) == 0 {
        // Started outside any input, so the caller's token is the only teardown
        // signal there is and the workers follow it as before.
        let linked = token.clone();
        tokio::spawn(async move {
            shutdown.cancelled().await;
            linked.cancel();
        });
    }
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

    fn fake_sink(shutdown_done: &Arc<AtomicBool>) -> Arc<dyn RequestTraceSink> {
        Arc::new(FakeSink {
            emitted: Arc::new(AtomicUsize::new(0)),
            shutdown_done: shutdown_done.clone(),
            dropped: 0,
            hang_on_shutdown: false,
        })
    }

    fn install_workers(sink: Arc<dyn RequestTraceSink>, shutdown: CancellationToken) {
        WORKERS_STARTED.store(true, Ordering::SeqCst);
        *WORKERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(spawn_workers(vec![sink], shutdown));
    }

    fn workers_are_cancelled() -> bool {
        WORKERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .expect("the workers should still be installed")
            .token
            .is_cancelled()
    }

    /// `ACTIVE_INPUTS`, `WORKERS` and `WORKERS_STARTED` are process-wide, and
    /// the entrypoint tests register an input of their own, so this shares a
    /// serialization group with them rather than racing their registrations.
    #[tokio::test]
    #[serial_test::serial(request_trace_lifecycle)]
    async fn only_the_last_input_out_stops_the_shared_workers() {
        crate::request_trace::init_bus_for_test(64);
        let shutdown_done = Arc::new(AtomicBool::new(false));

        let first = ActiveInput::register();
        let second = ActiveInput::register();
        // The runtime of the input that initialized tracing. Every later input
        // has a runtime of its own, so this one ending says nothing about them.
        let first_runtime = CancellationToken::new();
        install_workers(fake_sink(&shutdown_done), first_runtime.clone());

        first_runtime.cancel();
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            !workers_are_cancelled(),
            "one input's runtime must not stop the shared workers while another input is still publishing"
        );

        assert!(
            !first.release(),
            "an input that finishes while another is still running must not drain"
        );

        // Dropping the last guard is the cancelled or panicking input. It
        // cannot await the drain, but it must still start one.
        drop(second);
        assert_eq!(ACTIVE_INPUTS.load(Ordering::SeqCst), 0);
        assert!(
            workers_are_cancelled(),
            "dropping the last input must cancel the workers"
        );

        let workers = WORKERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .expect("the drop path must leave the workers joinable");
        tokio::time::timeout(
            Duration::from_secs(5),
            futures::future::join_all(workers.handles),
        )
        .await
        .expect("the cancelled workers should finish on their own");
        assert!(
            shutdown_done.load(Ordering::SeqCst),
            "each sink should have run its own shutdown after the cancellation"
        );

        // A second set of inputs in the same process: the last one out gets the
        // bounded drain, and the drain reopens the start gate so a set after
        // that one can be spawned at all.
        let drained = Arc::new(AtomicBool::new(false));
        install_workers(fake_sink(&drained), CancellationToken::new());
        let only = ActiveInput::register();

        let report = only
            .release_and_drain()
            .await
            .expect("the last input out drains the workers it can still see");

        assert!(!report.timed_out);
        assert!(drained.load(Ordering::SeqCst));
        assert!(
            !WORKERS_STARTED.load(Ordering::SeqCst),
            "a drained worker set must leave the start gate open for the next one"
        );
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
