// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mocker module - runtime integration for the mock scheduler.
//!
//! The core mocker logic lives in the `dynamo-mocker` crate.
//! This module provides the runtime-dependent engine wrapper.

mod metrics;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::backend::ExecutionContext;
use crate::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig, WorkerMetricsPublisher};
use crate::protocols::TokenIdType;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use anyhow::Result;
use dashmap::DashMap;
use dynamo_kv_router::protocols::{KvCacheEvent, StorageTier};
use dynamo_mocker::common::protocols::{
    DirectRequest, EngineType, KvCacheEventSink, KvEventPublishers, MockEngineArgs, OutputSignal,
    RawKvEventSink,
};
use dynamo_mocker::common::utils::sleep_precise;
use dynamo_mocker::engine::create_engine;
use dynamo_mocker::scheduler::SchedulerHandle;
use dynamo_mocker::services::bootstrap::{BootstrapServer, connect_to_prefill};
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{
    component::Component,
    engine::AsyncEngineContextProvider,
    pipeline::{
        AsyncEngine, Error, ManyOut, ResponseStream, SingleIn, async_trait, network::Ingress,
    },
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{Notify, OnceCell, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use self::metrics::NativeMockerMetrics;

pub const MOCKER_COMPONENT: &str = "mocker";

/// vLLM-flavored `disaggregated_params` payload emitted by a mocker prefill in
/// its first output and read back by the matching decode.
///
/// This mirrors the *shape* of real vLLM `kv_transfer_params` (NIXL): the
/// prefill announces an opaque transfer handle plus where to pull it from.
/// `transfer_id` is the channel key (a [`dynamo_mocker::services::bootstrap::TransferId`]);
/// `prefill_host`/`prefill_port` address the prefill's bootstrap server so the
/// decode can connect for the modeled pull. Sourced **post-prefill-compute**
/// (unlike sglang's frontend-precomputed bootstrap room).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct VllmDisaggParams {
    transfer_id: u64,
    prefill_host: String,
    prefill_port: u16,
}

impl VllmDisaggParams {
    fn parse(value: &serde_json::Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }
}

/// Wrapper to adapt KvEventPublisher to the KvCacheEventSink trait
struct KvEventSinkAdapter(KvEventPublisher);

impl KvCacheEventSink for KvEventSinkAdapter {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        self.0
            .publish(event)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e))
    }

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> anyhow::Result<()> {
        self.0
            .publish_with_storage_tier(event, storage_tier)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e))
    }
}

fn generate_random_token() -> TokenIdType {
    let mut rng = rand::rng();
    rng.random_range(1000..2000)
}

/// AsyncEngine wrapper around the Scheduler that generates random character tokens
pub struct MockEngine {
    active_requests: Arc<DashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>,
    request_senders: OnceCell<Vec<mpsc::UnboundedSender<DirectRequest>>>,
    senders_ready: Notify,
    engine_args: MockEngineArgs,
    unset_dp_rank_counter: AtomicU32,
    /// Monotonic source of vLLM disagg `transfer_id`s (the channel/pin key).
    /// Process-local and collision-free — the channel registry is long-lived,
    /// so a random id could collide and silently mis-release/leak a pinned KV.
    transfer_id_counter: std::sync::atomic::AtomicU64,
    /// Bootstrap server for prefill workers in disaggregated mode
    bootstrap_server: Arc<OnceCell<Arc<BootstrapServer>>>,
    native_metrics: Arc<NativeMockerMetrics>,
    /// Keep schedulers alive so their CancelGuards don't fire prematurely.
    _schedulers: OnceCell<Vec<Box<dyn SchedulerHandle>>>,
    /// Forward pass metrics publisher (kept alive for the engine lifetime).
    _fpm_publisher: OnceCell<crate::fpm_publisher::FpmDirectPublisher>,
}

struct MockSessionControlEngine;

#[async_trait]
impl AsyncEngine<SingleIn<serde_json::Value>, ManyOut<Annotated<serde_json::Value>>, Error>
    for MockSessionControlEngine
{
    async fn generate(
        &self,
        request: SingleIn<serde_json::Value>,
    ) -> Result<ManyOut<Annotated<serde_json::Value>>, Error> {
        let (body, context) = request.into_parts();
        let action = body.get("action").and_then(|value| value.as_str());
        let session_id = body.get("session_id").and_then(|value| value.as_str());

        let response = match (action, session_id) {
            (Some("open_session" | "close_session"), Some(session_id)) => {
                serde_json::json!({
                    "status": "ok",
                    "session_id": session_id,
                })
            }
            (_, None) => {
                serde_json::json!({
                    "status": "error",
                    "message": "session_id required",
                })
            }
            (other, Some(session_id)) => {
                serde_json::json!({
                    "status": "error",
                    "session_id": session_id,
                    "message": format!("unsupported action {:?}", other),
                })
            }
        };

        let stream = futures::stream::iter(vec![Annotated::from_data(response)]);
        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

impl MockEngine {
    /// Create a new MockEngine with the given parameters
    pub fn new(engine_args: MockEngineArgs) -> Self {
        let native_metrics = NativeMockerMetrics::new(engine_args.engine_type, engine_args.dp_size)
            .expect("mocker native metrics collectors should be valid");
        Self {
            active_requests: Arc::new(DashMap::new()),
            request_senders: OnceCell::new(),
            senders_ready: Notify::new(),
            engine_args,
            unset_dp_rank_counter: AtomicU32::new(0),
            transfer_id_counter: std::sync::atomic::AtomicU64::new(1),
            bootstrap_server: Arc::new(OnceCell::new()),
            native_metrics,
            _schedulers: OnceCell::new(),
            _fpm_publisher: OnceCell::new(),
        }
    }

    fn resolve_dp_rank(&self, request: &PreprocessedRequest) -> u32 {
        if let Some(dp_rank) = request.routing.as_ref().and_then(|routing| routing.dp_rank) {
            return dp_rank;
        }

        self.unset_dp_rank_counter.fetch_add(1, Ordering::Relaxed) % self.engine_args.dp_size
    }

    fn is_vllm(&self) -> bool {
        self.engine_args.engine_type == EngineType::Vllm
    }

    pub async fn start(&self, component: Component) -> Result<()> {
        // Use primary_token() instead of child_token() so the mocker continues running
        // during graceful shutdown (Phase 1/2) and only stops in Phase 3.
        // child_token() is a child of endpoint_shutdown_token which is cancelled in Phase 1.
        // primary_token() is only cancelled in Phase 3, after waiting for inflight requests.
        let cancel_token = component.drt().primary_token();
        self.native_metrics
            .register(component.get_metrics_registry())?;

        // Simulate engine startup time if configured
        if let Some(startup_time_secs) = self.engine_args.startup_time {
            tracing::info!("Simulating engine startup time: {:.2}s", startup_time_secs);
            tokio::time::sleep(Duration::from_secs_f64(startup_time_secs)).await;
            tracing::info!("Engine startup simulation completed");
        }

        Self::start_session_control_endpoint(component.clone());

        // Start the KV-transfer (bootstrap) server for prefill workers in
        // disaggregated mode. sglang (or vLLM with an explicit --bootstrap-ports)
        // binds the advertised port — UNCHANGED. A vLLM prefill with no
        // bootstrap_port still needs the cross-process channel for the
        // disaggregated_params (NIXL-style) pull, so it binds an OS-assigned
        // port (0); the real port is emitted in disaggregated_params and the
        // decode reads it. sglang never reaches the port-0 branch.
        let transfer_port = self
            .engine_args
            .bootstrap_port
            .or_else(|| (self.engine_args.is_prefill() && self.is_vllm()).then_some(0));
        if self.engine_args.is_prefill()
            && let Some(port) = transfer_port
        {
            let server = BootstrapServer::start(port, cancel_token.clone()).await?;
            tracing::info!(
                requested_port = port,
                bound_port = server.port(),
                "KV-transfer server started for prefill worker"
            );
            let _ = self.bootstrap_server.set(server);
        }

        let kv_component = if self.engine_args.needs_kv_publisher() {
            tracing::info!(
                "Initializing KV event publisher with block_size {}, enable_local_indexer={}",
                self.engine_args.block_size,
                self.engine_args.enable_local_indexer
            );
            Some(&component)
        } else {
            None
        };

        // Create FPM publisher upfront and get per-dp-rank sink handles.
        let worker_id = component.drt().connection_id().to_string();
        let fpm_sinks = match crate::fpm_publisher::FpmDirectPublisher::new(
            component.clone(),
            worker_id,
            self.engine_args.dp_size,
        )
        .await
        {
            Ok((publisher, sinks)) => {
                let _ = self._fpm_publisher.set(publisher);
                sinks
            }
            Err(e) => {
                tracing::error!("Failed to start FPM publisher: {e}");
                (0..self.engine_args.dp_size)
                    .map(|_| dynamo_mocker::common::protocols::FpmPublisher::default())
                    .collect()
            }
        };

        let schedulers = self
            .start_schedulers(kv_component, cancel_token.clone(), fpm_sinks)
            .await;

        Self::start_metrics_publishing(
            &schedulers,
            component.clone(),
            self.native_metrics.clone(),
            cancel_token.clone(),
        )
        .await?;

        let _ = self._schedulers.set(schedulers);

        Ok(())
    }

    fn start_session_control_endpoint(component: Component) {
        let ingress = match Ingress::for_engine(Arc::new(MockSessionControlEngine)) {
            Ok(ingress) => ingress,
            Err(e) => {
                tracing::error!("Failed to build mocker session_control ingress: {e}");
                return;
            }
        };

        tokio::spawn(async move {
            if let Err(e) = component
                .endpoint("session_control")
                .endpoint_builder()
                .handler(ingress)
                .graceful_shutdown(true)
                .start()
                .await
            {
                tracing::error!("Mocker session_control endpoint failed: {e}");
            }
        });
    }

    /// Wait until the scheduler at `dp_rank` reports headroom on **both** budgets
    /// vLLM/sglang gate the disaggregated KV recv on — block/token capacity AND a
    /// free sequence slot (`max_num_seqs`). Either alone blocks the recv; gating on
    /// blocks only would make the prefill-side abort path unreachable under
    /// seq-bound load. Graceful during warmup (`total_blocks == 0` → Ok) and when
    /// no seq cap is configured (`max_num_seqs == 0`). No timer on this side —
    /// cancellation is delegated to the request context; the prefill's
    /// `kv_transfer_abort_timeout_ms` bounds the strand. See the design doc.
    pub async fn wait_for_decode_kv_capacity(&self, dp_rank: u32) -> Result<()> {
        let schedulers = self
            ._schedulers
            .get()
            .ok_or_else(|| anyhow::anyhow!("schedulers not initialized"))?;
        let scheduler_idx = dp_rank as usize;
        if scheduler_idx >= schedulers.len() {
            return Err(anyhow::anyhow!(
                "dp_rank {dp_rank} out of bounds (have {} schedulers)",
                schedulers.len()
            ));
        }
        let mut rx = schedulers[scheduler_idx].metrics_receiver();

        loop {
            let metrics = rx.borrow().clone();
            // total_blocks == 0 means the scheduler hasn't published yet (still warming up).
            // Don't block on warmup — the scheduler's own queue will handle admission
            // once metrics start flowing.
            let has_block_capacity =
                metrics.total_blocks == 0 || metrics.active_decode_blocks < metrics.total_blocks;
            // max_num_seqs == 0 means "no seq cap configured" — mirror vLLM/sglang's
            // treatment of unset max_running_requests as effectively unbounded.
            // running_requests is the live sequence count (vLLM len(self.running) /
            // sglang req_to_token_pool occupancy).
            let has_seq_capacity =
                metrics.max_num_seqs == 0 || metrics.running_requests < metrics.max_num_seqs;
            if has_block_capacity && has_seq_capacity {
                return Ok(());
            }
            // Wait indefinitely for the next metrics update — no decode-side timer.
            // Cancellation lands via the request context being dropped.
            rx.changed()
                .await
                .map_err(|_| anyhow::anyhow!("scheduler metrics channel closed"))?;
        }
    }

    /// Send a request to the appropriate scheduler, waiting for initialization if needed.
    pub async fn direct(&self, request: DirectRequest, dp_rank: usize) {
        let sender = self.request_sender(dp_rank).await;
        let _ = sender.send(request);
    }

    async fn request_sender(&self, dp_rank: usize) -> mpsc::UnboundedSender<DirectRequest> {
        if let Some(senders) = self.request_senders.get() {
            return senders[dp_rank].clone();
        }

        // Register the waiter *before* re-checking to avoid a TOCTOU race
        // where `start_schedulers` sets + notifies between our check and subscribe.
        let notified = self.senders_ready.notified();
        if let Some(senders) = self.request_senders.get() {
            return senders[dp_rank].clone();
        }
        notified.await;

        let senders = self
            .request_senders
            .get()
            .expect("must be set after notify");
        senders[dp_rank].clone()
    }

    /// Create schedulers and spawn their background tasks for distributing token notifications.
    async fn start_schedulers(
        &self,
        component: Option<&Component>,
        cancel_token: CancellationToken,
        fpm_sinks: Vec<dynamo_mocker::common::protocols::FpmPublisher>,
    ) -> Vec<Box<dyn SchedulerHandle>> {
        let args = &self.engine_args;
        let mut schedulers = Vec::<Box<dyn SchedulerHandle>>::new();
        let mut senders = Vec::with_capacity(args.dp_size as usize);

        for (dp_rank, fpm_publisher) in (0..args.dp_size).zip(fpm_sinks) {
            let (output_tx, mut output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();

            let (kv_event_publishers, relay_publisher): (
                KvEventPublishers,
                Option<KvEventPublisher>,
            ) = match component {
                Some(comp) if args.zmq_kv_events_port.is_some() => {
                    let zmq_port = args.zmq_kv_events_port.unwrap() + dp_rank as u16;
                    let replay_port = args.zmq_replay_port.map(|p| p + dp_rank as u16);
                    match ZmqKvEventSink::new(
                        zmq_port,
                        replay_port,
                        dp_rank,
                        args.block_size as u32,
                    )
                    .await
                    {
                        Ok(sink) => {
                            let source_config = Some(KvEventSourceConfig::Zmq {
                                endpoint: format!("tcp://127.0.0.1:{zmq_port}"),
                                topic: String::new(),
                                image_token_id: None,
                            });
                            match KvEventPublisher::new_with_local_indexer(
                                comp.clone(),
                                args.block_size as u32,
                                source_config,
                                args.enable_local_indexer,
                                dp_rank,
                                None,
                            ) {
                                Ok(publisher) => (
                                    KvEventPublishers::new(
                                        None,
                                        Some(Arc::new(sink) as Arc<dyn RawKvEventSink>),
                                    ),
                                    Some(publisher),
                                ),
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to create KV event relay for dp_rank {dp_rank}: {e}"
                                    );
                                    (KvEventPublishers::default(), None)
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to create ZMQ KV event sink for dp_rank {dp_rank}: {e}"
                            );
                            (KvEventPublishers::default(), None)
                        }
                    }
                }
                Some(comp) => {
                    match KvEventPublisher::new_with_local_indexer(
                        comp.clone(),
                        args.block_size as u32,
                        None,
                        args.enable_local_indexer,
                        dp_rank,
                        None,
                    ) {
                        Ok(publisher) => (
                            KvEventPublishers::new(
                                Some(Arc::new(KvEventSinkAdapter(publisher))
                                    as Arc<dyn KvCacheEventSink>),
                                None,
                            ),
                            None,
                        ),
                        Err(e) => {
                            tracing::error!(
                                "Failed to create KV event publisher for dp_rank {dp_rank}: {e}"
                            );
                            (KvEventPublishers::default(), None)
                        }
                    }
                }
                None => (KvEventPublishers::default(), None),
            };

            let scheduler = create_engine(
                args.clone(),
                dp_rank,
                Some(output_tx),
                kv_event_publishers,
                Some(cancel_token.clone()),
                fpm_publisher,
            );

            senders.push(scheduler.request_sender());
            schedulers.push(scheduler);

            let active_requests_clone = self.active_requests.clone();
            let cancel_token_cloned = cancel_token.clone();

            tokio::spawn(async move {
                // Keep the relay publisher alive for the lifetime of this task.
                // Dropping it would cancel its background ZMQ→NATS relay tasks.
                let _relay_publisher = relay_publisher;

                loop {
                    tokio::select! {
                        signal_result = output_rx.recv() => {
                            let Some(output_batch) = signal_result else {
                                break; // Channel closed
                            };

                            for signal in output_batch {
                                if let Some(request_tx) = active_requests_clone.get(&signal.uuid) {
                                    let _ = request_tx.send(signal);
                                }
                            }
                        }
                        _ = cancel_token_cloned.cancelled() => {
                            tracing::info!("Scheduler output task cancelled, clearing active requests");
                            active_requests_clone.clear();
                            break;
                        }
                    }
                }
            });
        }

        // Set the senders once and notify waiters
        self.request_senders
            .set(senders)
            .expect("Already initialized");
        self.senders_ready.notify_waiters();

        schedulers
    }

    /// Start background tasks to publish metrics on change
    async fn start_metrics_publishing(
        schedulers: &[Box<dyn SchedulerHandle>],
        component: Component,
        native_metrics: Arc<NativeMockerMetrics>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);

        if let Err(e) = metrics_publisher.create_endpoint(component).await {
            tracing::error!("Metrics endpoint failed: {e}");
        }
        for scheduler in schedulers.iter() {
            let mut metrics_rx = scheduler.metrics_receiver();
            let publisher = metrics_publisher.clone();
            let native_metrics = native_metrics.clone();
            let cancel_token = cancel_token.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        // Watch for metrics changes
                        Ok(_) = metrics_rx.changed() => {
                            // Get the latest metrics
                            let metrics = metrics_rx.borrow().clone();
                            native_metrics.update_scheduler_snapshot(&metrics);

                            // Publish metrics using flat API
                            if let Err(e) = publisher.publish(
                                Some(metrics.dp_rank),
                                None,
                                Some(metrics.active_decode_blocks),
                            ) {
                                tracing::warn!("Failed to publish metrics for DP rank {}: {e}", metrics.dp_rank);
                            } else {
                                tracing::debug!(
                                    dp_rank = metrics.dp_rank,
                                    active_decode_blocks = metrics.active_decode_blocks,
                                    total_blocks = metrics.total_blocks,
                                    gpu_cache_usage_perc = metrics.gpu_cache_usage_perc,
                                    "published mocker load metrics"
                                );
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            tracing::debug!("Metrics publishing cancelled");
                            break;
                        }
                    }
                }
            });
        }
        tracing::info!("Metrics background tasks started");
        Ok(())
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LLMEngineOutput>, Error> for MockEngine {
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LLMEngineOutput>, Error> {
        let (request, ctx) = input.into_parts();
        let request_start = Instant::now();

        let dp_rank = self.resolve_dp_rank(&request);

        // Validate dp_rank
        if dp_rank >= self.engine_args.dp_size {
            return Err(Error::msg(format!(
                "dp_rank {} is out of bounds for dp_size {}",
                dp_rank, self.engine_args.dp_size
            )));
        }

        let request_uuid = ctx.id().parse().unwrap_or(Uuid::new_v4());
        let is_prefill = self.engine_args.is_prefill();
        let max_output_tokens = if is_prefill {
            1
        } else {
            request
                .stop_conditions
                .max_tokens
                .ok_or_else(|| Error::msg("max_output_tokens must be specified for mocker"))?
                as usize
        };
        let native_timing = self
            .native_metrics
            .request_timing(&request.model, dp_rank, is_prefill, request_start)
            .await;

        // Disaggregated-serving KV-transfer coordination. The channel and the
        // scheduler pin/release core are shared; only the **keying and source**
        // of the transfer handle differ by path (not by engine — a vLLM mocker
        // can use either path):
        //
        // - **Bootstrap path** (sglang, and vLLM-with-`--bootstrap-ports`): the
        //   frontend precomputes the channel key and plumbs the bootstrap triple
        //   on the request's `bootstrap_info`. UNCHANGED behavior.
        // - **disaggregated_params path** (vLLM NIXL `kv_transfer_params`): the
        //   prefill EMITS the channel key in its output
        //   (`disaggregated_params = {transfer_id, prefill_host, prefill_port}`)
        //   and the decode reads it back from `prefill_result.disaggregated_params`.
        //   Selected only when no `bootstrap_info` is present (the frontend took
        //   the output-`disaggregated_params` route).
        //
        // Both resolve to a single `u64` channel key (the prefill's pin is keyed
        // by request `uuid`; the channel/correlation is keyed by that `u64`).
        let is_vllm = self.is_vllm();
        let bootstrap_server = self.bootstrap_server.clone();
        let abort_timeout = self
            .engine_args
            .kv_transfer_abort_timeout_ms
            .map(Duration::from_millis);

        // Resolve the decode-side connect target (host, port, transfer_id), if any.
        let decode_connect: Option<(String, u16, u64)> = if self.engine_args.is_decode() {
            if let Some(b) = request.bootstrap_info.as_ref() {
                // Bootstrap path (both engines): frontend-precomputed triple.
                Some((b.bootstrap_host.clone(), b.bootstrap_port, b.bootstrap_room))
            } else if is_vllm {
                // vLLM disaggregated_params path: the transfer handle rides on
                // the decode request's `prefill_result.disaggregated_params`,
                // emitted by the prefill.
                request
                    .prefill_result
                    .as_ref()
                    .and_then(|r| VllmDisaggParams::parse(&r.disaggregated_params))
                    .map(|p| (p.prefill_host, p.prefill_port, p.transfer_id))
            } else {
                None
            }
        } else {
            None
        };

        if let Some((host, port, transfer_id)) = &decode_connect {
            // Gate decode on local KV capacity before connecting (see
            // wait_for_decode_kv_capacity). Only when abort_timeout is configured, to
            // preserve the legacy "skip the wait" path for older DGDs.
            if abort_timeout.is_some() {
                self.wait_for_decode_kv_capacity(dp_rank)
                    .await
                    .map_err(|e| Error::msg(format!("Decode KV wait failed: {e}")))?;
            }
            // Forensic logging: emit decode-side wait-start event so post-hoc
            // analysis can reconstruct (decode_worker, target_prefill, transfer)
            // stranding graphs.
            tracing::info!(
                target: "mocker::kv_abort",
                decode_dp_rank = dp_rank,
                transfer_id = transfer_id,
                target_host = %host,
                target_port = port,
                "decode_kv_wait_start"
            );
            connect_to_prefill(host, *port, *transfer_id)
                .await
                .map_err(|e| Error::msg(format!("Bootstrap connection failed: {e}")))?;
        }

        // Resolve the prefill-side transfer id (the `u64` channel key) and, for
        // vLLM, the `disaggregated_params` handle to emit. The prefill pins only
        // when it can hand off over a running bootstrap server, so the strand is
        // always releasable (decode connect or abort-timeout); without a server
        // the prefill frees normally (no strand) — e.g. aggregated requests.
        //
        // The TWO disagg paths differ in WHO drives the pin release, and the
        // difference is intentional — vLLM is the NIXL-faithful reframe:
        //
        // - **sglang bootstrap path** (`bootstrap_info` present): the decode
        //   already knows the frontend-precomputed room, so the prefill blocks
        //   on `wait_for_decode_ready` in the spawned task (KV pinned throughout)
        //   and `complete_room`s after the first token. UNCHANGED.
        // - **vLLM disaggregated_params path**: the prefill emits the transfer
        //   handle in its output and its request stream COMPLETES NORMALLY — it
        //   does NOT hold the stream open waiting for the pull. The KV pin is
        //   decoupled from the request lifecycle (as NixlConnector holds KV
        //   independent of the request): at pin time the prefill REGISTERS the
        //   release trigger on the long-lived channel server, keyed by
        //   `transfer_id`. The server fires the registered `release_pin` when the
        //   matching decode connects (then ACKs it) or on abort-timeout.
        #[allow(clippy::type_complexity)]
        let (prefill_transfer_id, emitted_disagg_params, vllm_disagg_handoff): (
            Option<u64>,
            Option<serde_json::Value>,
            bool,
        ) = if is_prefill {
            if let Some(b) = request.bootstrap_info.as_ref() {
                // sglang bootstrap path: pin keyed by the frontend's precomputed
                // room; the prefill blocks on the pre-emit wait (unchanged).
                (Some(b.bootstrap_room), None, false)
            } else if is_vllm && let Some(server) = bootstrap_server.get() {
                // vLLM disaggregated_params path: generate a transfer id
                // post-compute and announce it in the output. Release is driven
                // by the channel server, not by holding the prefill stream.
                let transfer_id = self
                    .transfer_id_counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // VllmDisaggParams is three primitives; json! is infallible (no expect).
                let value = serde_json::json!({
                    "transfer_id": transfer_id,
                    // The mocker prefill/decode run as local processes; the
                    // bootstrap server binds 0.0.0.0, so loopback is the
                    // reachable pull address for the modeled transfer.
                    "prefill_host": "127.0.0.1",
                    "prefill_port": server.port(),
                });
                (Some(transfer_id), Some(value), true)
            } else {
                (None, None, false)
            }
        } else {
            (None, None, false)
        };

        // Convert PreprocessedRequest to DirectRequest for scheduler.
        //
        // For a disagg prefill with a transfer id, tag the request with
        // `bootstrap_room` (the scheduler's pin trigger — an engine-neutral
        // `u64` regardless of which engine produced it) so the scheduler PINS
        // its KV on prefill completion (keeps the blocks counted active) instead
        // of freeing them immediately.
        let direct_request = DirectRequest {
            tokens: request.token_ids.clone(),
            max_output_tokens,
            uuid: Some(request_uuid),
            dp_rank,
            arrival_timestamp_ms: request.request_timestamp_ms,
            bootstrap_room: prefill_transfer_id,
            ..Default::default()
        };

        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<OutputSignal>();
        self.active_requests.insert(request_uuid, request_tx);

        // ordering fix: submit the prefill to the scheduler FIRST so
        // its KV is genuinely allocated and (on completion) pinned. Previously
        // submission was deferred until AFTER `wait_for_decode_ready` resolved,
        // so the scheduler held no request during the wait — no KV, no
        // pressure, no cascade.
        //
        // - sglang: capture `(server, room_id, sender)` so the spawned task can
        //   run the pre-emit wait that governs the *release* of the pinned KV.
        // - vLLM: REGISTER the release trigger on the channel server here at pin
        //   time, then let the stream complete normally. The server fires the
        //   registered `release_pin` on decode-connect / abort-timeout — the
        //   prefill task does NOT wait. (This is the decoupling that lets the
        //   prefill_router drain the prefill stream and route the decode.)
        let sglang_pin_release = if is_prefill {
            match (bootstrap_server.get().cloned(), prefill_transfer_id) {
                (Some(server), Some(transfer_id)) => {
                    let sender = self.request_sender(dp_rank as usize).await;
                    self.direct(direct_request, dp_rank as usize).await;
                    if vllm_disagg_handoff {
                        // vLLM: register the release on the channel server. The
                        // closure sends the scheduler's `release_pin` for this
                        // prefill `uuid` when the server fires it (decode pull or
                        // abort-timeout). The prefill stream is NOT held open.
                        let pin_start = std::time::Instant::now();
                        tracing::info!(
                            target: "mocker::kv_abort",
                            prefill_dp_rank = dp_rank,
                            transfer_id,
                            "prefill_kv_pin_start"
                        );
                        let release_sender = sender.clone();
                        server.register_pin(
                            transfer_id,
                            abort_timeout,
                            Box::new(move || {
                                // Best-effort: if the channel is closed the
                                // scheduler is shutting down and the KV is moot.
                                let _ = release_sender
                                    .send(DirectRequest::release_pin(request_uuid, dp_rank));
                                tracing::info!(
                                    target: "mocker::kv_abort",
                                    prefill_dp_rank = dp_rank,
                                    transfer_id,
                                    duration_ms = pin_start.elapsed().as_millis() as u64,
                                    "prefill_kv_pin_end"
                                );
                            }),
                        );
                        None
                    } else {
                        // sglang: the spawned task runs the pre-emit wait.
                        Some((server, transfer_id, sender))
                    }
                }
                _ => {
                    self.direct(direct_request, dp_rank as usize).await;
                    None
                }
            }
        } else {
            self.direct(direct_request, dp_rank as usize).await;
            None
        };

        // Create a simple channel for the stream
        let (stream_tx, stream_rx) = mpsc::unbounded_channel::<LLMEngineOutput>();

        let active_requests = self.active_requests.clone();
        let async_context = ctx.context();
        let reasoning = self.engine_args.reasoning.clone();
        let mut native_timing = native_timing;
        // The `disaggregated_params` a prefill emits in its output:
        // - vLLM (disagg): the real `{transfer_id, prefill_host, prefill_port}`
        //   handle the decode reads back. The decode connects by `transfer_id`.
        // - sglang (disagg) / any other prefill: the legacy opaque marker,
        //   unchanged — sglang coordinates over the frontend-precomputed room.
        let prefill_disagg_params: Option<serde_json::Value> = if is_prefill {
            emitted_disagg_params.or_else(|| Some(serde_json::json!("dummy")))
        } else {
            None
        };

        // Spawn a task to handle the complex async logic
        tokio::spawn(async move {
            // sglang bootstrap path: the decode already knows the
            // frontend-precomputed room, so it connects *before* the prefill
            // emits. The prefill request is ALREADY submitted (above), so its KV
            // is pinned on completion; this wait is the *release* trigger — the
            // strand persists for exactly as long as the decode takes to arrive
            // (event-driven). The vLLM disagg path does NOT reach here: its
            // release is driven by the channel server (registered above) so the
            // stream completes normally.
            if let Some((server, room_id, sender)) = sglang_pin_release {
                let pin_start = std::time::Instant::now();
                tracing::info!(
                    target: "mocker::kv_abort",
                    prefill_dp_rank = dp_rank,
                    room_id,
                    "prefill_kv_pin_start"
                );
                tokio::select! {
                    result = server.wait_for_decode_ready(room_id, abort_timeout) => {
                        let outcome = if let Err(e) = result {
                            tracing::warn!(
                                "Prefill aborting transfer for room {room_id}: {e}"
                            );
                            server.abort_room(room_id);
                            Some(e)
                        } else {
                            None
                        };
                        let _ = sender.send(DirectRequest::release_pin(request_uuid, dp_rank));
                        tracing::info!(
                            target: "mocker::kv_abort",
                            prefill_dp_rank = dp_rank,
                            room_id,
                            outcome = if outcome.is_some() { "aborted" } else { "completed" },
                            duration_ms = pin_start.elapsed().as_millis() as u64,
                            "prefill_kv_pin_end"
                        );
                        if let Some(e) = outcome {
                            let _ = stream_tx.send(LLMEngineOutput::error(format!(
                                "NIXL transfer aborted: {e}"
                            )));
                            active_requests.remove(&request_uuid);
                            return;
                        }
                    }
                    _ = async_context.stopped() => {
                        let _ = sender.send(DirectRequest::release_pin(request_uuid, dp_rank));
                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                        active_requests.remove(&request_uuid);
                        return;
                    }
                }
            }

            let mut token_count = 0;
            let think_len = reasoning
                .as_ref()
                .map(|cfg| cfg.num_thinking_tokens(max_output_tokens))
                .unwrap_or(0);

            loop {
                tokio::select! {
                    maybe_signal = request_rx.recv() => {
                        let Some(signal) = maybe_signal else {
                            let _ = stream_tx.send(LLMEngineOutput::error("All output transmitters closed".to_string()));
                            break;
                        };

                        // A terminally rejected request never ran (its footprint
                        // exceeds the KV pool): emit no token and do not complete the
                        // bootstrap room — surface the rejection and end the stream
                        // before any token/prefill bookkeeping.
                        if signal.rejected {
                            let _ = stream_tx.send(LLMEngineOutput::error(
                                "request rejected: KV footprint exceeds pool capacity".to_string(),
                            ));
                            break;
                        }

                        // Generate a token (with thinking boundaries if configured)
                        let token_id = if token_count == 0 && think_len > 0 {
                            reasoning.as_ref().unwrap().start_thinking_token_id
                        } else if think_len > 0 && token_count == think_len - 1 {
                            reasoning.as_ref().unwrap().end_thinking_token_id
                        } else {
                            generate_random_token()
                        };
                        token_count += 1;

                        let output = LLMEngineOutput {
                            token_ids: vec![token_id],
                            disaggregated_params: prefill_disagg_params.clone(),
                            ..Default::default()
                        };

                        if signal.completed && token_count < max_output_tokens {
                            let _ = stream_tx.send(LLMEngineOutput::error("Completion signal received before max tokens reached".to_string()));
                            break;
                        }

                        if signal.completed {
                            // Emit the prefill's completion token. For vLLM disagg
                            // this carries the real `disaggregated_params` handle
                            // the decode reads back; the stream then COMPLETES
                            // normally (no pin-wait here — release is driven by the
                            // channel server, registered at submit time). sglang
                            // already waited pre-emit (KV stayed pinned) and ACKs
                            // the decode (`complete_room`) below — unchanged.
                            if stream_tx.send(output).is_err() {
                                tracing::error!("Output stream receiver closed.");
                                break;
                            }
                            native_timing.record_tokens(1);

                            // Prefill-to-decode handoff delay is emitted by the shared mocker core.
                            if is_prefill
                                && let Some(delay_ms) = signal.handoff_delay_ms
                            {
                                sleep_precise(Duration::from_secs_f64(delay_ms / 1000.0)).await;
                            }

                            // sglang bootstrap path: after first token, mark the
                            // room complete (unblocks the decode). The vLLM disagg
                            // path already marked the room complete in
                            // `register_pin`, so it is skipped here (its release is
                            // server-driven, decoupled from the stream).
                            if is_prefill
                                && !vllm_disagg_handoff
                                && let (Some(server), Some(transfer_id)) =
                                    (bootstrap_server.get(), prefill_transfer_id)
                            {
                                server.complete_room(transfer_id);
                            }

                            if stream_tx.send(LLMEngineOutput::length()).is_err() {
                                tracing::error!("Output stream receiver closed.");
                                break;
                            }
                            native_timing.record_normal_completion();
                            break;
                        }

                        if stream_tx.send(output).is_err() {
                            tracing::error!("Output stream receiver closed.");
                            break;
                        }
                        native_timing.record_tokens(1);
                    }

                    _ = async_context.stopped() => {
                        let _ = stream_tx.send(LLMEngineOutput::cancelled());
                        break;
                    }
                }
            }

            active_requests.remove(&request_uuid);
        });

        let stream = UnboundedReceiverStream::new(stream_rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

pub struct AnnotatedMockEngine {
    inner: Arc<MockEngine>,
}

impl AnnotatedMockEngine {
    pub fn new(
        inner: MockEngine,
        distributed_runtime: DistributedRuntime,
        endpoint_id: dynamo_runtime::protocols::EndpointId,
    ) -> Self {
        let inner = Arc::new(inner);
        let inner_clone = inner.clone();

        // Start background task to wait for component service and start the engine
        let cancel_token = distributed_runtime.primary_token();
        tokio::spawn(async move {
            let component = loop {
                if cancel_token.is_cancelled() {
                    tracing::debug!("Mocker engine startup cancelled");
                    return;
                }

                let ready = distributed_runtime
                    .namespace(&endpoint_id.namespace)
                    .and_then(|ns| ns.component(&endpoint_id.component))
                    .ok();

                if let Some(comp) = ready
                    && let Ok(instances) = comp.list_instances().await
                    && !instances.is_empty()
                {
                    break comp;
                }

                tracing::debug!("Component service not available yet, retrying...");
                tokio::time::sleep(Duration::from_millis(100)).await;
            };

            tracing::debug!("Component service is now available, starting mocker engine");
            if let Err(e) = inner_clone.start(component).await {
                tracing::error!("Failed to start mocker engine: {e}");
            }
        });

        Self { inner }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for AnnotatedMockEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let stream = self.inner.generate(input).await?;
        let context = stream.context();

        // Convert stream of LLMEngineOutput to Annotated<LLMEngineOutput>
        let annotated_stream = stream.map(Annotated::from_data);

        Ok(ResponseStream::new(Box::pin(annotated_stream), context))
    }
}

/// Create a mocker engine as ExecutionContext
pub async fn make_mocker_engine(
    distributed_runtime: DistributedRuntime,
    endpoint_id: dynamo_runtime::protocols::EndpointId,
    args: MockEngineArgs,
) -> Result<ExecutionContext, Error> {
    // Create the mocker engine
    tracing::info!("Creating mocker engine with config: {args:?}");
    let annotated_engine =
        AnnotatedMockEngine::new(MockEngine::new(args), distributed_runtime, endpoint_id);

    Ok(Arc::new(annotated_engine))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vLLM disagg handle the prefill emits must round-trip through the
    /// `disaggregated_params` JSON the decode reads back, so the decode can
    /// recover `(prefill_host, prefill_port, transfer_id)` and connect.
    #[test]
    fn vllm_disagg_params_round_trip() {
        let params = VllmDisaggParams {
            transfer_id: 0xDEAD_BEEF_u64,
            prefill_host: "127.0.0.1".to_string(),
            prefill_port: 51234,
        };
        let value = serde_json::to_value(&params).unwrap();
        // It is a structured object (vLLM `kv_transfer_params` shape), not the
        // legacy opaque marker.
        assert!(value.get("transfer_id").is_some());
        assert!(value.get("prefill_host").is_some());
        assert!(value.get("prefill_port").is_some());

        let parsed = VllmDisaggParams::parse(&value).expect("round-trips");
        assert_eq!(parsed.transfer_id, 0xDEAD_BEEF_u64);
        assert_eq!(parsed.prefill_host, "127.0.0.1");
        assert_eq!(parsed.prefill_port, 51234);
    }

    /// A request carrying no (or a non-conforming) `disaggregated_params`
    /// yields no decode connect target — the decode does not strand waiting on
    /// a transfer that was never announced (aggregated vLLM path).
    #[test]
    fn vllm_disagg_params_absent_or_malformed_parse_to_none() {
        assert!(VllmDisaggParams::parse(&serde_json::json!("dummy")).is_none());
        assert!(VllmDisaggParams::parse(&serde_json::json!({})).is_none());
        assert!(
            VllmDisaggParams::parse(&serde_json::json!({ "transfer_id": 1 })).is_none(),
            "missing host/port must not parse"
        );
    }
}
