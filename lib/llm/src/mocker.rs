// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mocker module - runtime integration for the mock scheduler.
//!
//! The core mocker logic lives in the `dynamo-mocker` crate.
//! This module provides the runtime-dependent engine wrapper.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use futures::StreamExt;
use rand::Rng;
use serde::Serialize;
use tokio::sync::{Notify, OnceCell, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use zeromq::{Socket, SocketSend};

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{
    component::Component,
    engine::AsyncEngineContextProvider,
    pipeline::{AsyncEngine, Error, ManyOut, ResponseStream, SingleIn, async_trait},
    traits::DistributedRuntimeProvider,
};

use crate::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig, WorkerMetricsPublisher};
use crate::protocols::TokenIdType;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};

// Re-export from dynamo-mocker for convenience
use dynamo_mocker::common::bootstrap::{BootstrapServer, connect_to_prefill};
use dynamo_mocker::common::protocols::OutputSignal;
pub use dynamo_mocker::common::protocols::{
    DirectRequest, KvCacheEventSink, MockEngineArgs, MockEngineArgsBuilder,
};
use dynamo_mocker::common::utils::{compute_kv_transfer_delay, sleep_precise};
pub use dynamo_mocker::common::{bootstrap, perf_model, protocols, running_mean, sequence};
pub use dynamo_mocker::scheduler::Scheduler;
pub use dynamo_mocker::{kv_manager, scheduler};

pub const MOCKER_COMPONENT: &str = "mocker";

/// Wrapper to adapt KvEventPublisher to the KvCacheEventSink trait
struct KvEventSinkAdapter(KvEventPublisher);

impl KvCacheEventSink for KvEventSinkAdapter {
    fn publish(
        &self,
        event: KvCacheEvent,
        _block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        self.0
            .publish(event)
            .map_err(|e| anyhow::anyhow!("Failed to send KV event: {}", e))
    }
}

// ---------------------------------------------------------------------------
// ZMQ KV event publishing (vLLM native wire format)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(tag = "type")]
enum ZmqRawKvEvent {
    BlockStored {
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<u32>,
        block_size: u32,
    },
    BlockRemoved {
        block_hashes: Vec<u64>,
    },
}

struct ZmqKvEventMsg {
    event: KvCacheEvent,
    block_token_ids: Option<Vec<Vec<u32>>>,
}

struct ZmqKvEventSink {
    tx: mpsc::UnboundedSender<ZmqKvEventMsg>,
}

impl ZmqKvEventSink {
    async fn new(port: u16, dp_rank: u32, block_size: u32) -> Result<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel::<ZmqKvEventMsg>();

        // Bind the PUB socket before returning so that any SUB connect()
        // that follows is guaranteed to find the endpoint already listening.
        let mut pub_socket = zeromq::PubSocket::new();
        let endpoint = format!("tcp://0.0.0.0:{port}");
        pub_socket
            .bind(&endpoint)
            .await
            .map_err(|e| anyhow::anyhow!("ZMQ PUB bind to {endpoint} failed: {e}"))?;
        tracing::info!("ZmqKvEventSink bound to {endpoint} for dp_rank {dp_rank}");

        tokio::spawn(async move {
            let mut seq_num: u64 = 0;

            while let Some(msg) = rx.recv().await {
                let events =
                    convert_to_zmq_events(&msg.event, msg.block_token_ids.as_deref(), block_size);
                if events.is_empty() {
                    continue;
                }

                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();

                let batch: (f64, Vec<ZmqRawKvEvent>, Option<i32>) =
                    (timestamp, events, Some(dp_rank as i32));
                let payload = match rmp_serde::to_vec(&batch) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!("Failed to serialize ZMQ KV event: {e}");
                        continue;
                    }
                };

                let frames = vec![
                    Bytes::from(""),
                    Bytes::from(seq_num.to_be_bytes().to_vec()),
                    Bytes::from(payload),
                ];
                let zmq_msg = zeromq::ZmqMessage::try_from(frames)
                    .expect("Failed to create ZMQ multipart message");

                if let Err(e) = pub_socket.send(zmq_msg).await {
                    tracing::warn!("Failed to send ZMQ KV event: {e}");
                }

                seq_num += 1;
            }
        });

        Ok(Self { tx })
    }
}

impl KvCacheEventSink for ZmqKvEventSink {
    fn publish(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        self.tx
            .send(ZmqKvEventMsg {
                event,
                block_token_ids: block_token_ids.map(|t| t.to_vec()),
            })
            .map_err(|_| anyhow::anyhow!("ZMQ event sink channel closed"))
    }
}

fn convert_to_zmq_events(
    event: &KvCacheEvent,
    block_token_ids: Option<&[Vec<u32>]>,
    block_size: u32,
) -> Vec<ZmqRawKvEvent> {
    match &event.data {
        KvCacheEventData::Stored(store_data) => {
            let block_hashes: Vec<u64> = store_data.blocks.iter().map(|b| b.block_hash.0).collect();
            let parent_block_hash = store_data.parent_hash.map(|h| h.0);

            let token_ids: Vec<u32> = block_token_ids
                .map(|tids| tids.iter().flatten().copied().collect())
                .unwrap_or_default();

            assert_eq!(
                token_ids.len(),
                block_hashes.len() * block_size as usize,
                "token_ids length ({}) must equal block_hashes.len() ({}) * block_size ({block_size})",
                token_ids.len(),
                block_hashes.len(),
            );

            vec![ZmqRawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                token_ids,
                block_size,
            }]
        }
        KvCacheEventData::Removed(remove_data) => {
            let block_hashes: Vec<u64> = remove_data.block_hashes.iter().map(|h| h.0).collect();
            vec![ZmqRawKvEvent::BlockRemoved { block_hashes }]
        }
        KvCacheEventData::Cleared => vec![],
    }
}

fn generate_random_token() -> TokenIdType {
    let mut rng = rand::rng();
    rng.random_range(1000..2000)
}

/// AsyncEngine wrapper around the Scheduler that generates random character tokens
pub struct MockVllmEngine {
    active_requests: Arc<DashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>,
    request_senders: OnceCell<Vec<mpsc::UnboundedSender<DirectRequest>>>,
    senders_ready: Notify,
    engine_args: MockEngineArgs,
    /// Bootstrap server for prefill workers in disaggregated mode
    bootstrap_server: Arc<OnceCell<Arc<BootstrapServer>>>,
    /// Keep schedulers alive so their CancelGuards don't fire prematurely.
    _schedulers: OnceCell<Vec<Scheduler>>,
}

impl MockVllmEngine {
    /// Create a new MockVllmEngine with the given parameters
    pub fn new(engine_args: MockEngineArgs) -> Self {
        Self {
            active_requests: Arc::new(DashMap::new()),
            request_senders: OnceCell::new(),
            senders_ready: Notify::new(),
            engine_args,
            bootstrap_server: Arc::new(OnceCell::new()),
            _schedulers: OnceCell::new(),
        }
    }

    pub async fn start(&self, component: Component) -> Result<()> {
        // Use primary_token() instead of child_token() so the mocker continues running
        // during graceful shutdown (Phase 1/2) and only stops in Phase 3.
        // child_token() is a child of endpoint_shutdown_token which is cancelled in Phase 1.
        // primary_token() is only cancelled in Phase 3, after waiting for inflight requests.
        let cancel_token = component.drt().primary_token();

        // Simulate engine startup time if configured
        if let Some(startup_time_secs) = self.engine_args.startup_time {
            tracing::info!("Simulating engine startup time: {:.2}s", startup_time_secs);
            tokio::time::sleep(Duration::from_secs_f64(startup_time_secs)).await;
            tracing::info!("Engine startup simulation completed");
        }

        // Start bootstrap server for prefill workers in disaggregated mode
        if self.engine_args.is_prefill()
            && let Some(port) = self.engine_args.bootstrap_port
        {
            let server = BootstrapServer::start(port, cancel_token.clone()).await?;
            let _ = self.bootstrap_server.set(server);
            tracing::info!(port = port, "Bootstrap server started for prefill worker");
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

        let schedulers = self
            .start_schedulers(kv_component, cancel_token.clone())
            .await;

        Self::start_metrics_publishing(&schedulers, component, cancel_token.clone()).await?;

        let _ = self._schedulers.set(schedulers);

        Ok(())
    }

    /// Send a request to the appropriate scheduler, waiting for initialization if needed.
    pub async fn direct(&self, request: DirectRequest, dp_rank: usize) {
        if let Some(senders) = self.request_senders.get() {
            let _ = senders[dp_rank].send(request);
            return;
        }

        // Register the waiter *before* re-checking to avoid a TOCTOU race
        // where `start_schedulers` sets + notifies between our check and subscribe.
        let notified = self.senders_ready.notified();
        if let Some(senders) = self.request_senders.get() {
            let _ = senders[dp_rank].send(request);
            return;
        }
        notified.await;

        let senders = self
            .request_senders
            .get()
            .expect("must be set after notify");
        let _ = senders[dp_rank].send(request);
    }

    /// Create schedulers and spawn their background tasks for distributing token notifications
    async fn start_schedulers(
        &self,
        component: Option<&Component>,
        cancel_token: CancellationToken,
    ) -> Vec<Scheduler> {
        let args = &self.engine_args;
        let mut schedulers = Vec::<Scheduler>::new();
        let mut senders = Vec::with_capacity(args.dp_size as usize);

        for dp_rank in 0..args.dp_size {
            let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

            let (kv_event_sink, relay_publisher): (
                Option<Arc<dyn KvCacheEventSink>>,
                Option<KvEventPublisher>,
            ) = match component {
                Some(comp) if args.zmq_kv_events_port.is_some() => {
                    let zmq_port = args.zmq_kv_events_port.unwrap() + dp_rank as u16;
                    match ZmqKvEventSink::new(zmq_port, dp_rank, args.block_size as u32).await {
                        Ok(sink) => {
                            let source_config = Some(KvEventSourceConfig::Zmq {
                                endpoint: format!("tcp://127.0.0.1:{zmq_port}"),
                                topic: String::new(),
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
                                    Some(Arc::new(sink) as Arc<dyn KvCacheEventSink>),
                                    Some(publisher),
                                ),
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to create KV event relay for dp_rank {dp_rank}: {e}"
                                    );
                                    (None, None)
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to create ZMQ KV event sink for dp_rank {dp_rank}: {e}"
                            );
                            (None, None)
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
                            Some(Arc::new(KvEventSinkAdapter(publisher))
                                as Arc<dyn KvCacheEventSink>),
                            None,
                        ),
                        Err(e) => {
                            tracing::error!(
                                "Failed to create KV event publisher for dp_rank {dp_rank}: {e}"
                            );
                            (None, None)
                        }
                    }
                }
                None => (None, None),
            };

            let scheduler = Scheduler::new(
                args.clone(),
                dp_rank,
                Some(output_tx),
                kv_event_sink,
                Some(cancel_token.clone()),
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
                            let Some(signal) = signal_result else {
                                break; // Channel closed
                            };

                            if let Some(request_tx) = active_requests_clone.get(&signal.uuid) {
                                let _ = request_tx.send(signal);
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
        schedulers: &[Scheduler],
        component: Component,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);

        if let Err(e) = metrics_publisher.create_endpoint(component).await {
            tracing::error!("Metrics endpoint failed: {e}");
        }
        for scheduler in schedulers.iter() {
            let mut metrics_rx = scheduler.metrics_receiver();
            let publisher = metrics_publisher.clone();
            let cancel_token = cancel_token.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        // Watch for metrics changes
                        Ok(_) = metrics_rx.changed() => {
                            // Get the latest metrics
                            let metrics = metrics_rx.borrow().clone();

                            // Publish metrics using flat API
                            if let Err(e) = publisher.publish(Some(metrics.dp_rank), metrics.active_decode_blocks) {
                                tracing::warn!("Failed to publish metrics for DP rank {}: {e}", metrics.dp_rank);
                            } else {
                                tracing::trace!("Published metrics for DP rank {}", metrics.dp_rank);
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
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LLMEngineOutput>, Error>
    for MockVllmEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LLMEngineOutput>, Error> {
        let (request, ctx) = input.into_parts();

        // Extract dp_rank from routing hints (defaults to 0 if not set)
        let dp_rank = request
            .routing
            .as_ref()
            .and_then(|r| r.dp_rank)
            .unwrap_or(0);

        // Validate dp_rank
        if dp_rank >= self.engine_args.dp_size {
            return Err(Error::msg(format!(
                "dp_rank {} is out of bounds for dp_size {}",
                dp_rank, self.engine_args.dp_size
            )));
        }

        // Bootstrap rendezvous for disaggregated serving
        // - Decode: connect to prefill's server, block until prefill completes
        // - Prefill: complete_room() is called after first token (see below)
        let bootstrap_room = request.bootstrap_info.as_ref().map(|b| b.bootstrap_room);
        if let Some(bootstrap_info) = &request.bootstrap_info
            && self.engine_args.is_decode()
        {
            connect_to_prefill(
                &bootstrap_info.bootstrap_host,
                bootstrap_info.bootstrap_port,
                bootstrap_info.bootstrap_room,
            )
            .await
            .map_err(|e| Error::msg(format!("Bootstrap connection failed: {e}")))?;
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

        // Convert PreprocessedRequest to DirectRequest for scheduler
        let direct_request = DirectRequest {
            tokens: request.token_ids.clone(),
            max_output_tokens,
            uuid: Some(request_uuid),
            dp_rank,
        };

        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<OutputSignal>();
        self.active_requests.insert(request_uuid, request_tx);

        // Send the request to the appropriate scheduler based on dp_rank
        self.direct(direct_request, dp_rank as usize).await;

        // Create a simple channel for the stream
        let (stream_tx, stream_rx) = mpsc::unbounded_channel::<LLMEngineOutput>();

        let active_requests = self.active_requests.clone();
        let async_context = ctx.context();
        let bootstrap_server = self.bootstrap_server.clone();
        let reasoning = self.engine_args.reasoning.clone();

        // Compute KV transfer delay for prefill workers.
        // Simulates the time to transfer KV cache from prefill to decode worker.
        let kv_transfer_delay = if is_prefill {
            compute_kv_transfer_delay(&self.engine_args, request.token_ids.len())
        } else {
            None
        };

        // Spawn a task to handle the complex async logic
        tokio::spawn(async move {
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
                            disaggregated_params: is_prefill.then(|| serde_json::json!("dummy")),
                            ..Default::default()
                        };

                        if signal.completed && token_count < max_output_tokens {
                            let _ = stream_tx.send(LLMEngineOutput::error("Completion signal received before max tokens reached".to_string()));
                            break;
                        }

                        if signal.completed {
                            let _ = stream_tx.send(output);

                            // Simulate KV transfer delay before prefill's first (and only) token.
                            // This models the time to transfer KV cache to the decode worker.
                            if token_count == 1
                                && let Some(delay) = kv_transfer_delay
                            {
                                sleep_precise(delay).await;
                            }

                            // Prefill: after first token, mark room complete (unblocks decode)
                            if is_prefill
                                && token_count == 1
                                && let (Some(server), Some(room_id)) = (bootstrap_server.get(), bootstrap_room)
                            {
                                server.complete_room(room_id);
                            }

                            let _ = stream_tx.send(LLMEngineOutput::length());
                            break;
                        }

                        if stream_tx.send(output).is_err() {
                            tracing::error!("Output stream receiver closed.");
                            break;
                        }
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
    inner: Arc<MockVllmEngine>,
}

impl AnnotatedMockEngine {
    pub fn new(
        inner: MockVllmEngine,
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
) -> Result<crate::backend::ExecutionContext, Error> {
    // Create the mocker engine
    tracing::info!("Creating mocker engine with config: {args:?}");
    let annotated_engine =
        AnnotatedMockEngine::new(MockVllmEngine::new(args), distributed_runtime, endpoint_id);

    Ok(Arc::new(annotated_engine))
}
