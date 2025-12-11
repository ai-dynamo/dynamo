// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MockSchedulerEngine - AsyncEngine wrapper around the Scheduler
//!
//! This module provides an AsyncEngine implementation that wraps the Scheduler
//! to provide streaming token generation with realistic timing simulation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{Mutex, OnceCell, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::{
    component::Component,
    engine::AsyncEngineContextProvider,
    pipeline::{AsyncEngine, Error, ManyOut, ResponseStream, SingleIn, async_trait},
    traits::DistributedRuntimeProvider,
};

use crate::kv_router::publisher::WorkerMetricsPublisher;
use crate::mocker::protocols::DirectRequest;
use crate::mocker::protocols::{MockEngineArgs, OutputSignal, WorkerType};
use crate::mocker::scheduler::Scheduler;
use crate::protocols::TokenIdType;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

pub const MOCKER_COMPONENT: &str = "mocker";

fn generate_random_token() -> TokenIdType {
    let mut rng = rand::rng();
    rng.random_range(1000..2000)
}

/// AsyncEngine wrapper around the Scheduler that generates random character tokens
#[derive(Clone)]
pub struct MockVllmEngine {
    active_requests: Arc<Mutex<HashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>>,
    request_senders: Arc<OnceCell<Vec<mpsc::UnboundedSender<DirectRequest>>>>,
    engine_args: MockEngineArgs,
}

impl MockVllmEngine {
    /// Create a new MockVllmEngine with the given parameters
    pub fn new(args: MockEngineArgs) -> Self {
        Self {
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            request_senders: Arc::new(OnceCell::new()),
            engine_args: args,
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

        // Pass component to schedulers only if prefix caching is enabled and not a decode worker
        let scheduler_component = if self.engine_args.enable_prefix_caching
            && self.engine_args.worker_type != WorkerType::Decode
        {
            Some(component.clone())
        } else {
            None
        };

        let schedulers = self.start_schedulers(
            self.engine_args.clone(),
            self.active_requests.clone(),
            scheduler_component,
            cancel_token.clone(),
        );

        Self::start_metrics_publishing(&schedulers, Some(component.clone()), cancel_token.clone())
            .await?;

        Ok(())
    }

    pub fn direct(&self, request: DirectRequest, dp_rank: usize) {
        let senders = self.request_senders.get().expect("Not initialized");
        let _ = senders[dp_rank].send(request);
    }

    /// Create schedulers and spawn their background tasks for distributing token notifications
    fn start_schedulers(
        &self,
        args: MockEngineArgs,
        active_requests: Arc<Mutex<HashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>>,
        component: Option<Component>,
        cancel_token: CancellationToken,
    ) -> Vec<Scheduler> {
        let mut schedulers = Vec::<Scheduler>::new();
        let mut senders = Vec::with_capacity(args.dp_size as usize);

        // Create multiple schedulers and their background tasks
        for dp_rank in 0..args.dp_size {
            // Create a shared output channel that this scheduler will use
            let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

            let scheduler = Scheduler::new(
                args.clone(),
                dp_rank,
                Some(output_tx),
                component.clone(),
                Some(cancel_token.clone()),
            );

            senders.push(scheduler.request_sender());
            schedulers.push(scheduler);

            // Spawn a background task for this scheduler to distribute token notifications to active requests
            // let output_rx = Arc::new(Mutex::new(output_rx));
            let active_requests_clone = active_requests.clone();
            let cancel_token_cloned = cancel_token.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        signal_result = output_rx.recv() => {
                            let Some(signal) = signal_result else {
                                break; // Channel closed
                            };

                            // Notify the specific request that a token was generated
                            let active = active_requests_clone.lock().await;
                            if let Some(request_tx) = active.get(&signal.uuid) {
                                let _ = request_tx.send(signal);
                            }
                        }
                        _ = cancel_token_cloned.cancelled() => {
                            tracing::info!("Scheduler output task cancelled, clearing active requests");
                            // Clear all active requests to unblock waiting request handlers
                            // This will cause their request_rx.recv() to return None
                            let mut active = active_requests_clone.lock().await;
                            active.clear();
                            break;
                        }
                    }
                }
            });
        }

        // Set the senders once
        self.request_senders
            .set(senders)
            .expect("Already initialized");

        schedulers
    }

    /// Start background tasks to publish metrics on change
    async fn start_metrics_publishing(
        schedulers: &[Scheduler],
        component: Option<Component>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        tracing::debug!("Creating metrics publisher");
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);
        tracing::debug!("Metrics publisher created");

        if let Some(comp) = component {
            tracing::debug!("Creating metrics endpoint");
            tokio::spawn({
                let publisher = metrics_publisher.clone();
                async move {
                    if let Err(e) = publisher.create_endpoint(comp.clone()).await {
                        tracing::error!("Metrics endpoint failed: {e}");
                    }
                }
            });

            // Give it a moment to start
            tokio::time::sleep(Duration::from_millis(100)).await;
            tracing::debug!("Metrics endpoint started (background)");
        }

        tracing::debug!("Starting metrics background tasks");
        for (dp_rank, scheduler) in schedulers.iter().enumerate() {
            let mut metrics_rx = scheduler.metrics_receiver();
            let publisher = metrics_publisher.clone();
            let dp_rank = dp_rank as u32;
            let cancel_token = cancel_token.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        // Watch for metrics changes
                        Ok(_) = metrics_rx.changed() => {
                            // Get the latest metrics
                            let metrics = metrics_rx.borrow().clone();

                            // Publish metrics
                            if let Err(e) = publisher.publish(Arc::new(metrics)) {
                                tracing::warn!("Failed to publish metrics for DP rank {dp_rank}: {e}");
                            } else {
                                tracing::trace!("Published metrics for DP rank {}", dp_rank);
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            tracing::debug!("Metrics publishing cancelled for DP rank {dp_rank}");
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

        // Extract dp_rank from request field (defaults to 0 if not set)
        let dp_rank = request.dp_rank.unwrap_or(0);

        // Validate dp_rank
        if dp_rank >= self.engine_args.dp_size {
            return Err(Error::msg(format!(
                "dp_rank {} is out of bounds for dp_size {}",
                dp_rank, self.engine_args.dp_size
            )));
        }

        let request_uuid = ctx.id().parse().unwrap_or(Uuid::new_v4());

        // For prefill workers, override max_tokens to 1
        let is_prefill = self.engine_args.worker_type == WorkerType::Prefill;
        let max_output_tokens = if is_prefill {
            1
        } else {
            request
                .stop_conditions
                .max_tokens
                .expect("max_output_tokens must be specified for mocker") as usize
        };

        // Convert PreprocessedRequest to DirectRequest for scheduler
        let direct_request = DirectRequest {
            tokens: request.token_ids.clone(),
            max_output_tokens,
            uuid: Some(request_uuid),
            dp_rank,
        };

        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<OutputSignal>();
        {
            let mut active = self.active_requests.lock().await;
            active.insert(request_uuid, request_tx);
        }

        // Send the request to the appropriate scheduler based on dp_rank
        self.direct(direct_request, dp_rank as usize);

        // Create a simple channel for the stream
        let (stream_tx, stream_rx) = mpsc::unbounded_channel::<LLMEngineOutput>();

        let active_requests = self.active_requests.clone();
        let async_context = ctx.context();

        // Spawn a task to handle the complex async logic
        tokio::spawn(async move {
            let mut token_count = 0;

            loop {
                tokio::select! {
                    maybe_signal = request_rx.recv() => {
                        let Some(signal) = maybe_signal else {
                            let _ = stream_tx.send(LLMEngineOutput::error("All output transmitters closed".to_string()));
                            break;
                        };

                        // Generate a new token
                        let token_id = generate_random_token();
                        token_count += 1;

                        let output = LLMEngineOutput {
                            token_ids: vec![token_id],
                            tokens: None,  // Let backend handle detokenization
                            text: None,
                            cum_log_probs: None,
                            log_probs: None,
                            top_logprobs: None,
                            finish_reason: None,
                            index: None,
                            // Add dummy disaggregated_params for prefill workers
                            disaggregated_params: if is_prefill {
                                Some(serde_json::json!("dummy"))
                            } else {
                                None
                            },
                            extra_args: None,
                            completion_usage: None,
                        };

                        if signal.completed && token_count < max_output_tokens {
                            let _ = stream_tx.send(LLMEngineOutput::error("Completion signal received before max tokens reached".to_string()));
                            break;
                        }

                        if signal.completed {
                            let _ = stream_tx.send(output);
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

            // Clean up: remove this request from active requests
            let mut active = active_requests.lock().await;
            active.remove(&request_uuid);
        });

        // Create a simple UnboundedReceiverStream which is naturally Send + Sync
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
        tokio::spawn(async move {
            loop {
                // Try to create component
                let Ok(namespace) = distributed_runtime.namespace(&endpoint_id.namespace) else {
                    tracing::debug!("Namespace not available yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                };

                let Ok(component) = namespace.component(&endpoint_id.component) else {
                    tracing::debug!("Component not available yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                };

                // Check if service is available by trying to list instances
                let Ok(instances) = component.list_instances().await else {
                    tracing::debug!("Cannot list instances yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                };

                if instances.is_empty() {
                    tracing::debug!("No instances available yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }

                tracing::debug!("Component service is now available, starting mocker engine");

                // Start the engine with the component
                if let Err(e) = inner_clone.start(component).await {
                    tracing::error!("Failed to start mocker engine: {e}");
                }
                break;
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
