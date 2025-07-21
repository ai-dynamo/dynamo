// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_runtime::{
    component::{Component, InstanceSource},
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        ResponseStream, SingleIn,
    },
    prelude::*,
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tokio::sync::Mutex;

pub mod approx;
pub mod indexer;
pub mod metrics_aggregator;
pub mod protocols;
pub mod publisher;
pub mod recorder;
pub mod scheduler;
pub mod scoring;
pub mod sequence;

use crate::{
    kv_router::{
        approx::ApproxKvIndexer,
        indexer::{
            compute_block_hash_for_seq, KvIndexer, KvIndexerInterface, KvRouterError,
            OverlapScores, RouterEvent,
        },
        metrics_aggregator::EndpointCollector,
        protocols::{LocalBlockHash, RouterRequest, RouterResponse, WorkerSelectionResult},
        scheduler::{KvScheduler, KvSchedulerError, SchedulingRequest},
        scoring::ProcessedEndpoints,
    },
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
};

use dynamo_runtime::traits::events::EventSubscriber;

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component
pub const KV_EVENT_SUBJECT: &str = "kv_events";
pub const KV_HIT_RATE_SUBJECT: &str = "kv-hit-rate";
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

/// A trait that users can implement to define custom selection logic
pub trait WorkerSelector {
    fn select_worker(
        &self,
        workers: &ProcessedEndpoints,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

/// KV Router configuration parameters
#[derive(Debug, Clone, Copy)]
pub struct KvRouterConfig {
    pub overlap_score_weight: f64,

    pub router_temperature: f64,

    pub use_kv_events: bool,

    // note: this is not actually used for now
    pub max_num_batched_tokens: u32,
}

impl Default for KvRouterConfig {
    fn default() -> Self {
        Self {
            overlap_score_weight: 1.0,
            router_temperature: 0.0,
            use_kv_events: true,
            max_num_batched_tokens: 8192,
        }
    }
}

impl KvRouterConfig {
    /// Create a new KvRouterConfig with optional weight values.
    /// If a weight is None, the default value will be used.
    pub fn new(
        overlap_score_weight: Option<f64>,
        temperature: Option<f64>,
        use_kv_events: Option<bool>,
        max_num_batched_tokens: Option<u32>,
    ) -> Self {
        let default = Self::default();
        Self {
            overlap_score_weight: overlap_score_weight.unwrap_or(default.overlap_score_weight),
            router_temperature: temperature.unwrap_or(default.router_temperature),
            use_kv_events: use_kv_events.unwrap_or(default.use_kv_events),
            max_num_batched_tokens: max_num_batched_tokens
                .unwrap_or(default.max_num_batched_tokens),
        }
    }
}

// TODO: is there a way (macro) to auto-derive the KvIndexerInterface trait for this
// since both variants implement it
pub enum Indexer {
    KvIndexer(KvIndexer),
    ApproxKvIndexer(ApproxKvIndexer),
}

impl Indexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Indexer::KvIndexer(indexer) => indexer.find_matches(sequence).await,
            Indexer::ApproxKvIndexer(indexer) => indexer.find_matches(sequence).await,
        }
    }
}

/// A KvRouter only decides which worker you should use. It doesn't send you there.
/// TODO: Rename this to indicate it only selects a worker, it does not route.
pub struct KvRouter {
    indexer: Indexer,

    // How about a Box<dyn KvIndexerInterface>
    scheduler: KvScheduler,

    block_size: u32,

    // To ensure blocking reads / writes
    // TODO: benchmark tradeoffs
    find_best_match_mutex: Mutex<()>,
}

impl KvRouter {
    pub async fn new(
        component: Component,
        block_size: u32,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        use_kv_events: bool,
    ) -> Result<Self> {
        let cancellation_token = component
            .drt()
            .primary_lease()
            .expect("Cannot KV route static workers")
            .primary_token();
        let metrics_aggregator =
            EndpointCollector::new(component.clone(), cancellation_token.clone()).await;

        let indexer = if use_kv_events {
            Indexer::KvIndexer(KvIndexer::new(cancellation_token.clone(), block_size))
        } else {
            // hard code 120 seconds for now
            Indexer::ApproxKvIndexer(ApproxKvIndexer::new(
                cancellation_token.clone(),
                block_size,
                Duration::from_secs(120),
            ))
        };

        let scheduler = KvScheduler::start(
            component.namespace().clone(),
            block_size,
            metrics_aggregator.endpoints_watcher(),
            selector,
        )
        .await?;

        // [gluo TODO] try subscribe_with_type::<RouterEvent>,
        // error checking below will be different.
        if let Indexer::KvIndexer(ref kv_indexer) = indexer {
            let mut kv_events_rx = component.subscribe(KV_EVENT_SUBJECT).await?;
            let kv_events_tx = kv_indexer.event_sender();

            tokio::spawn(async move {
                while let Some(event) = kv_events_rx.next().await {
                    let event: RouterEvent = match serde_json::from_slice(&event.payload) {
                        Ok(event) => event,
                        Err(e) => {
                            tracing::warn!("Failed to deserialize RouterEvent: {:?}", e);
                            // Choosing warn and continue to process other events from other workers
                            // A bad event likely signals a problem with a worker, but potentially other workers are still healthy
                            continue;
                        }
                    };
                    if let Err(e) = kv_events_tx.send(event).await {
                        tracing::debug!(
                            "failed to send kv event to indexer; shutting down: {:?}",
                            e
                        );
                    }
                }
            });
        }

        tracing::info!("KV Routing initialized");
        Ok(Self {
            indexer,
            scheduler,
            block_size,
            find_best_match_mutex: Mutex::new(()), // Add this
        })
    }

    /// Give these tokens, find the worker with the best match in it's KV cache.
    /// Returned overlap amount is in number of blocks.
    /// Now also takes context_id for request tracking
    async fn find_best_match(
        &self,
        context_id: &str,
        tokens: &[u32],
    ) -> anyhow::Result<(i64, u32)> {
        // Acquire mutex to serialize access
        // TODO: may as well make all the subroutines synchronous if benchmarking favors this
        let _guard = self.find_best_match_mutex.lock().await;

        let isl_tokens = tokens.len();
        let block_size = self.block_size;

        let local_block_hashes = compute_block_hash_for_seq(tokens, self.block_size);
        let overlap_scores = self.indexer.find_matches(local_block_hashes).await?;

        let best_worker_id = self
            .scheduler
            .schedule(
                context_id.to_string(),
                isl_tokens,
                block_size,
                tokens,
                overlap_scores.clone(),
            )
            .await?;

        if let Indexer::ApproxKvIndexer(ref indexer) = self.indexer {
            indexer
                .process_routing_decision_for_request(tokens, best_worker_id)
                .await
                .unwrap();
        };

        let overlap_amount = overlap_scores
            .scores
            .get(&best_worker_id)
            .copied()
            .unwrap_or(0);
        Ok((best_worker_id, overlap_amount))
    }

    /// Push tokens to a specific request's sequence
    pub async fn push(&self, request_id: &String, tokens: &[u32]) {
        self.scheduler.push(request_id, tokens).await
    }

    /// Free all blocks associated with a request
    pub async fn free(&self, request_id: &String) {
        self.scheduler.free(request_id).await
    }

    /// Get the block size this router was configured with
    pub fn block_size(&self) -> u32 {
        self.block_size
    }
}

// NOTE: this would not be usable for now, should deprecate
#[async_trait]
impl AsyncEngine<SingleIn<RouterRequest>, ManyOut<Annotated<RouterResponse>>, Error> for KvRouter {
    async fn generate(
        &self,
        request: SingleIn<RouterRequest>,
    ) -> Result<ManyOut<Annotated<RouterResponse>>> {
        let (request, ctx) = request.into_parts();
        let (worker_id, _) = self.find_best_match(ctx.id(), &request.tokens).await?;

        let response = RouterResponse { worker_id };
        let response = Annotated::from_data(response);
        let stream = stream::iter(vec![response]);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    chooser: Arc<KvRouter>,
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
    ) -> Self {
        KvPushRouter { inner, chooser }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for KvPushRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        match self.inner.client.instance_source.as_ref() {
            InstanceSource::Static => self.inner.r#static(request).await,
            InstanceSource::Dynamic(_) => {
                // Extract context ID for request tracking
                let context_id = request.context().id().to_string();

                let (instance_id, overlap_amount) = self
                    .chooser
                    .find_best_match(&context_id, &request.token_ids)
                    .await?;
                // Update the request with the estimated prefix hit blocks
                let (mut backend_input, context) = request.into_parts();
                let isl = backend_input.token_ids.len();
                backend_input.estimated_prefix_hit_num_blocks = Some(overlap_amount);
                let updated_request = context.map(|_| backend_input);

                // Get the response stream from the worker
                let mut response_stream = self.inner.direct(updated_request, instance_id).await?;

                // Wrap the stream to track tokens
                let stream_context = response_stream.context();
                let chooser = self.chooser.clone();
                let request_id = context_id.clone();
                let block_size = chooser.block_size() as usize;

                let wrapped_stream = Box::pin(async_stream::stream! {
                    let mut accumulated_tokens = Vec::new();
                    let mut total_output_length = 0usize;
                    let mut last_block_index = (isl.saturating_sub(1)) / block_size;
                    let mut first_push_done = false;

                    while let Some(item) = response_stream.next().await {
                        // Track tokens if they exist in the response
                        let Some(ref output) = item.data else {
                            yield item;
                            continue;
                        };
                        if output.token_ids.is_empty() {
                            yield item;
                            continue;
                        }

                        // Add tokens to accumulator
                        accumulated_tokens.extend_from_slice(&output.token_ids);
                        total_output_length += output.token_ids.len();

                        // Always push for the first generated token (to mark prefill done)
                        // or when we've moved to a new block
                        let current_block_index = (isl + total_output_length).saturating_sub(1) / block_size;
                        let should_push = (!first_push_done && total_output_length >= 1) ||
                                      (first_push_done && current_block_index > last_block_index);

                        if should_push {
                            chooser.push(&request_id, &accumulated_tokens).await;
                            accumulated_tokens.clear();
                            last_block_index = current_block_index;
                            if !first_push_done {
                                first_push_done = true;
                            }
                        }

                        yield item;
                    }

                    chooser.free(&request_id).await;
                });

                Ok(ResponseStream::new(wrapped_stream, stream_context))
            }
        }
    }
}
