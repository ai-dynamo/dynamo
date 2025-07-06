// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

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
        indexer::{KvIndexer, KvIndexerInterface, RouterEvent},
        metrics_aggregator::EndpointCollector,
        protocols::{LocalBlockHash, RouterRequest, RouterResponse, WorkerSelectionResult},
        scheduler::{KvScheduler, KvSchedulerError, SchedulingRequest},
        scoring::ProcessedEndpoints,
    },
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
    tokens::TokenBlockSequence,
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
#[derive(Debug, Clone)]
pub struct KvRouterConfig {
    /// Weight for overlap score in worker selection.
    /// Higher values prioritize KV cache reuse. Default: 2.0
    pub overlap_score_weight: f64,

    pub router_temperature: f64,
}

impl Default for KvRouterConfig {
    fn default() -> Self {
        Self {
            overlap_score_weight: 1.0,
            router_temperature: 0.5,
        }
    }
}

impl KvRouterConfig {
    /// Create a new KvRouterConfig with optional weight values.
    /// If a weight is None, the default value will be used.
    pub fn new(overlap_score_weight: Option<f64>, temperature: Option<f64>) -> Self {
        let default = Self::default();
        Self {
            overlap_score_weight: overlap_score_weight.unwrap_or(default.overlap_score_weight),
            router_temperature: temperature.unwrap_or(default.router_temperature),
        }
    }
}

/// A KvRouter only decides which worker you should use. It doesn't send you there.
/// TODO: Rename this to indicate it only selects a worker, it does not route.
pub struct KvRouter {
    indexer: KvIndexer,
    scheduler: KvScheduler,
    block_size: u32,
}

impl KvRouter {
    pub async fn new(
        component: Component,
        block_size: u32,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
    ) -> Result<Self> {
        let cancellation_token = component
            .drt()
            .primary_lease()
            .expect("Cannot KV route static workers")
            .primary_token();
        let metrics_aggregator =
            EndpointCollector::new(component.clone(), cancellation_token.clone()).await;

        let indexer = KvIndexer::new(cancellation_token.clone(), block_size);
        let scheduler = KvScheduler::start(
            component.namespace().clone(),
            block_size,
            metrics_aggregator.endpoints_watcher(),
            selector,
        )
        .await?;

        // [gluo TODO] try subscribe_with_type::<RouterEvent>,
        // error checking below will be different.
        let mut kv_events_rx = component.subscribe(KV_EVENT_SUBJECT).await?;
        let kv_events_tx = indexer.event_sender();

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
                    tracing::debug!("failed to send kv event to indexer; shutting down: {:?}", e);
                }
            }
        });

        tracing::info!("KV Routing initialized");
        Ok(Self {
            scheduler,
            indexer,
            block_size,
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
        let isl_tokens = tokens.len();
        let block_size = self.block_size;

        let (complete_blocks, _partial_block) =
            TokenBlockSequence::split_tokens(tokens, block_size, 1337_u64);

        let local_block_hashes = complete_blocks
            .into_iter()
            .map(|block| LocalBlockHash(block.block_hash()))
            .collect();
        let overlap_scores = self.indexer.find_matches(local_block_hashes).await?;

        let best_worker_id = self
            .scheduler
            .schedule_and_update(
                context_id.to_string(),
                isl_tokens,
                block_size,
                tokens,
                overlap_scores.clone(),
            )
            .await?;

        let overlap_amount = overlap_scores
            .scores
            .get(&best_worker_id)
            .copied()
            .unwrap_or(0);
        Ok((best_worker_id, overlap_amount))
    }

    /// Push a token to a specific request's sequence
    pub async fn push(&self, request_id: &String, token: u32) {
        self.scheduler.push(request_id, token).await
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
                backend_input.estimated_prefix_hit_num_blocks = Some(overlap_amount);
                let updated_request = context.map(|_| backend_input);

                // Get the response stream from the worker
                let mut response_stream = self.inner.direct(updated_request, instance_id).await?;

                // Wrap the stream to track tokens
                let stream_context = response_stream.context();
                let chooser = self.chooser.clone();
                let request_id = context_id.clone();

                let wrapped_stream = Box::pin(async_stream::stream! {
                    while let Some(item) = response_stream.next().await {
                        // Track tokens if they exist in the response
                        if let Some(ref output) = item.data {
                            for token_id in &output.token_ids {
                                chooser.push(&request_id, *token_id).await;
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
