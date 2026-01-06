// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode Disaggregation Operator
//!
//! Routes requests to different decode tiers based on sequence length.
//! When a request's sequence length exceeds the current tier's capacity,
//! it migrates to a higher-capacity tier.

use std::{pin::Pin, sync::Arc, time::Instant};

use anyhow::Result;
use futures::StreamExt;
use parking_lot::RwLock;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Endpoint,
    engine::{AsyncEngineContext, AsyncEngineStream},
    pipeline::{
        AsyncEngineContextProvider, Context, ManyOut, Operator, PushRouter, ResponseStream,
        RouterMode, ServerStreamingEngine, SingleIn, async_trait,
        network::egress::push_router::RoutedManyOut,
    },
    protocols::annotated::Annotated,
};

use crate::{
    discovery::{DecodeTierNotification, ModelManager},
    http::service::metrics::Metrics,
    kv_router::{KvPushRouter, KvRouterConfig},
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        preprocessor::{MigrationRequest, MigrationResponse},
    },
};
use dynamo_runtime::metrics::prometheus_names::frontend_service::decode_disagg::failure_reason;

/// Shared tiers storage wrapped in Arc for cloning across async tasks
type SharedTiers = RwLock<Vec<Arc<Tier>>>;

fn find_tier(tiers: &Arc<SharedTiers>, seqlen: u32) -> Option<Arc<Tier>> {
    tiers.read().iter().find(|t| t.seqlen > seqlen).cloned()
}

/// A tier represents a set of workers with a specific sequence length capacity.
#[derive(Clone)]
struct Tier {
    /// Maximum sequence length this tier can handle
    seqlen: u32,
    /// Router to workers in this tier (KV-aware or simple)
    router: TierRouter,
    /// Router for migration requests (to call migrate endpoint on source worker)
    /// Uses Annotated wrapper to match streaming protocol format
    migrate_router: Arc<PushRouter<MigrationRequest, Annotated<MigrationResponse>>>,
}

/// The router type for a tier
#[derive(Clone)]
enum TierRouter {
    /// KV-aware routing
    Kv(Arc<KvPushRouter>),
    /// Simple routing (RoundRobin, Random)
    Simple(Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>),
}

impl TierRouter {
    async fn generate_routed(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<RoutedManyOut<Annotated<LLMEngineOutput>>> {
        match self {
            TierRouter::Kv(router) => router.generate_routed(request).await,
            TierRouter::Simple(router) => router.generate_routed(request).await,
        }
    }
}

/// DecodeDisagger routes requests to different decode tiers based on sequence length.
///
/// Tiers are identified by their sequence length capacity (e.g., 8192, 32768).
/// Workers register to components named `llm-{seqlen}` (e.g., `llm-8192`).
///
/// Routing logic:
/// 1. If ISL > lowest tier's seqlen, route to appropriate tier
/// 2. During decode, if ISL + OSL exceeds current tier's seqlen, migrate to higher tier
pub struct DecodeDisagger {
    /// Tiers sorted by seqlen ascending (wrapped in Arc for async cloning)
    tiers: Arc<SharedTiers>,
    /// Cancellation token for background tasks
    cancel_token: CancellationToken,
    /// Metrics for decode disaggregation
    metrics: Arc<Metrics>,
    /// Model name for metrics labels
    model_name: Arc<String>,
}

impl DecodeDisagger {
    /// Create a disabled DecodeDisagger that just forwards to next (passthrough mode)
    pub fn disabled(model_name: &str, metrics: Arc<Metrics>) -> Arc<Self> {
        Arc::new(Self {
            tiers: Arc::new(RwLock::new(Vec::new())),
            cancel_token: CancellationToken::new(),
            metrics,
            model_name: Arc::new(model_name.to_string()),
        })
    }

    /// Create a new DecodeDisagger (no tiers yet, can be added dynamically)
    pub fn new(model_name: &str, metrics: Arc<Metrics>) -> Self {
        Self {
            tiers: Arc::new(RwLock::new(Vec::new())),
            cancel_token: CancellationToken::new(),
            metrics,
            model_name: Arc::new(model_name.to_string()),
        }
    }

    /// Create a DecodeDisagger with dynamic tier discovery.
    /// Subscribes to ModelManager for tier notifications as workers with `this_seqlen` come online.
    ///
    /// # Arguments
    /// * `model_name` - The model name to subscribe to tier notifications for
    /// * `model_manager` - The model manager for tier discovery and KV routing
    /// * `router_mode` - Router mode for tier routers
    /// * `kv_router_config` - Optional KV router config for tier routers
    /// * `kv_cache_block_size` - KV cache block size for KV routing
    /// * `metrics` - Metrics for decode disaggregation tracking
    pub fn with_dynamic_tiers(
        model_name: &str,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_router_config: Option<KvRouterConfig>,
        kv_cache_block_size: u32,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        let tiers = Arc::new(RwLock::new(Vec::new()));
        let cancel_token = CancellationToken::new();
        let model_name_arc = Arc::new(model_name.to_string());

        let disagger = Arc::new(Self {
            tiers: tiers.clone(),
            cancel_token: cancel_token.clone(),
            metrics: metrics.clone(),
            model_name: model_name_arc.clone(),
        });

        // Get existing tiers that were activated before we subscribed
        let existing_tiers = model_manager.get_existing_decode_tiers(model_name);
        tracing::info!(
            model_name = model_name,
            existing_tiers = existing_tiers.len(),
            "DecodeDisagger checking for existing tiers"
        );

        // Subscribe to tier notifications and spawn background task
        let mut tier_rx = model_manager.subscribe_decode_tiers(model_name);
        let model_name = model_name.to_string();
        let metrics_clone = metrics.clone();

        tokio::spawn(async move {
            // First, activate any existing tiers
            for decode_tier in existing_tiers {
                let seqlen = decode_tier.seqlen();
                tracing::info!(
                    model_name = %model_name,
                    seqlen = seqlen,
                    "Activating existing decode tier"
                );

                if let Err(e) = Self::activate_tier_inner(
                    &tiers,
                    seqlen,
                    decode_tier.decode_endpoint().clone(),
                    decode_tier.migrate_endpoint().clone(),
                    router_mode,
                    kv_router_config,
                    kv_cache_block_size,
                    Some(model_manager.clone()),
                )
                .await
                {
                    tracing::error!(
                        seqlen = seqlen,
                        error = %e,
                        "Failed to activate existing tier"
                    );
                } else {
                    metrics_clone.inc_active_tiers(&model_name);
                }
            }

            // Now listen for tier add/remove notifications
            loop {
                tokio::select! {
                    result = tier_rx.recv() => {
                        match result {
                            Ok(DecodeTierNotification::Added { seqlen, decode_endpoint, migrate_endpoint }) => {
                                tracing::info!(
                                    model_name = %model_name,
                                    seqlen = seqlen,
                                    "Decode tier added"
                                );

                                if let Err(e) = Self::activate_tier_inner(
                                    &tiers,
                                    seqlen,
                                    decode_endpoint,
                                    migrate_endpoint,
                                    router_mode,
                                    kv_router_config,
                                    kv_cache_block_size,
                                    Some(model_manager.clone()),
                                ).await {
                                    tracing::error!(
                                        seqlen = seqlen,
                                        error = %e,
                                        "Failed to activate tier"
                                    );
                                } else {
                                    metrics_clone.inc_active_tiers(&model_name);
                                }
                            }
                            Ok(DecodeTierNotification::Removed { seqlen }) => {
                                tracing::info!(
                                    model_name = %model_name,
                                    seqlen = seqlen,
                                    "Decode tier removed"
                                );

                                let mut tiers_guard = tiers.write();
                                tiers_guard.retain(|t| t.seqlen != seqlen);
                                metrics_clone.dec_active_tiers(&model_name);
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                tracing::warn!(
                                    model_name = %model_name,
                                    lagged = n,
                                    "Tier notification receiver lagged"
                                );
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                tracing::debug!(
                                    model_name = %model_name,
                                    "Tier notification channel closed"
                                );
                                break;
                            }
                        }
                    }
                    _ = cancel_token.cancelled() => {
                        tracing::debug!(
                            model_name = %model_name,
                            "Tier discovery task cancelled"
                        );
                        break;
                    }
                }
            }
        });

        disagger
    }

    /// Activate a tier with the provided endpoints
    async fn activate_tier_inner(
        tiers: &SharedTiers,
        seqlen: u32,
        decode_endpoint: Endpoint,
        migrate_endpoint: Endpoint,
        router_mode: RouterMode,
        kv_router_config: Option<KvRouterConfig>,
        kv_cache_block_size: u32,
        model_manager: Option<Arc<ModelManager>>,
    ) -> Result<()> {
        tracing::info!(seqlen, ?router_mode, "Activating decode tier");

        let router = if router_mode.is_kv_routing() {
            let Some(manager) = model_manager else {
                anyhow::bail!("KV routing requires ModelManager");
            };

            let kv_chooser = manager
                .kv_chooser_for(&decode_endpoint, kv_cache_block_size, kv_router_config)
                .await?;

            let client = kv_chooser.client().clone();
            let push_router =
                PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                    client,
                    RouterMode::KV,
                    None,
                    None,
                )
                .await?;

            TierRouter::Kv(Arc::new(KvPushRouter::new(push_router, kv_chooser)))
        } else {
            let client = decode_endpoint.client().await?;
            let push_router =
                PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                    client,
                    router_mode,
                    None,
                    None,
                )
                .await?;

            TierRouter::Simple(Arc::new(push_router))
        };

        // Create migrate router for calling migrate endpoint on source workers
        // Uses Annotated wrapper to match streaming protocol format
        let migrate_client = migrate_endpoint.client().await?;
        let migrate_push_router =
            PushRouter::<MigrationRequest, Annotated<MigrationResponse>>::from_client_with_threshold(
                migrate_client,
                RouterMode::RoundRobin, // Will use direct() for specific instance targeting
                None,
                None,
            )
            .await?;

        let tier = Arc::new(Tier {
            seqlen,
            router,
            migrate_router: Arc::new(migrate_push_router),
        });

        // Insert tier in sorted order
        let mut tiers_guard = tiers.write();
        let insert_pos = tiers_guard
            .iter()
            .position(|t| t.seqlen > seqlen)
            .unwrap_or(tiers_guard.len());
        tiers_guard.insert(insert_pos, tier);

        tracing::info!(
            seqlen,
            total_tiers = tiers_guard.len(),
            "Decode tier activated successfully"
        );

        Ok(())
    }
}

// Holds all the mutable state for a migration
struct MigrationContext {
    // These are the objects which get updated as we migrate
    previous_stream: Option<Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>>,
    current_stream: Pin<Box<dyn AsyncEngineStream<Annotated<LLMEngineOutput>>>>,
    current_instance_id: u64,
    current_tier: Arc<Tier>,
    /// Tracks when a migration started and from which seqlen (cleared after first token)
    migration_started_at: Option<(Instant, u32)>,
    /// Chunks drained during migration that should be yielded to the client
    pending_chunks: Vec<Annotated<LLMEngineOutput>>,
    /// Original prompt length (used to calculate tokens generated for max_tokens adjustment)
    original_prompt_len: usize,
    /// Metrics for tracking migration performance
    metrics: Arc<Metrics>,
    /// Model name for metrics labels
    model_name: Arc<String>,
}

impl MigrationContext {
    pub async fn new(
        current_tier: Arc<Tier>,
        request: PreprocessedRequest,
        parent_ctx: Arc<dyn AsyncEngineContext>,
        metrics: Arc<Metrics>,
        model_name: Arc<String>,
    ) -> Result<Self> {
        let original_prompt_len = request.token_ids.len();
        let inner_ctx = Context::with_id(request.clone(), parent_ctx.id().to_string());
        parent_ctx.link_child(inner_ctx.context());
        let routed_out = current_tier.router.generate_routed(inner_ctx).await?;
        let (current_stream, current_instance_id) = routed_out.take();

        // Track inflight request for this tier
        metrics.inc_tier_inflight(&model_name, current_tier.seqlen);

        Ok(Self {
            previous_stream: None,
            current_stream,
            current_instance_id,
            current_tier,
            migration_started_at: None,
            pending_chunks: Vec::new(),
            original_prompt_len,
            metrics,
            model_name,
        })
    }

    pub async fn send_migration(
        &mut self,
        request_id: &str,
        tokens_seen: u32,
        parent_ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Option<MigrationResponse> {
        // Call migrate endpoint on the source worker to initiate KV transfer
        // The response includes bootstrap_info and all token_ids from the request
        let migrate_req = MigrationRequest {
            rid: request_id.to_string(),
            tokens_seen,
        };
        let migrate_context = Context::with_id(migrate_req, format!("migrate-{}", request_id));
        parent_ctx.link_child(migrate_context.context());

        // Use direct routing to the specific instance that has the request
        let migration_result = self
            .current_tier
            .migrate_router
            .direct(migrate_context, self.current_instance_id)
            .await;

        match migration_result {
            Ok(routed_migrate) => {
                let (mut migrate_stream, _) = routed_migrate.take();
                match migrate_stream.next().await {
                    Some(annotated_response) => annotated_response.data,
                    None => {
                        tracing::warn!("Migration endpoint returned empty response");
                        self.metrics.inc_tier_migration_failure(
                            &self.model_name,
                            failure_reason::MIGRATE_EMPTY_RESPONSE,
                        );
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to call migrate endpoint");
                self.metrics.inc_tier_migration_failure(
                    &self.model_name,
                    failure_reason::MIGRATE_ENDPOINT_FAILED,
                );
                None
            }
        }
    }

    pub async fn tick(
        mut self,
        tiers: &Arc<SharedTiers>,
        request_id: &str,
        running_request: &PreprocessedRequest,
        parent_ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Result<MigrationContext> {
        // If we've reached the end of the current tier, migrate to the next tier
        if running_request.token_ids.len() >= self.current_tier.seqlen as usize {
            // We've reached the end of the current tier, so migrate to the next tier
            tracing::info!(
                request_token_count = running_request.token_ids.len(),
                current_tier_seqlen = self.current_tier.seqlen,
                request_id = request_id,
                "Reached the end of the current tier, migrating to the next tier",
            );
            let Some(new_tier) = find_tier(tiers, (running_request.token_ids.len() + 1) as u32)
            else {
                self.metrics
                    .inc_tier_migration_failure(&self.model_name, failure_reason::NO_NEXT_TIER);
                return Err(anyhow::anyhow!("No next tier available for migration"));
            };

            // Start timing the migration
            let migration_start = Instant::now();
            let from_seqlen = self.current_tier.seqlen;

            // Call migrate endpoint - this de-schedules the request on decode1 and returns
            // the full token_ids (origin_input_ids + output_ids) so we know exactly what to send to decode2
            let tokens_seen = running_request.token_ids.len() as u32;
            let migration_response = self
                .send_migration(&request_id, tokens_seen, parent_ctx)
                .await;
            let Some(response) = migration_response else {
                // Migration failure already recorded in send_migration
                return Err(anyhow::anyhow!(
                    "Migration failed: no response from migrate endpoint"
                ));
            };

            tracing::info!(
                bootstrap_room = response.bootstrap_info.bootstrap_room,
                tokens_seen = tokens_seen,
                pending_outputs_count = response.pending_outputs.len(),
                request_id = request_id,
                "Migration response received",
            );

            // Build the full token_ids for the migrated request by extending with pending outputs
            let mut migrated_token_ids = running_request.token_ids.clone();
            for output in &response.pending_outputs {
                migrated_token_ids.extend(output.token_ids.iter().copied());
            }

            // Use the pending_outputs from the response - these are the actual LLMEngineOutput
            // chunks that the frontend hasn't seen yet, preserving API behavior
            let pending_chunks: Vec<Annotated<LLMEngineOutput>> = response
                .pending_outputs
                .into_iter()
                .map(Annotated::from_data)
                .collect();

            // Calculate tokens generated so far and reduce max_tokens accordingly
            let tokens_generated = migrated_token_ids
                .len()
                .saturating_sub(self.original_prompt_len);

            // Record migration detail metrics
            self.metrics
                .observe_migration_pending_chunks(&self.model_name, pending_chunks.len());
            self.metrics
                .observe_migration_tokens_transferred(&self.model_name, migrated_token_ids.len());
            self.metrics
                .observe_migration_tokens_generated_before(&self.model_name, tokens_generated);

            // Create the migrated request with the full token_ids and adjusted max_tokens
            let mut migrated_request = running_request.clone();
            migrated_request.token_ids = migrated_token_ids;

            // Reduce max_tokens by tokens already generated
            if let Some(max_tokens) = migrated_request.stop_conditions.max_tokens {
                let remaining_tokens = max_tokens.saturating_sub(tokens_generated as u32);
                migrated_request.stop_conditions.max_tokens = Some(remaining_tokens);
                tracing::info!(
                    original_max_tokens = max_tokens,
                    tokens_generated = tokens_generated,
                    remaining_max_tokens = remaining_tokens,
                    request_id = request_id,
                    "Adjusted max_tokens for migrated request",
                );
            }

            // Reduce min_tokens by tokens already generated
            if let Some(min_tokens) = migrated_request.stop_conditions.min_tokens {
                let remaining_min = min_tokens.saturating_sub(tokens_generated as u32);
                migrated_request.stop_conditions.min_tokens = Some(remaining_min);
                tracing::info!(
                    original_min_tokens = min_tokens,
                    remaining_min_tokens = remaining_min,
                    request_id = request_id,
                    "Adjusted min_tokens for migrated request",
                );
            }

            let mut new_context = Context::with_id(migrated_request, request_id.to_string());
            new_context.bootstrap_info = Some(response.bootstrap_info);

            parent_ctx.link_child(new_context.context());
            let routed_out = match new_tier.router.generate_routed(new_context).await {
                Ok(out) => out,
                Err(e) => {
                    self.metrics.inc_tier_migration_failure(
                        &self.model_name,
                        failure_reason::ROUTING_FAILED,
                    );
                    return Err(e);
                }
            };
            let (new_stream, new_instance_id) = routed_out.take();
            tracing::info!(
                new_instance_id = new_instance_id,
                new_tier_seqlen = new_tier.seqlen,
                pending_chunks = pending_chunks.len(),
                request_id = request_id,
                "Created new stream for next tier",
            );

            // Update tier inflight counts
            self.metrics
                .dec_tier_inflight(&self.model_name, from_seqlen);
            self.metrics
                .inc_tier_inflight(&self.model_name, new_tier.seqlen);

            // Keep the old stream alive until we receive the first token from the new tier.
            // This prevents the old context from being cancelled before KV transfer completes.
            // The old stream will be dropped on the next tick() call (after first token received).
            return Ok(MigrationContext {
                previous_stream: Some(self.current_stream),
                current_stream: new_stream,
                current_instance_id: new_instance_id,
                current_tier: new_tier,
                migration_started_at: Some((migration_start, from_seqlen)),
                pending_chunks,
                original_prompt_len: self.original_prompt_len,
                metrics: self.metrics.clone(),
                model_name: self.model_name.clone(),
            });
        }
        Ok(self)
    }

    /// Wait for the next chunk from the current stream.
    /// ALWAYS pulls a chunk from the current stream.
    /// If there any pending chunks from a migration drain, prepend them to the current stream's chunks.
    pub async fn tock(
        &mut self,
        running_request: &mut PreprocessedRequest,
        request_id: &str,
    ) -> (StreamResult, Vec<Annotated<LLMEngineOutput>>) {
        // First, return any pending chunks from the migration drain
        let mut chunks_to_return = Vec::new();
        if !self.pending_chunks.is_empty() {
            let chunks = std::mem::take(&mut self.pending_chunks);
            for chunk in &chunks {
                if let Some(ref data) = chunk.data {
                    running_request.token_ids.extend(data.token_ids.iter());
                }
            }
            tracing::info!(
                pending_chunks = chunks.len(),
                request_id = request_id,
                "Found pending chunks from migration drain",
            );
            chunks_to_return.extend(chunks);
        }

        let Some(chunk) = self.current_stream.next().await else {
            // No more chunks from the current stream, so we're done
            return (StreamResult::Stop, chunks_to_return);
        };

        if let Some(ref data) = chunk.data {
            // Keep track of the request + tokens we've received so far
            running_request.token_ids.extend(data.token_ids.iter());
        }

        chunks_to_return.push(chunk);

        // Record migration duration if we just migrated
        if let Some((migration_start, from_seqlen)) = self.migration_started_at.take() {
            let duration = migration_start.elapsed();
            tracing::info!(
                duration_ms = duration.as_millis(),
                from_seqlen = from_seqlen,
                to_seqlen = self.current_tier.seqlen,
                request_id = request_id,
                "Migration completed, received first token from new tier"
            );

            // Record migration metrics
            self.metrics.observe_tier_migration(
                &self.model_name,
                from_seqlen,
                self.current_tier.seqlen,
                duration.as_secs_f64(),
            );
        }

        if let Some(previous_stream) = self.previous_stream.take() {
            drop(previous_stream);
        }

        (StreamResult::Continue, chunks_to_return)
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum StreamResult {
    Continue,
    Stop,
}

impl Drop for DecodeDisagger {
    fn drop(&mut self) {
        tracing::debug!("Dropping DecodeDisagger, cancelling background tasks");
        self.cancel_token.cancel();
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for DecodeDisagger
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let Some(tier) = find_tier(&self.tiers, request.token_ids.len() as u32) else {
            let tier_seqlens: Vec<u32> = self.tiers.read().iter().map(|t| t.seqlen).collect();
            tracing::warn!(
                request_token_count = request.token_ids.len(),
                available_tiers = ?tier_seqlens,
                "No tier found for request, routing to next tier"
            );
            // Record passthrough metric
            self.metrics.inc_tier_passthrough(&self.model_name);
            return next.generate(request).await;
        };

        tracing::info!(
            request_token_count = request.token_ids.len(),
            tier_seqlen = tier.seqlen,
            "Routing request to tier"
        );

        // Record tier request metric
        self.metrics.inc_tier_request(&self.model_name, tier.seqlen);

        let (mut running_request, context) = request.transfer(());
        let parent_engine_ctx = context.context();
        let mut migration_context = MigrationContext::new(
            tier,
            running_request.clone(),
            parent_engine_ctx.clone(),
            self.metrics.clone(),
            self.model_name.clone(),
        )
        .await?;

        let request_id = context.id().to_string();
        let tiers = self.tiers.clone();
        let metrics = self.metrics.clone();
        let model_name = self.model_name.clone();
        let migrating_stream = async_stream::stream! {
            let parent_engine_ctx = context.context();
            // Track the current tier seqlen for inflight decrement on completion
            let mut current_tier_seqlen = migration_context.current_tier.seqlen;
            loop {
                if parent_engine_ctx.is_stopped() || parent_engine_ctx.is_killed() {
                    break;
                }
                match migration_context.tick(&tiers, &request_id, &running_request, &parent_engine_ctx).await {
                    Ok(ctx) => {
                        migration_context = ctx;
                        current_tier_seqlen = migration_context.current_tier.seqlen;
                    }
                    Err(e)=> {
                        yield Annotated::from_error(e.to_string());
                        break;
                    }
                };
                let (result, chunks) = migration_context.tock(&mut running_request, &request_id).await;
                for chunk in chunks {
                    yield chunk;
                }
                if result == StreamResult::Stop {
                    break;
                }
            }
            // Decrement inflight for the final tier when stream completes
            metrics.dec_tier_inflight(&model_name, current_tier_seqlen);
        };

        Ok(ResponseStream::new(
            Box::pin(migrating_stream),
            parent_engine_ctx.clone(),
        ))
    }
}
