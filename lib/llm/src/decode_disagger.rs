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

use crate::protocols::common::preprocessor::BootstrapInfo;
use crate::{
    discovery::{DecodeTierNotification, ModelManager},
    kv_router::{KvPushRouter, KvRouterConfig},
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        preprocessor::{MigrationRequest, MigrationResponse},
    },
};

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
}

impl DecodeDisagger {
    /// Create a disabled DecodeDisagger that just forwards to next (passthrough mode)
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            tiers: Arc::new(RwLock::new(Vec::new())),
            cancel_token: CancellationToken::new(),
        })
    }

    /// Create a new DecodeDisagger (no tiers yet, can be added dynamically)
    pub fn new() -> Self {
        Self {
            tiers: Arc::new(RwLock::new(Vec::new())),
            cancel_token: CancellationToken::new(),
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
    pub fn with_dynamic_tiers(
        model_name: &str,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_router_config: Option<KvRouterConfig>,
        kv_cache_block_size: u32,
    ) -> Arc<Self> {
        let tiers = Arc::new(RwLock::new(Vec::new()));
        let cancel_token = CancellationToken::new();

        let disagger = Arc::new(Self {
            tiers: tiers.clone(),
            cancel_token: cancel_token.clone(),
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
}

impl MigrationContext {
    pub async fn new(
        current_tier: Arc<Tier>,
        request: PreprocessedRequest,
        parent_ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<Self> {
        let inner_ctx = Context::with_id(request.clone(), parent_ctx.id().to_string());
        parent_ctx.link_child(inner_ctx.context());
        let routed_out = current_tier.router.generate_routed(inner_ctx).await?;
        let (current_stream, current_instance_id) = routed_out.take();
        Ok(Self {
            previous_stream: None,
            current_stream,
            current_instance_id,
            current_tier,
            migration_started_at: None,
        })
    }

    pub async fn send_migration(
        &mut self,
        request_id: &str,
        parent_ctx: &Arc<dyn AsyncEngineContext>,
    ) -> Option<BootstrapInfo> {
        // Step 1: Call migrate endpoint on the source worker to initiate KV transfer
        let migrate_req = MigrationRequest {
            rid: request_id.to_string(),
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
                    Some(annotated_response) => annotated_response.data.map(|r| r.bootstrap_info),
                    None => {
                        tracing::warn!("Migration endpoint returned empty response");
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to call migrate endpoint");
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
        // If we just migrated and then got the next token, drop the previous stream
        if let Some(previous_stream) = self.previous_stream.take() {
            drop(previous_stream);
        }

        // If we've reached the end of the current tier, migrate to the next tier
        if running_request.token_ids.len() >= self.current_tier.seqlen as usize {
            // We've reached the end of the current tier, so migrate to the next tier
            tracing::info!(
                request_token_count = running_request.token_ids.len(),
                current_tier_seqlen = self.current_tier.seqlen,
                "Reached the end of the current tier, migrating to the next tier",
            );
            let Some(new_tier) = find_tier(tiers, (running_request.token_ids.len() + 1) as u32)
            else {
                return Err(anyhow::anyhow!("No next tier available for migration"));
            };

            // Start timing the migration
            let migration_start = Instant::now();
            let from_seqlen = self.current_tier.seqlen;

            let bootstrap_info = self.send_migration(&request_id, parent_ctx).await;
            let mut new_context = Context::with_id(running_request.clone(), request_id.to_string());
            if let Some(info) = bootstrap_info {
                tracing::info!(
                    bootstrap_info = ?info.clone(),
                    "Migration successful, received bootstrap info",
                );
                new_context.bootstrap_info = Some(info);
            } else {
                tracing::warn!("no bootstrap info received");
            }
            parent_ctx.link_child(new_context.context());
            let routed_out = new_tier.router.generate_routed(new_context).await?;
            let (new_stream, new_instance_id) = routed_out.take();
            tracing::info!(
                new_instance_id = new_instance_id,
                new_tier_seqlen = new_tier.seqlen,
                "Created new stream for next tier",
            );
            return Ok(MigrationContext {
                previous_stream: Some(self.current_stream),
                current_stream: new_stream,
                current_instance_id: new_instance_id,
                current_tier: new_tier,
                migration_started_at: Some((migration_start, from_seqlen)),
            });
        }
        Ok(self)
    }

    pub async fn tock(
        &mut self,
        running_request: &mut PreprocessedRequest,
    ) -> Option<Annotated<LLMEngineOutput>> {
        let Some(chunk) = self.current_stream.next().await else {
            return None;
        };

        // Log migration duration if we just migrated
        if let Some((migration_start, from_seqlen)) = self.migration_started_at.take() {
            let duration = migration_start.elapsed();
            tracing::info!(
                duration_ms = duration.as_millis(),
                from_seqlen = from_seqlen,
                to_seqlen = self.current_tier.seqlen,
                "Migration completed, received first token from new tier"
            );
        }

        if let Some(ref data) = chunk.data {
            // Keep track of the request + tokens we've received so far
            running_request.token_ids.extend(data.token_ids.iter());
        }
        Some(chunk)
    }
}

impl Default for DecodeDisagger {
    fn default() -> Self {
        Self::new()
    }
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
            return next.generate(request).await;
        };

        tracing::info!(
            request_token_count = request.token_ids.len(),
            tier_seqlen = tier.seqlen,
            "Routing request to tier"
        );

        let (mut running_request, context) = request.transfer(());
        let parent_engine_ctx = context.context();
        let mut migration_context =
            MigrationContext::new(tier, running_request.clone(), parent_engine_ctx.clone()).await?;

        let request_id = context.id().to_string();
        let tiers = self.tiers.clone();
        let migrating_stream = async_stream::stream! {
            let parent_engine_ctx = context.context();
            loop {
                if parent_engine_ctx.is_stopped() || parent_engine_ctx.is_killed() {
                    break;
                }
                match migration_context.tick(&tiers, &request_id, &running_request, &parent_engine_ctx).await {
                    Ok(ctx) => {
                        migration_context = ctx;
                    }
                    Err(e)=> {
                        yield Annotated::from_error(e.to_string());
                        break;
                    }
                };
                if let Some(chunk) = migration_context.tock(&mut running_request).await {
                    yield chunk;
                } else {
                    break;
                }
            }
        };

        Ok(ResponseStream::new(
            Box::pin(migrating_stream),
            parent_engine_ctx.clone(),
        ))
    }
}
