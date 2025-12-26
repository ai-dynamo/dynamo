// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode Disaggregation Operator
//!
//! Routes requests to different decode tiers based on sequence length.
//! When a request's sequence length exceeds the current tier's capacity,
//! it migrates to a higher-capacity tier.

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use parking_lot::RwLock;
use tokio::sync::{broadcast, oneshot};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Endpoint,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Context, ManyOut, Operator, PushRouter,
        ResponseStream, RouterMode, ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};

use crate::{
    discovery::{DecodeTierNotification, ModelManager},
    kv_router::{KvPushRouter, KvRouterConfig},
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};

/// Shared tiers storage wrapped in Arc for cloning across async tasks
type SharedTiers = Arc<RwLock<Vec<Tier>>>;

/// A tier represents a set of workers with a specific sequence length capacity.
#[derive(Clone)]
struct Tier {
    /// Maximum sequence length this tier can handle
    seqlen: u32,
    /// Router to workers in this tier (KV-aware or simple)
    router: TierRouter,
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
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        match self {
            TierRouter::Kv(router) => router.generate(request).await,
            TierRouter::Simple(router) => router.generate(request).await,
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
    tiers: SharedTiers,
    /// Router mode for creating tier routers
    router_mode: RouterMode,
    /// KV router config for creating KV routers
    kv_router_config: Option<KvRouterConfig>,
    /// KV cache block size
    kv_cache_block_size: u32,
    /// Model manager for creating KV choosers
    model_manager: Option<Arc<ModelManager>>,
    /// Cancellation token for background tasks
    cancel_token: CancellationToken,
}

impl DecodeDisagger {
    /// Create a disabled DecodeDisagger that just forwards to next (passthrough mode)
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            tiers: Arc::new(RwLock::new(Vec::new())),
            router_mode: RouterMode::RoundRobin,
            kv_router_config: None,
            kv_cache_block_size: 0,
            model_manager: None,
            cancel_token: CancellationToken::new(),
        })
    }

    /// Create a new DecodeDisagger (no tiers yet, can be added dynamically)
    pub fn new() -> Self {
        Self {
            tiers: Arc::new(RwLock::new(Vec::new())),
            router_mode: RouterMode::RoundRobin,
            kv_router_config: None,
            kv_cache_block_size: 0,
            model_manager: None,
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
            router_mode,
            kv_router_config,
            kv_cache_block_size,
            model_manager: Some(model_manager.clone()),
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
            for (seqlen, endpoint) in existing_tiers {
                tracing::info!(
                    model_name = %model_name,
                    seqlen = seqlen,
                    "Activating existing decode tier"
                );

                if let Err(e) = Self::activate_tier_inner(
                    &tiers,
                    seqlen,
                    endpoint,
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
                            Ok(DecodeTierNotification::Added { seqlen, endpoint }) => {
                                tracing::info!(
                                    model_name = %model_name,
                                    seqlen = seqlen,
                                    "Decode tier added"
                                );

                                if let Err(e) = Self::activate_tier_inner(
                                    &tiers,
                                    seqlen,
                                    endpoint,
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

    /// Register a tier for activation. Returns a sender to provide the endpoint when discovered.
    pub fn register_tier(&self, seqlen: u32) -> oneshot::Sender<Endpoint> {
        let (tx, rx) = oneshot::channel();

        // Spawn background task to wait for activation
        let tiers = self.tiers.clone();
        let router_mode = self.router_mode;
        let kv_router_config = self.kv_router_config;
        let kv_cache_block_size = self.kv_cache_block_size;
        let model_manager = self.model_manager.clone();
        let cancel_token = self.cancel_token.clone();

        tokio::spawn(async move {
            tokio::select! {
                result = rx => {
                    let Ok(endpoint) = result else {
                        tracing::debug!(seqlen, "Tier activation channel closed without endpoint");
                        return;
                    };

                    if let Err(e) = Self::activate_tier_inner(
                        &tiers,
                        seqlen,
                        endpoint,
                        router_mode,
                        kv_router_config,
                        kv_cache_block_size,
                        model_manager,
                    ).await {
                        tracing::error!(seqlen, error = %e, "Failed to activate tier");
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::debug!(seqlen, "Tier activation cancelled");
                }
            }
        });

        tx
    }

    /// Activate a tier with the provided endpoint
    async fn activate_tier_inner(
        tiers: &SharedTiers,
        seqlen: u32,
        endpoint: Endpoint,
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
                .kv_chooser_for(&endpoint, kv_cache_block_size, kv_router_config)
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
            let client = endpoint.client().await?;
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

        let tier = Tier { seqlen, router };

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

    /// Check if any tiers are available
    pub fn has_tiers(&self) -> bool {
        !self.tiers.read().is_empty()
    }

    /// Select the appropriate tier for a given sequence length
    fn select_tier(&self, seqlen: usize) -> Option<Tier> {
        let tiers = self.tiers.read();
        // Find first tier that can handle this seqlen
        tiers.iter().find(|t| seqlen <= t.seqlen as usize).cloned()
    }

    /// Check if this is the maximum tier
    fn is_max_tier(&self, seqlen: u32) -> bool {
        let tiers = self.tiers.read();
        tiers.last().map(|t| t.seqlen == seqlen).unwrap_or(true)
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
        let num_tiers = self.tiers.read().len();
        tracing::info!(num_tiers, "DecodeDisagger::generate called");

        // If no tiers configured, just forward to next (passthrough mode)
        if !self.has_tiers() {
            tracing::info!("DecodeDisagger: No tiers configured, using passthrough mode");
            return next.generate(request).await;
        }

        let (req, context) = request.into_parts();
        let isl = req.token_ids.len();

        tracing::info!(isl, num_tiers, "DecodeDisagger: Routing request with ISL");

        // Log available tiers
        {
            let tiers = self.tiers.read();
            for tier in tiers.iter() {
                tracing::info!(tier_seqlen = tier.seqlen, "DecodeDisagger: Available tier");
            }
        }

        // Select initial tier based on ISL
        let Some(current_tier) = self.select_tier(isl) else {
            // No tier can handle this, forward to next (let it fail there)
            tracing::warn!(
                isl,
                "No tier available for sequence length, forwarding to next"
            );
            return next.generate(context.map(|_| req)).await;
        };

        let current_seqlen = current_tier.seqlen;

        tracing::info!(
            isl,
            selected_tier = current_seqlen,
            "DecodeDisagger: Selected tier for request"
        );

        // If this is the max tier, just forward (no migration possible)
        if self.is_max_tier(current_seqlen) {
            tracing::info!(
                isl,
                tier_seqlen = current_seqlen,
                "DecodeDisagger: Using max tier, forwarding to next"
            );
            return next.generate(context.map(|_| req)).await;
        }

        // Route to current tier and monitor for migration
        let request_for_tier = context.map(|_| req.clone());
        let mut stream = current_tier.router.generate(request_for_tier).await?;
        let stream_context = stream.context();
        let stream_context_for_return = stream_context.clone();
        let request_id = stream_context.id().to_string();

        let tiers = self.tiers.clone();
        let mut req_clone = req.clone();

        // Create a stream that monitors output and migrates if needed
        let migrating_stream = async_stream::stream! {
            let mut output_tokens = 0usize;

            while let Some(chunk) = stream.next().await {
                // Count tokens in this chunk
                if let Some(ref data) = chunk.data {
                    output_tokens += data.token_ids.len();
                    req_clone.token_ids.extend(data.token_ids.iter());
                }

                // Check if we need to migrate
                let total_seqlen = isl + output_tokens;
                if total_seqlen > current_seqlen as usize {
                    tracing::info!(
                        isl,
                        output_tokens,
                        total_seqlen,
                        current_tier = current_seqlen,
                        "Sequence length exceeded tier capacity, migrating"
                    );

                    // Drop current stream (cancels the request)
                    drop(stream);

                    // Get next tier (clone inside block to avoid holding lock across await)
                    let higher_tier = {
                        let tiers_guard = tiers.read();
                        tiers_guard
                            .iter()
                            .find(|t| t.seqlen > current_seqlen)
                            .cloned()
                    };

                    let Some(higher_tier) = higher_tier else {
                        tracing::error!("No higher tier available for migration");
                        return;
                    };

                    tracing::info!(
                        new_tier = higher_tier.seqlen,
                        "Migrating to higher tier"
                    );

                    // Re-send full request to higher tier
                    // Note: This triggers full prefill on the new tier
                    let migration_context = Context::with_id(req_clone.clone(), request_id.clone());
                    match higher_tier.router.generate(migration_context).await {
                        Ok(mut new_stream) => {
                            // Continue streaming from new tier
                            while let Some(new_chunk) = new_stream.next().await {
                                yield new_chunk;
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Failed to migrate to higher tier");
                        }
                    }
                    return;
                }

                yield chunk;
            }
        };

        Ok(ResponseStream::new(
            Box::pin(migrating_stream),
            stream_context_for_return,
        ))
    }
}
