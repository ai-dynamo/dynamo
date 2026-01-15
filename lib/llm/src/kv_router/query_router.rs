// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query Router for query-only routing operations
//!
//! This router is designed for use with the Inference Gateway EPP (External Pre-Processor)
//! to query optimal worker IDs without sending the actual request.
//!
//! The QueryRouter supports both aggregated and disaggregated modes:
//! - Aggregated: Returns a single decode_worker_id
//! - Disaggregated: Returns prefill_worker_id + decode_worker_id (when prefill workers available)

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::RwLock;

use dynamo_runtime::DistributedRuntime;

use crate::{
    discovery::ModelManager,
    kv_router::{KvRouter, KvRouterConfig, RouterConfigOverride},
};

/// Result of a routing query
#[derive(Debug, Clone, Default)]
pub struct RouteQueryResult {
    /// Worker ID for prefill phase (only valid if is_disaggregated is true)
    pub prefill_worker_id: u64,
    /// Worker ID for decode phase (always valid)
    pub decode_worker_id: u64,
    /// True if disaggregated mode is active (prefill_worker_id is valid)
    pub is_disaggregated: bool,
}

/// Query Router for EPP integration
///
/// This router only queries for optimal worker IDs without sending requests.
/// It supports both aggregated and disaggregated modes based on prefill worker availability.
pub struct QueryRouter {
    /// KV-aware router for decode workers (also handles aggregated mode)
    decode_router: Arc<KvRouter>,

    /// Optional KV-aware router for prefill workers (None = aggregated mode)
    /// This is lazily activated when prefill workers are discovered
    prefill_router: Arc<RwLock<Option<Arc<KvRouter>>>>,

    /// Whether to enforce disaggregated mode (fail if prefill unavailable)
    enforce_disagg: bool,
}

impl QueryRouter {
    /// Create a new QueryRouter
    ///
    /// The router starts in aggregated mode and automatically switches to
    /// disaggregated mode when prefill workers become available.
    pub async fn new(
        distributed_runtime: &DistributedRuntime,
        namespace: &str,
        component_name: &str,
        model_name: &str,
        block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        enforce_disagg: bool,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();

        let component = distributed_runtime
            .namespace(namespace)?
            .component(component_name)?;
        let endpoint = component.endpoint("generate");

        let model_manager = Arc::new(ModelManager::new());

        // Create KV router for decode workers
        let decode_router = model_manager
            .kv_chooser_for(&endpoint, block_size, Some(kv_router_config.clone()))
            .await
            .context("Failed to create decode router")?;

        // Register for prefill worker discovery
        let prefill_rx = model_manager.register_prefill_router(model_name.to_string());

        let prefill_router: Arc<RwLock<Option<Arc<KvRouter>>>> = Arc::new(RwLock::new(None));

        // Spawn background task to activate prefill router when workers discovered
        if let Some(rx) = prefill_rx {
            let prefill_router_clone = prefill_router.clone();
            let model_manager_clone = model_manager.clone();
            let kv_config = kv_router_config.clone();

            tokio::spawn(async move {
                match rx.await {
                    Ok(endpoint) => {
                        tracing::info!("Prefill workers discovered, activating disaggregated mode");

                        // Create prefill-specific config
                        let mut prefill_config = kv_config;
                        prefill_config.router_track_active_blocks = false;

                        // Create KV router for prefill workers
                        match model_manager_clone
                            .kv_chooser_for(&endpoint, block_size, Some(prefill_config))
                            .await
                        {
                            Ok(new_prefill_router) => {
                                let mut guard = prefill_router_clone.write().await;
                                *guard = Some(new_prefill_router);
                                tracing::info!("Disaggregated mode activated");
                            }
                            Err(e) => {
                                tracing::error!(error = %e, "Failed to create prefill router");
                            }
                        }
                    }
                    Err(_) => {
                        tracing::debug!("Prefill router activation channel closed");
                    }
                }
            });
        }

        Ok(Self {
            decode_router,
            prefill_router,
            enforce_disagg,
        })
    }

    /// Query the optimal worker(s) for a request
    ///
    /// This method does NOT send the request, it only queries for worker IDs.
    /// State is optionally updated based on `update_states` parameter.
    ///
    /// Returns:
    /// - In aggregated mode: decode_worker_id only
    /// - In disaggregated mode: both prefill_worker_id and decode_worker_id
    pub async fn query_route(
        &self,
        request_id: Option<&str>,
        token_ids: &[u32],
        update_states: bool,
    ) -> Result<RouteQueryResult> {
        // Check if disaggregated mode is available
        let prefill_router_guard = self.prefill_router.read().await;
        let is_disaggregated = prefill_router_guard.is_some();

        if self.enforce_disagg && !is_disaggregated {
            anyhow::bail!("Disaggregated mode enforced but no prefill workers available");
        }

        if is_disaggregated {
            let prefill_router = prefill_router_guard.as_ref().unwrap().clone();
            drop(prefill_router_guard); // Release lock before async operations

            // Disaggregated mode: query both prefill and decode routers
            // Query prefill worker
            let (prefill_worker, _prefill_overlap) = prefill_router
                .find_best_match(request_id, token_ids, None, update_states)
                .await
                .context("Failed to query prefill worker")?;

            // Query decode worker with overlap_score_weight = 0
            // (we want to minimize load, not maximize overlap for decode)
            let mut decode_override = RouterConfigOverride::default();
            decode_override.overlap_score_weight = Some(0.0);

            let (decode_worker, _decode_overlap) = self
                .decode_router
                .find_best_match(request_id, token_ids, Some(&decode_override), update_states)
                .await
                .context("Failed to query decode worker")?;

            Ok(RouteQueryResult {
                prefill_worker_id: prefill_worker.worker_id,
                decode_worker_id: decode_worker.worker_id,
                is_disaggregated: true,
            })
        } else {
            drop(prefill_router_guard); // Release lock

            // Aggregated mode: query decode router only
            let (worker, _overlap) = self
                .decode_router
                .find_best_match(request_id, token_ids, None, update_states)
                .await
                .context("Failed to query worker")?;

            Ok(RouteQueryResult {
                prefill_worker_id: 0,
                decode_worker_id: worker.worker_id,
                is_disaggregated: false,
            })
        }
    }

    /// Check if disaggregated mode is currently active
    pub async fn is_disaggregated(&self) -> bool {
        self.prefill_router.read().await.is_some()
    }

    /// Mark prefill as completed for a request (for state tracking)
    pub async fn mark_prefill_complete(&self, request_id: &str) -> Result<()> {
        self.decode_router
            .mark_prefill_completed(request_id)
            .await
            .context("Failed to mark prefill complete")
    }

    /// Free a request (release resources)
    pub async fn free(&self, request_id: &str) -> Result<()> {
        self.decode_router
            .free(request_id)
            .await
            .context("Failed to free request")
    }
}
