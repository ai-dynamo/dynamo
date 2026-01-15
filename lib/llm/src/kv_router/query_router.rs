// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query Router for query-only routing operations
//!
//! This router is designed for use with the Inference Gateway EPP (External Pre-Processor)
//! to query optimal worker IDs without sending the actual request.
//!
//! The QueryRouter supports both aggregated and disaggregated modes:
//! - Aggregated: Returns a single decode_worker_id (uses KvRouter directly)
//! - Disaggregated: Returns prefill_worker_id + decode_worker_id (uses PrefillRouter + KvRouter)

use std::sync::Arc;

use anyhow::{Context, Result};

use dynamo_runtime::DistributedRuntime;

use crate::{
    discovery::ModelManager,
    kv_router::{KvRouter, KvRouterConfig, PrefillRouter, RouterConfigOverride},
};
use dynamo_runtime::pipeline::RouterMode;

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
///
/// Internally uses:
/// - `PrefillRouter` for prefill worker selection (reuses existing logic)
/// - `KvRouter` for decode worker selection
pub struct QueryRouter {
    /// KV-aware router for decode workers (also handles aggregated mode)
    decode_router: Arc<KvRouter>,

    /// PrefillRouter for prefill worker selection (reuses pipeline logic)
    /// None until prefill workers are discovered
    prefill_router: Arc<PrefillRouter>,

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

        // Create PrefillRouter (reuses existing activation logic)
        // It will auto-activate when prefill workers are discovered
        let prefill_router = model_manager
            .register_prefill_router(model_name.to_string())
            .map(|rx| {
                // Create prefill-specific config
                let mut prefill_config = kv_router_config.clone();
                prefill_config.router_track_active_blocks = false;

                PrefillRouter::new(
                    rx,
                    model_manager.clone(),
                    RouterMode::KV, // Use KV routing for prefill
                    block_size,
                    Some(prefill_config),
                    enforce_disagg,
                )
            })
            .unwrap_or_else(|| {
                PrefillRouter::disabled(model_manager.clone(), RouterMode::KV, enforce_disagg)
            });

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
        // Check if disaggregated mode is available (prefill router activated)
        let is_disaggregated = self.prefill_router.is_activated();

        if self.enforce_disagg && !is_disaggregated {
            anyhow::bail!("Disaggregated mode enforced but no prefill workers available");
        }

        if is_disaggregated {
            // Disaggregated mode: query both prefill and decode routers
            // Use PrefillRouter's query method (reuses existing logic)
            let (prefill_worker_id, _dp_rank) = self
                .prefill_router
                .query_prefill_worker(token_ids, update_states)
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
                prefill_worker_id,
                decode_worker_id: decode_worker.worker_id,
                is_disaggregated: true,
            })
        } else {
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
    pub fn is_disaggregated(&self) -> bool {
        self.prefill_router.is_activated()
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
