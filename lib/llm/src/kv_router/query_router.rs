// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query Router for query-only routing operations
//!
//! This is a thin wrapper that holds a PrefillRouter and decode KvRouter together,
//! replicating the `query_instance_id` flow but standalone (not through pipeline).
//!
//! The actual routing logic is delegated to `PrefillRouter.query_route()`.

use std::sync::Arc;

use anyhow::{Context, Result};

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::pipeline::RouterMode;

use crate::{
    discovery::ModelManager,
    kv_router::{KvRouter, KvRouterConfig, PrefillRouter},
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
/// Holds a PrefillRouter and decode KvRouter together, enabling query-only
/// routing that returns worker IDs without executing requests.
///
/// This replicates the `query_instance_id` annotation behavior but as a
/// standalone API for C/Go bindings.
pub struct QueryRouter {
    /// Decode router (also handles aggregated mode)
    decode_router: Arc<KvRouter>,

    /// PrefillRouter handles prefill worker selection and disagg logic
    prefill_router: Arc<PrefillRouter>,
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

        // Create decode router
        let decode_router = model_manager
            .kv_chooser_for(&endpoint, block_size, Some(kv_router_config.clone()))
            .await
            .context("Failed to create decode router")?;

        // Create PrefillRouter (auto-activates when prefill workers discovered)
        let prefill_router = model_manager
            .register_prefill_router(model_name.to_string())
            .map(|rx| {
                let mut prefill_config = kv_router_config.clone();
                prefill_config.router_track_active_blocks = false;

                PrefillRouter::new(
                    rx,
                    model_manager.clone(),
                    RouterMode::KV,
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
        })
    }

    /// Query optimal worker IDs without executing a request.
    ///
    /// Delegates to `PrefillRouter.query_route()` which handles both
    /// aggregated and disaggregated modes.
    pub async fn query_route(
        &self,
        token_ids: &[u32],
        update_states: bool,
    ) -> Result<RouteQueryResult> {
        self.prefill_router
            .query_route(&self.decode_router, token_ids, update_states)
            .await
    }

    /// Check if disaggregated mode is currently active
    pub fn is_disaggregated(&self) -> bool {
        self.prefill_router.is_activated()
    }

    /// Get direct access to the decode router
    pub fn decode_router(&self) -> &Arc<KvRouter> {
        &self.decode_router
    }

    /// Get direct access to the prefill router
    pub fn prefill_router(&self) -> &Arc<PrefillRouter> {
        &self.prefill_router
    }
}
