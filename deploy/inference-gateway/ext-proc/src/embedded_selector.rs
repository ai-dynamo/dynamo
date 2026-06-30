// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process selection backend (single-image evaluation mode).
//!
//! Built only with the `selector-embedded` feature. The EPP and the runtime-free
//! `SelectionCore` are compiled into one binary, and the EPP calls the selector's
//! Rust API directly — no HTTP client, no second process, no replication. This is
//! intended for single-replica evaluation, not production.
//!
//! Requests are built as the selection service's own public types and passed to
//! the core by value (no JSON serialization in the hot path). The HTTP backend
//! keeps the small mirror wire types so the production/thin-client build does not
//! have to compile the selection service.

use std::sync::Arc;

use anyhow::{Result, anyhow};

use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectRequest as CoreSelectRequest, SelectionCore, SelectionError,
    WorkerRequest as CoreWorkerRequest,
};
use tokio_util::sync::CancellationToken;

use crate::selection_backend::{
    OverlapSummary, SelectRequest, SelectResponse, SelectionBackend, WorkerRegistration,
};

/// Tenant scope for router-only mode. Must match between worker registration and
/// selection; the selection service's own default is `"default"`.
const DEFAULT_TENANT: &str = "default";

/// Embedded [`SelectionBackend`] wrapping an in-process [`SelectionCore`].
pub struct EmbeddedSelectionBackend {
    core: Arc<SelectionCore>,
    cancel: CancellationToken,
}

impl EmbeddedSelectionBackend {
    /// Build an in-process selector. `indexer_threads` sizes the KV indexer pool;
    /// router scheduling behavior comes from the standard `DYN_ROUTER_*`
    /// environment (same as `dynamo.select_service`).
    pub fn new(indexer_threads: usize) -> Result<Self> {
        let cancel = CancellationToken::new();
        let kv_router_config = kv_router_config_from_dynamo_env();
        let core = Arc::new(SelectionCore::new(
            kv_router_config,
            indexer_threads,
            cancel.clone(),
        ));
        tracing::info!(
            indexer_threads,
            "Initialized embedded (in-process) selection core"
        );
        Ok(Self { core, cancel })
    }
}

impl Drop for EmbeddedSelectionBackend {
    fn drop(&mut self) {
        // Stop the core's KV-event listeners, scheduling, and expiry tasks.
        self.core.shutdown();
        self.cancel.cancel();
    }
}

#[tonic::async_trait]
impl SelectionBackend for EmbeddedSelectionBackend {
    async fn upsert_worker(&self, reg: &WorkerRegistration) -> Result<()> {
        let req = CoreWorkerRequest {
            worker_id: reg.worker_id,
            model_name: reg.model_name.clone(),
            tenant_id: DEFAULT_TENANT.to_string(),
            endpoint: Some(reg.endpoint.clone()),
            block_size: Some(reg.block_size),
            data_parallel_size: Some(reg.data_parallel_size),
            kv_events_endpoints: reg.kv_events_endpoints.clone(),
            replay_endpoint: reg.replay_endpoint.clone(),
            total_kv_blocks: reg.total_kv_blocks,
            max_num_batched_tokens: reg.max_num_batched_tokens,
            stable_routing_id: reg.stable_routing_id.clone(),
            ..Default::default()
        };
        self.core
            .upsert_worker(req)
            .await
            .map_err(|e| anyhow!("embedded upsert_worker failed: {e}"))?;
        Ok(())
    }

    async fn delete_worker(&self, worker_id: u64) -> Result<()> {
        match self.core.delete_worker(worker_id).await {
            Ok(_) => Ok(()),
            // A worker that was never registered is not an error (idempotent).
            Err(SelectionError::NotFound(_)) => Ok(()),
            Err(e) => Err(anyhow!("embedded delete_worker failed: {e}")),
        }
    }

    async fn select(&self, req: &SelectRequest) -> Result<SelectResponse> {
        let core_req = CoreSelectRequest {
            model_name: req.model_name.clone(),
            tenant_id: DEFAULT_TENANT.to_string(),
            selection_id: req.selection_id.clone(),
            prompt: PromptRequest {
                token_ids: Some(req.token_ids.clone()),
                ..Default::default()
            },
            allowed_worker_ids: req.allowed_worker_ids.clone(),
            priority_jump: req.priority_jump,
            strict_priority: req.strict_priority,
            ..Default::default()
        };
        let resp = self
            .core
            .select(core_req)
            .await
            .map_err(|e| anyhow!("embedded select failed: {e}"))?;
        Ok(SelectResponse {
            selection_id: resp.selection_id,
            worker_id: resp.worker_id,
            dp_rank: resp.dp_rank,
            endpoint: resp.endpoint,
            block_size: resp.block_size,
            overlap: OverlapSummary {
                longest_matched: resp.overlap.longest_matched,
                gpu: resp.overlap.gpu,
                cpu: resp.overlap.cpu,
                disk: resp.overlap.disk,
            },
            effective_prefill_tokens: resp.effective_prefill_tokens,
        })
    }

    async fn any_ready(&self) -> bool {
        self.core.ready().ready
    }
}
