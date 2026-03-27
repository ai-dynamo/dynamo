// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;

use dynamo_runtime::{
    pipeline::{AsyncEngine, ManyOut, PushRouter, SingleIn},
    protocols::annotated::Annotated,
};

use crate::{
    kv_router::KvPushRouter,
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};

/// The inner router used by PrefillRouter
#[derive(Clone)]
pub(super) enum InnerPrefillRouter {
    /// KV-aware routing using KvPushRouter
    KvRouter(Arc<KvPushRouter>),
    /// Simple routing (RoundRobin, Random, Direct)
    /// Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
    /// available in KV routing mode where the router has actual bookkeeping.
    SimpleRouter(Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>),
}

impl InnerPrefillRouter {
    /// Generate with optional direct routing to specific worker.
    /// For KvRouter, target_worker is ignored since prefill_worker_id is already set on the request.
    /// For SimpleRouter, target_worker triggers direct routing via router.direct().
    pub(super) async fn generate_to_worker(
        &self,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        match (self, target_worker) {
            // KvRouter: prefill_worker_id already set on request, KvPushRouter::select_worker uses it
            (InnerPrefillRouter::KvRouter(router), _) => router.generate(request).await,
            (InnerPrefillRouter::SimpleRouter(router), Some(worker_id)) => {
                router.direct(request, worker_id).await
            }
            (InnerPrefillRouter::SimpleRouter(router), None) => router.generate(request).await,
        }
    }

    /// Select next worker (for non-KV modes only)
    pub(super) fn select_next_worker(&self) -> Option<u64> {
        match self {
            InnerPrefillRouter::SimpleRouter(router) => router.select_next_worker(),
            InnerPrefillRouter::KvRouter(_) => None,
        }
    }
}
