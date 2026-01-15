// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct Router for pre-routed requests
//!
//! This router is used when worker IDs are pre-determined by an external orchestrator
//! (such as the Inference Gateway EPP). It extracts the worker ID from the request's
//! routing hints and routes directly to that worker.

use dynamo_runtime::{
    component::Client,
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        RouterMode, SingleIn,
    },
    protocols::annotated::Annotated,
};

use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
};

/// Router that reads worker ID from the request's routing hints
///
/// This router is used in the direct routing pipeline when worker selection
/// is done externally (e.g., by EPP). It extracts the worker ID from:
/// - `routing.decode_worker_id` (preferred)
/// - `routing.backend_instance_id` (fallback)
///
/// If neither is set, the request fails.
pub struct DirectFromRequestRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
}

impl DirectFromRequestRouter {
    /// Create a new DirectFromRequestRouter
    pub async fn new(client: Client) -> anyhow::Result<Self> {
        // Create PushRouter in Direct mode with a placeholder worker ID
        // The actual worker ID will be extracted from each request
        let inner = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client(
            client,
            RouterMode::Direct(0), // Placeholder - we'll override per-request
        )
        .await?;

        Ok(Self { inner })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for DirectFromRequestRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        // Extract worker ID from request routing hints
        let worker_id = request
            .routing
            .as_ref()
            .and_then(|r| r.decode_worker_id.or(r.backend_instance_id))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Direct routing requires decode_worker_id or backend_instance_id in request. \
                     This pipeline expects worker IDs to be set by an external orchestrator (e.g., EPP)."
                )
            })?;

        tracing::debug!(
            worker_id = worker_id,
            request_id = %request.context().id(),
            "Direct routing to pre-selected worker"
        );

        // Route directly to the specified worker
        self.inner.direct(request, worker_id).await
    }
}
