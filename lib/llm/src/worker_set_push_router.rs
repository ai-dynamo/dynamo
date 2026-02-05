// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WorkerSet-aware PushRouter wrapper for Random/RoundRobin routing across multiple worker sets.
//!
//! During rollouts, workers from different versions coexist in separate sets (namespaces).
//! This router selects a worker set (weighted by worker count), then selects a worker
//! within that set, ensuring traffic distributes naturally during rollouts.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{Error, ManyOut, PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};

use crate::{
    discovery::WorkerSetManager,
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};

/// Wrapper around PushRouter that handles worker set selection for multi-set routing.
///
/// For Random/RoundRobin modes in prefix (multi-set) deployments:
/// 1. Selects a WorkerSet using weighted random selection
/// 2. Selects a worker within that set (random or round-robin)
/// 3. Routes to that specific worker via the underlying PushRouter
pub struct WorkerSetPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    worker_set_manager: Arc<WorkerSetManager>,
    router_mode: RouterMode,
}

impl WorkerSetPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        worker_set_manager: Arc<WorkerSetManager>,
        router_mode: RouterMode,
    ) -> Self {
        Self {
            inner,
            worker_set_manager,
            router_mode,
        }
    }

    /// Select a worker using worker set selection.
    ///
    /// Returns the instance ID to route to, or an error if no workers are available.
    fn select_worker(&self) -> Result<u64, Error> {
        // Step 1: Select a worker set (weighted by worker count)
        let worker_set = self
            .worker_set_manager
            .select_weighted()
            .ok_or_else(|| {
                Error::from(anyhow::anyhow!(
                    "No worker sets available for routing"
                ))
            })?;

        // Step 2: Select a worker within that set
        let instance_id = match self.router_mode {
            RouterMode::Random => worker_set.select_random(),
            RouterMode::RoundRobin => worker_set.select_round_robin(),
            _ => {
                return Err(Error::from(anyhow::anyhow!(
                    "WorkerSetPushRouter only supports Random and RoundRobin modes"
                )))
            }
        };

        instance_id.ok_or_else(|| {
            Error::from(anyhow::anyhow!(
                "Selected worker set has no available instances"
            ))
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for WorkerSetPushRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        // Select worker using worker set selection
        let instance_id = self.select_worker()?;

        tracing::debug!(
            instance_id,
            mode = ?self.router_mode,
            set_count = self.worker_set_manager.set_count(),
            "WorkerSetPushRouter selected worker"
        );

        // Route to the selected worker via the underlying PushRouter
        self.inner.direct(request, instance_id).await
    }
}
