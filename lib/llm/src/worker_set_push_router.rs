// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WorkerSet-aware PushRouter wrapper for Random/RoundRobin routing across multiple worker sets.
//!
//! During rollouts, workers from different versions coexist in separate sets (namespaces).
//! This router selects a worker set (weighted by worker count), then selects a worker
//! within that set, ensuring traffic distributes naturally during rollouts.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{Error, ManyOut, PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};

use crate::{
    discovery::{WorkerSet, WorkerSetManager},
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};

/// Wrapper around PushRouter that handles worker set selection for multi-set routing.
///
/// For Random/RoundRobin modes in prefix (multi-set) deployments:
/// 1. Selects a WorkerSet using weighted random selection
/// 2. Selects a worker within that set (random or round-robin)
/// 3. Routes to that specific worker via the underlying PushRouter
pub struct WorkerSetPushRouter {
    #[allow(dead_code)] // Kept for constructor compatibility; routing uses per-set cached routers
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    worker_set_manager: Arc<WorkerSetManager>,
    router_mode: RouterMode,
    /// Cached PushRouters per namespace for cross-namespace routing.
    set_routers: DashMap<String, Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>>,
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
            set_routers: DashMap::new(),
        }
    }

    /// Select a worker using worker set selection.
    ///
    /// Returns the selected worker set and instance ID, or an error if no workers are available.
    fn select_worker(&self) -> Result<(Arc<WorkerSet>, u64), Error> {
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

        let instance_id = instance_id.ok_or_else(|| {
            Error::from(anyhow::anyhow!(
                "Selected worker set has no available instances"
            ))
        })?;

        Ok((worker_set, instance_id))
    }

    /// Get or create a cached PushRouter for the given worker set's namespace.
    async fn get_or_create_router(
        &self,
        set: &Arc<WorkerSet>,
    ) -> Result<Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>, Error> {
        let namespace = set.namespace().to_string();
        if let Some(router) = self.set_routers.get(&namespace) {
            return Ok(router.value().clone());
        }
        let router = Arc::new(
            PushRouter::from_client(set.client().clone(), RouterMode::KV).await?,
        );
        self.set_routers.insert(namespace, router.clone());
        Ok(router)
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
        // Select worker set and worker using worker set selection
        let (worker_set, instance_id) = self.select_worker()?;

        tracing::debug!(
            instance_id,
            mode = ?self.router_mode,
            set_count = self.worker_set_manager.set_count(),
            "WorkerSetPushRouter selected worker"
        );

        // Route via a cached PushRouter for the selected set's namespace
        let router = self.get_or_create_router(&worker_set).await?;
        router.direct(request, instance_id).await
    }
}
