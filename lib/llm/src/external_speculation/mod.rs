// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-plane state for externally hosted speculative decoding.
//!
//! Target and draft ranks are selected independently with the configured Dynamo
//! routing policy. Discovery composes those selectors with the exact registered
//! ranks and the draft transport coordinates needed by the target.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use anyhow::Context as _;
use async_trait::async_trait;
use dynamo_kv_router::{
    protocols::WorkerWithDpRank, scheduling::AdmissionAttempt, selector::WorkerSelector,
};
use dynamo_runtime::{
    pipeline::{OccupancyReservation, RouterMode, SingleIn},
    protocols::EndpointId,
};

use crate::{
    kv_router::{FindBestMatchOutcome, KvRouter, RoutingHost},
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::{
        common::preprocessor::PreprocessedRequest, external_speculation::DraftTransportDescriptorV1,
    },
    worker_role::ExternalDraftBinding,
};

mod metrics;
mod router;

pub(crate) use metrics::ExternalSpeculationMetrics;
pub(crate) use router::ExternalSpeculationRouter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeculationPool {
    Target,
    Draft,
}

impl SpeculationPool {
    fn as_str(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Draft => "draft",
        }
    }
}

pub(crate) struct SpeculationSelection {
    pub worker: WorkerWithDpRank,
    pub overlap_blocks: u32,
    pub cached_tokens: usize,
    pub attempt: AdmissionAttempt,
    pub occupancy: Option<OccupancyReservation>,
}

#[async_trait]
pub(crate) trait SpeculationChooser: Send + Sync {
    async fn select_and_reserve(
        &self,
        request_id: &str,
        request: &SingleIn<PreprocessedRequest>,
        candidates: &HashSet<WorkerWithDpRank>,
    ) -> anyhow::Result<SpeculationSelection>;

    async fn release(
        &self,
        request_id: &str,
        worker: WorkerWithDpRank,
        attempt: AdmissionAttempt,
    ) -> anyhow::Result<()>;

    fn router_mode(&self) -> RouterMode;
}

/// Role-scoped adapter around Dynamo's normal KV selector.
pub(crate) struct KvSpeculationChooser<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig>,
{
    router: Arc<KvRouter<Sel>>,
    pool: SpeculationPool,
}

impl<Sel> KvSpeculationChooser<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig>,
{
    pub(crate) fn new(router: Arc<KvRouter<Sel>>, pool: SpeculationPool) -> Self {
        Self { router, pool }
    }
}

#[async_trait]
impl<Sel> SpeculationChooser for KvSpeculationChooser<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
{
    async fn select_and_reserve(
        &self,
        request_id: &str,
        request: &SingleIn<PreprocessedRequest>,
        candidates: &HashSet<WorkerWithDpRank>,
    ) -> anyhow::Result<SpeculationSelection> {
        let routing = request.routing.as_ref();
        let (tokens, block_mm_infos) = request.block_mm_routing_info();
        let mut allowed_workers = candidates.clone();
        if self.pool == SpeculationPool::Target
            && let Some(request_allowed) =
                routing.and_then(|routing| routing.allowed_worker_ids.as_ref())
        {
            allowed_workers.retain(|worker| request_allowed.contains(&worker.worker_id));
        }
        anyhow::ensure!(
            !allowed_workers.is_empty(),
            "no committed {} rank satisfies the request's worker constraints",
            self.pool.as_str()
        );
        let admitted = self
            .router
            .find_best_match_details_for_workers(
                request_id,
                tokens,
                block_mm_infos,
                request.router_config_override.as_ref(),
                routing.and_then(|routing| routing.lora_name.clone()),
                routing.and_then(|routing| routing.cache_namespace.clone()),
                routing
                    .and_then(|routing| routing.priority_jump)
                    .unwrap_or(0.0),
                routing
                    .and_then(|routing| routing.strict_priority)
                    .unwrap_or(0),
                routing.and_then(|routing| routing.expected_output_tokens),
                allowed_workers,
                routing
                    .and_then(|routing| routing.routing_constraints.clone())
                    .unwrap_or_default(),
            )
            .await?;
        let (outcome, attempt) = admitted.into_parts();
        match outcome {
            FindBestMatchOutcome::Routed {
                worker,
                overlap_blocks,
                cached_tokens,
                ..
            } => Ok(SpeculationSelection {
                worker,
                overlap_blocks,
                cached_tokens,
                attempt,
                occupancy: None,
            }),
            FindBestMatchOutcome::QueueRejected { rejection } => Err(rejection.into()),
        }
    }

    async fn release(
        &self,
        request_id: &str,
        worker: WorkerWithDpRank,
        attempt: AdmissionAttempt,
    ) -> anyhow::Result<()> {
        let AdmissionAttempt::Tracked(attempt_id) = attempt else {
            anyhow::bail!(
                "{} speculative selection for request {request_id} had no booking attempt",
                self.pool.as_str()
            );
        };
        self.router
            .free_if_booking(request_id, worker, attempt_id)
            .await
            .with_context(|| {
                format!(
                    "failed to release {} speculative booking for request {request_id} on worker {} rank {}",
                    self.pool.as_str(),
                    worker.worker_id,
                    worker.dp_rank
                )
            })
    }

    fn router_mode(&self) -> RouterMode {
        RouterMode::KV
    }
}

/// Role-scoped adapter around Dynamo's ordinary routing host.
pub(crate) struct PolicySpeculationChooser {
    host: RoutingHost,
    pool: SpeculationPool,
}

impl PolicySpeculationChooser {
    pub(crate) fn new(host: RoutingHost, pool: SpeculationPool) -> Self {
        Self { host, pool }
    }
}

#[async_trait]
impl SpeculationChooser for PolicySpeculationChooser {
    async fn select_and_reserve(
        &self,
        _request_id: &str,
        request: &SingleIn<PreprocessedRequest>,
        candidates: &HashSet<WorkerWithDpRank>,
    ) -> anyhow::Result<SpeculationSelection> {
        let constrained_candidates;
        let candidates = if self.pool == SpeculationPool::Target
            && let Some(request_allowed) = request
                .routing
                .as_ref()
                .and_then(|routing| routing.allowed_worker_ids.as_ref())
        {
            constrained_candidates = candidates
                .iter()
                .filter(|worker| request_allowed.contains(&worker.worker_id))
                .copied()
                .collect::<HashSet<_>>();
            &constrained_candidates
        } else {
            candidates
        };
        anyhow::ensure!(
            !candidates.is_empty(),
            "no committed {} rank satisfies the request's worker constraints",
            self.pool.as_str()
        );
        let (worker_id, occupancy) = self
            .host
            .select_for_external_speculation(request, candidates)?;
        // Ordinary policies select endpoint instances. Without KV affinity, ranks within the
        // selected instance are equivalent, so choose one uniformly for exact dispatch/hints.
        let rank_count = candidates
            .iter()
            .filter(|worker| worker.worker_id == worker_id)
            .count();
        anyhow::ensure!(
            rank_count > 0,
            "selected {} worker {worker_id} has no registered speculative rank",
            self.pool.as_str()
        );
        let rank_offset = if rank_count == 1 {
            0
        } else {
            rand::random_range(0..rank_count)
        };
        let worker = candidates
            .iter()
            .filter(|worker| worker.worker_id == worker_id)
            .nth(rank_offset)
            .copied()
            .expect("rank count was checked before selection");
        Ok(SpeculationSelection {
            worker,
            overlap_blocks: 0,
            cached_tokens: 0,
            attempt: AdmissionAttempt::Untracked,
            occupancy,
        })
    }

    async fn release(
        &self,
        _request_id: &str,
        _worker: WorkerWithDpRank,
        _attempt: AdmissionAttempt,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn router_mode(&self) -> RouterMode {
        self.host.router_mode()
    }
}

/// Complete request-plane composition retained for one target request.
#[derive(Clone)]
pub struct SpeculationCompositionSnapshot {
    pub target_endpoint: EndpointId,
    pub draft_endpoint: EndpointId,
    pub binding: ExternalDraftBinding,
    pub target_workers: Arc<HashSet<WorkerWithDpRank>>,
    draft_workers: Arc<HashSet<WorkerWithDpRank>>,
    pub draft_transports: Arc<HashMap<WorkerWithDpRank, DraftTransportDescriptorV1>>,
    pub(crate) target_chooser: Arc<dyn SpeculationChooser>,
    pub(crate) draft_chooser: Arc<dyn SpeculationChooser>,
}

impl fmt::Debug for SpeculationCompositionSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpeculationCompositionSnapshot")
            .field("target_endpoint", &self.target_endpoint)
            .field("draft_endpoint", &self.draft_endpoint)
            .field("binding", &self.binding)
            .field("target_workers", &self.target_workers)
            .field("draft_transports", &self.draft_transports)
            .finish_non_exhaustive()
    }
}

impl SpeculationCompositionSnapshot {
    pub(crate) fn new(
        target_endpoint: EndpointId,
        draft_endpoint: EndpointId,
        binding: ExternalDraftBinding,
        target_workers: HashSet<WorkerWithDpRank>,
        draft_transports: HashMap<WorkerWithDpRank, DraftTransportDescriptorV1>,
        target_chooser: Arc<dyn SpeculationChooser>,
        draft_chooser: Arc<dyn SpeculationChooser>,
    ) -> anyhow::Result<Arc<Self>> {
        let draft_workers = Arc::new(draft_transports.keys().copied().collect());
        let snapshot = Arc::new(Self {
            target_endpoint,
            draft_endpoint,
            binding,
            target_workers: Arc::new(target_workers),
            draft_workers,
            draft_transports: Arc::new(draft_transports),
            target_chooser,
            draft_chooser,
        });
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.target_workers.is_empty(),
            "speculative composition has no selectable target ranks"
        );
        anyhow::ensure!(
            !self.draft_transports.is_empty(),
            "speculative composition has no selectable draft ranks"
        );
        anyhow::ensure!(
            self.target_chooser.router_mode() == self.draft_chooser.router_mode(),
            "speculative target and draft must use the same router mode"
        );
        anyhow::ensure!(
            self.binding.endpoint == self.draft_endpoint,
            "speculative composition draft endpoint does not match its binding"
        );
        for transport in self.draft_transports.values() {
            transport.validate()?;
            anyhow::ensure!(
                transport.protocol == self.binding.protocol,
                "draft transport protocol does not match the target binding"
            );
        }
        Ok(())
    }
}

/// Discovery indexes used to invalidate all targets bound to a changed draft endpoint.
#[derive(Default)]
pub(crate) struct SpeculationCompositionRegistry {
    state: parking_lot::RwLock<SpeculationCompositionRegistryState>,
}

#[derive(Default)]
struct SpeculationCompositionRegistryState {
    endpoint_groups: HashMap<EndpointId, HashSet<String>>,
    draft_dependents: HashMap<EndpointId, HashSet<String>>,
    ready_targets: HashSet<String>,
}

impl SpeculationCompositionRegistry {
    pub(crate) fn replace(
        &self,
        endpoint_groups: HashMap<EndpointId, HashSet<String>>,
        draft_dependents: HashMap<EndpointId, HashSet<String>>,
        ready_targets: HashSet<String>,
    ) {
        *self.state.write() = SpeculationCompositionRegistryState {
            endpoint_groups,
            draft_dependents,
            ready_targets,
        };
    }

    pub(crate) fn counts(&self) -> (usize, usize, usize) {
        let state = self.state.read();
        (
            state.endpoint_groups.len(),
            state.draft_dependents.values().map(HashSet::len).sum(),
            state.ready_targets.len(),
        )
    }

    pub(crate) fn dependent_targets(&self, endpoint: &EndpointId) -> HashSet<String> {
        self.state
            .read()
            .draft_dependents
            .get(endpoint)
            .cloned()
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn group_count_for_endpoint(&self, endpoint: &EndpointId) -> usize {
        self.state
            .read()
            .endpoint_groups
            .get(endpoint)
            .map_or(0, HashSet::len)
    }

    #[cfg(test)]
    pub(crate) fn dependent_count_for_endpoint(&self, endpoint: &EndpointId) -> usize {
        self.state
            .read()
            .draft_dependents
            .get(endpoint)
            .map_or(0, HashSet::len)
    }

    #[cfg(test)]
    pub(crate) fn is_ready(&self, group_id: &str) -> bool {
        self.state.read().ready_targets.contains(group_id)
    }
}
