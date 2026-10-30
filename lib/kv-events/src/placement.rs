// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publisher-side helpers that turn a worker's placement decision into a [`RouterEvent`].

use serde::{Deserialize, Serialize};

use dynamo_router_protocol::residency::{ResidencyDomain, StorageTier};
use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};

use crate::event::{KvCacheEvent, RouterEvent};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PlacementOwner {
    LocalWorker(WorkerWithDpRank),
    Shared,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Placement {
    pub owner: PlacementOwner,
    pub tier: StorageTier,
    #[serde(default)]
    pub residency_domain: ResidencyDomain,
}

impl Placement {
    pub fn local_worker(worker_id: WorkerId, dp_rank: DpRank, tier: StorageTier) -> Self {
        Self {
            owner: PlacementOwner::LocalWorker(WorkerWithDpRank::new(worker_id, dp_rank)),
            tier,
            residency_domain: ResidencyDomain::Worker,
        }
    }

    pub fn local_gpu(worker_id: WorkerId, dp_rank: DpRank) -> Self {
        Self::local_worker(worker_id, dp_rank, StorageTier::Device)
    }

    pub fn is_local_gpu(&self) -> bool {
        matches!(self.owner, PlacementOwner::LocalWorker(_)) && self.tier.is_gpu()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlacementEvent {
    pub placement: Placement,
    pub event: KvCacheEvent,
    /// Session that triggered this store or reuse report, if provided by the engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl PlacementEvent {
    pub fn new(placement: Placement, event: KvCacheEvent) -> Self {
        Self {
            placement,
            event,
            session_id: None,
        }
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn local_gpu(worker_id: WorkerId, event: KvCacheEvent) -> Self {
        Self::new(Placement::local_gpu(worker_id, event.dp_rank), event)
    }

    pub fn into_router_event(self) -> Option<RouterEvent> {
        let PlacementOwner::LocalWorker(worker) = self.placement.owner else {
            return None;
        };
        let mut event = RouterEvent::with_residency_domain(
            worker.worker_id,
            self.event,
            self.placement.tier,
            self.placement.residency_domain,
        );
        event.session_id = self.session_id;
        Some(event)
    }
}
