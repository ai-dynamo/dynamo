// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend's requests in flight per pool, plus picks the core has
//! made that no request has taken over yet.

use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use dynamo_global_common::PoolId;
use dynamo_kv_router::WorkerId;

use crate::load::{InFlightGuard, InFlightTracker};

#[derive(Debug)]
pub(crate) struct FrontendSignals {
    own: Arc<InFlightTracker>,
    /// Pending picks by worker id.
    pending: DashMap<WorkerId, u64>,
}

impl FrontendSignals {
    /// `own` must be the tracker the Global Frontend counts its requests in.
    pub(crate) fn new(own: Arc<InFlightTracker>) -> Self {
        Self {
            own,
            pending: DashMap::new(),
        }
    }

    pub(crate) fn start(&self, pool: &PoolId) -> InFlightGuard {
        self.own.start(pool)
    }

    /// Requests in flight to `pool` (worker `worker`), pending picks included.
    pub(crate) fn in_flight(&self, pool: &PoolId, worker: WorkerId) -> u64 {
        let pending = self.pending.get(&worker).map_or(0, |count| *count);
        self.own.get(pool) + pending
    }

    pub(crate) fn add_pending(&self, worker: WorkerId) {
        *self.pending.entry(worker).or_default() += 1;
    }

    pub(crate) fn release_pending(&self, worker: WorkerId) {
        if let Entry::Occupied(mut entry) = self.pending.entry(worker) {
            *entry.get_mut() -= 1;
            if *entry.get() == 0 {
                entry.remove();
            }
        }
    }

    /// Turns a pending pick into the request's count. Counting first means a
    /// reader may briefly see it twice, never not at all.
    pub(crate) fn take_pending(&self, pool: &PoolId, worker: WorkerId) -> InFlightGuard {
        let guard = self.own.start(pool);
        self.release_pending(worker);
        guard
    }

    #[cfg(test)]
    pub(crate) fn pending(&self) -> u64 {
        self.pending.iter().map(|entry| *entry.value()).sum()
    }
}
