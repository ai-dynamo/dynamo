// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{SystemTime, UNIX_EPOCH};

use super::AffinityTarget;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct AffinityRevision {
    pub(super) sequence: u64,
    pub(super) router_id: u64,
}

#[derive(Clone, Copy)]
pub(super) struct ReplicaBinding {
    pub(super) target: AffinityTarget,
    pub(super) revision: AffinityRevision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplicaApplyOutcome {
    Inserted,
    Refreshed,
    ReplacedNewer,
    DeferredInitializing,
    IgnoredStale,
    RejectedSessionId,
    RejectedCapacity,
}

pub(super) fn apply_replica_binding(
    binding: &mut ReplicaBinding,
    target: AffinityTarget,
    revision: AffinityRevision,
) -> ReplicaApplyOutcome {
    if revision > binding.revision {
        binding.target = target;
        binding.revision = revision;
        return ReplicaApplyOutcome::ReplacedNewer;
    }
    if revision == binding.revision && target == binding.target {
        return ReplicaApplyOutcome::Refreshed;
    }
    ReplicaApplyOutcome::IgnoredStale
}

pub(super) fn revision_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX - 1)
}
