// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::Instant;

use super::AffinityTarget;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct AffinityRevision {
    pub(super) sequence: u64,
    pub(super) router_id: u64,
}

impl AffinityRevision {
    // Compatibility with v1.2-v1.3 routers during v1.4-v1.5 rolling upgrades.
    // TODO(v1.6): Remove revision-zero handling when those routers leave the N-2 window.
    pub(super) const fn is_legacy(self) -> bool {
        self.sequence == 0
    }

    pub(super) const fn is_versioned(self) -> bool {
        !self.is_legacy()
    }
}

#[derive(Clone, Copy)]
pub(super) struct ReplicaBinding {
    pub(super) target: AffinityTarget,
    pub(super) revision: AffinityRevision,
    pub(super) legacy_fence: Option<LegacyFence>,
}

#[derive(Clone, Copy)]
pub(super) struct LegacyFence {
    pub(super) deadline: Instant,
    pub(super) pending_versioned_target: Option<AffinityTarget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReplicaApplyOutcome {
    Inserted,
    Refreshed,
    ReplacedLegacy,
    ReplacedNewer,
    DeferredInitializing,
    DeferredLegacy,
    IgnoredStale,
    RejectedSessionId,
    RejectedCapacity,
}

pub(super) fn apply_replica_binding(
    binding: &mut ReplicaBinding,
    target: AffinityTarget,
    revision: AffinityRevision,
    now: Instant,
    ttl: Duration,
) -> ReplicaApplyOutcome {
    normalize_expired_replica_fence(binding, now);
    if revision.is_legacy() {
        let refreshed = binding.target == target && binding.legacy_fence.is_some();
        let pending_versioned_target = binding
            .legacy_fence
            .and_then(|fence| fence.pending_versioned_target)
            .or(binding.revision.is_versioned().then_some(binding.target));
        binding.target = target;
        binding.legacy_fence = Some(LegacyFence {
            deadline: now + ttl,
            pending_versioned_target,
        });
        return if refreshed {
            ReplicaApplyOutcome::Refreshed
        } else {
            ReplicaApplyOutcome::ReplacedLegacy
        };
    }

    if let Some(fence) = binding.legacy_fence.as_mut() {
        if revision <= binding.revision {
            return ReplicaApplyOutcome::IgnoredStale;
        }
        binding.revision = revision;
        fence.pending_versioned_target = Some(target);
        return ReplicaApplyOutcome::DeferredLegacy;
    }

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

pub(super) fn normalize_expired_replica_fence(binding: &mut ReplicaBinding, now: Instant) {
    normalize_expired_legacy_fence(&mut binding.target, &mut binding.legacy_fence, now);
}

pub(super) fn normalize_expired_legacy_fence(
    target: &mut AffinityTarget,
    legacy_fence: &mut Option<LegacyFence>,
    now: Instant,
) {
    let Some(fence) = *legacy_fence else {
        return;
    };
    if fence.deadline > now {
        return;
    }
    if let Some(pending_target) = fence.pending_versioned_target {
        *target = pending_target;
    }
    *legacy_fence = None;
}

pub(super) fn revision_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX - 1)
}
