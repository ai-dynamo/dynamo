// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owner identities for the lower tier's storage tests.

use dynamo_router_protocol::identity::{
    CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
    RoutingScopeId, StableDpSlotId,
};

use super::IndexedResidencyOwner;
use crate::prelude::{ResidencyOwner, WorkerWithDpRank};

pub(super) fn cache_owner(slot: u8) -> CacheOwnerId {
    CacheOwnerId::new(
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::Explicit),
            ),
            ClusterId::new(3),
        ),
        StableDpSlotId::new([slot; 16], IdentitySource::Explicit),
    )
}

/// `count` owners: even indexes are workers, odd indexes are cache owners.
pub(super) fn residency_owners(count: usize) -> Vec<ResidencyOwner> {
    (0..count)
        .map(|idx| {
            if idx % 2 == 0 {
                ResidencyOwner::worker(WorkerWithDpRank::new(100 + idx as u64, 0))
            } else {
                ResidencyOwner::cache_owner(cache_owner(idx as u8))
            }
        })
        .collect()
}

pub(super) fn indexed_owners(count: usize) -> Vec<IndexedResidencyOwner> {
    residency_owners(count)
        .into_iter()
        .map(IndexedResidencyOwner::from_exact)
        .collect()
}
