// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Residency owners projected onto workers: the compact owner key, the exact-owner
//! projection, and the per-lookup routing snapshot the lower tiers pin.

use rustc_hash::FxHashMap;

use dynamo_router_protocol::identity::CacheOwnerId;
use dynamo_router_protocol::metadata::RouterHintSourceMetadata;
use dynamo_router_protocol::residency::{ResidencyOwner, ResidencyOwnerKey};
use dynamo_router_protocol::worker::WorkerWithDpRank;

use crate::kv_hints::CacheOwnerKvTransferKeys;

/// Compact BLAKE3 keying of a residency owner. Hashing is router logic, so it lives here and
/// not in the wire crate that owns [`ResidencyOwner`].
pub trait ResidencyOwnerKeyExt {
    fn compact_key(self) -> ResidencyOwnerKey;
}

impl ResidencyOwnerKeyExt for ResidencyOwner {
    fn compact_key(self) -> ResidencyOwnerKey {
        let mut hasher = blake3::Hasher::new();
        match self {
            Self::Worker(worker) => {
                hasher.update(b"dynamo/residency-owner/worker/v1");
                hasher.update(&worker.worker_id.to_be_bytes());
                hasher.update(&worker.dp_rank.to_be_bytes());
            }
            Self::CacheOwner(cache_owner) => {
                hasher.update(b"dynamo/residency-owner/cache-owner/v1");
                update_cache_owner_hash(&mut hasher, cache_owner);
            }
        }
        let digest = hasher.finalize();
        ResidencyOwnerKey::from_digest(
            digest.as_bytes()[..16]
                .try_into()
                .expect("BLAKE3 output is 32 bytes"),
        )
    }
}

fn update_cache_owner_hash(hasher: &mut blake3::Hasher, owner: CacheOwnerId) {
    let domain = owner.pool().indexer_domain();
    hasher.update(&domain.cache_semantics().digest());
    hasher.update(&[domain.cache_semantics().source() as u8]);
    hasher.update(&domain.routing_scope().digest());
    hasher.update(&[domain.routing_scope().source() as u8]);
    hasher.update(&owner.pool().cluster_id().get().to_be_bytes());
    hasher.update(&owner.slot().digest());
    hasher.update(&[owner.slot().source() as u8]);
}

/// Immutable control-plane projection from exact residency owners to workers
/// that may currently consume those blocks.
///
/// Router-core stores no discovery state. The lib/llm wrapper resolves and
/// swaps this snapshot when source, attachment, or readability membership
/// changes; one snapshot is pinned for each tiered lookup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyProjection {
    exact_owners: FxHashMap<ResidencyOwnerKey, WorkerWithDpRank>,
}

impl ResidencyProjection {
    pub fn new(
        cache_owners: impl IntoIterator<Item = (CacheOwnerId, WorkerWithDpRank)>,
    ) -> Result<Self, ResidencyProjectionError> {
        let mut projection = Self::default();
        for (cache_owner, worker) in cache_owners {
            let key = ResidencyOwner::cache_owner(cache_owner).compact_key();
            match projection.exact_owners.insert(key, worker) {
                Some(existing) if existing != worker => {
                    return Err(ResidencyProjectionError::ConflictingAttachment {
                        cache_owner,
                        existing,
                        proposed: worker,
                    });
                }
                _ => {}
            }
        }
        Ok(projection)
    }

    #[inline]
    pub fn project(&self, owner: ResidencyOwner) -> Option<WorkerWithDpRank> {
        match owner {
            ResidencyOwner::Worker(worker) => Some(worker),
            ResidencyOwner::CacheOwner(_) => self.project_key(owner.compact_key()),
        }
    }

    #[inline]
    pub fn project_key(&self, key: ResidencyOwnerKey) -> Option<WorkerWithDpRank> {
        self.exact_owners.get(&key).copied()
    }

    pub fn cache_owner_worker(&self, cache_owner: CacheOwnerId) -> Option<WorkerWithDpRank> {
        self.project(ResidencyOwner::cache_owner(cache_owner))
    }

    pub fn is_empty(&self) -> bool {
        self.exact_owners.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRouterHintSource {
    pub metadata: RouterHintSourceMetadata,
    pub attached_worker: Option<WorkerWithDpRank>,
    pub kv_transfer_keys: CacheOwnerKvTransferKeys,
}

/// One immutable lookup snapshot for scheduling projection and persistent hint sources.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyRoutingSnapshot {
    projection: ResidencyProjection,
    router_hint_sources: FxHashMap<ResidencyOwnerKey, ResolvedRouterHintSource>,
}

impl ResidencyRoutingSnapshot {
    pub fn from_projection(projection: ResidencyProjection) -> Self {
        Self {
            projection,
            router_hint_sources: FxHashMap::default(),
        }
    }

    pub fn new(
        projection: ResidencyProjection,
        router_hint_sources: impl IntoIterator<
            Item = (
                CacheOwnerId,
                RouterHintSourceMetadata,
                Option<WorkerWithDpRank>,
            ),
        >,
    ) -> Self {
        let router_hint_sources = router_hint_sources
            .into_iter()
            .map(|(owner, metadata, attached_worker)| {
                (
                    ResidencyOwner::cache_owner(owner).compact_key(),
                    ResolvedRouterHintSource {
                        kv_transfer_keys: CacheOwnerKvTransferKeys::new(&metadata),
                        metadata,
                        attached_worker,
                    },
                )
            })
            .collect();
        Self {
            projection,
            router_hint_sources,
        }
    }

    pub fn projection(&self) -> &ResidencyProjection {
        &self.projection
    }

    pub fn router_hint_source(
        &self,
        owner: ResidencyOwnerKey,
    ) -> Option<&ResolvedRouterHintSource> {
        self.router_hint_sources.get(&owner)
    }

    pub fn has_router_hint_source(&self, owner: ResidencyOwnerKey) -> bool {
        self.router_hint_sources.contains_key(&owner)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ResidencyProjectionError {
    #[error("cache owner {cache_owner} is projected to both {existing:?} and {proposed:?}")]
    ConflictingAttachment {
        cache_owner: CacheOwnerId,
        existing: WorkerWithDpRank,
        proposed: WorkerWithDpRank,
    },
}
