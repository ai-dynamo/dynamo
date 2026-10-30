// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache events: what an engine tells the router about its cache, and the router-side
//! envelope ([`RouterEvent`]) that carries one such event with its worker and residency owner.
//!
//! The engine-specific ZMQ encoding of these events is [`crate::zmq_wire`]; the replay cursor
//! that keeps a subscriber in step with a publisher is [`crate::cursor`].

use serde::{Deserialize, Serialize};

use dynamo_router_protocol::block_hash::{ExternalSequenceBlockHash, LocalBlockHash};
use dynamo_router_protocol::identity::CacheOwnerId;
use dynamo_router_protocol::mm::BlockExtraInfo;
use dynamo_router_protocol::residency::{
    ResetScope, ResidencyDomain, ResidencyOwner, StorageTier, UnsupportedResidencyDomain,
    WireResidencyDomain,
};
use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};

/// The event subject that workers publish KV cache events on.
pub const KV_EVENT_SUBJECT: &str = "kv-events";

/// Represents a collection of cache events and a shutdown flag.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheEvents {
    /// A list of cache events.
    pub events: Vec<KvCacheEvent>,
    /// A flag indicating whether the cache is shutting down.
    pub shutdown: bool,
}

/// Represents a single cache event with an ID and associated data.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KvCacheEvent {
    /// The unique identifier of the event.
    pub event_id: u64,
    /// The data associated with the event.
    pub data: KvCacheEventData,
    /// The data parallel rank of the worker emitting this event (0 if DP not enabled).
    #[serde(default)]
    pub dp_rank: DpRank,
}

/// Represents the data associated with a cache event.
///
/// Data is either stored or removed.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheEventData {
    Stored(KvCacheStoreData),
    Removed(KvCacheRemoveData),
    /// Remove all KV ownership for the emitting `(worker_id, dp_rank)`.
    ///
    /// This is ordered only within that rank publisher's event sequence. Worker-wide removal is
    /// a separate serving-membership lifecycle operation.
    Cleared,
}

/// Represents the data associated with a stored cache event.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KvCacheStoreData {
    /// The optional hash of the parent block.
    pub parent_hash: Option<ExternalSequenceBlockHash>,
    /// Absolute position of the first block in this batch for positional replay.
    pub start_position: Option<u32>,
    /// A list of stored blocked data.
    pub blocks: Vec<KvCacheStoredBlockData>,
}

/// Represents data for a stored block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KvCacheStoredBlockData {
    /// The hash of the block.
    pub block_hash: ExternalSequenceBlockHash,
    /// The hash of the tokens in the block.
    pub tokens_hash: LocalBlockHash,
    /// Extra multimodal metadata for this block
    /// Note: Do NOT use skip_serializing_if with bincode - it breaks deserialization
    /// because bincode is positional and expects all fields to be present.
    pub mm_extra_info: Option<BlockExtraInfo>,
}

/// Represents the data associated with a removed cache event.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KvCacheRemoveData {
    /// A list of block hashes to remove.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

// ------
// Router Event Types
// ------

/// Errors that can occur during KV Cache Event processing.
///
/// Indexer backends may introduce additional failure modes.
/// Downstream matches must include a wildcard arm because this enum is non-exhaustive.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum KvCacheEventError {
    #[error("Failed to find parent block")]
    ParentBlockNotFound,

    #[error("Failed to find block")]
    BlockNotFound,

    #[error("Invalid block sequence")]
    InvalidBlockSequence,

    /// A bounded physical-index omission; this does not prove the backing table is full.
    ///
    /// An indexer may still commit exact source lineage and ownership so removal and later
    /// re-admission remain correct.
    #[error("Indexer capacity exhausted")]
    CapacityExhausted,

    /// A pre-commit allocation or reservation failed; no lossy-capacity policy is implied.
    #[error("Indexer allocation failed")]
    AllocationFailed,

    /// An exact ownership degree overflowed before mutation.
    #[error("Indexer ownership degree overflow")]
    OwnershipDegreeOverflow,

    #[error("Indexer invariant violated")]
    IndexerInvariantViolation,

    #[error("Unsupported residency domain")]
    UnsupportedResidencyDomain,
}

/// A [`KvCacheEvent`] on a specific LLM worker denoted by [`WorkerId`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterEvent {
    /// The ID of the worker emitting the event.
    pub worker_id: WorkerId,
    /// The storage tier associated with the event.
    #[serde(default)]
    pub storage_tier: StorageTier,
    /// Optional logical residency owner. Absence is interpreted by event kind
    /// for compatibility with domainless publishers.
    #[serde(default, skip_serializing_if = "WireResidencyDomain::is_missing")]
    pub residency_domain: WireResidencyDomain,
    /// The cache event associated with the worker.
    pub event: KvCacheEvent,
    /// Stable state source that owns this event stream.
    ///
    /// This is absent on the legacy Worker-only wire. CacheOwner events are
    /// valid only on a versioned, residency-aware source where this field is
    /// present; they must never be sent to legacy consumers. MessagePack
    /// compatibility relies on `to_vec_named`; positional encoding is unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_source: Option<CacheOwnerId>,
    /// Session that triggered this store or reuse report, if provided by the engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl RouterEvent {
    /// Create a new `RouterEvent`.
    ///
    /// ### Arguments
    ///
    /// * `worker_id` - The ID of the worker emitting the event.
    /// * `event` - The cache event.
    ///
    /// ### Returns
    ///
    /// A new `RouterEvent`.
    pub fn new(worker_id: WorkerId, event: KvCacheEvent) -> Self {
        Self::with_storage_tier(worker_id, event, StorageTier::Device)
    }

    pub fn with_storage_tier(
        worker_id: WorkerId,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> Self {
        // Canonical in-process events are worker-owned, including clears. A
        // missing domain is reserved for decoding legacy wire payloads.
        Self::with_residency_domain(worker_id, event, storage_tier, ResidencyDomain::Worker)
    }

    pub fn with_residency_domain(
        worker_id: WorkerId,
        event: KvCacheEvent,
        storage_tier: StorageTier,
        residency_domain: ResidencyDomain,
    ) -> Self {
        Self {
            worker_id,
            state_source: None,
            session_id: None,
            storage_tier,
            residency_domain: WireResidencyDomain::explicit(residency_domain),
            event,
        }
    }

    /// Create a CacheOwner event with its required stable state source.
    pub fn with_cache_owner(
        worker_id: WorkerId,
        event: KvCacheEvent,
        storage_tier: StorageTier,
        cache_owner: CacheOwnerId,
    ) -> Self {
        Self::with_residency_domain(worker_id, event, storage_tier, ResidencyDomain::CacheOwner)
            .with_state_source(cache_owner)
    }

    /// Attach an explicit stable source to a residency-aware event.
    pub fn with_state_source(mut self, state_source: CacheOwnerId) -> Self {
        self.state_source = Some(state_source);
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Resolve a storage mutation to its canonical logical domain.
    pub fn resolved_residency_domain(&self) -> Result<ResidencyDomain, UnsupportedResidencyDomain> {
        let domain = self.residency_domain.parse()?.unwrap_or_default();
        // A domain-scoped clear has no physical residency tier. Raw clears use Device
        // as a wire placeholder; allow that sentinel without permitting CacheOwner
        // stores or removes in Device.
        if domain == ResidencyDomain::CacheOwner
            && self.storage_tier.is_gpu()
            && !matches!(self.event.data, KvCacheEventData::Cleared)
        {
            return Err(UnsupportedResidencyDomain);
        }
        Ok(domain)
    }

    pub fn residency_owner(&self) -> Result<ResidencyOwner, UnsupportedResidencyDomain> {
        Ok(match self.resolved_residency_domain()? {
            ResidencyDomain::Worker => {
                let worker = WorkerWithDpRank::new(self.worker_id, self.event.dp_rank);
                ResidencyOwner::worker(worker)
            }
            ResidencyDomain::CacheOwner => {
                ResidencyOwner::cache_owner(self.state_source.ok_or(UnsupportedResidencyDomain)?)
            }
        })
    }

    /// Resolve `Cleared` semantics while preserving legacy all-domain clears.
    pub fn reset_scope(&self) -> Result<Option<ResetScope>, UnsupportedResidencyDomain> {
        if !matches!(self.event.data, KvCacheEventData::Cleared) {
            return Ok(None);
        }
        Ok(Some(match self.residency_domain.parse()? {
            Some(domain) => ResetScope::Domain(domain),
            None => ResetScope::All,
        }))
    }

    pub fn targets_primary(&self) -> Result<bool, UnsupportedResidencyDomain> {
        if let Some(scope) = self.reset_scope()? {
            return Ok(!matches!(
                scope,
                ResetScope::Domain(ResidencyDomain::CacheOwner)
            ));
        }
        self.resolved_residency_domain()?;
        Ok(self.storage_tier.is_gpu())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_hashing::routing::compute_block_hash;

    mod residency_domain_compat {
        use super::*;

        #[derive(Serialize, Deserialize)]
        struct LegacyRouterEvent {
            worker_id: WorkerId,
            #[serde(default)]
            storage_tier: StorageTier,
            event: KvCacheEvent,
        }

        #[derive(Serialize)]
        struct ExplicitDomainEvent {
            worker_id: WorkerId,
            storage_tier: StorageTier,
            residency_domain: serde_json::Value,
            event: KvCacheEvent,
        }

        fn event(event_id: u64, data: KvCacheEventData) -> KvCacheEvent {
            KvCacheEvent {
                event_id,
                data,
                dp_rank: 0,
            }
        }

        fn stored(event_id: u64) -> KvCacheEvent {
            event(
                event_id,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: Vec::new(),
                }),
            )
        }

        #[test]
        fn named_messagepack_keeps_worker_only_mixed_versions_operational() {
            let legacy = vec![
                LegacyRouterEvent {
                    worker_id: 7,
                    storage_tier: StorageTier::HostPinned,
                    event: stored(1),
                },
                LegacyRouterEvent {
                    worker_id: 7,
                    storage_tier: StorageTier::Device,
                    event: event(2, KvCacheEventData::Cleared),
                },
            ];
            let decoded: Vec<RouterEvent> =
                rmp_serde::from_slice(&rmp_serde::to_vec_named(&legacy).unwrap()).unwrap();
            assert_eq!(decoded[0].session_id, None);
            assert_eq!(
                decoded[0].resolved_residency_domain(),
                Ok(ResidencyDomain::Worker)
            );
            assert_eq!(decoded[1].reset_scope(), Ok(Some(ResetScope::All)));

            let explicit_worker = RouterEvent::with_residency_domain(
                7,
                stored(3),
                StorageTier::Disk,
                ResidencyDomain::Worker,
            )
            .with_session_id("session-1");
            let old_reader: LegacyRouterEvent =
                rmp_serde::from_slice(&rmp_serde::to_vec_named(&explicit_worker).unwrap()).unwrap();
            assert_eq!(old_reader.event.event_id, 3);
            assert_eq!(old_reader.storage_tier, StorageTier::Disk);

            let round_trip: RouterEvent =
                rmp_serde::from_slice(&rmp_serde::to_vec_named(&explicit_worker).unwrap()).unwrap();
            assert_eq!(round_trip.session_id.as_deref(), Some("session-1"));

            let mixed = vec![
                ExplicitDomainEvent {
                    worker_id: 7,
                    storage_tier: StorageTier::HostPinned,
                    residency_domain: serde_json::json!("worker"),
                    event: stored(4),
                },
                ExplicitDomainEvent {
                    worker_id: 7,
                    storage_tier: StorageTier::HostPinned,
                    residency_domain: serde_json::json!("future_owner"),
                    event: stored(5),
                },
                ExplicitDomainEvent {
                    worker_id: 7,
                    storage_tier: StorageTier::HostPinned,
                    residency_domain: serde_json::Value::Null,
                    event: stored(6),
                },
                ExplicitDomainEvent {
                    worker_id: 7,
                    storage_tier: StorageTier::HostPinned,
                    residency_domain: serde_json::json!("worker"),
                    event: stored(7),
                },
            ];
            let decoded: Vec<RouterEvent> =
                rmp_serde::from_slice(&rmp_serde::to_vec_named(&mixed).unwrap()).unwrap();
            let applied_ids: Vec<_> = decoded
                .iter()
                .filter(|event| event.resolved_residency_domain().is_ok())
                .map(|event| event.event.event_id)
                .collect();
            assert_eq!(applied_ids, vec![4, 7]);
        }
    }

    #[test]
    fn test_router_event_new() {
        let worker_id = 0;
        let kv_cache_event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(0),
                    mm_extra_info: None,
                    tokens_hash: LocalBlockHash(13226331709069118873),
                }],
            }),
            dp_rank: 0,
        };
        let router_event = RouterEvent::new(worker_id, kv_cache_event);

        assert_eq!(router_event.worker_id, worker_id);
        assert_eq!(router_event.event.event_id, 1);
        if let KvCacheEventData::Stored(store_op) = &router_event.event.data {
            assert_eq!(store_op.blocks.len(), 1);
            assert_eq!(
                store_op.blocks[0].tokens_hash,
                compute_block_hash(b"test data")
            );
            assert_eq!(store_op.blocks[0].block_hash, ExternalSequenceBlockHash(0));
        } else {
            panic!("Expected KvCacheEventData::Stored");
        }
    }
}
