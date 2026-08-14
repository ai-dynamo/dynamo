// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery contracts for persistent KV state sources and engine attachments.
//!
//! Advertisements are immutable hints, not liveness. Reconciliation activates
//! a projection only after the advertised recovery/control endpoint returns an
//! exact matching status and recovery reaches the attachment cursor barrier.

use std::collections::HashSet;

use dynamo_kv_router::{
    identity::{CacheOwnerId, IndexerDomainId},
    indexer::{
        KvStateAgentIdentity, KvStateAgentStatus, KvStateProtocolVersion, KvStateRecoveryReceipt,
    },
    protocols::{ResidencyProjection, ResidencyProjectionError, WorkerWithDpRank},
};
use dynamo_runtime::component::Instance;
use serde::{Deserialize, Serialize};

use super::PublisherId;
use crate::kv_router::indexer::Indexer;

pub const KV_STATE_HOST_TOPIC_V2: &str = "kv-state-hosts-v2";
pub const KV_STATE_ATTACHMENT_INTENT_TOPIC_V2: &str = "kv-state-attachment-intents-v2";
pub const KV_STATE_SOURCE_TOPIC_V2: &str = "kv-state-sources-v2";
pub const KV_STATE_ATTACHMENT_TOPIC_V2: &str = "kv-state-attachments-v2";
pub const KV_STATE_EVENT_TOPIC_V2: &str = "kv-state-events-v2";

/// Immutable raw ingress contract selected for one stable slot.
///
/// This is deliberately independent of [`KvStateProtocolVersion`], which
/// versions the Dynamo discovery, recovery, and event protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvStateIngressProtocol {
    FrameworkV1,
    VllmResidencyV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStateHostAdvertisement {
    pub protocol_version: KvStateProtocolVersion,
    pub host_instance: Instance,
    pub control_target: Instance,
    pub max_slots: usize,
}

/// Lease-owned request to attach one producer-owned raw stream to a host slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStateAttachmentIntent {
    pub target_host: Instance,
    pub producer_instance: Instance,
    pub intent_incarnation: u64,
    pub cache_owner_id: CacheOwnerId,
    pub worker: WorkerWithDpRank,
    pub kv_state_endpoint: dynamo_runtime::protocols::EndpointId,
    pub indexer_domain_id: IndexerDomainId,
    pub kv_block_size: u32,
    pub ingress_protocol: KvStateIngressProtocol,
    pub raw_zmq_endpoint: String,
    pub raw_topic: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_token_id: Option<u32>,
    pub cache_readable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvStateHostControlRequest {
    Status,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStateHostStatus {
    pub healthy: bool,
    pub total_slots: usize,
    pub active_slots: usize,
    pub detached_slots: usize,
    pub failed_slots: usize,
    pub capacity_rejected_total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl dynamo_runtime::protocols::maybe_error::MaybeError for KvStateHostStatus {
    fn from_err(error: impl std::error::Error + 'static) -> Self {
        Self {
            healthy: false,
            error: Some(error.to_string()),
            ..Default::default()
        }
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        self.error
            .as_ref()
            .map(|error| dynamo_runtime::error::DynamoError::msg(error.clone()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStateSourceAdvertisement {
    pub cache_owner_id: CacheOwnerId,
    pub global_dp_rank: u32,
    pub kv_state_endpoint: dynamo_runtime::protocols::EndpointId,
    pub indexer_domain_id: IndexerDomainId,
    pub kv_block_size: u32,
    pub ingress_protocol: KvStateIngressProtocol,
    pub publisher_id: PublisherId,
    pub protocol_version: KvStateProtocolVersion,
    pub event_topic: String,
    pub recovery_control_target: Instance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStateAttachmentAdvertisement {
    pub cache_owner_id: CacheOwnerId,
    pub publisher_id: PublisherId,
    pub protocol_version: KvStateProtocolVersion,
    pub recovery_control_target: Instance,
    pub attachment_generation: u64,
    pub producer_instance: Instance,
    pub intent_incarnation: u64,
    pub worker: WorkerWithDpRank,
    pub ingress_protocol: KvStateIngressProtocol,
    pub raw_zmq_endpoint: String,
    pub raw_topic: String,
    pub cache_readable: bool,
    pub ready_at_outbound_cursor: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvStateUnknownReason {
    WatchUncertain,
    AmbiguousSource,
    AmbiguousAttachment,
    EndpointUnreachable,
    StatusMismatch,
    RecoveryIdentityMismatch,
    RecoveryBehindBarrier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvStateProjectionResolution {
    Unavailable,
    Unknown(KvStateUnknownReason),
    Ready {
        cache_owner_id: CacheOwnerId,
        worker: WorkerWithDpRank,
        ready_at_outbound_cursor: u64,
    },
}

impl KvStateProjectionResolution {
    fn ready_mapping(&self) -> Option<(CacheOwnerId, WorkerWithDpRank)> {
        match self {
            Self::Ready {
                cache_owner_id,
                worker,
                ..
            } => Some((*cache_owner_id, *worker)),
            Self::Unavailable | Self::Unknown(_) => None,
        }
    }
}

/// Model-level owner of the one immutable projection snapshot consumed by lookups.
pub struct KvStateProjectionController {
    indexer: Indexer,
}

impl KvStateProjectionController {
    pub fn new(indexer: Indexer) -> Self {
        Self { indexer }
    }

    pub fn publish<'a>(
        &self,
        resolutions: impl IntoIterator<Item = &'a KvStateProjectionResolution>,
    ) -> Result<(), ResidencyProjectionError> {
        let projection = aggregate_residency_projection(resolutions)?;
        self.indexer.set_residency_projection(projection);
        Ok(())
    }
}

pub fn aggregate_residency_projection<'a>(
    resolutions: impl IntoIterator<Item = &'a KvStateProjectionResolution>,
) -> Result<ResidencyProjection, ResidencyProjectionError> {
    ResidencyProjection::new(
        resolutions
            .into_iter()
            .filter_map(KvStateProjectionResolution::ready_mapping),
    )
}

/// Resolve one cache owner's eligibility from an authoritative discovery
/// snapshot and its callable status result.
///
/// `None` status means timeout or endpoint failure. It is deliberately
/// Unknown rather than deletion: callers unproject immediately and retry, but
/// never clear retained CacheOwner state from this outcome.
pub fn resolve_kv_state_projection(
    discovery_known: bool,
    cache_owner_id: CacheOwnerId,
    sources: &[KvStateSourceAdvertisement],
    attachments: &[KvStateAttachmentAdvertisement],
    live_workers: &HashSet<WorkerWithDpRank>,
    status: Option<&KvStateAgentStatus>,
    recovery_receipt: Option<&KvStateRecoveryReceipt>,
) -> KvStateProjectionResolution {
    if !discovery_known {
        return KvStateProjectionResolution::Unknown(KvStateUnknownReason::WatchUncertain);
    }

    let matching_sources: Vec<_> = sources
        .iter()
        .filter(|source| source.cache_owner_id == cache_owner_id)
        .collect();
    let [source] = matching_sources.as_slice() else {
        return if matching_sources.is_empty() {
            KvStateProjectionResolution::Unavailable
        } else {
            KvStateProjectionResolution::Unknown(KvStateUnknownReason::AmbiguousSource)
        };
    };

    let matching_attachments: Vec<_> = attachments
        .iter()
        .filter(|attachment| attachment.cache_owner_id == cache_owner_id)
        .collect();
    let [attachment] = matching_attachments.as_slice() else {
        return if matching_attachments.is_empty() {
            KvStateProjectionResolution::Unavailable
        } else {
            KvStateProjectionResolution::Unknown(KvStateUnknownReason::AmbiguousAttachment)
        };
    };

    if !live_workers.contains(&attachment.worker) || !attachment.cache_readable {
        return KvStateProjectionResolution::Unavailable;
    }

    let Some(status) = status else {
        return KvStateProjectionResolution::Unknown(KvStateUnknownReason::EndpointUnreachable);
    };
    let status_attachment = status.attachment.as_ref();
    if status.identity.cache_owner_id != cache_owner_id
        || !status.cache_owner_ready
        || attachment.publisher_id != source.publisher_id
        || status.identity.publisher_id != source.publisher_id
        || status.identity.protocol_version != source.protocol_version
        || source.protocol_version != attachment.protocol_version
        || source.event_topic != KV_STATE_EVENT_TOPIC_V2
        || source.recovery_control_target != attachment.recovery_control_target
        || status_attachment.is_none_or(|current| {
            current.generation != attachment.attachment_generation
                || current.worker != attachment.worker
                || !current.ready
                || !current.cache_readable
                || current.ready_at_outbound_cursor != attachment.ready_at_outbound_cursor
        })
    {
        return KvStateProjectionResolution::Unknown(KvStateUnknownReason::StatusMismatch);
    }

    let expected_identity = KvStateAgentIdentity {
        cache_owner_id,
        publisher_id: source.publisher_id,
        protocol_version: source.protocol_version,
    };
    let Some(recovery_receipt) = recovery_receipt else {
        return KvStateProjectionResolution::Unknown(KvStateUnknownReason::RecoveryBehindBarrier);
    };
    if recovery_receipt.identity != expected_identity
        || recovery_receipt.attachment_generation != Some(attachment.attachment_generation)
    {
        return KvStateProjectionResolution::Unknown(
            KvStateUnknownReason::RecoveryIdentityMismatch,
        );
    }
    if status.outbound_cursor < attachment.ready_at_outbound_cursor
        || recovery_receipt.recovered_through_cursor < attachment.ready_at_outbound_cursor
    {
        return KvStateProjectionResolution::Unknown(KvStateUnknownReason::RecoveryBehindBarrier);
    }

    KvStateProjectionResolution::Ready {
        cache_owner_id,
        worker: attachment.worker,
        ready_at_outbound_cursor: attachment.ready_at_outbound_cursor,
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
        StableDpSlotId,
    };
    use dynamo_kv_router::indexer::{KvStateAgentIdentity, KvStateAttachmentStatus};
    use dynamo_runtime::component::TransportType;

    use super::*;

    fn owner() -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        )
    }

    fn endpoint() -> Instance {
        Instance {
            component: "worker".to_string(),
            endpoint: "state-agent".to_string(),
            namespace: "ns".to_string(),
            instance_id: 17,
            transport: TransportType::Tcp("tcp://127.0.0.1:1234".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
    }

    #[test]
    fn projection_requires_exact_status_and_recovery_barrier() {
        let cache_owner_id = owner();
        let worker = WorkerWithDpRank::new(17, 3);
        let identity = KvStateAgentIdentity {
            cache_owner_id,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
        };
        let source = KvStateSourceAdvertisement {
            cache_owner_id,
            global_dp_rank: worker.dp_rank,
            kv_state_endpoint: "ns.worker.generate".into(),
            indexer_domain_id: cache_owner_id.pool().indexer_domain(),
            kv_block_size: 4,
            ingress_protocol: KvStateIngressProtocol::VllmResidencyV1,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
            event_topic: KV_STATE_EVENT_TOPIC_V2.to_string(),
            recovery_control_target: endpoint(),
        };
        let attachment = KvStateAttachmentAdvertisement {
            cache_owner_id,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
            recovery_control_target: endpoint(),
            attachment_generation: 7,
            producer_instance: endpoint(),
            intent_incarnation: 71,
            worker,
            ingress_protocol: KvStateIngressProtocol::VllmResidencyV1,
            raw_zmq_endpoint: "tcp://framework".to_string(),
            raw_topic: "kv-events-v2".to_string(),
            cache_readable: true,
            ready_at_outbound_cursor: 9,
        };
        let status = KvStateAgentStatus {
            identity: identity.clone(),
            attachment: Some(KvStateAttachmentStatus {
                generation: 7,
                worker,
                ready: true,
                cache_readable: true,
                ready_at_outbound_cursor: 9,
            }),
            cache_owner_ready: true,
            outbound_cursor: 12,
        };
        let live = HashSet::from([worker]);
        let receipt = KvStateRecoveryReceipt {
            identity: identity.clone(),
            attachment_generation: Some(7),
            recovered_through_cursor: 8,
        };

        assert_eq!(
            resolve_kv_state_projection(
                true,
                cache_owner_id,
                std::slice::from_ref(&source),
                std::slice::from_ref(&attachment),
                &live,
                Some(&status),
                Some(&receipt),
            ),
            KvStateProjectionResolution::Unknown(KvStateUnknownReason::RecoveryBehindBarrier)
        );
        let receipt = KvStateRecoveryReceipt {
            recovered_through_cursor: 9,
            ..receipt
        };
        assert!(matches!(
            resolve_kv_state_projection(
                true,
                cache_owner_id,
                std::slice::from_ref(&source),
                std::slice::from_ref(&attachment),
                &live,
                Some(&status),
                Some(&receipt),
            ),
            KvStateProjectionResolution::Ready { worker: ready, .. } if ready == worker
        ));

        let wrong_publisher_receipt = KvStateRecoveryReceipt {
            identity: KvStateAgentIdentity {
                publisher_id: 42,
                ..identity
            },
            attachment_generation: Some(7),
            recovered_through_cursor: 12,
        };
        assert_eq!(
            resolve_kv_state_projection(
                true,
                cache_owner_id,
                std::slice::from_ref(&source),
                std::slice::from_ref(&attachment),
                &live,
                Some(&status),
                Some(&wrong_publisher_receipt),
            ),
            KvStateProjectionResolution::Unknown(KvStateUnknownReason::RecoveryIdentityMismatch)
        );

        let second_owner = CacheOwnerId::new(
            cache_owner_id.pool(),
            StableDpSlotId::new([5; 16], IdentitySource::Explicit),
        );
        let second_worker = WorkerWithDpRank::new(18, 4);
        let resolutions = [
            KvStateProjectionResolution::Ready {
                cache_owner_id,
                worker,
                ready_at_outbound_cursor: 9,
            },
            KvStateProjectionResolution::Ready {
                cache_owner_id: second_owner,
                worker: second_worker,
                ready_at_outbound_cursor: 4,
            },
        ];
        let projection = aggregate_residency_projection(&resolutions).unwrap();
        assert_eq!(projection.cache_owner_worker(cache_owner_id), Some(worker));
        assert_eq!(
            projection.cache_owner_worker(second_owner),
            Some(second_worker)
        );

        let mut stale = attachment.clone();
        stale.attachment_generation = 8;
        assert_eq!(
            resolve_kv_state_projection(
                true,
                cache_owner_id,
                std::slice::from_ref(&source),
                &[attachment, stale],
                &live,
                Some(&status),
                Some(&receipt),
            ),
            KvStateProjectionResolution::Unknown(KvStateUnknownReason::AmbiguousAttachment)
        );
        assert_eq!(
            resolve_kv_state_projection(false, cache_owner_id, &[source], &[], &live, None, None,),
            KvStateProjectionResolution::Unknown(KvStateUnknownReason::WatchUncertain)
        );
    }
}
