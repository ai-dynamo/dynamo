// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::block::nixl::{BlockDescriptorList, SerializedNixlBlockSet};
use crate::tokens::{SequenceHash, compute_hash_v2};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

pub type G2pbChecksum = u64;

pub const G2PB_NAMESPACE: &str = "kvbm-g2pb";
pub const G2PB_COMPONENT_NAME: &str = "service";
pub const G2PB_ENDPOINT_NAME: &str = "g2pb";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbPeer {
    pub instance_id: u64,
    pub endpoint: String,
    pub hostname: String,
}

impl G2pbPeer {
    pub fn routing_id(&self) -> u64 {
        if self.hostname.is_empty() {
            return self.instance_id;
        }

        compute_hash_v2(self.hostname.as_bytes(), 0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbPutBlock {
    pub sequence_hash: SequenceHash,
    pub size_bytes: usize,
    pub checksum: G2pbChecksum,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbQueryHit {
    pub instance_id: u64,
    pub sequence_hash: SequenceHash,
    pub size_bytes: usize,
    pub checksum: G2pbChecksum,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbTransferBlock {
    pub meta: G2pbPutBlock,
    pub payload: Vec<u8>,
}

impl G2pbTransferBlock {
    pub fn compute_checksum(payload: &[u8]) -> G2pbChecksum {
        xxh3_64(payload)
    }

    fn validate_payload_size(&self) -> Result<(), G2pbError> {
        let actual_size_bytes = self.payload.len();
        if actual_size_bytes != self.meta.size_bytes {
            return Err(G2pbError::InvalidPayloadSize {
                sequence_hash: self.meta.sequence_hash,
                expected_size_bytes: self.meta.size_bytes,
                actual_size_bytes,
            });
        }

        Ok(())
    }

    fn with_computed_checksum(mut self) -> Self {
        self.meta.checksum = Self::compute_checksum(&self.payload);
        self
    }

    fn validate_payload_checksum(&self) -> Result<(), G2pbError> {
        let actual_checksum = Self::compute_checksum(&self.payload);
        if actual_checksum != self.meta.checksum {
            return Err(G2pbError::NotFound {
                instance_id: 0,
                sequence_hashes: vec![self.meta.sequence_hash],
            });
        }

        Ok(())
    }

    fn validate_payload_integrity(&self) -> Result<(), G2pbError> {
        self.validate_payload_size()?;
        self.validate_payload_checksum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbOfferRequest {
    pub blocks: Vec<G2pbPutBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbHealthResponse {
    pub instance_id: u64,
    pub listen: String,
    pub hostname: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbQueryRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbOfferResponse {
    pub accepted: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbPutPayloadRequest {
    pub blocks: Vec<G2pbTransferBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbFetchRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbFetchResponse {
    pub blocks: Vec<G2pbTransferBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G2pbRpcRequest {
    Health,
    PutBlocks(Vec<G2pbPutBlock>),
    Offer(G2pbOfferRequest),
    PutPayload(G2pbPutPayloadRequest),
    Query(G2pbQueryRequest),
    Fetch(G2pbFetchRequest),
    StagePut(G2pbStageBlocksRequest),
    CommitPut(G2pbCommitRequest),
    LoadRemote(G2pbLoadRemoteRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G2pbRpcResponse {
    Ack,
    Health(G2pbHealthResponse),
    Offer(G2pbOfferResponse),
    PutPayload(Vec<G2pbTransferBlock>),
    Query(Vec<G2pbQueryHit>),
    Fetch(G2pbFetchBlocksResponse),
    StagePut(G2pbStageBlocksResponse),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbStageBlocksRequest {
    pub blocks: Vec<G2pbPutBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pbStageBlocksResponse {
    pub instance_id: u64,
    pub blockset: SerializedNixlBlockSet,
    pub descriptors: BlockDescriptorList,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G2pbCommitRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pbLoadRemoteRequest {
    pub blockset: SerializedNixlBlockSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2pbFetchBlocksResponse {
    pub instance_id: u64,
    pub blockset: SerializedNixlBlockSet,
    pub descriptors: BlockDescriptorList,
}

#[derive(Debug, thiserror::Error)]
pub enum G2pbError {
    #[error("no live G2PB peers are available")]
    NoPeers,
    #[error("owning G2PB peer instance {instance_id} is not available")]
    UnknownPeer { instance_id: u64 },
    #[error(
        "requested G2PB blocks were not found on peer instance {instance_id}: {sequence_hashes:?}"
    )]
    NotFound {
        instance_id: u64,
        sequence_hashes: Vec<SequenceHash>,
    },
    #[error(
        "G2PB payload for sequence hash {sequence_hash} had size {actual_size_bytes}, expected {expected_size_bytes}"
    )]
    InvalidPayloadSize {
        sequence_hash: SequenceHash,
        expected_size_bytes: usize,
        actual_size_bytes: usize,
    },
}

mod g2pb_client;
mod g2pb_service;

pub use g2pb_client::{
    G2pbDiscoveredPeers, G2pbPeerInstance, G2pbPeerResolver, G2pbRequestPlaneClient,
    G2pbStorageClient, discover_g2pb_peers, route_g2pb_put_blocks_by_owner,
    route_g2pb_sequence_hashes_by_owner, route_g2pb_transfer_blocks_by_owner, select_g2pb_owner,
};
pub use g2pb_service::{
    G2pbCacheStorage, G2pbStorageAgent, G2pbStorageConfig,
};
#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::Instant;

    fn peers() -> Vec<G2pbPeer> {
        vec![
            G2pbPeer {
                instance_id: 101,
                endpoint: "tcp://peer-10".to_string(),
                hostname: "g2pb-10".to_string(),
            },
            G2pbPeer {
                instance_id: 202,
                endpoint: "tcp://peer-20".to_string(),
                hostname: "g2pb-20".to_string(),
            },
            G2pbPeer {
                instance_id: 303,
                endpoint: "tcp://peer-30".to_string(),
                hostname: "g2pb-30".to_string(),
            },
        ]
    }

    #[test]
    fn owner_selection_is_deterministic() {
        let peers = peers();
        let sequence_hash = 0xdead_beef_u64;

        let owner_a = select_g2pb_owner(sequence_hash, &peers).unwrap();

        let mut reversed = peers.clone();
        reversed.reverse();
        let owner_b = select_g2pb_owner(sequence_hash, &reversed).unwrap();

        assert_eq!(owner_a, owner_b);
    }

    #[test]
    fn route_sequence_hashes_groups_by_owner_and_preserves_owner_order() {
        let peer_list = peers();
        let sequence_hashes = vec![1_u64, 2_u64, 3_u64, 4_u64];

        let routed = route_g2pb_sequence_hashes_by_owner(&sequence_hashes, &peer_list).unwrap();

        for hashes in routed.values() {
            let mut expected = hashes.clone();
            expected.sort_by_key(|sequence_hash| {
                sequence_hashes
                    .iter()
                    .position(|candidate| candidate == sequence_hash)
                    .unwrap()
            });
            assert_eq!(*hashes, expected);
        }
    }

    #[test]
    fn discovered_peers_reject_duplicate_instance_ids() {
        let err = G2pbDiscoveredPeers::from_health_responses(vec![
            (
                101,
                G2pbHealthResponse {
                    instance_id: 10,
                    listen: "tcp://peer-10-a".to_string(),
                    hostname: "g2pb-10-a".to_string(),
                },
            ),
            (
                202,
                G2pbHealthResponse {
                    instance_id: 10,
                    listen: "tcp://peer-10-b".to_string(),
                    hostname: "g2pb-10-b".to_string(),
                },
            ),
        ])
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("duplicate remote instance_id 10 discovered")
        );
    }

    #[test]
    fn discovered_peers_sort_instances_by_routing_id() {
        let discovered = G2pbDiscoveredPeers::from_health_responses(vec![
            (
                303,
                G2pbHealthResponse {
                    instance_id: 30,
                    listen: "tcp://peer-30".to_string(),
                    hostname: "g2pb-30".to_string(),
                },
            ),
            (
                101,
                G2pbHealthResponse {
                    instance_id: 10,
                    listen: "tcp://peer-10".to_string(),
                    hostname: "g2pb-10".to_string(),
                },
            ),
            (
                202,
                G2pbHealthResponse {
                    instance_id: 20,
                    listen: "tcp://peer-20".to_string(),
                    hostname: "g2pb-20".to_string(),
                },
            ),
        ])
        .unwrap();

        let expected = {
            let mut instances = discovered.instances();
            instances.sort_by_key(|resolved| resolved.peer.routing_id());
            instances
                .into_iter()
                .map(|resolved| resolved.peer.instance_id)
                .collect::<Vec<_>>()
        };

        assert_eq!(
            discovered
                .instances()
                .into_iter()
                .map(|resolved| resolved.peer.instance_id)
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[tokio::test]
    async fn agent_query_and_fetch_use_in_memory_peer_cache() {
        let agent = G2pbStorageAgent::new(7);
        let checksum_11 = G2pbTransferBlock::compute_checksum(&[1, 2, 3, 4]);
        let checksum_12 = G2pbTransferBlock::compute_checksum(&[5, 6]);

        agent
            .offer_and_put_payload_blocks(vec![
                G2pbTransferBlock {
                    meta: G2pbPutBlock {
                        sequence_hash: 11,
                        size_bytes: 4,
                        checksum: checksum_11,
                    },
                    payload: vec![1, 2, 3, 4],
                },
                G2pbTransferBlock {
                    meta: G2pbPutBlock {
                        sequence_hash: 12,
                        size_bytes: 2,
                        checksum: checksum_12,
                    },
                    payload: vec![5, 6],
                },
            ])
            .await
            .unwrap();

        let hits = agent.query_blocks(&[11, 12, 13]).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].instance_id, 7);
        assert_eq!(hits[0].size_bytes, 4);
        assert_eq!(hits[1].size_bytes, 2);

        let fetched = agent.fetch_blocks(&[11, 12]).await.unwrap();
        assert_eq!(
            fetched,
            vec![
                G2pbTransferBlock {
                    meta: G2pbPutBlock {
                        sequence_hash: 11,
                        size_bytes: 4,
                        checksum: checksum_11,
                    },
                    payload: vec![1, 2, 3, 4],
                },
                G2pbTransferBlock {
                    meta: G2pbPutBlock {
                        sequence_hash: 12,
                        size_bytes: 2,
                        checksum: checksum_12,
                    },
                    payload: vec![5, 6],
                },
            ]
        );
    }

    #[tokio::test]
    async fn storage_rejects_payloads_with_mismatched_checksum() -> Result<()> {
        use super::g2pb_service::{G2pbPeerStorage, InMemoryG2pbPeerStorage};

        let storage = InMemoryG2pbPeerStorage::default();
        let err = G2pbPeerStorage::put_payload_blocks(
            &storage,
            vec![G2pbTransferBlock {
                meta: G2pbPutBlock {
                    sequence_hash: 77,
                    size_bytes: 4,
                    checksum: 123,
                },
                payload: vec![1, 2, 3, 4],
            }],
        )
        .await
        .unwrap_err();

        assert!(matches!(
            err,
            G2pbError::NotFound {
                instance_id: 0,
                sequence_hashes
            } if sequence_hashes == vec![77]
        ));

        Ok(())
    }

    #[tokio::test]
    async fn agent_fetch_reports_not_found() {
        let agent = G2pbStorageAgent::new(3);

        agent
            .put_blocks(vec![G2pbPutBlock {
                sequence_hash: 44,
                size_bytes: 8,
                checksum: 0,
            }])
            .await;

        let err = agent.fetch_blocks(&[44]).await.unwrap_err();
        assert!(matches!(
            err,
            G2pbError::NotFound {
                instance_id: 3,
                sequence_hashes
            } if sequence_hashes == vec![44]
        ));
    }

    #[tokio::test]
    async fn agent_offer_only_accepts_missing_blocks() {
        let agent = G2pbStorageAgent::new(7);
        agent
            .put_blocks(vec![G2pbPutBlock {
                sequence_hash: 100,
                size_bytes: 16,
                checksum: 0,
            }])
            .await;

        let accepted = agent
            .offered_blocks(vec![
                G2pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 16,
                    checksum: 0,
                },
                G2pbPutBlock {
                    sequence_hash: 200,
                    size_bytes: 32,
                    checksum: 0,
                },
                G2pbPutBlock {
                    sequence_hash: 200,
                    size_bytes: 32,
                    checksum: 0,
                },
            ])
            .await;

        assert_eq!(
            accepted,
            vec![G2pbPutBlock {
                sequence_hash: 200,
                size_bytes: 32,
                checksum: 0,
            }]
        );
    }

    #[tokio::test]
    async fn agent_offer_and_put_registers_only_missing_blocks() {
        let agent = G2pbStorageAgent::new(7);
        let accepted = agent
            .offer_and_put_blocks(vec![
                G2pbPutBlock {
                    sequence_hash: 5,
                    size_bytes: 16,
                    checksum: 0,
                },
                G2pbPutBlock {
                    sequence_hash: 6,
                    size_bytes: 32,
                    checksum: 9,
                },
                G2pbPutBlock {
                    sequence_hash: 6,
                    size_bytes: 32,
                    checksum: 9,
                },
            ])
            .await;

        assert_eq!(accepted.len(), 2);

        let hits = agent.query_blocks(&[5, 6]).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].sequence_hash, 5);
        assert_eq!(hits[1].sequence_hash, 6);
    }

    #[tokio::test]
    async fn agent_offer_payload_blocks_rejects_size_mismatch() {
        let agent = G2pbStorageAgent::new(7);

        let err = agent
            .offered_payload_blocks(vec![G2pbTransferBlock {
                meta: G2pbPutBlock {
                    sequence_hash: 5,
                    size_bytes: 4,
                    checksum: 0,
                },
                payload: vec![1, 2, 3],
            }])
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            G2pbError::InvalidPayloadSize {
                sequence_hash: 5,
                expected_size_bytes: 4,
                actual_size_bytes: 3
            }
        ));
    }

    #[tokio::test]
    async fn client_put_and_query_route_by_owner() {
        let peer_list = peers();
        let owner = select_g2pb_owner(1234, &peer_list).unwrap();
        let agent = Arc::new(G2pbStorageAgent::new(owner.instance_id));
        let client = G2pbStorageClient::new(
            peer_list,
            HashMap::from_iter([(owner.instance_id, agent.clone())]),
        );

        client
            .put_blocks(vec![G2pbPutBlock {
                sequence_hash: 1234,
                size_bytes: 64,
                checksum: 7,
            }])
            .await
            .unwrap();

        let hits = client.query_blocks(&[1234, 9999]).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[&1234].instance_id, owner.instance_id);
        assert_eq!(hits[&1234].size_bytes, 64);
    }

    #[tokio::test]
    async fn client_offer_routes_by_owner_and_preserves_input_order() {
        let peer_list = peers();
        let mut agents = HashMap::new();

        for peer in &peer_list {
            agents.insert(
                peer.instance_id,
                Arc::new(G2pbStorageAgent::new(peer.instance_id)),
            );
        }

        let existing_hash = 200_u64;
        let existing_owner = select_g2pb_owner(existing_hash, &peer_list).unwrap();
        agents[&existing_owner.instance_id]
            .put_blocks(vec![G2pbPutBlock {
                sequence_hash: existing_hash,
                size_bytes: 8,
                checksum: 0,
            }])
            .await;

        let client = G2pbStorageClient::new(peer_list, agents);
        let accepted = client
            .offer_blocks(vec![
                G2pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 8,
                    checksum: 0,
                },
                G2pbPutBlock {
                    sequence_hash: existing_hash,
                    size_bytes: 8,
                    checksum: 0,
                },
                G2pbPutBlock {
                    sequence_hash: 300,
                    size_bytes: 8,
                    checksum: 0,
                },
                G2pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 8,
                    checksum: 0,
                },
            ])
            .await
            .unwrap();

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.sequence_hash)
                .collect::<Vec<_>>(),
            vec![100, 300]
        );
    }

    #[tokio::test]
    async fn client_offer_and_put_payload_rejects_duplicate_hashes_within_batch() {
        let peer_list = peers();
        let mut agents = HashMap::new();
        for peer in &peer_list {
            agents.insert(
                peer.instance_id,
                Arc::new(G2pbStorageAgent::new(peer.instance_id)),
            );
        }

        let client = G2pbStorageClient::new(peer_list.clone(), agents.clone());
        let accepted = client
            .offer_and_put_payload_blocks(vec![
                G2pbTransferBlock {
                    meta: G2pbPutBlock {
                        sequence_hash: 1,
                        size_bytes: 2,
                        checksum: 0,
                    },
                    payload: vec![1, 2],
                },
                G2pbTransferBlock {
                    meta: G2pbPutBlock {
                        sequence_hash: 2,
                        size_bytes: 2,
                        checksum: 0,
                    },
                    payload: vec![3, 4],
                },
                G2pbTransferBlock {
                    meta: G2pbPutBlock {
                        sequence_hash: 1,
                        size_bytes: 2,
                        checksum: 0,
                    },
                    payload: vec![5, 6],
                },
            ])
            .await
            .unwrap();

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.meta.sequence_hash)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        let fetched = client.fetch_blocks(&[1, 2]).await;
        assert_eq!(
            fetched
                .iter()
                .map(|block| block.meta.sequence_hash)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[tokio::test]
    async fn client_query_blocks_fans_out_across_owners_concurrently() {
        let peer_list = peers();
        let sequence_hashes = vec![100_u64, 200_u64, 300_u64];
        let mut agents = HashMap::new();

        for sequence_hash in &sequence_hashes {
            let owner = select_g2pb_owner(*sequence_hash, &peer_list).unwrap();
            agents.entry(owner.instance_id).or_insert_with(|| {
                Arc::new(
                    G2pbStorageAgent::new(owner.instance_id)
                        .with_query_delay(Duration::from_millis(75)),
                )
            });
            agents[&owner.instance_id]
                .put_blocks(vec![G2pbPutBlock {
                    sequence_hash: *sequence_hash,
                    size_bytes: 8,
                    checksum: 0,
                }])
                .await;
        }

        let client = G2pbStorageClient::new(peer_list, agents);
        let start = Instant::now();
        let hits = client.query_blocks(&sequence_hashes).await;
        let elapsed = start.elapsed();

        assert_eq!(hits.len(), sequence_hashes.len());
        assert!(elapsed < Duration::from_millis(200));
    }

    #[tokio::test]
    async fn client_treats_missing_fetches_as_cache_miss() {
        let peer_list = peers();
        let owner = select_g2pb_owner(1234, &peer_list).unwrap();
        let agent = Arc::new(G2pbStorageAgent::new(owner.instance_id));
        agent
            .put_blocks(vec![G2pbPutBlock {
                sequence_hash: 1234,
                size_bytes: 4,
                checksum: 0,
            }])
            .await;

        let client =
            G2pbStorageClient::new(peer_list, HashMap::from_iter([(owner.instance_id, agent)]));
        let fetched = client.fetch_blocks(&[1234]).await;
        assert!(fetched.is_empty());
    }

    #[tokio::test]
    async fn g2pb_cache_storage_supports_basic_operations() -> Result<()> {
        let mut config = G2pbStorageConfig::new(0);
        config.g2_capacity_bytes = 4 * 1024;

        let storage = Arc::new(G2pbCacheStorage::new(config).await?);
        let agent = G2pbStorageAgent::new_with_storage(77, storage.clone());

        agent
            .offer_and_put_payload_blocks(vec![G2pbTransferBlock {
                meta: G2pbPutBlock {
                    sequence_hash: 1001,
                    size_bytes: 8,
                    checksum: 0,
                },
                payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }])
            .await?;

        let hits = agent.query_blocks(&[1001]).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].sequence_hash, 1001);
        assert_eq!(hits[0].size_bytes, 8);

        // Test offer blocks
        let accepted = agent
            .offer_blocks(&[
                G2pbPutBlock {
                    sequence_hash: 1001,
                    size_bytes: 8,
                    checksum: 0,
                },
                G2pbPutBlock {
                    sequence_hash: 2001,
                    size_bytes: 16,
                    checksum: 0,
                },
            ])
            .await;
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0], 2001);

        agent.delete_blocks(&[1001]).await?;
        assert!(agent.query_blocks(&[1001]).await.is_empty());
        assert!(matches!(
            agent.fetch_blocks(&[1001]).await,
            Err(G2pbError::NotFound {
                instance_id: 77,
                sequence_hashes,
            }) if sequence_hashes == vec![1001]
        ));

        Ok(())
    }
}
