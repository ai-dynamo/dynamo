// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::block::nixl::{BlockDescriptorList, SerializedNixlBlockSet};
use crate::tokens::{SequenceHash, compute_hash_v2};

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const G3PB_NAMESPACE: &str = "kvbm-g3pb";
pub const G3PB_COMPONENT_NAME: &str = "peer-cache";
pub const G3PB_ENDPOINT_NAME: &str = "g3pb";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPeer {
    pub instance_id: u64,
    pub endpoint: String,
    pub hostname: String,
}

impl G3pbPeer {
    pub fn routing_id(&self) -> u64 {
        if self.hostname.is_empty() {
            return self.instance_id;
        }

        compute_hash_v2(self.hostname.as_bytes(), 0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPutBlock {
    pub sequence_hash: SequenceHash,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbQueryHit {
    pub instance_id: u64,
    pub sequence_hash: SequenceHash,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbTransferBlock {
    pub meta: G3pbPutBlock,
    pub payload: Vec<u8>,
}

impl G3pbTransferBlock {
    fn validate_payload_size(&self) -> Result<(), G3pbError> {
        let actual_size_bytes = self.payload.len();
        if actual_size_bytes != self.meta.size_bytes {
            return Err(G3pbError::InvalidPayloadSize {
                sequence_hash: self.meta.sequence_hash,
                expected_size_bytes: self.meta.size_bytes,
                actual_size_bytes,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbOfferRequest {
    pub blocks: Vec<G3pbPutBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbHealthResponse {
    pub instance_id: u64,
    pub listen: String,
    pub hostname: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbQueryRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbOfferResponse {
    pub accepted: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPutPayloadRequest {
    pub blocks: Vec<G3pbTransferBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbFetchRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbFetchResponse {
    pub blocks: Vec<G3pbTransferBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G3pbRpcRequest {
    Health,
    PutBlocks(Vec<G3pbPutBlock>),
    Offer(G3pbOfferRequest),
    PutPayload(G3pbPutPayloadRequest),
    Query(G3pbQueryRequest),
    Fetch(G3pbFetchRequest),
    StagePut(G3pbStageBlocksRequest),
    CommitPut(G3pbCommitRequest),
    LoadRemote(G3pbLoadRemoteRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G3pbRpcResponse {
    Ack,
    Health(G3pbHealthResponse),
    Offer(G3pbOfferResponse),
    PutPayload(Vec<G3pbTransferBlock>),
    Query(Vec<G3pbQueryHit>),
    Fetch(G3pbFetchBlocksResponse),
    StagePut(G3pbStageBlocksResponse),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbStageBlocksRequest {
    pub blocks: Vec<G3pbPutBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3pbStageBlocksResponse {
    pub instance_id: u64,
    pub blockset: SerializedNixlBlockSet,
    pub descriptors: BlockDescriptorList,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbCommitRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3pbLoadRemoteRequest {
    pub blockset: SerializedNixlBlockSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3pbFetchBlocksResponse {
    pub instance_id: u64,
    pub blockset: SerializedNixlBlockSet,
    pub descriptors: BlockDescriptorList,
}

#[derive(Debug, thiserror::Error)]
pub enum G3pbError {
    #[error("no live G3PB peers are available")]
    NoPeers,
    #[error("owning G3PB peer instance {instance_id} is not available")]
    UnknownPeer { instance_id: u64 },
    #[error(
        "requested G3PB blocks were not found on peer instance {instance_id}: {sequence_hashes:?}"
    )]
    NotFound {
        instance_id: u64,
        sequence_hashes: Vec<SequenceHash>,
    },
    #[error(
        "G3PB payload for sequence hash {sequence_hash} had size {actual_size_bytes}, expected {expected_size_bytes}"
    )]
    InvalidPayloadSize {
        sequence_hash: SequenceHash,
        expected_size_bytes: usize,
        actual_size_bytes: usize,
    },
}

mod g3pb_client;
mod g3pb_service;

pub use g3pb_client::{
    G3pbDiscoveredPeers, G3pbPeerInstance, G3pbPeerResolver, G3pbRequestPlaneClient,
    G3pbStorageClient, discover_g3pb_peers, route_g3pb_put_blocks_by_owner,
    route_g3pb_sequence_hashes_by_owner, route_g3pb_transfer_blocks_by_owner, select_g3pb_owner,
};
pub use g3pb_service::{
    G3pbCacheStorage, G3pbStorageAgent, G3pbStorageConfig,
};
#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use tokio::time::Instant;

    fn peers() -> Vec<G3pbPeer> {
        vec![
            G3pbPeer {
                instance_id: 101,
                endpoint: "tcp://peer-10".to_string(),
                hostname: "g3pb-10".to_string(),
            },
            G3pbPeer {
                instance_id: 202,
                endpoint: "tcp://peer-20".to_string(),
                hostname: "g3pb-20".to_string(),
            },
            G3pbPeer {
                instance_id: 303,
                endpoint: "tcp://peer-30".to_string(),
                hostname: "g3pb-30".to_string(),
            },
        ]
    }

    #[test]
    fn owner_selection_is_deterministic() {
        let peers = peers();
        let sequence_hash = 0xdead_beef_u64;

        let owner_a = select_g3pb_owner(sequence_hash, &peers).unwrap();

        let mut reversed = peers.clone();
        reversed.reverse();
        let owner_b = select_g3pb_owner(sequence_hash, &reversed).unwrap();

        assert_eq!(owner_a, owner_b);
    }

    #[test]
    fn route_sequence_hashes_groups_by_owner_and_preserves_owner_order() {
        let peer_list = peers();
        let sequence_hashes = vec![1_u64, 2_u64, 3_u64, 4_u64];

        let routed = route_g3pb_sequence_hashes_by_owner(&sequence_hashes, &peer_list).unwrap();

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
        let err = G3pbDiscoveredPeers::from_health_responses(vec![
            (
                101,
                G3pbHealthResponse {
                    instance_id: 10,
                    listen: "tcp://peer-10-a".to_string(),
                    hostname: "g3pb-10-a".to_string(),
                },
            ),
            (
                202,
                G3pbHealthResponse {
                    instance_id: 10,
                    listen: "tcp://peer-10-b".to_string(),
                    hostname: "g3pb-10-b".to_string(),
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
        let discovered = G3pbDiscoveredPeers::from_health_responses(vec![
            (
                303,
                G3pbHealthResponse {
                    instance_id: 30,
                    listen: "tcp://peer-30".to_string(),
                    hostname: "g3pb-30".to_string(),
                },
            ),
            (
                101,
                G3pbHealthResponse {
                    instance_id: 10,
                    listen: "tcp://peer-10".to_string(),
                    hostname: "g3pb-10".to_string(),
                },
            ),
            (
                202,
                G3pbHealthResponse {
                    instance_id: 20,
                    listen: "tcp://peer-20".to_string(),
                    hostname: "g3pb-20".to_string(),
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
        let agent = G3pbStorageAgent::new(7);

        agent
            .offer_and_put_payload_blocks(vec![
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 11,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![1, 2, 3, 4],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 12,
                        size_bytes: 2,
                        checksum: Some([3; 32]),
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
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 11,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![1, 2, 3, 4],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 12,
                        size_bytes: 2,
                        checksum: Some([3; 32]),
                    },
                    payload: vec![5, 6],
                },
            ]
        );
    }

    #[tokio::test]
    async fn agent_fetch_reports_not_found() {
        let agent = G3pbStorageAgent::new(3);

        agent
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 44,
                size_bytes: 8,
                checksum: None,
            }])
            .await;

        let err = agent.fetch_blocks(&[44]).await.unwrap_err();
        assert!(matches!(
            err,
            G3pbError::NotFound {
                instance_id: 3,
                sequence_hashes
            } if sequence_hashes == vec![44]
        ));
    }

    #[tokio::test]
    async fn agent_offer_only_accepts_missing_blocks() {
        let agent = G3pbStorageAgent::new(7);
        agent
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 100,
                size_bytes: 16,
                checksum: None,
            }])
            .await;

        let accepted = agent
            .offered_blocks(vec![
                G3pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 16,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 200,
                    size_bytes: 32,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 200,
                    size_bytes: 32,
                    checksum: None,
                },
            ])
            .await;

        assert_eq!(
            accepted,
            vec![G3pbPutBlock {
                sequence_hash: 200,
                size_bytes: 32,
                checksum: None,
            }]
        );
    }

    #[tokio::test]
    async fn agent_offer_and_put_registers_only_missing_blocks() {
        let agent = G3pbStorageAgent::new(7);
        let accepted = agent
            .offer_and_put_blocks(vec![
                G3pbPutBlock {
                    sequence_hash: 5,
                    size_bytes: 16,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 6,
                    size_bytes: 32,
                    checksum: Some([9; 32]),
                },
                G3pbPutBlock {
                    sequence_hash: 6,
                    size_bytes: 32,
                    checksum: Some([9; 32]),
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
        let agent = G3pbStorageAgent::new(7);

        let err = agent
            .offered_payload_blocks(vec![G3pbTransferBlock {
                meta: G3pbPutBlock {
                    sequence_hash: 5,
                    size_bytes: 4,
                    checksum: None,
                },
                payload: vec![1, 2, 3],
            }])
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            G3pbError::InvalidPayloadSize {
                sequence_hash: 5,
                expected_size_bytes: 4,
                actual_size_bytes: 3
            }
        ));
    }

    #[tokio::test]
    async fn client_put_and_query_route_by_owner() {
        let peer_list = peers();
        let owner = select_g3pb_owner(1234, &peer_list).unwrap();
        let agent = Arc::new(G3pbStorageAgent::new(owner.instance_id));
        let client = G3pbStorageClient::new(
            peer_list,
            HashMap::from_iter([(owner.instance_id, agent.clone())]),
        );

        client
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 1234,
                size_bytes: 64,
                checksum: Some([7; 32]),
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
                Arc::new(G3pbStorageAgent::new(peer.instance_id)),
            );
        }

        let existing_hash = 200_u64;
        let existing_owner = select_g3pb_owner(existing_hash, &peer_list).unwrap();
        agents[&existing_owner.instance_id]
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: existing_hash,
                size_bytes: 8,
                checksum: None,
            }])
            .await;

        let client = G3pbStorageClient::new(peer_list, agents);
        let accepted = client
            .offer_blocks(vec![
                G3pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: existing_hash,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 300,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 100,
                    size_bytes: 8,
                    checksum: None,
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
                Arc::new(G3pbStorageAgent::new(peer.instance_id)),
            );
        }

        let client = G3pbStorageClient::new(peer_list.clone(), agents.clone());
        let accepted = client
            .offer_and_put_payload_blocks(vec![
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 1,
                        size_bytes: 2,
                        checksum: None,
                    },
                    payload: vec![1, 2],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 2,
                        size_bytes: 2,
                        checksum: None,
                    },
                    payload: vec![3, 4],
                },
                G3pbTransferBlock {
                    meta: G3pbPutBlock {
                        sequence_hash: 1,
                        size_bytes: 2,
                        checksum: None,
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
            let owner = select_g3pb_owner(*sequence_hash, &peer_list).unwrap();
            agents.entry(owner.instance_id).or_insert_with(|| {
                Arc::new(
                    G3pbStorageAgent::new(owner.instance_id)
                        .with_query_delay(Duration::from_millis(75)),
                )
            });
            agents[&owner.instance_id]
                .put_blocks(vec![G3pbPutBlock {
                    sequence_hash: *sequence_hash,
                    size_bytes: 8,
                    checksum: None,
                }])
                .await;
        }

        let client = G3pbStorageClient::new(peer_list, agents);
        let start = Instant::now();
        let hits = client.query_blocks(&sequence_hashes).await;
        let elapsed = start.elapsed();

        assert_eq!(hits.len(), sequence_hashes.len());
        assert!(elapsed < Duration::from_millis(200));
    }

    #[tokio::test]
    async fn client_treats_missing_fetches_as_cache_miss() {
        let peer_list = peers();
        let owner = select_g3pb_owner(1234, &peer_list).unwrap();
        let agent = Arc::new(G3pbStorageAgent::new(owner.instance_id));
        agent
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 1234,
                size_bytes: 4,
                checksum: None,
            }])
            .await;

        let client =
            G3pbStorageClient::new(peer_list, HashMap::from_iter([(owner.instance_id, agent)]));
        let fetched = client.fetch_blocks(&[1234]).await;
        assert!(fetched.is_empty());
    }

    #[tokio::test]
    async fn g3pb_cache_storage_supports_basic_operations() -> Result<()> {
        let mut config = G3pbStorageConfig::new(0);
        config.g2_capacity_bytes = 4 * 1024;

        let storage = Arc::new(G3pbCacheStorage::new(config).await?);
        let agent = G3pbStorageAgent::new_with_storage(77, storage.clone());

        agent
            .offer_and_put_payload_blocks(vec![G3pbTransferBlock {
                meta: G3pbPutBlock {
                    sequence_hash: 1001,
                    size_bytes: 8,
                    checksum: None,
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
                G3pbPutBlock {
                    sequence_hash: 1001,
                    size_bytes: 8,
                    checksum: None,
                },
                G3pbPutBlock {
                    sequence_hash: 2001,
                    size_bytes: 16,
                    checksum: None,
                },
            ])
            .await;
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0], 2001);

        agent.delete_blocks(&[1001]).await?;
        assert!(agent.query_blocks(&[1001]).await.is_empty());
        assert!(matches!(
            agent.fetch_blocks(&[1001]).await,
            Err(G3pbError::NotFound {
                instance_id: 77,
                sequence_hashes,
            }) if sequence_hashes == vec![1001]
        ));

        Ok(())
    }
}
