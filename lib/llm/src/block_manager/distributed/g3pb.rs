// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::WorkerID;
use crate::tokens::{SequenceHash, compute_hash_v2};

use async_trait::async_trait;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::RwLock;
#[cfg(test)]
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPeer {
    pub worker_id: WorkerID,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbPutBlock {
    pub sequence_hash: SequenceHash,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G3pbQueryHit {
    pub worker_id: WorkerID,
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
    pub worker_id: WorkerID,
    pub listen: String,
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

#[derive(Debug, thiserror::Error)]
pub enum G3pbError {
    #[error("no live G3PB peers are available")]
    NoPeers,
    #[error("owning G3PB peer {worker_id} is not available")]
    UnknownPeer { worker_id: WorkerID },
    #[error("requested G3PB blocks were not found on peer {worker_id}: {sequence_hashes:?}")]
    NotFound {
        worker_id: WorkerID,
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

#[async_trait]
pub trait G3pbPeerStorage: Send + Sync {
    async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>);
    async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash>;
    async fn put_payload_blocks(&self, blocks: Vec<G3pbTransferBlock>) -> Result<(), G3pbError>;
    async fn query_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G3pbQueryHit>;
    async fn fetch_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError>;
}

#[derive(Clone, Debug)]
struct InMemoryG3pbEntry {
    meta: G3pbPutBlock,
    payload: Option<Vec<u8>>,
}

#[derive(Default)]
pub struct InMemoryG3pbPeerStorage {
    blocks: RwLock<HashMap<SequenceHash, InMemoryG3pbEntry>>,
}

#[async_trait]
impl G3pbPeerStorage for InMemoryG3pbPeerStorage {
    async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>) {
        let mut guard = self.blocks.write().expect("g3pb peer storage poisoned");
        for block in blocks {
            let payload = guard
                .remove(&block.sequence_hash)
                .and_then(|entry| entry.payload);
            guard.insert(
                block.sequence_hash,
                InMemoryG3pbEntry {
                    meta: block,
                    payload,
                },
            );
        }
    }

    async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash> {
        let guard = self.blocks.read().expect("g3pb peer storage poisoned");
        let mut seen = HashSet::new();
        blocks
            .iter()
            .filter_map(|block| {
                let sequence_hash = block.sequence_hash;
                (!guard.contains_key(&sequence_hash) && seen.insert(sequence_hash))
                    .then_some(sequence_hash)
            })
            .collect()
    }

    async fn put_payload_blocks(&self, blocks: Vec<G3pbTransferBlock>) -> Result<(), G3pbError> {
        let mut guard = self.blocks.write().expect("g3pb peer storage poisoned");
        for block in blocks {
            block.validate_payload_size()?;
            guard.insert(
                block.meta.sequence_hash,
                InMemoryG3pbEntry {
                    meta: block.meta.clone(),
                    payload: Some(block.payload),
                },
            );
        }

        Ok(())
    }

    async fn query_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G3pbQueryHit> {
        let guard = self.blocks.read().expect("g3pb peer storage poisoned");
        sequence_hashes
            .iter()
            .filter_map(|sequence_hash| {
                guard.get(sequence_hash).map(|entry| G3pbQueryHit {
                    worker_id,
                    sequence_hash: *sequence_hash,
                    size_bytes: entry.meta.size_bytes,
                    checksum: entry.meta.checksum,
                })
            })
            .collect()
    }

    async fn fetch_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let guard = self.blocks.read().expect("g3pb peer storage poisoned");
        let mut missing = Vec::new();
        let mut blocks = Vec::with_capacity(sequence_hashes.len());

        for sequence_hash in sequence_hashes {
            match guard.get(sequence_hash) {
                Some(entry) => {
                    let Some(payload) = entry.payload.clone() else {
                        missing.push(*sequence_hash);
                        continue;
                    };
                    blocks.push(G3pbTransferBlock {
                        meta: entry.meta.clone(),
                        payload,
                    });
                }
                None => missing.push(*sequence_hash),
            }
        }

        if !missing.is_empty() {
            return Err(G3pbError::NotFound {
                worker_id,
                sequence_hashes: missing,
            });
        }

        Ok(blocks)
    }
}

pub fn select_g3pb_owner(sequence_hash: SequenceHash, peers: &[G3pbPeer]) -> Option<G3pbPeer> {
    peers
        .iter()
        .cloned()
        .max_by_key(|peer| rendezvous_score(sequence_hash, peer.worker_id))
}

fn rendezvous_score(sequence_hash: SequenceHash, worker_id: WorkerID) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&sequence_hash.to_le_bytes());
    bytes[8..].copy_from_slice(&worker_id.to_le_bytes());
    compute_hash_v2(&bytes, 0)
}

fn route_items_by_g3pb_owner<T, F>(
    items: impl IntoIterator<Item = T>,
    peers: &[G3pbPeer],
    sequence_hash: F,
) -> Result<HashMap<WorkerID, Vec<T>>, G3pbError>
where
    F: Fn(&T) -> SequenceHash,
{
    let mut routed = HashMap::<WorkerID, Vec<T>>::new();

    for item in items {
        let owner = select_g3pb_owner(sequence_hash(&item), peers).ok_or(G3pbError::NoPeers)?;
        routed.entry(owner.worker_id).or_default().push(item);
    }

    Ok(routed)
}

fn filter_first_sequence_hash_matches<T, F>(
    items: Vec<T>,
    mut sequence_hash: F,
    accepted_hashes: &HashSet<SequenceHash>,
) -> Vec<T>
where
    F: FnMut(&T) -> SequenceHash,
{
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|item| {
            let sequence_hash = sequence_hash(item);
            accepted_hashes.contains(&sequence_hash) && seen.insert(sequence_hash)
        })
        .collect()
}

pub fn route_g3pb_sequence_hashes_by_owner(
    sequence_hashes: &[SequenceHash],
    peers: &[G3pbPeer],
) -> Result<HashMap<WorkerID, Vec<SequenceHash>>, G3pbError> {
    route_items_by_g3pb_owner(sequence_hashes.iter().copied(), peers, |sequence_hash| {
        *sequence_hash
    })
}

pub fn route_g3pb_put_blocks_by_owner(
    blocks: Vec<G3pbPutBlock>,
    peers: &[G3pbPeer],
) -> Result<HashMap<WorkerID, Vec<G3pbPutBlock>>, G3pbError> {
    route_items_by_g3pb_owner(blocks, peers, |block| block.sequence_hash)
}

pub fn route_g3pb_transfer_blocks_by_owner(
    blocks: Vec<G3pbTransferBlock>,
    peers: &[G3pbPeer],
) -> Result<HashMap<WorkerID, Vec<G3pbTransferBlock>>, G3pbError> {
    route_items_by_g3pb_owner(blocks, peers, |block| block.meta.sequence_hash)
}

#[derive(Clone)]
pub struct G3pbStorageAgent {
    worker_id: WorkerID,
    storage: Arc<dyn G3pbPeerStorage>,
    #[cfg(test)]
    query_delay: Option<Duration>,
}

impl G3pbStorageAgent {
    pub fn new(worker_id: WorkerID) -> Self {
        Self::new_with_storage(worker_id, Arc::new(InMemoryG3pbPeerStorage::default()))
    }

    pub fn new_with_storage(worker_id: WorkerID, storage: Arc<dyn G3pbPeerStorage>) -> Self {
        Self {
            worker_id,
            storage,
            #[cfg(test)]
            query_delay: None,
        }
    }

    pub fn worker_id(&self) -> WorkerID {
        self.worker_id
    }

    #[cfg(test)]
    fn with_query_delay(mut self, query_delay: Duration) -> Self {
        self.query_delay = Some(query_delay);
        self
    }

    pub async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>) {
        self.storage.put_blocks(blocks).await;
    }

    pub async fn offer_blocks(&self, blocks: &[G3pbPutBlock]) -> Vec<SequenceHash> {
        self.storage.offer_blocks(blocks).await
    }

    pub async fn offered_blocks(&self, blocks: Vec<G3pbPutBlock>) -> Vec<G3pbPutBlock> {
        let accepted_hashes: HashSet<_> = self.offer_blocks(&blocks).await.into_iter().collect();
        filter_first_sequence_hash_matches(blocks, |block| block.sequence_hash, &accepted_hashes)
    }

    pub async fn offer_and_put_blocks(&self, blocks: Vec<G3pbPutBlock>) -> Vec<G3pbPutBlock> {
        let accepted_blocks = self.offered_blocks(blocks).await;
        if !accepted_blocks.is_empty() {
            self.put_blocks(accepted_blocks.clone()).await;
        }
        accepted_blocks
    }

    pub async fn offered_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        for block in &blocks {
            block.validate_payload_size()?;
        }

        let metadata: Vec<_> = blocks.iter().map(|block| block.meta.clone()).collect();
        let accepted_hashes: HashSet<_> = self.offer_blocks(&metadata).await.into_iter().collect();

        Ok(filter_first_sequence_hash_matches(
            blocks,
            |block| block.meta.sequence_hash,
            &accepted_hashes,
        ))
    }

    pub async fn offer_and_put_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let accepted_blocks = self.offered_payload_blocks(blocks).await?;
        if !accepted_blocks.is_empty() {
            self.storage
                .put_payload_blocks(accepted_blocks.clone())
                .await?;
        }
        Ok(accepted_blocks)
    }

    pub async fn query_blocks(&self, sequence_hashes: &[SequenceHash]) -> Vec<G3pbQueryHit> {
        #[cfg(test)]
        if let Some(query_delay) = self.query_delay {
            tokio::time::sleep(query_delay).await;
        }

        self.storage
            .query_blocks(self.worker_id, sequence_hashes)
            .await
    }

    pub async fn fetch_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        self.storage
            .fetch_blocks(self.worker_id, sequence_hashes)
            .await
    }
}

pub struct G3pbStorageClient {
    peers: Vec<G3pbPeer>,
    agents: HashMap<WorkerID, Arc<G3pbStorageAgent>>,
}

impl G3pbStorageClient {
    pub fn new(peers: Vec<G3pbPeer>, agents: HashMap<WorkerID, Arc<G3pbStorageAgent>>) -> Self {
        Self { peers, agents }
    }

    pub fn owner_for(&self, sequence_hash: SequenceHash) -> Result<G3pbPeer, G3pbError> {
        select_g3pb_owner(sequence_hash, &self.peers).ok_or(G3pbError::NoPeers)
    }

    pub async fn put_blocks(&self, blocks: Vec<G3pbPutBlock>) -> Result<(), G3pbError> {
        let routed = route_g3pb_put_blocks_by_owner(blocks, &self.peers)?;
        for (worker_id, blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            agent.put_blocks(blocks).await;
        }

        Ok(())
    }

    pub async fn offer_blocks(
        &self,
        blocks: Vec<G3pbPutBlock>,
    ) -> Result<Vec<G3pbPutBlock>, G3pbError> {
        let routed = route_g3pb_put_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offer_blocks(&owner_blocks).await;
            accepted_by_owner.insert(worker_id, accepted.into_iter().collect());
        }

        Ok(filter_accepted_put_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn offer_and_put_blocks(
        &self,
        blocks: Vec<G3pbPutBlock>,
    ) -> Result<Vec<G3pbPutBlock>, G3pbError> {
        let routed = route_g3pb_put_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offer_and_put_blocks(owner_blocks).await;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.sequence_hash)
                    .collect(),
            );
        }

        Ok(filter_accepted_put_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn offer_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let routed = route_g3pb_transfer_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offered_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.meta.sequence_hash)
                    .collect(),
            );
        }

        Ok(filter_accepted_transfer_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn offer_and_put_payload_blocks(
        &self,
        blocks: Vec<G3pbTransferBlock>,
    ) -> Result<Vec<G3pbTransferBlock>, G3pbError> {
        let routed = route_g3pb_transfer_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G3pbError::UnknownPeer { worker_id })?;
            let accepted = agent.offer_and_put_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.meta.sequence_hash)
                    .collect(),
            );
        }

        Ok(filter_accepted_transfer_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn query_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> HashMap<SequenceHash, G3pbQueryHit> {
        let grouped =
            route_g3pb_sequence_hashes_by_owner(sequence_hashes, &self.peers).unwrap_or_default();

        let queries = grouped
            .into_iter()
            .filter_map(|(worker_id, sequence_hashes)| {
                self.agents
                    .get(&worker_id)
                    .cloned()
                    .map(|agent| async move { agent.query_blocks(&sequence_hashes).await })
            });

        let mut hits = HashMap::new();
        for owner_hits in join_all(queries).await {
            for hit in owner_hits {
                hits.insert(hit.sequence_hash, hit);
            }
        }

        hits
    }

    pub async fn fetch_blocks(&self, sequence_hashes: &[SequenceHash]) -> Vec<G3pbTransferBlock> {
        let grouped =
            route_g3pb_sequence_hashes_by_owner(sequence_hashes, &self.peers).unwrap_or_default();

        let mut fetched = Vec::new();
        for (worker_id, owner_hashes) in grouped {
            let Some(agent) = self.agents.get(&worker_id) else {
                continue;
            };

            match agent.fetch_blocks(&owner_hashes).await {
                Ok(mut blocks) => fetched.append(&mut blocks),
                Err(G3pbError::NotFound { .. }) => {
                    // G3PB is a cache. Fetch failures degrade to cache misses.
                }
                Err(G3pbError::InvalidPayloadSize { .. }) => {}
                Err(G3pbError::NoPeers | G3pbError::UnknownPeer { .. }) => {}
            }
        }

        fetched.sort_by_key(|block| {
            sequence_hashes
                .iter()
                .position(|sequence_hash| *sequence_hash == block.meta.sequence_hash)
                .unwrap_or(usize::MAX)
        });
        fetched
    }
}

fn filter_accepted_put_blocks<F>(
    blocks: Vec<G3pbPutBlock>,
    owner_for: F,
    accepted_by_owner: &HashMap<WorkerID, HashSet<SequenceHash>>,
) -> Vec<G3pbPutBlock>
where
    F: Fn(SequenceHash) -> Option<G3pbPeer>,
{
    blocks
        .into_iter()
        .fold(
            (Vec::new(), HashSet::new()),
            |(mut accepted_blocks, mut seen), block| {
                let sequence_hash = block.sequence_hash;
                let is_accepted = owner_for(sequence_hash)
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&sequence_hash));

                if is_accepted && seen.insert(sequence_hash) {
                    accepted_blocks.push(block);
                }

                (accepted_blocks, seen)
            },
        )
        .0
}

fn filter_accepted_transfer_blocks<F>(
    blocks: Vec<G3pbTransferBlock>,
    owner_for: F,
    accepted_by_owner: &HashMap<WorkerID, HashSet<SequenceHash>>,
) -> Vec<G3pbTransferBlock>
where
    F: Fn(SequenceHash) -> Option<G3pbPeer>,
{
    blocks
        .into_iter()
        .fold(
            (Vec::new(), HashSet::new()),
            |(mut accepted_blocks, mut seen), block| {
                let sequence_hash = block.meta.sequence_hash;
                let is_accepted = owner_for(sequence_hash)
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&sequence_hash));

                if is_accepted && seen.insert(sequence_hash) {
                    accepted_blocks.push(block);
                }

                (accepted_blocks, seen)
            },
        )
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    use tokio::time::Instant;

    fn peers() -> Vec<G3pbPeer> {
        vec![
            G3pbPeer {
                worker_id: 10,
                endpoint: "tcp://peer-10".to_string(),
            },
            G3pbPeer {
                worker_id: 20,
                endpoint: "tcp://peer-20".to_string(),
            },
            G3pbPeer {
                worker_id: 30,
                endpoint: "tcp://peer-30".to_string(),
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
        assert_eq!(hits[0].worker_id, 7);
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
                worker_id: 3,
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
        let agent = Arc::new(G3pbStorageAgent::new(owner.worker_id));
        let client = G3pbStorageClient::new(
            peer_list,
            HashMap::from_iter([(owner.worker_id, agent.clone())]),
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
        assert_eq!(hits[&1234].worker_id, owner.worker_id);
        assert_eq!(hits[&1234].size_bytes, 64);
    }

    #[tokio::test]
    async fn client_offer_routes_by_owner_and_preserves_input_order() {
        let peer_list = peers();
        let mut agents = HashMap::new();

        for peer in &peer_list {
            agents.insert(
                peer.worker_id,
                Arc::new(G3pbStorageAgent::new(peer.worker_id)),
            );
        }

        let existing_hash = 200_u64;
        let existing_owner = select_g3pb_owner(existing_hash, &peer_list).unwrap();
        agents[&existing_owner.worker_id]
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
                peer.worker_id,
                Arc::new(G3pbStorageAgent::new(peer.worker_id)),
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
            agents.entry(owner.worker_id).or_insert_with(|| {
                Arc::new(
                    G3pbStorageAgent::new(owner.worker_id)
                        .with_query_delay(Duration::from_millis(75)),
                )
            });
            agents[&owner.worker_id]
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
        let agent = Arc::new(G3pbStorageAgent::new(owner.worker_id));
        agent
            .put_blocks(vec![G3pbPutBlock {
                sequence_hash: 1234,
                size_bytes: 4,
                checksum: None,
            }])
            .await;

        let client =
            G3pbStorageClient::new(peer_list, HashMap::from_iter([(owner.worker_id, agent)]));
        let fetched = client.fetch_blocks(&[1234]).await;
        assert!(fetched.is_empty());
    }
}
