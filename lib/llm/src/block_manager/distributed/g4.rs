// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{BlockTransferHandler, BlockTransferPool, BlockTransferRequest};

use crate::block_manager::{
    WorkerID,
    block::{
        BlockDataProvider, BlockMetadata, ImmutableBlock, data::BlockDataExt,
        locality::LocalityProvider,
    },
    offload::DiskBlockRegistrationObserver,
    storage::DiskStorage,
};
use crate::tokens::{SequenceHash, compute_hash_v2};

use anyhow::Result;
use async_trait::async_trait;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4StorageWorker {
    pub worker_id: WorkerID,
    pub endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4PutBlock {
    pub sequence_hash: SequenceHash,
    pub disk_block_idx: usize,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4QueryHit {
    pub worker_id: WorkerID,
    pub sequence_hash: SequenceHash,
    pub disk_block_idx: usize,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4FetchedBlock {
    pub worker_id: WorkerID,
    pub sequence_hash: SequenceHash,
    pub target_block_idx: usize,
    pub size_bytes: usize,
    pub checksum: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4TransferBlock {
    pub meta: G4PutBlock,
    pub payload: Vec<u8>,
}

impl G4TransferBlock {
    fn validate_payload_size(&self) -> Result<(), G4Error> {
        let actual_size_bytes = self.payload.len();
        if actual_size_bytes != self.meta.size_bytes {
            return Err(G4Error::InvalidPayloadSize {
                sequence_hash: self.meta.sequence_hash,
                expected_size_bytes: self.meta.size_bytes,
                actual_size_bytes,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4OfferRequest {
    pub blocks: Vec<G4PutBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4OfferResponse {
    pub accepted: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4PutPayloadRequest {
    pub blocks: Vec<G4TransferBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4FetchRequest {
    pub sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G4FetchResponse {
    pub blocks: Vec<G4TransferBlock>,
}

#[derive(Debug, thiserror::Error)]
pub enum G4Error {
    #[error("no live G4 storage workers are available")]
    NoStorageWorkers,
    #[error("owning G4 worker {worker_id} is not available")]
    UnknownWorker { worker_id: WorkerID },
    #[error("requested G4 blocks were not found on worker {worker_id}: {sequence_hashes:?}")]
    NotFound {
        worker_id: WorkerID,
        sequence_hashes: Vec<SequenceHash>,
    },
    #[error("G4 fetch from worker {worker_id} timed out after {timeout:?}")]
    Timeout {
        worker_id: WorkerID,
        timeout: Duration,
    },
    #[error("G4 transfer failed on worker {worker_id}: {source}")]
    Transfer {
        worker_id: WorkerID,
        #[source]
        source: anyhow::Error,
    },
    #[error(
        "G4 payload for sequence hash {sequence_hash} had size {actual_size_bytes}, expected {expected_size_bytes}"
    )]
    InvalidPayloadSize {
        sequence_hash: SequenceHash,
        expected_size_bytes: usize,
        actual_size_bytes: usize,
    },
}

#[async_trait]
pub trait G4TransferExecutor: Send + Sync {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()>;
}

#[async_trait]
impl G4TransferExecutor for BlockTransferHandler {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.execute_transfer_direct(request).await
    }
}

/// Pick the single owner with the highest rendezvous score.
pub fn select_g4_owner(
    sequence_hash: SequenceHash,
    workers: &[G4StorageWorker],
) -> Option<G4StorageWorker> {
    workers
        .iter()
        .cloned()
        .max_by_key(|worker| rendezvous_score(sequence_hash, worker.worker_id))
}

fn rendezvous_score(sequence_hash: SequenceHash, worker_id: WorkerID) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&sequence_hash.to_le_bytes());
    bytes[8..].copy_from_slice(&worker_id.to_le_bytes());
    compute_hash_v2(&bytes, 0)
}

#[derive(Default)]
pub struct G4BlockIndex {
    blocks: RwLock<HashMap<SequenceHash, G4PutBlock>>,
}

impl G4BlockIndex {
    pub async fn put_blocks(&self, blocks: Vec<G4PutBlock>) {
        let mut guard = self.blocks.write().expect("g4 block index poisoned");
        for block in blocks {
            guard.insert(block.sequence_hash, block);
        }
    }

    pub fn block(&self, sequence_hash: SequenceHash) -> Option<G4PutBlock> {
        self.blocks
            .read()
            .expect("g4 block index poisoned")
            .get(&sequence_hash)
            .cloned()
    }

    pub fn offer_blocks(&self, blocks: &[G4PutBlock]) -> Vec<SequenceHash> {
        let guard = self.blocks.read().expect("g4 block index poisoned");
        blocks
            .iter()
            .filter_map(|block| {
                (!guard.contains_key(&block.sequence_hash)).then_some(block.sequence_hash)
            })
            .collect()
    }

    async fn query_blocks(
        &self,
        worker_id: WorkerID,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<G4QueryHit> {
        let guard = self.blocks.read().expect("g4 block index poisoned");
        sequence_hashes
            .iter()
            .filter_map(|sequence_hash| {
                guard.get(sequence_hash).map(|block| G4QueryHit {
                    worker_id,
                    sequence_hash: *sequence_hash,
                    disk_block_idx: block.disk_block_idx,
                    size_bytes: block.size_bytes,
                    checksum: block.checksum,
                })
            })
            .collect()
    }

    async fn fetch_entries(
        &self,
        worker_id: WorkerID,
        entries: &[(SequenceHash, usize)],
    ) -> Result<(Vec<(usize, usize)>, Vec<G4FetchedBlock>), G4Error> {
        let guard = self.blocks.read().expect("g4 block index poisoned");
        let mut missing = Vec::new();
        let mut request_blocks = Vec::with_capacity(entries.len());
        let mut fetched_blocks = Vec::with_capacity(entries.len());

        for (sequence_hash, target_block_idx) in entries {
            match guard.get(sequence_hash) {
                Some(block) => {
                    request_blocks.push((block.disk_block_idx, *target_block_idx));
                    fetched_blocks.push(G4FetchedBlock {
                        worker_id,
                        sequence_hash: *sequence_hash,
                        target_block_idx: *target_block_idx,
                        size_bytes: block.size_bytes,
                        checksum: block.checksum,
                    });
                }
                None => missing.push(*sequence_hash),
            }
        }

        if !missing.is_empty() {
            return Err(G4Error::NotFound {
                worker_id,
                sequence_hashes: missing,
            });
        }

        Ok((request_blocks, fetched_blocks))
    }

    pub fn register_disk_blocks<Locality, Metadata>(
        &self,
        blocks: &[ImmutableBlock<DiskStorage, Locality, Metadata>],
    ) -> Result<()>
    where
        Locality: LocalityProvider,
        Metadata: BlockMetadata,
    {
        let mut registered = Vec::with_capacity(blocks.len());
        for block in blocks {
            let size_bytes = block.block_data().block_view()?.size();
            registered.push(G4PutBlock {
                sequence_hash: block.sequence_hash(),
                disk_block_idx: block.block_id(),
                size_bytes,
                checksum: None,
            });
        }

        let mut guard = self.blocks.write().expect("g4 block index poisoned");
        for block in registered {
            guard.insert(block.sequence_hash, block);
        }
        Ok(())
    }
}

impl<Locality, Metadata> DiskBlockRegistrationObserver<Locality, Metadata> for G4BlockIndex
where
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
{
    fn observe_registered_blocks(
        &self,
        blocks: &[ImmutableBlock<DiskStorage, Locality, Metadata>],
    ) -> Result<()> {
        self.register_disk_blocks(blocks)
    }
}

#[derive(Clone)]
pub struct G4StorageAgent {
    worker_id: WorkerID,
    transfer: Arc<dyn G4TransferExecutor>,
    block_index: Arc<G4BlockIndex>,
    #[cfg(test)]
    query_delay: Option<Duration>,
}

impl G4StorageAgent {
    pub fn new(worker_id: WorkerID, transfer: Arc<dyn G4TransferExecutor>) -> Self {
        Self::new_with_index(worker_id, transfer, Arc::new(G4BlockIndex::default()))
    }

    pub fn new_with_index(
        worker_id: WorkerID,
        transfer: Arc<dyn G4TransferExecutor>,
        block_index: Arc<G4BlockIndex>,
    ) -> Self {
        Self {
            worker_id,
            transfer,
            block_index,
            #[cfg(test)]
            query_delay: None,
        }
    }

    pub fn worker_id(&self) -> WorkerID {
        self.worker_id
    }

    pub fn block_index(&self) -> Arc<G4BlockIndex> {
        self.block_index.clone()
    }

    #[cfg(test)]
    fn with_query_delay(mut self, query_delay: Duration) -> Self {
        self.query_delay = Some(query_delay);
        self
    }

    pub async fn put_blocks(&self, blocks: Vec<G4PutBlock>) {
        self.block_index.put_blocks(blocks).await;
    }

    pub async fn offer_blocks(&self, blocks: &[G4PutBlock]) -> Vec<SequenceHash> {
        self.block_index.offer_blocks(blocks)
    }

    pub async fn offered_blocks(&self, blocks: Vec<G4PutBlock>) -> Vec<G4PutBlock> {
        let accepted_hashes: HashSet<_> = self.offer_blocks(&blocks).await.into_iter().collect();
        blocks
            .into_iter()
            .filter(|block| accepted_hashes.contains(&block.sequence_hash))
            .collect()
    }

    pub async fn offer_and_put_blocks(&self, blocks: Vec<G4PutBlock>) -> Vec<G4PutBlock> {
        let accepted_blocks = self.offered_blocks(blocks).await;

        if !accepted_blocks.is_empty() {
            self.put_blocks(accepted_blocks.clone()).await;
        }

        accepted_blocks
    }

    pub async fn offered_payload_blocks(
        &self,
        blocks: Vec<G4TransferBlock>,
    ) -> Result<Vec<G4TransferBlock>, G4Error> {
        for block in &blocks {
            block.validate_payload_size()?;
        }

        let metadata: Vec<_> = blocks.iter().map(|block| block.meta.clone()).collect();
        let accepted_hashes: HashSet<_> = self.offer_blocks(&metadata).await.into_iter().collect();

        Ok(blocks
            .into_iter()
            .filter(|block| accepted_hashes.contains(&block.meta.sequence_hash))
            .collect())
    }

    pub async fn offer_and_put_payload_blocks(
        &self,
        blocks: Vec<G4TransferBlock>,
    ) -> Result<Vec<G4TransferBlock>, G4Error> {
        let accepted_blocks = self.offered_payload_blocks(blocks).await?;

        if !accepted_blocks.is_empty() {
            self.put_blocks(
                accepted_blocks
                    .iter()
                    .map(|block| block.meta.clone())
                    .collect(),
            )
            .await;
        }

        Ok(accepted_blocks)
    }

    pub async fn query_blocks(&self, sequence_hashes: &[SequenceHash]) -> Vec<G4QueryHit> {
        #[cfg(test)]
        if let Some(query_delay) = self.query_delay {
            tokio::time::sleep(query_delay).await;
        }

        self.block_index
            .query_blocks(self.worker_id, sequence_hashes)
            .await
    }

    async fn fetch_entries(
        &self,
        target_pool: BlockTransferPool,
        entries: &[(SequenceHash, usize)],
        timeout: Duration,
    ) -> Result<Vec<G4FetchedBlock>, G4Error> {
        let (request_blocks, fetched_blocks) = self
            .block_index
            .fetch_entries(self.worker_id, entries)
            .await?;
        let request =
            BlockTransferRequest::new(BlockTransferPool::Disk, target_pool, request_blocks);

        match tokio::time::timeout(timeout, self.transfer.execute_transfer(request)).await {
            Ok(Ok(())) => Ok(fetched_blocks),
            Ok(Err(source)) => Err(G4Error::Transfer {
                worker_id: self.worker_id,
                source,
            }),
            Err(_) => Err(G4Error::Timeout {
                worker_id: self.worker_id,
                timeout,
            }),
        }
    }

    pub async fn fetch_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
        target_pool: BlockTransferPool,
        target_block_start_idx: usize,
        timeout: Duration,
    ) -> Result<Vec<G4FetchedBlock>, G4Error> {
        let entries: Vec<_> = sequence_hashes
            .iter()
            .enumerate()
            .map(|(offset, sequence_hash)| (*sequence_hash, target_block_start_idx + offset))
            .collect();
        self.fetch_entries(target_pool, &entries, timeout).await
    }
}

pub struct G4StorageClient {
    workers: Vec<G4StorageWorker>,
    agents: HashMap<WorkerID, Arc<G4StorageAgent>>,
    request_timeout: Duration,
}

impl G4StorageClient {
    pub fn new(
        workers: Vec<G4StorageWorker>,
        agents: HashMap<WorkerID, Arc<G4StorageAgent>>,
        request_timeout: Duration,
    ) -> Self {
        Self {
            workers,
            agents,
            request_timeout,
        }
    }

    pub fn owner_for(&self, sequence_hash: SequenceHash) -> Result<G4StorageWorker, G4Error> {
        select_g4_owner(sequence_hash, &self.workers).ok_or(G4Error::NoStorageWorkers)
    }

    pub async fn put_blocks(&self, blocks: Vec<G4PutBlock>) -> Result<(), G4Error> {
        let mut routed: HashMap<WorkerID, Vec<G4PutBlock>> = HashMap::new();

        for block in blocks {
            let owner = self.owner_for(block.sequence_hash)?;
            routed.entry(owner.worker_id).or_default().push(block);
        }

        for (worker_id, blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G4Error::UnknownWorker { worker_id })?;
            agent.put_blocks(blocks).await;
        }

        Ok(())
    }

    pub async fn offer_blocks(&self, blocks: Vec<G4PutBlock>) -> Result<Vec<G4PutBlock>, G4Error> {
        let mut routed: HashMap<WorkerID, Vec<G4PutBlock>> = HashMap::new();

        for block in &blocks {
            let owner = self.owner_for(block.sequence_hash)?;
            routed
                .entry(owner.worker_id)
                .or_default()
                .push(block.clone());
        }

        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G4Error::UnknownWorker { worker_id })?;
            let accepted = agent.offer_blocks(&owner_blocks).await;
            accepted_by_owner.insert(worker_id, accepted.into_iter().collect());
        }

        Ok(blocks
            .into_iter()
            .filter(|block| {
                self.owner_for(block.sequence_hash)
                    .ok()
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&block.sequence_hash))
            })
            .collect())
    }

    pub async fn offer_and_put_blocks(
        &self,
        blocks: Vec<G4PutBlock>,
    ) -> Result<Vec<G4PutBlock>, G4Error> {
        let mut routed: HashMap<WorkerID, Vec<G4PutBlock>> = HashMap::new();

        for block in &blocks {
            let owner = self.owner_for(block.sequence_hash)?;
            routed
                .entry(owner.worker_id)
                .or_default()
                .push(block.clone());
        }

        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G4Error::UnknownWorker { worker_id })?;
            let accepted = agent.offer_and_put_blocks(owner_blocks).await;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.sequence_hash)
                    .collect(),
            );
        }

        Ok(blocks
            .into_iter()
            .filter(|block| {
                self.owner_for(block.sequence_hash)
                    .ok()
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&block.sequence_hash))
            })
            .collect())
    }

    pub async fn offer_payload_blocks(
        &self,
        blocks: Vec<G4TransferBlock>,
    ) -> Result<Vec<G4TransferBlock>, G4Error> {
        let mut routed: HashMap<WorkerID, Vec<G4TransferBlock>> = HashMap::new();

        for block in &blocks {
            let owner = self.owner_for(block.meta.sequence_hash)?;
            routed
                .entry(owner.worker_id)
                .or_default()
                .push(block.clone());
        }

        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G4Error::UnknownWorker { worker_id })?;
            let accepted = agent.offered_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.meta.sequence_hash)
                    .collect(),
            );
        }

        Ok(blocks
            .into_iter()
            .filter(|block| {
                self.owner_for(block.meta.sequence_hash)
                    .ok()
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&block.meta.sequence_hash))
            })
            .collect())
    }

    pub async fn offer_and_put_payload_blocks(
        &self,
        blocks: Vec<G4TransferBlock>,
    ) -> Result<Vec<G4TransferBlock>, G4Error> {
        let mut routed: HashMap<WorkerID, Vec<G4TransferBlock>> = HashMap::new();

        for block in &blocks {
            let owner = self.owner_for(block.meta.sequence_hash)?;
            routed
                .entry(owner.worker_id)
                .or_default()
                .push(block.clone());
        }

        let mut accepted_by_owner = HashMap::<WorkerID, HashSet<SequenceHash>>::new();
        for (worker_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&worker_id)
                .ok_or(G4Error::UnknownWorker { worker_id })?;
            let accepted = agent.offer_and_put_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                worker_id,
                accepted
                    .into_iter()
                    .map(|block| block.meta.sequence_hash)
                    .collect(),
            );
        }

        Ok(blocks
            .into_iter()
            .filter(|block| {
                self.owner_for(block.meta.sequence_hash)
                    .ok()
                    .and_then(|owner| accepted_by_owner.get(&owner.worker_id))
                    .is_some_and(|accepted| accepted.contains(&block.meta.sequence_hash))
            })
            .collect())
    }

    pub async fn query_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> HashMap<SequenceHash, G4QueryHit> {
        let mut grouped = HashMap::<WorkerID, Vec<SequenceHash>>::new();
        for sequence_hash in sequence_hashes {
            if let Ok(owner) = self.owner_for(*sequence_hash) {
                grouped
                    .entry(owner.worker_id)
                    .or_default()
                    .push(*sequence_hash);
            }
        }

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

    pub async fn fetch_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
        target_pool: BlockTransferPool,
        target_block_start_idx: usize,
    ) -> Vec<G4FetchedBlock> {
        let mut grouped = HashMap::<WorkerID, Vec<(usize, SequenceHash)>>::new();
        for (offset, sequence_hash) in sequence_hashes.iter().enumerate() {
            if let Ok(owner) = self.owner_for(*sequence_hash) {
                grouped
                    .entry(owner.worker_id)
                    .or_default()
                    .push((offset, *sequence_hash));
            }
        }

        let mut fetched = Vec::new();
        for (worker_id, entries) in grouped {
            let Some(agent) = self.agents.get(&worker_id) else {
                continue;
            };

            let routed_entries: Vec<_> = entries
                .iter()
                .map(|(offset, hash)| (*hash, target_block_start_idx + *offset))
                .collect();

            match agent
                .fetch_entries(target_pool, &routed_entries, self.request_timeout)
                .await
            {
                Ok(mut blocks) => fetched.append(&mut blocks),
                Err(
                    G4Error::NotFound { .. } | G4Error::Timeout { .. } | G4Error::Transfer { .. },
                ) => {
                    // G4 is a cache. Fetch failures degrade to cache misses.
                }
                Err(G4Error::InvalidPayloadSize { .. }) => {}
                Err(G4Error::NoStorageWorkers | G4Error::UnknownWorker { .. }) => {}
            }
        }

        fetched.sort_by_key(|block| block.target_block_idx);
        fetched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::anyhow;
    use tokio::sync::Mutex;
    use tokio::time::Instant;

    #[derive(Default)]
    struct MockTransferExecutor {
        requests: Mutex<Vec<BlockTransferRequest>>,
        fail: bool,
        sleep: Option<Duration>,
    }

    #[async_trait]
    impl G4TransferExecutor for MockTransferExecutor {
        async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
            if let Some(sleep) = self.sleep {
                tokio::time::sleep(sleep).await;
            }

            self.requests.lock().await.push(request);
            if self.fail {
                Err(anyhow!("mock transfer failure"))
            } else {
                Ok(())
            }
        }
    }

    fn workers() -> Vec<G4StorageWorker> {
        vec![
            G4StorageWorker {
                worker_id: 10,
                endpoint: "tcp://worker-10".to_string(),
            },
            G4StorageWorker {
                worker_id: 20,
                endpoint: "tcp://worker-20".to_string(),
            },
            G4StorageWorker {
                worker_id: 30,
                endpoint: "tcp://worker-30".to_string(),
            },
        ]
    }

    #[test]
    fn owner_selection_is_deterministic() {
        let workers = workers();
        let sequence_hash = 0xdead_beef_u64;

        let owner_a = select_g4_owner(sequence_hash, &workers).unwrap();

        let mut reversed = workers.clone();
        reversed.reverse();
        let owner_b = select_g4_owner(sequence_hash, &reversed).unwrap();

        assert_eq!(owner_a, owner_b);
    }

    #[tokio::test]
    async fn agent_query_and_fetch_use_disk_transfer_request() {
        let transfer = Arc::new(MockTransferExecutor::default());
        let agent = G4StorageAgent::new(7, transfer.clone());

        agent
            .put_blocks(vec![
                G4PutBlock {
                    sequence_hash: 11,
                    disk_block_idx: 4,
                    size_bytes: 1024,
                    checksum: None,
                },
                G4PutBlock {
                    sequence_hash: 12,
                    disk_block_idx: 9,
                    size_bytes: 2048,
                    checksum: Some([3; 32]),
                },
            ])
            .await;

        let hits = agent.query_blocks(&[11, 12, 13]).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].worker_id, 7);
        assert_eq!(hits[0].disk_block_idx, 4);
        assert_eq!(hits[1].disk_block_idx, 9);

        let fetched = agent
            .fetch_blocks(
                &[11, 12],
                BlockTransferPool::Host,
                100,
                Duration::from_millis(50),
            )
            .await
            .unwrap();

        assert_eq!(
            fetched,
            vec![
                G4FetchedBlock {
                    worker_id: 7,
                    sequence_hash: 11,
                    target_block_idx: 100,
                    size_bytes: 1024,
                    checksum: None,
                },
                G4FetchedBlock {
                    worker_id: 7,
                    sequence_hash: 12,
                    target_block_idx: 101,
                    size_bytes: 2048,
                    checksum: Some([3; 32]),
                },
            ]
        );

        let requests = transfer.requests.lock().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].from_pool, BlockTransferPool::Disk);
        assert_eq!(requests[0].to_pool, BlockTransferPool::Host);
        assert_eq!(requests[0].blocks, vec![(4, 100), (9, 101)]);
    }

    #[tokio::test]
    async fn agent_fetch_reports_not_found() {
        let transfer = Arc::new(MockTransferExecutor::default());
        let agent = G4StorageAgent::new(3, transfer);

        agent
            .put_blocks(vec![G4PutBlock {
                sequence_hash: 11,
                disk_block_idx: 4,
                size_bytes: 1024,
                checksum: None,
            }])
            .await;

        let err = agent
            .fetch_blocks(
                &[11, 12],
                BlockTransferPool::Host,
                0,
                Duration::from_millis(50),
            )
            .await
            .unwrap_err();

        match err {
            G4Error::NotFound {
                worker_id,
                sequence_hashes,
            } => {
                assert_eq!(worker_id, 3);
                assert_eq!(sequence_hashes, vec![12]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn client_treats_transfer_failures_as_cache_miss() {
        let failing_transfer = Arc::new(MockTransferExecutor {
            fail: true,
            ..Default::default()
        });
        let agent = Arc::new(G4StorageAgent::new(20, failing_transfer));
        agent
            .put_blocks(vec![G4PutBlock {
                sequence_hash: 99,
                disk_block_idx: 7,
                size_bytes: 4096,
                checksum: None,
            }])
            .await;

        let client = G4StorageClient::new(
            workers(),
            HashMap::from([(20, agent)]),
            Duration::from_millis(50),
        );

        let fetched = client.fetch_blocks(&[99], BlockTransferPool::Host, 0).await;
        assert!(fetched.is_empty());
    }

    #[tokio::test]
    async fn client_put_and_query_route_by_owner() {
        let worker_list = workers();
        let owner = select_g4_owner(1234, &worker_list).unwrap();
        let agent = Arc::new(G4StorageAgent::new(
            owner.worker_id,
            Arc::new(MockTransferExecutor::default()),
        ));
        let client = G4StorageClient::new(
            worker_list,
            HashMap::from([(owner.worker_id, agent.clone())]),
            Duration::from_millis(50),
        );

        client
            .put_blocks(vec![G4PutBlock {
                sequence_hash: 1234,
                disk_block_idx: 55,
                size_bytes: 512,
                checksum: None,
            }])
            .await
            .unwrap();

        let hits = client.query_blocks(&[1234, 9999]).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[&1234].worker_id, owner.worker_id);
        assert_eq!(hits[&1234].disk_block_idx, 55);
    }

    #[tokio::test]
    async fn agent_offer_only_accepts_missing_blocks() {
        let transfer = Arc::new(MockTransferExecutor::default());
        let agent = G4StorageAgent::new(7, transfer);

        agent
            .put_blocks(vec![G4PutBlock {
                sequence_hash: 11,
                disk_block_idx: 4,
                size_bytes: 1024,
                checksum: None,
            }])
            .await;

        let offered = agent
            .offer_blocks(&[
                G4PutBlock {
                    sequence_hash: 11,
                    disk_block_idx: 40,
                    size_bytes: 1024,
                    checksum: None,
                },
                G4PutBlock {
                    sequence_hash: 12,
                    disk_block_idx: 5,
                    size_bytes: 2048,
                    checksum: None,
                },
            ])
            .await;

        assert_eq!(offered, vec![12]);
    }

    #[tokio::test]
    async fn client_offer_routes_by_owner_and_preserves_input_order() {
        let worker_list = workers();
        let sequence_hashes = vec![1_u64, 2_u64, 3_u64];
        let mut agents = HashMap::new();

        for sequence_hash in &sequence_hashes {
            let owner = select_g4_owner(*sequence_hash, &worker_list).unwrap();
            agents.entry(owner.worker_id).or_insert_with(|| {
                Arc::new(G4StorageAgent::new(
                    owner.worker_id,
                    Arc::new(MockTransferExecutor::default()),
                ))
            });
        }

        let existing_hash = sequence_hashes[1];
        let existing_owner = select_g4_owner(existing_hash, &worker_list).unwrap();
        agents
            .get(&existing_owner.worker_id)
            .unwrap()
            .put_blocks(vec![G4PutBlock {
                sequence_hash: existing_hash,
                disk_block_idx: 99,
                size_bytes: 4096,
                checksum: None,
            }])
            .await;

        let client = G4StorageClient::new(worker_list, agents, Duration::from_millis(50));
        let offered = client
            .offer_blocks(
                sequence_hashes
                    .iter()
                    .enumerate()
                    .map(|(idx, sequence_hash)| G4PutBlock {
                        sequence_hash: *sequence_hash,
                        disk_block_idx: idx,
                        size_bytes: 1024 * (idx + 1),
                        checksum: None,
                    })
                    .collect(),
            )
            .await
            .unwrap();

        assert_eq!(
            offered
                .iter()
                .map(|block| block.sequence_hash)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[tokio::test]
    async fn agent_offer_and_put_registers_only_missing_blocks() {
        let transfer = Arc::new(MockTransferExecutor::default());
        let agent = G4StorageAgent::new(7, transfer);

        agent
            .put_blocks(vec![G4PutBlock {
                sequence_hash: 11,
                disk_block_idx: 4,
                size_bytes: 1024,
                checksum: None,
            }])
            .await;

        let accepted = agent
            .offer_and_put_blocks(vec![
                G4PutBlock {
                    sequence_hash: 11,
                    disk_block_idx: 40,
                    size_bytes: 1024,
                    checksum: None,
                },
                G4PutBlock {
                    sequence_hash: 12,
                    disk_block_idx: 5,
                    size_bytes: 2048,
                    checksum: Some([7; 32]),
                },
            ])
            .await;

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.sequence_hash)
                .collect::<Vec<_>>(),
            vec![12]
        );

        let hits = agent.query_blocks(&[11, 12]).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits.iter().find(|hit| hit.sequence_hash == 12).map(|hit| (
                hit.disk_block_idx,
                hit.size_bytes,
                hit.checksum
            )),
            Some((5, 2048, Some([7; 32])))
        );
    }

    #[tokio::test]
    async fn client_offer_and_put_routes_by_owner_and_preserves_input_order() {
        let worker_list = workers();
        let sequence_hashes = vec![1_u64, 2_u64, 3_u64];
        let mut agents = HashMap::new();

        for sequence_hash in &sequence_hashes {
            let owner = select_g4_owner(*sequence_hash, &worker_list).unwrap();
            agents.entry(owner.worker_id).or_insert_with(|| {
                Arc::new(G4StorageAgent::new(
                    owner.worker_id,
                    Arc::new(MockTransferExecutor::default()),
                ))
            });
        }

        let existing_hash = sequence_hashes[1];
        let existing_owner = select_g4_owner(existing_hash, &worker_list).unwrap();
        agents
            .get(&existing_owner.worker_id)
            .unwrap()
            .put_blocks(vec![G4PutBlock {
                sequence_hash: existing_hash,
                disk_block_idx: 99,
                size_bytes: 4096,
                checksum: None,
            }])
            .await;

        let client = G4StorageClient::new(worker_list, agents.clone(), Duration::from_millis(50));
        let accepted = client
            .offer_and_put_blocks(
                sequence_hashes
                    .iter()
                    .enumerate()
                    .map(|(idx, sequence_hash)| G4PutBlock {
                        sequence_hash: *sequence_hash,
                        disk_block_idx: idx,
                        size_bytes: 1024 * (idx + 1),
                        checksum: Some([idx as u8; 32]),
                    })
                    .collect(),
            )
            .await
            .unwrap();

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.sequence_hash)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );

        let query_hashes = sequence_hashes.clone();
        for sequence_hash in query_hashes {
            let owner = select_g4_owner(sequence_hash, &workers()).unwrap();
            let hits = agents
                .get(&owner.worker_id)
                .unwrap()
                .query_blocks(&[sequence_hash])
                .await;
            assert_eq!(hits.len(), 1);
        }
    }

    #[tokio::test]
    async fn agent_offer_payload_blocks_rejects_size_mismatch() {
        let agent = G4StorageAgent::new(7, Arc::new(MockTransferExecutor::default()));

        let err = agent
            .offered_payload_blocks(vec![G4TransferBlock {
                meta: G4PutBlock {
                    sequence_hash: 11,
                    disk_block_idx: 4,
                    size_bytes: 8,
                    checksum: None,
                },
                payload: vec![1, 2, 3],
            }])
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            G4Error::InvalidPayloadSize {
                sequence_hash: 11,
                expected_size_bytes: 8,
                actual_size_bytes: 3,
            }
        ));
    }

    #[tokio::test]
    async fn client_offer_payload_routes_by_owner_and_preserves_input_order() {
        let worker_list = workers();
        let sequence_hashes = vec![1_u64, 2_u64, 3_u64];
        let mut agents = HashMap::new();

        for sequence_hash in &sequence_hashes {
            let owner = select_g4_owner(*sequence_hash, &worker_list).unwrap();
            agents.entry(owner.worker_id).or_insert_with(|| {
                Arc::new(G4StorageAgent::new(
                    owner.worker_id,
                    Arc::new(MockTransferExecutor::default()),
                ))
            });
        }

        let existing_hash = sequence_hashes[1];
        let existing_owner = select_g4_owner(existing_hash, &worker_list).unwrap();
        agents
            .get(&existing_owner.worker_id)
            .unwrap()
            .put_blocks(vec![G4PutBlock {
                sequence_hash: existing_hash,
                disk_block_idx: 99,
                size_bytes: 4,
                checksum: None,
            }])
            .await;

        let client = G4StorageClient::new(worker_list, agents, Duration::from_millis(50));
        let accepted = client
            .offer_payload_blocks(vec![
                G4TransferBlock {
                    meta: G4PutBlock {
                        sequence_hash: 1,
                        disk_block_idx: 1,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![1; 4],
                },
                G4TransferBlock {
                    meta: G4PutBlock {
                        sequence_hash: 2,
                        disk_block_idx: 2,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![2; 4],
                },
                G4TransferBlock {
                    meta: G4PutBlock {
                        sequence_hash: 3,
                        disk_block_idx: 3,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![3; 4],
                },
            ])
            .await
            .unwrap();

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.meta.sequence_hash)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[tokio::test]
    async fn agent_offer_and_put_payload_registers_only_missing_blocks() {
        let agent = G4StorageAgent::new(7, Arc::new(MockTransferExecutor::default()));

        agent
            .put_blocks(vec![G4PutBlock {
                sequence_hash: 11,
                disk_block_idx: 4,
                size_bytes: 4,
                checksum: None,
            }])
            .await;

        let accepted = agent
            .offer_and_put_payload_blocks(vec![
                G4TransferBlock {
                    meta: G4PutBlock {
                        sequence_hash: 11,
                        disk_block_idx: 40,
                        size_bytes: 4,
                        checksum: None,
                    },
                    payload: vec![1; 4],
                },
                G4TransferBlock {
                    meta: G4PutBlock {
                        sequence_hash: 12,
                        disk_block_idx: 5,
                        size_bytes: 4,
                        checksum: Some([7; 32]),
                    },
                    payload: vec![2; 4],
                },
            ])
            .await
            .unwrap();

        assert_eq!(
            accepted
                .iter()
                .map(|block| block.meta.sequence_hash)
                .collect::<Vec<_>>(),
            vec![12]
        );

        let hits = agent.query_blocks(&[11, 12]).await;
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits.iter().find(|hit| hit.sequence_hash == 12).map(|hit| (
                hit.disk_block_idx,
                hit.size_bytes,
                hit.checksum
            )),
            Some((5, 4, Some([7; 32])))
        );
    }

    #[tokio::test]
    async fn client_query_blocks_fans_out_across_owners_concurrently() {
        let worker_list = workers();
        let mut by_owner = HashMap::<WorkerID, SequenceHash>::new();

        for sequence_hash in 1..=1024_u64 {
            let owner = select_g4_owner(sequence_hash, &worker_list).unwrap();
            by_owner.entry(owner.worker_id).or_insert(sequence_hash);
            if by_owner.len() >= 2 {
                break;
            }
        }

        assert!(by_owner.len() >= 2, "expected at least two owners");

        let delay = Duration::from_millis(75);
        let mut agents = HashMap::new();
        let mut query_hashes = Vec::new();

        for worker in &worker_list {
            if let Some(sequence_hash) = by_owner.get(&worker.worker_id) {
                let agent = Arc::new(
                    G4StorageAgent::new(
                        worker.worker_id,
                        Arc::new(MockTransferExecutor::default()),
                    )
                    .with_query_delay(delay),
                );
                agent
                    .put_blocks(vec![G4PutBlock {
                        sequence_hash: *sequence_hash,
                        disk_block_idx: worker.worker_id as usize,
                        size_bytes: 4096,
                        checksum: None,
                    }])
                    .await;
                agents.insert(worker.worker_id, agent);
                query_hashes.push(*sequence_hash);
            }
        }

        let client = G4StorageClient::new(worker_list, agents, Duration::from_millis(50));

        let start = Instant::now();
        let hits = client.query_blocks(&query_hashes).await;
        let elapsed = start.elapsed();

        assert_eq!(hits.len(), query_hashes.len());
        assert!(
            elapsed < delay * query_hashes.len() as u32,
            "expected concurrent owner fanout, elapsed={elapsed:?}, delay={delay:?}, owners={}",
            query_hashes.len()
        );
    }

    #[tokio::test]
    async fn client_fetch_preserves_non_contiguous_target_indices() {
        let worker_list = workers();
        let sequence_hashes = vec![1_u64, 2_u64, 3_u64];

        let mut agents = HashMap::new();
        let mut request_log = Vec::new();

        for sequence_hash in &sequence_hashes {
            let owner = select_g4_owner(*sequence_hash, &worker_list).unwrap();
            let transfer = Arc::new(MockTransferExecutor::default());
            let agent = agents
                .entry(owner.worker_id)
                .or_insert_with(|| Arc::new(G4StorageAgent::new(owner.worker_id, transfer.clone())))
                .clone();

            agent
                .put_blocks(vec![G4PutBlock {
                    sequence_hash: *sequence_hash,
                    disk_block_idx: (*sequence_hash as usize) * 10,
                    size_bytes: 1024,
                    checksum: None,
                }])
                .await;

            request_log.push((owner.worker_id, transfer));
        }

        let client = G4StorageClient::new(worker_list, agents, Duration::from_millis(50));
        let fetched = client
            .fetch_blocks(&sequence_hashes, BlockTransferPool::Host, 100)
            .await;

        assert_eq!(fetched.len(), 3);
        assert_eq!(fetched[0].target_block_idx, 100);
        assert_eq!(fetched[1].target_block_idx, 101);
        assert_eq!(fetched[2].target_block_idx, 102);

        for (worker_id, transfer) in request_log {
            let requests = transfer.requests.lock().await;
            for request in requests.iter() {
                for (_, target_idx) in &request.blocks {
                    assert!(
                        (100..=102).contains(target_idx),
                        "worker {worker_id} produced unexpected target idx {target_idx}"
                    );
                }
            }
        }
    }
}
