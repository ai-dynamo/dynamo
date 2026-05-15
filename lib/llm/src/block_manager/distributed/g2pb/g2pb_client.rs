// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    G2PB_ENDPOINT_NAME, G2pbCommitRequest, G2pbError, G2pbFetchBlocksResponse, G2pbFetchRequest,
    G2pbHealthResponse, G2pbLoadRemoteRequest, G2pbOfferRequest, G2pbOfferResponse, G2pbPeer,
    G2pbPutBlock, G2pbPutPayloadRequest, G2pbQueryHit, G2pbQueryRequest, G2pbRpcRequest,
    G2pbRpcResponse, G2pbStageBlocksRequest, G2pbStageBlocksResponse, G2pbStorageAgent,
    G2pbTransferBlock,
};
use crate::block_manager::block::nixl::SerializedNixlBlockSet;
use crate::tokens::{SequenceHash, compute_hash_v2};

use anyhow::{Context, Result};
use dynamo_runtime::{
    component::Component,
    pipeline::{RouterMode, SingleIn, network::egress::push_router::PushRouter},
    protocols::annotated::Annotated,
};
use futures::{StreamExt, future::join_all};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock as AsyncRwLock;

pub fn select_g2pb_owner(sequence_hash: SequenceHash, peers: &[G2pbPeer]) -> Option<G2pbPeer> {
    peers
        .iter()
        .cloned()
        .max_by_key(|peer| rendezvous_score(sequence_hash, peer.routing_id()))
}

fn rendezvous_score(sequence_hash: SequenceHash, instance_id: u64) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&sequence_hash.to_le_bytes());
    bytes[8..].copy_from_slice(&instance_id.to_le_bytes());
    compute_hash_v2(&bytes, 0)
}

fn route_items_by_g2pb_owner<T, F>(
    items: impl IntoIterator<Item = T>,
    peers: &[G2pbPeer],
    sequence_hash: F,
) -> Result<HashMap<u64, Vec<T>>, G2pbError>
where
    F: Fn(&T) -> SequenceHash,
{
    let mut routed = HashMap::<u64, Vec<T>>::new();

    for item in items {
        let owner = select_g2pb_owner(sequence_hash(&item), peers).ok_or(G2pbError::NoPeers)?;
        routed.entry(owner.instance_id).or_default().push(item);
    }

    Ok(routed)
}

pub fn route_g2pb_sequence_hashes_by_owner(
    sequence_hashes: &[SequenceHash],
    peers: &[G2pbPeer],
) -> Result<HashMap<u64, Vec<SequenceHash>>, G2pbError> {
    route_items_by_g2pb_owner(sequence_hashes.iter().copied(), peers, |sequence_hash| {
        *sequence_hash
    })
}

pub fn route_g2pb_put_blocks_by_owner(
    blocks: Vec<G2pbPutBlock>,
    peers: &[G2pbPeer],
) -> Result<HashMap<u64, Vec<G2pbPutBlock>>, G2pbError> {
    route_items_by_g2pb_owner(blocks, peers, |block| block.sequence_hash)
}

pub fn route_g2pb_transfer_blocks_by_owner(
    blocks: Vec<G2pbTransferBlock>,
    peers: &[G2pbPeer],
) -> Result<HashMap<u64, Vec<G2pbTransferBlock>>, G2pbError> {
    route_items_by_g2pb_owner(blocks, peers, |block| block.meta.sequence_hash)
}
pub struct G2pbStorageClient {
    peers: Vec<G2pbPeer>,
    agents: HashMap<u64, Arc<G2pbStorageAgent>>,
}

#[derive(Clone)]
pub struct G2pbRequestPlaneClient {
    router: PushRouter<G2pbRpcRequest, Annotated<G2pbRpcResponse>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct G2pbPeerInstance {
    pub peer: G2pbPeer,
    pub instance_id: u64,
}

#[derive(Debug, Clone, Default)]
pub struct G2pbDiscoveredPeers {
    peers_by_instance: HashMap<u64, G2pbPeerInstance>,
}

#[derive(Clone)]
pub struct G2pbPeerResolver {
    request_client: G2pbRequestPlaneClient,
    peers: Arc<AsyncRwLock<G2pbDiscoveredPeers>>,
}

impl G2pbRequestPlaneClient {
    pub async fn new(component: Component) -> Result<Self> {
        let client = component.endpoint(G2PB_ENDPOINT_NAME).client().await?;
        client.wait_for_instances().await?;
        let router = PushRouter::from_client_no_fault_detection(client, RouterMode::Direct).await?;
        Ok(Self { router })
    }

    pub fn instance_ids(&self) -> Vec<u64> {
        self.router.client.instance_ids()
    }

    pub fn instance_avail_watcher(&self) -> tokio::sync::watch::Receiver<Vec<u64>> {
        self.router.client.instance_avail_watcher()
    }

    async fn request(&self, instance_id: u64, request: G2pbRpcRequest) -> Result<G2pbRpcResponse> {
        let mut stream = self
            .router
            .direct(SingleIn::new(request), instance_id)
            .await?;
        let response = stream.next().await.ok_or_else(|| {
            anyhow::anyhow!("G2PB request to instance {instance_id} returned no response")
        })?;

        response.into_result()?.ok_or_else(|| {
            anyhow::anyhow!("G2PB request to instance {instance_id} returned an empty response")
        })
    }

    pub async fn health(&self, instance_id: u64) -> Result<G2pbHealthResponse> {
        match self.request(instance_id, G2pbRpcRequest::Health).await? {
            G2pbRpcResponse::Health(response) => Ok(response),
            other => anyhow::bail!("unexpected G2PB health response: {other:?}"),
        }
    }

    pub async fn put_blocks(&self, instance_id: u64, blocks: Vec<G2pbPutBlock>) -> Result<()> {
        match self
            .request(instance_id, G2pbRpcRequest::PutBlocks(blocks))
            .await?
        {
            G2pbRpcResponse::Ack => Ok(()),
            other => anyhow::bail!("unexpected G2PB put_blocks response: {other:?}"),
        }
    }

    pub async fn offer(
        &self,
        instance_id: u64,
        request: G2pbOfferRequest,
    ) -> Result<G2pbOfferResponse> {
        match self
            .request(instance_id, G2pbRpcRequest::Offer(request))
            .await?
        {
            G2pbRpcResponse::Offer(response) => Ok(response),
            other => anyhow::bail!("unexpected G2PB offer response: {other:?}"),
        }
    }

    pub async fn put_payload(
        &self,
        instance_id: u64,
        request: G2pbPutPayloadRequest,
    ) -> Result<Vec<G2pbTransferBlock>> {
        match self
            .request(instance_id, G2pbRpcRequest::PutPayload(request))
            .await?
        {
            G2pbRpcResponse::PutPayload(response) => Ok(response),
            other => anyhow::bail!("unexpected G2PB put_payload response: {other:?}"),
        }
    }

    pub async fn query(
        &self,
        instance_id: u64,
        request: G2pbQueryRequest,
    ) -> Result<Vec<G2pbQueryHit>> {
        match self
            .request(instance_id, G2pbRpcRequest::Query(request))
            .await?
        {
            G2pbRpcResponse::Query(response) => Ok(response),
            other => anyhow::bail!("unexpected G2PB query response: {other:?}"),
        }
    }

    pub async fn fetch(
        &self,
        instance_id: u64,
        request: G2pbFetchRequest,
    ) -> Result<G2pbFetchBlocksResponse> {
        match self
            .request(instance_id, G2pbRpcRequest::Fetch(request))
            .await?
        {
            G2pbRpcResponse::Fetch(response) => Ok(response),
            other => anyhow::bail!("unexpected G2PB fetch response: {other:?}"),
        }
    }

    pub async fn stage_put(
        &self,
        instance_id: u64,
        request: G2pbStageBlocksRequest,
    ) -> Result<G2pbStageBlocksResponse> {
        match self
            .request(instance_id, G2pbRpcRequest::StagePut(request))
            .await?
        {
            G2pbRpcResponse::StagePut(response) => Ok(response),
            other => anyhow::bail!("unexpected G2PB stage_put response: {other:?}"),
        }
    }

    pub async fn commit_put(&self, instance_id: u64, request: G2pbCommitRequest) -> Result<()> {
        match self
            .request(instance_id, G2pbRpcRequest::CommitPut(request))
            .await?
        {
            G2pbRpcResponse::Ack => Ok(()),
            other => anyhow::bail!("unexpected G2PB commit_put response: {other:?}"),
        }
    }

    pub async fn load_remote(
        &self,
        instance_id: u64,
        request: G2pbLoadRemoteRequest,
    ) -> Result<()> {
        match self
            .request(instance_id, G2pbRpcRequest::LoadRemote(request))
            .await?
        {
            G2pbRpcResponse::Ack => Ok(()),
            other => anyhow::bail!("unexpected G2PB load_remote response: {other:?}"),
        }
    }
}

impl G2pbPeerResolver {
    pub async fn new(request_client: G2pbRequestPlaneClient) -> Result<Self> {
        let initial = discover_g2pb_peers(&request_client).await?;
        let peers = Arc::new(AsyncRwLock::new(initial));
        let resolver = Self {
            request_client,
            peers,
        };
        resolver.spawn_refresh_task();
        Ok(resolver)
    }

    fn spawn_refresh_task(&self) {
        let request_client = self.request_client.clone();
        let peers = self.peers.clone();
        let mut watcher = request_client.instance_avail_watcher();

        tokio::spawn(async move {
            loop {
                if watcher.changed().await.is_err() {
                    break;
                }

                match discover_g2pb_peers(&request_client).await {
                    Ok(refreshed) => {
                        *peers.write().await = refreshed;
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "failed to refresh G2PB peer snapshot from discovery watch");
                    }
                }
            }
        });
    }

    pub async fn snapshot(&self) -> G2pbDiscoveredPeers {
        self.peers.read().await.clone()
    }
}

impl G2pbDiscoveredPeers {
    pub fn from_health_responses(discovered: Vec<(u64, G2pbHealthResponse)>) -> Result<Self> {
        let mut peers_by_instance = HashMap::with_capacity(discovered.len());
        for (instance_id, health) in discovered {
            let peer = G2pbPeer {
                instance_id: health.instance_id,
                endpoint: health.listen,
                hostname: health.hostname,
            };
            let resolved = G2pbPeerInstance {
                peer: peer.clone(),
                instance_id,
            };

            if let Some(previous) = peers_by_instance.insert(health.instance_id, resolved) {
                anyhow::bail!(
                    "duplicate remote instance_id {} discovered at instance {} and {}",
                    health.instance_id,
                    previous.instance_id,
                    instance_id
                );
            }
        }

        Ok(Self { peers_by_instance })
    }

    pub fn is_empty(&self) -> bool {
        self.peers_by_instance.is_empty()
    }

    pub fn peers(&self) -> Vec<G2pbPeer> {
        let mut peers: Vec<_> = self
            .peers_by_instance
            .values()
            .map(|resolved| resolved.peer.clone())
            .collect();
        peers.sort_by_key(|peer| peer.routing_id());
        peers
    }

    pub fn instances(&self) -> Vec<G2pbPeerInstance> {
        let mut instances: Vec<_> = self.peers_by_instance.values().cloned().collect();
        instances.sort_by_key(|resolved| resolved.peer.routing_id());
        instances
    }

    pub fn instance_id(&self, instance_id: u64) -> Result<u64> {
        self.peers_by_instance
            .get(&instance_id)
            .map(|resolved| resolved.instance_id)
            .ok_or_else(|| {
                anyhow::anyhow!("missing backend instance for instance_id {instance_id}")
            })
    }

    pub async fn load_remote_blockset(
        &self,
        request_client: &G2pbRequestPlaneClient,
        blockset: SerializedNixlBlockSet,
    ) -> Result<()> {
        for resolved in self.instances() {
            request_client
                .load_remote(
                    resolved.instance_id,
                    G2pbLoadRemoteRequest {
                        blockset: blockset.clone(),
                    },
                )
                .await
                .with_context(|| {
                    format!(
                        "failed to publish local blockset to instance {}",
                        resolved.peer.instance_id
                    )
                })?;
        }

        Ok(())
    }
}

pub async fn discover_g2pb_peers(
    request_client: &G2pbRequestPlaneClient,
) -> Result<G2pbDiscoveredPeers> {
    let mut discovered = Vec::new();
    for instance_id in request_client.instance_ids() {
        let health = request_client.health(instance_id).await?;
        discovered.push((instance_id, health));
    }

    G2pbDiscoveredPeers::from_health_responses(discovered)
}

impl G2pbStorageClient {
    pub fn new(peers: Vec<G2pbPeer>, agents: HashMap<u64, Arc<G2pbStorageAgent>>) -> Self {
        Self { peers, agents }
    }

    pub fn owner_for(&self, sequence_hash: SequenceHash) -> Result<G2pbPeer, G2pbError> {
        select_g2pb_owner(sequence_hash, &self.peers).ok_or(G2pbError::NoPeers)
    }

    pub async fn put_blocks(&self, blocks: Vec<G2pbPutBlock>) -> Result<(), G2pbError> {
        let routed = route_g2pb_put_blocks_by_owner(blocks, &self.peers)?;
        for (instance_id, blocks) in routed {
            let agent = self
                .agents
                .get(&instance_id)
                .ok_or(G2pbError::UnknownPeer { instance_id })?;
            agent.put_blocks(blocks).await;
        }

        Ok(())
    }

    pub async fn offer_blocks(
        &self,
        blocks: Vec<G2pbPutBlock>,
    ) -> Result<Vec<G2pbPutBlock>, G2pbError> {
        let routed = route_g2pb_put_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<u64, HashSet<SequenceHash>>::new();
        for (instance_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&instance_id)
                .ok_or(G2pbError::UnknownPeer { instance_id })?;
            let accepted = agent.offer_blocks(&owner_blocks).await;
            accepted_by_owner.insert(instance_id, accepted.into_iter().collect());
        }

        Ok(filter_accepted_put_blocks(
            blocks,
            |sequence_hash| self.owner_for(sequence_hash).ok(),
            &accepted_by_owner,
        ))
    }

    pub async fn offer_and_put_blocks(
        &self,
        blocks: Vec<G2pbPutBlock>,
    ) -> Result<Vec<G2pbPutBlock>, G2pbError> {
        let routed = route_g2pb_put_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<u64, HashSet<SequenceHash>>::new();
        for (instance_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&instance_id)
                .ok_or(G2pbError::UnknownPeer { instance_id })?;
            let accepted = agent.offer_and_put_blocks(owner_blocks).await;
            accepted_by_owner.insert(
                instance_id,
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
        blocks: Vec<G2pbTransferBlock>,
    ) -> Result<Vec<G2pbTransferBlock>, G2pbError> {
        let routed = route_g2pb_transfer_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<u64, HashSet<SequenceHash>>::new();
        for (instance_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&instance_id)
                .ok_or(G2pbError::UnknownPeer { instance_id })?;
            let accepted = agent.offered_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                instance_id,
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
        blocks: Vec<G2pbTransferBlock>,
    ) -> Result<Vec<G2pbTransferBlock>, G2pbError> {
        let routed = route_g2pb_transfer_blocks_by_owner(blocks.clone(), &self.peers)?;
        let mut accepted_by_owner = HashMap::<u64, HashSet<SequenceHash>>::new();
        for (instance_id, owner_blocks) in routed {
            let agent = self
                .agents
                .get(&instance_id)
                .ok_or(G2pbError::UnknownPeer { instance_id })?;
            let accepted = agent.offer_and_put_payload_blocks(owner_blocks).await?;
            accepted_by_owner.insert(
                instance_id,
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
    ) -> HashMap<SequenceHash, G2pbQueryHit> {
        let grouped =
            route_g2pb_sequence_hashes_by_owner(sequence_hashes, &self.peers).unwrap_or_default();

        let queries = grouped
            .into_iter()
            .filter_map(|(instance_id, sequence_hashes)| {
                self.agents
                    .get(&instance_id)
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

    pub async fn fetch_blocks(&self, sequence_hashes: &[SequenceHash]) -> Vec<G2pbTransferBlock> {
        let grouped =
            route_g2pb_sequence_hashes_by_owner(sequence_hashes, &self.peers).unwrap_or_default();

        let mut fetched = Vec::new();
        for (instance_id, owner_hashes) in grouped {
            let Some(agent) = self.agents.get(&instance_id) else {
                continue;
            };

            match agent.fetch_blocks(&owner_hashes).await {
                Ok(mut blocks) => {
                    blocks.retain(|block| block.validate_payload_integrity().is_ok());
                    fetched.append(&mut blocks);
                }
                Err(G2pbError::NotFound { .. }) => {
                    // G2PB is a cache. Fetch failures degrade to cache misses.
                }
                Err(G2pbError::InvalidPayloadSize { .. }) => {}
                Err(G2pbError::NoPeers | G2pbError::UnknownPeer { .. }) => {}
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
    blocks: Vec<G2pbPutBlock>,
    owner_for: F,
    accepted_by_owner: &HashMap<u64, HashSet<SequenceHash>>,
) -> Vec<G2pbPutBlock>
where
    F: Fn(SequenceHash) -> Option<G2pbPeer>,
{
    blocks
        .into_iter()
        .fold(
            (Vec::new(), HashSet::new()),
            |(mut accepted_blocks, mut seen), block| {
                let sequence_hash = block.sequence_hash;
                let is_accepted = owner_for(sequence_hash)
                    .and_then(|owner| accepted_by_owner.get(&owner.instance_id))
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
    blocks: Vec<G2pbTransferBlock>,
    owner_for: F,
    accepted_by_owner: &HashMap<u64, HashSet<SequenceHash>>,
) -> Vec<G2pbTransferBlock>
where
    F: Fn(SequenceHash) -> Option<G2pbPeer>,
{
    blocks
        .into_iter()
        .fold(
            (Vec::new(), HashSet::new()),
            |(mut accepted_blocks, mut seen), block| {
                let sequence_hash = block.meta.sequence_hash;
                let is_accepted = owner_for(sequence_hash)
                    .and_then(|owner| accepted_by_owner.get(&owner.instance_id))
                    .is_some_and(|accepted| accepted.contains(&sequence_hash));

                if is_accepted && seen.insert(sequence_hash) {
                    accepted_blocks.push(block);
                }

                (accepted_blocks, seen)
            },
        )
        .0
}
