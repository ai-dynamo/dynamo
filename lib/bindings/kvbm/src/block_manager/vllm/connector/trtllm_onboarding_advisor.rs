// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::VllmBlockManager;
use crate::block_manager::vllm::KvbmRequest;
use crate::block_manager::vllm::connector::leader::slot::{ConnectorSlotManager, SlotManager};
use crate::get_current_tokio_handle;
use anyhow::Context;
use dynamo_llm::block_manager::block::transfer::{PoolConfig, TransferContext, read_from_remote};
use dynamo_llm::block_manager::distributed::{
    G2PB_COMPONENT_NAME, G2PB_NAMESPACE, G2pbFetchBlocksResponse, G2pbFetchRequest,
    G2pbPeerResolver, G2pbQueryHit, G2pbQueryRequest, G2pbRequestPlaneClient,
    route_g2pb_sequence_hashes_by_owner,
};
use dynamo_llm::block_manager::metrics_kvbm::KvbmMetrics;
use dynamo_llm::block_manager::offload::max_transfer_batch_size;
use dynamo_llm::block_manager::storage::DeviceAllocator;
use dynamo_llm::block_manager::{MutableBlock, Storage, block::locality::LocalityProvider};
use dynamo_llm::tokens::{SequenceHash, TokenBlock};
use dynamo_runtime::DistributedRuntime;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::oneshot;

#[derive(Clone)]
struct RemoteOnboardingContext {
    request_client: G2pbRequestPlaneClient,
    peer_resolver: G2pbPeerResolver,
    transfer_context: Arc<TransferContext>,
    imported_instances: Arc<Mutex<HashSet<u64>>>,
}

pub struct RemoteOnboardingAdvisor {
    drt: Option<Arc<DistributedRuntime>>,
    device_id: usize,
    slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
    kvbm_metrics: KvbmMetrics,
    remote_onboarding: Option<RemoteOnboardingContext>,
    inflight_remote_onboarding: Arc<Mutex<HashSet<String>>>,
}

impl std::fmt::Debug for RemoteOnboardingAdvisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteOnboardingAdvisor")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl RemoteOnboardingAdvisor {
    pub fn new(
        drt: Option<Arc<DistributedRuntime>>,
        device_id: usize,
        slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
        kvbm_metrics: KvbmMetrics,
    ) -> Self {
        Self {
            drt,
            device_id,
            slot_manager,
            kvbm_metrics,
            remote_onboarding: None,
            inflight_remote_onboarding: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    fn slot_manager(&self) -> anyhow::Result<&ConnectorSlotManager<String>> {
        self.slot_manager
            .get()
            .ok_or_else(|| anyhow::anyhow!("slot_manager not initialized"))
    }

    fn remote_onboarding_context(
        &mut self,
        timeout_ms: u64,
    ) -> anyhow::Result<Option<RemoteOnboardingContext>> {
        if let Some(context) = &self.remote_onboarding {
            return Ok(Some(context.clone()));
        }

        let Some(drt) = self.drt.clone() else {
            return Ok(None);
        };

        let component = drt
            .namespace(G2PB_NAMESPACE)?
            .component(G2PB_COMPONENT_NAME)?;
        let request_client = tokio::task::block_in_place(|| {
            get_current_tokio_handle().block_on(async {
                tokio::time::timeout(
                    Duration::from_millis(timeout_ms),
                    G2pbRequestPlaneClient::new(component),
                )
                .await
                .context("timed out waiting for G2PB request plane")?
            })
        })?;
        let peer_resolver = tokio::task::block_in_place(|| {
            get_current_tokio_handle().block_on(async {
                tokio::time::timeout(
                    Duration::from_millis(timeout_ms),
                    G2pbPeerResolver::new(request_client.clone()),
                )
                .await
                .context("timed out waiting for G2PB peer discovery")?
            })
        })?;
        let pool_config = PoolConfig {
            enable_pool: true,
            max_concurrent_transfers: 4,
            max_transfer_batch_size: max_transfer_batch_size(),
            num_outer_components: 1,
            num_layers: 1,
        };
        let transfer_context = Arc::new(TransferContext::new(
            self.slot_manager()?.block_manager().nixl_agent(),
            DeviceAllocator::new(self.device_id)?.ctx().new_stream()?,
            tokio::runtime::Handle::current(),
            Some(pool_config),
        )?);

        let context = RemoteOnboardingContext {
            request_client,
            peer_resolver,
            transfer_context,
            imported_instances: Arc::new(Mutex::new(HashSet::new())),
        };
        self.remote_onboarding = Some(context.clone());
        Ok(Some(context))
    }

    pub fn advise_async_onboarding(
        &mut self,
        request: KvbmRequest,
        timeout_ms: u64,
        min_blocks: u64,
    ) -> anyhow::Result<()> {
        let Some(context) = self.remote_onboarding_context(timeout_ms)? else {
            tracing::debug!(
                request_id = request.request_id,
                "remote onboarding advice ignored because no distributed runtime is available"
            );
            return Ok(());
        };

        {
            let mut inflight = self
                .inflight_remote_onboarding
                .lock()
                .map_err(|_| anyhow::anyhow!("remote onboarding set poisoned"))?;
            if !inflight.insert(request.request_id.clone()) {
                return Ok(());
            }
        }

        let request_id = request.request_id.clone();
        let salt_hash = request.salt_hash;
        let shared_slot = self.slot_manager()?.get_slot(&request_id)?;
        let sequence_blocks = {
            let slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
            slot.sequence().blocks().to_vec()
        };
        let block_manager = self.slot_manager()?.block_manager().clone();
        let kvbm_metrics = self.kvbm_metrics.clone();
        let inflight = self.inflight_remote_onboarding.clone();
        let (done_tx, done_rx) = oneshot::channel();

        tokio::spawn(async move {
            let result = execute_remote_onboarding(
                block_manager,
                sequence_blocks,
                salt_hash,
                min_blocks as usize,
                kvbm_metrics,
                context,
            )
            .await;

            match result {
                Ok(prefetched_blocks) => {
                    tracing::debug!(
                        request_id,
                        prefetched_blocks,
                        "completed advisory remote onboarding"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        request_id,
                        error = %error,
                        "advisory remote onboarding failed"
                    );
                }
            }

            if let Ok(mut guard) = inflight.lock() {
                guard.remove(&request_id);
            }

            let _ = done_tx.send(());
        });

        tokio::task::block_in_place(|| {
            get_current_tokio_handle().block_on(async move {
                let _ = tokio::time::timeout(Duration::from_millis(timeout_ms), done_rx).await;
            });
        });

        Ok(())
    }
}

async fn local_match_prefix_len<L>(
    block_manager: &dynamo_llm::block_manager::KvBlockManager<
        L,
        dynamo_llm::block_manager::BasicMetadata,
    >,
    sequence_hashes: &[SequenceHash],
) -> anyhow::Result<usize>
where
    L: dynamo_llm::block_manager::block::locality::LocalityProvider,
{
    let mut matched = 0usize;

    if let Some(device) = block_manager.device() {
        matched += device.match_sequence_hashes(sequence_hashes).await?.len();
    }

    if let Some(host) = block_manager.host() {
        matched += host
            .match_sequence_hashes(&sequence_hashes[matched..])
            .await?
            .len();
    }

    if let Some(disk) = block_manager.disk() {
        matched += disk
            .match_sequence_hashes(&sequence_hashes[matched..])
            .await?
            .len();
    }

    Ok(matched)
}

async fn execute_remote_onboarding(
    block_manager: VllmBlockManager,
    sequence_blocks: Vec<TokenBlock>,
    salt_hash: u64,
    min_blocks: usize,
    kvbm_metrics: KvbmMetrics,
    context: RemoteOnboardingContext,
) -> anyhow::Result<usize> {
    if sequence_blocks.is_empty() {
        return Ok(0);
    }

    let sequence_hashes: Vec<_> = sequence_blocks
        .iter()
        .map(|block| block.sequence_hash())
        .collect();
    let local_prefix_len = local_match_prefix_len(&block_manager, &sequence_hashes).await?;
    let missing_blocks = &sequence_blocks[local_prefix_len..];

    if missing_blocks.len() < min_blocks {
        return Ok(0);
    }

    let discovered_peers = context.peer_resolver.snapshot().await;
    if discovered_peers.is_empty() {
        return Ok(0);
    }

    let peer_list = discovered_peers.peers();
    let missing_hashes: Vec<_> = missing_blocks
        .iter()
        .map(|block| block.sequence_hash())
        .collect();
    let query_routes = route_g2pb_sequence_hashes_by_owner(&missing_hashes, &peer_list)?;
    let mut hit_map = HashMap::<SequenceHash, G2pbQueryHit>::new();

    for (instance_id, owner_hashes) in query_routes {
        let instance_id = discovered_peers.instance_id(instance_id)?;
        for hit in context
            .request_client
            .query(
                instance_id,
                G2pbQueryRequest {
                    sequence_hashes: owner_hashes,
                },
            )
            .await?
        {
            hit_map.insert(hit.sequence_hash, hit);
        }
    }

    let (contiguous_remote, post_gap_hits) =
        contiguous_remote_prefix_and_post_gap_hits(missing_blocks, &hit_map);

    if post_gap_hits > 0 {
        kvbm_metrics.g2pb_post_gap_requests.inc();
        kvbm_metrics
            .g2pb_post_gap_blocks
            .inc_by(post_gap_hits as u64);
    }

    if contiguous_remote.len() < min_blocks {
        return Ok(0);
    }

    let contiguous_hashes: Vec<_> = contiguous_remote
        .iter()
        .map(|block| block.sequence_hash())
        .collect();
    let token_blocks_by_hash: HashMap<SequenceHash, TokenBlock> = contiguous_remote
        .into_iter()
        .map(|block| (block.sequence_hash(), block))
        .collect();

    let host_pool = block_manager
        .host()
        .ok_or_else(|| anyhow::anyhow!("block manager has no host pool"))?;

    let fetch_routes = route_g2pb_sequence_hashes_by_owner(&contiguous_hashes, &peer_list)?;
    let mut prefetched = 0usize;

    for (instance_id, owner_hashes) in fetch_routes {
        let instance_id = discovered_peers.instance_id(instance_id)?;
        let fetched: G2pbFetchBlocksResponse = context
            .request_client
            .fetch(
                instance_id,
                G2pbFetchRequest {
                    sequence_hashes: owner_hashes.clone(),
                },
            )
            .await?;

        {
            let mut imported = context
                .imported_instances
                .lock()
                .map_err(|_| anyhow::anyhow!("imported instance set poisoned"))?;
            if imported.insert(instance_id) {
                block_manager.import_remote_blockset(fetched.blockset.clone())?;
            }
        }

        let remote_blocks = block_manager.get_remote_blocks_immutable(&fetched.descriptors)?;
        let mut local_host_blocks = host_pool.allocate_blocks(owner_hashes.len()).await?;

        for (block, sequence_hash) in local_host_blocks.iter_mut().zip(owner_hashes.iter()) {
            let token_block = token_blocks_by_hash.get(sequence_hash).ok_or_else(|| {
                anyhow::anyhow!("missing token block for sequence hash {sequence_hash}")
            })?;
            complete_block(block, salt_hash, token_block.tokens().as_ref())?;
        }

        let notify = read_from_remote(
            &remote_blocks,
            &mut local_host_blocks,
            context.transfer_context.clone(),
        )?;
        notify.await.context("remote onboarding transfer dropped")?;

        let immutable_host_blocks = host_pool.register_blocks(local_host_blocks).await?;
        prefetched += immutable_host_blocks.len();
    }

    Ok(prefetched)
}

fn contiguous_remote_prefix_and_post_gap_hits(
    missing_blocks: &[TokenBlock],
    hit_map: &HashMap<SequenceHash, G2pbQueryHit>,
) -> (Vec<TokenBlock>, usize) {
    let contiguous_remote: Vec<TokenBlock> = missing_blocks
        .iter()
        .take_while(|block| hit_map.contains_key(&block.sequence_hash()))
        .cloned()
        .collect();
    let post_gap_hits = missing_blocks[contiguous_remote.len()..]
        .iter()
        .filter(|block| hit_map.contains_key(&block.sequence_hash()))
        .count();

    (contiguous_remote, post_gap_hits)
}

fn complete_block<
    S: Storage,
    L: LocalityProvider,
    M: dynamo_llm::block_manager::block::BlockMetadata,
>(
    block: &mut MutableBlock<S, L, M>,
    salt_hash: u64,
    tokens: &[u32],
) -> anyhow::Result<()> {
    block.init_sequence(salt_hash)?;
    for token in tokens {
        block.add_token(*token)?;
    }
    block.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::block_manager::block::BasicMetadata;
    use dynamo_llm::block_manager::block::locality;
    use dynamo_llm::block_manager::config::{
        KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig, KvManagerRuntimeConfig,
    };
    use dynamo_llm::block_manager::storage::{PinnedStorage, cuda::PinnedAllocator};
    use dynamo_llm::tokens::{TokenBlockSequence, Tokens};
    use tokio_util::sync::CancellationToken;

    const BLOCK_SIZE: usize = 4;
    const SALT_HASH: u64 = 42;

    fn build_sequence_blocks(num_blocks: usize) -> Vec<TokenBlock> {
        let tokens: Vec<u32> = (1..=(num_blocks * BLOCK_SIZE) as u32).collect();
        TokenBlockSequence::new(Tokens::from(tokens), BLOCK_SIZE as u32, Some(SALT_HASH))
            .blocks()
            .to_vec()
    }

    async fn build_test_block_manager()
    -> anyhow::Result<dynamo_llm::block_manager::KvBlockManager<locality::Local, BasicMetadata>>
    {
        let cancel_token = CancellationToken::new();
        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(7)
                    .cancellation_token(cancel_token)
                    .disable_nixl()
                    .build()?,
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(1)
                    .outer_dim(1)
                    .page_size(BLOCK_SIZE)
                    .inner_dim(128)
                    .build()?,
            )
            .host_layout(
                KvManagerLayoutConfig::<PinnedStorage>::builder()
                    .num_blocks(8)
                    .allocator(PinnedAllocator::new(0)?)
                    .build()?,
            )
            .build()?;

        Ok(
            dynamo_llm::block_manager::KvBlockManager::<locality::Local, BasicMetadata>::new(
                config,
            )
            .await?,
        )
    }

    async fn register_blocks_with_hashes<
        S: Storage,
        L: dynamo_llm::block_manager::block::locality::LocalityProvider,
    >(
        pool: &dyn dynamo_llm::block_manager::BlockPool<S, L, BasicMetadata>,
        token_chunks: &[Vec<u32>],
    ) -> anyhow::Result<Vec<SequenceHash>> {
        let mut blocks = pool.allocate_blocks(token_chunks.len()).await?;
        let mut sequence_hashes = Vec::with_capacity(token_chunks.len());
        for (block, tokens) in blocks.iter_mut().zip(token_chunks.iter()) {
            complete_block(block, SALT_HASH, tokens)?;
            sequence_hashes.push(block.sequence_hash()?);
        }
        let _ = pool.register_blocks(blocks).await?;
        Ok(sequence_hashes)
    }

    #[test]
    fn contiguous_remote_prefix_stops_at_first_gap_and_counts_tail_hits() {
        let sequence_blocks = build_sequence_blocks(5);
        let mut hit_map = HashMap::<SequenceHash, G2pbQueryHit>::new();

        for index in [0usize, 1, 3, 4] {
            let block = &sequence_blocks[index];
            hit_map.insert(
                block.sequence_hash(),
                G2pbQueryHit {
                    instance_id: 11,
                    sequence_hash: block.sequence_hash(),
                    size_bytes: 1024,
                    checksum: 123u64,
                },
            );
        }

        let (contiguous_remote, post_gap_hits) =
            contiguous_remote_prefix_and_post_gap_hits(&sequence_blocks, &hit_map);

        assert_eq!(contiguous_remote.len(), 2);
        assert_eq!(post_gap_hits, 2);
        assert_eq!(
            contiguous_remote
                .iter()
                .map(|block| block.sequence_hash())
                .collect::<Vec<_>>(),
            sequence_blocks[..2]
                .iter()
                .map(|block| block.sequence_hash())
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn local_match_prefix_len_counts_host_prefix() -> anyhow::Result<()> {
        let block_manager = build_test_block_manager().await?;
        let host_pool = block_manager.host().expect("host pool");
        let present_hashes = register_blocks_with_hashes(
            host_pool,
            &[
                vec![11, 12, 13, 14],
                vec![21, 22, 23, 24],
                vec![31, 32, 33, 34],
            ],
        )
        .await?;
        let missing_hash = register_blocks_with_hashes(host_pool, &[vec![41, 42, 43, 44]])
            .await?
            .pop()
            .expect("missing hash");
        let query_hashes = vec![
            present_hashes[0],
            present_hashes[1],
            present_hashes[2],
            missing_hash ^ 0xDEADBEEF,
        ];

        let matched = local_match_prefix_len(&block_manager, &query_hashes).await?;
        assert_eq!(matched, 3);

        Ok(())
    }
}
