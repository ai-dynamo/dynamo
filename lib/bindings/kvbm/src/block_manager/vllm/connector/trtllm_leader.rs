// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::block_manager::BlockManagerBuilder;
use crate::block_manager::vllm::connector::leader::slot::{
    ConnectorSlotManager, SlotManager, SlotState,
};
use crate::block_manager::vllm::connector::leader::{
    kvbm_metrics_endpoint_enabled, parse_kvbm_metrics_port,
};
use crate::block_manager::{distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest};
use crate::get_current_tokio_handle;
use anyhow::Context;
use dynamo_llm::block_manager::block::transfer::{PoolConfig, TransferContext, read_from_remote};
use dynamo_llm::block_manager::connector::protocol::RequestType;
use dynamo_llm::block_manager::distributed::{
    G3PB_COMPONENT_NAME, G3PB_NAMESPACE, G3pbFetchBlocksResponse, G3pbFetchRequest,
    G3pbPeerResolver, G3pbQueryHit, G3pbQueryRequest, G3pbRequestPlaneClient,
    route_g3pb_sequence_hashes_by_owner,
};
use dynamo_llm::block_manager::kv_consolidator::EventSource;
use dynamo_llm::block_manager::metrics_kvbm::{KvbmMetrics, KvbmMetricsRegistry};
use dynamo_llm::block_manager::{MutableBlock, Storage, block::locality::LocalityProvider};
use dynamo_llm::block_manager::offload::max_transfer_batch_size;
use dynamo_llm::block_manager::storage::DeviceAllocator;
use dynamo_llm::tokens::{SequenceHash, TokenBlock};
use dynamo_runtime::DistributedRuntime;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::runtime::Handle;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct RemotePrefetchContext {
    request_client: G3pbRequestPlaneClient,
    peer_resolver: G3pbPeerResolver,
    transfer_context: Arc<TransferContext>,
    imported_instances: Arc<Mutex<HashSet<u64>>>,
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

pub trait Leader: Send + Sync + std::fmt::Debug {
    fn get_num_new_matched_tokens(
        &mut self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)>;

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        context_current_position: usize,
    ) -> anyhow::Result<()>;

    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>>;

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool>;

    fn has_slot(&self, request_id: String) -> bool;

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()>;

    fn advise_async_loading(
        &mut self,
        request: KvbmRequest,
        tokens: Vec<u32>,
        transfer_budget_ms: u64,
        min_blocks: u64,
    ) -> anyhow::Result<()>;

    fn slot_manager(&self) -> &ConnectorSlotManager<String>;
}

pub struct KvConnectorLeader {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
    drt: Option<Arc<DistributedRuntime>>,
    rank: usize,
    device_id: usize,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    iteration_counter: u64,
    inflight_request_to_num_external_tokens: HashMap<String, usize>,
    remote_prefetch: Option<RemotePrefetchContext>,
    inflight_remote_prefetch: Arc<Mutex<HashSet<String>>>,
    remote_prefetch_cancellation: Arc<Mutex<HashMap<String, CancellationToken>>>,
    kvbm_metrics: KvbmMetrics,
}

impl std::fmt::Debug for KvConnectorLeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvConnectorLeader")
            .field("rank", &self.rank)
            .field("device_id", &self.device_id)
            .field("block_size", &self.block_size)
            .field("iteration_counter", &self.iteration_counter)
            .finish()
    }
}

impl KvConnectorLeader {
    fn new(
        rank: u64,
        device_id: u64,
        drt: Option<Arc<DistributedRuntime>>,
        page_size: usize,
        leader_py: PyKvbmLeader,
        consolidator_trtllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
    ) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with rank: {}, device_id: {}",
            rank,
            device_id
        );

        let leader = leader_py.get_inner().clone();
        let handle: Handle = get_current_tokio_handle();

        let kvbm_metrics = KvbmMetrics::new(
            &KvbmMetricsRegistry::default(),
            kvbm_metrics_endpoint_enabled(),
            parse_kvbm_metrics_port(),
        );

        let kvbm_metrics_clone = kvbm_metrics.clone();

        let slot_manager_cell = Arc::new(OnceLock::new());

        {
            let slot_manager_cell = slot_manager_cell.clone();
            // Capture consolidator endpoints for the async block
            let consolidator_trtllm_ep = consolidator_trtllm_endpoint.clone();
            let consolidator_output_ep = consolidator_output_endpoint.clone();

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                let mut block_manager_builder = BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .kvbm_metrics(kvbm_metrics_clone.clone());

                // Add consolidator config if endpoint is provided
                // For TRTLLM: engine_endpoint is where TRTLLM publishes, output_endpoint is where consolidator publishes
                if let Some(trtllm_ep) = consolidator_trtllm_ep.clone() {
                    tracing::info!(
                        "Consolidator config: trtllm_endpoint={}, consolidated_output_endpoint={:?}",
                        trtllm_ep,
                        consolidator_output_ep
                    );

                    block_manager_builder = block_manager_builder.consolidator_config(
                        trtllm_ep,
                        consolidator_output_ep,
                        EventSource::Trtllm,
                    );
                }

                let block_manager = match block_manager_builder.build().await
                {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

                // Create the slot manager now that everything is ready
                let sm = ConnectorSlotManager::new(
                    block_manager.get_block_manager().clone(),
                    leader.clone(),
                    kvbm_metrics_clone.clone(),
                    Some(format!("worker-{}", rank)),
                );

                let _ = slot_manager_cell.set(sm);

                tracing::info!("KvConnectorLeader init complete.");
            });
        }

        Self {
            slot_manager: slot_manager_cell,
            drt,
            rank: rank as usize,
            device_id: device_id as usize,
            block_size: page_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
            inflight_request_to_num_external_tokens: HashMap::new(),
            remote_prefetch: None,
            inflight_remote_prefetch: Arc::new(Mutex::new(HashSet::new())),
            remote_prefetch_cancellation: Arc::new(Mutex::new(HashMap::new())),
            kvbm_metrics,
        }
    }

    fn remote_prefetch_context(
        &mut self,
        timeout_ms: u64,
    ) -> anyhow::Result<Option<RemotePrefetchContext>> {
        if let Some(context) = &self.remote_prefetch {
            return Ok(Some(context.clone()));
        }

        let Some(drt) = self.drt.clone() else {
            return Ok(None);
        };

        let component = drt.namespace(G3PB_NAMESPACE)?.component(G3PB_COMPONENT_NAME)?;
        let request_client = tokio::task::block_in_place(|| {
            get_current_tokio_handle().block_on(async {
                tokio::time::timeout(
                    Duration::from_millis(timeout_ms),
                    G3pbRequestPlaneClient::new(component),
                )
                .await
                .context("timed out waiting for G3PB request plane")?
            })
        })?;
        let peer_resolver = tokio::task::block_in_place(|| {
            get_current_tokio_handle().block_on(async {
                tokio::time::timeout(
                    Duration::from_millis(timeout_ms),
                    G3pbPeerResolver::new(request_client.clone()),
                )
                .await
                .context("timed out waiting for G3PB peer discovery")?
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
            self.slot_manager().block_manager().nixl_agent(),
            DeviceAllocator::new(self.device_id)?.ctx().new_stream()?,
            tokio::runtime::Handle::current(),
            Some(pool_config),
        )?);

        let context = RemotePrefetchContext {
            request_client,
            peer_resolver,
            transfer_context,
            imported_instances: Arc::new(Mutex::new(HashSet::new())),
        };
        self.remote_prefetch = Some(context.clone());
        Ok(Some(context))
    }
}

async fn local_match_prefix_len<L>(
    block_manager: &dynamo_llm::block_manager::KvBlockManager<L, dynamo_llm::block_manager::BasicMetadata>,
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
        matched += host.match_sequence_hashes(&sequence_hashes[matched..]).await?.len();
    }

    if let Some(disk) = block_manager.disk() {
        matched += disk.match_sequence_hashes(&sequence_hashes[matched..]).await?.len();
    }

    Ok(matched)
}

async fn execute_remote_prefetch(
    block_manager: crate::block_manager::VllmBlockManager,
    sequence_blocks: Vec<TokenBlock>,
    salt_hash: u64,
    min_blocks: usize,
    kvbm_metrics: KvbmMetrics,
    context: RemotePrefetchContext,
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
    let query_routes = route_g3pb_sequence_hashes_by_owner(&missing_hashes, &peer_list)?;
    let mut hit_map = HashMap::<SequenceHash, G3pbQueryHit>::new();

    for (instance_id, owner_hashes) in query_routes {
        let instance_id = discovered_peers.instance_id(instance_id)?;
        for hit in context
            .request_client
            .query(
                instance_id,
                G3pbQueryRequest {
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
        kvbm_metrics.g3pb_post_gap_requests.inc();
        kvbm_metrics.g3pb_post_gap_blocks.inc_by(post_gap_hits as u64);
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

    let fetch_routes = route_g3pb_sequence_hashes_by_owner(&contiguous_hashes, &peer_list)?;
    let mut prefetched = 0usize;

    for (instance_id, owner_hashes) in fetch_routes {
        let instance_id = discovered_peers.instance_id(instance_id)?;
        let fetched: G3pbFetchBlocksResponse = context
            .request_client
            .fetch(
                instance_id,
                G3pbFetchRequest {
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
            let token_block = token_blocks_by_hash
                .get(sequence_hash)
                .ok_or_else(|| anyhow::anyhow!("missing token block for sequence hash {sequence_hash}"))?;
            complete_block(block, salt_hash, token_block.tokens().as_ref())?;
        }

        let notify = read_from_remote(
            &remote_blocks,
            &mut local_host_blocks,
            context.transfer_context.clone(),
        )?;
        notify.await.context("remote prefetch transfer dropped")?;

        let immutable_host_blocks = host_pool.register_blocks(local_host_blocks).await?;
        prefetched += immutable_host_blocks.len();
    }

    Ok(prefetched)
}

fn contiguous_remote_prefix_and_post_gap_hits(
    missing_blocks: &[TokenBlock],
    hit_map: &HashMap<SequenceHash, G3pbQueryHit>,
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

impl Leader for KvConnectorLeader {
    #[inline]
    fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.slot_manager
            .get()
            .expect("slot_manager not initialized")
    }

    /// Match the tokens in the request with the available block pools.
    /// Note: the necessary details of the request are captured prior to this call. For trtllm,
    /// we make a create slot call prior to this call, so a slot is guaranteed to exist.
    ///
    /// To align with the connector interface, we must ensure that if no blocks are matched, we return (0, false).
    /// In our implementation, if we match any block, we return (num_matched_tokens, true).
    #[tracing::instrument(level = "debug", skip(self, request_num_tokens, num_computed_tokens))]
    fn get_num_new_matched_tokens(
        &mut self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)> {
        tracing::debug!(
            "request_num_tokens: {request_num_tokens}; num_computed_tokens: {num_computed_tokens}"
        );

        // TRTLLM could match partial blocks if enable_partial_reuse = True,
        // immediately return 0 to simplify things.
        if !num_computed_tokens.is_multiple_of(self.block_size) {
            return Ok((0, false));
        }

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        // early exit if we cannot match full block
        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            let total_tokens = slot.sequence().total_tokens();
            tracing::debug!(
                "total_tokens in sequence: {total_tokens}; num_computed_tokens: {num_computed_tokens}; can not match full block."
            );
            return Ok((0, false));
        }

        // find matches for any remaining tokens
        // this will advance the computed position and hold any newly matched blocks in the slot
        slot.acquire_local_matches(num_computed_tokens)?;

        // return the number of external tokens that are ready for onboarding
        // we always return true here as we always asynchronously onboard matched blocks
        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!(
                (num_computed_tokens + num_external_tokens).is_multiple_of(self.block_size)
            );
            tracing::debug!(
                request_id = request_id,
                "scheduling onboarding for {} external tokens",
                num_external_tokens
            );
            // Add to the map so that onboarding can be triggered in update_state_after_alloc.
            self.inflight_request_to_num_external_tokens
                .insert(request_id, num_external_tokens);

            self.kvbm_metrics
                .matched_tokens
                .inc_by(num_external_tokens as u64);
            Ok((num_external_tokens, true))
        } else {
            Ok((0, false))
        }
    }

    /// Note: TRTLLM will not provide any scheduler output data for requests that are onboarding. it is entirely
    /// on the connector's implementation to handle this case.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id))]
    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        context_current_position: usize,
    ) -> anyhow::Result<()> {
        tracing::debug!(
            request_id,
            "num_device_blocks: {}, context_current_position: {}",
            block_ids.len(),
            context_current_position
        );

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        // we have not yet advanced the computed position, but now we can, since we have an indication that we have
        // necessary gpu blocks into which we will load the external tokens.

        slot.append_mutable_device_blocks(&block_ids)?;

        if let Some(&num_external_tokens) = self
            .inflight_request_to_num_external_tokens
            .get(&request_id)
        {
            if num_external_tokens > 0 {
                let num_computed_tokens = context_current_position - num_external_tokens;
                slot.record_cached_device_tokens(num_computed_tokens);
                slot.advance_computed_position(num_computed_tokens)?;

                tracing::debug!(
                    request_id = request_id,
                    "triggering onboarding for {} external tokens",
                    num_external_tokens
                );
                slot.trigger_onboarding(num_external_tokens)?;
                self.onboarding_slots.insert(request_id.clone());
            }

            self.inflight_request_to_num_external_tokens
                .remove(&request_id);
        }

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(iteration = self.iteration_counter + 1))]
    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>> {
        // the iteration counter is used to track the number of times we have built the connector metadata
        // all connetor operations have the iteration counter at which they were issued.
        // this allows operations to be lazily enqueued to the transfer engine
        // the worker side of the connector will track all operations for completion before the request is
        // allowed to be marked as finished.
        self.iteration_counter += 1;
        let iteration = self.iteration_counter;

        tracing::debug!("Building connector metadata");
        tracing::debug!("SchedulerOutput: {scheduler_output:#?}");

        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);

        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        // Worker-side - we create a request slot for onboarding, then delete it when onboarding is finished, then
        // recreate it again when we start the prefill/decode phase.
        //
        // This is kind of a nice abstraction as it keeps the events simplier; however, we now create the request-slot
        // once for onboarding (this loop), then again for prefill/decode (new_requests loop).
        //
        // TODO(krish): Consider a more deterministic way to count immediate ops.
        // Currently we count by filtering pending_ops at runtime. A higher-level approach
        // (e.g., tracking count when onboard_blocks is called, or deriving from architecture
        // config) might be more robust against potential timing-related issues.
        for request_id in onboarding_slots.iter() {
            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                // Count immediate (onboard) operations for this slot
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

                // Create slot with expected immediate ops BEFORE adding operations
                md.create_slot(request_id.clone(), num_immediate);
                md.add_operations(pending_ops);
            } else {
                md.create_slot(request_id.clone(), 0);
            }
        }

        // todo: update the code and abstraction to account for this two-phase lifecycle.
        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;

            let already_created = md.new_slots.iter().any(|s| &s.request_id == request_id);

            // Skip if this slot was already created in the onboarding_slots loop above.
            // This prevents overwriting the slot with expected_immediate_ops=0 when it should have the correct count.
            if already_created {
                assert!(
                    inflight_requests.remove(request_id),
                    "request_id {request_id} not found in inflight_requests: "
                );
                continue;
            }

            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            slot.record_start_iteration(iteration)?;

            debug_assert!(
                matches!(
                    slot.state(),
                    SlotState::Initialized | SlotState::Onboarding(_)
                ),
                "current slot state: {:?}",
                slot.state()
            );

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(&new_req.request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &[],
                &new_req.block_ids,
                new_req.num_computed_tokens,
                scheduled_tokens,
                new_req.priorities.as_deref(),
            )?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                // Count immediate (onboard) operations for this slot
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

                // Create slot with expected immediate ops BEFORE adding operations
                md.create_slot(new_req.request_id.clone(), num_immediate);

                tracing::debug!(
                    "adding {} pending operations for slot {} ({} immediate)",
                    pending_ops.len(),
                    new_req.request_id,
                    num_immediate
                );
                md.add_operations(pending_ops);
            } else {
                md.create_slot(new_req.request_id.clone(), 0);
            }
        }

        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;

            // note: evicition might trigger this assert
            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(&cached_req.request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &cached_req.new_token_ids,
                &cached_req.new_block_ids,
                cached_req.num_computed_tokens,
                scheduled_tokens,
                cached_req.priorities.as_deref(),
            )?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!(
                    "adding {} pending operations for slot {}",
                    pending_ops.len(),
                    request_id
                );
                md.add_operations(pending_ops);
            }
        }

        tracing::debug!("metadata: {md:#?}");
        serde_json::to_vec(&md)
            .map_err(|e| anyhow::anyhow!("Failed to serialize connector metadata: {}", e))
    }

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        tracing::debug!("Request finished: {request_id}; block_ids: {block_ids:?}");
        // grab the slot
        let shared_slot = self.slot_manager().get_slot(&request_id)?;

        // mark the slot as finished
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
        slot.mark_as_finished(self.iteration_counter)?;

        // todo: allow the request to resolve when it should exit
        // the request may have some outstanding operations
        // we would like to inform it to shutdown, then have it signal to the work that is officially gone,
        // then we can remove the slot and trigger the worker to clean up as well.

        // remove it from the manager as we will never use it again
        self.slot_manager().remove_slot(&request_id)?;
        self.inflight_request_to_num_external_tokens
            .remove(&request_id);
        if let Ok(mut inflight) = self.inflight_remote_prefetch.lock() {
            inflight.remove(&request_id);
        }
        if let Ok(mut cancellations) = self.remote_prefetch_cancellation.lock()
            && let Some(token) = cancellations.remove(&request_id)
        {
            token.cancel();
        }

        // if the slot has finished, we can return false to trtllm, indicating all gpu blocks are free to be reused
        // otherwise, we return true, which means there are still outstanding operations on gpu blocks which
        // must be awaited before the gpu blocks can be reused. if we return true, then it is the worker side
        // of the connector api which will be used to inform trtllm that the request is finished.
        if let SlotState::Finished = slot.state() {
            Ok(false)
        } else {
            debug_assert!(matches!(slot.state(), SlotState::Finishing));
            Ok(true)
        }
    }

    fn has_slot(&self, request_id: String) -> bool {
        self.slot_manager().has_slot(&request_id)
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()> {
        self.slot_manager()
            .create_slot(&request.request_id, tokens, request.salt_hash)?;

        self.inflight_requests.insert(request.request_id);

        Ok(())
    }

    fn advise_async_loading(
        &mut self,
        request: KvbmRequest,
        tokens: Vec<u32>,
        transfer_budget_ms: u64,
        min_blocks: u64,
    ) -> anyhow::Result<()> {
        // This is a best-effort pre-scheduling hint, not a load promise.
        //
        // Intended call timing:
        // - after TRT-LLM has activated request objects for the upcoming iteration
        // - before the scheduler runs for that iteration
        //
        // Intended caller shape:
        // - inspect active/context requests that are likely to need remote KV soon
        // - call advise_async_loading(request, tokens, budget, min_blocks)
        // - allow a bounded grace period for remote-to-local prefetch work
        // - then run normal scheduling using only local KVBM matches
        //
        // This API must not be used to promise remote availability to TRT-LLM.
        // It starts best-effort leader-side remote prefetch into local KVBM state.
        if !self.has_slot(request.request_id.clone()) {
            self.create_slot(request.clone(), tokens.clone())?;
        }

        let Some(context) = self.remote_prefetch_context(transfer_budget_ms)? else {
            tracing::debug!(
                request_id = request.request_id,
                "remote advice ignored because no distributed runtime is available"
            );
            return Ok(());
        };

        {
            let mut inflight = self
                .inflight_remote_prefetch
                .lock()
                .map_err(|_| anyhow::anyhow!("remote prefetch set poisoned"))?;
            if !inflight.insert(request.request_id.clone()) {
                return Ok(());
            }
        }
        let cancellation = CancellationToken::new();
        {
            let mut cancellations = self
                .remote_prefetch_cancellation
                .lock()
                .map_err(|_| anyhow::anyhow!("remote prefetch cancellation map poisoned"))?;
            cancellations.insert(request.request_id.clone(), cancellation.clone());
        }

        let request_id = request.request_id.clone();
        let salt_hash = request.salt_hash;
        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let sequence_blocks = {
            let slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
            slot.sequence().blocks().to_vec()
        };
        let block_manager = self.slot_manager().block_manager().clone();
        let kvbm_metrics = self.kvbm_metrics.clone();
        let inflight = self.inflight_remote_prefetch.clone();
        let cancellations = self.remote_prefetch_cancellation.clone();
        let (done_tx, done_rx) = oneshot::channel();

        tokio::spawn(async move {
            let result = tokio::select! {
                _ = cancellation.cancelled() => {
                    tracing::debug!(request_id, "cancelled advisory remote prefetch");
                    Ok(0usize)
                }
                result = execute_remote_prefetch(
                    block_manager,
                    sequence_blocks,
                    salt_hash,
                    min_blocks as usize,
                    kvbm_metrics,
                    context,
                ) => result,
            };

            match result {
                Ok(prefetched_blocks) => {
                    tracing::debug!(
                        request_id,
                        prefetched_blocks,
                        "completed advisory remote prefetch"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        request_id,
                        error = %error,
                        "advisory remote prefetch failed"
                    );
                }
            }

            if let Ok(mut guard) = inflight.lock() {
                guard.remove(&request_id);
            }
            if let Ok(mut guard) = cancellations.lock() {
                guard.remove(&request_id);
            }

            let _ = done_tx.send(());
        });

        tokio::task::block_in_place(|| {
            get_current_tokio_handle().block_on(async move {
                let _ = tokio::time::timeout(Duration::from_millis(transfer_budget_ms), done_rx)
                    .await;
            });
        });

        Ok(())
    }
}

#[pyclass]
pub struct PyTrtllmKvConnectorLeader {
    connector_leader: Box<dyn Leader>,
}

#[pymethods]
impl PyTrtllmKvConnectorLeader {
    #[new]
    #[pyo3(signature = (rank, device_id, drt, page_size, leader, consolidator_trtllm_endpoint=None, consolidator_output_endpoint=None))]
    pub fn new(
        rank: u64,
        device_id: u64,
        drt: Option<PyObject>,
        page_size: usize,
        leader: PyKvbmLeader,
        consolidator_trtllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
    ) -> PyResult<Self> {
        let drt: Option<Arc<DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = drt {
                crate::extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let connector_leader: Box<dyn Leader> = Box::new(KvConnectorLeader::new(
            rank,
            device_id,
            drt,
            page_size,
            leader,
            consolidator_trtllm_endpoint,
            consolidator_output_endpoint,
        ));
        Ok(Self { connector_leader })
    }

    fn get_num_new_matched_tokens(
        &mut self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(usize, bool)> {
        self.connector_leader
            .get_num_new_matched_tokens(request_id, request_num_tokens, num_computed_tokens)
            .map_err(to_pyerr)
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        context_current_position: usize,
    ) -> PyResult<()> {
        self.connector_leader
            .update_state_after_alloc(request_id, block_ids, context_current_position)
            .map_err(to_pyerr)
    }

    fn build_connector_metadata(&mut self, scheduler_output: SchedulerOutput) -> PyResult<Vec<u8>> {
        self.connector_leader
            .build_connector_metadata(scheduler_output)
            .map_err(to_pyerr)
    }

    fn request_finished(&mut self, request_id: &str, block_ids: Vec<BlockId>) -> PyResult<bool> {
        self.connector_leader
            .request_finished(request_id.to_string(), block_ids)
            .map_err(to_pyerr)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.connector_leader.has_slot(request_id.to_string())
    }

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.connector_leader
            .create_slot(request, tokens)
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (request, tokens, transfer_budget_ms=100, min_blocks=10))]
    fn advise_async_loading(
        &mut self,
        request: KvbmRequest,
        tokens: Vec<u32>,
        transfer_budget_ms: u64,
        min_blocks: u64,
    ) -> PyResult<()> {
        self.connector_leader
            .advise_async_loading(request, tokens, transfer_budget_ms, min_blocks)
            .map_err(to_pyerr)
    }
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

    async fn build_test_block_manager(
    ) -> anyhow::Result<
        dynamo_llm::block_manager::KvBlockManager<locality::Local, BasicMetadata>,
    > {
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
        let mut hit_map = HashMap::<SequenceHash, G3pbQueryHit>::new();

        for index in [0usize, 1, 3, 4] {
            let block = &sequence_blocks[index];
            hit_map.insert(
                block.sequence_hash(),
                G3pbQueryHit {
                    instance_id: 11,
                    sequence_hash: block.sequence_hash(),
                    size_bytes: 1024,
                    checksum: None,
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
            .expect("missing hash")
            ;
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
