// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use dynamo_llm::block_manager::block::nixl::{BlockDescriptorList, BlockMutability};
use dynamo_llm::block_manager::block::{BasicMetadata, BlockDataExt, BlockDataProvider};
use dynamo_llm::block_manager::block::{MutableBlock, locality::Local};
use dynamo_llm::block_manager::config::{
    KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig, KvManagerRuntimeConfig,
};
use dynamo_llm::block_manager::distributed::{
    G2PB_COMPONENT_NAME, G2PB_ENDPOINT_NAME, G2PB_NAMESPACE, G2pbCacheStorage, G2pbError,
    G2pbFetchBlocksResponse, G2pbHealthResponse, G2pbPutBlock, G2pbQueryHit, G2pbRpcRequest,
    G2pbRpcResponse, G2pbStageBlocksResponse, G2pbStorageAgent, G2pbStorageConfig,
};
use dynamo_llm::block_manager::locality::Local as LocalityLocal;
use dynamo_llm::block_manager::storage::{PinnedAllocator, PinnedStorage, nixl::NixlAgent};
use dynamo_llm::block_manager::{CancellationToken, KvBlockManager};
use dynamo_runtime::{
    DistributedRuntime, Runtime, Worker, logging,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait, network::Ingress,
    },
    protocols::annotated::Annotated,
    stream,
};
use serde_json::json;

fn make_descriptor_list(
    instance_id: u64,
    block_set_idx: usize,
    mutability: BlockMutability,
    block_indices: Vec<usize>,
) -> Result<BlockDescriptorList> {
    anyhow::ensure!(
        !block_indices.is_empty(),
        "block descriptor list cannot be empty"
    );

    // BlockDescriptorList still uses the lower-level transfer schema field
    // name `worker_id`; for G2PB remote peers we populate it with the
    // discovery/routing `instance_id`.
    Ok(serde_json::from_value(json!({
        "worker_id": instance_id,
        "block_set_idx": block_set_idx,
        "mutability": mutability,
        "block_indices": block_indices,
    }))?)
}

#[derive(Clone, Debug)]
struct Args {
    device_id: usize,
    host_blocks: usize,
    page_size: usize,
    inner_dim: usize,
    num_layers: usize,
    outer_dim: usize,
    dtype_width_bytes: usize,
    g2_bytes: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            device_id: 0,
            host_blocks: 64,
            page_size: 32,
            inner_dim: 128,
            num_layers: 1,
            outer_dim: 1,
            dtype_width_bytes: 2,
            g2_bytes: G2pbStorageConfig::DEFAULT_G2_CAPACITY_BYTES,
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self::default();
        let mut it = std::env::args().skip(1);

        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--device-id" => {
                    args.device_id = it
                        .next()
                        .context("missing value for --device-id")?
                        .parse()?
                }
                "--host-blocks" => {
                    args.host_blocks = it
                        .next()
                        .context("missing value for --host-blocks")?
                        .parse()?
                }
                "--page-size" => {
                    args.page_size = it
                        .next()
                        .context("missing value for --page-size")?
                        .parse()?
                }
                "--inner-dim" => {
                    args.inner_dim = it
                        .next()
                        .context("missing value for --inner-dim")?
                        .parse()?
                }
                "--num-layers" => {
                    args.num_layers = it
                        .next()
                        .context("missing value for --num-layers")?
                        .parse()?
                }
                "--outer-dim" => {
                    args.outer_dim = it
                        .next()
                        .context("missing value for --outer-dim")?
                        .parse()?
                }
                "--dtype-width-bytes" => {
                    args.dtype_width_bytes = it
                        .next()
                        .context("missing value for --dtype-width-bytes")?
                        .parse()?
                }
                "--g2-bytes" => {
                    args.g2_bytes = it.next().context("missing value for --g2-bytes")?.parse()?
                }
                "--help" | "-h" => {
                    println!(
                        "kvbm_g2pb_service
  --device-id <id>                CUDA device id for pinned host registration (default 0)
  --host-blocks <n>               pinned host staging capacity (default 64)
  --page-size <n>                 KVBM page size (default 32)
  --inner-dim <n>                 KVBM inner dim (default 128)
  --num-layers <n>                KVBM num layers (default 1)
  --outer-dim <n>                 KVBM outer dim (default 1)
  --dtype-width-bytes <bytes>     KVBM dtype width bytes (default 2)
  --g2-bytes <bytes>              remote host-memory G2 capacity"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown flag: {other}"),
            }
        }

        Ok(args)
    }
}

type HostBlockManager = KvBlockManager<LocalityLocal, BasicMetadata>;

fn read_host_block_payload(
    block: &impl BlockDataProvider<StorageType = PinnedStorage>,
) -> Result<Vec<u8>> {
    let block_data = block.block_data();
    let mut payload = Vec::new();

    for layer_idx in 0..block_data.num_layers() {
        for outer_idx in 0..block_data.num_outer_dims() {
            let layer_view = block_data.layer_view(layer_idx, outer_idx)?;
            unsafe {
                payload.extend_from_slice(std::slice::from_raw_parts(
                    layer_view.as_ptr(),
                    layer_view.size(),
                ));
            }
        }
    }

    Ok(payload)
}

struct StagedBlock {
    meta: G2pbPutBlock,
    block_id: usize,
    block: MutableBlock<PinnedStorage, Local, BasicMetadata>,
}

struct CommittedBlock {
    staged: StagedBlock,
    last_access_tick: u64,
}

#[derive(Default)]
struct G2pbPeerRuntimeState {
    reserved: HashSet<u64>,
    staged: HashMap<u64, StagedBlock>,
    committed: HashMap<u64, CommittedBlock>,
    access_clock: u64,
}

struct G2pbPeerRuntime {
    instance_id: u64,
    block_manager: HostBlockManager,
    blockset: dynamo_llm::block_manager::block::nixl::SerializedNixlBlockSet,
    state: RwLock<G2pbPeerRuntimeState>,
}

impl G2pbPeerRuntime {
    fn next_access_tick(state: &mut G2pbPeerRuntimeState) -> u64 {
        let tick = state.access_clock;
        state.access_clock += 1;
        tick
    }

    async fn new(args: &Args, instance_id: u64) -> Result<Self> {
        let agent = build_agent(instance_id)?;
        let cancel_token = CancellationToken::new();
        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(instance_id)
                    .cancellation_token(cancel_token)
                    .use_nixl_agent(agent)
                    .build()?,
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(args.num_layers)
                    .outer_dim(args.outer_dim)
                    .page_size(args.page_size)
                    .inner_dim(args.inner_dim)
                    .dtype_width_bytes(args.dtype_width_bytes)
                    .build()?,
            )
            .host_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(args.host_blocks)
                    .allocator(PinnedAllocator::new(args.device_id)?)
                    .build()?,
            )
            .build()?;

        let block_manager = HostBlockManager::new(config).await?;
        let blockset = block_manager.export_local_blockset()?;

        Ok(Self {
            instance_id,
            block_manager,
            blockset,
            state: RwLock::new(G2pbPeerRuntimeState::default()),
        })
    }

    async fn offer_blocks(&self, agent: &G2pbStorageAgent, blocks: &[G2pbPutBlock]) -> Vec<u64> {
        let offerable: Vec<_> = {
            let guard = self.state.read().expect("g2pb runtime state poisoned");
            blocks
                .iter()
                .filter(|block| {
                    !guard.reserved.contains(&block.sequence_hash)
                        && !guard.staged.contains_key(&block.sequence_hash)
                        && !guard.committed.contains_key(&block.sequence_hash)
                })
                .cloned()
                .collect()
        };

        agent.offer_blocks(&offerable).await
    }

    async fn stage_put_blocks(
        &self,
        agent: &G2pbStorageAgent,
        blocks: Vec<G2pbPutBlock>,
    ) -> Result<G2pbStageBlocksResponse> {
        {
            let state = self.state.read().expect("g2pb runtime state poisoned");
            for meta in &blocks {
                anyhow::ensure!(
                    !state.reserved.contains(&meta.sequence_hash)
                        && !state.staged.contains_key(&meta.sequence_hash)
                        && !state.committed.contains_key(&meta.sequence_hash),
                    "sequence hash {} is already staged or committed",
                    meta.sequence_hash
                );
            }
        }

        self.ensure_staging_capacity(agent, blocks.len()).await?;
        let host_pool = self
            .block_manager
            .host()
            .context("service runtime has no host staging pool")?;
        let mutable_blocks = host_pool.allocate_blocks(blocks.len()).await?;
        let block_set_idx = mutable_blocks
            .first()
            .map(|block| block.block_data().block_set_id())
            .context("no host blocks allocated for staging")?;
        let block_ids = mutable_blocks
            .iter()
            .map(|block| block.block_id())
            .collect::<Vec<_>>();

        let mut state = self.state.write().expect("g2pb runtime state poisoned");
        for (meta, block) in blocks.into_iter().zip(mutable_blocks.into_iter()) {
            state.reserved.insert(meta.sequence_hash);
            state.staged.insert(
                meta.sequence_hash,
                StagedBlock {
                    block_id: block.block_id(),
                    meta,
                    block,
                },
            );
        }
        drop(state);

        Ok(G2pbStageBlocksResponse {
            instance_id: self.instance_id,
            blockset: self.blockset.clone(),
            descriptors: make_descriptor_list(
                self.instance_id,
                block_set_idx,
                BlockMutability::Mutable,
                block_ids,
            )?,
        })
    }

    async fn commit_staged_blocks(
        &self,
        agent: &G2pbStorageAgent,
        sequence_hashes: &[u64],
    ) -> Result<(), G2pbError> {
        let mut committed_meta = Vec::with_capacity(sequence_hashes.len());
        {
            let mut state = self.state.write().expect("g2pb runtime state poisoned");
            for sequence_hash in sequence_hashes {
                let staged = state
                    .staged
                    .remove(sequence_hash)
                    .ok_or(G2pbError::NotFound {
                        instance_id: self.instance_id,
                        sequence_hashes: vec![*sequence_hash],
                    })?;
                state.reserved.remove(sequence_hash);
                let payload =
                    read_host_block_payload(&staged.block).map_err(|_| G2pbError::NotFound {
                        instance_id: self.instance_id,
                        sequence_hashes: vec![*sequence_hash],
                    })?;
                let actual_checksum =
                    dynamo_llm::block_manager::distributed::G2pbTransferBlock::compute_checksum(
                        &payload,
                    );
                if staged.meta.checksum != actual_checksum {
                    return Err(G2pbError::NotFound {
                        instance_id: self.instance_id,
                        sequence_hashes: vec![*sequence_hash],
                    });
                }
                let committed = staged.meta.clone();
                committed_meta.push(committed);
                let tick = Self::next_access_tick(&mut state);
                state.committed.insert(
                    *sequence_hash,
                    CommittedBlock {
                        staged: StagedBlock {
                            meta: committed_meta
                                .last()
                                .cloned()
                                .expect("committed metadata missing"),
                            ..staged
                        },
                        last_access_tick: tick,
                    },
                );
            }
        }

        agent.put_blocks(committed_meta).await;
        Ok(())
    }

    async fn query_blocks(
        &self,
        agent: &G2pbStorageAgent,
        sequence_hashes: &[u64],
    ) -> Vec<G2pbQueryHit> {
        let mut hits = Vec::new();
        let mut missing = Vec::new();
        {
            let mut state = self.state.write().expect("g2pb runtime state poisoned");
            for sequence_hash in sequence_hashes {
                let tick = Self::next_access_tick(&mut state);
                if let Some(block) = state.committed.get_mut(sequence_hash) {
                    block.last_access_tick = tick;
                    hits.push(G2pbQueryHit {
                        instance_id: self.instance_id,
                        sequence_hash: *sequence_hash,
                        size_bytes: block.staged.meta.size_bytes,
                        checksum: block.staged.meta.checksum,
                    });
                } else {
                    missing.push(*sequence_hash);
                }
            }
        }

        if !missing.is_empty() {
            hits.extend(agent.query_blocks(&missing).await);
        }

        hits
    }

    fn load_remote_blockset(
        &self,
        blockset: dynamo_llm::block_manager::block::nixl::SerializedNixlBlockSet,
    ) -> Result<()> {
        self.block_manager.import_remote_blockset(blockset)
    }

    fn fetch_descriptors(
        &self,
        sequence_hashes: &[u64],
    ) -> Result<G2pbFetchBlocksResponse, G2pbError> {
        let mut state = self.state.write().expect("g2pb runtime state poisoned");
        let mut block_ids = Vec::with_capacity(sequence_hashes.len());
        let mut block_set_idx = None;

        for sequence_hash in sequence_hashes {
            let tick = Self::next_access_tick(&mut state);
            let Some(block) = state.committed.get_mut(sequence_hash) else {
                return Err(G2pbError::NotFound {
                    instance_id: self.instance_id,
                    sequence_hashes: vec![*sequence_hash],
                });
            };

            block.last_access_tick = tick;
            block_ids.push(block.staged.block_id);
            block_set_idx.get_or_insert(block.staged.block.block_data().block_set_id());
        }

        let block_set_idx = block_set_idx.ok_or(G2pbError::NotFound {
            instance_id: self.instance_id,
            sequence_hashes: sequence_hashes.to_vec(),
        })?;
        Ok(G2pbFetchBlocksResponse {
            instance_id: self.instance_id,
            blockset: self.blockset.clone(),
            descriptors: make_descriptor_list(
                self.instance_id,
                block_set_idx,
                BlockMutability::Immutable,
                block_ids,
            )
            .map_err(|_| G2pbError::NotFound {
                instance_id: self.instance_id,
                sequence_hashes: sequence_hashes.to_vec(),
            })?,
        })
    }

    async fn ensure_staging_capacity(
        &self,
        agent: &G2pbStorageAgent,
        required_blocks: usize,
    ) -> Result<()> {
        let host_pool = self
            .block_manager
            .host()
            .context("service runtime has no host staging pool")?;
        let available_blocks = host_pool.available_blocks() as usize;
        if available_blocks >= required_blocks {
            return Ok(());
        }

        let blocks_to_reclaim = required_blocks - available_blocks;
        self.evict_committed_blocks(agent, blocks_to_reclaim)
            .await?;
        Ok(())
    }

    async fn evict_committed_blocks(&self, agent: &G2pbStorageAgent, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let (evicted_hashes, evicted_blocks) = {
            let mut state = self.state.write().expect("g2pb runtime state poisoned");
            let mut candidates: Vec<_> = state
                .committed
                .iter()
                .map(|(sequence_hash, block)| (*sequence_hash, block.last_access_tick))
                .collect();
            candidates.sort_by_key(|(_, last_access_tick)| *last_access_tick);

            let mut evicted_hashes = Vec::new();
            let mut evicted_blocks = Vec::new();

            for (sequence_hash, _) in candidates.into_iter().take(count) {
                if let Some(block) = state.committed.remove(&sequence_hash) {
                    evicted_hashes.push(sequence_hash);
                    evicted_blocks.push(block);
                }
            }

            (evicted_hashes, evicted_blocks)
        };

        if evicted_hashes.len() < count {
            anyhow::bail!(
                "unable to reclaim enough committed blocks, requested {}, reclaimed {}",
                count,
                evicted_hashes.len()
            );
        }

        let host_pool = self
            .block_manager
            .host()
            .context("service runtime has no host staging pool")?;
        let target_available_blocks = host_pool.available_blocks() as usize + evicted_hashes.len();
        drop(evicted_blocks);

        for _ in 0..100 {
            if host_pool.available_blocks() as usize >= target_available_blocks {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        anyhow::ensure!(
            host_pool.available_blocks() as usize >= target_available_blocks,
            "committed-block reclamation did not free enough host blocks: target {}, actual {}",
            target_available_blocks,
            host_pool.available_blocks()
        );

        agent.delete_blocks(&evicted_hashes).await?;
        Ok(())
    }
}

#[derive(Clone)]
struct G2pbBackendService {
    endpoint_id: String,
    agent: Arc<G2pbStorageAgent>,
    runtime: Arc<G2pbPeerRuntime>,
}

#[derive(Clone)]
struct G2pbEndpointHandler {
    service: Arc<G2pbBackendService>,
}

impl G2pbBackendService {
    async fn new(args: &Args, instance_id: u64) -> Result<Self> {
        let mut config = G2pbStorageConfig::new(args.device_id);
        config.g2_capacity_bytes = args.g2_bytes;
        let storage = Arc::new(G2pbCacheStorage::new(config).await?);
        let agent = Arc::new(G2pbStorageAgent::new_with_storage(instance_id, storage));
        let runtime = Arc::new(G2pbPeerRuntime::new(args, instance_id).await?);

        Ok(Self {
            endpoint_id: format!("{G2PB_NAMESPACE}/{G2PB_COMPONENT_NAME}/{G2PB_ENDPOINT_NAME}"),
            agent,
            runtime,
        })
    }

    fn instance_id(&self) -> u64 {
        self.agent.instance_id()
    }

    async fn handle_rpc(&self, request: G2pbRpcRequest) -> Result<G2pbRpcResponse> {
        match request {
            G2pbRpcRequest::Health => Ok(G2pbRpcResponse::Health(G2pbHealthResponse {
                instance_id: self.agent.instance_id(),
                listen: self.endpoint_id.clone(),
                hostname: std::env::var("HOSTNAME").unwrap_or_default(),
            })),
            G2pbRpcRequest::PutBlocks(blocks) => {
                self.agent.put_blocks(blocks).await;
                Ok(G2pbRpcResponse::Ack)
            }
            G2pbRpcRequest::Offer(request) => {
                let accepted = self
                    .runtime
                    .offer_blocks(&self.agent, &request.blocks)
                    .await;
                Ok(G2pbRpcResponse::Offer(
                    dynamo_llm::block_manager::distributed::G2pbOfferResponse { accepted },
                ))
            }
            G2pbRpcRequest::PutPayload(request) => Ok(G2pbRpcResponse::PutPayload(
                self.agent
                    .offer_and_put_payload_blocks(request.blocks)
                    .await?,
            )),
            G2pbRpcRequest::Query(request) => Ok(G2pbRpcResponse::Query(
                self.runtime
                    .query_blocks(&self.agent, &request.sequence_hashes)
                    .await,
            )),
            G2pbRpcRequest::Fetch(request) => Ok(G2pbRpcResponse::Fetch(
                self.runtime.fetch_descriptors(&request.sequence_hashes)?,
            )),
            G2pbRpcRequest::StagePut(request) => Ok(G2pbRpcResponse::StagePut(
                self.runtime
                    .stage_put_blocks(&self.agent, request.blocks)
                    .await?,
            )),
            G2pbRpcRequest::CommitPut(request) => {
                self.runtime
                    .commit_staged_blocks(&self.agent, &request.sequence_hashes)
                    .await?;
                Ok(G2pbRpcResponse::Ack)
            }
            G2pbRpcRequest::LoadRemote(request) => {
                self.runtime.load_remote_blockset(request.blockset)?;
                Ok(G2pbRpcResponse::Ack)
            }
        }
    }

    async fn serve(self: Arc<Self>, distributed: &DistributedRuntime) -> Result<()> {
        let ingress = Ingress::for_engine(G2pbEndpointHandler::new(self.clone()))?;
        let component = distributed
            .namespace(G2PB_NAMESPACE)?
            .component(G2PB_COMPONENT_NAME)?;

        component
            .endpoint(G2PB_ENDPOINT_NAME)
            .endpoint_builder()
            .handler(ingress)
            .graceful_shutdown(true)
            .start()
            .await
    }
}

impl G2pbEndpointHandler {
    fn new(service: Arc<G2pbBackendService>) -> Arc<Self> {
        Arc::new(Self { service })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<G2pbRpcRequest>, ManyOut<Annotated<G2pbRpcResponse>>, Error>
    for G2pbEndpointHandler
{
    async fn generate(
        &self,
        input: SingleIn<G2pbRpcRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<G2pbRpcResponse>>> {
        let (request, ctx) = input.into_parts();
        let response = match self.service.handle_rpc(request).await {
            Ok(response) => Annotated::from_data(response),
            Err(err) => Annotated::from_error(err.to_string()),
        };
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { response })),
            ctx.context(),
        ))
    }
}

fn build_agent(instance_id: u64) -> Result<NixlAgent> {
    let agent = NixlAgent::new(&instance_id.to_string())?;
    let (_, ucx_params) = agent.get_plugin_params("UCX")?;
    agent.create_backend("UCX", &ucx_params)?;
    let (_, posix_params) = agent.get_plugin_params("POSIX")?;
    agent.create_backend("POSIX", &posix_params)?;
    Ok(agent)
}

async fn app(runtime: Runtime) -> Result<()> {
    let args = Args::parse()?;
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    let instance_id = distributed.connection_id();
    let service = Arc::new(G2pbBackendService::new(&args, instance_id).await?);

    println!(
        "kvbm_g2pb_service registering instance_id={} on {G2PB_NAMESPACE}/{G2PB_COMPONENT_NAME}/{G2PB_ENDPOINT_NAME}",
        service.instance_id()
    );
    service.serve(&distributed).await
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}
