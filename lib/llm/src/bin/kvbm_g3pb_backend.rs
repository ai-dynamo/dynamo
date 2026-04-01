// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use dynamo_llm::block_manager::block::nixl::{BlockDescriptorList, BlockMutability};
use dynamo_llm::block_manager::block::{BasicMetadata, BlockDataExt, BlockDataProvider};
use dynamo_llm::block_manager::block::{MutableBlock, locality::Local};
use dynamo_llm::block_manager::config::{
    KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig, KvManagerRuntimeConfig,
};
use dynamo_llm::block_manager::distributed::{
    G3PB_COMPONENT_NAME, G3PB_ENDPOINT_NAME, G3PB_NAMESPACE, G3pbCacheStorage, G3pbError,
    G3pbFetchBlocksResponse, G3pbHealthResponse, G3pbPutBlock, G3pbQueryHit, G3pbRpcRequest,
    G3pbRpcResponse, G3pbStageBlocksResponse, G3pbStorageAgent, G3pbStorageConfig,
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
    worker_id: u64,
    block_set_idx: usize,
    mutability: BlockMutability,
    block_indices: Vec<usize>,
) -> Result<BlockDescriptorList> {
    anyhow::ensure!(
        !block_indices.is_empty(),
        "block descriptor list cannot be empty"
    );

    Ok(serde_json::from_value(json!({
        "worker_id": worker_id,
        "block_set_idx": block_set_idx,
        "mutability": mutability,
        "block_indices": block_indices,
    }))?)
}

#[derive(Clone, Debug)]
struct Args {
    worker_id: u64,
    device_id: usize,
    host_blocks: usize,
    page_size: usize,
    inner_dim: usize,
    num_layers: usize,
    outer_dim: usize,
    dtype_width_bytes: usize,
    foyer_dirs: Vec<PathBuf>,
    g2_bytes: usize,
    foyer_memory_bytes: usize,
    foyer_disk_bytes: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            worker_id: 41,
            device_id: 0,
            host_blocks: 64,
            page_size: 32,
            inner_dim: 128,
            num_layers: 1,
            outer_dim: 1,
            dtype_width_bytes: 2,
            foyer_dirs: vec![PathBuf::from(G3pbStorageConfig::DEFAULT_FOYER_DIR)],
            g2_bytes: G3pbStorageConfig::DEFAULT_G2_CAPACITY_BYTES,
            foyer_memory_bytes: G3pbStorageConfig::DEFAULT_FOYER_MEMORY_CAPACITY_BYTES,
            foyer_disk_bytes: G3pbStorageConfig::DEFAULT_FOYER_DISK_CAPACITY_BYTES,
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self::default();
        let mut it = std::env::args().skip(1);

        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--worker-id" => {
                    args.worker_id = it
                        .next()
                        .context("missing value for --worker-id")?
                        .parse()?
                }
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
                "--foyer-dir" => {
                    if args.foyer_dirs.len() == 1
                        && args.foyer_dirs[0] == PathBuf::from(G3pbStorageConfig::DEFAULT_FOYER_DIR)
                    {
                        args.foyer_dirs.clear();
                    }
                    args.foyer_dirs
                        .push(it.next().context("missing value for --foyer-dir")?.into())
                }
                "--g2-bytes" => {
                    args.g2_bytes = it.next().context("missing value for --g2-bytes")?.parse()?
                }
                "--foyer-memory-bytes" => {
                    args.foyer_memory_bytes = it
                        .next()
                        .context("missing value for --foyer-memory-bytes")?
                        .parse()?
                }
                "--foyer-disk-bytes" => {
                    args.foyer_disk_bytes = it
                        .next()
                        .context("missing value for --foyer-disk-bytes")?
                        .parse()?
                }
                "--help" | "-h" => {
                    println!(
                        "kvbm_g3pb_backend
  --worker-id <id>                backend worker id (default 41)
  --device-id <id>                CUDA device id for pinned host registration (default 0)
  --host-blocks <n>               pinned host staging capacity (default 64)
  --page-size <n>                 KVBM page size (default 32)
  --inner-dim <n>                 KVBM inner dim (default 128)
  --num-layers <n>                KVBM num layers (default 1)
  --outer-dim <n>                 KVBM outer dim (default 1)
  --dtype-width-bytes <bytes>     KVBM dtype width bytes (default 2)
  --foyer-dir <path>              foyer storage directory; repeat for multiple locations
  --g2-bytes <bytes>              pinned G2 staging capacity
  --foyer-memory-bytes <bytes>    foyer in-memory capacity
  --foyer-disk-bytes <bytes>      foyer disk capacity (split across dirs)"
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

struct StagedBlock {
    meta: G3pbPutBlock,
    block_id: usize,
    block: MutableBlock<PinnedStorage, Local, BasicMetadata>,
}

struct CommittedBlock {
    staged: StagedBlock,
    last_access_tick: u64,
}

#[derive(Default)]
struct G3pbPeerRuntimeState {
    reserved: HashSet<u64>,
    staged: HashMap<u64, StagedBlock>,
    committed: HashMap<u64, CommittedBlock>,
    access_clock: u64,
}

struct G3pbPeerRuntime {
    worker_id: u64,
    block_manager: HostBlockManager,
    blockset: dynamo_llm::block_manager::block::nixl::SerializedNixlBlockSet,
    state: RwLock<G3pbPeerRuntimeState>,
}

impl G3pbPeerRuntime {
    fn next_access_tick(state: &mut G3pbPeerRuntimeState) -> u64 {
        let tick = state.access_clock;
        state.access_clock += 1;
        tick
    }

    async fn new(args: &Args) -> Result<Self> {
        let agent = build_agent(args.worker_id)?;
        let cancel_token = CancellationToken::new();
        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(args.worker_id)
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
            worker_id: args.worker_id,
            block_manager,
            blockset,
            state: RwLock::new(G3pbPeerRuntimeState::default()),
        })
    }

    async fn offer_blocks(&self, agent: &G3pbStorageAgent, blocks: &[G3pbPutBlock]) -> Vec<u64> {
        let offerable: Vec<_> = {
            let guard = self.state.read().expect("g3pb runtime state poisoned");
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
        agent: &G3pbStorageAgent,
        blocks: Vec<G3pbPutBlock>,
    ) -> Result<G3pbStageBlocksResponse> {
        {
            let state = self.state.read().expect("g3pb runtime state poisoned");
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
            .context("backend runtime has no host staging pool")?;
        let mutable_blocks = host_pool.allocate_blocks(blocks.len()).await?;
        let block_set_idx = mutable_blocks
            .first()
            .map(|block| block.block_data().block_set_id())
            .context("no host blocks allocated for staging")?;
        let block_ids = mutable_blocks
            .iter()
            .map(|block| block.block_id())
            .collect::<Vec<_>>();

        let mut state = self.state.write().expect("g3pb runtime state poisoned");
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

        Ok(G3pbStageBlocksResponse {
            worker_id: self.worker_id,
            blockset: self.blockset.clone(),
            descriptors: make_descriptor_list(
                self.worker_id,
                block_set_idx,
                BlockMutability::Mutable,
                block_ids,
            )?,
        })
    }

    async fn commit_staged_blocks(
        &self,
        agent: &G3pbStorageAgent,
        sequence_hashes: &[u64],
    ) -> Result<()> {
        let mut committed_meta = Vec::with_capacity(sequence_hashes.len());
        {
            let mut state = self.state.write().expect("g3pb runtime state poisoned");
            for sequence_hash in sequence_hashes {
                let staged = state
                    .staged
                    .remove(sequence_hash)
                    .with_context(|| format!("sequence hash {sequence_hash} is not staged"))?;
                state.reserved.remove(sequence_hash);
                committed_meta.push(staged.meta.clone());
                let tick = Self::next_access_tick(&mut state);
                state.committed.insert(
                    *sequence_hash,
                    CommittedBlock {
                        staged,
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
        agent: &G3pbStorageAgent,
        sequence_hashes: &[u64],
    ) -> Vec<G3pbQueryHit> {
        let mut hits = Vec::new();
        let mut missing = Vec::new();
        {
            let mut state = self.state.write().expect("g3pb runtime state poisoned");
            for sequence_hash in sequence_hashes {
                let tick = Self::next_access_tick(&mut state);
                if let Some(block) = state.committed.get_mut(sequence_hash) {
                    block.last_access_tick = tick;
                    hits.push(G3pbQueryHit {
                        worker_id: self.worker_id,
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
    ) -> Result<G3pbFetchBlocksResponse, G3pbError> {
        let mut state = self.state.write().expect("g3pb runtime state poisoned");
        let mut block_ids = Vec::with_capacity(sequence_hashes.len());
        let mut block_set_idx = None;

        for sequence_hash in sequence_hashes {
            let tick = Self::next_access_tick(&mut state);
            let Some(block) = state.committed.get_mut(sequence_hash) else {
                return Err(G3pbError::NotFound {
                    worker_id: self.worker_id,
                    sequence_hashes: vec![*sequence_hash],
                });
            };

            block.last_access_tick = tick;
            block_ids.push(block.staged.block_id);
            block_set_idx.get_or_insert(block.staged.block.block_data().block_set_id());
        }

        let block_set_idx = block_set_idx.ok_or(G3pbError::NotFound {
            worker_id: self.worker_id,
            sequence_hashes: sequence_hashes.to_vec(),
        })?;
        Ok(G3pbFetchBlocksResponse {
            worker_id: self.worker_id,
            blockset: self.blockset.clone(),
            descriptors: make_descriptor_list(
                self.worker_id,
                block_set_idx,
                BlockMutability::Immutable,
                block_ids,
            )
            .map_err(|_| G3pbError::NotFound {
                worker_id: self.worker_id,
                sequence_hashes: sequence_hashes.to_vec(),
            })?,
        })
    }

    async fn ensure_staging_capacity(
        &self,
        agent: &G3pbStorageAgent,
        required_blocks: usize,
    ) -> Result<()> {
        let host_pool = self
            .block_manager
            .host()
            .context("backend runtime has no host staging pool")?;
        let available_blocks = host_pool.available_blocks() as usize;
        if available_blocks >= required_blocks {
            return Ok(());
        }

        let blocks_to_reclaim = required_blocks - available_blocks;
        self.evict_committed_blocks(agent, blocks_to_reclaim)
            .await?;
        Ok(())
    }

    async fn evict_committed_blocks(&self, agent: &G3pbStorageAgent, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let (evicted_hashes, evicted_blocks) = {
            let mut state = self.state.write().expect("g3pb runtime state poisoned");
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
            .context("backend runtime has no host staging pool")?;
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
struct G3pbBackendService {
    endpoint_id: String,
    agent: Arc<G3pbStorageAgent>,
    runtime: Arc<G3pbPeerRuntime>,
}

#[derive(Clone)]
struct G3pbEndpointHandler {
    service: Arc<G3pbBackendService>,
}

impl G3pbBackendService {
    async fn new(args: &Args) -> Result<Self> {
        let mut config = G3pbStorageConfig::new(args.foyer_dirs.clone(), args.device_id);
        config.g2_capacity_bytes = args.g2_bytes;
        config.foyer_memory_capacity_bytes = args.foyer_memory_bytes;
        config.foyer_disk_capacity_bytes = args.foyer_disk_bytes;
        let storage = Arc::new(G3pbCacheStorage::new(config).await?);
        let agent = Arc::new(G3pbStorageAgent::new_with_storage(args.worker_id, storage));
        let runtime = Arc::new(G3pbPeerRuntime::new(args).await?);

        Ok(Self {
            endpoint_id: format!("{G3PB_NAMESPACE}/{G3PB_COMPONENT_NAME}/{G3PB_ENDPOINT_NAME}"),
            agent,
            runtime,
        })
    }

    fn worker_id(&self) -> u64 {
        self.agent.worker_id()
    }

    async fn handle_rpc(&self, request: G3pbRpcRequest) -> Result<G3pbRpcResponse> {
        match request {
            G3pbRpcRequest::Health => Ok(G3pbRpcResponse::Health(G3pbHealthResponse {
                worker_id: self.agent.worker_id(),
                listen: self.endpoint_id.clone(),
            })),
            G3pbRpcRequest::PutBlocks(blocks) => {
                self.agent.put_blocks(blocks).await;
                Ok(G3pbRpcResponse::Ack)
            }
            G3pbRpcRequest::Offer(request) => {
                let accepted = self
                    .runtime
                    .offer_blocks(&self.agent, &request.blocks)
                    .await;
                Ok(G3pbRpcResponse::Offer(
                    dynamo_llm::block_manager::distributed::G3pbOfferResponse { accepted },
                ))
            }
            G3pbRpcRequest::PutPayload(request) => Ok(G3pbRpcResponse::PutPayload(
                self.agent
                    .offer_and_put_payload_blocks(request.blocks)
                    .await?,
            )),
            G3pbRpcRequest::Query(request) => Ok(G3pbRpcResponse::Query(
                self.runtime
                    .query_blocks(&self.agent, &request.sequence_hashes)
                    .await,
            )),
            G3pbRpcRequest::Fetch(request) => Ok(G3pbRpcResponse::Fetch(
                self.runtime.fetch_descriptors(&request.sequence_hashes)?,
            )),
            G3pbRpcRequest::StagePut(request) => Ok(G3pbRpcResponse::StagePut(
                self.runtime
                    .stage_put_blocks(&self.agent, request.blocks)
                    .await?,
            )),
            G3pbRpcRequest::CommitPut(request) => {
                self.runtime
                    .commit_staged_blocks(&self.agent, &request.sequence_hashes)
                    .await?;
                Ok(G3pbRpcResponse::Ack)
            }
            G3pbRpcRequest::LoadRemote(request) => {
                self.runtime.load_remote_blockset(request.blockset)?;
                Ok(G3pbRpcResponse::Ack)
            }
        }
    }

    async fn serve(self: Arc<Self>, distributed: &DistributedRuntime) -> Result<()> {
        let ingress = Ingress::for_engine(G3pbEndpointHandler::new(self.clone()))?;
        let component = distributed
            .namespace(G3PB_NAMESPACE)?
            .component(G3PB_COMPONENT_NAME)?;

        component
            .endpoint(G3PB_ENDPOINT_NAME)
            .endpoint_builder()
            .handler(ingress)
            .graceful_shutdown(true)
            .start()
            .await
    }
}

impl G3pbEndpointHandler {
    fn new(service: Arc<G3pbBackendService>) -> Arc<Self> {
        Arc::new(Self { service })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<G3pbRpcRequest>, ManyOut<Annotated<G3pbRpcResponse>>, Error>
    for G3pbEndpointHandler
{
    async fn generate(
        &self,
        input: SingleIn<G3pbRpcRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<G3pbRpcResponse>>> {
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

fn build_agent(worker_id: u64) -> Result<NixlAgent> {
    let agent = NixlAgent::new(&worker_id.to_string())?;
    let (_, ucx_params) = agent.get_plugin_params("UCX")?;
    agent.create_backend("UCX", &ucx_params)?;
    let (_, posix_params) = agent.get_plugin_params("POSIX")?;
    agent.create_backend("POSIX", &posix_params)?;
    Ok(agent)
}

async fn app(runtime: Runtime) -> Result<()> {
    let args = Args::parse()?;
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    let service = Arc::new(G3pbBackendService::new(&args).await?);

    println!(
        "kvbm_g3pb_backend registering worker_id={} on {G3PB_NAMESPACE}/{G3PB_COMPONENT_NAME}/{G3PB_ENDPOINT_NAME}",
        service.worker_id()
    );
    service.serve(&distributed).await
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}
