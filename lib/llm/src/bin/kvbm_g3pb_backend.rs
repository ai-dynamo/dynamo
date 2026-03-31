// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use dynamo_llm::block_manager::block::nixl::{BlockDescriptorList, BlockMutability};
use dynamo_llm::block_manager::block::{BasicMetadata, BlockDataExt, BlockDataProvider};
use dynamo_llm::block_manager::block::{MutableBlock, locality::Local};
use dynamo_llm::block_manager::config::{
    KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig, KvManagerRuntimeConfig,
};
use dynamo_llm::block_manager::distributed::{
    FoyerG3pbPeerStorage, G3pbCommitRequest, G3pbError, G3pbFetchBlocksResponse,
    G3pbFetchRequest, G3pbFoyerStorageConfig, G3pbHealthResponse, G3pbOfferRequest,
    G3pbOfferResponse, G3pbPutBlock, G3pbPutPayloadRequest, G3pbQueryHit, G3pbQueryRequest,
    G3pbStageBlocksRequest, G3pbStageBlocksResponse, G3pbStorageAgent,
};
use dynamo_llm::block_manager::locality::Local as LocalityLocal;
use dynamo_llm::block_manager::storage::{PinnedAllocator, PinnedStorage, nixl::NixlAgent};
use dynamo_llm::block_manager::{CancellationToken, KvBlockManager};

#[derive(Clone, Debug)]
struct Args {
    listen: SocketAddr,
    worker_id: u64,
    device_id: usize,
    host_blocks: usize,
    page_size: usize,
    inner_dim: usize,
    num_layers: usize,
    outer_dim: usize,
    dtype_width_bytes: usize,
    foyer_dir: Option<PathBuf>,
    foyer_memory_bytes: usize,
    foyer_disk_bytes: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:58080".parse().unwrap(),
            worker_id: 41,
            device_id: 0,
            host_blocks: 64,
            page_size: 32,
            inner_dim: 128,
            num_layers: 1,
            outer_dim: 1,
            dtype_width_bytes: 2,
            foyer_dir: None,
            foyer_memory_bytes: G3pbFoyerStorageConfig::DEFAULT_MEMORY_CAPACITY_BYTES,
            foyer_disk_bytes: G3pbFoyerStorageConfig::DEFAULT_DISK_CAPACITY_BYTES,
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self::default();
        let mut it = std::env::args().skip(1);

        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--listen" => {
                    args.listen = it.next().context("missing value for --listen")?.parse()?
                }
                "--worker-id" => {
                    args.worker_id = it
                        .next()
                        .context("missing value for --worker-id")?
                        .parse()?
                }
                "--device-id" => {
                    args.device_id = it.next().context("missing value for --device-id")?.parse()?
                }
                "--host-blocks" => {
                    args.host_blocks =
                        it.next().context("missing value for --host-blocks")?.parse()?
                }
                "--page-size" => {
                    args.page_size = it.next().context("missing value for --page-size")?.parse()?
                }
                "--inner-dim" => {
                    args.inner_dim = it.next().context("missing value for --inner-dim")?.parse()?
                }
                "--num-layers" => {
                    args.num_layers =
                        it.next().context("missing value for --num-layers")?.parse()?
                }
                "--outer-dim" => {
                    args.outer_dim = it.next().context("missing value for --outer-dim")?.parse()?
                }
                "--dtype-width-bytes" => {
                    args.dtype_width_bytes = it
                        .next()
                        .context("missing value for --dtype-width-bytes")?
                        .parse()?
                }
                "--foyer-dir" => {
                    args.foyer_dir =
                        Some(it.next().context("missing value for --foyer-dir")?.into())
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
  --listen <addr>                 HTTP listen address (default 127.0.0.1:58080)
  --worker-id <id>                backend worker id (default 41)
  --device-id <id>                CUDA device id for pinned host registration (default 0)
  --host-blocks <n>               pinned host staging capacity (default 64)
  --page-size <n>                 KVBM page size (default 32)
  --inner-dim <n>                 KVBM inner dim (default 128)
  --num-layers <n>                KVBM num layers (default 1)
  --outer-dim <n>                 KVBM outer dim (default 1)
  --dtype-width-bytes <bytes>     KVBM dtype width bytes (default 2)
  --foyer-dir <path>              enable foyer-backed peer storage at this path
  --foyer-memory-bytes <bytes>    foyer memory cache capacity
  --foyer-disk-bytes <bytes>      foyer disk cache capacity"
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

#[derive(Default)]
struct G3pbPeerRuntimeState {
    reserved: HashSet<u64>,
    staged: HashMap<u64, StagedBlock>,
    committed: HashMap<u64, StagedBlock>,
}

struct G3pbPeerRuntime {
    worker_id: u64,
    block_manager: HostBlockManager,
    blockset: dynamo_llm::block_manager::block::nixl::SerializedNixlBlockSet,
    state: RwLock<G3pbPeerRuntimeState>,
}

impl G3pbPeerRuntime {
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

    async fn offer_blocks(
        &self,
        agent: &G3pbStorageAgent,
        blocks: &[G3pbPutBlock],
    ) -> Vec<u64> {
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

    async fn stage_put_blocks(&self, blocks: Vec<G3pbPutBlock>) -> Result<G3pbStageBlocksResponse> {
        let host_pool = self
            .block_manager
            .host()
            .context("backend runtime has no host staging pool")?;
        let mutable_blocks = host_pool.allocate_blocks(blocks.len()).await?;

        let block_set_idx = mutable_blocks
            .first()
            .map(|block| block.block_data().block_set_id())
            .context("no host blocks allocated for staging")?;
        let block_ids = mutable_blocks.iter().map(|block| block.block_id()).collect::<Vec<_>>();

        let mut state = self.state.write().expect("g3pb runtime state poisoned");
        for meta in &blocks {
            anyhow::ensure!(
                !state.reserved.contains(&meta.sequence_hash)
                    && !state.staged.contains_key(&meta.sequence_hash)
                    && !state.committed.contains_key(&meta.sequence_hash),
                "sequence hash {} is already staged or committed",
                meta.sequence_hash
            );
        }

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
            descriptors: BlockDescriptorList::new(
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
                state.committed.insert(*sequence_hash, staged);
            }
        }

        agent.put_blocks(committed_meta).await;
        Ok(())
    }

    fn fetch_descriptors(&self, sequence_hashes: &[u64]) -> Result<G3pbFetchBlocksResponse, G3pbError> {
        let state = self.state.read().expect("g3pb runtime state poisoned");
        let mut block_ids = Vec::with_capacity(sequence_hashes.len());
        let mut block_set_idx = None;

        for sequence_hash in sequence_hashes {
            let Some(block) = state.committed.get(sequence_hash) else {
                return Err(G3pbError::NotFound {
                    worker_id: self.worker_id,
                    sequence_hashes: vec![*sequence_hash],
                });
            };

            block_ids.push(block.block_id);
            block_set_idx.get_or_insert(block.block.block_data().block_set_id());
        }

        let block_set_idx = block_set_idx.ok_or(G3pbError::NotFound {
            worker_id: self.worker_id,
            sequence_hashes: sequence_hashes.to_vec(),
        })?;
        Ok(G3pbFetchBlocksResponse {
            worker_id: self.worker_id,
            blockset: self.blockset.clone(),
            descriptors: BlockDescriptorList::new(
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
}

#[derive(Clone)]
struct AppState {
    agent: Arc<G3pbStorageAgent>,
    runtime: Arc<G3pbPeerRuntime>,
    listen: SocketAddr,
}

async fn health(State(state): State<AppState>) -> Json<G3pbHealthResponse> {
    Json(G3pbHealthResponse {
        worker_id: state.agent.worker_id(),
        listen: state.listen.to_string(),
    })
}

async fn put_blocks(
    State(state): State<AppState>,
    Json(blocks): Json<Vec<G3pbPutBlock>>,
) -> StatusCode {
    state.agent.put_blocks(blocks).await;
    StatusCode::NO_CONTENT
}

async fn query_blocks(
    State(state): State<AppState>,
    Json(request): Json<G3pbQueryRequest>,
) -> Json<Vec<G3pbQueryHit>> {
    Json(state.agent.query_blocks(&request.sequence_hashes).await)
}

async fn offer_blocks(
    State(state): State<AppState>,
    Json(request): Json<G3pbOfferRequest>,
) -> Json<G3pbOfferResponse> {
    let accepted = state.runtime.offer_blocks(&state.agent, &request.blocks).await;

    Json(G3pbOfferResponse { accepted })
}

async fn stage_put_blocks(
    State(state): State<AppState>,
    Json(request): Json<G3pbStageBlocksRequest>,
) -> Result<Json<G3pbStageBlocksResponse>, StatusCode> {
    match state.runtime.stage_put_blocks(request.blocks).await {
        Ok(response) => Ok(Json(response)),
        Err(_) => Err(StatusCode::BAD_REQUEST),
    }
}

async fn commit_put_blocks(
    State(state): State<AppState>,
    Json(request): Json<G3pbCommitRequest>,
) -> Result<StatusCode, StatusCode> {
    state
        .runtime
        .commit_staged_blocks(&state.agent, &request.sequence_hashes)
        .await
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(|_| StatusCode::BAD_REQUEST)
}

async fn put_payload_blocks(
    State(state): State<AppState>,
    Json(request): Json<G3pbPutPayloadRequest>,
) -> Result<StatusCode, StatusCode> {
    match state
        .agent
        .offer_and_put_payload_blocks(request.blocks)
        .await
    {
        Ok(_) => Ok(StatusCode::NO_CONTENT),
        Err(dynamo_llm::block_manager::distributed::G3pbError::InvalidPayloadSize { .. }) => {
            Err(StatusCode::BAD_REQUEST)
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn fetch_blocks(
    State(state): State<AppState>,
    Json(request): Json<G3pbFetchRequest>,
) -> Result<Json<G3pbFetchBlocksResponse>, StatusCode> {
    let blocks = state
        .runtime
        .fetch_descriptors(&request.sequence_hashes)
        .map_err(|err| match err {
            G3pbError::NotFound { .. } => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    Ok(Json(blocks))
}

fn build_agent(worker_id: u64) -> Result<NixlAgent> {
    let agent = NixlAgent::new(&worker_id.to_string())?;
    let (_, ucx_params) = agent.get_plugin_params("UCX")?;
    agent.create_backend("UCX", &ucx_params)?;
    Ok(agent)
}

async fn build_backend(args: &Args) -> Result<AppState> {
    let agent = if let Some(dir) = &args.foyer_dir {
        let mut config = G3pbFoyerStorageConfig::new(dir.clone());
        config.name = format!("g3pb-peer-cache-{}", args.worker_id);
        config.memory_capacity_bytes = args.foyer_memory_bytes;
        config.disk_capacity_bytes = args.foyer_disk_bytes;

        let storage = Arc::new(FoyerG3pbPeerStorage::new(config).await?);
        Arc::new(G3pbStorageAgent::new_with_storage(args.worker_id, storage))
    } else {
        Arc::new(G3pbStorageAgent::new(args.worker_id))
    };

    let runtime = Arc::new(G3pbPeerRuntime::new(args).await?);

    Ok(AppState {
        agent,
        runtime,
        listen: args.listen,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    let state = build_backend(&args).await?;

    let app = Router::new()
        .route("/health", get(health))
        .route("/offer", post(offer_blocks))
        .route("/put", post(put_blocks))
        .route("/stage_put", post(stage_put_blocks))
        .route("/commit_put", post(commit_put_blocks))
        .route("/put_payload", post(put_payload_blocks))
        .route("/query", post(query_blocks))
        .route("/fetch", post(fetch_blocks))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    println!("kvbm_g3pb_backend listening on http://{}", args.listen);
    axum::serve(listener, app).await?;
    Ok(())
}
