// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use dynamo_llm::block_manager::Storage;
use dynamo_llm::block_manager::distributed::{
    G4BlockIndex, G4FetchRequest, G4FetchResponse, G4HealthResponse, G4OfferRequest,
    G4OfferResponse, G4PutBlock, G4PutPayloadRequest, G4QueryHit, G4QueryRequest, G4StorageAgent,
    G4StorageWorker, G4TransferBlock, KvbmLeader, KvbmLeaderConfig, KvbmLeaderNumBlocksConfig,
    KvbmWorker, KvbmWorkerConfig,
};
use dynamo_llm::block_manager::storage::{
    DeviceAllocator, StorageAllocator,
    torch::{TorchDevice, TorchTensor},
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
struct MockTensor {
    ptr: u64,
    size: usize,
    shape: Vec<usize>,
}

impl MockTensor {
    fn new(shape: Vec<usize>, device_id: usize, dtype_width_bytes: usize) -> Result<Self> {
        let allocator = DeviceAllocator::new(device_id)?;
        let size = shape.iter().product::<usize>() * dtype_width_bytes;
        let device_storage = ManuallyDrop::new(allocator.allocate(size)?);

        Ok(Self {
            ptr: device_storage.addr(),
            size,
            shape,
        })
    }
}

impl TorchTensor for MockTensor {
    fn device(&self) -> TorchDevice {
        TorchDevice::Cuda(0)
    }

    fn data_ptr(&self) -> u64 {
        self.ptr
    }

    fn size_bytes(&self) -> usize {
        self.size
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn stride(&self) -> Vec<usize> {
        let mut stride = vec![1];
        for i in (0..self.shape.len() - 1).rev() {
            stride.push(stride.last().unwrap() * self.shape[i]);
        }
        stride.reverse();
        stride
    }
}

#[derive(Clone, Debug)]
struct Args {
    listen: SocketAddr,
    worker_id: u64,
    device_id: usize,
    num_device_blocks: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    leader_pub_url: String,
    leader_ack_url: String,
    host_blocks: usize,
    disk_blocks: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:58080".parse().unwrap(),
            worker_id: 41,
            device_id: 0,
            num_device_blocks: 8,
            page_size: 32,
            dtype_width_bytes: 2,
            leader_pub_url: "tcp://127.0.0.1:56011".to_string(),
            leader_ack_url: "tcp://127.0.0.1:56012".to_string(),
            host_blocks: 8,
            disk_blocks: 8,
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
                    args.device_id = it
                        .next()
                        .context("missing value for --device-id")?
                        .parse()?
                }
                "--num-device-blocks" => {
                    args.num_device_blocks = it
                        .next()
                        .context("missing value for --num-device-blocks")?
                        .parse()?
                }
                "--page-size" => {
                    args.page_size = it
                        .next()
                        .context("missing value for --page-size")?
                        .parse()?
                }
                "--dtype-width-bytes" => {
                    args.dtype_width_bytes = it
                        .next()
                        .context("missing value for --dtype-width-bytes")?
                        .parse()?
                }
                "--leader-pub-url" => {
                    args.leader_pub_url = it.next().context("missing value for --leader-pub-url")?
                }
                "--leader-ack-url" => {
                    args.leader_ack_url = it.next().context("missing value for --leader-ack-url")?
                }
                "--host-blocks" => {
                    args.host_blocks = it
                        .next()
                        .context("missing value for --host-blocks")?
                        .parse()?
                }
                "--disk-blocks" => {
                    args.disk_blocks = it
                        .next()
                        .context("missing value for --disk-blocks")?
                        .parse()?
                }
                "--help" | "-h" => {
                    println!(
                        "kvbm_g4_backend
  --listen <addr>                 HTTP listen address (default 127.0.0.1:58080)
  --worker-id <id>                backend worker id (default 41)
  --device-id <id>                CUDA device id (default 0)
  --num-device-blocks <n>         local worker device blocks (default 8)
  --page-size <n>                 KVBM page size (default 32)
  --dtype-width-bytes <n>         dtype width bytes (default 2)
  --leader-pub-url <url>          worker/leader pub url
  --leader-ack-url <url>          worker/leader ack url
  --host-blocks <n>               leader host blocks (default 8)
  --disk-blocks <n>               leader disk blocks (default 8)"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown flag: {other}"),
            }
        }

        Ok(args)
    }
}

#[derive(Clone)]
struct AppState {
    agent: Arc<G4StorageAgent>,
    payloads: Arc<tokio::sync::RwLock<HashMap<u64, G4TransferBlock>>>,
    listen: SocketAddr,
    _leader: Arc<KvbmLeader>,
    _worker: Arc<tokio::sync::Mutex<KvbmWorker>>,
}

async fn health(State(state): State<AppState>) -> Json<G4HealthResponse> {
    Json(G4HealthResponse {
        worker_id: state.agent.worker_id(),
        listen: state.listen.to_string(),
    })
}

async fn put_blocks(
    State(state): State<AppState>,
    Json(blocks): Json<Vec<G4PutBlock>>,
) -> StatusCode {
    state.agent.put_blocks(blocks).await;
    StatusCode::NO_CONTENT
}

async fn query_blocks(
    State(state): State<AppState>,
    Json(request): Json<G4QueryRequest>,
) -> Json<Vec<G4QueryHit>> {
    Json(state.agent.query_blocks(&request.sequence_hashes).await)
}

async fn offer_blocks(
    State(state): State<AppState>,
    Json(request): Json<G4OfferRequest>,
) -> Json<G4OfferResponse> {
    let accepted = state.agent.offer_blocks(&request.blocks).await;

    Json(G4OfferResponse { accepted })
}

async fn put_payload_blocks(
    State(state): State<AppState>,
    Json(request): Json<G4PutPayloadRequest>,
) -> Result<StatusCode, StatusCode> {
    let accepted = match state.agent.offered_payload_blocks(request.blocks).await {
        Ok(accepted) => accepted,
        Err(dynamo_llm::block_manager::distributed::G4Error::InvalidPayloadSize { .. }) => {
            return Err(StatusCode::BAD_REQUEST);
        }
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    if accepted.is_empty() {
        return Ok(StatusCode::NO_CONTENT);
    }

    let mut payloads = state.payloads.write().await;
    for block in &accepted {
        payloads.insert(block.meta.sequence_hash, block.clone());
    }

    state
        .agent
        .put_blocks(accepted.into_iter().map(|block| block.meta).collect())
        .await;

    Ok(StatusCode::NO_CONTENT)
}

async fn fetch_blocks(
    State(state): State<AppState>,
    Json(request): Json<G4FetchRequest>,
) -> Result<Json<G4FetchResponse>, StatusCode> {
    let payloads = state.payloads.read().await;
    let mut blocks = Vec::with_capacity(request.sequence_hashes.len());

    for sequence_hash in request.sequence_hashes {
        let Some(block) = payloads.get(&sequence_hash) else {
            return Err(StatusCode::NOT_FOUND);
        };
        blocks.push(block.clone());
    }

    Ok(Json(G4FetchResponse { blocks }))
}

async fn build_backend(
    args: &Args,
) -> Result<(
    Arc<KvbmLeader>,
    Arc<tokio::sync::Mutex<KvbmWorker>>,
    Arc<G4StorageAgent>,
    Arc<tokio::sync::RwLock<HashMap<u64, G4TransferBlock>>>,
)> {
    let shape = vec![args.num_device_blocks, 1, 2, args.page_size, 128];
    let tensors: Vec<Arc<dyn TorchTensor>> = vec![Arc::new(MockTensor::new(
        shape,
        args.device_id,
        args.dtype_width_bytes,
    )?)];

    let worker_config = KvbmWorkerConfig::builder()
        .cancel_token(CancellationToken::new())
        .num_device_blocks(args.num_device_blocks)
        .page_size(args.page_size)
        .dtype_width_bytes(args.dtype_width_bytes)
        .device_id(args.device_id)
        .tensors(tensors)
        .leader_pub_url(args.leader_pub_url.clone())
        .leader_ack_url(args.leader_ack_url.clone())
        .build()?;

    let mut worker = KvbmWorker::new(worker_config, false).await?;

    let leader_config = KvbmLeaderConfig::builder()
        .world_size(1)
        .leader_pub_url(args.leader_pub_url.clone())
        .leader_ack_url(args.leader_ack_url.clone())
        .host_blocks_config(KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 0.0,
            num_blocks_overriden: args.host_blocks,
        })
        .disk_blocks_config(KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 0.0,
            num_blocks_overriden: args.disk_blocks,
        })
        .build()?;

    let leader = Arc::new(KvbmLeader::new(leader_config).await?);
    let block_index = Arc::new(G4BlockIndex::default());
    let storage_worker = G4StorageWorker {
        worker_id: args.worker_id,
        endpoint: format!("http://{}", args.listen),
    };
    let agent = Arc::new(
        worker
            .into_g4_storage_agent(storage_worker, block_index)
            .await?,
    );
    let payloads = Arc::new(tokio::sync::RwLock::new(HashMap::new()));

    Ok((
        leader,
        Arc::new(tokio::sync::Mutex::new(worker)),
        agent,
        payloads,
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    let (leader, worker, agent, payloads) = build_backend(&args).await?;

    let state = AppState {
        agent,
        payloads,
        listen: args.listen,
        _leader: leader,
        _worker: worker,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/offer", post(offer_blocks))
        .route("/put", post(put_blocks))
        .route("/put_payload", post(put_payload_blocks))
        .route("/query", post(query_blocks))
        .route("/fetch", post(fetch_blocks))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    println!("kvbm_g4_backend listening on http://{}", args.listen);
    axum::serve(listener, app).await?;
    Ok(())
}
