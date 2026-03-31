// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use dynamo_llm::block_manager::distributed::{
    FoyerG3pbPeerStorage, G3pbError, G3pbFetchRequest, G3pbFetchResponse, G3pbFoyerStorageConfig,
    G3pbHealthResponse, G3pbOfferRequest, G3pbOfferResponse, G3pbPutBlock, G3pbPutPayloadRequest,
    G3pbQueryHit, G3pbQueryRequest, G3pbStorageAgent,
};

#[derive(Clone, Debug)]
struct Args {
    listen: SocketAddr,
    worker_id: u64,
    foyer_dir: Option<PathBuf>,
    foyer_memory_bytes: usize,
    foyer_disk_bytes: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:58080".parse().unwrap(),
            worker_id: 41,
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

#[derive(Clone)]
struct AppState {
    agent: Arc<G3pbStorageAgent>,
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
    let accepted = state.agent.offer_blocks(&request.blocks).await;

    Json(G3pbOfferResponse { accepted })
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
) -> Result<Json<G3pbFetchResponse>, StatusCode> {
    let blocks = state
        .agent
        .fetch_blocks(&request.sequence_hashes)
        .await
        .map_err(|err| match err {
            G3pbError::NotFound { .. } => StatusCode::NOT_FOUND,
            G3pbError::InvalidPayloadSize { .. } => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        })?;

    Ok(Json(G3pbFetchResponse { blocks }))
}

async fn build_backend(args: &Args) -> Result<Arc<G3pbStorageAgent>> {
    if let Some(dir) = &args.foyer_dir {
        let mut config = G3pbFoyerStorageConfig::new(dir.clone());
        config.name = format!("g3pb-peer-cache-{}", args.worker_id);
        config.memory_capacity_bytes = args.foyer_memory_bytes;
        config.disk_capacity_bytes = args.foyer_disk_bytes;

        let storage = Arc::new(FoyerG3pbPeerStorage::new(config).await?);
        return Ok(Arc::new(G3pbStorageAgent::new_with_storage(
            args.worker_id,
            storage,
        )));
    }

    Ok(Arc::new(G3pbStorageAgent::new(args.worker_id)))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    let agent = build_backend(&args).await?;

    let state = AppState {
        agent,
        listen: args.listen,
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
    println!("kvbm_g3pb_backend listening on http://{}", args.listen);
    axum::serve(listener, app).await?;
    Ok(())
}
