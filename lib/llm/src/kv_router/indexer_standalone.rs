// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};

use dynamo_runtime::{
    component::Component,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
        network::Ingress,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
    stream,
};

use crate::kv_router::{
    Indexer, KV_INDEXER_QUERY_ENDPOINT, KvRouterConfig,
    protocols::{
        BlockExtraInfo, LocalBlockHash, OverlapScores, RouterEvent, compute_block_hash_for_seq,
    },
    subscriber,
};

#[derive(Serialize, Deserialize, Debug)]
pub enum IndexerQueryRequest {
    FindMatchesHashed {
        block_hashes: Vec<LocalBlockHash>,
    },
    FindMatchesTokens {
        tokens: Vec<u32>,
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    },
    DumpTree,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum IndexerQueryResponse {
    Matches(OverlapScores),
    TreeDump(Vec<RouterEvent>),
    Error(String),
}

impl MaybeError for IndexerQueryResponse {
    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        IndexerQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<anyhow::Error> {
        match self {
            IndexerQueryResponse::Error(msg) => Some(anyhow::Error::msg(msg.clone())),
            _ => None,
        }
    }
}

struct IndexerQueryEngine {
    indexer: Indexer,
    block_size: u32,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<IndexerQueryRequest>,
        ManyOut<Annotated<IndexerQueryResponse>>,
        anyhow::Error,
    > for IndexerQueryEngine
{
    async fn generate(
        &self,
        request: SingleIn<IndexerQueryRequest>,
    ) -> Result<ManyOut<Annotated<IndexerQueryResponse>>> {
        let (request, ctx) = request.into_parts();

        if matches!(request, IndexerQueryRequest::DumpTree) {
            let response = match self.indexer.dump_events().await {
                Ok(events) => IndexerQueryResponse::TreeDump(events),
                Err(e) => IndexerQueryResponse::Error(format!("{e:?}")),
            };
            return Ok(ResponseStream::new(
                Box::pin(stream::iter(vec![Annotated::from_data(response)])),
                ctx.context(),
            ));
        }

        let block_hashes = match request {
            IndexerQueryRequest::FindMatchesHashed { block_hashes } => block_hashes,
            IndexerQueryRequest::FindMatchesTokens {
                tokens,
                block_mm_infos,
            } => compute_block_hash_for_seq(&tokens, self.block_size, block_mm_infos.as_deref()),
            IndexerQueryRequest::DumpTree => unreachable!(),
        };

        let response = match self.indexer.find_matches(block_hashes).await {
            Ok(scores) => IndexerQueryResponse::Matches(scores),
            Err(e) => IndexerQueryResponse::Error(format!("{e:?}")),
        };

        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![Annotated::from_data(response)])),
            ctx.context(),
        ))
    }
}

async fn start_indexer_query_endpoint(
    component: Component,
    indexer: Indexer,
    block_size: u32,
) -> Result<()> {
    let engine = std::sync::Arc::new(IndexerQueryEngine {
        indexer,
        block_size,
    });

    let ingress = Ingress::for_engine(engine)?;

    let fut = component
        .endpoint(KV_INDEXER_QUERY_ENDPOINT)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start();

    tokio::spawn(async move {
        if let Err(e) = fut.await {
            tracing::error!("Indexer query endpoint failed: {e:?}");
        }
    });

    Ok(())
}

pub async fn start_kv_block_indexer(
    component: &Component,
    kv_router_config: &KvRouterConfig,
    block_size: u32,
) -> Result<Indexer> {
    if kv_router_config.durable_kv_events {
        anyhow::bail!(
            "standalone indexer does not support durable_kv_events (JetStream): \
             consumer ID collisions, orphan cleanup conflicts, and snapshot/purge races \
             make it incompatible with an independent indexer"
        );
    }

    let indexer = Indexer::new(component, kv_router_config, block_size);

    subscriber::start_subscriber(component.clone(), kv_router_config, indexer.clone()).await?;

    start_indexer_query_endpoint(component.clone(), indexer.clone(), block_size).await?;

    tracing::info!(
        "Standalone KV indexer started with query endpoint '{KV_INDEXER_QUERY_ENDPOINT}'"
    );

    Ok(indexer)
}
