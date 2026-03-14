// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::stream;

use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};

use dynamo_kv_router::indexer::{IndexerQueryRequest, IndexerQueryResponse};

use dynamo_kv_router::standalone_indexer::registry::{IndexerKey, WorkerRegistry};

/// AsyncEngine that serves indexer queries over the request plane.
///
/// When a frontend sends an `IndexerQueryRequest` (model_name, namespace, block hashes),
/// this engine finds the appropriate indexer in the registry and returns overlap scores.
pub struct IndexerQueryEngine {
    pub registry: Arc<WorkerRegistry>,
}

#[async_trait]
impl AsyncEngine<SingleIn<IndexerQueryRequest>, ManyOut<IndexerQueryResponse>, anyhow::Error>
    for IndexerQueryEngine
{
    async fn generate(
        &self,
        request: SingleIn<IndexerQueryRequest>,
    ) -> Result<ManyOut<IndexerQueryResponse>> {
        let (req, ctx) = request.into_parts();

        let key = IndexerKey {
            model_name: req.model_name.clone(),
            tenant_id: req.namespace.clone(),
        };

        let response = match self.registry.get_indexer(&key) {
            Some(ie) => match ie.indexer.find_matches(req.block_hashes).await {
                Ok(scores) => IndexerQueryResponse::Scores(scores.into()),
                Err(e) => IndexerQueryResponse::Error(e.to_string()),
            },
            None => IndexerQueryResponse::Error(format!(
                "no indexer for model={} namespace={}",
                req.model_name, req.namespace
            )),
        };

        let resp_stream = stream::iter(vec![response]);
        Ok(ResponseStream::new(Box::pin(resp_stream), ctx.context()))
    }
}
