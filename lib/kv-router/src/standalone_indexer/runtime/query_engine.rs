// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};
use dynamo_runtime::stream;

use crate::indexer::{IndexerQueryRequest, IndexerQueryResponse};
use crate::standalone_indexer::registry::{IndexerKey, WorkerRegistry};

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
            Some(entry) => match entry.indexer.find_matches(req.block_hashes).await {
                Ok(scores) => IndexerQueryResponse::Scores(scores.into()),
                Err(err) => IndexerQueryResponse::Error(err.to_string()),
            },
            None => IndexerQueryResponse::Error(format!(
                "no indexer for model={} namespace={}",
                req.model_name, req.namespace
            )),
        };

        let response_stream = stream::iter(vec![response]);
        Ok(ResponseStream::new(
            Box::pin(response_stream),
            ctx.context(),
        ))
    }
}
