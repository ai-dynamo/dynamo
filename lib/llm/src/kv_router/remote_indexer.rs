// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use futures::StreamExt;

use dynamo_runtime::{
    component::Component,
    pipeline::{ManyOut, RouterMode, SingleIn, network::egress::push_router::PushRouter},
};

use dynamo_kv_router::{
    indexer::{IndexerQueryRequest, IndexerQueryResponse, KV_INDEXER_QUERY_ENDPOINT},
    protocols::{LocalBlockHash, OverlapScores},
};

/// A remote indexer that queries a standalone KV indexer via the request plane.
///
/// Used by the frontend when `remote_indexer_component` is configured. Instead of
/// maintaining a local radix tree, this forwards `find_matches` queries to the
/// standalone indexer service over the Dynamo request plane.
pub struct RemoteIndexer {
    router: PushRouter<IndexerQueryRequest, IndexerQueryResponse>,
    model_name: String,
    namespace: String,
}

impl RemoteIndexer {
    pub async fn new(
        component: &Component,
        indexer_component_name: &str,
        model_name: String,
    ) -> Result<Self> {
        let namespace = component.namespace().name();
        let indexer_ns = component.namespace();
        let indexer_component = indexer_ns.component(indexer_component_name)?;
        let endpoint = indexer_component.endpoint(KV_INDEXER_QUERY_ENDPOINT);
        let client = endpoint.client().await?;
        let router =
            PushRouter::from_client_no_fault_detection(client, RouterMode::RoundRobin).await?;
        Ok(Self {
            router,
            model_name,
            namespace,
        })
    }

    pub async fn find_matches(&self, block_hashes: Vec<LocalBlockHash>) -> Result<OverlapScores> {
        let request = IndexerQueryRequest {
            model_name: self.model_name.clone(),
            namespace: self.namespace.clone(),
            block_hashes,
        };
        let mut stream: ManyOut<IndexerQueryResponse> =
            self.router.round_robin(SingleIn::new(request)).await?;

        match stream.next().await {
            Some(IndexerQueryResponse::Scores(scores)) => Ok(scores.into()),
            Some(IndexerQueryResponse::Error(msg)) => {
                Err(anyhow::anyhow!("Remote indexer error: {}", msg))
            }
            None => Err(anyhow::anyhow!("Remote indexer returned empty response")),
        }
    }
}
