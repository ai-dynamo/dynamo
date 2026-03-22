// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::RadixTree;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{OverlapScores, RouterEvent, compute_block_hash_for_seq};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::shared::{
    ReplayScheduler, replay_policy, replay_selector, replay_slots, replay_workers_with_configs,
};
use crate::common::protocols::{DirectRequest, MockEngineArgs};

struct SyncReplayIndexer {
    block_size: u32,
    tree: RadixTree,
}

impl SyncReplayIndexer {
    fn new(block_size: u32) -> Self {
        Self {
            block_size,
            tree: RadixTree::new(),
        }
    }

    fn find_matches_for_request(&self, tokens: &[u32], lora_name: Option<&str>) -> OverlapScores {
        let sequence = compute_block_hash_for_seq(tokens, self.block_size, None, lora_name);
        self.tree.find_matches(sequence, false)
    }

    fn apply_event(&mut self, event: RouterEvent) -> Result<()> {
        self.tree.apply_event(event).map_err(Into::into)
    }
}

pub(crate) struct OfflineReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    runtime: tokio::runtime::Runtime,
    cancellation_token: CancellationToken,
    scheduler: Arc<ReplayScheduler>,
    indexer: SyncReplayIndexer,
}

impl OfflineReplayRouter {
    pub(crate) fn new(args: &MockEngineArgs, num_workers: usize) -> Result<Self> {
        let config = KvRouterConfig::default();
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let (_worker_config_tx, worker_config_rx) =
            tokio::sync::watch::channel(workers_with_configs);
        let selector = replay_selector(&config);
        let policy = replay_policy(&config, args);
        let cancellation_token = CancellationToken::new();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| anyhow!("failed to create offline replay router runtime: {e}"))?;
        let scheduler = {
            let _guard = runtime.enter();
            Arc::new(dynamo_kv_router::LocalScheduler::new(
                slots,
                worker_config_rx,
                None,
                args.block_size as u32,
                selector,
                policy,
                cancellation_token.clone(),
                "replay",
                false,
            ))
        };

        Ok(Self {
            config,
            block_size: args.block_size as u32,
            runtime,
            cancellation_token,
            scheduler,
            indexer: SyncReplayIndexer::new(args.block_size as u32),
        })
    }

    pub(crate) fn select_worker(&mut self, request: &DirectRequest) -> Result<usize> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("offline replay requires requests to have stable UUIDs"))?;
        let overlaps = self.indexer.find_matches_for_request(&request.tokens, None);
        let token_seq = self.config.compute_seq_hashes_for_tracking(
            &request.tokens,
            self.block_size,
            None,
            None,
        );
        let response = self.runtime.block_on(
            self.scheduler.schedule(
                Some(uuid.to_string()),
                request.tokens.len(),
                token_seq,
                overlaps,
                None,
                true,
                None,
                0.0,
                Some(
                    u32::try_from(request.max_output_tokens)
                        .context("max_output_tokens does not fit into u32")?,
                ),
                None,
            ),
        )?;

        usize::try_from(response.best_worker.worker_id)
            .map_err(|_| anyhow!("selected worker id does not fit into usize"))
    }

    pub(crate) fn apply_event(&mut self, event: RouterEvent) -> Result<()> {
        self.indexer.apply_event(event)
    }

    pub(crate) fn mark_prefill_completed(&mut self, uuid: Uuid) -> Result<()> {
        self.runtime
            .block_on(self.scheduler.mark_prefill_completed(&uuid.to_string()))
            .map_err(anyhow::Error::from)
    }

    pub(crate) fn free(&mut self, uuid: Uuid) -> Result<()> {
        self.runtime
            .block_on(self.scheduler.free(&uuid.to_string()))
            .map_err(anyhow::Error::from)
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.scheduler.pending_count()
    }

    pub(crate) fn shutdown(&mut self) {
        self.cancellation_token.cancel();
        self.runtime.block_on(async {
            tokio::task::yield_now().await;
        });
    }
}
