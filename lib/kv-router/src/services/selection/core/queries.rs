// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only partition queries: loads, potential loads and overlap scores.

use super::*;

impl SelectionCore {
    pub fn loads(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<ModelLoadResponse> {
        let entries = self.initialized_entries();
        let mut loads = Vec::new();
        for entry in entries {
            if model_name.is_some_and(|model_name| entry.key.model_name != model_name)
                || routing_group
                    .is_some_and(|routing_group| entry.key.routing_group != routing_group)
            {
                continue;
            }
            loads.push(ModelLoadResponse {
                model_name: entry.key.model_name.clone(),
                routing_group: entry.key.routing_group.clone(),
                loads: entry
                    .scheduler
                    .get_potential_loads(None, 0, Default::default(), false),
                pending_count: entry.scheduler.pending_count(),
                pending_isl_tokens: entry.scheduler.pending_isl_tokens(),
            });
        }
        loads.sort_by(|a, b| {
            (&a.model_name, &a.routing_group).cmp(&(&b.model_name, &b.routing_group))
        });
        loads
    }

    pub async fn potential_loads(
        &self,
        req: PotentialLoadsRequest,
    ) -> Result<Vec<PotentialLoad>, SelectionError> {
        let key = RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone());
        let entry = self.ready_entry(&key)?;
        let prepared = self
            .prepare_selection_inputs(
                &entry,
                &req.prompt.view(),
                Some(
                    self.kv_router_config
                        .assume_kv_reuse(req.router_config_override.as_ref()),
                ),
                false,
                false,
                &mut None,
            )
            .await?;
        let track_prefill_tokens = req
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.kv_router_config.router_track_prefill_tokens);
        Ok(entry.scheduler.get_potential_loads(
            Some(prepared.sequence_hashes),
            prepared.isl_tokens,
            prepared.overlap.effective_cached_tokens,
            track_prefill_tokens,
        ))
    }

    pub async fn overlap_scores(
        &self,
        req: OverlapScoresRequest,
    ) -> Result<OverlapScoresResponse, SelectionError> {
        let key = RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone());
        let entry = self.ready_entry(&key)?;
        let prompt = req.prompt.view();
        let max_reusable_blocks = prompt
            .routing_tokens_and_mm_infos()
            .map(|(tokens, _)| tokens.len())
            .or(prompt.isl_tokens)
            .map(|tokens| tokens.saturating_sub(1) / entry.block_size as usize);
        let block_hashes = req
            .prompt
            .view()
            .block_hashes_for_indexer(entry.block_size, entry.is_eagle)?;
        let num_blocks = block_hashes.len();
        let tiered = entry
            .indexer
            .find_tiered_matches_with_options(
                block_hashes,
                crate::indexer::LowerTierQueryOptions {
                    retain_kv_transfer_chain: false,
                    max_reusable_blocks,
                },
            )
            .await
            .map_err(|error| SelectionError::Internal(error.to_string()))?;
        let schedulable_workers = self.schedulable_worker_ranks(&key);
        Ok(
            OverlapAnalysis::new(&self.kv_router_config, entry.block_size, &tiered)
                .scores_response(num_blocks, schedulable_workers, false, None, None),
        )
    }

    fn schedulable_worker_ranks(&self, key: &RoutingPartitionId) -> Vec<WorkerWithDpRank> {
        let configs = self.catalog.scheduler_configs_for_key(key);
        let mut workers = Vec::new();
        for (worker_id, config) in configs {
            let start = config.data_parallel_start_rank;
            let end = start.saturating_add(config.data_parallel_size);
            for dp_rank in start..end {
                workers.push(WorkerWithDpRank::new(worker_id, dp_rank));
            }
        }
        workers
    }
}
