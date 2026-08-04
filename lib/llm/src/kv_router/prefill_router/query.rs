// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId};

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};

impl PrefillRouter {
    /// Query the best prefill worker without executing a request.
    ///
    /// This query is advisory and does not book scheduler or occupancy state;
    /// concurrent callers may observe the same worker. `policy_class` names the
    /// policy-class family to admit the request under; `None` uses the router's
    /// configured default family. A returned `PrefillQueryOutcome::QueueRejected`
    /// means the class refused admission outright (its
    /// `request_queue_limit_per_worker` is `0`) rather than queuing, so callers
    /// that want per-class prefill shedding must treat it as a shed, not as
    /// "prefill unavailable, fall back to aggregated."
    #[expect(clippy::too_many_arguments)]
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        policy_class: Option<String>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match prefill_router {
            InnerPrefillRouter::KvRouter(router) => {
                let outcome = router
                    .chooser
                    .find_best_match_details_with_policy_class(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        false,
                        false,
                        lora_name,
                        cache_namespace,
                        priority_jump,
                        strict_priority,
                        policy_class,
                        None,
                        None,
                        None,
                        allowed_worker_ids,
                        routing_constraints,
                    )
                    .await?;
                match outcome {
                    crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                        Ok(PrefillQueryOutcome::Routed {
                            worker_id: worker.worker_id,
                            dp_rank: Some(worker.dp_rank),
                        })
                    }
                    crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                        Ok(PrefillQueryOutcome::QueueRejected { rejection })
                    }
                }
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                let worker_id = router
                    .peek_next_worker()
                    .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                Ok(PrefillQueryOutcome::Routed {
                    worker_id,
                    dp_rank: None,
                })
            }
        }
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(InnerPrefillRouter::KvRouter(router)) = self.prefill_router.get() {
            router.chooser.register_workers(worker_ids);
        }
    }
}
