// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId};
use dynamo_kv_router::selector::WorkerSelector;

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};
use crate::{kv_router::sequence::SequenceError, local_model::runtime_config::ModelRuntimeConfig};

fn map_kv_query_outcome(outcome: crate::kv_router::FindBestMatchOutcome) -> PrefillQueryOutcome {
    match outcome {
        crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
            PrefillQueryOutcome::Routed {
                worker_id: worker.worker_id,
                dp_rank: Some(worker.dp_rank),
            }
        }
        crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
            PrefillQueryOutcome::QueueRejected { rejection }
        }
    }
}

fn ignore_missing_request(result: std::result::Result<(), SequenceError>) -> Result<()> {
    match result {
        Ok(()) | Err(SequenceError::RequestNotFound { .. }) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillRequestMode<'a> {
    Advisory,
    Tracked(&'a str),
}

impl<'a> PrefillRequestMode<'a> {
    fn scheduler_args(self) -> (Option<&'a str>, bool) {
        match self {
            Self::Advisory => (None, false),
            Self::Tracked(request_id) => (Some(request_id), true),
        }
    }
}

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// Query the best prefill worker without executing a request.
    ///
    /// This query is advisory and does not book scheduler or occupancy state;
    /// concurrent callers may observe the same worker.
    #[expect(clippy::too_many_arguments)]
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
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
                let (request_id, update_states) = PrefillRequestMode::Advisory.scheduler_args();
                let outcome = router
                    .chooser
                    .find_best_match_details(
                        request_id,
                        token_ids,
                        block_mm_infos,
                        None,
                        update_states,
                        false,
                        lora_name,
                        cache_namespace,
                        priority_jump,
                        strict_priority,
                        None,
                        None,
                        allowed_worker_ids,
                        routing_constraints,
                    )
                    .await?;
                Ok(map_kv_query_outcome(outcome))
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

    /// Select and reserve the best prefill worker for an externally-dispatched request.
    ///
    /// Unlike [`Self::query_prefill_worker`], this performs normal scheduler admission and
    /// books the request under `request_id`. The caller must later invoke
    /// [`Self::mark_prefill_completed`] and [`Self::free`] as the request progresses.
    #[expect(clippy::too_many_arguments)]
    pub async fn reserve_prefill_worker(
        &self,
        request_id: &str,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        if request_id.is_empty() {
            anyhow::bail!("request_id is required for a tracked prefill reservation");
        }
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match prefill_router {
            InnerPrefillRouter::KvRouter(router) => {
                let (request_id, update_states) =
                    PrefillRequestMode::Tracked(request_id).scheduler_args();
                let outcome = router
                    .chooser
                    .find_best_match_details(
                        request_id,
                        token_ids,
                        block_mm_infos,
                        None,
                        update_states,
                        false,
                        lora_name,
                        cache_namespace,
                        priority_jump,
                        strict_priority,
                        None,
                        None,
                        allowed_worker_ids,
                        routing_constraints,
                    )
                    .await?;
                Ok(map_kv_query_outcome(outcome))
            }
            InnerPrefillRouter::SimpleRouter(_) => Err(anyhow::anyhow!(
                "Tracked prefill reservations are not supported by the simple router"
            )),
        }
    }

    /// Release the prefill-token reservation for a tracked prefill request.
    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<()> {
        let Some(prefill_router) = self.prefill_router.get() else {
            return Ok(());
        };

        if let InnerPrefillRouter::KvRouter(router) = prefill_router {
            ignore_missing_request(router.chooser.mark_prefill_completed(request_id).await)?;
        }
        Ok(())
    }

    /// Remove a tracked request from prefill scheduler bookkeeping.
    pub async fn free(&self, request_id: &str) -> Result<()> {
        let Some(prefill_router) = self.prefill_router.get() else {
            return Ok(());
        };

        if let InnerPrefillRouter::KvRouter(router) = prefill_router {
            ignore_missing_request(router.chooser.free(request_id).await)?;
        }
        Ok(())
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(InnerPrefillRouter::KvRouter(router)) = self.prefill_router.get() {
            router.chooser.register_workers(worker_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_treats_missing_request_as_success() {
        assert!(
            ignore_missing_request(Err(SequenceError::RequestNotFound {
                request_id: "already-freed".to_string(),
            }))
            .is_ok()
        );
    }

    #[test]
    fn tracked_prefill_mode_supplies_request_id_and_enables_state_updates() {
        assert_eq!(PrefillRequestMode::Advisory.scheduler_args(), (None, false));
        assert_eq!(
            PrefillRequestMode::Tracked("reservation-1").scheduler_args(),
            (Some("reservation-1"), true)
        );
    }
}
