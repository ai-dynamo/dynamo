// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::selector::WorkerSelector;

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};
use crate::local_model::runtime_config::ModelRuntimeConfig;

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
        self.select_prefill_worker_inner(
            None,
            token_ids,
            block_mm_infos,
            lora_name,
            cache_namespace,
            priority_jump,
            strict_priority,
            allowed_worker_ids,
            routing_constraints,
        )
        .await
    }

    /// Atomically select and reserve the best prefill worker.
    ///
    /// The reservation contributes to scheduler load before this method returns,
    /// so concurrent callers cannot all select from the same stale load snapshot.
    /// The caller must eventually release it with [`Self::free_prefill_request`].
    #[expect(clippy::too_many_arguments)]
    pub async fn reserve_prefill_worker(
        &self,
        reservation_id: &str,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        if reservation_id.is_empty() {
            anyhow::bail!("prefill reservation ID must not be empty");
        }
        self.select_prefill_worker_inner(
            Some(reservation_id),
            token_ids,
            block_mm_infos,
            lora_name,
            cache_namespace,
            priority_jump,
            strict_priority,
            allowed_worker_ids,
            routing_constraints,
        )
        .await
    }

    #[expect(clippy::too_many_arguments)]
    async fn select_prefill_worker_inner(
        &self,
        reservation_id: Option<&str>,
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
        let binding = self
            .binding
            .load_full()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match &binding.router {
            InnerPrefillRouter::KvRouter(router) => {
                let outcome = router
                    .chooser
                    .find_best_match_details(
                        reservation_id,
                        token_ids,
                        block_mm_infos,
                        None,
                        reservation_id.is_some(),
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

    /// Release a prefill reservation if it is still owned by `worker`.
    ///
    /// Missing reservations and ownership mismatches are idempotent no-ops.
    pub async fn free_prefill_request(
        &self,
        reservation_id: &str,
        worker: WorkerWithDpRank,
    ) -> Result<()> {
        let Some(binding) = self.binding.load_full() else {
            return Ok(());
        };
        if let InnerPrefillRouter::KvRouter(router) = &binding.router {
            router
                .chooser
                .free_if_worker(reservation_id, worker)
                .await?;
        }
        Ok(())
    }

    /// Release a gateway-owned prefill reservation by its unforgeable ID.
    ///
    /// This variant is used by the native ext-proc cancellation guard, which
    /// may be dropped after admission but before the selected worker is
    /// returned to the caller. Gateway-generated UUIDs make an unqualified
    /// release ownership-safe; external C callers should use
    /// [`Self::free_prefill_request`] instead.
    pub async fn free_prefill_reservation(&self, reservation_id: &str) -> Result<()> {
        let Some(binding) = self.binding.load_full() else {
            return Ok(());
        };
        if let InnerPrefillRouter::KvRouter(router) = &binding.router {
            router.chooser.free(reservation_id).await?;
        }
        Ok(())
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(binding) = self.binding.load_full()
            && let InnerPrefillRouter::KvRouter(router) = &binding.router
        {
            router.chooser.register_workers(worker_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, atomic::Ordering},
    };

    use dynamo_kv_router::{config::KvRouterConfig, selector::DefaultWorkerSelector};
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{PushRouter, RouterMode},
        protocols::annotated::Annotated,
    };
    use tokio::sync::watch;

    use super::super::PrefillBinding;
    use super::*;
    use crate::{
        discovery::ModelManager,
        kv_router::{KvPushRouter, KvRouter},
        protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
        worker_type::WorkerType,
    };

    async fn tracked_prefill_router() -> (Arc<PrefillRouter>, Arc<KvRouter>) {
        let runtime = Runtime::from_current().unwrap();
        let distributed = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let component = distributed
            .namespace(format!("tracked-prefill-test-{}", uuid::Uuid::new_v4()))
            .unwrap()
            .component("workers".to_string())
            .unwrap();
        let endpoint = component.endpoint("generate");
        let endpoint_id = endpoint.id();
        let client = endpoint.client().await.unwrap();
        let (_workers_tx, workers_rx) = watch::channel(HashMap::from([
            (7, ModelRuntimeConfig::default()),
            (8, ModelRuntimeConfig::default()),
        ]));
        let config = KvRouterConfig {
            overlap_score_credit: 0.0,
            router_temperature: 0.0,
            use_kv_events: false,
            router_track_active_blocks: false,
            router_track_prefill_tokens: true,
            skip_initial_worker_wait: true,
            ..Default::default()
        };
        let chooser = Arc::new(
            KvRouter::new_with_worker_role(
                endpoint,
                client.clone(),
                workers_rx,
                None,
                16,
                DefaultWorkerSelector::new(Some(config.clone()), "prefill"),
                Some(config),
                None,
                Some(WorkerType::Prefill),
                "prefill",
                Some("tracked-prefill-test".to_string()),
                false,
                None,
                None,
            )
            .await
            .unwrap(),
        );
        let push_router =
            PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client(
                client,
                RouterMode::KV,
            )
            .await
            .unwrap();
        let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
        router.binding.store(Some(Arc::new(PrefillBinding {
            endpoint_id,
            router: InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new_with_coordinator(
                push_router,
                chooser.clone(),
                None,
            ))),
        })));
        router
            .lifecycle
            .store(PrefillLifecycleState::Active as u8, Ordering::Release);

        (router, chooser)
    }

    fn routed_worker(outcome: PrefillQueryOutcome) -> WorkerWithDpRank {
        match outcome {
            PrefillQueryOutcome::Routed {
                worker_id,
                dp_rank: Some(dp_rank),
            } => WorkerWithDpRank::new(worker_id, dp_rank),
            _ => panic!("expected routed prefill worker with a DP rank"),
        }
    }

    #[tokio::test]
    async fn tracked_admissions_observe_prior_prefill_booking() {
        let (router, chooser) = tracked_prefill_router().await;
        let tokens = vec![1; 64];

        let first = routed_worker(
            router
                .reserve_prefill_worker(
                    "reservation-1",
                    &tokens,
                    None,
                    None,
                    None,
                    0.0,
                    0,
                    None,
                    RoutingConstraints::default(),
                )
                .await
                .unwrap(),
        );
        let second = routed_worker(
            router
                .reserve_prefill_worker(
                    "reservation-2",
                    &tokens,
                    None,
                    None,
                    None,
                    0.0,
                    0,
                    None,
                    RoutingConstraints::default(),
                )
                .await
                .unwrap(),
        );

        assert_ne!(
            first, second,
            "the second request should select the idle worker"
        );
        let loads = chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert_eq!(loads.len(), 2);
        assert!(loads.iter().all(|load| load.potential_prefill_tokens == 64));

        router
            .free_prefill_request("reservation-1", first)
            .await
            .unwrap();
        router
            .free_prefill_request("reservation-2", second)
            .await
            .unwrap();
        let loads = chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(loads.iter().all(|load| load.potential_prefill_tokens == 0));
    }
}
