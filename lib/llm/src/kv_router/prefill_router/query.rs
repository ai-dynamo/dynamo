// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::{
    protocols::{BlockExtraInfo, RoutingConstraints, WorkerId, WorkerWithDpRank},
    scheduling::{
        AdmissionAttempt,
        queue::{SchedulerBookingCleanup, SchedulerBookingDescriptor},
    },
    selector::WorkerSelector,
};

use super::{PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter};
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// A prefill booking that owns cleanup of the exact scheduler attempt it admitted.
///
/// KV-routed bookings retain their exact scheduler booking descriptor, so
/// cleanup is independent of a later [`PrefillRouter`] binding change and
/// cannot release a newer booking that reused its request ID and worker.
/// Built-in router modes use an untracked, advisory booking whose release is a no-op.
#[must_use]
pub struct PrefillReservation {
    worker: WorkerWithDpRank,
    dp_rank: Option<u32>,
    release: ReservationRelease,
}

enum ReservationRelease {
    Kv {
        cleanup: SchedulerBookingCleanup,
        booking: Option<SchedulerBookingDescriptor>,
    },
    None,
}

impl PrefillReservation {
    pub fn worker_id(&self) -> WorkerId {
        self.worker.worker_id
    }

    pub fn dp_rank(&self) -> Option<u32> {
        self.dp_rank
    }

    /// Release this booking and wait for the scheduler to acknowledge it.
    pub async fn release(mut self) -> Result<()> {
        if let ReservationRelease::Kv { cleanup, booking } = &mut self.release
            && let Some(booking) = booking.take()
        {
            cleanup.enqueue_acknowledged(booking).wait().await?;
        }
        Ok(())
    }
}

impl Drop for PrefillReservation {
    fn drop(&mut self) {
        if let ReservationRelease::Kv { cleanup, booking } = &mut self.release
            && let Some(booking) = booking.take()
        {
            cleanup.enqueue(booking);
        }
    }
}

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// Select a prefill worker and reserve it when KV routing is enabled.
    ///
    /// If this future is dropped while queued, the scheduler retracts its
    /// pending admission. After success, the returned reservation owns the
    /// exact scheduler booking and enqueues cleanup when dropped; explicit
    /// release waits for that cleanup to be acknowledged. Built-in modes retain
    /// their existing advisory selection semantics and return an untracked
    /// reservation.
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
    ) -> Result<PrefillReservation> {
        if reservation_id.is_empty() {
            anyhow::bail!("prefill reservation ID must not be empty");
        }
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let binding = self
            .binding
            .load_full()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;
        let router = &binding.router;
        let Some(chooser) = router.kv_router_if_enabled() else {
            let worker_id = router
                .peek_next_worker()
                .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
            return Ok(PrefillReservation {
                worker: WorkerWithDpRank::new(worker_id, 0),
                dp_rank: None,
                release: ReservationRelease::None,
            });
        };
        let admitted = chooser
            .find_best_match_details_with_lifecycle(
                Some(reservation_id),
                token_ids,
                block_mm_infos,
                None,
                true,
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
        let (outcome, attempt) = admitted.into_parts();
        match outcome {
            crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                let AdmissionAttempt::Tracked(attempt_id) = attempt else {
                    anyhow::bail!("prefill reservation admission did not return a tracked attempt");
                };
                Ok(PrefillReservation {
                    worker,
                    dp_rank: Some(worker.dp_rank),
                    release: ReservationRelease::Kv {
                        cleanup: chooser.booking_cleanup(),
                        booking: Some(SchedulerBookingDescriptor {
                            request_id: reservation_id.to_string(),
                            worker,
                            attempt_id,
                        }),
                    },
                })
            }
            crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                anyhow::bail!(
                    "prefill router policy-class queue rejection: policy_class={}, limit_kind={}, current={}, limit={}",
                    rejection.policy_class,
                    rejection.limit_kind,
                    rejection.current,
                    rejection.limit,
                )
            }
        }
    }

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
        let binding = self
            .binding
            .load_full()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        let router = &binding.router;
        let Some(kv_router) = router.kv_router_if_enabled() else {
            let worker_id = router
                .peek_next_worker()
                .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
            return Ok(PrefillQueryOutcome::Routed {
                worker_id,
                dp_rank: None,
            });
        };
        let outcome = kv_router
            .find_best_match_details(
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

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(binding) = self.binding.load_full()
            && let Some(kv_router) = binding.router.kv_router_if_enabled()
        {
            kv_router.register_workers(worker_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use async_trait::async_trait;
    use dynamo_kv_router::{config::KvRouterConfig, selector::DefaultWorkerSelector};
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::Instance,
        discovery::{EndpointInstanceId, EventTransportKind},
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        engine::{AsyncEngine, AsyncEngineContext},
        pipeline::{
            AddressedRequest, Context, Error, ManyIn, ManyOut, PushRouter, ResponseStream,
            RouterMode, SingleIn, StreamingDispatch, context::Controller,
        },
        storage::kv,
        traits::DistributedRuntimeProvider,
    };
    use futures::{StreamExt, future::join_all};
    use tokio::sync::watch;

    use super::*;
    use crate::{
        discovery::ModelManager,
        kv_router::prefill_router::PrefillBinding,
        kv_router::{KvPushRouter, KvRouter, RouterLoadSource, RoutingHost, RoutingLoadContext},
        protocols::common::{
            FinishReason, llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest,
        },
        worker_type::WorkerType,
    };

    type LlmResponse = dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>;

    struct RecordingDispatch {
        worker_ids: Mutex<Vec<u64>>,
        pending_responses: bool,
    }

    impl RecordingDispatch {
        fn completed() -> Self {
            Self {
                worker_ids: Mutex::new(Vec::new()),
                pending_responses: false,
            }
        }

        fn pending() -> Self {
            Self {
                worker_ids: Mutex::new(Vec::new()),
                pending_responses: true,
            }
        }

        fn response_stream(&self) -> ManyOut<LlmResponse> {
            let context: Arc<dyn AsyncEngineContext> = Arc::new(Controller::default());
            if self.pending_responses {
                return ResponseStream::new(Box::pin(futures::stream::pending()), context);
            }
            ResponseStream::new(
                Box::pin(tokio_stream::iter(vec![LlmResponse::from_data(
                    LLMEngineOutput {
                        finish_reason: Some(FinishReason::EoS),
                        ..Default::default()
                    },
                )])),
                context,
            )
        }
    }

    #[async_trait]
    impl StreamingDispatch<PreprocessedRequest, LlmResponse> for RecordingDispatch {
        async fn generate(
            &self,
            request: SingleIn<AddressedRequest<PreprocessedRequest>>,
        ) -> Result<ManyOut<LlmResponse>, Error> {
            let (addressed, _) = request.transfer(());
            let (_, _, instance) = addressed.into_parts();
            self.worker_ids
                .lock()
                .unwrap()
                .push(instance.expect("selected instance").id());
            Ok(self.response_stream())
        }

        async fn generate_bidirectional(
            &self,
            _instance: Instance,
            _address: String,
            _input: ManyIn<PreprocessedRequest>,
        ) -> Result<ManyOut<LlmResponse>, Error> {
            anyhow::bail!("bidirectional dispatch is unused in this test")
        }

        async fn on_instance_removed(&self, _id: &EndpointInstanceId) {}
    }

    fn distributed_config(root: &std::path::Path) -> DistributedConfig {
        DistributedConfig {
            discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(root.to_path_buf())),
            nats_config: None,
            request_plane: RequestPlaneMode::Tcp,
            event_transport_kind: EventTransportKind::Zmq,
        }
    }

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    async fn query_worker(router: &PrefillRouter) -> u64 {
        match router
            .query_prefill_worker(
                &[1, 2, 3],
                None,
                None,
                None,
                0.0,
                0,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap()
        {
            PrefillQueryOutcome::Routed { worker_id, dp_rank } => {
                assert_eq!(dp_rank, None);
                worker_id
            }
            PrefillQueryOutcome::QueueRejected { .. } => panic!("RR query cannot queue"),
        }
    }

    async fn shared_router(
        runtime: &Runtime,
        discovery_root: &std::path::Path,
        namespace: &str,
        mode: RouterMode,
        dispatch: Arc<RecordingDispatch>,
    ) -> (
        Arc<RoutingHost>,
        Arc<PrefillRouter>,
        Vec<DistributedRuntime>,
        Vec<u64>,
    ) {
        let component = "workers";
        let endpoint_name = "generate";
        let mut worker_runtimes = Vec::new();
        for _ in 0..4 {
            let worker_runtime =
                DistributedRuntime::new(runtime.clone(), distributed_config(discovery_root))
                    .await
                    .unwrap();
            worker_runtime
                .namespace(namespace.to_string())
                .unwrap()
                .component(component.to_string())
                .unwrap()
                .endpoint(endpoint_name)
                .register_endpoint_instance()
                .await
                .unwrap();
            worker_runtimes.push(worker_runtime);
        }

        let router_runtime =
            DistributedRuntime::new(runtime.clone(), distributed_config(discovery_root))
                .await
                .unwrap();
        let client = router_runtime
            .namespace(namespace.to_string())
            .unwrap()
            .component(component.to_string())
            .unwrap()
            .endpoint(endpoint_name)
            .client()
            .await
            .unwrap();
        let instances = tokio::time::timeout(Duration::from_secs(5), async {
            let mut source = client.instance_source.as_ref().clone();
            loop {
                let instances = source.borrow_and_update().clone();
                if instances.len() == 4 {
                    return instances;
                }
                source
                    .changed()
                    .await
                    .expect("discovery source must remain open");
            }
        })
        .await
        .expect("all four workers must be discovered");
        let mut workers = instances.iter().map(Instance::id).collect::<Vec<_>>();
        workers.sort_unstable();
        let load_context = RoutingLoadContext::start(
            client.clone(),
            RouterLoadSource::Prefill,
            crate::discovery::LoadThresholdHandle::new(Default::default()),
            &client.endpoint.drt().child_token(),
            None,
        )
        .await
        .unwrap();
        let push_router = PushRouter::from_client_with_dispatch(client, mode, dispatch)
            .await
            .unwrap();
        let shared = Arc::new(
            RoutingHost::new_builtin_with_coordinator(
                push_router,
                load_context,
                None,
                crate::session_affinity::SessionAffinityMode::Hard,
            )
            .unwrap(),
        );
        let prefill = PrefillRouter::disabled(Arc::new(ModelManager::new()), mode, None);
        prefill.binding.store(Some(Arc::new(
            crate::kv_router::prefill_router::PrefillBinding {
                endpoint_id: dynamo_runtime::protocols::EndpointId {
                    namespace: namespace.to_string(),
                    component: component.to_string(),
                    name: endpoint_name.to_string(),
                },
                router: shared.clone(),
                prefill_router_mode: mode,
            },
        )));
        prefill.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        worker_runtimes.push(router_runtime);
        (shared, prefill, worker_runtimes, workers)
    }

    async fn tracked_binding(
        label: &str,
    ) -> (
        Arc<PrefillBinding<DefaultWorkerSelector>>,
        Arc<KvRouter<DefaultWorkerSelector>>,
    ) {
        let runtime = Runtime::from_current().unwrap();
        let distributed = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let component = distributed
            .namespace(format!(
                "prefill-reservation-{label}-{}",
                uuid::Uuid::new_v4()
            ))
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
                Some(format!("prefill-reservation-{label}")),
                false,
                None,
                None,
            )
            .await
            .unwrap(),
        );
        let push_router =
            PushRouter::<PreprocessedRequest, LlmResponse>::from_client(client, RouterMode::KV)
                .await
                .unwrap();
        let binding = Arc::new(PrefillBinding {
            endpoint_id,
            router: Arc::new(KvPushRouter::new(push_router, chooser.clone(), None).unwrap()),
            prefill_router_mode: RouterMode::KV,
        });
        (binding, chooser)
    }

    async fn tracked_prefill_router() -> (
        Arc<PrefillRouter<DefaultWorkerSelector>>,
        Arc<KvRouter<DefaultWorkerSelector>>,
    ) {
        let (binding, chooser) = tracked_binding("primary").await;
        let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
        router.binding.store(Some(binding));
        router.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        (router, chooser)
    }

    #[tokio::test]
    async fn prefill_reservation_cleans_exact_bookings_after_rebind() {
        let (router, original_chooser) = tracked_prefill_router().await;
        let reservation = router
            .reserve_prefill_worker(
                "epp-prefill/rebind",
                &[1u32; 64],
                None,
                None,
                None,
                0.0,
                0,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();

        assert!(reservation.dp_rank().is_some());
        let dropped_reservation = router
            .reserve_prefill_worker(
                "epp-prefill/drop",
                &[1u32; 64],
                None,
                None,
                None,
                0.0,
                0,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();

        let loads = original_chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(
            loads.iter().any(|load| load.potential_prefill_tokens > 0),
            "reservation did not add prefill load: {loads:?}"
        );

        let (replacement, _) = tracked_binding("replacement").await;
        router.binding.store(Some(replacement));
        reservation.release().await.unwrap();

        drop(dropped_reservation);
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let loads = original_chooser
                    .get_potential_loads(&[], None, None, None, None)
                    .await
                    .unwrap();
                if loads.iter().all(|load| load.potential_prefill_tokens == 0) {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropped reservation must release its admitted booking");
    }

    #[tokio::test]
    async fn rr_prefill_queries_and_reservations_do_not_advance_dispatch_cursor() {
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let namespace = "prefill-query-rr";
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (shared_router, prefill_router, worker_runtimes, expected_workers) = shared_router(
            &runtime,
            discovery_root.path(),
            namespace,
            RouterMode::RoundRobin,
            dispatch.clone(),
        )
        .await;

        let reservation = prefill_router
            .reserve_prefill_worker(
                "non-kv-reservation",
                &[1, 2, 3],
                None,
                None,
                None,
                0.0,
                0,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();
        assert_eq!(reservation.worker_id(), expected_workers[0]);
        assert_eq!(reservation.dp_rank(), None);
        reservation.release().await.unwrap();

        let concurrent_peeks = join_all((0..16).map(|_| query_worker(&prefill_router))).await;
        assert!(
            concurrent_peeks
                .iter()
                .all(|worker_id| *worker_id == expected_workers[0])
        );

        for expected_worker in &expected_workers {
            assert_eq!(query_worker(&prefill_router).await, *expected_worker);
            assert_eq!(query_worker(&prefill_router).await, *expected_worker);
            let mut stream = shared_router
                .generate(Context::new(request()))
                .await
                .unwrap();
            while stream.next().await.is_some() {}
        }

        assert_eq!(*dispatch.worker_ids.lock().unwrap(), expected_workers);
        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn advisory_prefill_query_never_acquires_local_occupancy() {
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();

        for (index, mode) in [
            RouterMode::PowerOfTwoChoices,
            RouterMode::LeastLoaded,
            RouterMode::DeviceAwareWeighted,
        ]
        .into_iter()
        .enumerate()
        {
            let dispatch = Arc::new(RecordingDispatch::pending());
            let (shared, prefill, runtimes, workers) = shared_router(
                &runtime,
                discovery_root.path(),
                &format!("prefill-query-occupancy-{index}"),
                mode,
                dispatch,
            )
            .await;

            let _ = join_all((0..16).map(|_| query_worker(&prefill))).await;
            assert_eq!(
                workers
                    .iter()
                    .map(|worker| shared.occupancy_for_test(*worker))
                    .sum::<u64>(),
                0,
                "{mode:?} advisory queries must not acquire occupancy"
            );

            let stream = shared.generate(Context::new(request())).await.unwrap();
            assert_eq!(
                workers
                    .iter()
                    .map(|worker| shared.occupancy_for_test(*worker))
                    .sum::<u64>(),
                1,
                "{mode:?} committed dispatch must retain exactly one lease"
            );
            drop(stream);
            assert_eq!(
                workers
                    .iter()
                    .map(|worker| shared.occupancy_for_test(*worker))
                    .sum::<u64>(),
                0,
                "{mode:?} dropping the response stream must release the lease"
            );
            drop(runtimes);
        }

        runtime.shutdown();
    }
}
