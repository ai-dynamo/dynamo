// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashSet, hash_map::Entry},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::selector::WorkerSelector;
use dynamo_kv_router::sequences::DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION;

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
    TrackedPrefillLease, TrackedPrefillLeaseState,
};
use crate::{
    kv_router::{KvRouter, sequence::SequenceError},
    local_model::runtime_config::ModelRuntimeConfig,
};

const ACTIVE_REQUEST_EXPIRY_ENV: &str = "DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS";
const TRACKED_RESERVATION_EXPIRY_GRACE: Duration = Duration::from_secs(60);

/// Keep the ownership entry at least as long as the scheduler can keep the
/// corresponding active request. The scheduler accepts a shorter environment
/// override, but retaining the small ownership entry for the default duration
/// is safer than dropping the only handle that can clean an old binding.
fn tracked_reservation_retention() -> Duration {
    let configured = std::env::var(ACTIVE_REQUEST_EXPIRY_ENV)
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION);
    configured
        .max(DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION)
        .saturating_add(TRACKED_RESERVATION_EXPIRY_GRACE)
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackedReservationActivation {
    Activated,
    CleanupRequested,
}

#[derive(Clone, Copy)]
enum CancelledReservationCleanup {
    AnyWorker,
    ExactWorker(WorkerWithDpRank),
}

/// Turns an abandoned pending admission into a tombstone. The scheduler's
/// lifecycle lease rolls the booking back; retaining this generation prevents
/// a same-ID admission from racing that rollback on the same worker.
struct PendingTrackedReservation<'a, Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    router: &'a PrefillRouter<Sel>,
    request_id: String,
    chooser: Arc<KvRouter<Sel>>,
    generation: Arc<()>,
    runtime_handle: tokio::runtime::Handle,
    armed: bool,
}

impl<Sel> PendingTrackedReservation<'_, Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl<Sel> Drop for PendingTrackedReservation<'_, Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Some(cleanup) = self.router.claim_cancelled_tracked_reservation(
            &self.request_id,
            &self.chooser,
            &self.generation,
        ) else {
            return;
        };

        let request_id = self.request_id.clone();
        let chooser = self.chooser.clone();
        drop(self.runtime_handle.spawn(async move {
            let result = match cleanup {
                CancelledReservationCleanup::AnyWorker => chooser.free(&request_id).await,
                CancelledReservationCleanup::ExactWorker(worker) => {
                    chooser.free_if_worker(&request_id, worker).await
                }
            };
            if let Err(error) = result
                && !matches!(error, SequenceError::RequestNotFound { .. })
            {
                tracing::warn!(
                    %request_id,
                    %error,
                    "Failed to roll back a cancelled tracked prefill reservation"
                );
            }
        }));
    }
}

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn prune_expired_tracked_reservations(&self) {
        let now = Instant::now();
        let retention = tracked_reservation_retention();
        self.tracked_reservations.lock().retain(|_, lease| {
            let admission_pending = matches!(
                lease.state,
                TrackedPrefillLeaseState::Pending
                    | TrackedPrefillLeaseState::Releasing {
                        admission_pending: true,
                        ..
                    }
            );
            // An in-flight admission may legitimately remain queued longer
            // than active-request expiry. Completed or abandoned generations
            // can be released once the scheduler's own expiry plus grace has
            // elapsed.
            admission_pending || now.saturating_duration_since(lease.updated_at) < retention
        });
    }

    fn begin_tracked_reservation<'a>(
        &'a self,
        request_id: &str,
        chooser: Arc<KvRouter<Sel>>,
    ) -> Result<PendingTrackedReservation<'a, Sel>> {
        self.prune_expired_tracked_reservations();
        let generation = Arc::new(());
        let runtime_handle = tokio::runtime::Handle::current();
        match self
            .tracked_reservations
            .lock()
            .entry(request_id.to_owned())
        {
            Entry::Vacant(entry) => {
                entry.insert(TrackedPrefillLease {
                    chooser: chooser.clone(),
                    generation: generation.clone(),
                    state: TrackedPrefillLeaseState::Pending,
                    updated_at: Instant::now(),
                });
                Ok(PendingTrackedReservation {
                    router: self,
                    request_id: request_id.to_owned(),
                    chooser,
                    generation,
                    runtime_handle,
                    armed: true,
                })
            }
            Entry::Occupied(_) => {
                anyhow::bail!("tracked prefill reservation {request_id:?} already exists")
            }
        }
    }

    fn tracked_reservation_matches(
        lease: &TrackedPrefillLease<Sel>,
        chooser: &Arc<KvRouter<Sel>>,
        generation: &Arc<()>,
    ) -> bool {
        Arc::ptr_eq(&lease.chooser, chooser) && Arc::ptr_eq(&lease.generation, generation)
    }

    fn remove_tracked_reservation_if_generation(
        &self,
        request_id: &str,
        chooser: &Arc<KvRouter<Sel>>,
        generation: &Arc<()>,
    ) {
        let mut reservations = self.tracked_reservations.lock();
        let should_remove = reservations
            .get(request_id)
            .is_some_and(|lease| Self::tracked_reservation_matches(lease, chooser, generation));
        if should_remove {
            reservations.remove(request_id);
        }
    }

    fn claim_cancelled_tracked_reservation(
        &self,
        request_id: &str,
        chooser: &Arc<KvRouter<Sel>>,
        generation: &Arc<()>,
    ) -> Option<CancelledReservationCleanup> {
        let mut reservations = self.tracked_reservations.lock();
        let lease = reservations.get_mut(request_id)?;
        if !Self::tracked_reservation_matches(lease, chooser, generation) {
            return None;
        }
        let cleanup = match lease.state {
            TrackedPrefillLeaseState::Pending
            | TrackedPrefillLeaseState::Releasing { worker: None, .. } => {
                lease.state = TrackedPrefillLeaseState::Releasing {
                    worker: None,
                    admission_pending: false,
                };
                CancelledReservationCleanup::AnyWorker
            }
            TrackedPrefillLeaseState::Active(worker)
            | TrackedPrefillLeaseState::Releasing {
                worker: Some(worker),
                ..
            } => {
                lease.state = TrackedPrefillLeaseState::Releasing {
                    worker: Some(worker),
                    admission_pending: false,
                };
                CancelledReservationCleanup::ExactWorker(worker)
            }
        };
        lease.updated_at = Instant::now();
        Some(cleanup)
    }

    fn finish_unbooked_tracked_reservation(
        &self,
        request_id: &str,
        chooser: &Arc<KvRouter<Sel>>,
        generation: &Arc<()>,
    ) {
        let mut reservations = self.tracked_reservations.lock();
        let should_remove = reservations.get(request_id).is_some_and(|lease| {
            Self::tracked_reservation_matches(lease, chooser, generation)
                && matches!(
                    lease.state,
                    TrackedPrefillLeaseState::Pending
                        | TrackedPrefillLeaseState::Releasing { worker: None, .. }
                )
        });
        if should_remove {
            reservations.remove(request_id);
        }
    }

    fn activate_tracked_reservation(
        &self,
        request_id: &str,
        chooser: &Arc<KvRouter<Sel>>,
        generation: &Arc<()>,
        worker: WorkerWithDpRank,
    ) -> TrackedReservationActivation {
        let mut reservations = self.tracked_reservations.lock();
        let Some(lease) = reservations.get_mut(request_id) else {
            return TrackedReservationActivation::CleanupRequested;
        };
        if !Self::tracked_reservation_matches(lease, chooser, generation) {
            return TrackedReservationActivation::CleanupRequested;
        }
        match lease.state {
            TrackedPrefillLeaseState::Pending => {
                lease.state = TrackedPrefillLeaseState::Active(worker);
                lease.updated_at = Instant::now();
                TrackedReservationActivation::Activated
            }
            TrackedPrefillLeaseState::Releasing { .. } => {
                lease.state = TrackedPrefillLeaseState::Releasing {
                    worker: Some(worker),
                    admission_pending: false,
                };
                lease.updated_at = Instant::now();
                TrackedReservationActivation::CleanupRequested
            }
            TrackedPrefillLeaseState::Active(_) => TrackedReservationActivation::CleanupRequested,
        }
    }

    async fn release_tracked_reservation(&self, request_id: &str) -> Result<()> {
        let lease = {
            let mut reservations = self.tracked_reservations.lock();
            let Some(lease) = reservations.get_mut(request_id) else {
                return Ok(());
            };
            match lease.state {
                TrackedPrefillLeaseState::Pending => {
                    lease.state = TrackedPrefillLeaseState::Releasing {
                        worker: None,
                        admission_pending: true,
                    };
                    lease.updated_at = Instant::now();
                    return Ok(());
                }
                TrackedPrefillLeaseState::Releasing {
                    worker: None,
                    admission_pending: true,
                } => return Ok(()),
                TrackedPrefillLeaseState::Active(worker)
                | TrackedPrefillLeaseState::Releasing {
                    worker: Some(worker),
                    ..
                } => {
                    lease.state = TrackedPrefillLeaseState::Releasing {
                        worker: Some(worker),
                        admission_pending: false,
                    };
                    lease.updated_at = Instant::now();
                    lease.clone()
                }
                TrackedPrefillLeaseState::Releasing {
                    worker: None,
                    admission_pending: false,
                } => return Ok(()),
            }
        };

        let worker = match lease.state {
            TrackedPrefillLeaseState::Releasing {
                worker: Some(worker),
                ..
            } => worker,
            TrackedPrefillLeaseState::Pending
            | TrackedPrefillLeaseState::Active(_)
            | TrackedPrefillLeaseState::Releasing { worker: None, .. } => {
                unreachable!("only worker-owned releases leave the registry lock")
            }
        };
        ignore_missing_request(lease.chooser.free_if_worker(request_id, worker).await)?;
        self.remove_tracked_reservation_if_generation(
            request_id,
            &lease.chooser,
            &lease.generation,
        );
        Ok(())
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

        match &binding.router {
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
        let binding = self
            .binding
            .load_full()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match &binding.router {
            InnerPrefillRouter::KvRouter(router) => {
                let chooser = router.chooser.clone();
                let mut pending = self.begin_tracked_reservation(request_id, chooser.clone())?;
                let generation = pending.generation.clone();
                let (tracked_request_id, update_states) =
                    PrefillRequestMode::Tracked(request_id).scheduler_args();
                let outcome = match chooser
                    .find_best_match_details_with_lifecycle(
                        tracked_request_id,
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
                    .await
                {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        self.finish_unbooked_tracked_reservation(request_id, &chooser, &generation);
                        pending.disarm();
                        return Err(error);
                    }
                };

                match outcome {
                    crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                        match self.activate_tracked_reservation(
                            request_id,
                            &chooser,
                            &generation,
                            worker,
                        ) {
                            TrackedReservationActivation::Activated => {
                                pending.disarm();
                                Ok(PrefillQueryOutcome::Routed {
                                    worker_id: worker.worker_id,
                                    dp_rank: Some(worker.dp_rank),
                                })
                            }
                            TrackedReservationActivation::CleanupRequested => {
                                ignore_missing_request(
                                    chooser.free_if_worker(request_id, worker).await,
                                )?;
                                self.remove_tracked_reservation_if_generation(
                                    request_id,
                                    &chooser,
                                    &generation,
                                );
                                anyhow::bail!(
                                    "tracked prefill reservation was released during admission"
                                );
                            }
                        }
                    }
                    crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                        self.finish_unbooked_tracked_reservation(request_id, &chooser, &generation);
                        pending.disarm();
                        Ok(PrefillQueryOutcome::QueueRejected { rejection })
                    }
                }
            }
            // The simple router has no scheduler state to reserve. Report the
            // typed unavailable condition so external routers safely fall back
            // to aggregated serving instead of pretending a tracked booking exists.
            InnerPrefillRouter::SimpleRouter(_) => Err(anyhow::anyhow!(PrefillError::NotActivated)),
        }
    }

    /// Release the complete prefill-side reservation for a tracked request.
    /// The decode router owns its independent booking until final completion.
    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<()> {
        self.release_tracked_reservation(request_id).await
    }

    /// Remove a tracked request from prefill scheduler bookkeeping.
    pub async fn free(&self, request_id: &str) -> Result<()> {
        self.release_tracked_reservation(request_id).await
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
    use super::*;
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex, atomic::Ordering},
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
        protocols::annotated::Annotated,
        storage::kv,
    };
    use futures::{StreamExt, future::join_all};
    use tokio::sync::watch;

    use crate::{
        discovery::ModelManager,
        kv_router::{KvPushRouter, KvRouter},
        protocols::common::{
            FinishReason,
            llm_backend::{LLMEngineOutput, PreprocessedRequest},
        },
        session_affinity::SessionAffinityPushRouter,
        worker_type::WorkerType,
    };

    type TestPrefillBinding = super::super::PrefillBinding<DefaultWorkerSelector>;
    type LlmResponse = Annotated<LLMEngineOutput>;

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
        Arc<SessionAffinityPushRouter>,
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
        let push_router = PushRouter::from_client_with_dispatch(client, mode, dispatch)
            .await
            .unwrap();
        let shared = Arc::new(SessionAffinityPushRouter::new(push_router, None, false).unwrap());
        let prefill = PrefillRouter::disabled(Arc::new(ModelManager::new()), mode, None);
        prefill.binding.store(Some(Arc::new(
            crate::kv_router::prefill_router::PrefillBinding {
                endpoint_id: dynamo_runtime::protocols::EndpointId {
                    namespace: namespace.to_string(),
                    component: component.to_string(),
                    name: endpoint_name.to_string(),
                },
                router: InnerPrefillRouter::SimpleRouter(shared.clone()),
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

    async fn make_kv_prefill_binding(worker_id: u64) -> (Arc<TestPrefillBinding>, Arc<KvRouter>) {
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
        let (_workers_tx, workers_rx) =
            watch::channel(HashMap::from([(worker_id, ModelRuntimeConfig::default())]));
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
        let binding = Arc::new(TestPrefillBinding {
            endpoint_id,
            router: InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new_with_coordinator(
                push_router,
                chooser.clone(),
                None,
            ))),
        });

        (binding, chooser)
    }

    fn install_binding(router: &PrefillRouter, binding: Arc<TestPrefillBinding>) {
        router.binding.store(Some(binding));
        router
            .lifecycle
            .store(PrefillLifecycleState::Active as u8, Ordering::Release);
    }

    async fn make_tracked_prefill_router() -> (Arc<PrefillRouter>, Arc<KvRouter>) {
        let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
        let (binding, chooser) = make_kv_prefill_binding(7).await;
        install_binding(&router, binding);

        (router, chooser)
    }

    async fn make_simple_prefill_binding() -> (Arc<TestPrefillBinding>, u64) {
        let runtime = Runtime::from_current().unwrap();
        let distributed = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let component = distributed
            .namespace(format!("simple-prefill-test-{}", uuid::Uuid::new_v4()))
            .unwrap()
            .component("workers".to_string())
            .unwrap();
        let endpoint = component.endpoint("generate");
        let endpoint_id = endpoint.id();
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let push_router =
            PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client(
                client,
                RouterMode::RoundRobin,
            )
            .await
            .unwrap();
        let simple_router = SessionAffinityPushRouter::new(push_router, None, false).unwrap();
        (
            Arc::new(TestPrefillBinding {
                endpoint_id,
                router: InnerPrefillRouter::SimpleRouter(Arc::new(simple_router)),
            }),
            worker_id,
        )
    }

    async fn current_prefill_load(chooser: &KvRouter) -> (usize, usize) {
        let loads = chooser
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert_eq!(loads.len(), 1);
        (loads[0].potential_prefill_tokens, loads[0].active_requests)
    }

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

    #[tokio::test]
    async fn pending_release_blocks_reentry_and_forces_worker_cleanup() {
        let (router, chooser) = make_tracked_prefill_router().await;
        let request_id = "release-during-admission";
        let pending = router
            .begin_tracked_reservation(request_id, chooser.clone())
            .unwrap();
        let generation = pending.generation.clone();

        router
            .release_tracked_reservation(request_id)
            .await
            .unwrap();
        assert!(matches!(
            router.tracked_reservations.lock()[request_id].state,
            TrackedPrefillLeaseState::Releasing {
                worker: None,
                admission_pending: true,
            }
        ));
        assert!(
            router
                .begin_tracked_reservation(request_id, chooser.clone())
                .is_err(),
            "a new generation must not reuse the ID while admission is unresolved"
        );

        let worker = WorkerWithDpRank::new(7, 0);
        assert_eq!(
            router.activate_tracked_reservation(request_id, &chooser, &generation, worker),
            TrackedReservationActivation::CleanupRequested
        );
        assert!(matches!(
            router.tracked_reservations.lock()[request_id].state,
            TrackedPrefillLeaseState::Releasing {
                worker: Some(selected),
                admission_pending: false,
            } if selected == worker
        ));

        // No scheduler booking is needed for this state-machine test:
        // RequestNotFound is the same idempotent completion used by retries.
        router.free(request_id).await.unwrap();
        assert!(router.tracked_reservations.lock().is_empty());
    }

    #[tokio::test]
    async fn dropping_pending_guard_rolls_back_delivered_booking() {
        let (router, chooser) = make_tracked_prefill_router().await;
        let request_id = "cancel-after-delivery";
        let pending = router
            .begin_tracked_reservation(request_id, chooser.clone())
            .unwrap();

        let token_ids = vec![1; 64];
        let outcome = chooser
            .find_best_match_details(
                Some(request_id),
                &token_ids,
                None,
                None,
                true,
                false,
                None,
                None,
                0.0,
                0,
                None,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            crate::kv_router::FindBestMatchOutcome::Routed { worker, .. }
                if worker == WorkerWithDpRank::new(7, 0)
        ));
        assert_eq!(current_prefill_load(&chooser).await, (64, 1));

        // Models cancellation after the scheduler has delivered its result but
        // before reserve_prefill_worker can promote the registry entry.
        drop(pending);
        tokio::time::timeout(Duration::from_secs(1), async {
            while current_prefill_load(&chooser).await != (0, 0) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("pending reservation drop did not roll back the scheduler booking");
        assert!(matches!(
            router.tracked_reservations.lock()[request_id].state,
            TrackedPrefillLeaseState::Releasing {
                worker: None,
                admission_pending: false,
            }
        ));
        assert!(
            router
                .begin_tracked_reservation(request_id, chooser)
                .is_err(),
            "the cancellation tombstone must block same-ID reentry"
        );
    }

    #[tokio::test]
    async fn tracked_prefill_reservation_books_active_tokens() {
        let (router, chooser) = make_tracked_prefill_router().await;
        assert_eq!(current_prefill_load(&chooser).await, (0, 0));

        let token_ids = vec![1; 64];
        let outcome = router
            .reserve_prefill_worker(
                "tracked-prefill",
                &token_ids,
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
        assert!(matches!(
            outcome,
            PrefillQueryOutcome::Routed { worker_id: 7, .. }
        ));
        assert_eq!(current_prefill_load(&chooser).await, (64, 1));

        router
            .mark_prefill_completed("tracked-prefill")
            .await
            .unwrap();
        assert_eq!(current_prefill_load(&chooser).await, (0, 0));

        router
            .mark_prefill_completed("tracked-prefill")
            .await
            .unwrap();
        router.free("tracked-prefill").await.unwrap();
        router.free("tracked-prefill").await.unwrap();
        router
            .mark_prefill_completed("tracked-prefill")
            .await
            .unwrap();
        assert_eq!(current_prefill_load(&chooser).await, (0, 0));
    }

    #[tokio::test]
    async fn tracked_cleanup_uses_original_binding_after_rebind() {
        let router = PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::KV, None);
        let (binding_a, chooser_a) = make_kv_prefill_binding(7).await;
        install_binding(&router, binding_a);

        let token_ids = vec![1; 64];
        let outcome = router
            .reserve_prefill_worker(
                "rebound-prefill",
                &token_ids,
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
        assert!(matches!(
            outcome,
            PrefillQueryOutcome::Routed { worker_id: 7, .. }
        ));
        assert_eq!(current_prefill_load(&chooser_a).await, (64, 1));

        let (binding_b, chooser_b) = make_kv_prefill_binding(8).await;
        install_binding(&router, binding_b);
        assert_eq!(current_prefill_load(&chooser_b).await, (0, 0));

        router
            .mark_prefill_completed("rebound-prefill")
            .await
            .unwrap();
        assert_eq!(current_prefill_load(&chooser_a).await, (0, 0));
        assert_eq!(current_prefill_load(&chooser_b).await, (0, 0));

        router.free("rebound-prefill").await.unwrap();
        assert_eq!(current_prefill_load(&chooser_a).await, (0, 0));
        assert_eq!(current_prefill_load(&chooser_b).await, (0, 0));
    }

    #[tokio::test]
    async fn simple_router_tracked_reservation_falls_back_without_booking() {
        let router =
            PrefillRouter::disabled(Arc::new(ModelManager::new()), RouterMode::RoundRobin, None);
        let (binding, worker_id) = make_simple_prefill_binding().await;
        install_binding(&router, binding);

        let advisory = router
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
            .unwrap();
        assert!(matches!(
            advisory,
            PrefillQueryOutcome::Routed {
                worker_id: selected,
                dp_rank: None,
            } if selected == worker_id
        ));

        let tracked_error = match router
            .reserve_prefill_worker(
                "simple-prefill",
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
        {
            Ok(_) => panic!("simple router must not report a tracked booking"),
            Err(error) => error,
        };
        assert!(matches!(
            tracked_error.downcast_ref::<PrefillError>(),
            Some(PrefillError::NotActivated)
        ));
        assert!(router.tracked_reservations.lock().is_empty());
        router
            .mark_prefill_completed("simple-prefill")
            .await
            .unwrap();
        router.free("simple-prefill").await.unwrap();
    }

    #[tokio::test]
    async fn rr_prefill_queries_do_not_advance_shared_dispatch_cursor() {
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
