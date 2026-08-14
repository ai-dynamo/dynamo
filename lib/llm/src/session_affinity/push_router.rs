// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc, time::Duration};

use dynamo_kv_router::{
    SimpleRoutingPolicy, SimpleWorkerPicker, SimpleWorkerScorer,
    selector::{CacheUnawareRequestContext, CacheUnawareWorkerSelectionPolicy},
};
use dynamo_runtime::engine::AsyncEngineContextProvider;
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{
    AsyncEngine, Error, ManyOut, PushRouter, ResolvedRoute, RouteAdmissionKind, RouteFallback,
    RoutePolicyCandidates, RoutePolicyDecision, RouteReservation, SingleIn,
    async_trait as pipeline_async_trait,
};

use super::{AffinityCoordinator, AffinityTarget, LlmResponse};
use crate::{
    lora::{
        LoraFilter,
        filtered_router::{LoadGuard, track_response},
        load_estimator::LoadEstimator,
    },
    preprocessor::PreprocessedRequest,
    protocols::common::timing::{
        RequestPhase, RequestTracker, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL,
    },
    routing_attempt::{AttemptBackend, AttemptKind, SelectionIntent},
};

pub(crate) struct SimpleAttempt {
    target: AffinityTarget,
    reservation: Option<RouteReservation>,
    exact: bool,
    allowed_fallback: Option<HashSet<u64>>,
    load_guard: Option<LoadGuard>,
    route: Option<ResolvedRoute>,
    tracker: Option<Arc<RequestTracker>>,
}

struct LoraConstraint {
    filter: Arc<LoraFilter>,
    load_estimator: Arc<LoadEstimator>,
    policy: Option<(SimpleWorkerScorer, SimpleWorkerPicker)>,
    custom_policy: Option<Arc<CacheUnawareWorkerSelectionPolicy>>,
}

pub struct SessionAffinityPushRouter {
    inner: PushRouter<PreprocessedRequest, LlmResponse>,
    policy: Option<(SimpleWorkerScorer, SimpleWorkerPicker)>,
    custom_policy: Option<Arc<CacheUnawareWorkerSelectionPolicy>>,
    affinity: Option<AffinityCoordinator>,
    direct: bool,
    lora: Option<LoraConstraint>,
}

impl SessionAffinityPushRouter {
    fn policy_for(
        mode: dynamo_runtime::pipeline::RouterMode,
    ) -> Result<Option<(SimpleWorkerScorer, SimpleWorkerPicker)>, Error> {
        use dynamo_runtime::pipeline::RouterMode;

        let policy = match mode {
            RouterMode::RoundRobin => SimpleRoutingPolicy::RoundRobin,
            RouterMode::Random => SimpleRoutingPolicy::Random,
            RouterMode::PowerOfTwoChoices => SimpleRoutingPolicy::PowerOfTwoChoices,
            RouterMode::LeastLoaded => SimpleRoutingPolicy::LeastLoaded,
            RouterMode::DeviceAwareWeighted => SimpleRoutingPolicy::DeviceAwareWeighted,
            RouterMode::Direct => return Ok(None),
            RouterMode::KV => anyhow::bail!("KV routing must use KvPushRouter"),
        };
        Ok(Some((
            SimpleWorkerScorer::new(policy),
            SimpleWorkerPicker::new(policy),
        )))
    }

    pub fn new(
        inner: PushRouter<PreprocessedRequest, LlmResponse>,
        ttl: Option<Duration>,
        direct: bool,
    ) -> Result<Self, Error> {
        let affinity = ttl.map(AffinityCoordinator::new).transpose()?;
        Self::new_with_coordinator(inner, affinity, direct)
    }

    pub(crate) fn new_with_coordinator(
        inner: PushRouter<PreprocessedRequest, LlmResponse>,
        affinity: Option<AffinityCoordinator>,
        direct: bool,
    ) -> Result<Self, Error> {
        let policy = Self::policy_for(inner.router_mode())?;
        Ok(Self {
            inner,
            policy,
            custom_policy: None,
            affinity,
            direct,
            lora: None,
        })
    }

    pub(crate) fn new_with_coordinator_and_lora(
        inner: PushRouter<PreprocessedRequest, LlmResponse>,
        affinity: Option<AffinityCoordinator>,
        direct: bool,
        lora: Option<(Arc<LoraFilter>, Arc<LoadEstimator>)>,
        custom_policy: Option<Arc<CacheUnawareWorkerSelectionPolicy>>,
        lora_custom_policy: Option<Arc<CacheUnawareWorkerSelectionPolicy>>,
    ) -> Result<Self, Error> {
        let mode = inner.router_mode();
        let policy = Self::policy_for(mode)?;
        let lora_policy = Self::policy_for(mode)?;
        Ok(Self {
            inner,
            policy,
            custom_policy,
            affinity,
            direct,
            lora: lora.map(|(filter, load_estimator)| LoraConstraint {
                filter,
                load_estimator,
                policy: lora_policy,
                custom_policy: lora_custom_policy,
            }),
        })
    }

    fn lora_candidates(
        &self,
        request: &PreprocessedRequest,
        intent: SelectionIntent,
    ) -> Result<(Option<HashSet<u64>>, Option<LoadGuard>), Error> {
        let Some(lora) = self.lora.as_ref() else {
            return Ok((None, None));
        };
        let Some(lora_name) = request
            .routing
            .as_ref()
            .and_then(|routing| routing.lora_name.as_deref())
        else {
            return Ok((None, None));
        };

        let routable = self.inner.client.instance_ids_avail();
        let replicas = lora
            .filter
            .filter_worker_ids_for_lora(Some(lora_name), &routable);
        if replicas.is_empty() {
            anyhow::bail!("No workers available after LoRA filtering (lora={lora_name})");
        }

        let free = self
            .inner
            .client
            .instance_ids_free()
            .into_iter()
            .collect::<HashSet<_>>();
        let candidates = replicas
            .into_iter()
            .filter(|worker_id| free.contains(worker_id))
            .collect::<HashSet<_>>();
        if candidates.is_empty() {
            return Err(DynamoError::builder()
                .error_type(ErrorType::ResourceExhausted)
                .message(format!(
                    "All eligible LoRA workers are overloaded (lora={lora_name})"
                ))
                .build()
                .into());
        }

        let guard = (intent == SelectionIntent::Committed)
            .then(|| LoadGuard::new(lora.load_estimator.clone(), lora_name.to_string()));
        Ok((Some(candidates), guard))
    }

    fn record_target(tracker: Option<&RequestTracker>, target: AffinityTarget) {
        let Some(tracker) = tracker else {
            return;
        };
        let worker_type = if tracker.phase() == RequestPhase::Prefill {
            WORKER_TYPE_PREFILL
        } else {
            WORKER_TYPE_DECODE
        };
        tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
    }

    fn custom_policy_decision(
        policy: &CacheUnawareWorkerSelectionPolicy,
        request: &CacheUnawareRequestContext<'_>,
        candidates: &RoutePolicyCandidates<'_>,
        allowed_worker_ids: &mut Option<HashSet<u64>>,
    ) -> Result<Option<RoutePolicyDecision>, Error> {
        let decision = policy.select(request, candidates, |index| candidates.worker_id(index))?;
        *allowed_worker_ids = Some(decision.allowed_worker_ids);
        Ok(Some(RoutePolicyDecision {
            index: decision.index,
            admission: RouteAdmissionKind::Occupancy,
        }))
    }

    #[cfg(test)]
    fn prepare_resolved_target(
        request: &mut PreprocessedRequest,
        requested: AffinityTarget,
        worker_id: u64,
    ) -> (Option<Arc<RequestTracker>>, AffinityTarget) {
        let dp_rank = requested
            .dp_rank
            .filter(|_| worker_id == requested.worker_id);
        request.routing_mut().dp_rank = dp_rank;
        (
            request.tracker.take(),
            AffinityTarget { worker_id, dp_rank },
        )
    }

    pub fn peek_next_worker(&self) -> Option<u64> {
        match self.policy.as_ref() {
            Some((scorer, picker)) => {
                self.inner
                    .peek_next_worker_with_policy(|candidates, context| {
                        picker.peek(scorer, candidates, context)
                    })
            }
            None => self.inner.peek_next_worker(),
        }
    }

    #[cfg(test)]
    pub(crate) fn occupancy_for_test(&self, worker_id: u64) -> u64 {
        self.inner.occupancy_for_test(worker_id)
    }

    pub async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<LlmResponse>), Error>
    where
        M: Send,
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error> + Send,
    {
        crate::routing_attempt::select_and_dispatch_prefill(self, request, prepare).await
    }
}

impl AttemptBackend for SessionAffinityPushRouter {
    type Attempt = SimpleAttempt;

    fn affinity(&self) -> Option<&AffinityCoordinator> {
        self.affinity.as_ref()
    }

    fn direct(&self) -> bool {
        self.direct
    }

    async fn select(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        intent: SelectionIntent,
        pinned_target: Option<AffinityTarget>,
    ) -> Result<Self::Attempt, Error> {
        let pinned_worker = pinned_target.map(|target| target.worker_id);
        let (mut allowed_fallback, load_guard) = self.lora_candidates(request.content(), intent)?;
        let native_policy = if allowed_fallback.is_some() {
            self.lora
                .as_ref()
                .and_then(|constraint| constraint.policy.as_ref())
        } else {
            self.policy.as_ref()
        };
        let custom_policy = if allowed_fallback.is_some() {
            self.lora
                .as_ref()
                .and_then(|constraint| constraint.custom_policy.as_deref())
        } else {
            self.custom_policy.as_deref()
        };
        let mut policy_allowed_workers = None;
        let custom_selection = if let Some(policy) = custom_policy {
            let session_context = request
                .agent_context
                .as_ref()
                .map(crate::kv_router::to_worker_selection_session_context);
            let routing = request.routing.as_ref();
            let engine_context = request.context();
            let custom_context = CacheUnawareRequestContext::new(
                engine_context.id(),
                request.token_ids.len(),
                intent == SelectionIntent::Advisory,
            )
            .with_session_context(session_context.as_ref())
            .with_expected_output_tokens(routing.and_then(|routing| routing.expected_output_tokens))
            .with_priority(
                routing
                    .and_then(|routing| routing.priority_jump)
                    .unwrap_or(0.0),
                routing
                    .and_then(|routing| routing.strict_priority)
                    .unwrap_or(0),
            );
            Some(match intent {
                SelectionIntent::Advisory => {
                    let target = self.inner.peek_within_policy(
                        request.content(),
                        pinned_worker,
                        allowed_fallback.as_ref(),
                        |candidates, _context| {
                            Self::custom_policy_decision(
                                policy,
                                &custom_context,
                                candidates,
                                &mut policy_allowed_workers,
                            )
                        },
                    )?;
                    (target, None)
                }
                SelectionIntent::Committed => {
                    let reservation = self
                        .inner
                        .reserve_within_policy(
                            request.content(),
                            pinned_worker,
                            allowed_fallback.as_ref(),
                            |candidates, _context| {
                                Self::custom_policy_decision(
                                    policy,
                                    &custom_context,
                                    candidates,
                                    &mut policy_allowed_workers,
                                )
                            },
                        )
                        .await?;
                    (reservation.target(), Some(reservation))
                }
            })
        } else {
            None
        };
        let (selected, reservation) = match (custom_selection, intent, native_policy) {
            (Some(selection), _, _) => selection,
            (None, SelectionIntent::Advisory, Some((scorer, picker))) => {
                let target = self.inner.peek_within_policy(
                    request.content(),
                    pinned_worker,
                    allowed_fallback.as_ref(),
                    |candidates, context| Ok(picker.peek(scorer, candidates, context)),
                )?;
                (target, None)
            }
            (None, SelectionIntent::Committed, Some((scorer, picker))) => {
                let reservation = self
                    .inner
                    .reserve_within_policy(
                        request.content(),
                        pinned_worker,
                        allowed_fallback.as_ref(),
                        |candidates, context| Ok(picker.select(scorer, candidates, context)),
                    )
                    .await?;
                (reservation.target(), Some(reservation))
            }
            (None, SelectionIntent::Advisory, None) => {
                let target = self.inner.peek_within(
                    request.content(),
                    pinned_worker,
                    allowed_fallback.as_ref(),
                )?;
                (target, None)
            }
            (None, SelectionIntent::Committed, None) => {
                let reservation = self
                    .inner
                    .reserve_within(request.content(), pinned_worker, allowed_fallback.as_ref())
                    .await?;
                (reservation.target(), Some(reservation))
            }
        };
        if policy_allowed_workers.is_some() {
            allowed_fallback = policy_allowed_workers;
        }
        let dp_rank = pinned_target
            .filter(|target| target.worker_id == selected.worker_id)
            .and_then(|target| target.dp_rank);
        Ok(SimpleAttempt {
            target: AffinityTarget::new(selected.worker_id, dp_rank),
            reservation,
            exact: crate::session_affinity::explicit_target(request, phase)?.is_some(),
            allowed_fallback,
            load_guard,
            route: None,
            tracker: None,
        })
    }

    fn observe_advisory(&self, request: &SingleIn<PreprocessedRequest>, attempt: &Self::Attempt) {
        Self::record_target(request.tracker.as_deref(), attempt.target);
    }

    async fn begin_dispatch(
        &self,
        request: &mut SingleIn<PreprocessedRequest>,
        attempt: &mut Self::Attempt,
        kind: AttemptKind,
    ) -> Result<AffinityTarget, Error> {
        let fallback = if attempt.exact || kind == AttemptKind::Prefill {
            RouteFallback::Deny
        } else if let Some(allowed) = attempt.allowed_fallback.as_ref() {
            RouteFallback::Within(allowed)
        } else {
            RouteFallback::Allow
        };
        let route = self
            .inner
            .resolve_route(attempt.target.worker_id, fallback)?;
        attempt
            .reservation
            .as_mut()
            .expect("committed simple attempt has a reservation")
            .retarget(route.target());
        let worker_id = route.target().worker_id;
        let dp_rank = attempt
            .target
            .dp_rank
            .filter(|_| worker_id == attempt.target.worker_id);
        request.routing_mut().dp_rank = dp_rank;
        let target = AffinityTarget::new(worker_id, dp_rank);
        attempt.target = target;
        attempt.route = Some(route);
        Ok(target)
    }

    fn after_prepare(
        &self,
        request: &mut SingleIn<PreprocessedRequest>,
        attempt: &mut Self::Attempt,
    ) {
        attempt.tracker = request.tracker.take();
    }

    async fn dispatch_prepared(
        &self,
        request: SingleIn<PreprocessedRequest>,
        attempt: &mut Self::Attempt,
        _kind: AttemptKind,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        let route = attempt
            .route
            .take()
            .expect("prepared simple attempt has a resolved route");
        self.inner.dispatch_resolved(request, route).await
    }

    async fn abort(&self, attempt: &mut Self::Attempt) {
        attempt.reservation.take();
        attempt.load_guard.take();
    }

    fn finish_dispatch(
        &self,
        mut attempt: Self::Attempt,
        target: AffinityTarget,
        stream: ManyOut<LlmResponse>,
    ) -> ManyOut<LlmResponse> {
        Self::record_target(attempt.tracker.as_deref(), target);
        let stream = attempt
            .reservation
            .take()
            .expect("dispatched simple attempt has a reservation")
            .into_tracked_stream(stream);
        match attempt.load_guard.take() {
            Some(guard) => track_response(stream, guard),
            None => stream,
        }
    }
}

#[pipeline_async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LlmResponse>, Error>
    for SessionAffinityPushRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        crate::routing_attempt::generate(self, request).await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use dynamo_kv_router::{
        KvRouterConfig, WorkerSelectionPolicyError,
        protocols::WorkerWithDpRank,
        selector::{
            WorkerCandidate, WorkerFilter, WorkerInputView, WorkerPicker, WorkerSelectionContext,
            WorkerSelectionPolicy,
        },
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::EventTransportKind,
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        pipeline::{Context, RouterMode},
        storage::kv::Selector,
    };
    use futures::StreamExt;

    use super::*;
    use crate::lora::{
        LoadEstimator, LoraFilter, LoraReplicaConfig, LoraRoutingTable, LoraStateTracker,
    };
    use crate::protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        preprocessor::RoutingHints,
        timing::RequestTracker,
    };
    use crate::session_affinity::AffinityAcquire;

    fn request(worker_id: Option<u64>, query_only: bool) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .annotations(if query_only {
                vec!["query_instance_id:true".to_string()]
            } else {
                Vec::new()
            })
            .routing(worker_id.map(|worker_id| RoutingHints {
                backend_instance_id: Some(worker_id),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    fn affinity_request(worker_id: Option<u64>, query_only: bool) -> SingleIn<PreprocessedRequest> {
        let mut request = Context::new(request(worker_id, query_only));
        request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("adapter-session"),
        );
        request
    }

    fn affinity(router: &SessionAffinityPushRouter) -> &AffinityCoordinator {
        router
            .affinity
            .as_ref()
            .expect("test router must enable affinity")
    }

    #[tokio::test]
    async fn constructor_rejects_kv_mode_without_panicking() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("simple_router_rejects_kv".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate");
        let inner = PushRouter::from_client(endpoint.client().await.unwrap(), RouterMode::KV)
            .await
            .unwrap();

        let error = SessionAffinityPushRouter::new(inner, None, false)
            .err()
            .expect("KV mode must be rejected");
        assert!(
            error
                .to_string()
                .contains("KV routing must use KvPushRouter")
        );

        runtime.shutdown();
    }

    #[test]
    fn direct_fallback_clears_stale_dp_rank() {
        let tracker = Arc::new(RequestTracker::new());
        let mut content = request(Some(7), false);
        content.routing_mut().dp_rank = Some(3);
        content.tracker = Some(tracker.clone());

        let (prepared_tracker, target) = SessionAffinityPushRouter::prepare_resolved_target(
            &mut content,
            AffinityTarget {
                worker_id: 7,
                dp_rank: Some(3),
            },
            8,
        );

        assert_eq!(content.routing.unwrap().dp_rank, None);
        assert_eq!(tracker.prefill_worker_id(), None);
        assert_eq!(tracker.decode_worker_id(), None);
        SessionAffinityPushRouter::record_target(prepared_tracker.as_deref(), target);
        assert_eq!(tracker.prefill_worker_id(), Some(8));
        assert_eq!(tracker.decode_worker_id(), Some(8));
    }

    #[tokio::test]
    async fn session_affinity_disabled_simple_router_has_no_coordinator() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let client = distributed
            .namespace("session_affinity_disabled".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate")
            .client()
            .await
            .unwrap();
        let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let router = SessionAffinityPushRouter::new(inner, None, false).unwrap();

        assert!(router.affinity.is_none());

        drop(router);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn advisory_query_uses_common_path_without_admission_or_dispatch() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("simple_advisory_attempt".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let inner = PushRouter::from_client(client, RouterMode::LeastLoaded)
            .await
            .unwrap();
        let router = SessionAffinityPushRouter::new(inner, None, false).unwrap();
        let tracker = Arc::new(RequestTracker::new());
        let mut content = request(None, true);
        content.tracker = Some(tracker.clone());

        let mut stream = router.generate(Context::new(content)).await.unwrap();
        let output = stream.next().await.unwrap().data.unwrap();

        assert_eq!(tracker.decode_worker_id(), Some(worker_id));
        assert_eq!(router.occupancy_for_test(worker_id), 0);
        assert!(output.routing_data.is_some());
        assert!(stream.next().await.is_none());

        runtime.shutdown();
    }

    #[tokio::test]
    async fn simple_round_robin_uses_native_picker_state() {
        async fn shared_drt(runtime: Runtime, store: &std::path::Path) -> DistributedRuntime {
            DistributedRuntime::new(
                runtime,
                DistributedConfig {
                    discovery_backend: DiscoveryBackend::KvStore(Selector::File(store.to_owned())),
                    nats_config: None,
                    request_plane: RequestPlaneMode::Tcp,
                    event_transport_kind: EventTransportKind::Zmq,
                },
            )
            .await
            .unwrap()
        }

        let runtime = Runtime::from_current().unwrap();
        let store = tempfile::tempdir().unwrap();
        let router_drt = shared_drt(runtime.clone(), store.path()).await;
        let first_worker_drt = shared_drt(runtime.clone(), store.path()).await;
        let second_worker_drt = shared_drt(runtime.clone(), store.path()).await;
        let endpoint_for = |drt: &DistributedRuntime| {
            drt.namespace("simple_native_round_robin".to_string())
                .unwrap()
                .component("workers".to_string())
                .unwrap()
                .endpoint("generate")
        };
        endpoint_for(&first_worker_drt)
            .register_endpoint_instance()
            .await
            .unwrap();
        endpoint_for(&second_worker_drt)
            .register_endpoint_instance()
            .await
            .unwrap();
        let endpoint = endpoint_for(&router_drt);
        let client = endpoint.client().await.unwrap();
        let workers = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let workers = client.instance_ids();
                if workers.len() == 2 {
                    return workers;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("both workers must be discovered");
        assert_eq!(workers.len(), 2);
        let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let router = SessionAffinityPushRouter::new(inner, None, false).unwrap();
        let request = Context::new(request(None, false));

        let advisory = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Advisory,
                None,
            )
            .await
            .unwrap();
        let first = advisory.target.worker_id;
        drop(advisory);

        let committed = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                None,
            )
            .await
            .unwrap();
        assert_eq!(committed.target.worker_id, first);
        drop(committed);

        assert_eq!(
            router.inner.peek_next_worker(),
            Some(first),
            "the compatibility runtime picker must not own normal LLM cursor state"
        );
        let second = router
            .peek_next_worker()
            .expect("native picker must expose the next worker");
        assert_ne!(second, first);

        let committed = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                None,
            )
            .await
            .unwrap();
        assert_eq!(committed.target.worker_id, second);
        drop(committed);

        runtime.shutdown();
    }

    #[tokio::test]
    async fn native_load_picker_reservation_owns_occupancy() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("simple_native_occupancy".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let inner = PushRouter::from_client(client, RouterMode::LeastLoaded)
            .await
            .unwrap();
        let router = SessionAffinityPushRouter::new(inner, None, false).unwrap();
        let request = Context::new(request(None, false));

        let committed = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                None,
            )
            .await
            .unwrap();
        assert_eq!(committed.target.worker_id, worker_id);
        assert_eq!(router.occupancy_for_test(worker_id), 1);
        drop(committed);
        assert_eq!(router.occupancy_for_test(worker_id), 0);

        runtime.shutdown();
    }

    #[tokio::test]
    async fn injected_policy_selects_non_kv_worker_and_owns_occupancy() {
        struct HighestWorkerPicker;

        impl WorkerPicker for HighestWorkerPicker {
            fn pick(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                Ok(input
                    .candidates()
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, candidate)| candidate.worker().worker_id)
                    .map(|(index, _)| index)
                    .expect("candidate"))
            }
        }

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("custom_simple_policy".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let expected = client
            .wait_for_instances()
            .await
            .unwrap()
            .iter()
            .map(|instance| instance.id())
            .max()
            .unwrap();
        let mut inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        inner.enable_policy_occupancy().await;
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            Vec::new(),
            Box::new(HighestWorkerPicker),
        )
        .into_cache_unaware()
        .unwrap();
        let router = SessionAffinityPushRouter::new_with_coordinator_and_lora(
            inner,
            None,
            false,
            None,
            Some(Arc::new(policy)),
            None,
        )
        .unwrap();
        let request = Context::new(request(None, false));

        let committed = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                None,
            )
            .await
            .unwrap();
        assert_eq!(committed.target.worker_id, expected);
        assert_eq!(router.occupancy_for_test(expected), 1);
        drop(committed);
        assert_eq!(router.occupancy_for_test(expected), 0);

        let pinned = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                Some(AffinityTarget::new(expected, None)),
            )
            .await
            .unwrap();
        assert_eq!(pinned.target.worker_id, expected);
        assert_eq!(router.occupancy_for_test(expected), 1);
        drop(pinned);
        assert_eq!(router.occupancy_for_test(expected), 0);

        runtime.shutdown();
    }

    #[tokio::test]
    async fn custom_policy_validates_pins_and_constrains_transport_fallback() {
        struct RejectWorker(u64);

        impl WorkerFilter for RejectWorker {
            fn keep(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                candidate: &WorkerCandidate,
            ) -> Result<bool, WorkerSelectionPolicyError> {
                Ok(candidate.worker().worker_id != self.0)
            }
        }

        struct HighestWorkerPicker;

        impl WorkerPicker for HighestWorkerPicker {
            fn pick(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                Ok(input
                    .candidates()
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, candidate)| candidate.worker().worker_id)
                    .map(|(index, _)| index)
                    .expect("candidate"))
            }
        }

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("custom_policy_fallback".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let real_worker = client.wait_for_instances().await.unwrap()[0].id();
        let stale_worker = real_worker.wrapping_add(1);
        let rejected_worker = real_worker.wrapping_add(2);
        client.override_instance_avail(vec![real_worker, stale_worker, rejected_worker]);

        let mut inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        inner.enable_policy_occupancy().await;
        let policy = WorkerSelectionPolicy::new_with_filters(
            KvRouterConfig::default(),
            "decode",
            vec![Box::new(RejectWorker(rejected_worker))],
            Vec::new(),
            Box::new(HighestWorkerPicker),
        )
        .into_cache_unaware()
        .unwrap();
        let router = SessionAffinityPushRouter::new_with_coordinator_and_lora(
            inner,
            None,
            false,
            None,
            Some(Arc::new(policy)),
            None,
        )
        .unwrap();
        let mut request = Context::new(request(None, false));

        assert!(
            router
                .select(
                    &request,
                    RequestPhase::Aggregated,
                    SelectionIntent::Committed,
                    Some(AffinityTarget::new(rejected_worker, None)),
                )
                .await
                .is_err(),
            "custom filters must reject an ineligible pinned target"
        );

        let mut pinned = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                Some(AffinityTarget::new(real_worker, None)),
            )
            .await
            .unwrap();
        assert_eq!(router.occupancy_for_test(real_worker), 1);
        router.abort(&mut pinned).await;
        assert_eq!(router.occupancy_for_test(real_worker), 0);

        let mut attempt = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                None,
            )
            .await
            .unwrap();
        assert_eq!(attempt.target.worker_id, stale_worker);
        let target = router
            .begin_dispatch(&mut request, &mut attempt, AttemptKind::Generate)
            .await
            .unwrap();
        assert_eq!(target.worker_id, real_worker);
        router.abort(&mut attempt).await;

        runtime.shutdown();
    }

    #[tokio::test]
    async fn lora_constraint_uses_common_advisory_and_reservation_lifecycle() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("lora_attempt_lifecycle".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let workers = client.wait_for_instances().await.unwrap();
        let allowed_worker = workers[0].id();

        let table = LoraRoutingTable::new();
        table.update_allocation(
            "adapter".to_string(),
            LoraReplicaConfig {
                lora_name: "adapter".to_string(),
                replica_factor: 1,
                replica_set: vec![WorkerWithDpRank::new(allowed_worker, 0)],
                updated_at: Instant::now(),
                is_active: true,
            },
        );
        let filter = Arc::new(LoraFilter::new(table, LoraStateTracker::new()));
        let estimator = Arc::new(LoadEstimator::new());
        let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let router = SessionAffinityPushRouter::new_with_coordinator_and_lora(
            inner,
            None,
            false,
            Some((filter, estimator.clone())),
            None,
            None,
        )
        .unwrap();
        let mut content = request(None, false);
        content.routing_mut().lora_name = Some("adapter".to_string());
        let request = Context::new(content);

        let advisory = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Advisory,
                None,
            )
            .await
            .unwrap();
        assert_eq!(advisory.target.worker_id, allowed_worker);
        assert!(estimator.get_inflight_counts().is_empty());
        drop(advisory);

        let committed = router
            .select(
                &request,
                RequestPhase::Aggregated,
                SelectionIntent::Committed,
                None,
            )
            .await
            .unwrap();
        assert_eq!(committed.target.worker_id, allowed_worker);
        assert_eq!(estimator.get_inflight_counts().get("adapter"), Some(&1));
        drop(committed);
        assert!(estimator.get_inflight_counts().is_empty());

        runtime.shutdown();
    }

    #[tokio::test]
    async fn lora_round_robin_cursor_is_independent_from_base_requests() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("lora_round_robin_cursor".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let real_worker = client.wait_for_instances().await.unwrap()[0].id();
        let workers = vec![
            real_worker,
            real_worker.wrapping_add(1),
            real_worker.wrapping_add(2),
        ];
        client.override_discovered_instances(workers.clone());
        let lora_workers = &workers[..2];

        let table = LoraRoutingTable::new();
        table.update_allocation(
            "adapter".to_string(),
            LoraReplicaConfig {
                lora_name: "adapter".to_string(),
                replica_factor: 2,
                replica_set: lora_workers
                    .iter()
                    .map(|worker_id| WorkerWithDpRank::new(*worker_id, 0))
                    .collect(),
                updated_at: Instant::now(),
                is_active: true,
            },
        );
        let filter = Arc::new(LoraFilter::new(table, LoraStateTracker::new()));
        let estimator = Arc::new(LoadEstimator::new());
        let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let router = SessionAffinityPushRouter::new_with_coordinator_and_lora(
            inner,
            None,
            false,
            Some((filter, estimator)),
            None,
            None,
        )
        .unwrap();
        let base_request = Context::new(request(None, false));
        let mut lora_content = request(None, false);
        lora_content.routing_mut().lora_name = Some("adapter".to_string());
        let lora_request = Context::new(lora_content);
        let mut selected_lora_workers = Vec::new();

        for _ in 0..2 {
            let base = router
                .select(
                    &base_request,
                    RequestPhase::Aggregated,
                    SelectionIntent::Committed,
                    None,
                )
                .await
                .unwrap();
            drop(base);

            let lora = router
                .select(
                    &lora_request,
                    RequestPhase::Aggregated,
                    SelectionIntent::Committed,
                    None,
                )
                .await
                .unwrap();
            selected_lora_workers.push(lora.target.worker_id);
            drop(lora);
        }

        assert_ne!(
            selected_lora_workers[0], selected_lora_workers[1],
            "base requests must not pin the LoRA round-robin cursor to one replica"
        );

        runtime.shutdown();
    }

    #[tokio::test]
    async fn failed_non_kv_dispatch_does_not_record_selected_worker() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let component = distributed
            .namespace("session_affinity_worker_disclosure".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap();

        for (index, mode) in [
            RouterMode::Random,
            RouterMode::RoundRobin,
            RouterMode::PowerOfTwoChoices,
            RouterMode::LeastLoaded,
            RouterMode::DeviceAwareWeighted,
            RouterMode::Direct,
        ]
        .into_iter()
        .enumerate()
        {
            let endpoint = component.endpoint(format!("mode-{index}"));
            let client = endpoint.client().await.unwrap();
            endpoint.register_endpoint_instance().await.unwrap();
            let worker_id = client.wait_for_instances().await.unwrap()[0].id();
            let inner = PushRouter::from_client(client, mode).await.unwrap();
            let router =
                SessionAffinityPushRouter::new(inner, None, mode.is_direct_routing()).unwrap();
            let tracker = Arc::new(RequestTracker::new());
            let mut content = request(mode.is_direct_routing().then_some(worker_id), false);
            content.tracker = Some(tracker.clone());

            let _ = tokio::time::timeout(
                Duration::from_millis(100),
                router.generate(Context::new(content)),
            )
            .await;

            assert_eq!(
                tracker.prefill_worker_id(),
                None,
                "{mode:?} must not disclose a worker before dispatch succeeds"
            );
            assert_eq!(
                tracker.decode_worker_id(),
                None,
                "{mode:?} must not disclose a worker before dispatch succeeds"
            );
        }

        runtime.shutdown();
    }

    #[tokio::test]
    async fn session_affinity_simple_modes_rollback_failed_initialization() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let namespace = distributed
            .namespace("session_affinity_adapters".to_string())
            .unwrap();
        let component = namespace.component("workers".to_string()).unwrap();

        for (index, mode) in [
            RouterMode::Random,
            RouterMode::RoundRobin,
            RouterMode::PowerOfTwoChoices,
            RouterMode::LeastLoaded,
            RouterMode::DeviceAwareWeighted,
            RouterMode::Direct,
        ]
        .into_iter()
        .enumerate()
        {
            let endpoint = component.endpoint(format!("mode-{index}"));
            let client = endpoint.client().await.unwrap();
            let inner = PushRouter::from_client(client, mode).await.unwrap();
            let router = SessionAffinityPushRouter::new(
                inner,
                Some(Duration::from_secs(10)),
                mode.is_direct_routing(),
            )
            .unwrap();
            let worker_id = mode.is_direct_routing().then_some(99);

            assert!(
                router
                    .generate(affinity_request(worker_id, false))
                    .await
                    .is_err()
            );
            assert_eq!(
                affinity(&router).entry_count(),
                0,
                "failed {mode:?} dispatch must release initialization"
            );
        }

        runtime.shutdown();
    }

    #[tokio::test]
    async fn session_affinity_query_and_direct_validation_do_not_create_state() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let namespace = distributed
            .namespace("session_affinity_read_only".to_string())
            .unwrap();
        let component = namespace.component("workers".to_string()).unwrap();

        let client = component
            .endpoint("query".to_string())
            .client()
            .await
            .unwrap();
        let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let router =
            SessionAffinityPushRouter::new(inner, Some(Duration::from_secs(10)), false).unwrap();
        assert!(router.generate(affinity_request(None, true)).await.is_err());
        assert_eq!(affinity(&router).entry_count(), 0);
        assert!(
            router
                .select_and_dispatch_prefill(affinity_request(None, true), |_, _| Ok(()))
                .await
                .is_err()
        );
        assert_eq!(affinity(&router).entry_count(), 0);

        let client = component
            .endpoint("direct".to_string())
            .client()
            .await
            .unwrap();
        let inner = PushRouter::from_client(client, RouterMode::Direct)
            .await
            .unwrap();
        let router =
            SessionAffinityPushRouter::new(inner, Some(Duration::from_secs(10)), true).unwrap();
        let error = router
            .generate(affinity_request(None, false))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("worker ID required"));
        assert_eq!(affinity(&router).entry_count(), 0);

        let error = router
            .generate(Context::new(request(None, false)))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("worker ID required for aggregated request in Direct routing mode")
        );
        let error = router
            .select_and_dispatch_prefill(Context::new(request(None, false)), |_, _| Ok(()))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("worker ID required for prefill request in Direct routing mode")
        );
        assert_eq!(affinity(&router).entry_count(), 0);

        runtime.shutdown();
    }

    #[tokio::test]
    async fn prefill_preparation_receives_explicit_rank_zero() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("session_affinity_prefill_target".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("prefill".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let expected = AffinityTarget {
            worker_id,
            dp_rank: Some(0),
        };

        for (mode, direct) in [(RouterMode::Direct, true), (RouterMode::RoundRobin, false)] {
            let inner = PushRouter::from_client(client.clone(), mode).await.unwrap();
            let router = SessionAffinityPushRouter::new(inner, None, direct).unwrap();
            let mut content = request(None, false);
            content.routing_mut().prefill_worker_id = Some(worker_id);
            content.routing_mut().prefill_dp_rank = Some(0);
            let tracker = Arc::new(RequestTracker::new());
            content.tracker = Some(tracker.clone());
            let mut observed = None;

            let error = router
                .select_and_dispatch_prefill(Context::new(content), |_, target| {
                    observed = Some(target);
                    Err::<(), _>(anyhow::anyhow!("stop before dispatch"))
                })
                .await
                .unwrap_err();

            assert!(error.to_string().contains("stop before dispatch"));
            assert_eq!(observed, Some(expected));
            assert_eq!(tracker.prefill_worker_id(), None);
            assert_eq!(tracker.decode_worker_id(), None);
        }

        runtime.shutdown();
    }

    #[tokio::test]
    async fn session_affinity_unavailable_target_is_invalidated() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let namespace = distributed
            .namespace("session_affinity_unavailable".to_string())
            .unwrap();
        let endpoint = namespace
            .component("workers".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let client = endpoint.client().await.unwrap();
        let inner = PushRouter::from_client(client, RouterMode::RoundRobin)
            .await
            .unwrap();
        let router =
            SessionAffinityPushRouter::new(inner, Some(Duration::from_secs(10)), false).unwrap();
        let session_id = SessionAffinityId::new("adapter-session");
        let AffinityAcquire::Initialize(initializer) =
            affinity(&router).acquire(&session_id, None).await.unwrap()
        else {
            panic!("first request must initialize");
        };
        drop(
            initializer
                .commit(AffinityTarget {
                    worker_id: 99,
                    dp_rank: None,
                })
                .unwrap(),
        );

        assert!(
            router
                .generate(affinity_request(None, false))
                .await
                .is_err()
        );
        assert_eq!(
            affinity(&router).query_target(&session_id, None).unwrap(),
            None
        );

        runtime.shutdown();
    }
}
