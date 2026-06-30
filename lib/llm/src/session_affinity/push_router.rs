// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

#[cfg(test)]
use std::sync::Arc;

use dynamo_runtime::{
    discovery::ClaimPayloadFuture,
    pipeline::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        SingleIn, async_trait as pipeline_async_trait,
    },
    traits::DistributedRuntimeProvider,
};

use super::{
    AffinityCoordinator, AffinityTarget, LlmResponse, ResolvedAffinity,
    coordinator::{affinity_id, invalid_argument},
    explicit_target, session_final,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::timing::{RequestPhase, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
};

#[cfg(test)]
type ExactDispatchProbe = Arc<dyn Fn(&PreprocessedRequest, AffinityTarget) + Send + Sync>;

pub struct SessionAffinityPushRouter {
    inner: PushRouter<PreprocessedRequest, LlmResponse>,
    affinity: Option<AffinityCoordinator>,
    direct: bool,
    #[cfg(test)]
    exact_dispatch_probe: Option<ExactDispatchProbe>,
}

impl SessionAffinityPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, LlmResponse>,
        ttl: Option<Duration>,
        direct: bool,
    ) -> Result<Self, Error> {
        let affinity = ttl
            .map(|ttl| {
                AffinityCoordinator::new_distributed(
                    ttl,
                    inner.client.endpoint.id().to_string(),
                    inner.client.endpoint.drt().discovery(),
                )
            })
            .transpose()?;
        Ok(Self {
            inner,
            affinity,
            direct,
            #[cfg(test)]
            exact_dispatch_probe: None,
        })
    }

    fn phase(request: &PreprocessedRequest) -> RequestPhase {
        request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated)
    }

    fn record_target(request: &PreprocessedRequest, target: AffinityTarget) {
        let Some(tracker) = request.tracker.as_ref() else {
            return;
        };
        let worker_type = if tracker.phase() == RequestPhase::Prefill {
            WORKER_TYPE_PREFILL
        } else {
            WORKER_TYPE_DECODE
        };
        tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
    }

    pub fn peek_next_worker(&self) -> Option<u64> {
        self.inner.peek_next_worker()
    }

    async fn resolve_affinity(
        &self,
        session_id: &crate::protocols::common::extensions::SessionAffinityId,
        phase: RequestPhase,
        request: &PreprocessedRequest,
        request_context: &dyn AsyncEngineContext,
    ) -> Result<(ResolvedAffinity, bool), Error> {
        let affinity = self
            .affinity
            .as_ref()
            .expect("affinity acquisition requires an enabled coordinator");
        let operation = affinity
            .acquire_with_context(session_id, request_context)
            .await?;
        let resolved = operation
            .resolve(|| -> ClaimPayloadFuture<'_> {
                Box::pin(async move {
                    let target = explicit_target(request, phase)?
                        .or_else(|| {
                            self.inner
                                .peek_worker_for_request(request)
                                .map(|worker_id| AffinityTarget {
                                    worker_id,
                                    dp_rank: None,
                                })
                        })
                        .ok_or_else(|| {
                            if self.direct {
                                invalid_argument(
                                    "worker ID required to create Direct session affinity",
                                )
                            } else {
                                anyhow::anyhow!("no worker available for session affinity")
                            }
                        })?;
                    Ok(serde_json::to_value(target)?)
                })
            })
            .await?;
        let proposal_was_explicit = if resolved.was_created() {
            explicit_target(request, phase)?.is_some()
        } else {
            false
        };
        Ok((resolved, proposal_was_explicit))
    }

    /// Adapts the generic worker-only router while keeping a known rank attached
    /// to its worker through preparation and exact dispatch.
    async fn select_and_dispatch_exact_target<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        pinned_target: Option<AffinityTarget>,
        prepare: F,
    ) -> Result<(M, ManyOut<LlmResponse>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        #[cfg(test)]
        let exact_dispatch_probe = self.exact_dispatch_probe.clone();
        self.inner
            .select_and_dispatch_exact(
                request,
                pinned_target.map(|target| target.worker_id),
                move |request, worker_id| {
                    let target = pinned_target.unwrap_or(AffinityTarget {
                        worker_id,
                        dp_rank: None,
                    });
                    debug_assert_eq!(target.worker_id, worker_id);
                    let metadata = prepare(request, target)?;
                    #[cfg(test)]
                    if let Some(probe) = exact_dispatch_probe {
                        probe(request, target);
                    }
                    Ok(metadata)
                },
            )
            .await
    }

    async fn book_and_dispatch_exact_target<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        target: AffinityTarget,
        advance_round_robin: bool,
        prepare: F,
    ) -> Result<(M, ManyOut<LlmResponse>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        #[cfg(test)]
        let exact_dispatch_probe = self.exact_dispatch_probe.clone();
        self.inner
            .book_and_dispatch_exact(
                request,
                target.worker_id,
                advance_round_robin,
                move |request, worker_id| {
                    debug_assert_eq!(target.worker_id, worker_id);
                    let metadata = prepare(request, target)?;
                    #[cfg(test)]
                    if let Some(probe) = exact_dispatch_probe {
                        probe(request, target);
                    }
                    Ok(metadata)
                },
            )
            .await
    }

    pub async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<LlmResponse>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error>,
    {
        let session_id = if self.affinity.is_some() {
            affinity_id(&request)?
        } else {
            None
        };
        if !self.direct && session_id.is_none() {
            let explicit = explicit_target(&request, RequestPhase::Prefill)?;
            return self
                .select_and_dispatch_exact_target(request, explicit, prepare)
                .await;
        }
        let Some(session_id) = session_id else {
            let explicit = explicit_target(&request, RequestPhase::Prefill)?;
            let Some(target) = explicit else {
                return Err(invalid_argument(
                    "worker ID required for prefill request in Direct routing mode",
                ));
            };
            return self
                .select_and_dispatch_exact_target(request, Some(target), prepare)
                .await;
        };
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        if is_query_only {
            let bound = self
                .affinity
                .as_ref()
                .expect("affinity query requires an enabled coordinator")
                .query_target(&session_id)?;
            let selected = match bound {
                Some(target) => Some(target),
                None => explicit_target(&request, RequestPhase::Prefill)?,
            };
            return self
                .select_and_dispatch_exact_target(request, selected, move |request, target| {
                    Self::record_target(request, target);
                    prepare(request, target)
                })
                .await;
        }

        let close_on_finish = session_final(request.content());
        let request_context = request.context();
        let (resolved, proposal_was_explicit) = self
            .resolve_affinity(
                &session_id,
                RequestPhase::Prefill,
                request.content(),
                request_context.as_ref(),
            )
            .await?;
        let target = resolved.target();
        let advance_round_robin = resolved.was_created() && !proposal_was_explicit;
        let (metadata, stream) = self
            .book_and_dispatch_exact_target(
                request,
                target,
                advance_round_robin,
                move |request, target| {
                    Self::record_target(request, target);
                    prepare(request, target)
                },
            )
            .await?;
        Ok((metadata, resolved.into_stream(stream, close_on_finish)))
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
        let phase = Self::phase(&request);
        let session_id = if self.affinity.is_some() {
            affinity_id(&request)?
        } else {
            None
        };
        if !self.direct && session_id.is_none() {
            return self.inner.generate(request).await;
        }
        let Some(session_id) = session_id else {
            let explicit = explicit_target(&request, phase)?;
            let Some(target) = explicit else {
                return Err(invalid_argument(format!(
                    "worker ID required for {phase} request in Direct routing mode"
                )));
            };
            return self.inner.direct(request, target.worker_id).await;
        };

        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        if is_query_only {
            let bound = self
                .affinity
                .as_ref()
                .expect("affinity query requires an enabled coordinator")
                .query_target(&session_id)?;
            let target = match bound {
                Some(target) => Some(target),
                None => explicit_target(&request, phase)?,
            };
            let (_, stream) = self
                .select_and_dispatch_exact_target(request, target, move |request, target| {
                    if target.dp_rank.is_some() {
                        request.routing_mut().dp_rank = target.dp_rank;
                    }
                    Self::record_target(request, target);
                    Ok(())
                })
                .await?;
            return Ok(stream);
        }

        let close_on_finish = session_final(request.content());
        let request_context = request.context();
        let (resolved, proposal_was_explicit) = self
            .resolve_affinity(
                &session_id,
                phase,
                request.content(),
                request_context.as_ref(),
            )
            .await?;
        let target = resolved.target();
        let advance_round_robin = resolved.was_created() && !proposal_was_explicit;
        let (_, stream) = self
            .book_and_dispatch_exact_target(
                request,
                target,
                advance_round_robin,
                move |request, target| {
                    if target.dp_rank.is_some() {
                        request.routing_mut().dp_rank = target.dp_rank;
                    }
                    Self::record_target(request, target);
                    Ok(())
                },
            )
            .await?;
        Ok(resolved.into_stream(stream, close_on_finish))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{Context, RouterMode},
    };
    use futures::poll;

    use super::*;
    use crate::protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        preprocessor::RoutingHints,
        timing::RequestTracker,
    };

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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct DispatchObservation {
        target: AffinityTarget,
        prepared_target: AffinityTarget,
    }

    struct ExactRouterHarness {
        runtime: Runtime,
        router: SessionAffinityPushRouter,
        worker_id: u64,
        observed: Arc<Mutex<Vec<DispatchObservation>>>,
    }

    impl ExactRouterHarness {
        async fn new(namespace: &str, mode: RouterMode) -> Self {
            Self::new_for_phase(namespace, mode, RequestPhase::Prefill).await
        }

        async fn new_for_phase(namespace: &str, mode: RouterMode, phase: RequestPhase) -> Self {
            let runtime = Runtime::from_current().unwrap();
            let distributed =
                DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                    .await
                    .unwrap();
            let endpoint = distributed
                .namespace(namespace.to_string())
                .unwrap()
                .component("workers".to_string())
                .unwrap()
                .endpoint("prefill".to_string());
            let client = endpoint.client().await.unwrap();
            endpoint.register_endpoint_instance().await.unwrap();
            let worker_id = client.wait_for_instances().await.unwrap()[0].id();
            let observed = Arc::new(Mutex::new(Vec::new()));
            let direct = mode == RouterMode::Direct;
            let inner = PushRouter::from_client(client, mode).await.unwrap();
            let mut router =
                SessionAffinityPushRouter::new(inner, Some(Duration::from_secs(10)), direct)
                    .unwrap();
            let client = router.inner.client.clone();
            let probe_observed = observed.clone();
            router.exact_dispatch_probe = Some(Arc::new(move |request, target| {
                let prepared_target = explicit_target(request, phase)
                    .expect("prepared request must contain valid routing metadata")
                    .expect("prepared request must contain a routing target");
                probe_observed.lock().unwrap().push(DispatchObservation {
                    target,
                    prepared_target,
                });
                client.set_overloaded_instances(&[target.worker_id]);
            }));
            Self {
                runtime,
                router,
                worker_id,
                observed,
            }
        }

        async fn dispatch_prefill(&self, request: SingleIn<PreprocessedRequest>) {
            let error = self
                .router
                .select_and_dispatch_prefill(request, |request, target| {
                    let routing = request.routing_mut();
                    routing.prefill_worker_id = Some(target.worker_id);
                    routing.prefill_dp_rank = target.dp_rank;
                    Ok(target)
                })
                .await
                .unwrap_err();
            assert!(error.to_string().contains("overloaded"));
        }

        fn observation(&self) -> DispatchObservation {
            let observed = self.observed.lock().unwrap();
            assert_eq!(observed.len(), 1);
            observed[0]
        }

        fn shutdown(self) {
            self.runtime.shutdown();
        }
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
    async fn session_affinity_failed_dispatch_preserves_created_claim() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let client = distributed
            .namespace("session_affinity_adapters".to_string())
            .unwrap()
            .component("workers".to_string())
            .unwrap()
            .endpoint("direct")
            .client()
            .await
            .unwrap();
        let inner = PushRouter::from_client(client, RouterMode::Direct)
            .await
            .unwrap();
        let router =
            SessionAffinityPushRouter::new(inner, Some(Duration::from_secs(10)), true).unwrap();

        assert!(
            router
                .generate(affinity_request(Some(99), false))
                .await
                .is_err()
        );
        assert_eq!(affinity(&router).entry_count(), 1);

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
    async fn direct_prefill_without_session_preserves_explicit_rank_zero() {
        let harness =
            ExactRouterHarness::new("session_affinity_direct_prefill_rank", RouterMode::Direct)
                .await;
        let target = AffinityTarget {
            worker_id: harness.worker_id,
            dp_rank: Some(0),
        };
        let mut content = request(None, false);
        content.routing_mut().prefill_worker_id = Some(target.worker_id);
        content.routing_mut().prefill_dp_rank = Some(0);

        harness.dispatch_prefill(Context::new(content)).await;
        assert_eq!(
            harness.observation(),
            DispatchObservation {
                target,
                prepared_target: target,
            }
        );
        harness.shutdown();
    }

    #[tokio::test]
    async fn non_direct_prefill_without_session_preserves_explicit_rank_zero() {
        let harness = ExactRouterHarness::new(
            "session_affinity_round_robin_prefill_rank",
            RouterMode::RoundRobin,
        )
        .await;
        let target = AffinityTarget {
            worker_id: harness.worker_id,
            dp_rank: Some(0),
        };
        let mut content = request(None, false);
        content.routing_mut().prefill_worker_id = Some(target.worker_id);
        content.routing_mut().prefill_dp_rank = target.dp_rank;

        harness.dispatch_prefill(Context::new(content)).await;
        assert_eq!(
            harness.observation(),
            DispatchObservation {
                target,
                prepared_target: target,
            }
        );
        harness.shutdown();
    }

    #[tokio::test]
    async fn distributed_affinity_target_overrides_conflicting_prefill_proposal() {
        let harness = ExactRouterHarness::new(
            "session_affinity_authoritative_prefill_rank",
            RouterMode::RoundRobin,
        )
        .await;
        let authoritative = AffinityTarget {
            worker_id: harness.worker_id,
            dp_rank: Some(0),
        };
        let seed_inner =
            PushRouter::from_client(harness.router.inner.client.clone(), RouterMode::RoundRobin)
                .await
                .unwrap();
        let seed_router =
            SessionAffinityPushRouter::new(seed_inner, Some(Duration::from_secs(10)), false)
                .unwrap();
        let session_id = SessionAffinityId::new("authoritative-prefill-session");
        let resolved = affinity(&seed_router)
            .acquire(&session_id)
            .await
            .unwrap()
            .resolve(|| Box::pin(async move { Ok(serde_json::to_value(authoritative)?) }))
            .await
            .unwrap();
        assert!(resolved.was_created());
        drop(resolved);
        assert_eq!(
            affinity(&harness.router).query_target(&session_id).unwrap(),
            None
        );

        let mut content = request(None, false);
        content.routing_mut().prefill_worker_id = Some(harness.worker_id + 1);
        content.routing_mut().prefill_dp_rank = Some(7);
        let mut request = Context::new(content);
        request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_id);

        harness.dispatch_prefill(request).await;
        assert_eq!(
            harness.observation(),
            DispatchObservation {
                target: authoritative,
                prepared_target: authoritative,
            }
        );
        drop(seed_router);
        harness.shutdown();
    }

    #[tokio::test]
    async fn selected_prefill_without_rank_dispatches_with_none() {
        let harness = ExactRouterHarness::new(
            "session_affinity_selected_prefill_without_rank",
            RouterMode::RoundRobin,
        )
        .await;
        let expected = AffinityTarget {
            worker_id: harness.worker_id,
            dp_rank: None,
        };

        harness
            .dispatch_prefill(Context::new(request(None, false)))
            .await;
        assert_eq!(
            harness.observation(),
            DispatchObservation {
                target: expected,
                prepared_target: expected,
            }
        );
        harness.shutdown();
    }

    #[tokio::test]
    async fn phase_barrier_keeps_prefill_and_decode_targets_separate() {
        let prefill = ExactRouterHarness::new_for_phase(
            "session_affinity_phase_barrier_prefill",
            RouterMode::Direct,
            RequestPhase::Prefill,
        )
        .await;
        let decode = ExactRouterHarness::new_for_phase(
            "session_affinity_phase_barrier_decode",
            RouterMode::Direct,
            RequestPhase::Decode,
        )
        .await;
        let tracker = Arc::new(RequestTracker::new());
        let prefill_target = AffinityTarget {
            worker_id: prefill.worker_id,
            dp_rank: Some(0),
        };
        let decode_target = AffinityTarget {
            worker_id: decode.worker_id,
            dp_rank: Some(7),
        };
        assert_ne!(prefill_target.worker_id, decode_target.worker_id);

        let prefill_permit = tracker.set_phase(RequestPhase::Prefill).await;
        let mut decode_transition = Box::pin(tracker.set_phase(RequestPhase::Decode));
        assert!(poll!(decode_transition.as_mut()).is_pending());

        let mut prefill_content = request(None, false);
        prefill_content.tracker = Some(tracker.clone());
        prefill_content.routing_mut().prefill_worker_id = Some(prefill_target.worker_id);
        prefill_content.routing_mut().prefill_dp_rank = prefill_target.dp_rank;
        let mut prefill_request = Context::new(prefill_content);
        prefill_request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("phase-barrier-session"),
        );
        prefill.dispatch_prefill(prefill_request).await;
        assert_eq!(
            prefill.observation(),
            DispatchObservation {
                target: prefill_target,
                prepared_target: prefill_target,
            }
        );
        assert_eq!(tracker.prefill_worker_id(), Some(prefill_target.worker_id));
        assert_eq!(tracker.prefill_dp_rank(), Some(0));
        assert_eq!(tracker.decode_worker_id(), None);
        assert_eq!(tracker.decode_dp_rank(), None);

        drop(prefill_permit);
        let decode_permit = decode_transition.await;

        let mut decode_content = request(None, false);
        decode_content.tracker = Some(tracker.clone());
        decode_content.routing_mut().decode_worker_id = Some(decode_target.worker_id);
        decode_content.routing_mut().dp_rank = decode_target.dp_rank;
        let mut decode_request = Context::new(decode_content);
        decode_request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("phase-barrier-session"),
        );
        let decode_error = decode.router.generate(decode_request).await.unwrap_err();
        assert!(decode_error.to_string().contains("overloaded"));
        assert_eq!(
            decode.observation(),
            DispatchObservation {
                target: decode_target,
                prepared_target: decode_target,
            }
        );
        assert_eq!(tracker.prefill_worker_id(), Some(prefill_target.worker_id));
        assert_eq!(tracker.prefill_dp_rank(), Some(0));
        assert_eq!(tracker.decode_worker_id(), Some(decode_target.worker_id));
        assert_eq!(tracker.decode_dp_rank(), Some(7));

        drop(decode_permit);
        prefill.shutdown();
        decode.shutdown();
    }

    #[tokio::test]
    async fn session_affinity_binding_wins_before_invalid_explicit_proposal() {
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
        let unavailable_target = AffinityTarget {
            worker_id: 99,
            dp_rank: None,
        };
        let resolved = affinity(&router)
            .acquire(&session_id)
            .await
            .unwrap()
            .resolve(|| Box::pin(async move { Ok(serde_json::to_value(unavailable_target)?) }))
            .await
            .unwrap();
        drop(resolved);

        let mut request = affinity_request(None, false);
        request.routing_mut().dp_rank = Some(0);
        let error = router.generate(request).await.unwrap_err();
        assert!(!error.to_string().contains("DP rank requires"));
        assert_eq!(
            affinity(&router).query_target(&session_id).unwrap(),
            Some(AffinityTarget {
                worker_id: 99,
                dp_rank: None,
            })
        );

        runtime.shutdown();
    }
}
