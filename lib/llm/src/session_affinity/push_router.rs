// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

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

pub struct SessionAffinityPushRouter {
    inner: PushRouter<PreprocessedRequest, LlmResponse>,
    affinity: Option<AffinityCoordinator>,
    direct: bool,
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
                    prepare(request, target)
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
        self.inner
            .book_and_dispatch_exact(
                request,
                target.worker_id,
                advance_round_robin,
                move |request, worker_id| {
                    debug_assert_eq!(target.worker_id, worker_id);
                    prepare(request, target)
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
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{Context, RouterMode},
    };

    use super::*;
    use crate::protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        preprocessor::RoutingHints,
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
        }

        runtime.shutdown();
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
