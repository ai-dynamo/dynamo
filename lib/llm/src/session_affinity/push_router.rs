// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_runtime::pipeline::{
    AsyncEngine, Error, ManyOut, PushRouter, SingleIn, async_trait as pipeline_async_trait,
};

use super::{
    AffinityCoordinator, AffinityTarget, LlmResponse,
    coordinator::{affinity_id, invalid_argument},
    explicit_target,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::timing::{RequestPhase, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
};

pub struct SessionAffinityPushRouter {
    inner: PushRouter<PreprocessedRequest, LlmResponse>,
    affinity: AffinityCoordinator,
    direct: bool,
}

impl SessionAffinityPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, LlmResponse>,
        ttl: Duration,
        direct: bool,
    ) -> Result<Self, Error> {
        Ok(Self {
            inner,
            affinity: AffinityCoordinator::new(ttl)?,
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

    fn direct_target(
        &self,
        explicit: Option<AffinityTarget>,
        phase: RequestPhase,
    ) -> Result<Option<AffinityTarget>, Error> {
        if !self.direct {
            return Ok(explicit);
        }
        explicit.map(Some).ok_or_else(|| {
            invalid_argument(format!(
                "worker ID required for {phase} request in Direct routing mode"
            ))
        })
    }

    pub fn peek_next_worker(&self) -> Option<u64> {
        self.inner.peek_next_worker()
    }

    pub async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<LlmResponse>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, u64, Option<u32>) -> Result<M, Error>,
    {
        let Some(session_id) = affinity_id(&request) else {
            let pinned_worker = phase_worker_id(&request, RequestPhase::Prefill);
            return self
                .inner
                .select_and_dispatch_exact(request, pinned_worker, move |request, worker_id| {
                    prepare(request, worker_id, None)
                })
                .await;
        };
        let explicit = self.direct_target(
            explicit_target(&request, RequestPhase::Prefill)?,
            RequestPhase::Prefill,
        )?;
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        if is_query_only {
            let selected = self
                .affinity
                .query_target(&session_id, explicit)?
                .or(explicit);
            let rank = selected.and_then(|target| target.dp_rank);
            return self
                .inner
                .select_and_dispatch_exact(
                    request,
                    selected.map(|target| target.worker_id),
                    move |request, worker_id| {
                        let target = AffinityTarget {
                            worker_id,
                            dp_rank: rank,
                        };
                        Self::record_target(request, target);
                        prepare(request, worker_id, rank)
                    },
                )
                .await;
        }

        let operation = self.affinity.acquire(&session_id, explicit).await?;
        let selected = operation.target().or(explicit);
        let rank = selected.and_then(|target| target.dp_rank);
        let ((metadata, target), stream) = self
            .inner
            .select_and_dispatch_exact(
                request,
                selected.map(|target| target.worker_id),
                move |request, worker_id| {
                    let target = AffinityTarget {
                        worker_id,
                        dp_rank: rank,
                    };
                    Self::record_target(request, target);
                    Ok((prepare(request, worker_id, rank)?, target))
                },
            )
            .await?;
        Ok((metadata, operation.into_stream(target, stream)?))
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
        let Some(session_id) = affinity_id(&request) else {
            if self.direct {
                let worker_id = phase_worker_id(&request, phase).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Worker ID required (--direct-route) but none found in request. Expected \
                         a phase-appropriate worker ID to be set by the external router."
                    )
                })?;
                return self.inner.direct(request, worker_id).await;
            }
            return self.inner.generate(request).await;
        };
        let explicit = self.direct_target(explicit_target(&request, phase)?, phase)?;

        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        if is_query_only {
            let target = self
                .affinity
                .query_target(&session_id, explicit)?
                .or(explicit);
            let rank = target.and_then(|target| target.dp_rank);
            let (_, stream) = self
                .inner
                .select_and_dispatch_exact(
                    request,
                    target.map(|target| target.worker_id),
                    move |request, worker_id| {
                        if rank.is_some() {
                            request.routing_mut().dp_rank = rank;
                        }
                        Self::record_target(
                            request,
                            AffinityTarget {
                                worker_id,
                                dp_rank: rank,
                            },
                        );
                        Ok(())
                    },
                )
                .await?;
            return Ok(stream);
        }

        let operation = self.affinity.acquire(&session_id, explicit).await?;
        let selected = operation.target().or(explicit);
        let rank = selected.and_then(|target| target.dp_rank);
        let (target, stream) = self
            .inner
            .select_and_dispatch_exact(
                request,
                selected.map(|target| target.worker_id),
                move |request, worker_id| {
                    if rank.is_some() {
                        request.routing_mut().dp_rank = rank;
                    }
                    let target = AffinityTarget {
                        worker_id,
                        dp_rank: rank,
                    };
                    Self::record_target(request, target);
                    Ok(target)
                },
            )
            .await?;
        operation.into_stream(target, stream)
    }
}

fn phase_worker_id(request: &PreprocessedRequest, phase: RequestPhase) -> Option<u64> {
    let routing = request.routing.as_ref()?;
    match phase {
        RequestPhase::Prefill => routing.prefill_worker_id.or(routing.backend_instance_id),
        RequestPhase::Decode => routing.decode_worker_id.or(routing.backend_instance_id),
        RequestPhase::Aggregated => routing.backend_instance_id,
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
    use crate::session_affinity::AffinityAcquire;

    fn request(worker_id: Option<u64>, query_only: bool) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .annotations(
                query_only
                    .then(|| vec!["query_instance_id:true".to_string()])
                    .unwrap_or_default(),
            )
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
                Duration::from_secs(10),
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
                router.affinity.entry_count(),
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
        let router = SessionAffinityPushRouter::new(inner, Duration::from_secs(10), false).unwrap();
        assert!(router.generate(affinity_request(None, true)).await.is_err());
        assert_eq!(router.affinity.entry_count(), 0);
        assert!(
            router
                .select_and_dispatch_prefill(affinity_request(None, true), |_, _, _| Ok(()))
                .await
                .is_err()
        );
        assert_eq!(router.affinity.entry_count(), 0);

        let client = component
            .endpoint("direct".to_string())
            .client()
            .await
            .unwrap();
        let inner = PushRouter::from_client(client, RouterMode::Direct)
            .await
            .unwrap();
        let router = SessionAffinityPushRouter::new(inner, Duration::from_secs(10), true).unwrap();
        let error = router
            .generate(affinity_request(None, false))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("worker ID required"));
        assert_eq!(router.affinity.entry_count(), 0);

        runtime.shutdown();
    }

    #[tokio::test]
    async fn session_affinity_unavailable_target_fails_without_rebinding() {
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
        let router = SessionAffinityPushRouter::new(inner, Duration::from_secs(10), false).unwrap();
        let session_id = SessionAffinityId::new("adapter-session");
        let AffinityAcquire::Initialize(initializer) =
            router.affinity.acquire(&session_id, None).await.unwrap()
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
            router.affinity.query_target(&session_id, None).unwrap(),
            Some(AffinityTarget {
                worker_id: 99,
                dp_rank: None,
            })
        );

        runtime.shutdown();
    }
}
