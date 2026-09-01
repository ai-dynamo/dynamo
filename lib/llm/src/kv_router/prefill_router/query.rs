// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId};
use dynamo_kv_router::selector::WorkerSelector;

use super::{PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter};
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
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use async_trait::async_trait;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::Instance,
        discovery::{EndpointInstanceId, EventTransportKind},
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        engine::{AsyncEngine, AsyncEngineContext},
        pipeline::{
            AddressedRequest, Context, Error, ManyIn, ManyOut, PushRouter, ResponseStream,
            RouterMode, ServerStreamingEngine, SingleIn, StreamingDispatch, context::Controller,
        },
        storage::kv,
        traits::DistributedRuntimeProvider,
    };
    use futures::{StreamExt, future::join_all};

    use super::*;
    use crate::{
        discovery::ModelManager,
        kv_router::{RouterLoadSource, RoutingHost, RoutingLoadContext},
        local_model::runtime_config::PREFILL_CONTINUE_CAPABILITY,
        protocols::common::{
            FinishReason, llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest,
        },
    };
    use dynamo_runtime::pipeline::Operator;

    type LlmResponse = dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>;

    #[derive(Default)]
    struct RecordingDispatch {
        worker_ids: Mutex<Vec<u64>>,
        pending_responses: bool,
        /// `stop_conditions.max_tokens` as it reached the worker, per dispatch.
        dispatched_max_tokens: Mutex<Vec<Option<u32>>>,
        /// Annotations as they reached the worker, per dispatch.
        dispatched_annotations: Mutex<Vec<Vec<String>>>,
    }

    impl RecordingDispatch {
        fn completed() -> Self {
            Self::default()
        }

        fn pending() -> Self {
            Self {
                pending_responses: true,
                ..Self::default()
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
            let (payload, _, instance) = addressed.into_parts();
            self.dispatched_max_tokens
                .lock()
                .unwrap()
                .push(payload.stop_conditions.max_tokens);
            self.dispatched_annotations
                .lock()
                .unwrap()
                .push(payload.annotations.clone());
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

    struct CountingDecodeHost {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LlmResponse>, Error>
        for CountingDecodeHost
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<LlmResponse>, Error> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let ctx: Arc<dyn AsyncEngineContext> = Arc::new(Controller::default());
            let _ = request;
            Ok(ResponseStream::new(Box::pin(futures::stream::empty()), ctx))
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

    /// How much of the prefill pool declares the continuation capability.
    #[derive(Clone, Copy)]
    enum PoolSupport {
        /// Every worker declares it, so the gate opens.
        Unanimous,
        /// All but one, so the gate must refuse the whole pool.
        AllButOne,
        /// Nobody declares it; for the tests that do not exercise the gate.
        None,
    }

    async fn shared_router(
        runtime: &Runtime,
        discovery_root: &std::path::Path,
        namespace: &str,
        mode: RouterMode,
        dispatch: Arc<RecordingDispatch>,
        prefill_continue: Option<&dynamo_kv_router::config::KvRouterConfig>,
        pool_support: PoolSupport,
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
            RoutingHost::new_builtin_with_coordinator(push_router, load_context, None).unwrap(),
        );
        let prefill = match prefill_continue {
            Some(config) => PrefillRouter::disabled_with_prefill_continue(
                Arc::new(ModelManager::new()),
                mode,
                config,
            ),
            None => PrefillRouter::disabled(Arc::new(ModelManager::new()), mode, None),
        };
        // Derived from the pool that was actually discovered, so a change to
        // the harness's worker count cannot quietly turn a unanimous pool into
        // a mixed one.
        let capable_workers = match pool_support {
            PoolSupport::Unanimous => workers.len(),
            PoolSupport::AllButOne => workers.len() - 1,
            PoolSupport::None => 0,
        };
        let runtime_configs = workers
            .iter()
            .enumerate()
            .map(|(index, worker_id)| {
                let mut config = ModelRuntimeConfig::default();
                if index < capable_workers {
                    config
                        .set_engine_specific(PREFILL_CONTINUE_CAPABILITY, true)
                        .unwrap();
                }
                (*worker_id, config)
            })
            .collect();
        // The sender is dropped immediately: a watch receiver keeps reading the
        // last published value, and no test republishes.
        let (_, prefill_runtime_configs) = tokio::sync::watch::channel(runtime_configs);
        prefill.binding.store(Some(Arc::new(
            crate::kv_router::prefill_router::PrefillBinding {
                endpoint_id: dynamo_runtime::protocols::EndpointId {
                    namespace: namespace.to_string(),
                    component: component.to_string(),
                    name: endpoint_name.to_string(),
                },
                router: shared.clone(),
                prefill_router_mode: mode,
                prefill_runtime_configs,
            },
        )));
        prefill.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        worker_runtimes.push(router_runtime);
        (shared, prefill, worker_runtimes, workers)
    }

    #[tokio::test]
    async fn a_continuation_is_forwarded_whole_and_dispatches_no_decode_leg() {
        // The whole of T5 in one test: a marked request is served by the prefill
        // worker, its stream reaches the client untouched, no decode worker is
        // asked for anything, and the request keeps its own token budget.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation",
            RouterMode::RoundRobin,
            dispatch.clone(),
            Some(&dynamo_kv_router::config::KvRouterConfig {
                prefill_continue_enabled: true,
                prefill_continue_force: true,
                prefill_continue_prefill_busy_threshold: Some(0.4),
                prefill_continue_max_concurrent: Some(2),
                ..Default::default()
            }),
            PoolSupport::Unanimous,
        )
        .await;

        let decode_calls = Arc::new(AtomicUsize::new(0));
        let next: ServerStreamingEngine<PreprocessedRequest, LlmResponse> =
            Arc::new(CountingDecodeHost {
                calls: decode_calls.clone(),
            });

        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);
        request
            .annotations
            .push(crate::kv_router::prefill_router::PREFILL_CONTINUE_ANNOTATION.to_string());

        let mut response = Operator::generate(prefill_router.as_ref(), Context::new(request), next)
            .await
            .expect("a continuation should be served by the prefill worker");

        let worker_id = dispatch.worker_ids.lock().unwrap()[0];
        assert_eq!(
            prefill_router.continuations.in_flight(worker_id),
            1,
            "the continuation must hold its place for as long as the stream lives"
        );

        let mut chunks = 0;
        while response.next().await.is_some() {
            chunks += 1;
        }

        // Still holding the stream, and the place is already back: it is
        // returned when the worker finishes, not when the client lets go.
        assert_eq!(
            prefill_router.continuations.in_flight(worker_id),
            0,
            "the place must come back at the end of the stream"
        );

        // the prefill worker's stream reached the client
        assert_eq!(
            chunks, 1,
            "the prefill stream must be forwarded, not drained"
        );
        // no decode leg was dispatched
        assert_eq!(
            decode_calls.load(Ordering::Relaxed),
            0,
            "a continuation must not dispatch a decode leg"
        );
        // and the worker was asked for the whole response, not one token
        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(256)],
            "a continuation keeps the request's own budget"
        );

        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn without_a_configured_cap_nothing_continues() {
        // Startup validation asks for a cap, but it only runs when the decode
        // set is KV-routed — this router is round-robin, so it reaches dispatch
        // with none. An unbounded continuation is worse than no continuation,
        // so the absent cap must refuse rather than waive the bound.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-uncapped",
            RouterMode::RoundRobin,
            dispatch.clone(),
            Some(&dynamo_kv_router::config::KvRouterConfig {
                prefill_continue_enabled: true,
                prefill_continue_force: true,
                prefill_continue_prefill_busy_threshold: Some(0.4),
                // Deliberately absent.
                prefill_continue_max_concurrent: None,
                ..Default::default()
            }),
            PoolSupport::Unanimous,
        )
        .await;

        let decode_calls = Arc::new(AtomicUsize::new(0));
        let next: ServerStreamingEngine<PreprocessedRequest, LlmResponse> =
            Arc::new(CountingDecodeHost {
                calls: decode_calls.clone(),
            });

        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);

        let _ = Operator::generate(prefill_router.as_ref(), Context::new(request), next).await;

        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(1)],
            "an uncapped router must hand off, not continue without a bound"
        );
        assert_eq!(
            *dispatch.dispatched_annotations.lock().unwrap(),
            vec![Vec::<String>::new()],
            "and the marker must not reach the worker"
        );

        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn a_worker_at_its_continuation_cap_is_demoted_to_a_handoff() {
        // Every worker declares support and the override is on, so the request
        // is marked before routing. The cap is what stops it, and the cap can
        // only be read once the worker is chosen — so this exercises the
        // dispatch-time demotion, not the pool check.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-capped",
            RouterMode::RoundRobin,
            dispatch.clone(),
            Some(&dynamo_kv_router::config::KvRouterConfig {
                prefill_continue_enabled: true,
                prefill_continue_force: true,
                prefill_continue_prefill_busy_threshold: Some(0.4),
                // No place for any continuation at all.
                prefill_continue_max_concurrent: Some(0),
                ..Default::default()
            }),
            PoolSupport::Unanimous,
        )
        .await;

        let decode_calls = Arc::new(AtomicUsize::new(0));
        let next: ServerStreamingEngine<PreprocessedRequest, LlmResponse> =
            Arc::new(CountingDecodeHost {
                calls: decode_calls.clone(),
            });

        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);

        let _ = Operator::generate(prefill_router.as_ref(), Context::new(request), next).await;

        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(1)],
            "a demoted request must carry the one-token clamp again"
        );
        let annotations = dispatch.dispatched_annotations.lock().unwrap().clone();
        assert_eq!(
            annotations,
            vec![Vec::<String>::new()],
            "the marker must come off, or the worker generates a whole response nothing returns"
        );
        let worker_id = dispatch.worker_ids.lock().unwrap()[0];
        assert_eq!(
            prefill_router.continuations.in_flight(worker_id),
            0,
            "a demoted request must hold no place in the census"
        );

        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn a_mixed_pool_refuses_to_continue_and_hands_off_as_today() {
        // Three of four workers declare the capability. The fourth would answer
        // a marked request with a handoff message and pin cache, so the gate
        // must refuse the whole pool rather than gamble on the selection.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-mixed",
            RouterMode::RoundRobin,
            dispatch.clone(),
            Some(&dynamo_kv_router::config::KvRouterConfig {
                prefill_continue_enabled: true,
                prefill_continue_force: true,
                prefill_continue_prefill_busy_threshold: Some(0.4),
                prefill_continue_max_concurrent: Some(2),
                ..Default::default()
            }),
            PoolSupport::AllButOne,
        )
        .await;

        let decode_calls = Arc::new(AtomicUsize::new(0));
        let next: ServerStreamingEngine<PreprocessedRequest, LlmResponse> =
            Arc::new(CountingDecodeHost {
                calls: decode_calls.clone(),
            });

        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);

        let _ = Operator::generate(prefill_router.as_ref(), Context::new(request), next).await;

        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(1)],
            "a pool that did not unanimously declare support gets today's one-token prefill"
        );

        drop(worker_runtimes);
        runtime.shutdown();
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
            None,
            PoolSupport::None,
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
                None,
                PoolSupport::None,
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
