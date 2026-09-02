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
            FinishReason,
            llm_backend::LLMEngineOutput,
            preprocessor::PreprocessedRequest,
            timing::{RequestPhase, RequestTracker},
        },
    };
    use dynamo_runtime::pipeline::Operator;

    type LlmResponse = dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>;

    /// A decode host and its call counter. Tests that only need *a* next engine
    /// bind the counter as `_`.
    fn counting_decode_host() -> (
        Arc<AtomicUsize>,
        ServerStreamingEngine<PreprocessedRequest, LlmResponse>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let next = Arc::new(CountingDecodeHost {
            calls: calls.clone(),
        });
        (calls, next)
    }

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
        let router_endpoint = router_runtime
            .namespace(namespace.to_string())
            .unwrap()
            .component(component.to_string())
            .unwrap()
            .endpoint(endpoint_name);
        let client = router_endpoint.client().await.unwrap();
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

        let push_router = PushRouter::from_client_with_dispatch(client.clone(), mode, dispatch)
            .await
            .unwrap();
        // The two planes reach `prepare_prefill_dispatch` by different routes,
        // so the continuation path is exercised on both rather than assumed to
        // be plane-agnostic.
        let shared = if mode.is_kv_routing() {
            let kv_config = dynamo_kv_router::config::KvRouterConfig {
                skip_initial_worker_wait: true,
                use_kv_events: false,
                router_track_active_blocks: false,
                ..Default::default()
            };
            let chooser = crate::kv_router::KvRouter::new(
                router_endpoint,
                client,
                prefill_runtime_configs.clone(),
                None,
                16,
                dynamo_kv_router::selector::DefaultWorkerSelector::new(
                    Some(kv_config.clone()),
                    "prefill",
                ),
                Some(kv_config),
                None,
                "prefill",
                None,
                false,
                None,
                None,
            )
            .await
            .unwrap();
            Arc::new(RoutingHost::new(push_router, Arc::new(chooser), None).unwrap())
        } else {
            // Only this arm needs a load context, and starting one spawns a
            // worker monitor.
            let load_context = RoutingLoadContext::start(
                client.clone(),
                RouterLoadSource::Prefill,
                crate::discovery::LoadThresholdHandle::new(Default::default()),
                &client.endpoint.drt().child_token(),
                None,
            )
            .await
            .unwrap();
            Arc::new(
                RoutingHost::new_builtin_with_coordinator(push_router, load_context, None).unwrap(),
            )
        };
        // The harness must build the plane the test asked for. The built-in
        // plane cannot read the interlock at all, so a KV test that silently
        // fell back would prove the opposite of what it claims.
        assert_eq!(
            shared.kv_router_if_enabled().is_some(),
            mode.is_kv_routing(),
            "harness built the wrong routing plane for {mode:?}"
        );
        let prefill = match prefill_continue {
            Some(config) => PrefillRouter::disabled_with_prefill_continue(
                Arc::new(ModelManager::new()),
                mode,
                config,
            ),
            None => PrefillRouter::disabled(Arc::new(ModelManager::new()), mode, None),
        };
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

    /// The continuation config every continuation test shares: force-on, so
    /// the decode-load condition cannot mask what is actually under test, plus
    /// a per-test cap. `None` means the cap is deliberately absent.
    ///
    /// Nothing validates this config. It reaches `PrefillContinuePolicy` via
    /// `disabled_with_prefill_continue`, not `KvRouter::new`, which the KV arm
    /// of the harness builds its own config for.
    fn continue_config(max_concurrent: Option<usize>) -> dynamo_kv_router::config::KvRouterConfig {
        dynamo_kv_router::config::KvRouterConfig {
            prefill_continue_enabled: true,
            prefill_continue_force: true,
            prefill_continue_prefill_busy_threshold: Some(0.4),
            prefill_continue_max_concurrent: max_concurrent,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn round_robin_has_no_interlock_signal_so_nothing_continues() {
        // The interlock reads a KV route preview, so on the built-in plane it
        // cannot be read at all. An unread safety check is not a passed one:
        // the feature refuses rather than continue unchecked.
        //
        // This is the whole reason the continuation tests below are KV-routed,
        // and it is what a benchmark arm has to get right — a round-robin arm
        // with the flag set is a control arm, not a treatment arm.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-no-interlock",
            RouterMode::RoundRobin,
            dispatch.clone(),
            Some(&continue_config(Some(2))),
            PoolSupport::Unanimous,
        )
        .await;

        let (_, next) = counting_decode_host();
        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);

        let _ = Operator::generate(prefill_router.as_ref(), Context::new(request), next).await;

        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(1)],
            "with no readable interlock the request must take today's handoff"
        );
        assert_eq!(
            *dispatch.dispatched_annotations.lock().unwrap(),
            vec![Vec::<String>::new()],
            "and the marker must never reach the worker"
        );

        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn without_a_readable_decode_pool_nothing_continues() {
        // The other continuation tests set `force`, which waives the
        // decode-load test and nothing else. Drop it and the decision has to
        // stand on a real decode signal — which this harness has no decode
        // routing host to provide. Unknown decode load is not an empty decode
        // pool, so the request hands off.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-no-decode-signal",
            RouterMode::KV,
            dispatch.clone(),
            Some(&dynamo_kv_router::config::KvRouterConfig {
                prefill_continue_force: false,
                prefill_continue_decode_busy_threshold: Some(0.9),
                ..continue_config(Some(2))
            }),
            PoolSupport::Unanimous,
        )
        .await;

        let (_, next) = counting_decode_host();
        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);

        let _ = Operator::generate(prefill_router.as_ref(), Context::new(request), next).await;

        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(1)],
            "an unreadable decode pool must not read as a full one"
        );

        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn a_continuation_is_forwarded_whole() {
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-kv",
            RouterMode::KV,
            dispatch.clone(),
            Some(&continue_config(Some(2))),
            PoolSupport::Unanimous,
        )
        .await;

        let (decode_calls, next) = counting_decode_host();

        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);
        let tracker = Arc::new(RequestTracker::new());
        request.tracker = Some(tracker.clone());

        let mut response = Operator::generate(prefill_router.as_ref(), Context::new(request), next)
            .await
            .expect("a continuation should be served by the prefill worker");

        let worker_id = dispatch.worker_ids.lock().unwrap()[0];
        assert_eq!(
            prefill_router.continuations.in_flight(worker_id),
            1,
            "the census must count a KV-routed continuation the same way"
        );

        while response.next().await.is_some() {}

        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(256)],
            "a KV-routed continuation keeps the request's own budget"
        );
        assert_eq!(
            decode_calls.load(Ordering::Relaxed),
            0,
            "a continuation must not dispatch a decode leg on either plane"
        );
        assert_eq!(
            prefill_router.continuations.in_flight(worker_id),
            0,
            "and gives its place back at the end of the stream"
        );

        // Finish-time gauges key on a recorded decode worker, and a request
        // that stays in the prefill phase never records one — so without this
        // transition inter-token latency is missing from exactly the arms
        // running the feature.
        assert_eq!(tracker.phase(), RequestPhase::Continuation);
        assert_eq!(
            tracker.decode_worker_id(),
            Some(worker_id),
            "the generating worker must be recorded as the decode worker"
        );
        assert_eq!(
            tracker.prefill_worker_id(),
            Some(worker_id),
            "and as the prefill worker, because it did both"
        );

        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn a_capped_worker_is_demoted() {
        // The dispatch-time demotion only fires when the pool looks free but
        // the *chosen* worker is not. A cap of zero cannot reach it — that
        // refuses before routing — and neither can a busy worker the scheduler
        // then avoids, because the scheduler balances load.
        //
        // The real shape is a census the scheduler cannot see: fill three of
        // the four workers, leave one free so the pool minimum still passes the
        // pre-routing check, then send one request per worker. Whichever ones
        // land on a filled worker must be demoted.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-kv-capped",
            RouterMode::KV,
            dispatch.clone(),
            Some(&continue_config(Some(1))),
            PoolSupport::Unanimous,
        )
        .await;

        let filled: Vec<_> = workers[..workers.len() - 1]
            .iter()
            .map(|worker_id| {
                prefill_router
                    .continuations
                    .try_admit(*worker_id, 1)
                    .expect("a fresh worker has its place free")
            })
            .collect();
        assert_eq!(
            prefill_router.continuations.min_in_flight(&workers),
            Some(0),
            "one worker must stay free, or the pre-routing check refuses instead"
        );

        for _ in 0..workers.len() {
            let mut request = request();
            request.stop_conditions.max_tokens = Some(256);
            let (_, next) = counting_decode_host();
            let _ = Operator::generate(prefill_router.as_ref(), Context::new(request), next).await;
        }

        // Pair each dispatch with the worker it went to. Counting demotions
        // would assume the scheduler spreads one request per worker, and it
        // does not.
        let dispatched = dispatch.dispatched_max_tokens.lock().unwrap().clone();
        let landed = dispatch.worker_ids.lock().unwrap().clone();
        let full: std::collections::HashSet<u64> =
            workers[..workers.len() - 1].iter().copied().collect();
        let mut demotions = 0;
        for (worker_id, max_tokens) in landed.iter().zip(dispatched.iter()) {
            if full.contains(worker_id) {
                assert_eq!(
                    *max_tokens,
                    Some(1),
                    "a request on filled worker {worker_id} must be demoted"
                );
                demotions += 1;
            } else {
                assert_eq!(
                    *max_tokens,
                    Some(256),
                    "a request on the free worker must continue"
                );
            }
        }
        assert!(
            demotions > 0,
            "the scheduler never chose a filled worker, so nothing was tested: {landed:?}"
        );

        drop(filled);
        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn a_mixed_pool_refuses_to_continue() {
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-kv-mixed",
            RouterMode::KV,
            dispatch.clone(),
            Some(&continue_config(Some(2))),
            PoolSupport::AllButOne,
        )
        .await;

        let (_, next) = counting_decode_host();

        let mut request = request();
        request.stop_conditions.max_tokens = Some(256);

        let _ = Operator::generate(prefill_router.as_ref(), Context::new(request), next).await;

        assert_eq!(
            *dispatch.dispatched_max_tokens.lock().unwrap(),
            vec![Some(1)],
            "a pool that did not unanimously declare support must hand off"
        );

        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn without_a_configured_cap_nothing_continues() {
        // Startup validation asks for a cap, but it does not run on every path
        // a router can be built from, so a config can reach dispatch with none.
        // An unbounded continuation is worse than no continuation, so the
        // absent cap must refuse rather than waive the bound.
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (_shared, prefill_router, worker_runtimes, _workers) = shared_router(
            &runtime,
            discovery_root.path(),
            "prefill-continuation-uncapped",
            RouterMode::KV,
            dispatch.clone(),
            // The cap is deliberately absent.
            Some(&continue_config(None)),
            PoolSupport::Unanimous,
        )
        .await;

        let (_, next) = counting_decode_host();

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
