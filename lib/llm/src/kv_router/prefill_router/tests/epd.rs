// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::kv_router::encoder_router::EncoderResultReleaseSupported;
use crate::local_model::runtime_config::DISAGG_PREFILL_CANCEL_ANYTIME_V1;
use crate::protocols::common::preprocessor::ENCODER_RESULT_RELEASE_SUPPORTED;
use crate::protocols::common::preprocessor::MultimodalData;

#[derive(Default)]
struct RecordingDecodeDispatch {
    annotations: Mutex<Vec<Vec<String>>>,
    phases: Mutex<Vec<RequestPhase>>,
    contexts: Mutex<Vec<Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>>>,
    gate: Mutex<Option<Arc<tokio::sync::Notify>>>,
}

#[async_trait]
impl dynamo_runtime::pipeline::StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>
    for RecordingDecodeDispatch
{
    async fn generate(
        &self,
        request: SingleIn<dynamo_runtime::pipeline::AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (request, _, _) = addressed.into_parts();
        self.annotations.lock().push(request.annotations.clone());
        self.phases.lock().push(request.phase());
        self.contexts.lock().push(context.context());
        let gate = self.gate.lock().clone();
        if let Some(gate) = gate {
            gate.notified().await;
        }
        let output = Annotated::from_data(LLMEngineOutput {
            token_ids: vec![7],
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        });
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { output })),
            context.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _instance: dynamo_runtime::component::Instance,
        _address: String,
        _input: dynamo_runtime::pipeline::ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("the routing host dispatches unary requests")
    }
}

fn conditional_router() -> Arc<PrefillRouter> {
    conditional_router_with(
        KvRouterConfig {
            conditional_disagg_enabled: true,
            ..Default::default()
        },
        false,
    )
}

fn conditional_router_with(
    config: KvRouterConfig,
    supports_cancellation: bool,
) -> Arc<PrefillRouter> {
    let (_activation_tx, activation_rx) = oneshot::channel();
    let router = PrefillRouter::new_with_selection_policy(
        Some(activation_rx),
        Arc::new(ModelManager::new()),
        RouterMode::KV,
        supports_cancellation,
        16,
        Some(config),
        SelectionPolicySource::Registry,
        None,
        None,
        SessionAffinityMode::Hard,
        "model".to_string(),
        "namespace".to_string(),
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        CancellationToken::new(),
        None,
    );
    assert!(router.conditional_disagg_enabled());
    router
}

pub(super) async fn local_prefill_binding(namespace: &str) -> Arc<PrefillBinding> {
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    let distributed = DistributedRuntime::new(
        Runtime::from_current().unwrap(),
        DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let endpoint = distributed
        .namespace(namespace)
        .unwrap()
        .component("prefill")
        .unwrap()
        .endpoint("generate");
    let client = endpoint.client().await.unwrap();
    let load_context = crate::kv_router::RoutingLoadContext::start(
        client.clone(),
        crate::kv_router::RouterLoadSource::Prefill,
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        &distributed.child_token(),
        None,
    )
    .await
    .unwrap();
    let host = RoutingHost::builder(client, RouterMode::RoundRobin)
        .load_context(load_context)
        .build()
        .await
        .unwrap();
    Arc::new(PrefillBinding {
        target_id: WorkerSetTargetId::Legacy(endpoint.id()),
        endpoint_id: endpoint.id(),
        router: Arc::new(host),
        prefill_router_mode: RouterMode::RoundRobin,
        encoder_result_requires_prefill: false,
    })
}

fn fresh_decode_router_with_mode(mode: RouterMode) -> Arc<PrefillRouter> {
    let (_activation_tx, activation_rx) = oneshot::channel();
    PrefillRouter::new(
        activation_rx,
        Arc::new(ModelManager::new()),
        mode,
        16,
        None,
        None,
        None,
        SessionAffinityMode::Hard,
        "model".to_string(),
        "namespace".to_string(),
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        CancellationToken::new(),
    )
}

struct CountingClassifier {
    classifications: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
    completions: Arc<AtomicUsize>,
}

#[async_trait]
impl dynamo_kv_router::plugins::request_classifier::RequestClassifier for CountingClassifier {
    fn classify(
        &mut self,
        request: dynamo_kv_router::plugins::request_classifier::ClassifyRequest,
        _: &dyn dynamo_kv_router::plugins::request_classifier::WorkerView,
    ) -> Result<
        dynamo_kv_router::plugins::request_classifier::ClassifyRequest,
        Box<dynamo_kv_router::plugins::request_classifier::ClassifierError>,
    > {
        self.classifications.fetch_add(1, Ordering::Relaxed);
        Ok(request)
    }
    async fn on_event(
        &mut self,
        event: dynamo_kv_router::plugins::request_classifier::ClassifyEvent,
    ) {
        match &event {
            dynamo_kv_router::plugins::request_classifier::ClassifyEvent::Aborted { .. } => {
                self.aborts.fetch_add(1, Ordering::Relaxed);
            }
            dynamo_kv_router::plugins::request_classifier::ClassifyEvent::Completed { .. } => {
                self.completions.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

struct ConditionalFixture {
    runtime: dynamo_runtime::Runtime,
    _distributed: dynamo_runtime::DistributedRuntime,
    endpoint: dynamo_runtime::component::Endpoint,
    _workers: watch::Sender<HashMap<u64, crate::local_model::runtime_config::ModelRuntimeConfig>>,
    router: Arc<PrefillRouter>,
    host: Arc<RoutingHost>,
    dispatch: Arc<RecordingDecodeDispatch>,
    affinity: crate::session_affinity::AffinityCoordinator,
    classifications: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
    completions: Arc<AtomicUsize>,
}

impl ConditionalFixture {
    async fn new(namespace: &str) -> Self {
        Self::with_cancellation(namespace, false).await
    }

    async fn with_cancellation(namespace: &str, supports_cancellation: bool) -> Self {
        use crate::kv_router::LlmDispatch;
        use crate::{kv_router::KvRouter, local_model::runtime_config::ModelRuntimeConfig};
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace(namespace)
            .unwrap()
            .component("decode")
            .unwrap()
            .endpoint("generate");
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let (workers_tx, workers) =
            watch::channel(HashMap::from([(worker_id, ModelRuntimeConfig::default())]));
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            ..Default::default()
        };
        let chooser = KvRouter::new(
            endpoint.clone(),
            client.clone(),
            workers,
            None,
            16,
            SelectionPolicySource::Registry,
            Some(config),
            None,
            "decode",
            None,
            false,
            None,
            None,
        )
        .await
        .unwrap();
        let classifications = Arc::new(AtomicUsize::new(0));
        let aborts = Arc::new(AtomicUsize::new(0));
        let completions = Arc::new(AtomicUsize::new(0));
        let chooser = chooser
            .with_request_classifier(CountingClassifier {
                classifications: classifications.clone(),
                aborts: aborts.clone(),
                completions: completions.clone(),
            })
            .unwrap();
        let dispatch = Arc::new(RecordingDecodeDispatch::default());
        let affinity = crate::session_affinity::AffinityCoordinator::new(
            std::time::Duration::from_secs(60),
            SessionAffinityMode::Hard,
        )
        .unwrap();
        let host = Arc::new(
            RoutingHost::builder(client, RouterMode::KV)
                .kv_router(Arc::new(chooser))
                .affinity_coordinator(Some(affinity.clone()))
                .dispatch(Arc::clone(&dispatch) as LlmDispatch)
                .build()
                .await
                .unwrap(),
        );
        // No KV events in this test, so nothing is cached and the net-new ratio is
        // 1.0; a ratio threshold above 1.0 makes the ISL policy bypass every request.
        let router = conditional_router_with(
            KvRouterConfig {
                conditional_disagg_enabled: true,
                conditional_disagg_eff_isl_ratio_threshold: 2.0,
                ..Default::default()
            },
            supports_cancellation,
        );
        assert!(router.decode_routing_host.set(host.clone()).is_ok());
        if supports_cancellation {
            let mut card = crate::model_card::ModelDeploymentCard::with_name_only("prefill");
            card.runtime_config.runtime_data.insert(
                DISAGG_PREFILL_CANCEL_ANYTIME_V1.into(),
                serde_json::json!(true),
            );
            distributed
                .discovery()
                .register(
                    dynamo_runtime::discovery::DiscoverySpec::from_model(
                        namespace.into(),
                        "decode".into(),
                        "generate".into(),
                        &card,
                    )
                    .unwrap(),
                )
                .await
                .unwrap();
            let mut configs = router
                .model_manager
                .get_or_create_runtime_config_watcher(&endpoint)
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while !router.model_manager.worker_supports_runtime_capability(
                    &endpoint.id(),
                    worker_id,
                    DISAGG_PREFILL_CANCEL_ANYTIME_V1,
                ) {
                    configs.changed().await.unwrap();
                }
            })
            .await
            .unwrap();
        }
        Self {
            runtime,
            _distributed: distributed,
            endpoint,
            _workers: workers_tx,
            router,
            host,
            dispatch,
            affinity,
            classifications,
            aborts,
            completions,
        }
    }

    fn binding(&self, requires_prefill: bool) -> Arc<PrefillBinding> {
        Arc::new(PrefillBinding {
            target_id: WorkerSetTargetId::Legacy(self.endpoint.id()),
            endpoint_id: self.endpoint.id(),
            router: self.host.clone(),
            prefill_router_mode: RouterMode::KV,
            encoder_result_requires_prefill: requires_prefill,
        })
    }
}

#[tokio::test]
async fn retired_prefill_enforces_handoff_consumption_policy() {
    use crate::kv_router::{RouterLoadSource, RoutingLoadContext};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("retired-encoder-prefill")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let replacement = distributed
        .namespace("retired-encoder-prefill")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("replacement");
    let client = endpoint.client().await.unwrap();
    let load_context = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::Prefill,
        crate::discovery::LoadThresholdHandle::new(Default::default()),
        &distributed.child_token(),
        None,
    )
    .await
    .unwrap();
    let prefill_host = Arc::new(
        RoutingHost::builder(client, RouterMode::RoundRobin)
            .load_context(load_context)
            .build()
            .await
            .unwrap(),
    );
    tokio::time::pause();
    for (requires_prefill, handoff, staged, query) in [
        (true, true, false, false),
        (false, true, false, false),
        (true, false, false, false),
        (true, true, true, false),
        (true, true, false, true),
    ] {
        for (retire_after_entry, replacement_kind) in [(false, 0), (true, 0), (true, 1), (true, 2)]
        {
            let router = fresh_decode_router_with_mode(if query {
                RouterMode::KV
            } else {
                RouterMode::RoundRobin
            });
            let mut card =
                crate::model_card::ModelDeploymentCard::with_name_only("optional-prefill");
            card.runtime_config.runtime_data.insert(
                crate::protocols::common::preprocessor::REQUIRES_ENCODER_RESULT.into(),
                serde_json::json!(requires_prefill),
            );
            assert_eq!(
                ModelManager::requires_encoder_result(&card),
                requires_prefill
            );
            let binding = Arc::new(PrefillBinding {
                target_id: WorkerSetTargetId::Legacy(endpoint.id()),
                endpoint_id: endpoint.id(),
                router: prefill_host.clone(),
                prefill_router_mode: RouterMode::RoundRobin,
                encoder_result_requires_prefill: ModelManager::requires_encoder_result(&card),
            });
            *router.target.lock() = Some(binding.target_id.clone());
            router.set_test_binding(Some(binding));
            let mut request = if query {
                query_only_request()
            } else {
                inference_request()
            };
            request.encoder_result = handoff.then(|| serde_json::json!({"opaque": "handoff"}));
            request.staged_kv_cleanup = staged;
            let host = Arc::new(CountingDecodeHost::default());
            let result = if retire_after_entry && !query {
                let tracker = Arc::new(RequestTracker::new());
                let barrier = tracker.set_phase(RequestPhase::Decode).await;
                request.tracker = Some(tracker);
                let in_flight = tokio::spawn({
                    let router = router.clone();
                    let host = host.clone();
                    async move { router.generate(SingleIn::new(request), host).await }
                });
                // Paused time advances after generate parks at the phase barrier.
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                if replacement_kind == 1 {
                    router.set_target(Some(WorkerSetTarget::Legacy(replacement.clone())));
                } else if replacement_kind == 2 {
                    let (_admissions, admitted_ids) = watch::channel(Vec::new());
                    router.set_target(Some(WorkerSetTarget::Committed(
                        crate::discovery::CommittedWorkerSetTarget {
                            endpoint: replacement.clone(),
                            group: "optional-prefill".into(),
                            generation: 1,
                            card: Arc::new(crate::model_card::ModelDeploymentCard::with_name_only(
                                "optional-prefill",
                            )),
                            admitted_ids,
                        },
                    )));
                }
                router.set_target(None);
                drop(barrier);
                tokio::time::timeout(std::time::Duration::from_secs(5), in_flight)
                    .await
                    .expect("retired prefill must resolve the waiting request")
                    .unwrap()
            } else {
                router.set_target(None);
                router.generate(SingleIn::new(request), host.clone()).await
            };
            assert_eq!(router.binding().is_none(), replacement_kind != 0 && !query);
            if requires_prefill && handoff && !staged && !query {
                let error = result.unwrap_err();
                assert!(crate::migration::is_migratable(error.as_ref()), "{error}");
                assert!(match_error_chain(
                    error.as_ref(),
                    &[ErrorType::WorkerUnavailable],
                    &[]
                ));
                assert_eq!(host.requests.load(Ordering::Relaxed), 0);
            } else {
                drop(result.unwrap());
                assert_eq!(host.requests.load(Ordering::Relaxed), 1);
            }
        }
    }
    tokio::time::resume();
    runtime.shutdown();
}

#[tokio::test]
async fn retired_committed_target_preserves_policy_before_binding_publication() {
    use crate::discovery::CommittedWorkerSetTarget;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("unpublished-prefill-policy")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let router = fresh_decode_router();
    let (_admissions, admitted_ids) = watch::channel(Vec::new());
    for (generation, required) in [(1, true), (2, false), (3, true)] {
        let mut card = crate::model_card::ModelDeploymentCard::with_name_only("prefill");
        card.runtime_config.runtime_data.insert(
            crate::protocols::common::preprocessor::REQUIRES_ENCODER_RESULT.into(),
            serde_json::json!(required),
        );
        router.set_target(Some(WorkerSetTarget::Committed(CommittedWorkerSetTarget {
            endpoint: endpoint.clone(),
            group: "prefill".into(),
            generation,
            card: Arc::new(card),
            admitted_ids: admitted_ids.clone(),
        })));
        assert!(matches!(
            router.activation_snapshot(),
            ActivationSnapshot::Pending
        ));
        router.set_target(None);
        let mut request = inference_request();
        request.encoder_result = Some(serde_json::json!({"opaque": "handoff"}));
        let host = Arc::new(CountingDecodeHost::default());
        let result = router.generate(SingleIn::new(request), host.clone()).await;
        if required {
            let error = result.unwrap_err();
            assert!(match_error_chain(
                error.as_ref(),
                &[ErrorType::WorkerUnavailable],
                &[]
            ));
            assert_eq!(host.requests.load(Ordering::Relaxed), 0);
        } else {
            drop(result.unwrap());
            assert_eq!(host.requests.load(Ordering::Relaxed), 1);
        }
    }
    drop(router);
    runtime.shutdown();
}

#[tokio::test]
async fn instance_annotation_does_not_bypass_prefill_on_executing_routes() {
    for mode in [RouterMode::RoundRobin, RouterMode::Random] {
        let router = fresh_decode_router_with_mode(mode);
        mark_target_pending(&router);
        let host = Arc::new(CountingDecodeHost::default());
        let error = router
            .generate(SingleIn::new(query_only_request()), host.clone())
            .await
            .unwrap_err();
        assert!(match_error_chain(
            error.as_ref(),
            &[ErrorType::WorkerUnavailable],
            &[]
        ));
        assert_eq!(host.requests.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn cancelled_encoder_handoff_never_starts_independent_prefill() {
    let binding = local_prefill_binding("cancelled-encoder").await;
    tokio::time::pause();
    for stop_before_routing in [true, false] {
        for kill in [false, true] {
            let router = conditional_router();
            router.set_test_binding(Some(binding.clone()));
            let tracker = Arc::new(RequestTracker::new());
            let barrier = tracker.set_phase(RequestPhase::Prefill).await;
            let mut request = inference_request();
            request.encoder_result = Some(serde_json::json!({"opaque": "handoff"}));
            request.tracker = Some(tracker);
            let mut request = SingleIn::new(request);
            request.insert(
                ENCODER_RESULT_RELEASE_SUPPORTED,
                EncoderResultReleaseSupported(true),
            );
            let context = request.context();
            let host = Arc::new(CountingDecodeHost::default());
            let cancel = || {
                if kill {
                    context.kill()
                } else {
                    context.stop_generating()
                }
            };
            if stop_before_routing {
                cancel();
            }
            let task = tokio::spawn({
                let router = router.clone();
                let host = host.clone();
                async move { router.generate(request, host).await }
            });
            if !stop_before_routing {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                cancel();
            }
            drop(barrier);
            let error = task.await.unwrap().unwrap_err();
            assert!(
                match_error_chain(error.as_ref(), &[ErrorType::Cancelled], &[]),
                "{error}"
            );
            assert_eq!(host.requests.load(Ordering::Relaxed), 0);
        }
    }
}

#[tokio::test]
async fn conditional_bypass_retains_text_releasable_and_staged_requests() {
    let fixture = ConditionalFixture::new("encoder-bypass-controls").await;
    fixture
        .router
        .set_test_binding(Some(fixture.binding(false)));
    fixture
        .router
        .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
    let downstream = Arc::new(QueryOnlyDecodeHost::default());
    for (has_encoder, release_supported, staged_kv) in [
        (false, false, false),
        (true, true, false),
        (true, false, true),
    ] {
        let mut request = SingleIn::new(inference_request());
        request.encoder_result = has_encoder.then(|| serde_json::json!({"opaque":"handoff"}));
        if has_encoder {
            request.multi_modal_data = Some(HashMap::from([(
                "image".into(),
                vec![MultimodalData::RawUrl(
                    "https://example.com/image.png".into(),
                )],
            )]));
        }
        request.staged_kv_cleanup = staged_kv;
        request.insert(
            ENCODER_RESULT_RELEASE_SUPPORTED,
            EncoderResultReleaseSupported(release_supported),
        );
        let mut response = fixture
            .router
            .generate(request, downstream.clone())
            .await
            .unwrap();
        assert_eq!(
            response.next().await.unwrap().data.unwrap().token_ids,
            vec![7]
        );
        assert!(response.next().await.is_none());
        assert_eq!(
            fixture.dispatch.phases.lock().last(),
            Some(&RequestPhase::Decode)
        );
        assert!(
            fixture
                .dispatch
                .annotations
                .lock()
                .last()
                .unwrap()
                .iter()
                .any(|value| value == BYPASS_REMOTE_PREFILL_ANNOTATION)
        );
    }
    assert_eq!(downstream.requests.load(Ordering::Relaxed), 0);
    fixture.runtime.shutdown();
}

#[tokio::test]
async fn nonreleasable_handoff_survives_late_client_cancellation() {
    for paired_cancellation in [false, true] {
        let fixture = ConditionalFixture::with_cancellation(
            &format!("nonreleasable-encoder-{paired_cancellation}"),
            paired_cancellation,
        )
        .await;
        let downstream = Arc::new(QueryOnlyDecodeHost::default());
        fixture
            .router
            .set_test_binding(Some(fixture.binding(false)));
        fixture
            .router
            .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
        let gate = Arc::new(tokio::sync::Notify::new());
        *fixture.dispatch.gate.lock() = Some(gate.clone());
        for kill in [false, true] {
            let session = SessionAffinityId::new(format!("nonreleasable-encoder-{kill}"));
            let leader = fixture.affinity.acquire(&session, None).await.unwrap();
            let mut request = SingleIn::new(inference_request());
            request.encoder_result = Some(serde_json::json!({"opaque": "must-consume"}));
            request.insert(SESSION_AFFINITY_CONTEXT_KEY, session);
            let context = request.context();
            let before = fixture.dispatch.phases.lock().len();
            let mut task = tokio::spawn({
                let router = fixture.router.clone();
                let downstream = downstream.clone();
                async move { router.generate(request, downstream).await }
            });
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                fixture.affinity.wait_for_initializing_waiter(),
            )
            .await
            .unwrap();
            if kill {
                context.kill();
            } else {
                context.stop_generating();
            }
            // Keep admission blocked after stop so cancellation cannot race with its completion.
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), &mut task)
                    .await
                    .is_err(),
                "a stopped caller must not abandon an undisposable encoder result"
            );
            assert_eq!(fixture.dispatch.phases.lock().len(), before);
            drop(leader);
            let consumer = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    if let Some(context) = fixture.dispatch.contexts.lock().get(before).cloned() {
                        break context;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            // Paired KV cancellation cannot dispose a separate producer's allocation.
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), consumer.stopped())
                    .await
                    .is_err()
            );
            gate.notify_one();
            let mut response = tokio::time::timeout(std::time::Duration::from_secs(5), task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let output = response.next().await.unwrap().data.unwrap();
            assert_eq!(output.token_ids, vec![7]);
            assert_eq!(
                fixture.dispatch.phases.lock().last(),
                Some(&RequestPhase::Prefill)
            );
            assert!(response.next().await.is_none());
        }
        fixture.runtime.shutdown();
    }
}

#[tokio::test]
async fn raw_media_encode_fallback_keeps_prefill_but_text_can_bypass() {
    let fixture = ConditionalFixture::new("raw-media-prefill-fallback").await;
    fixture
        .router
        .set_test_binding(Some(fixture.binding(false)));
    fixture
        .router
        .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
    let mut raw_media = inference_request();
    raw_media.multi_modal_data = Some(HashMap::from([(
        "image".into(),
        vec![MultimodalData::RawUrl(
            "https://example.com/image.png".into(),
        )],
    )]));
    assert!(raw_media.encoder_result.is_none());
    let mut response = fixture
        .router
        .generate(SingleIn::new(raw_media.clone()), fixture.host.clone())
        .await
        .unwrap();
    while response.next().await.is_some() {}
    assert_eq!(*fixture.dispatch.phases.lock(), vec![RequestPhase::Prefill]);
    assert!(
        !fixture.dispatch.annotations.lock()[0]
            .iter()
            .any(|a| a == BYPASS_REMOTE_PREFILL_ANNOTATION)
    );

    let mut response = fixture
        .router
        .generate(SingleIn::new(inference_request()), fixture.host.clone())
        .await
        .unwrap();
    while response.next().await.is_some() {}
    assert_eq!(
        *fixture.dispatch.phases.lock(),
        vec![RequestPhase::Prefill, RequestPhase::Decode]
    );
    assert!(
        fixture.dispatch.annotations.lock()[1]
            .iter()
            .any(|a| a == BYPASS_REMOTE_PREFILL_ANNOTATION)
    );
    fixture.runtime.shutdown();
}

pub(in crate::kv_router) async fn reject_stale_media_plan(
    mut request: SingleIn<PreprocessedRequest>,
    cleared: bool,
) -> Error {
    let fixture = ConditionalFixture::new(&format!("stale-media-plan-{cleared}")).await;
    fixture
        .router
        .set_test_binding(Some(fixture.binding(false)));
    fixture
        .router
        .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
    let tracker = Arc::new(RequestTracker::new());
    let barrier = tracker.set_phase(RequestPhase::Aggregated).await;
    request.tracker = Some(tracker.clone());
    let task = tokio::spawn({
        let router = fixture.router.clone();
        let host = fixture.host.clone();
        async move {
            request.insert(
                ENCODER_RESULT_RELEASE_SUPPORTED,
                EncoderResultReleaseSupported(true),
            );
            router.generate(request, host).await
        }
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let loads = fixture
                .host
                .kv_router()
                .get_potential_loads(&[], None, None, None, None)
                .await
                .unwrap();
            if loads.iter().map(|load| load.active_requests).sum::<usize>() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    if cleared {
        fixture.router.set_target(None);
    } else {
        fixture.router.set_test_binding(Some(fixture.binding(true)));
    }
    drop(barrier);
    let error = task
        .await
        .unwrap()
        .expect_err("stale plans cannot bypass mandatory P");
    assert!(
        match_error_chain(error.as_ref(), &[ErrorType::WorkerUnavailable], &[]),
        "{error:#}"
    );
    assert!(
        fixture.dispatch.phases.lock().is_empty(),
        "no native request was dispatched"
    );
    assert_eq!(tracker.phase(), RequestPhase::Prefill);
    let loads = fixture
        .host
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert_eq!(
        loads.iter().map(|load| load.active_requests).sum::<usize>(),
        0,
        "booking must settle before the error is returned"
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while fixture.aborts.load(Ordering::Relaxed) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(fixture.classifications.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.aborts.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.completions.load(Ordering::Relaxed), 0);
    fixture.runtime.shutdown();
    error
}

#[tokio::test]
async fn changed_prefill_binding_after_decode_tracking_aborts_before_dispatch() {
    let fixture = ConditionalFixture::new("post-tracking-prefill-change").await;
    fixture
        .router
        .set_test_binding(Some(fixture.binding(false)));
    fixture
        .router
        .set_target(Some(WorkerSetTarget::Legacy(fixture.endpoint.clone())));
    let mut request = SingleIn::new(inference_request());
    request.encoder_result = Some(serde_json::json!({"opaque": "handoff"}));
    request.ensure_tracker();
    let tracker = request.tracker.as_ref().unwrap().clone();
    {
        let _permit = tracker.set_phase(RequestPhase::Decode).await;
    }
    let decision = fixture
        .router
        .plan_conditional_disagg_decode(&request, request.id())
        .await
        .unwrap()
        .expect("optional Prefill permits an encoder handoff bypass");
    let worker = decision.plan.worker().worker_id;
    request.routing_mut().decode_worker_id = Some(worker);
    let expected_binding = decision.encoder_binding;
    assert!(
        fixture
            .router
            .allows_encoder_bypass(expected_binding.as_ref())
    );
    assert!(tracker.decode_worker_id().is_none());
    let error = fixture
        .host
        .dispatch_kv_plan_with_validation(request, decision.plan, || async {
            assert_eq!(
                tracker.decode_worker_id(),
                Some(worker),
                "binding validation must run after routing-decision tracking"
            );
            fixture.router.set_test_binding(Some(fixture.binding(true)));
            if fixture
                .router
                .allows_encoder_bypass(expected_binding.as_ref())
            {
                return Ok(());
            }
            let _permit = tracker.set_phase(RequestPhase::Prefill).await;
            Err(prefill_changed_error())
        })
        .await
        .expect_err("a newly mandatory Prefill stage cannot be bypassed");
    assert!(match_error_chain(
        error.as_ref(),
        &[ErrorType::WorkerUnavailable],
        &[]
    ));
    assert_eq!(tracker.phase(), RequestPhase::Prefill);
    assert!(fixture.dispatch.phases.lock().is_empty());
    let loads = fixture
        .host
        .kv_router()
        .get_potential_loads(&[], None, None, None, None)
        .await
        .unwrap();
    assert_eq!(
        loads.iter().map(|load| load.active_requests).sum::<usize>(),
        0,
        "a rejected tracked plan must return its booking"
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while fixture.aborts.load(Ordering::Relaxed) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(fixture.classifications.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.completions.load(Ordering::Relaxed), 0);
    fixture.runtime.shutdown();
}
