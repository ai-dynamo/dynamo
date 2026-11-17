// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod activation;
mod conditional;
mod media;
mod protocol;
mod replan;

use super::*;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_runtime::{
    engine::AsyncEngine,
    pipeline::{Context, Error},
};
use dynamo_scheduler::host::RouterConfigOverride;
use futures::StreamExt;
use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicUsize, Ordering},
};
use tokio::sync::oneshot;

use super::dispatch::independent_prefill_context;
use crate::kv_router::encoder_router::EncoderResultReleaseSupported;
use crate::local_model::runtime_config::DISAGG_PREFILL_CANCEL_ANYTIME_V1;
use crate::protocols::common::preprocessor::ENCODER_RESULT_RELEASE_SUPPORTED;
use crate::protocols::common::{
    FinishReason,
    preprocessor::{PreprocessedRequest, RoutingHints},
    timing::RoutingData,
};
use crate::protocols::response_merge::PrefillNeed;

const MAX_ROOM: u64 = i64::MAX as u64;

#[derive(Default)]
struct QueryOnlyDecodeHost {
    requests: AtomicUsize,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for QueryOnlyDecodeHost
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        assert!(request.get_annotation_value("query_instance_id").is_some());
        self.requests.fetch_add(1, Ordering::Relaxed);
        let output = Annotated::from_data(LLMEngineOutput {
            routing_data: Some(Box::new(RoutingData {
                token_ids: Some(request.token_ids.as_ref().clone()),
                ..Default::default()
            })),
            ..Default::default()
        });
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { output })),
            request.context(),
        ))
    }
}

pub(super) fn conditional_router() -> Arc<PrefillRouter> {
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

async fn local_prefill_binding(namespace: &str) -> Arc<PrefillBinding> {
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

fn query_only_request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .annotations(vec!["query_instance_id:".to_string()])
        .build()
        .unwrap()
}

#[derive(Default)]
struct CountingDecodeHost {
    requests: AtomicUsize,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for CountingDecodeHost
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        Ok(ResponseStream::new(
            Box::pin(stream::empty()),
            request.context(),
        ))
    }
}

fn inference_request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .build()
        .unwrap()
}

/// A decode WorkerSet's router as the watcher creates it: enabled, no
/// target yet, lifecycle `Pending`, no conditional disaggregation.
fn fresh_decode_router() -> Arc<PrefillRouter> {
    fresh_decode_router_with_mode(RouterMode::RoundRobin)
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

/// Set the pending state directly so the test does not need a live endpoint.
fn mark_target_pending(router: &PrefillRouter) {
    *router.target.lock() = Some(WorkerSetTargetId::Committed {
        group: "namespace/prefill/generate".to_string(),
        generation: 1,
    });
    router.set_test_binding(None);
}

// -- Activation-window behavior (worker rollout race) --
//
// Admission counts a prefill worker as live once it registers, but this
// router's binding is built asynchronously after `set_target`. A decode-only
// worker cannot run the aggregated request a passthrough would send it.

fn bypassed_stream() -> ManyOut<Annotated<LLMEngineOutput>> {
    let request = SingleIn::new(query_only_request());
    let output = Annotated::from_data(LLMEngineOutput {
        token_ids: vec![7],
        ..Default::default()
    });
    ResponseStream::new(
        Box::pin(stream::once(async move { output })),
        request.context(),
    )
}

#[derive(Default)]
struct RecordingDecodeDispatch {
    annotations: Mutex<Vec<Vec<String>>>,
    phases: Mutex<Vec<RequestPhase>>,
    contexts: Mutex<Vec<Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>>>,
    gate: Mutex<Option<Arc<tokio::sync::Notify>>>,
    fail_phase: Mutex<Option<RequestPhase>>,
    fail_stream_phase: Mutex<Option<RequestPhase>>,
    prefill_output: Mutex<Option<LLMEngineOutput>>,
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
        if self.fail_phase.lock().as_ref() == Some(&request.phase()) {
            return Err(DynamoError::builder()
                .class(ErrorClass::WorkerUnavailable)
                .message("injected replan dispatch failure")
                .build()
                .into());
        }
        let gate = self.gate.lock().clone();
        if let Some(gate) = gate {
            gate.notified().await;
        }
        let output = if self.fail_stream_phase.lock().as_ref() == Some(&request.phase()) {
            use dynamo_runtime::protocols::maybe_error::MaybeError;
            Annotated::from_err(
                DynamoError::builder()
                    .class(ErrorClass::WorkerUnavailable)
                    .message("injected replan dispatch failure")
                    .build(),
            )
        } else {
            Annotated::from_data(
                self.prefill_output
                    .lock()
                    .clone()
                    .filter(|_| request.phase() == RequestPhase::Prefill)
                    .unwrap_or(LLMEngineOutput {
                        token_ids: vec![7],
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    }),
            )
        };
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

struct ReplanClassifier {
    classifications: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
    completions: Arc<AtomicUsize>,
    events: Arc<Mutex<Vec<dynamo_kv_router::plugins::request_classifier::ClassifyEvent>>>,
}

#[async_trait]
impl dynamo_kv_router::plugins::request_classifier::RequestClassifier for ReplanClassifier {
    fn classify(
        &mut self,
        request: dynamo_kv_router::plugins::request_classifier::ClassifyRequest,
        _: &dyn dynamo_kv_router::plugins::request_classifier::WorkerView,
    ) -> dynamo_kv_router::plugins::request_classifier::ClassifyFuture {
        self.classifications.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move { Ok(request) })
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
        self.events.lock().push(event);
    }
}

struct CaptureHandoffHost {
    host: Arc<RoutingHost>,
    request: Mutex<Option<(RequestPhase, bool)>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for CaptureHandoffHost
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        self.request
            .lock()
            .replace((request.phase(), request.migration_state.is_some()));
        self.host.generate(request).await
    }
}

struct ConditionalFixture {
    runtime: dynamo_runtime::Runtime,
    distributed: dynamo_runtime::DistributedRuntime,
    endpoint: dynamo_runtime::component::Endpoint,
    workers: watch::Sender<HashMap<u64, crate::local_model::runtime_config::ModelRuntimeConfig>>,
    router: Arc<PrefillRouter>,
    host: Arc<RoutingHost>,
    dispatch: Arc<RecordingDecodeDispatch>,
    affinity: crate::session_affinity::AffinityCoordinator,
    classifications: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
    completions: Arc<AtomicUsize>,
    events: Arc<Mutex<Vec<dynamo_kv_router::plugins::request_classifier::ClassifyEvent>>>,
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
        let events = Arc::new(Mutex::new(Vec::new()));
        let chooser = chooser
            .with_request_classifier(ReplanClassifier {
                classifications: classifications.clone(),
                aborts: aborts.clone(),
                completions: completions.clone(),
                events: events.clone(),
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
            distributed,
            endpoint,
            workers: workers_tx,
            router,
            host,
            dispatch,
            affinity,
            classifications,
            aborts,
            completions,
            events,
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

fn request_with_constraints(
    routing_constraints: Option<RoutingConstraints>,
) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .routing(Some(RoutingHints {
            routing_constraints,
            ..Default::default()
        }))
        .build()
        .unwrap()
}
