// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::decision_profile::SGLANG_SYSTEMONE_SERIAL_V1;

use super::*;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Instance,
    pipeline::{AddressedRequest, ManyIn, RouterMode, StreamingDispatch},
};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
struct ScoreHop {
    pins: Mutex<Vec<Option<u64>>>,
}

struct StripScoringRegistry(crate::types::openai::generate::GenerateStreamingEngine);

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for StripScoringRegistry
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (wire, _) = request.transfer(());
        self.0.generate(Context::new(wire)).await
    }
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for ScoreHop {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (wire, _, instance) = addressed.into_parts();
        let worker = instance.unwrap().id();
        let hints = wire.routing.as_ref().unwrap();
        self.pins.lock().unwrap().push(hints.backend_instance_id);
        assert!(wire.multi_modal_data.is_none());
        wire.tracker
            .as_ref()
            .unwrap()
            .record_worker(worker, Some(0), "decode");
        let labels: Vec<u32> = serde_json::from_value(
            wire.extra_args.as_ref().unwrap()["sglang_tito"]["token_ids_logprob"].clone(),
        )
        .unwrap();
        Ok(ResponseStream::new(
            Box::pin(stream::iter([terminal_output(&labels)])),
            context.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _: Instance,
        _: String,
        _: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("score branches are unary")
    }
}

struct PausedFinalHop {
    inner: Arc<dyn StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>>>,
    paused: AtomicBool,
    entered: Notify,
    resume: Notify,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for PausedFinalHop {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        if self.paused.load(Ordering::SeqCst) {
            self.entered.notify_one();
            self.resume.notified().await;
        }
        self.inner.generate(request).await
    }

    async fn generate_bidirectional(
        &self,
        _: Instance,
        _: String,
        _: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("score branches are unary")
    }
}

#[tokio::test]
async fn causal_execution_preserves_guard_and_pins_through_ordinary_routing_host() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::causal_execution_preserves_guard_and_pins_through_ordinary_routing_host"
        ),
        &[
            ("DYN_ROUTER_MIN_INITIAL_WORKERS", "0"),
            ("RUST_MIN_STACK", "4194304"),
        ],
    ) {
        return;
    }
    assert_score_host(RouterMode::RoundRobin).await;
}

#[tokio::test]
async fn causal_kv_execution_preserves_guard_pins_and_rejects_stripped_registry() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::causal_kv_execution_preserves_guard_pins_and_rejects_stripped_registry"
        ),
        &[
            ("DYN_ROUTER_MIN_INITIAL_WORKERS", "0"),
            ("RUST_MIN_STACK", "4194304"),
        ],
    ) {
        return;
    }
    Box::pin(assert_score_host(RouterMode::KV)).await;
}

async fn assert_score_host(mode: RouterMode) {
    use crate::{
        discovery::{DecisionDispatch, LoadThresholdHandle},
        kv_router::{RouterLoadSource, RoutingHost, RoutingLoadContext},
        local_model::runtime_config::SGLANG_GENERATE_CAPABILITY,
        model_type::{ModelInput, ModelType},
        worker_type::WorkerType,
    };
    crate::generate_format::test_formats::install();
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(
        runtime.clone(),
        dynamo_runtime::distributed::DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let endpoint = drt
        .namespace("score-host")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate");
    let discovered = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker = discovered.wait_for_instances().await.unwrap()[0].id();
    let (ordinary, ordinary_rx) = tokio::sync::watch::channel(vec![worker]);
    let client = discovered
        .with_admitted_instances_and_cancellation(ordinary_rx.clone(), runtime.child_token());
    let scores = Arc::new(ScoreHop::default());
    let final_hop = Arc::new(PausedFinalHop {
        inner: DecisionDispatch::with_dispatch(
            scores.clone(),
            crate::discovery::GuardPolicy::Optional,
        ),
        paused: AtomicBool::new(false),
        entered: Notify::new(),
        resume: Notify::new(),
    });
    let load = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::Aggregated,
        LoadThresholdHandle::new(Default::default()),
        &runtime.child_token(),
        None,
    )
    .await
    .unwrap();
    let (_configs, configs) = tokio::sync::watch::channel(std::collections::HashMap::from([(
        worker,
        crate::local_model::runtime_config::ModelRuntimeConfig::default(),
    )]));
    let mut builder = RoutingHost::builder(client.clone(), mode)
        .load_context(load)
        .dispatch(final_hop.clone() as crate::kv_router::LlmDispatch);
    if mode == RouterMode::KV {
        let config = dynamo_kv_router::config::KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            ..Default::default()
        };
        let chooser = Box::pin(crate::kv_router::KvRouter::new(
            endpoint,
            client,
            configs,
            None,
            16,
            crate::kv_router::SelectionPolicySource::Registry,
            Some(config),
            None,
            "aggregated",
            None,
            false,
            None,
            None,
        ))
        .await
        .unwrap();
        builder = builder.kv_router(Arc::new(chooser));
    }
    let host = Arc::new(builder.build().await.unwrap());
    let routing = crate::entrypoint::input::PreprocessedRouting::from_score_host_for_test(host);
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.set_name("test-model");
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config.max_num_seqs = Some(1);
    for flag in [SGLANG_GENERATE_CAPABILITY, SGLANG_SYSTEMONE_SERIAL_V1] {
        card.runtime_config
            .runtime_data
            .insert(flag.into(), true.into());
    }
    let engine = routing.score_engine();
    let preprocessor = OpenAIPreprocessor::new(card.clone()).unwrap();
    let stripped_engine = Arc::new(StripScoringRegistry(engine.clone()));
    let stripped_preprocessor = preprocessor.clone();
    let mut workers = WorkerSet::new("score-host".into(), card.mdcsum().into(), card);
    let resources = crate::discovery::test_support::token_resources(engine, preprocessor);
    workers.set_instance_watcher(ordinary_rx);
    let initial = (Profile::SglangSerialV1, 255);
    let (qualification, fleet) = crate::discovery::test_support::fleet(Some(initial), vec![worker]);
    crate::discovery::test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    let selected =
        crate::discovery::test_support::prepared_selection_for_test("test-model".into(), &workers)
            .await
            .unwrap();
    let response = dispatch_selection_for_test(
        selected,
        vec![prepared_noul_branch(0), prepared_noul_branch(1)],
        "raw-score".into(),
        Context::new(()).context(),
        Arc::new(Semaphore::new(2))
            .acquire_many_owned(2)
            .await
            .unwrap(),
        Arc::new(super::super::Metrics::new()),
        canonical_request(2),
    )
    .await;
    let status = response.status();
    let body = response_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(*scores.pins.lock().unwrap(), vec![None, Some(worker)]);
    let mut stripped = WorkerSet::new(
        "stripped-score-host".into(),
        workers.card().mdcsum().into(),
        workers.card().clone(),
    );
    let resources =
        crate::discovery::test_support::token_resources(stripped_engine, stripped_preprocessor);
    stripped.set_instance_watcher(workers.decision_admissions().unwrap().clone());
    crate::discovery::test_support::attach_surface_for_test(
        &mut stripped,
        resources,
        workers.decision.as_ref().unwrap().fleet.clone(),
    )
    .await;
    let unchecked =
        crate::discovery::test_support::prepared_selection_for_test("test-model".into(), &stripped)
            .await
            .unwrap();
    let response = dispatch_selection_for_test(
        unchecked,
        vec![prepared_noul_branch(0)],
        "stripped-score".into(),
        Context::new(()).context(),
        Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        canonical_request(1),
    )
    .await;
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "an unchecked score stream must not become a successful response"
    );
    assert_eq!(
        scores.pins.lock().unwrap().len(),
        3,
        "a receipt rejects the unchecked result but cannot undo the deliberately stripped dispatch"
    );
    final_hop.paused.store(true, Ordering::SeqCst);
    let selected =
        crate::discovery::test_support::prepared_selection_for_test("test-model".into(), &workers)
            .await
            .unwrap();
    let pending = tokio::spawn(dispatch_selection_for_test(
        selected,
        vec![prepared_noul_branch(0)],
        "stale-score".into(),
        Context::new(()).context(),
        Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        canonical_request(1),
    ));
    tokio::time::timeout(Duration::from_secs(5), final_hop.entered.notified())
        .await
        .unwrap();
    qualification.send_replace(crate::discovery::DecisionAvailability::Unadvertised);
    crate::discovery::test_support::publish(
        &qualification,
        Some((Profile::VllmCausalV1, 26)),
        vec![worker],
    );
    assert_eq!(*ordinary.borrow(), vec![worker]);
    final_hop.resume.notify_one();
    assert_eq!(
        pending.await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(scores.pins.lock().unwrap().len(), 3);
    runtime.shutdown();
}
