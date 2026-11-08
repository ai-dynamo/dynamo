// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::decision_profile::{
    SGLANG_SYSTEMONE_SERIAL_V1, VLLM_CLEF_FLASH_DECISION_V1, VLLM_NATIVE_DECISION_MAX_CANDIDATES,
};

use super::*;
use crate::{
    discovery::{Model, SystemOneExecutionPlan, SystemOneSelectionError, test_support},
    local_model::runtime_config::SGLANG_GENERATE_CAPABILITY,
    migration::Migration,
    protocols::common::{
        OutputOptions, SamplingOptions, StopConditions,
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
    },
};
use dynamo_runtime::{
    Runtime,
    component::Instance,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{
        AddressedRequest, Context, Error, ManyIn, PushRouterOptions, ResponseStream,
        ServerStreamingEngine, StreamingDispatch,
    },
};
use futures::{StreamExt, stream};
use serde_json::json;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;

struct RetryTransport {
    calls: AtomicUsize,
    prompts: Mutex<Vec<Vec<u32>>>,
    release_failure: Arc<Notify>,
}

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for RetryTransport {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let (wire, _, _) = addressed.into_parts();
        self.prompts
            .lock()
            .unwrap()
            .push(wire.token_ids.as_ref().clone());
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        let data = Annotated::from_data(serde_json::from_value::<LLMEngineOutput>(json!({
            "token_ids": [42 + attempt as u32], "text": "ordinary", "finish_reason": if attempt == 0 { serde_json::Value::Null } else { json!("stop") }
        })).unwrap());
        let stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = Annotated<LLMEngineOutput>> + Send>,
        > = if attempt == 0 {
            let release = self.release_failure.clone();
            Box::pin(
                stream::once(async move { data }).chain(stream::once(async move {
                    release.notified().await;
                    Annotated {
                        data: None,
                        id: None,
                        event: Some("error".into()),
                        comment: None,
                        error: Some(
                            dynamo_runtime::error::DynamoError::builder()
                                .error_type(dynamo_runtime::error::ErrorType::Disconnected)
                                .message("scripted ordinary stream failure")
                                .build(),
                        ),
                    }
                })),
            )
        } else {
            Box::pin(stream::iter([data]))
        };
        Ok(ResponseStream::new(stream, context.context()))
    }

    async fn generate_bidirectional(
        &self,
        _: Instance,
        _: String,
        _: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("ordinary retry fixture is unary")
    }
}

#[tokio::test]
async fn retired_ordinary_pipeline_can_resolve_transport_for_migration() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::retired_ordinary_pipeline_can_resolve_transport_for_migration"
        ),
        &[("DYN_ROUTER_MIN_INITIAL_WORKERS", "0")],
    ) {
        return;
    }
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = drt
        .namespace("retired-ordinary")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let discovered = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker = discovered.wait_for_instances().await.unwrap()[0].id();
    let (admission, ids) = tokio::sync::watch::channel(vec![worker]);
    let cancellation = runtime.child_token();
    let client =
        discovered.with_admitted_instances_and_cancellation(ids.clone(), cancellation.clone());
    let transport = Arc::new(RetryTransport {
        calls: AtomicUsize::new(0),
        prompts: Mutex::new(Vec::new()),
        release_failure: Arc::new(Notify::new()),
    });
    let router = Arc::new(
        PushRouter::from_client_with_options(
            client.clone(),
            RouterMode::RoundRobin,
            PushRouterOptions {
                dispatch: Some(transport.clone()),
                fault_detection: false,
                ..crate::kv_router::plugins::push_router_options()
            },
        )
        .await
        .unwrap(),
    );
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config.max_num_seqs = Some(1);
    for flag in [SGLANG_SYSTEMONE_SERIAL_V1, SGLANG_GENERATE_CAPABILITY] {
        card.runtime_config.set_engine_specific(flag, true).unwrap();
    }
    assert!(
        card.decision_qualification().is_some(),
        "retired decision companion requires a qualified token card"
    );
    let preprocessor = OpenAIPreprocessor::new(card.clone()).unwrap();
    let mut old = WorkerSet::new("retired".into(), "old".into(), card.clone());
    old.generate_engine = Some(router.clone());
    let resources = test_support::token_resources(router.clone(), preprocessor);
    old.set_lifecycle_cancellation(cancellation.clone());
    old.set_instance_watcher(client.instance_avail_watcher());
    let (publisher, fleet) = test_support::fleet(card.decision_qualification(), vec![worker]);
    test_support::attach_surface_for_test(&mut old, resources, fleet).await;
    let stale = SystemOneExecutionPlan::from_worker_set("ordinary".into(), &old)
        .unwrap()
        .context;
    let model = Model::new("ordinary".into());
    model.add_worker_set("generation".into(), Arc::new(old));
    let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> = router;
    let request = PreprocessedRequest::builder()
        .model("ordinary".to_string())
        .token_ids(vec![1, 2, 3])
        .stop_conditions(StopConditions {
            max_tokens: Some(8),
            ..Default::default()
        })
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .build()
        .unwrap();
    let mut response = Operator::generate(
        Migration::new(1, None, "ordinary".into(), Arc::new(Metrics::new())).as_ref(),
        Context::new(request),
        next,
    )
    .await
    .unwrap();
    assert_eq!(
        response.next().await.unwrap().data.unwrap().token_ids,
        vec![42]
    );
    let mut successor = WorkerSet::new("retired".into(), "new".into(), card);
    successor.set_lifecycle_cancellation(runtime.child_token());
    model.add_worker_set("generation".into(), Arc::new(successor));
    drop(admission);
    assert!(cancellation.is_cancelled());
    assert!(matches!(
        stale.validate_dispatch(Some(worker)),
        Err(SystemOneSelectionError::Unavailable)
    ));
    transport.release_failure.notify_one();
    let mut replayed = Vec::new();
    while let Some(chunk) = response.next().await {
        assert!(
            chunk.error.is_none(),
            "retired ordinary lookup failed: {:?}",
            chunk.error
        );
        if let Some(data) = chunk.data {
            replayed.extend(data.token_ids);
        }
    }
    assert_eq!(replayed, vec![43]);
    assert_eq!(transport.calls.load(Ordering::SeqCst), 2);
    assert_eq!(transport.prompts.lock().unwrap()[1], vec![1, 2, 3, 42]);
    assert!(client.is_instance_live(worker));
    assert!(!runtime.primary_token().is_cancelled());
    drop(publisher);
    runtime.shutdown();
}

struct ClassifyEngine;

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateClassifyRequest>,
        ManyOut<Annotated<NvCreateClassifyResponse>>,
        Error,
    > for ClassifyEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateClassifyRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateClassifyResponse>>, Error> {
        Ok(ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_data(
                NvCreateClassifyResponse::empty(),
            )])),
            request.context(),
        ))
    }
}

#[tokio::test]
async fn unavailable_native_factory_does_not_block_ordinary_commit() {
    let mut card = ModelDeploymentCard::with_name_only("ordinary-classify");
    card.model_input = ModelInput::Text;
    card.model_type = ModelType::Classify | ModelType::Pooling;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config
        .set_engine_specific(VLLM_CLEF_FLASH_DECISION_V1, true)
        .unwrap();
    card.runtime_config
        .set_engine_specific(VLLM_NATIVE_DECISION_MAX_CANDIDATES, 26)
        .unwrap();
    let mut workers = WorkerSet::new("ordinary".into(), card.mdcsum().into(), card.clone());
    workers.classify_engine = Some(Arc::new(ClassifyEngine));
    assert!(workers.load_context().is_none());
    let (_publication, fleet) = test_support::fleet(card.decision_qualification(), vec![1]);
    let (_membership, admitted) = tokio::sync::watch::channel(vec![1]);
    workers.set_instance_watcher(admitted.clone());
    let teardown = workers.initialize_allocator_trim_on_teardown();
    let mut plan = DecisionPlan::new(&card, true, RouterMode::RoundRobin, teardown);
    plan.prepare_resources(None, None, None);
    plan.attach(&mut workers, fleet);
    let manager = ModelManager::new();
    manager.add_worker_set(card.name(), "ordinary", workers);
    let model = manager.get_committed_model(card.name()).unwrap();
    assert!(matches!(
        model.get_systemone_execution_plan(),
        Err(SystemOneSelectionError::Unsupported)
    ));
    let request =
        serde_json::from_value(json!({"model":card.name(), "input":"ordinary input"})).unwrap();
    let mut response = model
        .get_classify_engine()
        .unwrap()
        .generate(Context::new(request))
        .await
        .unwrap();
    assert_eq!(response.next().await.unwrap().data.unwrap().object, "list");
}
