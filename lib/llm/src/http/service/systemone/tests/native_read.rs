// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    model_type::{ModelInput, ModelType},
    protocols::{
        decision_native::NativeUsage,
        decision_read::{DecisionReadRequest, DecisionReadResponse, ReadOutcome},
    },
};

async fn read_selection_for_test(
    engine: crate::protocols::decision_read::DecisionReadStreamingEngine,
    profile: Profile,
) -> PreparedSelectionFixture {
    let mut card = ModelDeploymentCard::with_name_only("test-model");
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    card.runtime_config
        .runtime_data
        .insert(profile.decision_capability().into(), true.into());
    card.runtime_config.runtime_data.insert(
        "vllm_native_decision_max_candidates".into(),
        profile.limits().1.into(),
    );
    let workers = WorkerSet::new("reads".into(), card.mdcsum().into(), card);
    let resources = crate::discovery::test_support::native_resources(DecisionEngine::Read(engine));
    crate::discovery::test_support::forced_selection_for_test(
        "test-model".into(),
        profile,
        workers,
        resources,
    )
    .await
    .unwrap()
}

struct ReadEngine {
    malformed: bool,
    unknown_cache: bool,
    calls: Mutex<Vec<DecisionReadRequest>>,
}

#[tokio::test]
async fn native_reads_reject_single_level_scores_before_worker_dispatch() {
    for profile in [Profile::VllmLiquidV1, Profile::VllmDiffusionV1] {
        let engine = Arc::new(ReadEngine {
            malformed: false,
            unknown_cache: false,
            calls: Mutex::new(vec![]),
        });
        let request = single_level_score_request();
        let fixture = read_selection_for_test(engine.clone(), profile).await;
        let error = super::super::execution::prepare(&request, fixture.selection.clone(), 100)
            .err()
            .expect("native profile requires at least two score levels");
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(error.1.contains("at least two score levels"));
        assert!(engine.calls.lock().unwrap().is_empty());
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<DecisionReadRequest>, ManyOut<Annotated<DecisionReadResponse>>, Error>
    for ReadEngine
{
    async fn generate(
        &self,
        request: SingleIn<DecisionReadRequest>,
    ) -> Result<ManyOut<Annotated<DecisionReadResponse>>, Error> {
        let context = request.context();
        let (wire, _) = request.into_parts();
        self.calls.lock().unwrap().push(wire.clone());
        if self.unknown_cache {
            return Err(dynamo_runtime::error::DynamoError::builder()
                .class(dynamo_runtime::error::ErrorClass::BackendProtocol)
                .message("invalid measured cache usage")
                .build()
                .into());
        }
        let wire = wire.decision_read;
        let response = DecisionReadResponse {
            finish_reason: crate::protocols::decision_native::NativeFinishReason::Stop,
            protocol_version: 1,
            profile: wire.profile,
            outcomes: wire
                .questions
                .iter()
                .map(|q| ReadOutcome {
                    ordinal: q.ordinal,
                    probabilities: vec![0.25, 0.75],
                    label_mass: if self.malformed { 2.0 } else { 0.5 },
                })
                .collect(),
            usage: NativeUsage {
                input_tokens: 17,
                output_tokens: wire.questions.len() as u64 * 2,
                cached_tokens: 0,
            },
        };
        Ok(ResponseStream::new(
            Box::pin(stream::once(async { Annotated::from_data(response) })),
            context,
        ))
    }
}

#[tokio::test]
async fn native_read_selection_caches_the_published_candidate_capacity() {
    for profile in [Profile::VllmLiquidV1, Profile::VllmDiffusionV1] {
        let engine = Arc::new(ReadEngine {
            malformed: false,
            unknown_cache: false,
            calls: Mutex::new(vec![]),
        });
        let prototype = read_selection_for_test(engine, profile).await;
        let mut card = prototype.context.card.clone();
        card.runtime_config
            .runtime_data
            .insert("vllm_native_decision_max_candidates".into(), 3.into());
        let mut workers = WorkerSet::new("capacity".into(), card.mdcsum().into(), card);
        let (_admissions, admitted_ids) = tokio::sync::watch::channel(vec![1]);
        workers.set_instance_watcher(admitted_ids);
        let _qualification = crate::discovery::test_support::attach_card_surface_for_test(
            &mut workers,
            crate::discovery::test_support::native_resources(prototype.engine.clone()),
        )
        .await;
        let mut selection = crate::discovery::test_support::prepared_selection_for_test(
            "test-model".into(),
            &workers,
        )
        .await
        .unwrap();
        let expected = selection
            .context
            .card
            .decision_max_candidates(profile)
            .unwrap();
        assert_eq!(expected, 3);
        assert_eq!(
            selection.context.decision_capabilities().max_candidates,
            expected
        );
        selection
            .context
            .card
            .runtime_config
            .runtime_data
            .remove("vllm_native_decision_max_candidates");
        assert_eq!(
            selection.context.decision_capabilities().max_candidates,
            expected
        );
    }
}

#[tokio::test]
async fn native_read_uses_public_projection_with_real_mass_and_output_usage() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let engine = Arc::new(ReadEngine {
            malformed: false,
            unknown_cache: false,
            calls: Mutex::new(vec![]),
        });
        let selection = read_selection_for_test(engine.clone(), Profile::VllmDiffusionV1).await;
        let mut request = canonical_request(2).as_ref().clone();
        request.dialect = dialect;
        let prepared =
            super::super::execution::prepare(&request, selection.selection.clone(), 100).unwrap();
        let admission = Arc::new(Semaphore::new(2));
        let response = super::super::execution::dispatch(
            prepared,
            "native-read".into(),
            Context::new(()).context(),
            admission.clone().acquire_many_owned(2).await.unwrap(),
            Arc::new(super::super::Metrics::new()),
            "test-model".into(),
            Arc::new(request),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-request-id"], "native-read");
        let body = response_body(response).await;
        assert_eq!(body["usage"]["input_tokens"], 17);
        assert_eq!(body["usage"]["output_tokens"], 4);
        if dialect == dynamo_decisions::Dialect::OpenAi {
            assert_eq!(body["answers"].as_array().unwrap().len(), 2);
            assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 0);
        } else {
            assert_eq!(body["answers"].as_object().unwrap().len(), 2);
        }
        assert_eq!(engine.calls.lock().unwrap().len(), 1);
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn native_read_invalid_mass_is_not_a_normal_answer() {
    let engine = Arc::new(ReadEngine {
        malformed: true,
        unknown_cache: false,
        calls: Mutex::new(vec![]),
    });
    let selection = read_selection_for_test(engine, Profile::VllmDiffusionV1).await;
    let request = canonical_request(1);
    let prepared =
        super::super::execution::prepare(&request, selection.selection.clone(), 100).unwrap();
    let parent = Context::new(()).context();
    let admission = Arc::new(Semaphore::new(1));
    let response = super::super::execution::dispatch(
        prepared,
        "bad-native".into(),
        parent.clone(),
        admission.clone().acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        "test-model".into(),
        request,
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(response_body(response).await.get("answers").is_none());
    assert!(parent.is_killed());
    assert_eq!(admission.available_permits(), 1);
}

#[tokio::test]
async fn native_read_unknown_cache_is_not_invented_for_openai() {
    let engine = Arc::new(ReadEngine {
        malformed: false,
        unknown_cache: true,
        calls: Mutex::new(vec![]),
    });
    let selected = read_selection_for_test(engine, Profile::VllmDiffusionV1).await;
    let mut request = canonical_request(1).as_ref().clone();
    request.dialect = dynamo_decisions::Dialect::OpenAi;
    let request = Arc::new(request);
    let prepared =
        super::super::execution::prepare(&request, selected.selection.clone(), 100).unwrap();
    let admission = Arc::new(Semaphore::new(1));
    let response = super::super::execution::dispatch(
        prepared,
        "unknown-cache".into(),
        Context::new(()).context(),
        admission.acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        "test-model".into(),
        request,
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(response_body(response).await.get("answers").is_none());
}

#[tokio::test]
async fn native_reads_accept_core_validated_thinking_off_controls() {
    for profile in [Profile::VllmLiquidV1, Profile::VllmDiffusionV1] {
        let engine = Arc::new(ReadEngine {
            malformed: false,
            unknown_cache: false,
            calls: Mutex::new(vec![]),
        });
        let selected = read_selection_for_test(engine.clone(), profile).await;
        let mut request = canonical_request(1).as_ref().clone();
        request
            .chat_template_kwargs
            .insert("enable_thinking".into(), false.into());
        assert!(
            super::super::execution::prepare(&request, selected.selection.clone(), 100).is_ok()
        );
        assert!(engine.calls.lock().unwrap().is_empty());
    }
}

struct PendingReadEngine {
    started: Notify,
    context: Mutex<Option<Arc<dyn AsyncEngineContext>>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<DecisionReadRequest>, ManyOut<Annotated<DecisionReadResponse>>, Error>
    for PendingReadEngine
{
    async fn generate(
        &self,
        request: SingleIn<DecisionReadRequest>,
    ) -> Result<ManyOut<Annotated<DecisionReadResponse>>, Error> {
        let context = request.context();
        self.context.lock().unwrap().replace(context.clone());
        self.started.notify_one();
        Ok(ResponseStream::new(Box::pin(stream::pending()), context))
    }
}

#[tokio::test]
async fn native_read_deadline_kills_linked_child_and_releases_admission() {
    let engine = Arc::new(PendingReadEngine {
        started: Notify::new(),
        context: Mutex::new(None),
    });
    let selected = read_selection_for_test(engine.clone(), Profile::VllmLiquidV1).await;
    let request = canonical_request(1);
    let prepared =
        super::super::execution::prepare(&request, selected.selection.clone(), 100).unwrap();
    let admission = Arc::new(Semaphore::new(1));
    let parent = Context::new(()).context();
    let task = tokio::spawn(super::super::execution::dispatch(
        prepared,
        "deadline-read".into(),
        parent.clone(),
        admission.clone().acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        "test-model".into(),
        request,
    ));
    tokio::time::timeout(Duration::from_secs(2), engine.started.notified())
        .await
        .unwrap();
    let response = super::super::response::dispatch_with_deadline(
        task,
        tokio::time::Instant::now() + Duration::from_millis(10),
        parent.clone(),
        dynamo_decisions::Dialect::OpenAi,
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert!(parent.is_killed());
    assert!(engine.context.lock().unwrap().as_ref().unwrap().is_killed());
    let released = tokio::time::timeout(Duration::from_secs(2), admission.clone().acquire_owned())
        .await
        .expect("cancelled native read must release admission")
        .unwrap();
    drop(released);
    assert_eq!(admission.available_permits(), 1);
}
