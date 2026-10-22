// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    discovery::{DecisionExecutor, Model, WorkerSet},
    local_model::runtime_config::DecisionExecutionProfile,
    model_type::{ModelInput, ModelType},
    protocols::decision_head::{DecisionHeadRequest, DecisionHeadResponse, HeadOutcome, HeadUsage},
    worker_type::WorkerType,
};

#[derive(Clone, Copy)]
enum Behavior {
    Success,
    Malformed,
    Reordered,
    Error,
    InvalidRequest,
    Pending,
    Empty,
    Duplicate,
}

struct HeadEngine {
    behavior: Behavior,
    calls: Mutex<Vec<DecisionHeadRequest>>,
    contexts: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
    started: Notify,
}

impl HeadEngine {
    fn new(behavior: Behavior) -> Arc<Self> {
        Arc::new(Self {
            behavior,
            calls: Mutex::new(Vec::new()),
            contexts: Mutex::new(Vec::new()),
            started: Notify::new(),
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<DecisionHeadRequest>, ManyOut<Annotated<DecisionHeadResponse>>, Error>
    for HeadEngine
{
    async fn generate(
        &self,
        request: SingleIn<DecisionHeadRequest>,
    ) -> Result<ManyOut<Annotated<DecisionHeadResponse>>, Error> {
        let context = request.context();
        let (request, _) = request.into_parts();
        self.contexts.lock().unwrap().push(context.clone());
        self.calls.lock().unwrap().push(request.clone());
        self.started.notify_one();
        if matches!(self.behavior, Behavior::Error | Behavior::InvalidRequest) {
            return Err(dynamo_runtime::error::DynamoError::builder()
                .class(if matches!(self.behavior, Behavior::InvalidRequest) {
                    dynamo_runtime::error::ErrorClass::InvalidRequest
                } else {
                    dynamo_runtime::error::ErrorClass::CapacityExhausted
                })
                .message("private head details")
                .build()
                .into());
        }
        if matches!(self.behavior, Behavior::Pending) {
            return Ok(ResponseStream::new(Box::pin(stream::pending()), context));
        }
        let mut outcomes: Vec<_> = request
            .decision_head
            .questions
            .iter()
            .map(|q| HeadOutcome {
                ordinal: q.ordinal,
                probabilities: vec![0.25, 0.75],
            })
            .collect();
        if matches!(self.behavior, Behavior::Reordered) {
            outcomes.reverse();
        }
        if matches!(self.behavior, Behavior::Malformed) {
            outcomes[0].probabilities = vec![0.1, 0.2];
        }
        let response = DecisionHeadResponse {
            protocol_version: 1,
            profile: request.decision_head.profile,
            outcomes,
            usage: HeadUsage {
                input_tokens: 13,
                output_tokens: 0,
                cached_tokens: Some(2),
            },
        };
        let frames = match self.behavior {
            Behavior::Empty => vec![],
            Behavior::Duplicate => vec![
                Annotated::from_data(response.clone()),
                Annotated::from_data(response),
            ],
            _ => vec![Annotated::from_data(response)],
        };
        Ok(ResponseStream::new(Box::pin(stream::iter(frames)), context))
    }
}

fn selected(
    engine: Arc<HeadEngine>,
    profile: DecisionExecutionProfile,
) -> SystemOneExecutionSelection {
    let mut card = ModelDeploymentCard::with_name_only("test-model");
    card.model_input = ModelInput::Text;
    card.model_type = ModelType::Classify | ModelType::Pooling;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config
        .runtime_data
        .insert(profile.generate_capability().into(), true.into());
    SystemOneExecutionSelection {
        profile,
        canonical_model: "test-model".into(),
        executor: DecisionExecutor::Head { engine },
        card,
        admitted_ids: None,
    }
}

async fn dispatch_head(
    engine: Arc<HeadEngine>,
    dialect: dynamo_decisions::Dialect,
) -> (Response, Arc<dyn AsyncEngineContext>, Arc<Semaphore>) {
    let selection = selected(engine, DecisionExecutionProfile::VllmClefFlashV1);
    let mut request = canonical_request(2).as_ref().clone();
    request.dialect = dialect;
    let prepared = super::super::head::prepare(&request, &selection, 100).unwrap();
    let admission = Arc::new(Semaphore::new(2));
    let permit = admission.clone().acquire_many_owned(2).await.unwrap();
    let parent = Context::new(()).context();
    let response = super::super::head::dispatch(
        selection,
        prepared,
        "head-request".into(),
        parent.clone(),
        permit,
        Arc::new(super::super::Metrics::new()),
        Arc::new(request),
    )
    .await;
    (response, parent, admission)
}

#[tokio::test]
async fn head_dispatch_preserves_both_public_contracts_and_joint_usage() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let engine = HeadEngine::new(Behavior::Success);
        let (response, parent, admission) = dispatch_head(engine.clone(), dialect).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-request-id"], "head-request");
        let body = response_body(response).await;
        assert_eq!(body["usage"]["input_tokens"], 13);
        assert_eq!(body["usage"]["output_tokens"], 0);
        if dialect == dynamo_decisions::Dialect::OpenAi {
            assert_eq!(body["answers"][0]["probability"], 0.25);
            assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 2);
        } else {
            assert_eq!(body["answers"]["question-0"]["noul"], 0.25);
        }
        assert!(!parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
        let calls = engine.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].decision_head.questions.len(), 2);
        assert!(!calls[0].decision_head.cache_salt.is_empty());
    }
}

#[tokio::test]
async fn head_native_validation_preserves_dialect_specific_error_status() {
    for (dialect, status) in [
        (dynamo_decisions::Dialect::OpenAi, StatusCode::BAD_REQUEST),
        (
            dynamo_decisions::Dialect::Jev,
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let (response, parent, admission) =
            dispatch_head(HeadEngine::new(Behavior::InvalidRequest), dialect).await;
        assert_eq!(response.status(), status);
        let body = response_body(response).await;
        assert!(body.get("answers").is_none());
        assert!(!body.to_string().contains("private head details"));
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn head_errors_malformed_and_out_of_order_results_are_not_answers() {
    for behavior in [
        Behavior::Malformed,
        Behavior::Reordered,
        Behavior::Empty,
        Behavior::Duplicate,
        Behavior::Error,
    ] {
        let (response, parent, admission) =
            dispatch_head(HeadEngine::new(behavior), dynamo_decisions::Dialect::OpenAi).await;
        assert_eq!(
            response.status(),
            if matches!(behavior, Behavior::Error) {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        );
        let body = response_body(response).await;
        assert!(body.get("answers").is_none());
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn head_deadline_cancels_child_and_releases_question_weighted_admission() {
    let engine = HeadEngine::new(Behavior::Pending);
    let selection = selected(engine.clone(), DecisionExecutionProfile::VllmD3V1);
    let request = canonical_request(2);
    let prepared = super::super::head::prepare(&request, &selection, 100).unwrap();
    let admission = Arc::new(Semaphore::new(2));
    let permit = admission.clone().acquire_many_owned(2).await.unwrap();
    let parent = Context::new(()).context();
    let task = tokio::spawn(super::super::head::dispatch(
        selection,
        prepared,
        "deadline".into(),
        parent.clone(),
        permit,
        Arc::new(super::super::Metrics::new()),
        request,
    ));
    tokio::time::timeout(Duration::from_secs(2), engine.started.notified())
        .await
        .unwrap();
    let response = super::super::response::dispatch_with_deadline(
        task,
        tokio::time::Instant::now() + Duration::from_millis(10),
        parent.clone(),
        dynamo_decisions::Dialect::Jev,
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert!(parent.is_killed());
    assert!(
        engine
            .contexts
            .lock()
            .unwrap()
            .iter()
            .all(|context| context.is_killed())
    );
    assert_eq!(admission.available_permits(), 2);
}

#[test]
fn head_discovery_needs_no_rust_tokenizer_and_fences_retired_admissions() {
    for profile in [
        DecisionExecutionProfile::VllmD3V1,
        DecisionExecutionProfile::VllmClefFlashV1,
        DecisionExecutionProfile::VllmClefOmniV1,
    ] {
        let selected = selected(HeadEngine::new(Behavior::Success), profile);
        assert!(!selected.card.has_tokenizer());
        let mut workers = WorkerSet::new(
            "heads".into(),
            selected.card.mdcsum().into(),
            selected.card.clone(),
        );
        if let DecisionExecutor::Head { engine } = selected.executor {
            workers.decision_head_engine = Some(engine);
        }
        let (sender, receiver) = tokio::sync::watch::channel(vec![7]);
        workers.set_instance_watcher(receiver);
        let model = Model::new("test-model".into());
        model.add_worker_set("heads".into(), Arc::new(workers));
        let selection = model.get_systemone_execution_selection().unwrap();
        assert!(matches!(selection.executor, DecisionExecutor::Head { .. }));
        assert!(selection.token_parts().is_none());
        sender.send(vec![]).unwrap();
        assert!(selection.validate_dispatch(None).is_err());
    }
}

#[test]
fn head_capabilities_and_checksums_fail_closed_without_affecting_legacy_cards() {
    let base = selected(
        HeadEngine::new(Behavior::Success),
        DecisionExecutionProfile::VllmD3V1,
    )
    .card;
    let mut absent = base.clone();
    absent.runtime_config.runtime_data.clear();
    let mut disabled = absent.clone();
    disabled
        .runtime_config
        .runtime_data
        .insert("vllm_d3_decision_v1".into(), false.into());
    assert_eq!(absent.mdcsum(), disabled.mdcsum());
    assert_ne!(base.mdcsum(), absent.mdcsum());
    let mut dp = base.clone();
    dp.runtime_config.data_parallel_size = 2;
    assert!(dp.decision_execution_profile().is_none());
    let mut ambiguous = base.clone();
    ambiguous
        .runtime_config
        .runtime_data
        .insert("vllm_clef_flash_joint_decision_v1".into(), true.into());
    assert!(ambiguous.decision_execution_profile().is_none());
}
