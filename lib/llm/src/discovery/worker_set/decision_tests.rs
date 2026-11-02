// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::tests::StubEngine;
use super::*;
use crate::{
    decision_profile::DecisionExecutionProfile,
    discovery::{DecisionEngine, DecisionResources},
    protocols::common::{llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest},
};

fn decision_card(profile: DecisionExecutionProfile) -> ModelDeploymentCard {
    use crate::decision_profile::{DecisionProfileKind, NativeExecutorKind};
    use crate::model_type::{ModelInput, ModelType};
    let mut card = ModelDeploymentCard::with_name_only("decision-model");
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config
        .runtime_data
        .insert(profile.decision_capability().into(), true.into());
    match profile.kind() {
        DecisionProfileKind::Token(backend) => {
            card.model_input = ModelInput::Tokens;
            card.model_type = ModelType::Chat;
            card.runtime_config.max_num_seqs = Some(1);
            card.runtime_config.runtime_data.insert(
                crate::model_card::decision_profile::transport_capability(backend).into(),
                true.into(),
            );
        }
        DecisionProfileKind::Native(kind) => {
            (card.model_input, card.model_type) = match kind {
                NativeExecutorKind::Head => {
                    (ModelInput::Text, ModelType::Classify | ModelType::Pooling)
                }
                NativeExecutorKind::Read => (ModelInput::Tokens, ModelType::Chat),
            };
            card.runtime_config.runtime_data.insert(
                crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES.into(),
                3.into(),
            );
        }
    }
    card
}

#[tokio::test]
async fn decision_qualification_does_not_change_structural_worker_role() {
    use crate::discovery::Model;
    use crate::protocols::decision_read::{DecisionReadRequest, DecisionReadResponse};
    let qualification = (DecisionExecutionProfile::VllmLiquidV1, 3);
    let mut workers = WorkerSet::new(
        "decisions".into(),
        "revision".into(),
        decision_card(qualification.0),
    );
    let resources =
        crate::discovery::test_support::native_resources(DecisionEngine::Read(StubEngine::<
            DecisionReadRequest,
            DecisionReadResponse,
        >::new()));
    let (_ordinary, ordinary_rx) = watch::channel(vec![1]);
    workers.set_instance_watcher(ordinary_rx);
    let (qualifications, fleet) =
        crate::discovery::test_support::fleet(Some(qualification), vec![1]);
    crate::discovery::test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    let workers = Arc::new(workers);
    let model = Model::new("model".into());
    model.add_worker_set("decisions".into(), workers.clone());
    assert!(!workers.is_prefill_set());
    assert!(!model.has_prefill());
    assert!(model.prepared_selection_for_test().await.is_ok());
    assert!(model.is_ready_to_serve());
    qualifications.send_replace(crate::discovery::DecisionAvailability::Unadvertised);
    assert!(workers.has_any_serving_engine());
    assert!(!workers.is_prefill_set());
    assert!(!model.has_prefill());
    assert!(model.prepared_selection_for_test().await.is_err());
    assert!(!model.is_ready_to_serve());
    drop(qualifications);
    assert!(workers.has_any_serving_engine());
    assert!(!workers.is_prefill_set());
    assert!(!model.has_prefill());
    assert!(!model.is_ready_to_serve());
}

#[tokio::test]
async fn known_qualified_empty_decision_snapshot_is_unavailable() {
    use crate::discovery::{Model, SystemOneSelectionError};
    use crate::protocols::decision_read::{DecisionReadRequest, DecisionReadResponse};
    let qualification = (DecisionExecutionProfile::VllmLiquidV1, 3);
    let mut workers = WorkerSet::new(
        "decisions".into(),
        "revision".into(),
        decision_card(qualification.0),
    );
    let resources =
        crate::discovery::test_support::native_resources(DecisionEngine::Read(StubEngine::<
            DecisionReadRequest,
            DecisionReadResponse,
        >::new()));
    let (_ordinary, ordinary_rx) = watch::channel(vec![1]);
    workers.set_instance_watcher(ordinary_rx.clone());
    let (_sender, fleet) = crate::discovery::test_support::fleet(Some(qualification), vec![]);
    let context = crate::discovery::test_support::context_for_test(fleet.clone(), ordinary_rx);
    assert!(context.validate_dispatch(None).is_err());
    crate::discovery::test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    assert!(workers.decision_state().is_qualified());
    assert!(workers.has_decision_engine());
    assert_eq!(workers.decision_worker_count(), 0);
    let model = Model::new("model".into());
    model.add_worker_set("decisions".into(), Arc::new(workers));
    assert!(!model.is_ready_to_serve());
    assert!(matches!(
        model.get_systemone_execution_plan(),
        Err(SystemOneSelectionError::Unavailable)
    ));
}

#[tokio::test]
async fn cold_decision_plan_without_admitted_workers_is_transiently_unavailable() {
    use crate::discovery::{Model, ScoreFactory, SystemOneSelectionError};
    let (_runtime, load) = crate::discovery::test_support::routing_load_for_test().await;
    let card = decision_card(DecisionExecutionProfile::VllmCausalV1);
    let mut workers = WorkerSet::new("decisions".into(), "revision".into(), card.clone());
    let teardown = workers.initialize_allocator_trim_on_teardown();
    let resources = DecisionResources::for_worker(
        card,
        Some(ScoreFactory::new(
            StubEngine::<PreprocessedRequest, LLMEngineOutput>::new(),
            crate::discovery::TokenizerLoad::Card,
        )),
        Some(load),
        dynamo_runtime::pipeline::RouterMode::RoundRobin,
        teardown,
    )
    .into_shared();
    let (_ordinary, ordinary_rx) = watch::channel(vec![]);
    workers.set_instance_watcher(ordinary_rx);
    let _qualification =
        crate::discovery::test_support::attach_card_surface_for_test(&mut workers, resources).await;
    assert!(workers.has_decision_engine());
    assert!(workers.decision_state().is_qualified());
    let model = Model::new("model".into());
    model.add_worker_set("decisions".into(), Arc::new(workers));
    assert!(matches!(
        model.get_systemone_execution_plan(),
        Err(SystemOneSelectionError::Unavailable)
    ));
}

#[tokio::test]
async fn decision_state_keeps_cold_readiness_and_structural_role_consistent() {
    assert!(DecisionResources::default().into_shared().is_none());
    use crate::discovery::{Model, ScoreFactory, SystemOneSelectionError};
    let (_runtime, load) = crate::discovery::test_support::routing_load_for_test().await;
    let card = decision_card(DecisionExecutionProfile::VllmCausalV1);
    let tokenizer_card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    let preprocessor = crate::preprocessor::OpenAIPreprocessor::new(tokenizer_card).unwrap();
    let qualification = (DecisionExecutionProfile::VllmCausalV1, 26);
    for (name, prepared, qualified, live, closed) in [
        ("cold", false, true, true, false),
        ("prepared", true, true, true, false),
        ("empty", false, true, false, false),
        ("unqualified", false, false, true, false),
        ("closed", false, true, true, true),
    ] {
        let mut workers = WorkerSet::new("decisions".into(), "revision".into(), card.clone());
        let mut resources = DecisionResources::for_worker(
            card.clone(),
            Some(ScoreFactory::new(
                StubEngine::<PreprocessedRequest, LLMEngineOutput>::new(),
                crate::discovery::TokenizerLoad::Card,
            )),
            Some(load.clone()),
            dynamo_runtime::pipeline::RouterMode::RoundRobin,
            workers.initialize_allocator_trim_on_teardown(),
        );
        if prepared {
            resources.insert_token(crate::discovery::TokenResources {
                engine: StubEngine::<PreprocessedRequest, LLMEngineOutput>::new(),
                preprocessor: preprocessor.clone(),
            });
        }
        let resources = resources.into_shared().unwrap();
        let (_ordinary, ordinary_rx) = watch::channel(vec![1]);
        workers.set_instance_watcher(ordinary_rx);
        let (publisher, fleet) = crate::discovery::test_support::fleet(
            qualified.then_some(qualification),
            if live { vec![1] } else { vec![] },
        );
        crate::discovery::test_support::attach_surface_for_test(
            &mut workers,
            Some(resources.clone()),
            fleet,
        )
        .await;
        let _publisher = if closed {
            drop(publisher);
            None
        } else {
            Some(publisher)
        };
        let workers = Arc::new(workers);
        let model = Model::new("model".into());
        model.add_worker_set("decisions".into(), workers.clone());
        let ready = qualified && live && !closed;
        assert_eq!(
            (
                workers.has_any_serving_engine(),
                workers.has_decision_engine(),
                workers.is_prefill_set(),
                model.is_ready_to_serve(),
                model.is_displayable(),
                model.get_systemone_execution_plan().is_ok()
            ),
            (true, qualified && !closed, false, ready, ready, ready),
            "{name}",
        );
        assert_eq!(
            resources.is_prepared(qualification.0.kind()),
            prepared,
            "readiness must not initialize resources: {name}"
        );
        if !ready {
            let expected = if qualified || closed {
                SystemOneSelectionError::Unavailable
            } else {
                SystemOneSelectionError::Unsupported
            };
            assert_eq!(
                model.get_systemone_execution_plan().err(),
                Some(expected),
                "{name}"
            );
        }
    }
}

#[tokio::test]
async fn prebuilt_decision_resources_follow_live_profile_kind_without_reconstruction() {
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    use crate::discovery::{Model, TokenResources};
    use crate::protocols::decision_read::{DecisionReadRequest, DecisionReadResponse};
    let tokenizer_card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    let preprocessor = crate::preprocessor::OpenAIPreprocessor::new(tokenizer_card).unwrap();
    let mut card = decision_card(Profile::VllmCausalV1);
    card.runtime_config
        .runtime_data
        .remove(Profile::VllmCausalV1.decision_capability());
    assert!(card.decision_qualification().is_none());
    let mut workers = WorkerSet::new("decisions".into(), "revision".into(), card);
    let mut resources = DecisionResources::default();
    resources.insert_token(TokenResources {
        engine: StubEngine::<PreprocessedRequest, LLMEngineOutput>::new(),
        preprocessor,
    });
    resources.insert_read(StubEngine::<DecisionReadRequest, DecisionReadResponse>::new());
    let resources = resources.into_shared();
    let (_ordinary, ordinary_rx) = watch::channel(vec![1]);
    workers.set_instance_watcher(ordinary_rx);
    let (qualification, fleet) = crate::discovery::test_support::fleet(None, vec![1]);
    crate::discovery::test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    let workers = Arc::new(workers);
    let model = Model::new("model".into());
    model.add_worker_set("decisions".into(), workers.clone());
    assert!(!model.is_ready_to_serve());
    assert!(!workers.is_prefill_set());
    for profile in [
        Profile::SglangSerialV1,
        Profile::VllmCausalV1,
        Profile::VllmLiquidV1,
        Profile::VllmDiffusionV1,
    ] {
        crate::discovery::test_support::publish(
            &qualification,
            Some((profile, profile.limits().1)),
            vec![1],
        );
        let selection = model.prepared_selection_for_test().await.unwrap();
        assert_eq!(selection.context.profile(), profile);
        match &selection.engine {
            DecisionEngine::Token(executor) => {
                assert_eq!(
                    profile.kind(),
                    crate::decision_profile::DecisionProfileKind::Token(executor.backend())
                );
            }
            DecisionEngine::Read(_) => assert!(matches!(
                profile,
                Profile::VllmLiquidV1 | Profile::VllmDiffusionV1
            )),
            DecisionEngine::Head(_) => panic!("head resources were not prepared"),
        }
        assert!(model.is_ready_to_serve());
        assert_eq!(workers.worker_count(), 1);
        assert!(Arc::ptr_eq(&workers, &model.worker_sets()[0]));
    }
    crate::discovery::test_support::publish(&qualification, Some((Profile::VllmD3V1, 26)), vec![1]);
    assert!(!model.is_ready_to_serve());
    assert!(model.prepared_selection_for_test().await.is_err());
    assert!(!workers.is_prefill_set());
}

#[tokio::test]
async fn decision_context_revalidates_fleet_profile_capacity_and_generation() {
    use crate::protocols::decision_read::{DecisionReadRequest, DecisionReadResponse};
    let qualification = (DecisionExecutionProfile::VllmLiquidV1, 3);
    let mut workers = WorkerSet::new(
        "decisions".into(),
        "revision".into(),
        decision_card(qualification.0),
    );
    let resources =
        crate::discovery::test_support::native_resources(DecisionEngine::Read(StubEngine::<
            DecisionReadRequest,
            DecisionReadResponse,
        >::new()));
    let (ordinary, ordinary_rx) = watch::channel(vec![1]);
    workers.set_instance_watcher(ordinary_rx);
    let (qualifications, fleet) =
        crate::discovery::test_support::fleet(Some(qualification), vec![1]);
    crate::discovery::test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    let selected =
        crate::discovery::test_support::prepared_selection_for_test("model".into(), &workers)
            .await
            .unwrap();
    assert!(selected.context.validate_dispatch(Some(1)).is_ok());
    for changed in [
        Some((DecisionExecutionProfile::VllmDiffusionV1, 3)),
        Some((DecisionExecutionProfile::VllmLiquidV1, 4)),
        None,
    ] {
        crate::discovery::test_support::publish(&qualifications, changed, vec![1]);
        assert_eq!(workers.has_decision_engine(), changed.is_some());
        assert!(selected.context.validate_dispatch(Some(1)).is_err());
        assert_eq!(workers.worker_count(), 1);
    }
    crate::discovery::test_support::publish(&qualifications, Some(qualification), vec![1]);
    assert!(workers.has_decision_engine());
    assert!(
        selected.context.validate_dispatch(Some(1)).is_err(),
        "revoked snapshots cannot revive after an ABA transition"
    );
    let selected =
        crate::discovery::test_support::prepared_selection_for_test("model".into(), &workers)
            .await
            .unwrap();
    drop(ordinary);
    assert!(
        selected.context.validate_dispatch(Some(1)).is_err(),
        "a retired ordinary generation cannot revive through the shared fleet gate"
    );
    drop(qualifications);
    assert!(!workers.has_decision_engine());
}

#[tokio::test]
async fn decision_route_card_matrix_uses_one_structural_classifier() {
    use crate::decision_profile::{DecisionProfileKind, NativeExecutorKind, TokenBackend};
    use crate::discovery::{Model, SystemOneSelectionError};
    use crate::protocols::{
        decision_head::{DecisionHeadRequest, DecisionHeadResponse},
        decision_read::{DecisionReadRequest, DecisionReadResponse},
    };
    let tokenizer_card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    let preprocessor = crate::preprocessor::OpenAIPreprocessor::new(tokenizer_card).unwrap();
    let routes =
        [
            (None, None),
            (
                crate::discovery::test_support::token_resources(
                    StubEngine::<PreprocessedRequest, LLMEngineOutput>::new(),
                    preprocessor.clone(),
                ),
                Some(DecisionProfileKind::Token(TokenBackend::Sglang)),
            ),
            (
                crate::discovery::test_support::token_resources(
                    StubEngine::<PreprocessedRequest, LLMEngineOutput>::new(),
                    preprocessor,
                ),
                Some(DecisionProfileKind::Token(TokenBackend::VllmCausal)),
            ),
            (
                crate::discovery::test_support::native_resources(DecisionEngine::Head(
                    StubEngine::<DecisionHeadRequest, DecisionHeadResponse>::new(),
                )),
                Some(DecisionProfileKind::Native(NativeExecutorKind::Head)),
            ),
            (
                crate::discovery::test_support::native_resources(DecisionEngine::Read(
                    StubEngine::<DecisionReadRequest, DecisionReadResponse>::new(),
                )),
                Some(DecisionProfileKind::Native(NativeExecutorKind::Read)),
            ),
        ];
    for profile in [
        DecisionExecutionProfile::SglangSerialV1,
        DecisionExecutionProfile::VllmCausalV1,
        DecisionExecutionProfile::VllmD3V1,
        DecisionExecutionProfile::VllmClefFlashV1,
        DecisionExecutionProfile::VllmClefOmniV1,
        DecisionExecutionProfile::VllmLiquidV1,
        DecisionExecutionProfile::VllmDiffusionV1,
    ] {
        let qualified = decision_card(profile);
        let mut missing_profile = qualified.clone();
        missing_profile
            .runtime_config
            .runtime_data
            .remove(profile.decision_capability());
        let mut invalid_shape = qualified.clone();
        invalid_shape.runtime_config.data_parallel_size = 2;
        let mut invalid_capacity = qualified.clone();
        invalid_capacity.runtime_config.runtime_data.insert(
            crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES.into(),
            0.into(),
        );
        let capacity_required = matches!(profile.kind(), DecisionProfileKind::Native(_));
        for (card, valid) in [
            (qualified, true),
            (missing_profile, false),
            (invalid_shape, false),
            (invalid_capacity, !capacity_required),
        ] {
            for (route, kind) in &routes {
                let kind_matches = *kind == Some(profile.kind())
                    || matches!(
                        (kind, profile.kind()),
                        (
                            Some(DecisionProfileKind::Token(_)),
                            DecisionProfileKind::Token(_)
                        )
                    );
                let expected = valid && kind_matches;
                let mut workers =
                    WorkerSet::new("decisions".into(), "revision".into(), card.clone());
                let (_admissions, admitted_ids) = watch::channel(vec![1]);
                workers.set_instance_watcher(admitted_ids);
                let _qualification = crate::discovery::test_support::attach_card_surface_for_test(
                    &mut workers,
                    route.clone(),
                )
                .await;
                assert_eq!(
                    workers.decision_state().is_ready(),
                    expected,
                    "{profile:?}, {kind:?}"
                );
                assert_eq!(workers.has_decision_engine(), expected);
                if let DecisionState::Qualified {
                    profile: selected,
                    capacity,
                    ..
                } = workers.decision_state()
                {
                    assert_eq!(selected, profile);
                    assert_eq!(
                        capacity,
                        if capacity_required {
                            3
                        } else {
                            profile.limits().1
                        }
                    );
                }
                assert_eq!(
                    crate::discovery::test_support::prepared_selection_for_test(
                        "model".into(),
                        &workers
                    )
                    .await
                    .is_some(),
                    expected
                );
                let model = Model::new("model".into());
                model.add_worker_set("decisions".into(), Arc::new(workers));
                assert_eq!(model.has_decision_engine(), expected);
                match model.prepared_selection_for_test().await {
                    Ok(_) => assert!(expected),
                    Err(error) => {
                        assert!(!expected);
                        assert!(matches!(error, SystemOneSelectionError::Unsupported));
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn decision_liveness_uses_the_canonical_worker_lifecycle() {
    use crate::protocols::decision_read::{DecisionReadRequest, DecisionReadResponse};
    let profile = DecisionExecutionProfile::VllmLiquidV1;
    let mut workers = WorkerSet::new(
        "decisions".into(),
        "revision".into(),
        decision_card(profile),
    );
    let (_membership, available) = watch::channel(vec![1]);
    workers.set_instance_watcher(available);
    let (_publication, fleet) = crate::discovery::test_support::fleet(Some((profile, 3)), vec![1]);
    let resources =
        crate::discovery::test_support::native_resources(DecisionEngine::Read(StubEngine::<
            DecisionReadRequest,
            DecisionReadResponse,
        >::new()));
    crate::discovery::test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    let cancellation = CancellationToken::new();
    workers.set_lifecycle_cancellation(cancellation.clone());
    assert_eq!(workers.decision_worker_count(), 1);
    cancellation.cancel();
    assert_eq!(workers.decision_worker_count(), 0);
}
