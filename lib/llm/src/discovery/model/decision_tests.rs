// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::tests::{StubGenerateEngine, systemone_worker_set};
use super::*;
use crate::{
    decision_profile::{
        DecisionExecutionProfile, SGLANG_SYSTEMONE_SERIAL_V1, VLLM_CAUSAL_DECISION_V1,
    },
    discovery::{DecisionEngine, SystemOneExecutionSelection, TokenExecutor},
    local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    model_card::{LoraInfo, ModelDeploymentCard},
};
use tokio::sync::watch;

fn worker_token_executor(
    workers: &super::WorkerSet,
) -> std::sync::Arc<super::super::TokenResources> {
    let Some(resources) = &workers.decision else {
        panic!("token worker fixture")
    };
    resources
        .resources
        .prepared_token()
        .expect("token worker fixture")
}
fn token_executor(selection: &SystemOneExecutionSelection) -> &TokenExecutor {
    let DecisionEngine::Token(executor) = &selection.engine else {
        panic!("expected token executor")
    };
    executor
}
#[tokio::test]
async fn recoverable_decision_alternate_takes_precedence_over_failed_qualification() {
    use crate::discovery::{DecisionAvailability, test_support};
    let model = Model::new("test-model".into());
    let (mut failed, _failed_workers, _failed_qualification) =
        systemone_worker_set("failed", |_| {}).await;
    let (failure, fleet) = test_support::fleet(None, vec![]);
    failure.send_replace(DecisionAvailability::Failed);
    Arc::get_mut(&mut failed)
        .unwrap()
        .decision
        .as_mut()
        .unwrap()
        .fleet = fleet;
    model.add_worker_set("failed".into(), failed);
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::QualificationFailed)
    ));
    let (mut alternative, workers, _alternative_qualification) =
        systemone_worker_set("alternative", |_| {}).await;
    let qualification = alternative.card().decision_qualification().unwrap();
    let (publication, fleet) = test_support::fleet(Some(qualification), vec![1]);
    Arc::get_mut(&mut alternative)
        .unwrap()
        .decision
        .as_mut()
        .unwrap()
        .fleet = fleet;
    workers.send_replace(vec![]);
    model.add_worker_set("alternative".into(), alternative);
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unavailable)
    ));
    publication.send_replace(DecisionAvailability::Pending);
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unavailable)
    ));
    test_support::publish(&publication, Some(qualification), vec![1]);
    workers.send_replace(vec![1]);
    assert!(model.prepared_selection_for_test().await.is_ok());
}

#[tokio::test]
async fn decision_presence_survives_admission_churn() {
    let model = Model::new("test-model".into());
    let (workers, admissions, _qualification) = systemone_worker_set("decisions", |_| {}).await;
    model.add_worker_set("decisions".into(), workers);
    assert!(model.has_decision_engine());
    assert!(model.prepared_selection_for_test().await.is_ok());

    admissions.send_replace(Vec::new());
    assert!(model.has_decision_engine());
    assert!(!model.is_ready_to_serve());
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unavailable)
    ));

    admissions.send_replace(vec![1]);
    assert!(model.has_decision_engine());
    assert!(model.prepared_selection_for_test().await.is_ok());
    drop(admissions);
    assert!(model.has_decision_engine());
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unavailable)
    ));
}

#[tokio::test]
async fn systemone_selection_derives_token_backend_from_live_qualification() {
    use crate::decision_profile::{DecisionExecutionProfile as Profile, TokenBackend};
    let (mut workers, _ordinary, _qualification) = systemone_worker_set("decisions", |_| {}).await;
    let initial = (Profile::SglangSerialV1, 255);
    let (qualifications, fleet) = crate::discovery::test_support::fleet(Some(initial), vec![1]);
    Arc::get_mut(&mut workers)
        .unwrap()
        .decision
        .as_mut()
        .unwrap()
        .fleet = fleet;
    let first =
        crate::discovery::test_support::prepared_selection_for_test("model".into(), &workers)
            .await
            .unwrap();
    assert_eq!(token_executor(&first).backend(), TokenBackend::Sglang);
    crate::discovery::test_support::publish(
        &qualifications,
        Some((Profile::VllmCausalV1, 26)),
        vec![1],
    );
    let second =
        crate::discovery::test_support::prepared_selection_for_test("model".into(), &workers)
            .await
            .unwrap();
    assert_eq!(token_executor(&second).backend(), TokenBackend::VllmCausal);
    assert!(Arc::ptr_eq(
        token_executor(&first).engine(),
        token_executor(&second).engine()
    ));
    assert!(first.context.validate_dispatch(Some(1)).is_err());
    assert!(second.context.validate_dispatch(Some(1)).is_ok());
}

#[tokio::test]
async fn vllm_decision_selection_uses_its_own_profile_and_limits() {
    use crate::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY;

    for max_num_seqs in [1, 16] {
        let model = Model::new("scoring-model".into());
        let (worker_set, _workers, _qualification) = systemone_worker_set("vllm", |card| {
            card.model_input = crate::model_type::ModelInput::Tokens;
            card.model_type = crate::model_type::ModelType::Chat;
            card.runtime_config
                .runtime_data
                .remove(SGLANG_GENERATE_CAPABILITY);
            card.runtime_config
                .runtime_data
                .remove(SGLANG_SYSTEMONE_SERIAL_V1);
            card.runtime_config
                .runtime_data
                .insert(VLLM_INFERENCE_V1_GENERATE_CAPABILITY.into(), true.into());
            card.runtime_config
                .runtime_data
                .insert(VLLM_CAUSAL_DECISION_V1.into(), true.into());
            card.runtime_config.max_num_seqs = Some(max_num_seqs);
            card.runtime_config.data_parallel_size = 1;
        })
        .await;
        model.add_worker_set("vllm".into(), worker_set.clone());
        assert!(!model.has_generate_engine());
        if max_num_seqs != 1 {
            assert!(matches!(
                model.prepared_selection_for_test().await,
                Err(SystemOneSelectionError::Unsupported)
            ));
            continue;
        }
        let selected = model.prepared_selection_for_test().await.unwrap();
        assert_eq!(
            selected.context.profile(),
            DecisionExecutionProfile::VllmCausalV1
        );
        assert_eq!(selected.context.decision_capabilities().max_questions, 64);
        assert_eq!(selected.context.decision_capabilities().max_candidates, 26);
        assert_eq!(
            Some(selected.context.decision_capabilities().max_candidates),
            selected
                .context
                .card
                .decision_max_candidates(selected.context.profile())
        );
        assert!(Arc::ptr_eq(
            token_executor(&selected).engine(),
            &worker_token_executor(&worker_set).engine
        ));
        assert!(Arc::ptr_eq(
            token_executor(&selected).preprocessor(),
            &worker_token_executor(&worker_set).preprocessor
        ));
    }
}

#[tokio::test]
async fn systemone_selection_keeps_engine_card_and_assets_atomic() {
    let model = Model::new("scoring-model".to_string());
    let (worker_set_a, workers_a, _qualification_a) = systemone_worker_set("ns-a", |card| {
        card.kv_cache_block_size = 16;
    })
    .await;
    let (worker_set_b, workers_b, qualification_b) = systemone_worker_set("ns-b", |card| {
        card.kv_cache_block_size = 32;
    })
    .await;
    workers_b.send(vec![]).unwrap();
    model.add_worker_set("ns-a".to_string(), worker_set_a.clone());
    model.add_worker_set("ns-b".to_string(), worker_set_b.clone());
    let selection_a = model.prepared_selection_for_test().await.unwrap();
    assert_eq!(
        Some(selection_a.context.decision_capabilities().max_candidates),
        selection_a
            .context
            .card
            .decision_max_candidates(selection_a.context.profile())
    );
    assert!(Arc::ptr_eq(
        token_executor(&selection_a).engine(),
        &worker_token_executor(&worker_set_a).engine
    ));
    assert!(Arc::ptr_eq(
        token_executor(&selection_a).preprocessor(),
        &worker_token_executor(&worker_set_a).preprocessor
    ));
    assert_eq!(selection_a.context.card.kv_cache_block_size, 16);
    assert_eq!(selection_a.context.canonical_model, "scoring-model");

    workers_a.send(vec![]).unwrap();
    workers_b.send(vec![2]).unwrap();
    crate::discovery::test_support::publish(
        &qualification_b,
        worker_set_b.card().decision_qualification(),
        vec![2],
    );
    let selection_b = model.prepared_selection_for_test().await.unwrap();
    assert!(Arc::ptr_eq(
        token_executor(&selection_b).engine(),
        &worker_token_executor(&worker_set_b).engine
    ));
    assert!(Arc::ptr_eq(
        token_executor(&selection_b).preprocessor(),
        &worker_token_executor(&worker_set_b).preprocessor
    ));
    assert_eq!(selection_b.context.card.kv_cache_block_size, 32);
    assert_eq!(selection_a.context.card.kv_cache_block_size, 16);
}

#[tokio::test]
async fn systemone_selection_distinguishes_unsupported_from_unavailable() {
    let unsupported: [fn(&mut ModelDeploymentCard); 9] = [
        |card| {
            card.runtime_config
                .runtime_data
                .remove("sglang_systemone_serial_v1");
        },
        |card| {
            card.runtime_config.max_num_seqs = None;
        },
        |card| {
            card.runtime_config.max_num_seqs = Some(2);
        },
        |card| {
            card.runtime_config
                .runtime_data
                .remove(crate::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY);
        },
        |card| {
            card.worker_type = None;
        },
        |card| {
            card.worker_type = Some(crate::worker_type::WorkerType::Prefill);
        },
        |card| {
            card.lora = Some(LoraInfo {
                name: "adapter".to_string(),
                max_gpu_lora_count: None,
            });
        },
        |card| {
            card.needs = vec![vec![crate::worker_type::WorkerType::Prefill]];
        },
        |card| {
            card.runtime_config
                .runtime_data
                .insert("spec_decode".to_string(), serde_json::json!({}));
        },
    ];
    for configure in unsupported {
        let model = Model::new("scoring-model".to_string());
        let (worker_set, _workers, _qualification) = systemone_worker_set("ns", configure).await;
        model.add_worker_set("ns".to_string(), worker_set);
        assert!(matches!(
            model.prepared_selection_for_test().await,
            Err(SystemOneSelectionError::Unsupported)
        ));
    }
    let model = Model::new("scoring-model".to_string());
    let (mut worker_set, _workers, _qualification) = systemone_worker_set("ns", |_| {}).await;
    Arc::get_mut(&mut worker_set).unwrap().decision = None;
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unsupported)
    ));
    let model = Model::new("scoring-model".to_string());
    let (mut worker_set, _workers, _qualification) = systemone_worker_set("ns", |_| {}).await;
    let worker = Arc::get_mut(&mut worker_set).unwrap();
    worker.decision = None;
    worker.generate_engine = Some(Arc::new(StubGenerateEngine));
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(model.get_generate_engine().is_ok());
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unsupported)
    ));
    let model = Model::new("scoring-model".to_string());
    let (worker_set, workers, _qualification) = systemone_worker_set("ns", |_| {}).await;
    workers.send(vec![]).unwrap();
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unavailable)
    ));
}

#[tokio::test]
async fn systemone_selection_ignores_unrelated_card_migration_limit() {
    let model = Model::new("scoring-model".to_string());
    let (worker_set, _workers, _qualification) = systemone_worker_set("ns", |card| {
        card.migration_limit = 1;
    })
    .await;
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(model.prepared_selection_for_test().await.is_ok());
}

#[tokio::test]
async fn systemone_retained_selection_rejects_lifecycle_retirement() {
    let model = Model::new("scoring-model".to_string());
    let (workers, _tx, _qualification) = systemone_worker_set("ns", |_| {}).await;
    let cancellation = workers.lifecycle_cancellation().unwrap().clone();
    model.add_worker_set("ns".to_string(), workers);
    let selection = model.prepared_selection_for_test().await.unwrap();
    assert_eq!(selection.context.validate_dispatch(Some(1)), Ok(()));
    assert!(model.has_decision_engine());
    cancellation.cancel();
    assert!(model.has_decision_engine());
    assert_eq!(
        selection.context.validate_dispatch(Some(1)),
        Err(SystemOneSelectionError::Unavailable)
    );
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unavailable)
    ));
}

#[tokio::test]
async fn systemone_in_process_selection_requires_explicit_decision_surface() {
    let model = Model::new("scoring-model".to_string());
    let (registered, _tx, _qualification) = systemone_worker_set("ns", |_| {}).await;
    let local = WorkerSet::new(
        "local".to_string(),
        "local".to_string(),
        registered.card().clone(),
    );
    assert_eq!(local.decision_worker_count(), 0);
    model.add_worker_set("local".to_string(), Arc::new(local));
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unsupported)
    ));
}

#[tokio::test]
async fn systemone_selection_excludes_discovered_but_unadmitted_workers() {
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let component = distributed
        .namespace("decision-admission-test".to_string())
        .unwrap()
        .component("workers".to_string())
        .unwrap();
    let model = Model::new("scoring-model".to_string());
    let (mut unadmitted, discovered, _unadmitted_qualification) =
        systemone_worker_set("ns-a", |_| {}).await;
    let (admitted_tx, admitted_rx) = watch::channel(Vec::new());
    let selected_card = Arc::new(unadmitted.card().clone());
    Arc::get_mut(&mut unadmitted).unwrap().set_topology_target(
        super::super::worker_set::CommittedWorkerSetTarget {
            endpoint: component.endpoint("generate".to_string()),
            group: "group-a".to_string(),
            generation: 1,
            card: selected_card,
            admitted_ids: admitted_rx,
        },
    );
    model.add_worker_set("ns-a".to_string(), unadmitted);
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(SystemOneSelectionError::Unavailable)
    ));
    let (available, _available_tx, _available_qualification) =
        systemone_worker_set("ns-b", |_| {}).await;
    model.add_worker_set("ns-b".to_string(), available.clone());
    let selection = model.prepared_selection_for_test().await.unwrap();
    assert!(Arc::ptr_eq(
        token_executor(&selection).engine(),
        &worker_token_executor(&available).engine
    ));
    admitted_tx.send(vec![1, 2]).unwrap();
    let workers = model.get_worker_set("ns-a").unwrap();
    assert_eq!(workers.worker_count(), 1);
    assert_eq!(workers.decision_worker_count(), 1);
    admitted_tx.send(vec![2]).unwrap();
    assert_eq!(workers.decision_worker_count(), 0);
    admitted_tx.send(vec![1]).unwrap();
    assert_eq!(workers.decision_worker_count(), 1);
    drop(discovered);
    assert_eq!(workers.decision_worker_count(), 0);
    drop(admitted_tx);
    assert_eq!(workers.decision_worker_count(), 0);
    let selection = model.prepared_selection_for_test().await.unwrap();
    assert!(Arc::ptr_eq(
        token_executor(&selection).engine(),
        &worker_token_executor(&available).engine
    ));
    runtime.shutdown();
}
