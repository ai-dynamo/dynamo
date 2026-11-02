// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::tests::{StubGenerateEngine, systemone_worker_set};
use super::*;
use crate::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY;
use crate::model_card::{LoraInfo, ModelDeploymentCard};
use tokio::sync::watch;

fn worker_token_executor(workers: &super::WorkerSet) -> &super::TokenExecutor {
    let Some(super::DecisionRoute::Available(super::DecisionEngine::Token(executor))) =
        &workers.decision
    else {
        panic!("token worker fixture")
    };
    executor
}
fn token_executor(selection: &super::SystemOneExecutionSelection) -> &super::TokenExecutor {
    let super::DecisionEngine::Token(executor) = selection.engine() else {
        panic!("expected token executor")
    };
    executor
}
#[test]
fn systemone_selection_rejects_a_mismatched_token_backend() {
    let model = Model::new("test-model".into());
    let (mut workers, _admissions) = systemone_worker_set("mismatched", |_| {});
    let executor = worker_token_executor(&workers);
    let mismatched = TokenExecutor::new(
        crate::local_model::runtime_config::TokenBackend::VllmCausal,
        executor.engine().clone(),
        executor.preprocessor().clone(),
    );
    Arc::get_mut(&mut workers).unwrap().decision =
        Some(DecisionRoute::Available(DecisionEngine::Token(mismatched)));
    assert!(SystemOneExecutionSelection::from_worker_set("test-model".into(), &workers).is_none());
    model.add_worker_set("mismatched".into(), workers);
    assert!(!model.has_decision_engine());
    assert!(model.get_systemone_execution_selection().is_err());
}

#[test]
fn vllm_decision_selection_uses_its_own_profile_and_limits() {
    use crate::local_model::runtime_config::{
        DecisionExecutionProfile, SGLANG_GENERATE_CAPABILITY, SGLANG_SYSTEMONE_SERIAL_V1,
        VLLM_CAUSAL_DECISION_V1,
    };

    for max_num_seqs in [1, 16] {
        let model = Model::new("scoring-model".into());
        let (worker_set, _workers) = systemone_worker_set("vllm", |card| {
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
        });
        model.add_worker_set("vllm".into(), worker_set.clone());
        assert!(!model.has_generate_engine());
        if max_num_seqs != 1 {
            assert!(matches!(
                model.get_systemone_execution_selection(),
                Err(SystemOneSelectionError::Unsupported)
            ));
            continue;
        }
        let selected = model.get_systemone_execution_selection().unwrap();
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
            &token_executor(&selected).engine,
            worker_token_executor(&worker_set).engine()
        ));
        assert!(Arc::ptr_eq(
            &token_executor(&selected).preprocessor,
            worker_token_executor(&worker_set).preprocessor()
        ));
    }
}

#[test]
fn systemone_selection_keeps_engine_card_and_assets_atomic() {
    let model = Model::new("scoring-model".to_string());
    let (worker_set_a, workers_a) = systemone_worker_set("ns-a", |card| {
        card.kv_cache_block_size = 16;
    });
    let (worker_set_b, workers_b) = systemone_worker_set("ns-b", |card| {
        card.kv_cache_block_size = 32;
    });
    workers_b.send(vec![]).unwrap();
    model.add_worker_set("ns-a".to_string(), worker_set_a.clone());
    model.add_worker_set("ns-b".to_string(), worker_set_b.clone());
    let selection_a = model.get_systemone_execution_selection().unwrap();
    assert_eq!(
        Some(selection_a.context.decision_capabilities().max_candidates),
        selection_a
            .context
            .card
            .decision_max_candidates(selection_a.context.profile())
    );
    assert!(Arc::ptr_eq(
        &token_executor(&selection_a).engine,
        worker_token_executor(&worker_set_a).engine()
    ));
    assert!(Arc::ptr_eq(
        &token_executor(&selection_a).preprocessor,
        worker_token_executor(&worker_set_a).preprocessor()
    ));
    assert_eq!(selection_a.context.card.kv_cache_block_size, 16);
    assert_eq!(selection_a.context.canonical_model, "scoring-model");

    workers_a.send(vec![]).unwrap();
    workers_b.send(vec![2]).unwrap();
    let selection_b = model.get_systemone_execution_selection().unwrap();
    assert!(Arc::ptr_eq(
        &token_executor(&selection_b).engine,
        worker_token_executor(&worker_set_b).engine()
    ));
    assert!(Arc::ptr_eq(
        &token_executor(&selection_b).preprocessor,
        worker_token_executor(&worker_set_b).preprocessor()
    ));
    assert_eq!(selection_b.context.card.kv_cache_block_size, 32);
    assert_eq!(selection_a.context.card.kv_cache_block_size, 16);
}

#[test]
fn systemone_selection_distinguishes_unsupported_from_unavailable() {
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
        let (worker_set, _workers) = systemone_worker_set("ns", configure);
        model.add_worker_set("ns".to_string(), worker_set);
        assert!(matches!(
            model.get_systemone_execution_selection(),
            Err(SystemOneSelectionError::Unsupported)
        ));
    }
    let model = Model::new("scoring-model".to_string());
    let (mut worker_set, _workers) = systemone_worker_set("ns", |_| {});
    Arc::get_mut(&mut worker_set).unwrap().decision = None;
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(matches!(
        model.get_systemone_execution_selection(),
        Err(SystemOneSelectionError::Unavailable)
    ));
    let model = Model::new("scoring-model".to_string());
    let (mut worker_set, _workers) = systemone_worker_set("ns", |_| {});
    let worker = Arc::get_mut(&mut worker_set).unwrap();
    worker.decision = None;
    worker.generate_engine = Some(Arc::new(StubGenerateEngine));
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(model.get_generate_engine().is_ok());
    assert!(matches!(
        model.get_systemone_execution_selection(),
        Err(SystemOneSelectionError::Unavailable)
    ));
    let model = Model::new("scoring-model".to_string());
    let (worker_set, workers) = systemone_worker_set("ns", |_| {});
    workers.send(vec![]).unwrap();
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(matches!(
        model.get_systemone_execution_selection(),
        Err(SystemOneSelectionError::Unavailable)
    ));
}

#[test]
fn systemone_selection_ignores_unrelated_card_migration_limit() {
    let model = Model::new("scoring-model".to_string());
    let (worker_set, _workers) = systemone_worker_set("ns", |card| {
        card.migration_limit = 1;
    });
    model.add_worker_set("ns".to_string(), worker_set);
    assert!(model.get_systemone_execution_selection().is_ok());
}

#[test]
fn systemone_retained_selection_rejects_lifecycle_retirement() {
    let model = Model::new("scoring-model".to_string());
    let (mut workers, _tx) = systemone_worker_set("ns", |_| {});
    let cancellation = tokio_util::sync::CancellationToken::new();
    Arc::get_mut(&mut workers)
        .unwrap()
        .set_lifecycle_cancellation(cancellation.clone());
    model.add_worker_set("ns".to_string(), workers);
    let selection = model.get_systemone_execution_selection().unwrap();
    assert_eq!(selection.context.validate_dispatch(Some(1)), Ok(()));
    cancellation.cancel();
    assert_eq!(
        selection.context.validate_dispatch(Some(1)),
        Err(SystemOneSelectionError::Unavailable)
    );
    assert!(matches!(
        model.get_systemone_execution_selection(),
        Err(SystemOneSelectionError::Unavailable)
    ));
}

#[test]
fn systemone_in_process_selection_does_not_require_discovery_admissions() {
    let model = Model::new("scoring-model".to_string());
    let (registered, _tx) = systemone_worker_set("ns", |_| {});
    let mut local = WorkerSet::new(
        "local".to_string(),
        "local".to_string(),
        registered.card().clone(),
    );
    local.decision = registered.decision.clone();
    let cancellation = tokio_util::sync::CancellationToken::new();
    local.set_lifecycle_cancellation(cancellation.clone());
    assert_eq!(local.decision_worker_count(), 1);
    model.add_worker_set("local".to_string(), Arc::new(local));
    let selection = model.get_systemone_execution_selection().unwrap();
    assert!(selection.context.admitted_ids.is_none());
    assert_eq!(selection.context.validate_dispatch(None), Ok(()));
    cancellation.cancel();
    assert_eq!(
        selection.context.validate_dispatch(None),
        Err(SystemOneSelectionError::Unavailable)
    );
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
    let (mut unadmitted, discovered) = systemone_worker_set("ns-a", |_| {});
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
        model.get_systemone_execution_selection(),
        Err(SystemOneSelectionError::Unavailable)
    ));
    let (available, _available_tx) = systemone_worker_set("ns-b", |_| {});
    model.add_worker_set("ns-b".to_string(), available.clone());
    let selection = model.get_systemone_execution_selection().unwrap();
    assert!(Arc::ptr_eq(
        &token_executor(&selection).engine,
        worker_token_executor(&available).engine()
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
    let selection = model.get_systemone_execution_selection().unwrap();
    assert!(Arc::ptr_eq(
        &token_executor(&selection).engine,
        worker_token_executor(&available).engine()
    ));
    runtime.shutdown();
}
