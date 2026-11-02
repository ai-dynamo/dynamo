// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::test_support::{
    apply_discovery_event, discovered_card, wait_for_model, watch_test_cards_with_decisions,
};
use super::*;
use dynamo_runtime::{Runtime, distributed::DistributedConfig};

#[tokio::test]
async fn decision_only_capability_keeps_mixed_workers_serving_chat() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::decision_only_capability_keeps_mixed_workers_serving_chat"
        ),
        &[
            ("DYN_ROUTER_MIN_INITIAL_WORKERS", "0"),
            ("RUST_MIN_STACK", "4194304"),
        ],
    ) {
        return;
    }
    use crate::decision_profile::{VLLM_LIQUID_DECISION_V1, VLLM_NATIVE_DECISION_MAX_CANDIDATES};
    use dynamo_runtime::{distributed::DiscoveryBackend, storage::kv};
    let runtime = Runtime::from_current().unwrap();
    let store = tempfile::tempdir().unwrap();
    let config = || DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(store.path().into())),
        ..DistributedConfig::process_local()
    };
    let drt = DistributedRuntime::new(runtime.clone(), config())
        .await
        .unwrap();
    let peer = DistributedRuntime::new(runtime.clone(), config())
        .await
        .unwrap();
    for worker in [&drt, &peer] {
        worker
            .namespace("mixed")
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("generate")
            .register_endpoint_instance()
            .await
            .unwrap();
    }
    let first_id = drt.discovery().instance_id();
    let second_id = peer.discovery().instance_id();
    assert_ne!(
        first_id, second_id,
        "the fleet fixture requires distinct worker incarnations"
    );
    let all_workers = drt
        .namespace("mixed")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate")
        .client()
        .await
        .unwrap();
    let mut available = all_workers.instance_avail_watcher();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if available.borrow_and_update().len() == 2 {
                break;
            }
            available.changed().await.unwrap();
        }
    })
    .await
    .expect("both worker endpoints must be discovered before model cards");
    let discovered_ids = all_workers.instance_ids();
    assert!(discovered_ids.contains(&first_id) && discovered_ids.contains(&second_id));
    let mut qualified = ModelDeploymentCard::with_name_only("mixed-decisions");
    qualified.model_input = ModelInput::Text;
    qualified.model_type = ModelType::Chat;
    qualified.worker_type = Some(WorkerType::Aggregated);
    qualified
        .runtime_config
        .set_engine_specific(VLLM_LIQUID_DECISION_V1, true)
        .unwrap();
    qualified
        .runtime_config
        .set_engine_specific(VLLM_NATIVE_DECISION_MAX_CANDIDATES, 26)
        .unwrap();
    let mut ordinary = qualified.clone();
    ordinary
        .runtime_config
        .runtime_data
        .remove(VLLM_LIQUID_DECISION_V1);
    let manager = Arc::new(ModelManager::new());
    let (events, task) = watch_test_cards_with_decisions(
        drt,
        manager.clone(),
        RouterConfig {
            router_mode: RouterMode::RoundRobin,
            ..Default::default()
        },
        true,
    );
    apply_discovery_event(&events, discovered_card("mixed", first_id, &ordinary)).await;
    let model = wait_for_model(&manager, "mixed-decisions", |model| {
        model.has_chat_engine() && model.total_workers() == 1
    })
    .await;
    assert!(!model.has_decision_engine());
    assert!(matches!(
        model.get_systemone_execution_plan(),
        Err(crate::discovery::SystemOneSelectionError::Unsupported)
    ));
    let initial_workers = model.worker_sets().into_iter().next().unwrap();
    apply_discovery_event(&events, discovered_card("mixed", first_id, &qualified)).await;
    let model = wait_for_model(&manager, "mixed-decisions", |model| {
        model.get_systemone_execution_plan().is_ok() && model.total_workers() == 1
    })
    .await;
    assert!(model.has_decision_engine());
    assert!(initial_workers.decision_state().is_ready());
    let Some(resources) = &initial_workers.decision else {
        panic!("late decision kind retains its factory");
    };
    assert!(
        !resources
            .resources
            .is_prepared(crate::decision_profile::DecisionProfileKind::Native(
                crate::decision_profile::NativeExecutorKind::Read
            )),
        "cold readiness must not initialize the native route"
    );
    let plan = model.get_systemone_execution_plan().unwrap();
    plan.prepare_native().await.unwrap();
    let selected = plan.prepare().unwrap();
    assert!(model.has_decision_engine());
    assert!(Arc::ptr_eq(&initial_workers, &model.worker_sets()[0]));
    apply_discovery_event(&events, discovered_card("mixed", second_id, &ordinary)).await;
    let model = wait_for_model(&manager, "mixed-decisions", |model| {
        model.total_workers() == 2
    })
    .await;
    assert!(model.get_chat_engine().is_ok());
    assert!(Arc::ptr_eq(&initial_workers, &model.worker_sets()[0]));
    assert!(!runtime.primary_token().is_cancelled());
    for workers in model.worker_sets() {
        let client = workers.load_context().unwrap().client();
        assert!(client.is_instance_live(first_id));
        assert!(client.is_instance_live(second_id));
    }
    assert!(matches!(
        model.prepared_selection_for_test().await,
        Err(crate::discovery::SystemOneSelectionError::Unavailable)
    ));
    assert!(selected.context.validate_dispatch(None).is_err());
    assert_eq!(qualified.mdcsum(), ordinary.mdcsum());
    apply_discovery_event(&events, discovered_card("mixed", second_id, &qualified)).await;
    let model = wait_for_model(&manager, "mixed-decisions", |model| {
        model.total_workers() == 2 && model.has_decision_engine()
    })
    .await;
    assert!(model.prepared_selection_for_test().await.is_ok());
    assert!(model.get_chat_engine().is_ok());
    assert!(Arc::ptr_eq(&initial_workers, &model.worker_sets()[0]));
    assert!(!runtime.primary_token().is_cancelled());
    let mut conflicting = qualified.clone();
    conflicting
        .runtime_config
        .set_engine_specific(VLLM_NATIVE_DECISION_MAX_CANDIDATES, 25)
        .unwrap();
    apply_discovery_event(&events, discovered_card("mixed", second_id, &conflicting)).await;
    let model = manager.get_committed_model("mixed-decisions").unwrap();
    assert!(matches!(
        model.get_systemone_execution_plan(),
        Err(crate::discovery::SystemOneSelectionError::Unavailable)
    ));
    assert!(model.get_chat_engine().is_ok());
    assert_eq!(model.total_workers(), 2);
    assert!(Arc::ptr_eq(&initial_workers, &model.worker_sets()[0]));
    apply_discovery_event(&events, discovered_card("mixed", second_id, &qualified)).await;
    let model = wait_for_model(&manager, "mixed-decisions", |model| {
        model.get_systemone_execution_plan().is_ok()
    })
    .await;
    assert!(Arc::ptr_eq(&initial_workers, &model.worker_sets()[0]));
    drop(events);
    task.await.unwrap();
    runtime.shutdown();
}
