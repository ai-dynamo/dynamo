// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::discovery::{DecisionEngine, DecisionRoute};
use dynamo_runtime::{Runtime, distributed::DistributedConfig};

#[test]
fn decision_whole_request_router_mode_preserves_supported_modes() {
    for mode in [
        RouterMode::RoundRobin,
        RouterMode::Random,
        RouterMode::PowerOfTwoChoices,
        RouterMode::LeastLoaded,
        RouterMode::DeviceAwareWeighted,
    ] {
        assert!(DecisionPlan::supports_native_routing(mode));
    }
}

#[test]
fn decision_whole_request_router_mode_rejects_missing_routing_inputs() {
    for mode in [RouterMode::KV, RouterMode::Direct] {
        assert!(!DecisionPlan::supports_native_routing(mode));
    }
}

#[tokio::test]
async fn decision_only_activation_builds_internal_generate_pipeline() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::decision_only_activation_builds_internal_generate_pipeline"
        ),
        &[
            ("DYN_ROUTER_MIN_INITIAL_WORKERS", "0"),
            ("RUST_MIN_STACK", "4194304"),
        ],
    ) {
        return;
    }
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let mut watcher = ModelWatcher::new(
        drt,
        Arc::new(ModelManager::new()),
        RouterConfig {
            router_mode: RouterMode::RoundRobin,
            ..Default::default()
        },
        0,
        None,
        None,
        None,
        Arc::new(Metrics::new()),
    );
    assert!(watcher.generate_engine_capabilities.is_empty());
    for (enabled, qualified) in [(false, true), (true, false), (true, true)] {
        watcher.set_systemone_enabled(enabled);
        let mut card = ModelDeploymentCard::load_from_disk(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
            None,
        )
        .unwrap();
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::Chat | ModelType::Completions;
        card.worker_type = Some(WorkerType::Aggregated);
        card.runtime_config.max_num_seqs = Some(1);
        for capability in [
            VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
            crate::local_model::runtime_config::VLLM_CAUSAL_DECISION_V1,
        ] {
            card.runtime_config
                .set_engine_specific(capability, qualified)
                .unwrap();
        }
        let spec = decision_preparation_spec(card);
        let (_admissions, admission_rx) = tokio::sync::watch::channel(vec![7]);
        let prepared = watcher
            .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
            .await
            .expect("chat remains available without public Generate activation");
        let workers = prepared.worker_set.as_ref().unwrap();
        assert!(workers.chat_engine.is_some());
        assert!(workers.completions_engine.is_some());
        assert!(workers.generate_engine.is_none());
        assert_eq!(
            matches!(
                workers.decision,
                Some(DecisionRoute::Available(DecisionEngine::Token(_)))
            ),
            enabled && qualified
        );
        assert!(watcher.generate_engine_capabilities.is_empty());
    }
    runtime.shutdown();
}

fn decision_preparation_spec(card: ModelDeploymentCard) -> GroupSpec {
    let mcid = ModelCardInstanceId {
        namespace: "decision-preparation-ns".into(),
        component: "workers".into(),
        endpoint: "generate".into(),
        instance_id: 7,
        model_suffix: None,
    };
    let endpoint_id = model_card_endpoint_id(&mcid);
    let key = GroupKey {
        model_name: card.name().to_string(),
        worker_set_key: worker_set_key(&endpoint_id, card.model_type, card.worker_type),
    };
    let desired = DesiredInstance {
        key: mcid.to_path(),
        mcid,
        endpoint_id,
        mdc_checksum: card.mdcsum().to_string(),
        projection_fingerprint: lora_projection_fingerprint(&card).unwrap(),
        video_contract: None,
        card,
        group_key: key.clone(),
    };
    GroupSpec {
        decision_kinds: crate::model_card::decision_profile::DecisionKinds::default(),
        decision_fleet: crate::discovery::DecisionFleet {
            state: tokio::sync::watch::channel(
                crate::discovery::DecisionAvailability::Unadvertised,
            )
            .1,
        },
        key,
        mdc_checksum: desired.mdc_checksum.clone(),
        fingerprint: desired.mdc_checksum.clone(),
        generation: 1,
        representative: desired,
        video_contract: None,
    }
}

#[tokio::test]
async fn text_routes_retain_graph_and_preserve_declared_surfaces() {
    use dynamo_runtime::{Runtime, distributed::DistributedConfig};

    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
        .await
        .unwrap();
    let manager = Arc::new(ModelManager::new());
    let router_config = RouterConfig {
        load_threshold_config: crate::discovery::LoadThresholdConfig {
            active_decode_blocks_threshold: Some(0.8),
            ..Default::default()
        },
        ..Default::default()
    };
    let watcher = Arc::new(ModelWatcher::new(
        drt,
        manager.clone(),
        router_config.clone(),
        0,
        None,
        None,
        None,
        Arc::new(Metrics::new_with_prefix(Some(
            "watcher_mixed_text_test".to_string(),
        ))),
    ));
    let mcid = ModelCardInstanceId {
        namespace: "mixed-text-ns".to_string(),
        component: "workers".to_string(),
        endpoint: "generate".to_string(),
        instance_id: 1,
        model_suffix: None,
    };
    let mut card = ModelDeploymentCard::with_name_only("mixed-text-model");
    card.model_input = ModelInput::Text;
    card.model_type = ModelType::Chat | ModelType::Classify | ModelType::Pooling;
    card.worker_type = Some(WorkerType::Aggregated);

    let endpoint_id = model_card_endpoint_id(&mcid);
    let key = GroupKey {
        model_name: card.name().to_string(),
        worker_set_key: worker_set_key(&endpoint_id, card.model_type, card.worker_type),
    };
    let desired = DesiredInstance {
        key: mcid.to_path(),
        mcid,
        endpoint_id,
        mdc_checksum: card.mdcsum().to_string(),
        projection_fingerprint: lora_projection_fingerprint(&card).unwrap(),
        video_contract: qwen_video_contract_digest(&card),
        card,
        group_key: key.clone(),
    };
    let spec = GroupSpec {
        decision_kinds: crate::model_card::decision_profile::DecisionKinds::default(),
        decision_fleet: crate::discovery::DecisionFleet {
            state: tokio::sync::watch::channel(
                crate::discovery::DecisionAvailability::Unadvertised,
            )
            .1,
        },
        key,
        mdc_checksum: desired.mdc_checksum.clone(),
        fingerprint: desired.mdc_checksum.clone(),
        generation: 1,
        representative: desired.clone(),
        video_contract: desired.video_contract.clone(),
    };
    let (_admission_tx, admission_rx) = tokio::sync::watch::channel(vec![1]);
    let prepared = watcher
        .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
        .await
        .unwrap();
    let worker_set = prepared.worker_set.as_ref().unwrap();
    let load_context = worker_set
        .load_context()
        .expect("text routing must retain its routing load context");
    assert_eq!(load_context.source(), RouterLoadSource::Aggregated);
    assert!(load_context.monitor().is_some());
    assert_eq!(load_context.client().endpoint.id(), desired.endpoint_id);
    watcher
        .commit_group(&spec, prepared, &[desired], &[])
        .unwrap();

    let model = manager.get_model("mixed-text-model").unwrap();
    assert!(model.has_chat_engine());
    assert!(model.has_classify_engine());
    assert!(model.has_pooling_engine());
    assert_eq!(
        model
            .load_threshold_config(None)
            .unwrap()
            .active_decode_blocks_threshold,
        Some(0.8)
    );
}

#[test]
fn native_profiles_do_not_build_routes_before_optional_activation() {
    use crate::local_model::runtime_config::{
        DecisionExecutionProfile as Profile, DecisionProfileKind, NativeExecutorKind,
        VLLM_NATIVE_DECISION_MAX_CANDIDATES,
    };
    for profile in [
        Profile::VllmD3V1,
        Profile::VllmClefFlashV1,
        Profile::VllmClefOmniV1,
        Profile::VllmLiquidV1,
        Profile::VllmDiffusionV1,
    ] {
        for input in [ModelInput::Text, ModelInput::Tokens] {
            let mut card = ModelDeploymentCard::with_name_only("inactive-native");
            card.model_input = input;
            card.worker_type = Some(WorkerType::Aggregated);
            card.model_type = match profile.kind() {
                DecisionProfileKind::Native(NativeExecutorKind::Head) => {
                    ModelType::Classify | ModelType::Pooling
                }
                DecisionProfileKind::Native(NativeExecutorKind::Read) => ModelType::Chat,
                DecisionProfileKind::Token(_) => unreachable!(),
            };
            card.runtime_config
                .set_engine_specific(profile.decision_capability(), true)
                .unwrap();
            card.runtime_config
                .set_engine_specific(VLLM_NATIVE_DECISION_MAX_CANDIDATES, 26)
                .unwrap();
            assert_eq!(card.decision_execution_profile(), None);
            assert!(matches!(
                DecisionPlan::new(&card, true, RouterMode::RoundRobin),
                DecisionPlan::Disabled
            ));
        }
    }
}
