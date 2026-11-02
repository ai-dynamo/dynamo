// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::decision_profile::{
    DecisionProfileKind::Native,
    NativeExecutorKind::{Head, Read},
};
use dynamo_runtime::{Runtime, component::Client, distributed::DistributedConfig};

#[test]
fn decision_plan_separates_routing_from_internal_generation() {
    use crate::decision_profile::{
        VLLM_CAUSAL_DECISION_V1, VLLM_LIQUID_DECISION_V1, VLLM_NATIVE_DECISION_MAX_CANDIDATES,
    };

    let mut card = ModelDeploymentCard::with_name_only("planned-model");
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config
        .set_engine_specific(VLLM_LIQUID_DECISION_V1, true)
        .unwrap();
    card.runtime_config
        .set_engine_specific(VLLM_NATIVE_DECISION_MAX_CANDIDATES, 26)
        .unwrap();

    let teardown = Arc::new(super::super::allocator::AllocatorTrimOnDrop::new());
    let disabled = DecisionPlan::new(&card, false, RouterMode::RoundRobin, teardown.clone());
    assert!(!disabled.needs_load_context());
    let native = DecisionPlan::new(&card, true, RouterMode::RoundRobin, teardown.clone());
    assert!(native.needs_load_context());
    let unsupported = DecisionPlan::new(&card, true, RouterMode::Direct, teardown.clone());
    assert!(!unsupported.needs_load_context());

    card.runtime_config
        .runtime_data
        .remove(VLLM_LIQUID_DECISION_V1);
    card.runtime_config.max_num_seqs = Some(1);
    for capability in [
        VLLM_CAUSAL_DECISION_V1,
        VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    ] {
        card.runtime_config
            .set_engine_specific(capability, true)
            .unwrap();
    }
    let token = DecisionPlan::new(&card, true, RouterMode::KV, teardown);
    assert!(token.needs_load_context());
}

#[tokio::test]
async fn native_load_context_excludes_discovered_but_unadmitted_workers() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::native_load_context_excludes_discovered_but_unadmitted_workers"
        ),
        &[
            ("DYN_ROUTER_MIN_INITIAL_WORKERS", "0"),
            ("RUST_MIN_STACK", "4194304"),
        ],
    ) {
        return;
    }
    use crate::decision_profile::{
        VLLM_CLEF_FLASH_DECISION_V1, VLLM_LIQUID_DECISION_V1, VLLM_NATIVE_DECISION_MAX_CANDIDATES,
    };

    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = drt
        .namespace("decision-preparation-ns")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let discovered = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker_id = discovered.wait_for_instances().await.unwrap()[0].id();
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
    watcher.set_systemone_enabled(true);

    for (input, capability) in [
        (ModelInput::Tokens, VLLM_LIQUID_DECISION_V1),
        (ModelInput::Text, VLLM_LIQUID_DECISION_V1),
        (ModelInput::Text, VLLM_CLEF_FLASH_DECISION_V1),
    ] {
        let mut card = ModelDeploymentCard::with_name_only("native-admission-model");
        card.model_input = input;
        card.model_type = if capability == VLLM_CLEF_FLASH_DECISION_V1 {
            ModelType::Classify | ModelType::Pooling
        } else {
            ModelType::Chat
        };
        card.worker_type = Some(WorkerType::Aggregated);
        card.runtime_config
            .set_engine_specific(capability, true)
            .unwrap();
        card.runtime_config
            .set_engine_specific(VLLM_NATIVE_DECISION_MAX_CANDIDATES, 26)
            .unwrap();
        let (spec, _qualification) = decision_preparation_spec(card);
        let (admissions, admission_rx) = tokio::sync::watch::channel(Vec::new());
        let prepared = watcher
            .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
            .await
            .unwrap();
        let worker_set = prepared.worker_set.as_ref().unwrap();
        assert!(matches!(
            &worker_set.decision,
            Some(surface) if !surface.resources.is_prepared(Native(Head)) && !surface.resources.is_prepared(Native(Read))
        ));
        let client = worker_set.load_context().unwrap().client();
        assert_eq!(discovered.instances().len(), 1);
        assert!(client.instances().is_empty());
        assert!(client.instance_ids_avail().is_empty());

        admissions.send_replace(vec![worker_id]);
        wait_for_native_route_workers(client, &[worker_id]).await;
        admissions.send_replace(Vec::new());
        wait_for_native_route_workers(client, &[]).await;
        assert_eq!(discovered.instances().len(), 1);
    }
    runtime.shutdown();
}

async fn wait_for_native_route_workers(client: &Client, expected: &[u64]) {
    let mut workers = client.instance_avail_watcher();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if workers.borrow_and_update().as_slice() == expected {
                break;
            }
            workers.changed().await.unwrap();
        }
    })
    .await
    .expect("native routing must track controller admission");
}

#[test]
fn decision_whole_request_router_mode_preserves_supported_modes() {
    let mut card = ModelDeploymentCard::with_name_only("native-routing");
    card.worker_type = Some(WorkerType::Aggregated);
    card.model_input = ModelInput::Text;
    card.model_type = ModelType::Chat;
    for mode in [
        RouterMode::RoundRobin,
        RouterMode::Random,
        RouterMode::PowerOfTwoChoices,
        RouterMode::LeastLoaded,
        RouterMode::DeviceAwareWeighted,
    ] {
        assert!(
            crate::discovery::decision_policy::DecisionCapabilities::for_card(&card, mode, true)
                .kinds
                .read
        );
    }
}

#[test]
fn decision_whole_request_router_mode_rejects_missing_routing_inputs() {
    let mut card = ModelDeploymentCard::with_name_only("native-routing");
    card.worker_type = Some(WorkerType::Aggregated);
    card.model_input = ModelInput::Text;
    card.model_type = ModelType::Chat;
    for mode in [RouterMode::KV, RouterMode::Direct] {
        assert!(
            !crate::discovery::decision_policy::DecisionCapabilities::for_card(&card, mode, true)
                .kinds
                .read
        );
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
    for (enabled, qualified, serial, aggregated) in [
        (false, true, true, true),
        (true, false, true, true),
        (true, true, true, true),
        (true, false, false, true),
        (true, false, true, false),
    ] {
        watcher.set_systemone_enabled(enabled);
        let mut card = ModelDeploymentCard::load_from_disk(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
            None,
        )
        .unwrap();
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::Chat | ModelType::Completions;
        card.worker_type = aggregated.then_some(WorkerType::Aggregated);
        card.runtime_config.max_num_seqs = Some(if serial { 1 } else { 16 });
        for capability in [
            VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
            crate::decision_profile::VLLM_CAUSAL_DECISION_V1,
        ] {
            card.runtime_config
                .set_engine_specific(capability, qualified)
                .unwrap();
        }
        let (mut spec, _initial_qualification) = decision_preparation_spec(card);
        let (qualifications, fleet) = crate::discovery::test_support::fleet(
            spec.representative.card.decision_qualification(),
            vec![7],
        );
        spec.decision_fleet = fleet;
        let (_admissions, admission_rx) = tokio::sync::watch::channel(vec![7]);
        let prepared = watcher
            .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
            .await
            .expect("chat remains available without public Generate activation");
        let workers = prepared.worker_set.as_ref().unwrap();
        assert!(workers.chat_engine.is_some());
        assert!(workers.completions_engine.is_some());
        assert!(workers.generate_engine.is_none());
        let ordinary_chat = workers.chat_engine.as_ref().unwrap().clone();
        let ordinary_completions = workers.completions_engine.as_ref().unwrap().clone();
        let ordinary_routing = workers
            .decision
            .as_ref()
            .and_then(|surface| surface.resources.score_engine_for_test().cloned());
        assert_eq!(ordinary_routing.is_some(), enabled && aggregated);
        assert_eq!(
            matches!(
                &workers.decision,
                Some(surface) if surface.resources.prepared_token().is_some()
            ),
            enabled && qualified
        );
        assert_eq!(workers.has_decision_engine(), enabled && qualified);
        if let Some(resources) = &workers.decision {
            let resources = &resources.resources;
            assert!(
                !resources.is_prepared(Native(Read)),
                "causal-only cards must not build native read routers"
            );
            assert!(!resources.is_prepared(Native(Head)));
        }
        if enabled && !qualified && aggregated {
            use crate::decision_profile::DecisionExecutionProfile as Profile;
            use crate::discovery::SystemOneExecutionPlan;
            let Some(resources) = &workers.decision else {
                panic!("ordinary route retains a late-kind factory");
            };
            let resources = &resources.resources;
            assert!(
                resources.is_empty(),
                "no advertised decision kind must materialize eagerly"
            );
            crate::discovery::test_support::publish(
                &qualifications,
                Some((Profile::VllmCausalV1, 26)),
                vec![7],
            );
            assert!(
                workers.has_decision_engine(),
                "a qualified route exists before token resource initialization"
            );
            let plan =
                SystemOneExecutionPlan::from_worker_set("planned-model".into(), workers).unwrap();
            tokio::task::spawn_blocking(move || plan.prepare())
                .await
                .unwrap()
                .unwrap();
            assert!(workers.has_decision_engine());
            let token = resources.prepared_token().unwrap().engine.clone();
            qualifications.send_replace(crate::discovery::DecisionAvailability::Unadvertised);
            assert!(!workers.has_decision_engine());
            crate::discovery::test_support::publish(
                &qualifications,
                Some((Profile::SglangSerialV1, 255)),
                vec![7],
            );
            assert!(
                workers.has_decision_engine(),
                "same token kind reuses its prepared transport across backend/version gates"
            );
            assert!(Arc::ptr_eq(
                &token,
                &resources.prepared_token().unwrap().engine
            ));
            crate::discovery::test_support::publish(
                &qualifications,
                Some((Profile::VllmLiquidV1, 26)),
                vec![7],
            );
            assert!(
                workers.has_decision_engine(),
                "a late native kind is supported before initialization"
            );
            let plan =
                SystemOneExecutionPlan::from_worker_set("planned-model".into(), workers).unwrap();
            plan.prepare_native().await.unwrap();
            assert!(matches!(
                &plan.prepare().unwrap().engine,
                crate::discovery::DecisionEngine::Read(_)
            ));
            assert!(workers.has_decision_engine());
            assert!(Arc::ptr_eq(
                &token,
                &resources.prepared_token().unwrap().engine
            ));
            assert!(Arc::ptr_eq(
                ordinary_routing.as_ref().unwrap(),
                resources.score_engine_for_test().unwrap(),
            ));
            assert!(Arc::ptr_eq(
                &ordinary_chat,
                workers.chat_engine.as_ref().unwrap()
            ));
            assert!(Arc::ptr_eq(
                &ordinary_completions,
                workers.completions_engine.as_ref().unwrap()
            ));
        }
        assert!(watcher.generate_engine_capabilities.is_empty());
    }
    runtime.shutdown();
}

pub(super) fn decision_preparation_spec(
    card: ModelDeploymentCard,
) -> (
    GroupSpec,
    tokio::sync::watch::Sender<crate::discovery::DecisionAvailability>,
) {
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
    let (qualification, fleet) = crate::discovery::test_support::fleet(None, vec![]);
    (
        GroupSpec {
            key,
            mdc_checksum: desired.mdc_checksum.clone(),
            fingerprint: desired.mdc_checksum.clone(),
            generation: 1,
            decision_kinds: crate::model_card::decision_profile::DecisionKinds::advertised(
                &desired.card,
            ),
            decision_fleet: fleet,
            representative: desired,
            video_contract: None,
        },
        qualification,
    )
}

#[tokio::test]
async fn unsupported_optional_decision_routing_preserves_declared_surfaces() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::unsupported_optional_decision_routing_preserves_declared_surfaces"
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
    use crate::decision_profile::{VLLM_CLEF_FLASH_DECISION_V1, VLLM_LIQUID_DECISION_V1};
    for (input, capability, mode) in [
        (ModelInput::Tokens, VLLM_LIQUID_DECISION_V1, RouterMode::KV),
        (
            ModelInput::Tokens,
            VLLM_LIQUID_DECISION_V1,
            RouterMode::Direct,
        ),
        (
            ModelInput::Text,
            VLLM_LIQUID_DECISION_V1,
            RouterMode::Direct,
        ),
        (
            ModelInput::Text,
            VLLM_CLEF_FLASH_DECISION_V1,
            RouterMode::Direct,
        ),
    ] {
        let mut watcher = ModelWatcher::new(
            drt.clone(),
            Arc::new(ModelManager::new()),
            RouterConfig {
                router_mode: mode,
                ..Default::default()
            },
            0,
            None,
            None,
            None,
            Arc::new(Metrics::new()),
        );
        watcher.set_systemone_enabled(true);
        let mut card = ModelDeploymentCard::load_from_disk(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
            None,
        )
        .unwrap();
        card.model_input = input;
        card.kv_cache_block_size = 16;
        card.model_type = ModelType::Chat | ModelType::Completions;
        if capability == VLLM_CLEF_FLASH_DECISION_V1 {
            card.model_type |= ModelType::Classify | ModelType::Pooling;
        }
        card.worker_type = Some(WorkerType::Aggregated);
        card.runtime_config
            .set_engine_specific(capability, true)
            .unwrap();
        card.runtime_config
            .set_engine_specific(
                crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES,
                26,
            )
            .unwrap();
        assert!(card.decision_execution_profile().is_some());
        let (spec, _qualification) = decision_preparation_spec(card);
        let (_admissions, admission_rx) = tokio::sync::watch::channel(vec![7]);
        let prepared = watcher
            .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
            .await
            .expect("unsupported decision routing must preserve other surfaces");
        let workers = prepared.worker_set.as_ref().unwrap();
        assert!(workers.chat_engine.is_some());
        assert!(workers.completions_engine.is_some());
        assert!(!workers.decision_state().is_ready());
        if capability == VLLM_CLEF_FLASH_DECISION_V1 {
            assert!(workers.classify_engine.is_some());
            assert!(workers.pooling_engine.is_some());
        }
        let model = crate::discovery::Model::new("unsupported-decision-route".into());
        model.add_worker_set(
            "unsupported-route".into(),
            Arc::new(prepared.worker_set.unwrap()),
        );
        assert!(matches!(
            model.prepared_selection_for_test().await,
            Err(crate::discovery::SystemOneSelectionError::Unsupported)
        ));
    }
    runtime.shutdown();
}

#[tokio::test]
async fn token_native_read_without_frontend_tokenizer_retains_load_context() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::token_native_read_without_frontend_tokenizer_retains_load_context"
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
    watcher.set_systemone_enabled(true);
    assert!(watcher.generate_engine_capabilities.is_empty());

    for (model_type, native_enabled, failure_marker, enabled) in [
        (ModelType::Chat, false, None, false),
        (ModelType::Chat, false, None, true),
        (ModelType::Chat, true, None, true),
        (ModelType::Chat | ModelType::Completions, true, None, true),
        (
            ModelType::Chat,
            false,
            Some(serde_json::json!({"profile":"vllm_liquid_decision_v1"})),
            true,
        ),
        (
            ModelType::Chat,
            false,
            Some(serde_json::json!({"profile":"unknown"})),
            true,
        ),
        (ModelType::Chat, false, Some(serde_json::json!(true)), true),
    ] {
        watcher.set_systemone_enabled(enabled);
        let mut card = ModelDeploymentCard::with_name_only("native-token-model");
        card.model_input = ModelInput::Tokens;
        card.model_type = model_type;
        card.worker_type = Some(WorkerType::Aggregated);
        if let Some(marker) = failure_marker {
            card.runtime_config.runtime_data.insert(
                crate::decision_profile::VLLM_NATIVE_DECISION_QUALIFICATION_FAILURE_V1.into(),
                marker,
            );
        }
        let failed = card.decision_qualification_failure().is_some();
        if native_enabled {
            card.runtime_config
                .set_engine_specific(crate::decision_profile::VLLM_LIQUID_DECISION_V1, true)
                .unwrap();
            card.runtime_config
                .set_engine_specific(
                    crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES,
                    26,
                )
                .unwrap();
        }
        assert!(!card.has_tokenizer());
        let (mut spec, _initial_qualification) = decision_preparation_spec(card);
        let expected_qualification = spec.representative.card.decision_qualification();
        let (qualifications, fleet) = crate::discovery::test_support::fleet(None, vec![7]);
        spec.decision_fleet = fleet;
        let (admission_tx, admission_rx) = tokio::sync::watch::channel(vec![7]);
        let prepared = watcher
            .prepare_worker_set(&spec, admission_rx, CancellationToken::new())
            .await;
        if !native_enabled && !failed {
            let error = prepared.err().expect(
                "no tokenizer, no factory and no decision advertisement must fail identically with System One enabled",
            );
            assert_eq!(
                error.to_string(),
                "Model has no supported Rust tokenizer and no chat_engine_factory. \
                 Use --dyn-chat-processor vllm/sglang or provide a supported \
                 tokenizer file (tokenizer.json, tiktoken.model, or *.tiktoken)."
            );
            continue;
        }
        let mut prepared = prepared.expect("native reads need no frontend tokenizer or Generate");
        let worker_set = prepared.worker_set.as_ref().unwrap();
        assert!(matches!(
            &worker_set.decision,
            Some(surface) if !surface.resources.is_prepared(Native(Read))
        ));
        assert!(
            !worker_set.has_decision_engine(),
            "prepared native resources are not live before cohort qualification"
        );
        if failed {
            qualifications.send_replace(crate::discovery::DecisionAvailability::Failed);
        } else {
            crate::discovery::test_support::publish(
                &qualifications,
                expected_qualification,
                vec![7],
            );
        }
        assert_eq!(worker_set.has_decision_engine(), native_enabled);
        assert!(!worker_set.is_prefill_set());
        assert!(worker_set.chat_engine.is_none());
        assert!(worker_set.completions_engine.is_none());
        assert!(worker_set.generate_engine.is_none());
        assert!(worker_set.encoder_router.is_none());
        assert!(worker_set.prefill_router.is_none());
        assert!(worker_set.load_thresholds.is_some());
        let load_context = worker_set.load_context().expect("native route load owner");
        let resources = &worker_set.decision.as_ref().unwrap().resources;
        assert!(
            resources.score_engine_for_test().is_none(),
            "native-only resources have no ordinary score host"
        );
        assert!(Arc::ptr_eq(
            load_context,
            resources.load_context_for_test().unwrap()
        ));
        if native_enabled {
            let plan = crate::discovery::SystemOneExecutionPlan::from_worker_set(
                "native-read".into(),
                worker_set,
            )
            .unwrap();
            plan.prepare_native().await.unwrap();
            assert!(resources.is_prepared(Native(Read)));
            assert!(resources.score_engine_for_test().is_none());
            assert!(Arc::ptr_eq(
                load_context,
                resources.load_context_for_test().unwrap()
            ));
        }
        assert_eq!(load_context.source(), RouterLoadSource::Aggregated);
        assert!(load_context.monitor().is_some());
        if failed {
            assert!(matches!(
                worker_set.decision_state(),
                crate::discovery::DecisionState::Failed
            ));
            assert_eq!(worker_set.decision_worker_count(), 0);
        }
        let cancellation = load_context.cancellation_token();
        assert!(!cancellation.is_cancelled());
        let admissions = worker_set.decision_admissions().unwrap();
        assert_eq!(*admissions.borrow(), vec![7]);
        admission_tx.send(vec![]).unwrap();
        assert!(admissions.borrow().is_empty());
        if failed {
            let model = crate::discovery::Model::new("failed-read".into());
            model.add_worker_set(
                "failed-read".into(),
                Arc::new(prepared.worker_set.take().unwrap()),
            );
            assert!(matches!(
                model.get_systemone_execution_plan(),
                Err(crate::discovery::SystemOneSelectionError::QualificationFailed)
            ));
        }
        drop(prepared);
        assert!(cancellation.is_cancelled());
    }
    runtime.shutdown();
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
        key,
        mdc_checksum: desired.mdc_checksum.clone(),
        fingerprint: desired.mdc_checksum.clone(),
        generation: 1,
        decision_kinds: crate::model_card::decision_profile::DecisionKinds::advertised(
            &desired.card,
        ),
        decision_fleet: crate::discovery::test_support::fleet(None, vec![]).1,
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
