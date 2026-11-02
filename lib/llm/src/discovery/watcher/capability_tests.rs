// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    decision_profile::{
        DecisionExecutionProfile as Profile, DecisionProfileKind, NativeExecutorKind,
    },
    discovery::{CommittedWorkerSetTarget, test_support},
};

#[derive(Debug, Clone, Copy)]
struct Case {
    profile: Profile,
    input: ModelInput,
    model_type: ModelType,
    worker_type: Option<WorkerType>,
    tokenizer: bool,
    mode: RouterMode,
    score: bool,
    load: bool,
    enabled: bool,
}

struct Fixture {
    card: ModelDeploymentCard,
    tokenizer: crate::tokenizers::Tokenizer,
    routing: entrypoint::input::PreprocessedRouting,
    load: Arc<RoutingLoadContext>,
}

impl Fixture {
    async fn check(&self, case: Case) {
        let mut card = self.card.clone();
        card.model_input = case.input;
        card.model_type = case.model_type;
        card.worker_type = case.worker_type;
        card.runtime_config.data_parallel_size = 1;
        card.runtime_config.max_num_seqs = Some(1);
        card.runtime_config.runtime_data.clear();
        if !case.tokenizer {
            card.tokenizer = None;
        }
        card.runtime_config
            .set_engine_specific(case.profile.decision_capability(), true)
            .unwrap();
        card.runtime_config
            .set_engine_specific(
                crate::decision_profile::VLLM_NATIVE_DECISION_MAX_CANDIDATES,
                26,
            )
            .unwrap();
        if let DecisionProfileKind::Token(backend) = case.profile.kind() {
            card.runtime_config
                .set_engine_specific(
                    crate::model_card::decision_profile::transport_capability(backend),
                    true,
                )
                .unwrap();
        }
        let token_model = case.input == ModelInput::Tokens
            && (case.model_type.supports_chat() || case.model_type.supports_completions());
        let native_mode = case.mode == RouterMode::RoundRobin;
        let shape = match case.profile.kind() {
            DecisionProfileKind::Token(_) => token_model,
            DecisionProfileKind::Native(NativeExecutorKind::Head) => {
                case.input == ModelInput::Text
                    && case.model_type.supports_classify()
                    && case.model_type.supports_pooling()
                    && native_mode
            }
            DecisionProfileKind::Native(NativeExecutorKind::Read) => {
                matches!(case.input, ModelInput::Tokens | ModelInput::Text)
                    && (case.model_type.supports_chat() || case.model_type.supports_completions())
                    && native_mode
            }
        } && case.enabled
            && case.worker_type == Some(WorkerType::Aggregated);
        let expected = shape
            && case.load
            && (!matches!(case.profile.kind(), DecisionProfileKind::Token(_))
                || (case.score && case.tokenizer));
        let mut workers = WorkerSet::new(
            "capability-table".into(),
            card.mdcsum().into(),
            card.clone(),
        );
        let (_membership, ids) = tokio::sync::watch::channel(vec![1]);
        workers.set_instance_watcher(ids.clone());
        workers.set_lifecycle_cancellation(self.load.cancellation_token());
        workers.set_topology_target(CommittedWorkerSetTarget {
            endpoint: self.load.client().endpoint.clone(),
            group: "capability-table".into(),
            generation: 1,
            card: Arc::new(card.clone()),
            admitted_ids: ids,
        });
        let mut plan = DecisionPlan::new(
            &card,
            case.enabled,
            case.mode,
            workers.initialize_allocator_trim_on_teardown(),
        );
        assert_eq!(plan.needs_load_context(), shape, "planning: {case:?}");
        plan.prepare_resources(
            Some(&self.tokenizer),
            case.score.then_some(&self.routing),
            case.load.then(|| self.load.clone()),
        );
        let (_publication, fleet) = test_support::fleet(card.decision_qualification(), vec![1]);
        plan.attach(&mut workers, fleet);
        assert_eq!(
            workers.decision_state().is_ready(),
            expected,
            "readiness: {case:?}"
        );
        if let Some(surface) = &workers.decision {
            let resources = &surface.resources;
            assert_eq!(
                resources.supports(case.profile.kind()),
                expected,
                "support: {case:?}"
            );
            assert_eq!(
                resources.prepare_native(case.profile.kind()).await.is_ok(),
                expected,
                "native prep: {case:?}"
            );
            assert_eq!(
                resources.prepare(case.profile.kind()).is_ok(),
                expected,
                "prepare: {case:?}"
            );
        } else {
            assert!(!expected, "missing resources: {case:?}");
        }
    }
}

#[tokio::test]
async fn decision_capabilities_align_planning_readiness_and_preparation() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::decision_capabilities_align_planning_readiness_and_preparation"
        ),
        &[
            ("DYN_ROUTER_MIN_INITIAL_WORKERS", "0"),
            ("RUST_MIN_STACK", "4194304"),
        ],
    ) {
        return;
    }
    let (_runtime, load) = test_support::routing_load_for_test().await;
    let card = ModelDeploymentCard::load_from_disk(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
        None,
    )
    .unwrap();
    let tokenizer = card.tokenizer().unwrap();
    let host = Arc::new(
        crate::kv_router::RoutingHost::builder(load.client().clone(), RouterMode::RoundRobin)
            .load_context(load.clone())
            .build()
            .await
            .unwrap(),
    );
    let fixture = Fixture {
        card,
        tokenizer,
        routing: entrypoint::input::PreprocessedRouting::from_score_host_for_test(host),
        load,
    };
    for profile in [
        Profile::SglangSerialV1,
        Profile::VllmCausalV1,
        Profile::VllmD3V1,
        Profile::VllmClefFlashV1,
        Profile::VllmClefOmniV1,
        Profile::VllmLiquidV1,
        Profile::VllmDiffusionV1,
    ] {
        for (input, model_type, worker_type, tokenizer) in [
            (
                ModelInput::Tokens,
                ModelType::Chat,
                Some(WorkerType::Aggregated),
                true,
            ),
            (
                ModelInput::Tokens,
                ModelType::Completions,
                Some(WorkerType::Aggregated),
                true,
            ),
            (
                ModelInput::Tokens,
                ModelType::Chat,
                Some(WorkerType::Aggregated),
                false,
            ),
            (
                ModelInput::Text,
                ModelType::Chat,
                Some(WorkerType::Aggregated),
                true,
            ),
            (
                ModelInput::Text,
                ModelType::Classify | ModelType::Pooling,
                Some(WorkerType::Aggregated),
                true,
            ),
            (
                ModelInput::Tokens,
                ModelType::Classify | ModelType::Pooling,
                Some(WorkerType::Aggregated),
                true,
            ),
            (
                ModelInput::Tokens,
                ModelType::Chat,
                Some(WorkerType::Decode),
                true,
            ),
            (ModelInput::Tensor, ModelType::TensorBased, None, true),
        ] {
            for mode in [RouterMode::RoundRobin, RouterMode::KV, RouterMode::Direct] {
                for (score, load) in [(false, false), (false, true), (true, false), (true, true)] {
                    for enabled in [false, true] {
                        fixture
                            .check(Case {
                                profile,
                                input,
                                model_type,
                                worker_type,
                                tokenizer,
                                mode,
                                score,
                                load,
                                enabled,
                            })
                            .await;
                    }
                }
            }
        }
    }
}
