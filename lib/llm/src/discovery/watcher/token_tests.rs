// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{common::checked_file::CheckedFile, model_card::TokenizerKind};
use dynamo_runtime::{
    Runtime,
    distributed::DistributedConfig,
    engine::AsyncEngineContextProvider,
    pipeline::{AsyncEngine, Error, ResponseStream},
};

#[tokio::test]
async fn token_factory_reuses_supplied_ordinary_routing_host() {
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let client = drt
        .namespace("shared-decision-host")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate")
        .client()
        .await
        .unwrap();
    let cancellation = runtime.child_token();
    let load = RoutingLoadContext::start(
        client.clone(),
        RouterLoadSource::Aggregated,
        LoadThresholdHandle::new(Default::default()),
        &cancellation,
        None,
    )
    .await
    .unwrap();
    let host = Arc::new(
        crate::kv_router::RoutingHost::builder(client, RouterMode::RoundRobin)
            .load_context(load.clone())
            .build()
            .await
            .unwrap(),
    );
    let ordinary: crate::types::openai::generate::GenerateStreamingEngine = host.clone();
    let routing = entrypoint::input::PreprocessedRouting::from_score_host_for_test(host);
    let mut card = ModelDeploymentCard::load_from_disk(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
        None,
    )
    .unwrap();
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(WorkerType::Aggregated);
    let mut workers = WorkerSet::new("shared-host".into(), card.mdcsum().into(), card.clone());
    let mut plan = DecisionPlan::new(
        &card,
        true,
        RouterMode::RoundRobin,
        workers.initialize_allocator_trim_on_teardown(),
    );
    let resources = plan
        .prepare_resources(None, Some(&routing), Some(load))
        .unwrap();
    assert!(Arc::ptr_eq(&ordinary, &routing.score_engine()));
    assert!(Arc::ptr_eq(
        &ordinary,
        resources.score_engine_for_test().unwrap()
    ));
    runtime.shutdown();
}

struct AvailableChat;

struct RenamedFixture {
    path: PathBuf,
    backup: PathBuf,
}

impl RenamedFixture {
    fn hide(path: PathBuf, model: &str) -> Self {
        assert!(path.components().any(|part| part.as_os_str() == model));
        let backup = path.with_extension("token-recovery-backup");
        assert!(!backup.exists());
        std::fs::rename(&path, &backup).unwrap();
        Self { path, backup }
    }

    fn restore(self) {
        std::fs::rename(&self.backup, &self.path).unwrap();
    }
}

impl Drop for RenamedFixture {
    fn drop(&mut self) {
        if self.backup.exists() {
            let _ = std::fs::rename(&self.backup, &self.path);
        }
    }
}

#[async_trait::async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for AvailableChat
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        Ok(ResponseStream::new(
            Box::pin(futures::stream::empty()),
            request.context(),
        ))
    }
}

#[tokio::test]
async fn failed_optional_token_preparation_preserves_ordinary_surfaces_without_retry() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::failed_optional_token_preparation_preserves_ordinary_surfaces_without_retry"
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
    let directory = tempfile::tempdir().unwrap();
    let invalid = directory.path().join("tokenizer.json");
    std::fs::write(&invalid, b"{}").unwrap();
    for bad_tokenizer in [true, false] {
        let factory: Option<ChatEngineFactoryCallback> = if bad_tokenizer {
            Some(Arc::new(|_, _, _| {
                Box::pin(async {
                    Ok(Arc::new(AvailableChat) as crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine)
                })
            }))
        } else {
            None
        };
        let mut watcher = ModelWatcher::new(
            drt.clone(),
            Arc::new(ModelManager::new()),
            RouterConfig {
                router_mode: RouterMode::RoundRobin,
                ..Default::default()
            },
            0,
            None,
            factory,
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
        card.model_input = ModelInput::Tokens;
        card.model_type = if bad_tokenizer {
            ModelType::Chat
        } else {
            ModelType::Completions
        };
        card.worker_type = Some(WorkerType::Aggregated);
        card.runtime_config.data_parallel_size = 1;
        card.runtime_config.max_num_seqs = Some(1);
        for capability in [
            crate::decision_profile::VLLM_CAUSAL_DECISION_V1,
            VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
        ] {
            card.runtime_config
                .set_engine_specific(capability, true)
                .unwrap();
        }
        if bad_tokenizer {
            card.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                CheckedFile::from_disk(&invalid).unwrap(),
            ));
        } else {
            card.prompt_formatter = None;
        }
        let (mut spec, _initial_publication) =
            super::decision_tests::decision_preparation_spec(card);
        let (_publication, fleet) = crate::discovery::test_support::fleet(
            spec.representative.card.decision_qualification(),
            vec![7],
        );
        spec.decision_fleet = fleet;
        let (_ordinary, admitted) = tokio::sync::watch::channel(vec![7]);
        let prepared = watcher
            .prepare_worker_set(&spec, admitted, CancellationToken::new())
            .await
            .unwrap();
        let workers = prepared.worker_set.unwrap();
        assert_eq!(workers.chat_engine.is_some(), bad_tokenizer);
        assert_eq!(workers.completions_engine.is_some(), !bad_tokenizer);
        assert!(
            !workers.has_decision_engine(),
            "known token construction failure cannot become lazy-ready"
        );
        assert!(
            matches!(
                workers.decision_state(),
                crate::discovery::DecisionState::Unqualified
            ),
            "eager and lazy permanent failures must retain the same structural decision state"
        );
        assert!(
            workers
                .decision
                .as_ref()
                .unwrap()
                .resources
                .token_failed_for_test(),
            "the canonical cell, not an absent factory, must own an eager permanent failure"
        );
        let model = crate::discovery::Model::new("token-failure".into());
        model.add_worker_set("token-failure".into(), Arc::new(workers));
        for _ in 0..3 {
            assert!(matches!(
                model.get_systemone_execution_plan(),
                Err(crate::discovery::SystemOneSelectionError::Unsupported)
            ));
        }
    }
    runtime.shutdown();
}

#[tokio::test]
async fn same_checksum_republication_recovers_transient_token_io_without_rebuilding() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::same_checksum_republication_recovers_transient_token_io_without_rebuilding"
        ),
        &[
            ("DYN_ROUTER_MIN_INITIAL_WORKERS", "0"),
            ("RUST_MIN_STACK", "4194304"),
        ],
    ) {
        return;
    }
    use super::test_support::{
        apply_discovery_event, discovered_card, wait_for_model, watch_test_cards_with_decisions,
    };
    use crate::decision_profile::VLLM_CAUSAL_DECISION_V1;
    use crate::discovery::SystemOneSelectionError;
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = drt
        .namespace("token-recovery")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    endpoint.register_endpoint_instance().await.unwrap();
    let worker = drt.discovery().instance_id();
    let name = format!("token-recovery-{worker}");
    let mut card = ModelDeploymentCard::load_from_disk(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
        None,
    )
    .unwrap();
    card.set_name(&name);
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config.max_num_seqs = Some(16);
    card.runtime_config
        .set_engine_specific(VLLM_INFERENCE_V1_GENERATE_CAPABILITY, true)
        .unwrap();
    let manager = Arc::new(ModelManager::new());
    let (events, watcher) = watch_test_cards_with_decisions(
        drt,
        manager.clone(),
        RouterConfig {
            router_mode: RouterMode::RoundRobin,
            ..Default::default()
        },
        true,
    );
    apply_discovery_event(&events, discovered_card("token-recovery", worker, &card)).await;
    let model = wait_for_model(&manager, &name, |model| {
        model.total_workers() == 1 && model.get_chat_engine().is_ok()
    })
    .await;
    let workers = model.worker_sets()[0].clone();
    let Some(crate::model_card::PromptFormatterArtifact::HfTokenizerConfigJson(formatter)) =
        &workers.card().prompt_formatter
    else {
        panic!("fixture formatter");
    };
    let hidden = RenamedFixture::hide(formatter.path().unwrap().to_owned(), &name);
    let mut qualified = card.clone();
    qualified.runtime_config.max_num_seqs = Some(1);
    qualified
        .runtime_config
        .set_engine_specific(VLLM_CAUSAL_DECISION_V1, true)
        .unwrap();
    assert_eq!(card.mdcsum(), qualified.mdcsum());
    apply_discovery_event(
        &events,
        discovered_card("token-recovery", worker, &qualified),
    )
    .await;
    let model = wait_for_model(&manager, &name, |model| {
        model.get_systemone_execution_plan().is_ok()
    })
    .await;
    let plan = model.get_systemone_execution_plan().unwrap();
    let context = plan.context.clone();
    let failed = tokio::task::spawn_blocking(move || plan.prepare())
        .await
        .unwrap();
    hidden.restore();
    assert!(matches!(
        failed
            .err()
            .unwrap()
            .downcast_ref::<SystemOneSelectionError>(),
        Some(SystemOneSelectionError::Unavailable)
    ));
    assert!(model.has_decision_engine());
    assert!(model.get_chat_engine().is_ok());
    apply_discovery_event(
        &events,
        discovered_card("token-recovery", worker, &qualified),
    )
    .await;
    assert!(Arc::ptr_eq(&workers, &model.worker_sets()[0]));
    context.validate_dispatch(Some(worker)).unwrap();
    tokio::time::sleep(Duration::from_millis(1050)).await;
    let plan = model.get_systemone_execution_plan().unwrap();
    tokio::task::spawn_blocking(move || plan.prepare())
        .await
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&workers, &model.worker_sets()[0]));
    assert!(
        workers
            .decision
            .as_ref()
            .unwrap()
            .resources
            .prepared_token()
            .is_some()
    );
    assert!(model.get_chat_engine().is_ok());
    context.validate_dispatch(Some(worker)).unwrap();
    drop(events);
    watcher.await.unwrap();
    runtime.shutdown();
}
