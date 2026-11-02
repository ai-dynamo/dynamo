// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::protocols::Annotated;
use crate::protocols::common::{llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest};
use dynamo_runtime::pipeline::{AsyncEngine, Error, ManyOut, SingleIn};
use std::sync::atomic::{AtomicUsize, Ordering};

struct UnusedScoreEngine;
#[async_trait::async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for UnusedScoreEngine
{
    async fn generate(
        &self,
        _: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        anyhow::bail!("resource lifecycle fixture must not dispatch")
    }
}

fn token_resources() -> TokenResources {
    let card = ModelDeploymentCard::load_from_disk(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
        None,
    )
    .unwrap();
    TokenResources {
        engine: Arc::new(UnusedScoreEngine),
        preprocessor: OpenAIPreprocessor::new(card).unwrap(),
    }
}

#[tokio::test]
async fn eager_lazy_and_retry_preparation_share_builder_and_load_cancellation() {
    use crate::discovery::LoadThresholdHandle;
    use crate::kv_router::RouterLoadSource;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use std::cell::{Cell, RefCell};

    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let client = drt
        .namespace("canonical-token-builder")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate")
        .client()
        .await
        .unwrap();
    let parent = CancellationToken::new();
    let load = RoutingLoadContext::start(
        client,
        RouterLoadSource::Aggregated,
        LoadThresholdHandle::new(Default::default()),
        &parent,
        None,
    )
    .await
    .unwrap();
    let mut card = ModelDeploymentCard::load_from_disk(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct"),
        None,
    )
    .unwrap();
    card.model_input = crate::model_type::ModelInput::Tokens;
    card.model_type = crate::model_type::ModelType::Chat;
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    let tokenizer = card.tokenizer().unwrap();
    let ready = token_resources();
    let seen = RefCell::new(Vec::new());
    let factory = || {
        DecisionResources::for_worker(
            card.clone(),
            Some(ScoreFactory::new(
                ready.engine.clone(),
                TokenizerLoad::Loaded(tokenizer.clone()),
            )),
            Some(load.clone()),
            RouterMode::RoundRobin,
            Arc::new(AllocatorTrimOnDrop::new()),
        )
    };
    for eager in [true, false] {
        let resources = factory();
        let calls = Cell::new(0);
        let build = |_: &ModelDeploymentCard, _: Tokenizer, cancellation: &CancellationToken| {
            calls.set(calls.get() + 1);
            seen.borrow_mut().push(cancellation.clone());
            Ok(ready.preprocessor.clone())
        };
        if eager {
            resources.prepare_token_with(build).unwrap();
        }
        let resources = resources.into_shared().unwrap();
        let first = resources.prepare_token_with(build).unwrap();
        let cached = resources.prepare_token_with(build).unwrap();
        assert!(Arc::ptr_eq(&first, &cached));
        assert_eq!(calls.get(), 1, "eager={eager}");
    }
    let resources = factory();
    let calls = Cell::new(0);
    let build = |_: &ModelDeploymentCard, _: Tokenizer, cancellation: &CancellationToken| {
        calls.set(calls.get() + 1);
        seen.borrow_mut().push(cancellation.clone());
        if calls.get() == 1 {
            return Err(std::io::Error::other("temporary builder I/O").into());
        }
        Ok(ready.preprocessor.clone())
    };
    for _ in 0..2 {
        let error = resources.prepare_token_with(build).err().unwrap();
        assert!(matches!(
            error.downcast_ref::<SystemOneSelectionError>(),
            Some(SystemOneSelectionError::Unavailable)
        ));
    }
    assert_eq!(calls.get(), 1, "backoff must not invoke the builder");
    {
        let mut state = resources.token.state.lock();
        let TokenInitState::Backoff { until, .. } = &mut *state else {
            panic!("retry state")
        };
        *until = Instant::now();
    }
    let recovered = resources.prepare_token_with(build).unwrap();
    assert!(Arc::ptr_eq(
        &recovered,
        &resources.prepare_token_with(build).unwrap()
    ));
    assert_eq!(calls.get(), 2);
    let seen = seen.borrow();
    assert_eq!(seen.len(), 4);
    assert!(seen.iter().all(|token| !token.is_cancelled()));
    parent.cancel();
    assert!(seen.iter().all(CancellationToken::is_cancelled));
    runtime.shutdown();
}

#[tokio::test]
async fn decision_resources_retain_teardown_once_for_prebuilt_and_lazy_commit() {
    let (_runtime, load) = super::super::test_support::routing_load_for_test().await;
    for prebuilt in [false, true] {
        let mut token = Some(token_resources());
        let mut card = ModelDeploymentCard::with_name_only("teardown-fixture");
        card.model_input = crate::model_type::ModelInput::Tokens;
        card.model_type = crate::model_type::ModelType::Chat;
        card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
        let teardown = Arc::new(AllocatorTrimOnDrop::new());
        let mut resources = DecisionResources::for_worker(
            card.clone(),
            Some(ScoreFactory::new(
                token.as_ref().unwrap().engine.clone(),
                TokenizerLoad::Card,
            )),
            Some(load.clone()),
            RouterMode::RoundRobin,
            teardown.clone(),
        );
        if prebuilt {
            resources.insert_token(token.take().unwrap());
        }
        let mut workers = super::super::WorkerSet::new("fixture".into(), "revision".into(), card);
        let (_admissions, admitted_ids) = tokio::sync::watch::channel(vec![1]);
        workers.set_instance_watcher(admitted_ids);
        let _qualification = super::super::test_support::attach_card_surface_for_test(
            &mut workers,
            resources.into_shared(),
        )
        .await;
        let adopted = workers.initialize_allocator_trim_on_teardown();
        assert!(Arc::ptr_eq(&adopted, &teardown));
        drop(adopted);
        let before_commit = Arc::strong_count(&teardown);
        workers.enable_allocator_trim_on_teardown();
        assert_eq!(
            Arc::strong_count(&teardown),
            before_commit,
            "commit must not wrap decision engines a second time"
        );
        let Some(resources) = &workers.decision else {
            unreachable!()
        };
        if let Some(token) = token.take() {
            resources
                .resources
                .token
                .get_or_init(|| Ok(Arc::new(retain_token(token, &resources.resources.teardown))))
                .unwrap();
        }
        resources.resources.prepare_token().unwrap();
        let after_prepare = Arc::strong_count(&teardown);
        assert_eq!(after_prepare, before_commit + usize::from(!prebuilt));
        workers.enable_allocator_trim_on_teardown();
        assert_eq!(Arc::strong_count(&teardown), after_prepare);
    }
}

#[test]
fn token_selections_share_one_owned_resource_pair() {
    let mut resources = DecisionResources::default();
    resources.insert_token(token_resources());
    let token = resources.token.get().unwrap();
    let owners = (
        Arc::strong_count(&token.engine),
        Arc::strong_count(&token.preprocessor),
    );
    let selections: Vec<_> = (0..3)
        .map(|_| {
            resources
                .prepare(DecisionProfileKind::Token(
                    crate::decision_profile::TokenBackend::VllmCausal,
                ))
                .unwrap()
        })
        .collect();
    assert_eq!(
        (
            Arc::strong_count(&token.engine),
            Arc::strong_count(&token.preprocessor)
        ),
        owners,
        "request executors must share the resource pair rather than clone both members"
    );
    assert_eq!(selections.len(), 3);
}

#[tokio::test]
async fn late_token_initialization_drains_cancelled_admission_and_caches_outcome() {
    let cell = Arc::new(TokenResourceCell::default());
    let attempts = Arc::new(AtomicUsize::new(0));
    let admission = Arc::new(tokio::sync::Semaphore::new(1));
    let permit = admission.clone().acquire_owned().await.unwrap();
    let (entered, entered_rx) = tokio::sync::oneshot::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    let resources = Arc::new(token_resources());
    let observed = cell.clone();
    let builds = attempts.clone();
    let waiter = tokio::spawn(async move {
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            observed
                .get_or_init(|| {
                    builds.fetch_add(1, Ordering::SeqCst);
                    entered.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(resources)
                })
                .map(|_| ())
        })
        .await
        .unwrap()
    });
    entered_rx.await.unwrap();
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    assert!(
        admission.clone().try_acquire_owned().is_err(),
        "cancelled observer cannot release blocking initialization admission"
    );
    assert!(!cell.is_ready());
    release.send(()).unwrap();
    let _drained = tokio::time::timeout(std::time::Duration::from_secs(5), admission.acquire())
        .await
        .unwrap()
        .unwrap();
    assert!(cell.get().is_some());
    cell.get_or_init(|| {
        attempts.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("cached initializer must not run")
    })
    .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

#[test]
fn late_token_failure_is_terminal_for_its_worker_set() {
    for kind in [
        None,
        Some(std::io::ErrorKind::InvalidData),
        Some(std::io::ErrorKind::InvalidInput),
    ] {
        let cell = TokenResourceCell::default();
        let attempts = AtomicUsize::new(0);
        for _ in 0..3 {
            assert!(
                cell.get_or_init(|| {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    if let Some(kind) = kind {
                        Err(
                            std::io::Error::new(kind, "deterministic token configuration failure")
                                .into(),
                        )
                    } else {
                        anyhow::bail!("deterministic token configuration failure")
                    }
                })
                .is_err()
            );
            assert!(cell.has_failed());
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn token_io_failure_retries_after_bounded_backoff_without_revoking_support() {
    let cell = TokenResourceCell::default();
    let attempts = AtomicUsize::new(0);
    assert!(
        cell.get_or_init(|| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(std::io::Error::other("transient tokenizer I/O").into())
        })
        .is_err()
    );
    assert!(
        !cell.has_failed(),
        "transient I/O must not permanently withdraw token support"
    );
    assert!(
        cell.get_or_init(|| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(token_resources()))
        })
        .is_err()
    );
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        1,
        "backoff bounds retry work"
    );
    std::thread::sleep(std::time::Duration::from_millis(1050));
    cell.get_or_init(|| {
        attempts.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(token_resources()))
    })
    .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert!(cell.get().is_some());
}

#[test]
fn opaque_tiktoken_failure_uses_bounded_unavailable_retry_policy() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokenizer.tiktoken");
    std::fs::write(&path, b"not-base64 missing-rank\n").unwrap();
    std::fs::write(
        directory.path().join("config.json"),
        br#"{"model_type":"unsupported-opaque-fixture"}"#,
    )
    .unwrap();
    let mut card = ModelDeploymentCard::with_name_only("opaque-fixture");
    card.model_input = crate::model_type::ModelInput::Tokens;
    card.model_type = crate::model_type::ModelType::Chat;
    card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
    card.tokenizer = Some(crate::model_card::TokenizerKind::TikTokenModel(
        crate::common::checked_file::CheckedFile::from_disk(&path).unwrap(),
    ));
    let resources = DecisionResources::for_worker(
        card,
        Some(ScoreFactory::new(
            Arc::new(UnusedScoreEngine),
            TokenizerLoad::Card,
        )),
        None,
        RouterMode::RoundRobin,
        Arc::new(AllocatorTrimOnDrop::new()),
    );
    for _ in 0..2 {
        let cause = resources.prepare_token().err().unwrap();
        assert!(matches!(
            cause.downcast_ref::<SystemOneSelectionError>(),
            Some(SystemOneSelectionError::Unavailable)
        ));
        assert!(!resources.token.has_failed());
    }
    let error = match &*resources.token.state.lock() {
        TokenInitState::Backoff { error, .. } => error.clone(),
        _ => panic!("opaque failures must retain a retry gate"),
    };
    resources.prepare_token().err().unwrap();
    let state = resources.token.state.lock();
    assert!(
        matches!(&*state, TokenInitState::Backoff { error: current, .. } if Arc::ptr_eq(&error, current)),
        "requests inside backoff must not reparse the opaque artifact"
    );
}
