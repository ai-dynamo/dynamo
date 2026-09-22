// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::{RouterPluginRegistry, RouterPlugins};

const EXAMPLE: &str = include_str!("../examples/thunderagent.yaml");

fn resolve(yaml: &str) -> (KvRouterConfig, RouterPlugins) {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), yaml).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(file.path().display().to_string()),
        ..Default::default()
    };
    let mut registry = RouterPluginRegistry::default();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    let plugins = registry.resolve_plugins(&config).unwrap();
    (config, plugins)
}

#[test]
fn shipping_plugins_does_not_enable_them_by_default() {
    let mut registry = RouterPluginRegistry::default();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    assert!(
        registry
            .resolve_plugins(&KvRouterConfig::default())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn worker_selection_rejects_admission_parameters() {
    for (parameters, accepted) in [
        ("", true),
        ("      parameters: {}\n", true),
        ("      parameters: {pause_threshold: 0.9}\n", false),
    ] {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), format!(
            "worker_selection:\n  aggregated: thunderagent\n  instances:\n    - name: thunderagent\n      type: thunderagent\n{parameters}"
        )).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        let mut registry = RouterPluginRegistry::default();
        dynamo_custom_policy_builtin::register(&mut registry).unwrap();
        let result = registry.resolve_plugins(&config);
        if accepted {
            assert!(result.unwrap().worker_selection().is_some());
        } else {
            let Err(error) = result else {
                panic!("misplaced admission settings must fail startup")
            };
            assert!(error.to_string().contains("pause_threshold"));
        }
    }
}

#[test]
fn thunderagent_parameters_are_validated_at_startup() {
    for parameters in ["pause_target: 1.1", "unknown_parameter: 1"] {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            format!("request_classifier:\n  type: thunderagent\n  parameters:\n    {parameters}\n"),
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        let mut registry = RouterPluginRegistry::default();
        dynamo_custom_policy_builtin::register(&mut registry).unwrap();
        let error = match registry.resolve_plugins(&config) {
            Ok(_) => panic!("invalid ThunderAgent parameters must fail startup"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains(parameters.split(':').next().unwrap())
        );
    }
}

mod scheduler {
    use std::collections::HashMap;
    use std::future::{Future, poll_fn};
    use std::sync::Arc;
    use std::task::Poll;
    use std::time::Duration;

    use dynamo_kv_router::config::RouterQueuePolicy;
    use dynamo_kv_router::plugins::request_classifier::{
        RequestClassifierContext, RequestClassifierWorker,
    };
    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{
        LocalScheduler, NoopOverlapScoresRefresh, OverlapSignals, PolicyProfile, ScheduleMode,
        ScheduleRequest, SessionContext,
    };
    use dynamo_kv_router::{
        ActiveSequencesMultiWorker, NoopSequencePublisher, RoutingPartitionRef, WorkerType,
    };
    use tokio::sync::watch;
    use tokio_util::sync::CancellationToken;

    use super::*;

    #[derive(Clone, PartialEq)]
    struct Worker;

    impl WorkerConfigLike for Worker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            Some(4096)
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(128)
        }
    }

    fn request(id: &str, session: Option<&str>) -> ScheduleRequest {
        ScheduleRequest {
            mode: ScheduleMode::TrackedWithLifecycle {
                request_id: id.into(),
            },
            token_seq: Some(vec![1]),
            block_hashes: None,
            isl_tokens: 64,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: session.map(|id| SessionContext::new(id.into(), None, None, None)),
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
        }
    }

    /// Load the shipped YAML through the same catalog as the frontend, then exercise both plugin
    /// roles through the host scheduler and its asynchronous lifecycle delivery.
    #[tokio::test]
    async fn shipped_catalog_serializes_sessions_and_releases_them_after_abort() {
        let (config, plugins) = resolve(EXAMPLE);
        let worker = WorkerWithDpRank::new(1, 0);
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            64,
            HashMap::from([(1, (0, 1))]),
            false,
            0,
            "test",
        ));
        let (_worker_tx, worker_rx) = watch::channel(HashMap::from([(1, Worker)]));
        let shutdown = CancellationToken::new();
        let _shutdown_guard = shutdown.clone().drop_guard();
        let selector = plugins.worker_selection().unwrap()(
            &config,
            WorkerType::Aggregated,
            RoutingPartitionRef::new("model", "default"),
        );
        let scheduler = LocalScheduler::new(
            slots,
            worker_rx,
            PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
            64,
            selector,
            None,
            None::<Arc<NoopOverlapScoresRefresh>>,
            None,
            None,
            Duration::from_secs(60),
            true,
            shutdown.clone(),
            "test",
            false,
        );
        let classifier =
            plugins.request_classifier().unwrap()(RequestClassifierContext::new(64, move || {
                vec![RequestClassifierWorker::new(worker, Some(128))]
            }));
        assert!(scheduler.install_request_classifier(classifier, shutdown));

        let mut first = scheduler.begin_request_lifecycle("first").unwrap().unwrap();
        let selected = scheduler
            .schedule_request(request("first", Some("session")))
            .await
            .unwrap();
        assert_eq!(selected.best_worker, worker);
        first.sent(worker);

        let mut second = scheduler
            .begin_request_lifecycle("second")
            .unwrap()
            .unwrap();
        let mut pending = Box::pin(scheduler.schedule_request(request("second", Some("session"))));
        assert!(poll_fn(|cx| Poll::Ready(pending.as_mut().poll(cx).is_pending())).await);

        // An unrelated request without session context still passes through admission.
        let mut anonymous = scheduler
            .begin_request_lifecycle("anonymous")
            .unwrap()
            .unwrap();
        let selected = tokio::time::timeout(
            Duration::from_secs(2),
            scheduler.schedule_request(request("anonymous", None)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(selected.best_worker, worker);
        anonymous.sent(worker);
        anonymous.complete();
        scheduler.free("anonymous").await.unwrap();

        first.abort(None);
        scheduler.free("first").await.unwrap();
        let selected = tokio::time::timeout(Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(selected.best_worker, worker);
        second.sent(worker);
        second.complete();
        scheduler.free("second").await.unwrap();

        // Completion also releases the session for its next turn.
        let mut third = scheduler.begin_request_lifecycle("third").unwrap().unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            scheduler.schedule_request(request("third", Some("session"))),
        )
        .await
        .unwrap()
        .unwrap();
        third.sent(worker);
        third.complete();
        scheduler.free("third").await.unwrap();
    }
}
