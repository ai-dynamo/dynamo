// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use dynamo_custom_policy_builtin::default_factory;
use dynamo_kv_router::protocols::RoutingConstraints;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionCacheConfig, SelectionCore, WorkerRequest,
};
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionId};

fn worker(worker_id: u64) -> WorkerRequest {
    WorkerRequest {
        worker_id,
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        endpoint: Some(format!("http://worker-{worker_id}:8000")),
        block_size: Some(4),
        max_num_batched_tokens: Some(1024),
        ..Default::default()
    }
}

fn reserve_request(selection_id: &str) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        selection_id: Some(selection_id.to_string()),
        prompt: PromptRequest {
            token_ids: Some(vec![1, 2, 3, 4, 5]),
            ..Default::default()
        },
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
    }
}

#[tokio::test]
async fn shared_core_books_and_releases_with_the_builtin_default() {
    let core = SelectionCore::try_new_local(
        KvRouterConfig {
            use_kv_events: false,
            router_queue_threshold: None,
            ..Default::default()
        },
        1,
        Default::default(),
        SelectionCacheConfig::default(),
        default_factory(),
    )
    .unwrap();
    core.upsert_worker(worker(1)).await.unwrap();
    let partition = core
        .partition(&RoutingPartitionId::new("model", "default"))
        .unwrap();
    assert!(!partition.scheduler().has_request("request"));
    core.select_and_reserve(reserve_request("request"))
        .await
        .unwrap();
    assert!(partition.scheduler().has_request("request"));
    core.prefill_complete("request").await.unwrap();
    core.free_reservation("request").await.unwrap();
    assert!(!partition.scheduler().has_request("request"));
    assert!(core.free_reservation("request").await.is_err());
    core.shutdown();
}

#[tokio::test]
async fn scoring_error_does_not_book_and_allows_retry() {
    use dynamo_kv_router::{
        WorkerCandidates, WorkerInputView, WorkerPicker, WorkerScorer, WorkerSelectionContext,
        WorkerSelectionPolicy, WorkerSelectionPolicyError,
    };
    use std::sync::Arc;

    struct FailOnce(bool);
    impl WorkerScorer for FailOnce {
        fn score(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            if std::mem::take(&mut self.0) {
                return Err(WorkerSelectionPolicyError::failed("score failed"));
            }
            costs.fill(0.0);
            Ok(())
        }
    }
    struct First;
    impl WorkerPicker for First {
        fn pick(
            &mut self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(0)
        }
    }
    let core = SelectionCore::try_new_local(
        KvRouterConfig {
            use_kv_events: false,
            router_queue_threshold: None,
            ..Default::default()
        },
        1,
        Default::default(),
        SelectionCacheConfig::default(),
        Arc::new(|config, role, _| {
            WorkerSelectionPolicy::new(
                config.clone(),
                role.default_selector_label(),
                vec![Box::new(FailOnce(true))],
                Box::new(First),
            )
        }),
    )
    .unwrap();
    core.upsert_worker(worker(1)).await.unwrap();
    let partition = core
        .partition(&RoutingPartitionId::new("model", "default"))
        .unwrap();
    let error = core
        .select_and_reserve(reserve_request("request"))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("score failed"));
    assert!(!partition.scheduler().has_request("request"));
    core.select_and_reserve(reserve_request("request"))
        .await
        .unwrap();
    assert!(partition.scheduler().has_request("request"));
    core.free_reservation("request").await.unwrap();
    assert!(!partition.scheduler().has_request("request"));
    core.shutdown();
}
