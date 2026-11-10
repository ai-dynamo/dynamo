// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::collections::HashMap;

use dynamo_kv_router::indexer::KvIndexerInterface;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, RoutingConstraints, StorageTier, WorkerWithDpRank,
    compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_router::services::indexer::backend::Indexer;
use dynamo_kv_router::services::selection::{
    PromptRequest, SelectAndReserveRequest, SelectionCacheConfig, SelectionCore, SelectionService,
    SelectionServiceBuilder, WorkerRequest,
};
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionId, WorkerType};
use dynamo_router_policies::{default_factory, default_registry, register};

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
    // Replacing scorer/picker state must leave active reservations intact.
    use dynamo_kv_router::plugins::worker_selection::RuntimePolicyUpdate;
    let control = partition.scheduler().policy_control();
    let initial = control.get().await.unwrap();
    control
        .update(RuntimePolicyUpdate::patch(
            initial.revision(),
            serde_json::json!({"prefill_load_scale": 2.0})
                .as_object()
                .unwrap()
                .clone(),
        ))
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
async fn runtime_updates_are_conditional_and_cancelled_writes_do_not_apply() {
    use dynamo_kv_router::plugins::worker_selection::{RuntimePolicyError, RuntimePolicyUpdate};
    let core = SelectionCore::try_new_local(
        KvRouterConfig {
            use_kv_events: false,
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
    let control = partition.scheduler().policy_control();
    let patch = || {
        serde_json::json!({"prefill_load_scale": 3.0})
            .as_object()
            .unwrap()
            .clone()
    };
    let (a, b) = tokio::join!(
        control.update(RuntimePolicyUpdate::patch(0, patch())),
        control.update(RuntimePolicyUpdate::patch(0, patch()))
    );
    assert!(matches!(
        (&a, &b),
        (Ok(_), Err(RuntimePolicyError::Conflict { .. }))
            | (Err(RuntimePolicyError::Conflict { .. }), Ok(_))
    ));

    // Poll once on the current-thread runtime to enqueue, then abandon the receiver before
    // the actor can process the command. A following get is an actor-ordering barrier.
    let mut abandoned = Box::pin(control.update(RuntimePolicyUpdate::reset(1)));
    std::future::poll_fn(|cx| {
        assert!(abandoned.as_mut().poll(cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    drop(abandoned);
    let current = control.get().await.unwrap();
    assert_eq!(current.revision(), 1);
    assert_eq!(current.parameters()["prefill_load_scale"], 3.0);
    core.shutdown();
}

#[tokio::test]
async fn scoring_error_does_not_book_and_allows_retry() {
    use dynamo_kv_router::WorkerSelectionPolicyError;
    use dynamo_kv_router::plugins::worker_selection::{
        WorkerCandidates, WorkerInputView, WorkerPicker, WorkerScorer, WorkerSelectionContext,
        WorkerSelectionPolicy,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct FailOnce(AtomicBool);
    impl WorkerScorer for FailOnce {
        fn score(
            &self,
            _: &WorkerSelectionContext<'_>,
            _: WorkerCandidates<'_>,
            costs: &mut [f64],
        ) -> Result<(), WorkerSelectionPolicyError> {
            if self.0.swap(false, Ordering::Relaxed) {
                return Err(WorkerSelectionPolicyError::failed("score failed"));
            }
            costs.fill(0.0);
            Ok(())
        }
    }
    struct First;
    impl WorkerPicker for First {
        fn pick(
            &self,
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
                vec![Box::new(FailOnce(AtomicBool::new(true)))],
                Box::new(First),
            )
        }),
    )
    .unwrap();
    core.upsert_worker(worker(1)).await.unwrap();
    let partition = core
        .partition(&RoutingPartitionId::new("model", "default"))
        .unwrap();
    assert!(matches!(
        partition.scheduler().policy_control().get().await,
        Err(dynamo_kv_router::plugins::worker_selection::RuntimePolicyError::Unsupported)
    ));
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

/// Tokens of the prompt the `kvcr` tests book: eight blocks of four.
fn prompt_tokens() -> Vec<u32> {
    (1..=32).collect()
}

/// The prompt's sequence hashes, block 0 first, as an engine publishes them.
fn prompt_hashes() -> Vec<u64> {
    compute_seq_hash_for_block(&compute_block_hash_for_seq(
        &prompt_tokens(),
        4,
        BlockHashOptions::default(),
    ))
}

/// Index `worker_id` as holding the prompt's first `blocks` blocks, as its KV events would.
async fn store_prefix(service: &SelectionService, worker_id: u64, blocks: usize) {
    let local = compute_block_hash_for_seq(&prompt_tokens(), 4, BlockHashOptions::default());
    let blocks = local
        .iter()
        .zip(prompt_hashes())
        .take(blocks)
        .map(|(&tokens_hash, sequence_hash)| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(sequence_hash),
            tokens_hash,
            mm_extra_info: None,
        })
        .collect();
    let partition = service
        .core()
        .partition(&RoutingPartitionId::new("model", "default"))
        .unwrap();
    partition
        .indexer()
        .apply_event_routed(RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks,
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        ))
        .await
        .unwrap();
    if let Indexer::Single { primary, .. } = partition.indexer() {
        primary.flush().await;
    }
}

/// A selection service built from the shipped registry with the router policy `policy_yaml`, as
/// the standalone service builds it. Worker 1 holds the whole prompt and is busy; idle worker 2
/// holds its first two blocks. Both advertise `router_hint` sources.
async fn kvcr_service(policy_yaml: &str) -> (SelectionService, tempfile::NamedTempFile) {
    let policy_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy_file.path(), policy_yaml).unwrap();
    let config = KvRouterConfig {
        use_kv_events: true,
        router_queue_threshold: None,
        router_policy_config: Some(policy_file.path().display().to_string()),
        ..Default::default()
    };
    let mut registry = default_registry();
    register(&mut registry).unwrap();
    let service = SelectionServiceBuilder::new(config, WorkerType::Aggregated, registry)
        .indexer_threads(1)
        .build()
        .await
        .unwrap();
    for worker_id in [1, 2] {
        service
            .upsert_worker(WorkerRequest {
                kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
                router_hint_worker_type: Some("decode".to_string()),
                router_hint_source_control_endpoints: HashMap::from([(
                    0,
                    format!("tcp://worker-{worker_id}:9000"),
                )]),
                ..worker(worker_id)
            })
            .await
            .unwrap();
    }
    store_prefix(&service, 1, 8).await;
    store_prefix(&service, 2, 2).await;
    // Busy worker 1 (8 prefill tokens, 2 decode blocks) costs 2 + 10 = 12 for the prompt. Worker 2
    // costs 6 + 8 = 14, less half its 6 copyable blocks at a 0.5 multiplier: 11.
    let mut load = reserve_request("load");
    load.prompt.token_ids = Some((100..108).collect());
    load.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
    service.select_and_reserve(load).await.unwrap();
    (service, policy_file)
}

fn prompt_request(selection_id: &str) -> SelectAndReserveRequest {
    let mut request = reserve_request(selection_id);
    request.prompt.token_ids = Some(prompt_tokens());
    request
}

/// `kvcr` credits worker 2 for the six blocks it can copy from worker 1, books it, and commits
/// it to that copy; `kv_fetch`, on by default under `kvcr`, sends the copy as `kv.fetch@1.0`.
#[tokio::test]
async fn kvcr_books_a_copy_that_kv_fetch_sends() {
    let (service, _policy_file) = kvcr_service(
        r#"
worker_selection:
  aggregated:
    type: kvcr
    parameters: {shared_cache_multiplier: 0.5, overlap_score_credit_decay: 0.0}
"#,
    )
    .await;
    let response = service
        .select_and_reserve(prompt_request("copy"))
        .await
        .unwrap();
    assert_eq!(response.worker_id, 2);
    let hint = response.kv_hint.expect("the committed copy");
    let actions: Vec<_> = hint
        .actions
        .iter()
        .map(|action| {
            (
                action.action_id.as_str(),
                action.action_type.as_str(),
                action.action_version.as_str(),
            )
        })
        .collect();
    assert_eq!(actions, [("a1", "kv.fetch", "1.0")]);
    // Every hash from block 0 through the source's prefix, including the two worker 2 holds.
    assert_eq!(
        serde_json::to_value(&hint.actions[0].payload).unwrap(),
        serde_json::json!({
            "source_control_endpoint": "tcp://worker-1:9000",
            "block_hashes": prompt_hashes(),
        })
    );
}

/// `kvcr` commits copies, so its role needs a hint that sends them: the default `kv_fetch` when the
/// entry lists no `hints`, or a listed `kv_fetch`. Listing `hints` without one fails the
/// partition's setup, and the worker can't be scheduled, with the reason naming the fix.
#[tokio::test]
async fn kvcr_partitions_need_a_hint_that_sends_copies() {
    for (hints, sends_copies) in [
        ("", true),
        ("    hints:\n      - type: kv_fetch\n", true),
        ("    hints: []\n", false),
    ] {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            format!(
                "worker_selection:\n  aggregated:\n    type: kvcr\n    parameters: {{shared_cache_multiplier: 0.5}}\n{hints}"
            ),
        )
        .unwrap();
        let config = KvRouterConfig {
            use_kv_events: true,
            router_queue_threshold: None,
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };
        let mut registry = default_registry();
        register(&mut registry).unwrap();
        let service = SelectionServiceBuilder::new(config, WorkerType::Aggregated, registry)
            .indexer_threads(1)
            .build()
            .await
            .unwrap();
        let record = service
            .upsert_worker(WorkerRequest {
                kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
                ..worker(1)
            })
            .await
            .unwrap();
        let reasons = record.not_schedulable_reasons.join("; ");
        assert_eq!(
            reasons.contains(
                "commits KV copies, but none of the hints in worker_selection.aggregated.hints sends them"
            ),
            !sends_copies,
            "hints={hints:?}: {reasons}"
        );
    }
}

/// `kv_fetch` sends a fetch `kvcr` didn't commit only when it adds at least
/// `min_additional_blocks` beyond the booked worker's own prefix, and always sends a committed one.
/// Without copy credit, `kvcr` commits nothing.
#[tokio::test]
async fn kv_fetch_holds_only_uncommitted_fetches_to_min_additional_blocks() {
    // Booked on worker 2, a fetch from worker 1 adds 8 - 2 = 6 blocks.
    for (multiplier, min_additional_blocks, sends) in
        [(0.0, 6, true), (0.0, 7, false), (0.5, 7, true)]
    {
        let (service, _policy_file) = kvcr_service(&format!(
            r#"
worker_selection:
  aggregated:
    type: kvcr
    parameters: {{shared_cache_multiplier: {multiplier}}}
    hints:
      - type: kv_fetch
        parameters: {{min_additional_blocks: {min_additional_blocks}}}
"#
        ))
        .await;
        let mut request = prompt_request("pinned");
        request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
        let response = service.select_and_reserve(request).await.unwrap();
        assert_eq!(
            response.kv_hint.is_some(),
            sends,
            "multiplier={multiplier} min_additional_blocks={min_additional_blocks}"
        );
    }
}
