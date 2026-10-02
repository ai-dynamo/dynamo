// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_kv_router::config::RouterQueuePolicy;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData,
};
use dynamo_kv_router::scheduling::RouterPolicyConfig;

type Policy = KvRouterPlacement;

fn request(id: u128) -> DirectRequest {
    DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens: vec![7; 64],
        max_output_tokens: 8,
        ..Default::default()
    }
}

fn placement(queue_policy: Option<RouterQueuePolicy>) -> Policy {
    let args = MockEngineArgs::builder()
        .block_size(16)
        .dp_size(2)
        .max_num_batched_tokens(Some(16))
        .build()
        .unwrap();
    let config = KvRouterConfig {
        router_temperature: 0.0,
        router_track_prefill_tokens: true,
        router_queue_threshold: queue_policy.map(|_| 0.5),
        router_queue_policy: queue_policy.unwrap_or_default(),
        ..Default::default()
    };
    let evidence = Arc::new(Mutex::new(RouterEvidence {
        capture_decisions: true,
        ..Default::default()
    }));
    Policy::new_with_selector_seed(&args, Some(config), None, 1, None)
        .unwrap()
        .with_affinity_and_evidence(
            Some(ReplayAffinityConfig {
                mode: ReplayAffinityMode::Session,
                ttl_seconds: 1.25,
            }),
            Some(evidence),
            "aggregated",
        )
        .unwrap()
}

fn place(policy: &mut Policy, id: u128, session: &str, now: f64) -> PlacementDecision {
    policy
        .place(
            &request(id),
            KvReplayMetadata::default(),
            Some(session.into()),
            now,
        )
        .unwrap()
        .decision
}

#[test]
#[ignore = "manual before/after affinity backlog benchmark; run with --ignored --nocapture"]
fn benchmark_affinity_blocked_backlog() {
    for pending in [128, 1024, 4096] {
        let mut policy = placement(Some(RouterQueuePolicy::Fcfs));
        for (id, session) in [(1, "busy-a"), (2, "busy-b")] {
            immediate(place(&mut policy, id, session, 0.0));
            commit(&mut policy, id, 0.0);
        }
        for id in 3..pending + 3 {
            assert!(matches!(
                place(&mut policy, id, "busy-a", 0.0),
                PlacementDecision::Queued
            ));
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = std::time::Instant::now();
            for _ in 0..100 {
                assert!(complete(&mut policy, 99_999, 0.0).is_empty());
            }
            samples.push(started.elapsed().as_nanos() / 100);
        }
        samples.sort_unstable();
        assert_eq!(policy.router.pending_count(), pending as usize);
        println!(
            "affinity_blocked_backlog pending={pending} median_ns_per_drain={}",
            samples[2]
        );
    }
}

fn immediate(decision: PlacementDecision) -> Placement {
    match decision {
        PlacementDecision::Immediate(placement) => placement,
        PlacementDecision::Queued => panic!("expected immediate placement"),
    }
}

fn commit(policy: &mut Policy, id: u128, now: f64) {
    <Policy as PlacementPolicy<DirectRequest>>::dispatch_committed(
        policy,
        Uuid::from_u128(id),
        now,
    )
    .unwrap();
}

fn advance(policy: &mut Policy, now: f64) -> Vec<Placement> {
    <Policy as PlacementPolicy<DirectRequest>>::advance_clock(policy, now).unwrap()
}

fn complete(policy: &mut Policy, id: u128, now: f64) -> Vec<Placement> {
    <Policy as PlacementPolicy<DirectRequest>>::request_terminal(policy, Uuid::from_u128(id), now)
        .unwrap()
}

fn cache_request_on_rank(policy: &mut Policy, request: &DirectRequest, dp_rank: u32) {
    let hashes = ReplayRequestHashes::from_tokens(&request.tokens, policy.router.block_size);
    policy
        .router
        .on_kv_events(vec![RouterEvent::new(
            0,
            KvCacheEvent {
                event_id: 1,
                dp_rank,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: hashes
                        .local_block_hashes
                        .into_iter()
                        .enumerate()
                        .map(|(index, hash)| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(index as u64 + 1),
                            tokens_hash: LocalBlockHash(hash),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
            },
        )])
        .unwrap();
}

#[test]
fn affinity_wspt_ignores_cache_on_another_dp_rank() {
    for commit_before_arrival in [true, false] {
        let mut policy = placement(Some(RouterQueuePolicy::Wspt));
        let bound = immediate(place(&mut policy, 1, "conversation", 0.0));
        if commit_before_arrival {
            commit(&mut policy, 1, 0.0);
        }
        let long = request(2);
        cache_request_on_rank(&mut policy, &long, 1 - bound.scheduler_id as u32);
        let mut short = request(3);
        short.tokens = vec![9; 16];
        for request in [&long, &short] {
            assert!(matches!(
                policy
                    .place(
                        request,
                        KvReplayMetadata::default(),
                        Some("conversation".into()),
                        0.0
                    )
                    .unwrap()
                    .decision,
                PlacementDecision::Queued
            ));
        }
        if !commit_before_arrival {
            commit(&mut policy, 1, 0.0);
        }
        let released = complete(&mut policy, 1, 0.0);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(3));
        assert_eq!(released[0].scheduler_id, bound.scheduler_id);
        commit(&mut policy, 3, 0.0);
        let released = complete(&mut policy, 3, 0.0);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(2));
        assert_eq!(released[0].scheduler_id, bound.scheduler_id);
    }
}

#[test]
fn affinity_cache_bucket_ignores_cache_on_another_dp_rank() {
    for commit_before_arrival in [true, false] {
        let mut policy = placement(Some(RouterQueuePolicy::Wspt));
        let profile = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 8
  - name: uncached
    policy_family: standard
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 8
"#,
        )
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Wspt);
        policy.router.pending = PolicyQueue::new(profile.clone());
        policy.router.profile = profile;
        let bound = immediate(place(&mut policy, 1, "conversation", 0.0));
        if commit_before_arrival {
            commit(&mut policy, 1, 0.0);
        }
        cache_request_on_rank(&mut policy, &request(2), 1 - bound.scheduler_id as u32);
        assert!(matches!(
            place(&mut policy, 2, "conversation", 0.0),
            PlacementDecision::Queued
        ));
        if !commit_before_arrival {
            commit(&mut policy, 1, 0.0);
        }
        assert!(advance(&mut policy, 0.0).is_empty());
        let queued = policy.router.pending.entries().next().unwrap();
        assert_eq!(
            policy.router.profile.class(queued.class_index()).name,
            "uncached"
        );
        assert_eq!(queued.snapshot().cached_tokens, 0);
        assert_eq!(queued.snapshot().scheduling_cost_tokens, 64);
    }
}

#[test]
fn initialization_waits_for_dispatch_then_releases_same_worker_and_dp() {
    let mut policy = placement(None);
    let first = immediate(place(&mut policy, 1, "conversation", 0.0));
    assert!(matches!(
        place(&mut policy, 2, "conversation", 0.0),
        PlacementDecision::Queued
    ));
    assert!(advance(&mut policy, 0.0).is_empty());

    commit(&mut policy, 1, 0.0);
    let released = advance(&mut policy, 0.0);
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].scheduler_id, first.scheduler_id);
    commit(&mut policy, 2, 0.0);
    complete(&mut policy, 1, 20.0);
    complete(&mut policy, 2, 25.0);
    let evidence = policy.router.evidence.as_ref().unwrap().lock().unwrap();
    assert_eq!(evidence.post_dispatch_checks, 2);
    assert_eq!(evidence.affinity_hits, 1);
    assert_eq!(
        evidence.decisions[0]["dp_rank"],
        evidence.decisions[1]["dp_rank"]
    );
}

#[test]
fn committed_group_reaches_waiters_behind_another_initializer() {
    for queue_policy in [None, Some(RouterQueuePolicy::Fcfs)] {
        let mut policy = placement(queue_policy);
        immediate(place(&mut policy, 1, "still-initializing", 0.0));
        let bound = immediate(place(&mut policy, 2, "committed", 0.0));
        for (id, session) in [(3, "still-initializing"), (4, "committed")] {
            assert!(matches!(
                place(&mut policy, id, session, 0.0),
                PlacementDecision::Queued
            ));
        }
        commit(&mut policy, 2, 0.0);
        let mut released = advance(&mut policy, 0.0);
        if queue_policy.is_some() {
            assert!(released.is_empty(), "the bound rank is still full");
            let queued = policy
                .router
                .pending
                .entries()
                .find(|entry| entry.payload().uuid == Uuid::from_u128(4))
                .unwrap();
            assert!(
                matches!(
                    *queued.payload().affinity_hold.borrow(),
                    Some(Hold::Bound { .. })
                ),
                "a hidden waiter must hold its native lease as soon as the binding is published"
            );
            released = complete(&mut policy, 2, 2_000.0);
        }
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(4));
        assert_eq!(released[0].scheduler_id, bound.scheduler_id);
        commit(&mut policy, 4, 2_000.0);
        assert_eq!(
            policy
                .router
                .evidence
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .affinity_hits,
            1
        );
    }
}

#[test]
fn failed_dispatch_releases_native_slots_and_initialization() {
    let mut policy = placement(None);
    immediate(place(&mut policy, 1, "conversation", 0.0));
    assert!(matches!(
        place(&mut policy, 2, "conversation", 0.0),
        PlacementDecision::Queued
    ));
    <Policy as PlacementPolicy<DirectRequest>>::dispatch_aborted(
        &mut policy,
        Uuid::from_u128(1),
        0.0,
    )
    .unwrap();
    assert!(
        policy
            .router
            .slots
            .active_tokens(policy.router.decay_now(0.0))
            .values()
            .all(|tokens| *tokens == 0)
    );
    let released = advance(&mut policy, 0.0);
    assert_eq!(released.len(), 1);
    commit(&mut policy, 2, 0.0);
    complete(&mut policy, 2, 1.0);
    let evidence = policy.router.evidence.as_ref().unwrap().lock().unwrap();
    assert_eq!(evidence.dispatch_aborts, 1);
    assert_eq!(
        evidence.affinity_hits, 0,
        "an aborted dispatch must not publish a binding"
    );
}

#[test]
fn virtual_ttl_starts_at_last_terminal_and_accepts_fractional_seconds() {
    let mut policy = placement(None);
    immediate(place(&mut policy, 1, "conversation", 0.0));
    commit(&mut policy, 1, 0.0);
    // Active leases survive arbitrarily long replay execution.
    immediate(place(&mut policy, 2, "conversation", 5_000.0));
    commit(&mut policy, 2, 5_000.0);
    complete(&mut policy, 1, 5_000.0);
    complete(&mut policy, 2, 5_100.0);
    immediate(place(&mut policy, 3, "conversation", 6_349.0));
    commit(&mut policy, 3, 6_349.0);
    complete(&mut policy, 3, 6_400.0);
    immediate(place(&mut policy, 4, "conversation", 7_650.0));
    commit(&mut policy, 4, 7_650.0);
    let evidence = policy.router.evidence.as_ref().unwrap().lock().unwrap();
    assert_eq!(
        evidence
            .decisions
            .iter()
            .map(|row| row["binding_reused"].as_bool().unwrap())
            .collect::<Vec<_>>(),
        vec![false, true, true, false]
    );
}

#[test]
fn lcfs_waits_for_the_native_initialization_owner() {
    let mut policy = placement(Some(RouterQueuePolicy::Lcfs));
    // Occupy both DP ranks before either of the two queued siblings is selected.
    immediate(place(&mut policy, 1, "busy-a", 0.0));
    commit(&mut policy, 1, 0.0);
    immediate(place(&mut policy, 2, "busy-b", 0.0));
    commit(&mut policy, 2, 0.0);
    assert!(matches!(
        place(&mut policy, 3, "siblings", 1.0),
        PlacementDecision::Queued
    ));
    assert!(matches!(
        place(&mut policy, 4, "siblings", 2.0),
        PlacementDecision::Queued
    ));
    assert_eq!(policy.router.pending.pending_count(), 1);
    assert_eq!(policy.router.pending_count(), 2);
    let released = complete(&mut policy, 1, 3.0);
    assert_eq!(released.len(), 1);
    assert_eq!(
        released[0].request_id,
        Uuid::from_u128(3),
        "the live host keeps initialization through queueing; its sibling has not entered LCFS yet"
    );
    commit(&mut policy, 3, 3.0);
    assert!(
        advance(&mut policy, 3.0).is_empty(),
        "bound busy rank must retain the sibling in the policy queue"
    );
    let last = complete(&mut policy, 3, 4.0);
    assert_eq!(last.len(), 1);
    assert_eq!(last[0].request_id, Uuid::from_u128(4));
    assert_eq!(last[0].scheduler_id, released[0].scheduler_id);
    commit(&mut policy, 4, 4.0);
}

#[test]
fn canceling_a_queued_initializer_wakes_its_waiting_sibling() {
    let mut policy = placement(Some(RouterQueuePolicy::Wspt));
    for (id, session) in [(1, "busy-a"), (2, "busy-b")] {
        immediate(place(&mut policy, id, session, 0.0));
        commit(&mut policy, id, 0.0);
    }
    for id in 3..=5 {
        assert!(matches!(
            place(&mut policy, id, "waiting", 0.0),
            PlacementDecision::Queued
        ));
    }
    assert_eq!(policy.router.pending_count(), 3);
    assert!(policy.router.cancel_pending(Uuid::from_u128(5)));
    assert!(policy.router.cancel_pending(Uuid::from_u128(3)));
    assert_eq!(policy.router.pending_count(), 1);
    assert_eq!(policy.router.wakeup_ms, Some(0.0));
    assert!(advance(&mut policy, 0.0).is_empty());
    let released = complete(&mut policy, 1, 0.0);
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].request_id, Uuid::from_u128(4));
    assert_eq!(policy.router.pending_count(), 0);
    commit(&mut policy, 4, 0.0);
}

#[test]
fn affinity_waiters_observe_cache_at_binding_readiness() {
    for compact_hashes in [false, true] {
        let mut policy = placement(Some(RouterQueuePolicy::Wspt));
        let bound = immediate(place(&mut policy, 1, "conversation", 0.0));
        assert!(matches!(
            policy
                .place(
                    &request(2),
                    KvReplayMetadata::from_hashes(compact_hashes.then(|| {
                        ReplayRequestHashes::from_tokens(
                            &request(2).tokens,
                            policy.router.block_size,
                        )
                    })),
                    Some("conversation".into()),
                    0.0,
                )
                .unwrap()
                .decision,
            PlacementDecision::Queued
        ));
        assert_eq!(policy.router.pending.pending_count(), 0);
        cache_request_on_rank(&mut policy, &request(2), bound.scheduler_id as u32);
        commit(&mut policy, 1, 0.0);
        assert!(advance(&mut policy, 0.0).is_empty());
        let queued = policy.router.pending.entries().next().unwrap();
        assert_eq!(queued.snapshot().cached_tokens, 64);
        assert_eq!(queued.snapshot().scheduling_cost_tokens, 1);
        assert!(
            queued
                .payload()
                .token_seq
                .as_ref()
                .is_some_and(|sequence| !sequence.is_empty())
        );
    }
}

#[test]
fn affinity_waiters_use_normal_queue_admission_limits() {
    let mut policy = placement(Some(RouterQueuePolicy::Wspt));
    let profile = RouterPolicyConfig::from_yaml(
        r#"
default_policy_family: limited
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: limited
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 8
  - name: limited
    policy_family: limited
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 8
    request_queue_limit_per_worker: 1
"#,
    )
    .unwrap()
    .resolve_profile(None, None, RouterQueuePolicy::Wspt);
    policy.router.pending = PolicyQueue::new(profile.clone());
    policy.router.profile = profile;
    let bound = immediate(place(&mut policy, 1, "conversation", 0.0));
    cache_request_on_rank(&mut policy, &request(2), 1 - bound.scheduler_id as u32);
    for id in 2..=4 {
        assert!(matches!(
            place(&mut policy, id, "conversation", 0.0),
            PlacementDecision::Queued
        ));
    }
    commit(&mut policy, 1, 0.0);
    let error =
        <Policy as PlacementPolicy<DirectRequest>>::advance_clock(&mut policy, 0.0).unwrap_err();
    let rejection = error
        .downcast_ref::<dynamo_kv_router::scheduling::QueueRejection>()
        .unwrap();
    assert_eq!(rejection.policy_class, "limited");
    assert_eq!((rejection.current, rejection.limit), (2, 2));
    assert_eq!(policy.router.pending_count(), 2);
}

#[test]
fn removed_affinity_worker_retries_queue_admission_and_waiters() {
    let mut policy = placement(Some(RouterQueuePolicy::Wspt));
    immediate(place(&mut policy, 1, "conversation", 0.0));
    commit(&mut policy, 1, 0.0);
    for id in 2..=3 {
        assert!(matches!(
            place(&mut policy, id, "conversation", 0.0),
            PlacementDecision::Queued
        ));
    }
    policy.router.remove_worker(0).unwrap();
    assert!(
        policy
            .router
            .on_topology_changed(0.0)
            .unwrap()
            .admissions
            .is_empty()
    );
    assert_eq!(policy.router.pending_count(), 2);
    policy.router.add_worker(1).unwrap();
    let released = policy.router.on_topology_changed(0.0).unwrap().admissions;
    assert_eq!(released.len(), 1);
    let first = released[0].uuid;
    assert!(released[0].worker_idx >= 2);
    commit(&mut policy, first.as_u128(), 0.0);
    assert!(advance(&mut policy, 0.0).is_empty());
    let released = complete(&mut policy, first.as_u128(), 0.0);
    assert_eq!(released.len(), 1);
    assert_ne!(released[0].request_id, first);
    assert!(released[0].scheduler_id >= 2);
    assert_eq!(policy.router.pending_count(), 0);
}

#[test]
fn queued_affinity_does_not_block_another_idle_rank() {
    let mut policy = placement(Some(RouterQueuePolicy::Fcfs));
    let busy = immediate(place(&mut policy, 1, "busy", 0.0));
    commit(&mut policy, 1, 0.0);
    let freeing = immediate(place(&mut policy, 2, "freeing", 0.0));
    commit(&mut policy, 2, 0.0);
    assert_ne!(busy.scheduler_id, freeing.scheduler_id);
    assert!(matches!(
        place(&mut policy, 3, "busy", 1.0),
        PlacementDecision::Queued
    ));
    assert!(matches!(
        place(&mut policy, 4, "freeing", 2.0),
        PlacementDecision::Queued
    ));
    let released = complete(&mut policy, 2, 3.0);
    assert_eq!(
        released.len(),
        1,
        "an available bound rank must make progress"
    );
    assert_eq!(released[0].request_id, Uuid::from_u128(4));
    assert_eq!(released[0].scheduler_id, freeing.scheduler_id);
    commit(&mut policy, 4, 3.0);
    let released = complete(&mut policy, 1, 4.0);
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].request_id, Uuid::from_u128(3));
    assert_eq!(released[0].scheduler_id, busy.scheduler_id);
    commit(&mut policy, 3, 4.0);
}

#[test]
fn new_arrival_to_idle_affinity_rank_bypasses_other_rank_backlog() {
    let mut policy = placement(Some(RouterQueuePolicy::Fcfs));
    let busy = immediate(place(&mut policy, 1, "busy", 0.0));
    commit(&mut policy, 1, 0.0);
    let freeing = immediate(place(&mut policy, 2, "freeing", 0.0));
    commit(&mut policy, 2, 0.0);
    assert_ne!(busy.scheduler_id, freeing.scheduler_id);
    assert!(complete(&mut policy, 2, 1.0).is_empty());
    assert!(matches!(
        place(&mut policy, 3, "busy", 2.0),
        PlacementDecision::Queued
    ));
    let placement = immediate(place(&mut policy, 4, "freeing", 3.0));
    assert_eq!(placement.scheduler_id, freeing.scheduler_id);
    commit(&mut policy, 4, 3.0);
}

#[test]
fn initialized_siblings_do_not_block_a_new_session_on_an_idle_rank() {
    let mut policy = placement(Some(RouterQueuePolicy::Fcfs));
    let busy = immediate(place(&mut policy, 1, "busy", 0.0));
    commit(&mut policy, 1, 0.0);
    immediate(place(&mut policy, 2, "freeing", 0.0));
    commit(&mut policy, 2, 0.0);
    for (id, session) in [(3, "siblings"), (4, "siblings"), (5, "new-session")] {
        assert!(matches!(
            place(&mut policy, id, session, id as f64),
            PlacementDecision::Queued
        ));
    }
    let initialized = complete(&mut policy, 2, 6.0);
    assert_eq!(initialized.len(), 1);
    assert_eq!(initialized[0].request_id, Uuid::from_u128(3));
    commit(&mut policy, 3, 6.0);
    assert!(advance(&mut policy, 6.0).is_empty());
    let released = complete(&mut policy, 1, 7.0);
    assert_eq!(
        released.len(),
        1,
        "a sibling's pin must not block another rank"
    );
    assert_eq!(released[0].request_id, Uuid::from_u128(5));
    assert_eq!(released[0].scheduler_id, busy.scheduler_id);
    commit(&mut policy, 5, 7.0);
}

#[test]
fn aborted_initializer_unblocks_its_rank_despite_another_bound_backlog() {
    let mut policy = placement(Some(RouterQueuePolicy::Fcfs));
    immediate(place(&mut policy, 1, "busy", 0.0));
    commit(&mut policy, 1, 0.0);
    let aborted = immediate(place(&mut policy, 2, "aborted", 0.0));
    assert!(matches!(
        place(&mut policy, 3, "busy", 1.0),
        PlacementDecision::Queued
    ));
    assert!(matches!(
        place(&mut policy, 4, "aborted", 2.0),
        PlacementDecision::Queued
    ));
    <Policy as PlacementPolicy<DirectRequest>>::dispatch_aborted(
        &mut policy,
        Uuid::from_u128(2),
        3.0,
    )
    .unwrap();
    let released = advance(&mut policy, 3.0);
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].request_id, Uuid::from_u128(4));
    assert_eq!(released[0].scheduler_id, aborted.scheduler_id);
    commit(&mut policy, 4, 3.0);
}

#[test]
fn oversized_affinity_clock_returns_error_without_advancing() {
    let mut policy = placement(None);
    for now in [f64::MAX, 1e22] {
        let error = policy
            .place(
                &request(1),
                KvReplayMetadata::default(),
                Some("session".into()),
                now,
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("replay affinity time is too large")
        );
        assert_eq!(policy.router.affinity.as_ref().unwrap().clock.now_ms(), 0.0);
    }
    immediate(place(&mut policy, 2, "session", 1.0));
    commit(&mut policy, 2, 1.0);
}

#[test]
fn sibling_group_uses_conversation_ancestry_and_play_namespace() {
    use aisimulate_core::replay::{
        AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1, AgenticConversationLineage, AgenticRuntimeIdentity,
        ReplayRequestContext,
    };
    let config = ReplayAffinityConfig {
        mode: ReplayAffinityMode::SiblingGroup,
        ttl_seconds: 3600.0,
    };
    let child = |play: &str, conversation: &str, parent: Option<&str>| {
        let mut request = request(1);
        request.replay_context = Some(ReplayRequestContext {
            authored_id: "request".into(),
            session_id: None,
            turn_index: None,
            metadata: serde_json::Value::Null,
            prompt_token_source: Default::default(),
            agentic: Some(AgenticRuntimeIdentity {
                request_id: "request".into(),
                play_id: play.into(),
                conversation_id: conversation.into(),
                lane_id: None,
                root_id: None,
                parent_id: None,
                cache_id: None,
                lineage: Some(AgenticConversationLineage {
                    schema: AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1.into(),
                    root_conversation_id: "root".into(),
                    parent_conversation_id: parent.map(Into::into),
                }),
            }),
        });
        request
    };
    let first = config
        .group_key(
            &child("play", "child-a", Some("parent-before-snapshot")),
            None,
        )
        .unwrap();
    let sibling = config
        .group_key(
            &child("play", "child-b", Some("parent-before-snapshot")),
            None,
        )
        .unwrap();
    assert_eq!(first, sibling);
    assert_ne!(
        first,
        config
            .group_key(
                &child("other-play", "child-b", Some("parent-before-snapshot")),
                None
            )
            .unwrap()
    );
    assert_ne!(
        first,
        config
            .group_key(&child("play", "parent-before-snapshot", None), None)
            .unwrap()
    );
    assert!(
        config
            .group_key(&request(2), None)
            .unwrap_err()
            .to_string()
            .contains("Agentic identity")
    );
    let mut ambiguous = child("play", "ambiguous", Some("parent"));
    ambiguous
        .replay_context
        .as_mut()
        .unwrap()
        .agentic
        .as_mut()
        .unwrap()
        .lineage = None;
    assert!(
        config
            .group_key(&ambiguous, None)
            .unwrap_err()
            .to_string()
            .contains("conversation lineage")
    );
}

#[cfg(feature = "python-replay")]
#[test]
fn canonical_replay_reuses_native_cache_in_aggregated_and_disaggregated_dp() {
    use serde_json::json;
    for topology in [
        json!({"kind":"aggregated", "workers":{"initial_workers":2}}),
        json!({"kind":"disaggregated", "prefill":{"initial_workers":2}, "decode":{"initial_workers":2}}),
    ] {
        let payload = json!({
            "version":1, "topology":topology,
            "engine":{"dp_size":2,"rank":{"block_size":16,"num_gpu_blocks":64,
                "timing_model":{"type":"fixed","prefill_ms":1.0,"decode_ms":1.0}}},
            "adapters":{"placement":{"provider":"round_robin"},"scaling":{"provider":"none"}},
            "record_per_request":true,
            "requests":[
                {"id":"first","arrival_time_ms":0.0,"input_tokens":64,"input_token_ids":vec![7;64],"output_tokens":2,"session_id":"conversation"},
                {"id":"second","arrival_time_ms":100.0,"input_tokens":64,"input_token_ids":vec![7;64],"output_tokens":2,"session_id":"conversation"}
            ]
        });
        let result = crate::replay::run_canonical_replay_json(
            &payload.to_string(),
            None,
            None,
            Some(ReplayAffinityConfig {
                mode: ReplayAffinityMode::Session,
                ttl_seconds: 3600.0,
            }),
        )
        .unwrap();
        let report: serde_json::Value = serde_json::from_str(&result).unwrap();
        let evidence = &report["dynamo_policy"];
        assert!(evidence["physical_kv_events"].as_u64().unwrap() > 0);
        assert_eq!(evidence["decision_count"], evidence["post_dispatch_checks"]);
        let decisions = evidence["decisions"].as_array().unwrap();
        for role in ["aggregated", "prefill", "decode"] {
            let role_decisions: Vec<_> =
                decisions.iter().filter(|row| row["role"] == role).collect();
            if role_decisions.is_empty() {
                continue;
            }
            assert_eq!(role_decisions.len(), 2);
            assert_eq!(
                role_decisions[0]["worker_id"],
                role_decisions[1]["worker_id"]
            );
            assert_eq!(role_decisions[0]["dp_rank"], role_decisions[1]["dp_rank"]);
            assert_eq!(role_decisions[1]["binding_reused"], true);
        }
        let rows = report["per_request"].as_array().unwrap();
        assert!(
            rows.iter()
                .flat_map(|row| row["routing_history"].as_array().unwrap())
                .any(|route| { route["reported_overlap_tokens"].as_u64().unwrap_or(0) > 0 }),
            "{report}"
        );
    }
}
