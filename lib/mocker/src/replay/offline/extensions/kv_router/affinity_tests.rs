// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_kv_router::config::RouterQueuePolicy;

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
fn lcfs_siblings_do_not_hide_a_queued_initializer() {
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
    let released = complete(&mut policy, 1, 3.0);
    assert_eq!(released.len(), 1);
    assert_eq!(
        released[0].request_id,
        Uuid::from_u128(4),
        "native LCFS order must be preserved"
    );
    commit(&mut policy, 4, 3.0);
    assert!(
        advance(&mut policy, 3.0).is_empty(),
        "bound busy rank must retain the sibling in the policy queue"
    );
    let last = complete(&mut policy, 4, 4.0);
    assert_eq!(last.len(), 1);
    assert_eq!(last[0].request_id, Uuid::from_u128(3));
    assert_eq!(last[0].scheduler_id, released[0].scheduler_id);
    commit(&mut policy, 3, 4.0);
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
