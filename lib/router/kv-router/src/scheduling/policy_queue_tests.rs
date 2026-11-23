// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The policy queue driven through profiles the YAML resolver built; the queue itself lives
//! in the admission stage, which exposes its indexes to these tests through `testing` hooks.

use tokio::time::Instant;

use crate::scheduling::policy_config::{RouterPolicyConfig, RouterQueuePolicy};
use dynamo_admission::{
    BlockReason, LaneCheck, LaneKey, PolicyClassConfig, PolicyProfile, PolicyQueue,
    PolicyQueueEntry, PolicyQueueStats, QueueLane, QueueLimitKind, QueueMetadata, QueueSnapshot,
    RecheckScope,
};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_session_affinity::WorkerAffinityTarget;

/// A session lane named after a number, where a test wants several distinct lanes.
fn lane_of(id: u64) -> QueueLane {
    QueueLane::Keyed(lane_key(id))
}

fn lane_key(id: u64) -> LaneKey {
    LaneKey::new(&format!("lane-{id}"))
}

fn profile(yaml: &str) -> PolicyProfile {
    RouterPolicyConfig::from_yaml(yaml)
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
}

/// Metadata for an entry with no explicit or class due time: how the kernel enqueues a
/// request outside EDF.
fn no_due_metadata(
    class_index: usize,
    snapshot: QueueSnapshot,
    arrival_offset_secs: f64,
) -> QueueMetadata {
    QueueMetadata {
        class_index,
        snapshot,
        due_at: None,
        class_due_at: None,
        arrival_offset_secs,
        lane_key: None,
    }
}

fn admission_profile() -> PolicyProfile {
    profile(
        r#"
policy_classes:
  - name: agents
    queue_policy: fcfs
    quantum: 10
"#,
    )
}

#[test]
fn per_worker_caps_scale_and_remain_pre_add() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
policy_classes:
  - name: capped
    quantum: 10
    request_queue_limit_per_worker: 1
    raw_isl_token_queue_limit_per_worker: 5
    cached_token_queue_limit_per_worker: 3
"#,
    ));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(8, 4), 0.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "first",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(100, 100), 1.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "overshoot",
        )
        .unwrap();
    let (rejection, payload) = queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 2.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "rejected",
        )
        .unwrap_err();
    assert_eq!(payload, "rejected");
    assert_eq!(rejection.limit_kind, QueueLimitKind::Requests);
    assert_eq!(rejection.current, 2);
    assert_eq!(rejection.limit, 2);
    assert_eq!(queue.class_stats(0).raw_isl_tokens, 108);
    assert_eq!(queue.class_stats(0).cached_tokens, 104);
}

#[test]
fn earliest_due_request_precedes_fcfs_arrival_within_a_class() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let now = Instant::now();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(now + std::time::Duration::from_secs(20)),
                class_due_at: None,
                arrival_offset_secs: 0.0,
                lane_key: None,
            },
            1,
            0.0,
            0,
            QueueLane::Shared,
            "later-due",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(now + std::time::Duration::from_secs(10)),
                class_due_at: None,
                arrival_offset_secs: 1.0,
                lane_key: None,
            },
            1,
            0.0,
            0,
            QueueLane::Shared,
            "earlier-due",
        )
        .unwrap();

    assert_eq!(
        queue.next_due_at(0),
        Some(now + std::time::Duration::from_secs(10))
    );
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "earlier-due"
    );
    assert_eq!(queue.due_entry_count(0), 1);
    assert_eq!(
        queue.next_due_at(0),
        Some(now + std::time::Duration::from_secs(20))
    );
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "later-due"
    );
    assert!((queue.due_entry_count(0) == 0));
    assert_eq!(queue.next_due_at(0), None);
}

#[test]
fn due_time_order_preserves_sub_millisecond_precision() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let due_at = Instant::now() + std::time::Duration::from_secs(10);
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(due_at + std::time::Duration::from_nanos(1)),
                class_due_at: None,
                arrival_offset_secs: 0.0,
                lane_key: None,
            },
            1,
            0.0,
            0,
            lane_of(1),
            "later-due-better-fcfs",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(due_at),
                class_due_at: None,
                arrival_offset_secs: 1.0,
                lane_key: None,
            },
            1,
            0.0,
            0,
            lane_of(2),
            "earlier-due-worse-fcfs",
        )
        .unwrap();

    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "earlier-due-worse-fcfs"
    );
}

#[test]
fn deadline_count_restores_the_no_deadline_fast_path_after_removal() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let due_at = Instant::now() + std::time::Duration::from_secs(10);
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 0.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "no-deadline",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(due_at),
                class_due_at: None,
                arrival_offset_secs: 1.0,
                lane_key: None,
            },
            2,
            0.0,
            0,
            lane_of(1),
            "deadline",
        )
        .unwrap();

    assert_eq!(queue.next_due_at(0), Some(due_at));
    queue.retain(|payload| *payload != "deadline");
    assert!((queue.due_entry_count(0) == 0));
    assert_eq!(queue.next_due_at(0), None);
    assert_eq!(queue.pending_count(), 1);
}

#[test]
fn deadline_timer_removes_an_expired_blocked_worker_lane() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let worker = WorkerWithDpRank::new(7, 0);
    let due_at = Instant::now() + std::time::Duration::from_secs(1);
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(due_at),
                class_due_at: None,
                arrival_offset_secs: 0.0,
                lane_key: None,
            },
            1,
            0.0,
            0,
            lane_of(worker.worker_id),
            "blocked",
        )
        .unwrap();

    assert!(queue.pop_next(|_, _, _| false).is_none());
    assert!(
        queue
            .class_lane_block_reason(0, lane_of(worker.worker_id))
            .is_some()
    );
    assert_eq!(queue.next_due_at(0), Some(due_at));

    let expired = queue.take_expired(0, due_at);
    assert_eq!(expired.len(), 1);
    assert_eq!(
        expired.into_iter().next().unwrap().into_payload(),
        "blocked"
    );
    assert_eq!(queue.next_due_at(0), None);
    assert!(
        queue
            .pop_next(|_, _, _| -> bool { panic!("expired lane revisited") })
            .is_none()
    );
}

#[test]
fn strict_priority_precedes_earlier_due_request_within_a_class() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let now = Instant::now();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(now + std::time::Duration::from_secs(20)),
                class_due_at: None,
                arrival_offset_secs: 0.0,
                lane_key: None,
            },
            1,
            0.0,
            10,
            QueueLane::Shared,
            "high-priority-later-due",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(now + std::time::Duration::from_secs(10)),
                class_due_at: None,
                arrival_offset_secs: 1.0,
                lane_key: None,
            },
            1,
            0.0,
            0,
            QueueLane::Shared,
            "low-priority-earlier-due",
        )
        .unwrap();

    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "high-priority-later-due"
    );
}

fn edf_profile() -> PolicyProfile {
    // A configured class without queue_policy is EDF.
    profile(
        r#"
policy_classes:
  - name: edf
    quantum: 10
"#,
    )
}

fn enqueue_due(
    queue: &mut PolicyQueue<&'static str>,
    due_at: Option<Instant>,
    class_due_at: Option<Instant>,
    lane: QueueLane,
    payload: &'static str,
) {
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at,
                class_due_at,
                arrival_offset_secs: 0.0,
                lane_key: None,
            },
            2,
            0.0,
            0,
            lane,
            payload,
        )
        .unwrap();
}

fn drain_payloads<V: Into<LaneCheck>>(
    queue: &mut PolicyQueue<&'static str>,
    is_dispatchable: impl FnMut(usize, &PolicyClassConfig, &&'static str) -> V + Copy,
) -> Vec<&'static str> {
    std::iter::from_fn(|| queue.pop_next(is_dispatchable))
        .map(PolicyQueueEntry::into_payload)
        .collect()
}

#[test]
fn edf_orders_by_due_time_then_enqueue_order_and_ignores_priority_modifiers() {
    let mut queue = PolicyQueue::new(&edf_profile());
    assert_eq!(queue.class_config(0).queue_policy, RouterQueuePolicy::Edf);
    let now = Instant::now();
    let at = |secs| Some(now + std::time::Duration::from_secs(secs));
    for (payload, due_at, class_due_at, priority_jump, strict_priority) in [
        ("absent-first", None, None, 0.0, 0),
        ("class-30-boosted", None, at(30), 100.0, 9),
        ("explicit-20-over-class-5", at(20), at(5), 0.0, 0),
        ("class-10", None, at(10), 0.0, 0),
        ("absent-boosted", None, None, 1_000.0, 9),
        ("class-10-tie", None, at(10), 0.0, 0),
        ("explicit-40-over-class-1", at(40), at(1), 0.0, 0),
    ] {
        queue
            .enqueue_with_due_at(
                QueueMetadata {
                    class_index: 0,
                    snapshot: QueueSnapshot::new(1, 0),
                    due_at,
                    class_due_at,
                    arrival_offset_secs: 0.0,
                    lane_key: None,
                },
                1,
                priority_jump,
                strict_priority,
                QueueLane::Shared,
                payload,
            )
            .unwrap();
    }
    // Only the explicit deadlines are indexed for expiry.
    assert_eq!(queue.due_entry_count(0), 2);
    assert_eq!(queue.next_due_at(0), at(20));

    // Earliest due first; an explicit deadline replaces the class due time, earlier or
    // later. Equal and absent due times keep enqueue order.
    assert_eq!(
        drain_payloads(&mut queue, |_, _, _| true),
        [
            "class-10",
            "class-10-tie",
            "explicit-20-over-class-5",
            "class-30-boosted",
            "explicit-40-over-class-1",
            "absent-first",
            "absent-boosted",
        ]
    );
    assert!((queue.due_entry_count(0) == 0));
}

#[test]
fn edf_lanes_compare_due_times_and_a_blocked_lane_holds_only_itself() {
    let mut queue = PolicyQueue::new(&edf_profile());
    let now = Instant::now();
    let at = |secs| Some(now + std::time::Duration::from_secs(secs));
    let worker = |id: u64| lane_of(id);
    for (payload, class_due_at, placement) in [
        ("shared-30", at(30), QueueLane::Shared),
        ("full-worker-10", at(10), worker(1)),
        ("idle-worker-20", at(20), worker(2)),
        ("shared-25", at(25), QueueLane::Shared),
        ("full-worker-15", at(15), worker(1)),
    ] {
        enqueue_due(&mut queue, None, class_due_at, placement, payload);
    }

    // The earliest head is pinned to a full worker. It holds only its own lane: the other
    // lane and the shared heap still dispatch, in due-time order.
    assert_eq!(
        drain_payloads(&mut queue, |_, _, payload| !payload
            .starts_with("full-worker")),
        ["idle-worker-20", "shared-25", "shared-30"]
    );
    assert_eq!(queue.pending_count(), 2);
    queue.recheck_blocked_lanes(RecheckScope::All);
    assert_eq!(
        drain_payloads(&mut queue, |_, _, _| true),
        ["full-worker-10", "full-worker-15"]
    );
}

#[test]
fn capacity_recheck_skips_gate_held_lanes() {
    let mut queue = PolicyQueue::new(&edf_profile());
    enqueue_due(&mut queue, None, None, lane_of(1), "gate-held");
    enqueue_due(&mut queue, None, None, lane_of(2), "capacity-blocked");
    enqueue_due(&mut queue, None, None, QueueLane::Shared, "shared");
    let asked = std::cell::RefCell::new(Vec::new());
    let predicate = |_: usize, _: &PolicyClassConfig, payload: &&'static str| {
        asked.borrow_mut().push(*payload);
        match *payload {
            "gate-held" => LaneCheck::Blocked(BlockReason::Gate),
            "capacity-blocked" => LaneCheck::Blocked(BlockReason::Capacity),
            _ => LaneCheck::Dispatch,
        }
    };

    // Each blocked lane keeps the reason its head was refused for.
    assert_eq!(
        queue
            .pop_next(predicate)
            .map(PolicyQueueEntry::into_payload),
        Some("shared")
    );
    assert_eq!(
        queue.class_lane_block_reason(0, lane_of(1)).unwrap(),
        BlockReason::Gate
    );
    assert_eq!(
        queue.class_lane_block_reason(0, lane_of(2)).unwrap(),
        BlockReason::Capacity
    );
    asked.borrow_mut().clear();
    assert!(queue.pop_next(predicate).is_none());
    assert!(
        asked.borrow().is_empty(),
        "blocked lanes are not re-asked per pop"
    );

    // A capacity change asks only the lane blocked on capacity.
    queue.recheck_blocked_lanes(RecheckScope::Capacity);
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(asked.borrow().as_slice(), ["capacity-blocked"]);
    asked.borrow_mut().clear();

    // The gate's own wake asks every blocked lane.
    queue.recheck_blocked_lanes(RecheckScope::All);
    assert!(queue.pop_next(predicate).is_none());
    let mut all = asked.borrow().clone();
    all.sort_unstable();
    assert_eq!(all, ["capacity-blocked", "gate-held"]);

    // A narrower request never shrinks a wider pending scope.
    queue.recheck_blocked_lanes(RecheckScope::All);
    queue.recheck_blocked_lanes(RecheckScope::Capacity);
    assert_eq!(queue.class_pending_recheck(0), Some(RecheckScope::All));
}

#[test]
fn shared_group_stand_in_is_handed_on_when_it_is_cancelled() {
    // Three keyed lanes refused as SharedCapacity; the pinned lane keeps its own reason.
    let mut queue = PolicyQueue::new(&edf_profile());
    for (id, payload) in [(1, "s1"), (2, "s2"), (3, "s3")] {
        enqueue_due(&mut queue, None, None, lane_of(id), payload);
    }
    enqueue_due(
        &mut queue,
        None,
        None,
        QueueLane::Exact(WorkerWithDpRank::new(9, 0)),
        "pinned",
    );
    let room = std::cell::Cell::new(false);
    let predicate = |_: usize, _: &PolicyClassConfig, payload: &&'static str| {
        if *payload == "pinned" {
            LaneCheck::Blocked(BlockReason::Capacity)
        } else if room.get() {
            LaneCheck::Dispatch
        } else {
            LaneCheck::Blocked(BlockReason::SharedCapacity)
        }
    };
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(queue.class_blocked_lane_count(0), 4);

    // Room appears: the recheck promotes lane 1, it dispatches, and lane 2 becomes the
    // stand-in, a candidate waiting for the next pass.
    room.set(true);
    queue.recheck_blocked_lanes(RecheckScope::Capacity);
    assert_eq!(queue.pop_next(predicate).unwrap().into_payload(), "s1");
    assert_eq!(queue.class_blocked_lane_count(0), 2);

    // The stand-in is cancelled between passes. Without a handoff lane 3 would wait for the
    // next capacity change; with it the next pass promotes and dispatches lane 3.
    let (removed, _) = queue.take_if_in_class(0, |payload| *payload == "s2");
    assert_eq!(removed.len(), 1);
    assert_eq!(queue.pop_next(predicate).unwrap().into_payload(), "s3");
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(
        queue.class_lane_block_reason(0, QueueLane::Exact(WorkerWithDpRank::new(9, 0))),
        Some(BlockReason::Capacity)
    );
}

#[test]
fn shared_group_stand_in_expiry_hands_the_role_on() {
    // A class whose budget expires lane roots through `expiring_roots`, so an expired
    // stand-in leaves by `pop_expiring_root`.
    let mut queue = PolicyQueue::with_class_budget_expiry(&controlled_delay_profile());
    let now = Instant::now();
    let after = |secs| now + std::time::Duration::from_secs(secs);
    // EDF puts lane 1 first, then lane 2, then lane 3.
    enqueue_due(&mut queue, None, Some(after(1)), lane_of(1), "s1");
    enqueue_due(&mut queue, None, Some(after(30)), lane_of(2), "s2");
    enqueue_due(&mut queue, None, Some(after(60)), lane_of(3), "s3");
    let room = std::cell::Cell::new(false);
    let predicate = |_: usize, _: &PolicyClassConfig, _: &&'static str| {
        if room.get() {
            LaneCheck::Dispatch
        } else {
            LaneCheck::Blocked(BlockReason::SharedCapacity)
        }
    };
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(queue.class_blocked_lane_count(0), 3);

    // Room appears: lane 1 dispatches and lane 2 becomes the stand-in.
    room.set(true);
    queue.recheck_blocked_lanes(RecheckScope::Capacity);
    assert_eq!(queue.pop_next(predicate).unwrap().into_payload(), "s1");
    assert_eq!(queue.class_blocked_lane_count(0), 1);

    // The stand-in's only entry expires before the next pass; the role moves to lane 3.
    let expired = queue.take_expired(0, after(31));
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].payload(), &"s2");
    assert_eq!(queue.pop_next(predicate).unwrap().into_payload(), "s3");
    assert!(queue.pop_next(predicate).is_none());
}

#[test]
fn shared_capacity_lanes_are_probed_once_per_capacity_recheck() {
    let mut queue = PolicyQueue::new(&edf_profile());
    for id in 1..=3 {
        enqueue_due(&mut queue, None, None, lane_of(id), "session");
    }
    enqueue_due(
        &mut queue,
        None,
        None,
        QueueLane::Exact(WorkerWithDpRank::new(9, 0)),
        "pinned",
    );
    let asks = std::cell::Cell::new(0usize);
    let room = std::cell::Cell::new(false);
    let predicate = |_: usize, _: &PolicyClassConfig, payload: &&'static str| {
        asks.set(asks.get() + 1);
        match (*payload, room.get()) {
            ("session", false) => LaneCheck::Blocked(BlockReason::SharedCapacity),
            ("session", true) => LaneCheck::Dispatch,
            _ => LaneCheck::Blocked(BlockReason::Capacity),
        }
    };

    // Every lane is refused once; the three sessions join the shared group.
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(asks.get(), 4);
    assert_eq!(
        queue.class_lane_block_reason(0, lane_of(1)),
        Some(BlockReason::SharedCapacity)
    );
    assert_eq!(queue.class_blocked_lane_count(0), 4);

    // A capacity change probes one shared lane and the pinned lane: two asks, not four.
    asks.set(0);
    queue.recheck_blocked_lanes(RecheckScope::Capacity);
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(asks.get(), 2);
    assert_eq!(queue.class_blocked_lane_count(0), 4);

    // Once the probe may go, only the group's best head is promoted: the probe, the pinned
    // lane, and the promoted head at dispatch are asked, the other shared lanes stay where
    // they are without a visit.
    room.set(true);
    asks.set(0);
    queue.recheck_blocked_lanes(RecheckScope::Capacity);
    assert_eq!(queue.pop_next(predicate).unwrap().into_payload(), "session");
    assert_eq!(asks.get(), 3);
    // The dispatch hands the stand-in to the next best head, so more room is used up in
    // later passes at one ask each, without another capacity change: stand-in, then the
    // pinned lane is not asked again as no recheck was filed.
    assert_eq!(queue.class_blocked_lane_count(0), 2);
    asks.set(0);
    assert_eq!(queue.pop_next(predicate).unwrap().into_payload(), "session");
    assert_eq!(asks.get(), 1);
    assert_eq!(queue.class_blocked_lane_count(0), 1);
    // The room is gone: the last stand-in is refused for the shared reason and rejoins the
    // group, which now waits for the next capacity change.
    room.set(false);
    asks.set(0);
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(asks.get(), 1);
    assert_eq!(queue.class_blocked_lane_count(0), 2);
    asks.set(0);
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(asks.get(), 0);
    room.set(true);
    queue.recheck_blocked_lanes(RecheckScope::Capacity);
    assert_eq!(queue.pop_next(predicate).unwrap().into_payload(), "session");
    assert!(queue.pop_next(predicate).is_none());
    assert_eq!(queue.class_blocked_lane_count(0), 1);
    assert_eq!(
        queue.class_lane_block_reason(0, QueueLane::Exact(WorkerWithDpRank::new(9, 0))),
        Some(BlockReason::Capacity)
    );
}

#[test]
fn edf_class_due_times_order_but_never_expire() {
    let mut queue = PolicyQueue::new(&edf_profile());
    let now = Instant::now();
    let after = |secs| now + std::time::Duration::from_secs(secs);
    enqueue_due(
        &mut queue,
        None,
        Some(after(1)),
        QueueLane::Shared,
        "class-1",
    );
    enqueue_due(
        &mut queue,
        Some(after(5)),
        Some(after(2)),
        QueueLane::Shared,
        "explicit-5",
    );
    assert_eq!(queue.next_due_at(0), Some(after(5)));

    // Long after both due times, expiry takes only the explicit deadline.
    let expired = queue
        .take_expired(0, after(60))
        .into_iter()
        .map(PolicyQueueEntry::into_payload)
        .collect::<Vec<_>>();
    assert_eq!(expired, ["explicit-5"]);
    assert!((queue.due_entry_count(0) == 0));
    assert_eq!(queue.next_due_at(0), None);
    assert_eq!(drain_payloads(&mut queue, |_, _, _| true), ["class-1"]);
}

#[tokio::test(start_paused = true)]
async fn due_time_order_survives_the_expiry_index_emptying() {
    let mut queue = PolicyQueue::new(&edf_profile());
    let start = Instant::now();
    let after = |secs| start + std::time::Duration::from_secs(secs);
    enqueue_due(
        &mut queue,
        None,
        Some(after(10)),
        QueueLane::Shared,
        "class-10",
    );
    // The expiry index is empty when this deadline arrives 20 s later; due-time keys
    // keep one origin regardless.
    tokio::time::advance(std::time::Duration::from_secs(20)).await;
    enqueue_due(
        &mut queue,
        Some(after(120)),
        None,
        QueueLane::Shared,
        "explicit-120",
    );
    // This class due time has already passed, and still orders after the earlier one.
    enqueue_due(
        &mut queue,
        None,
        Some(after(12)),
        QueueLane::Shared,
        "class-12",
    );
    assert_eq!(
        drain_payloads(&mut queue, |_, _, _| true),
        ["class-10", "class-12", "explicit-120"]
    );
}

#[tokio::test(start_paused = true)]
async fn due_times_before_the_queue_origin_clamp_to_it() {
    // A due time can precede the queue's construction, for example one counted from an
    // ingress passed in from an earlier clock reading. Such due times all key as the
    // origin, so they tie there and keep enqueue order.
    let early = Instant::now();
    tokio::time::advance(std::time::Duration::from_secs(60)).await;
    let mut queue = PolicyQueue::new(&edf_profile());
    let at = |secs| early + std::time::Duration::from_secs(secs);
    enqueue_due(&mut queue, None, Some(at(3)), QueueLane::Shared, "class-3");
    enqueue_due(
        &mut queue,
        None,
        Some(at(120)),
        QueueLane::Shared,
        "class-120",
    );
    enqueue_due(&mut queue, None, Some(at(1)), QueueLane::Shared, "class-1");
    enqueue_due(
        &mut queue,
        Some(at(2)),
        None,
        QueueLane::Shared,
        "explicit-2",
    );

    // The explicit deadline's timer wakes at the origin.
    assert_eq!(queue.next_due_at(0), Some(queue.deadline_origin()));
    assert_eq!(
        drain_payloads(&mut queue, |_, _, _| true),
        ["class-3", "class-1", "explicit-2", "class-120"]
    );
    assert!((queue.due_entry_count(0) == 0));
}

#[test]
fn fcfs_and_wspt_do_not_order_by_class_due_times() {
    let now = Instant::now();
    let at = |secs| Some(now + std::time::Duration::from_secs(secs));
    let queue_for = |policy: &str| {
        PolicyQueue::new(&profile(&format!(
            "policy_classes:\n  - name: c\n    queue_policy: {policy}\n    quantum: 100\n"
        )))
    };
    let metadata = |tokens, due_at, class_due_at, arrival_offset_secs| QueueMetadata {
        class_index: 0,
        snapshot: QueueSnapshot::new(tokens, 0),
        due_at,
        class_due_at,
        arrival_offset_secs,
        lane_key: None,
    };

    // FCFS keeps arrival order although the later arrival's class due time is earlier.
    let mut fcfs = queue_for("fcfs");
    for (payload, arrival_offset_secs, class_due_at) in
        [("first", 0.0, at(30)), ("second", 1.0, at(1))]
    {
        fcfs.enqueue_with_due_at(
            metadata(1, None, class_due_at, arrival_offset_secs),
            1,
            0.0,
            0,
            QueueLane::Shared,
            payload,
        )
        .unwrap();
    }
    assert!((fcfs.due_entry_count(0) == 0));
    assert_eq!(
        drain_payloads(&mut fcfs, |_, _, _| true),
        ["first", "second"]
    );

    // WSPT dispatches the shorter request first although the longer one's class due time
    // is earlier. An explicit deadline still precedes the score.
    let mut wspt = queue_for("wspt");
    for (payload, tokens, due_at, class_due_at) in [
        ("long", 50, None, at(1)),
        ("short", 1, None, at(30)),
        ("long-explicit", 50, at(60), None),
    ] {
        wspt.enqueue_with_due_at(
            metadata(tokens, due_at, class_due_at, 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            payload,
        )
        .unwrap();
    }
    assert_eq!(wspt.due_entry_count(0), 1);
    assert_eq!(
        drain_payloads(&mut wspt, |_, _, _| true),
        ["long-explicit", "short", "long"]
    );
}

/// An EDF class that enables controlled delay. A queue built with `PolicyQueue::new` still
/// expires only its explicit deadlines.
fn controlled_delay_profile() -> PolicyProfile {
    profile(
        r#"
policy_classes:
  - name: edf
    quantum: 1000
    controlled_delay: true
"#,
    )
}

fn timed(cost: usize, due_at: Option<Instant>, class_due_at: Option<Instant>) -> QueueMetadata {
    QueueMetadata {
        class_index: 0,
        snapshot: QueueSnapshot::new(cost, 0),
        due_at,
        class_due_at,
        arrival_offset_secs: 0.0,
        lane_key: None,
    }
}

fn push_to<T>(queue: &mut PolicyQueue<T>, metadata: QueueMetadata, lane: QueueLane, payload: T) {
    assert!(
        queue
            .enqueue_with_due_at(metadata, 1, 0.0, 0, lane, payload)
            .is_ok()
    );
}

fn push<T>(queue: &mut PolicyQueue<T>, metadata: QueueMetadata, payload: T) {
    push_to(queue, metadata, QueueLane::Shared, payload);
}

/// Every entry of the class that expires by `now`, earliest first with head expiry.
fn take_expired_payloads<T>(
    queue: &mut PolicyQueue<T>,
    class_index: usize,
    now: Instant,
) -> Vec<T> {
    queue
        .take_expired(class_index, now)
        .into_iter()
        .map(PolicyQueueEntry::into_payload)
        .collect()
}

fn drain_all<T>(queue: &mut PolicyQueue<T>) -> Vec<T> {
    std::iter::from_fn(|| queue.pop_next(|_, _, _| true))
        .map(PolicyQueueEntry::into_payload)
        .collect()
}

#[test]
fn class_budgets_expire_per_class_and_only_in_online_queues() {
    let profile = profile(
        r#"
default_policy_class: disabled
policy_classes:
  - name: enabled
    quantum: 1
    slo_ttft_ms: 100
    controlled_delay: true
  - name: disabled
    quantum: 1
    slo_ttft_ms: 100
"#,
    );
    let start = Instant::now();
    let budget = start + std::time::Duration::from_millis(100);
    let explicit = start + std::time::Duration::from_millis(200);
    let metadata = |class_index, due_at| QueueMetadata {
        class_index,
        ..timed(1, due_at, Some(budget))
    };
    let mut online = PolicyQueue::with_class_budget_expiry(&profile);
    let mut offline = PolicyQueue::new(&profile);
    // Online, only the opted-in class enforces its budget; offline replay enforces none.
    assert_eq!(online.enforced_due_at(&metadata(0, None)), Some(budget));
    assert_eq!(online.enforced_due_at(&metadata(1, None)), None);
    assert_eq!(offline.enforced_due_at(&metadata(0, None)), None);
    // An explicit deadline wins in every class, even when later than the budget.
    for class_index in [0, 1] {
        for queue in [&online, &offline] {
            assert_eq!(
                queue.enforced_due_at(&metadata(class_index, Some(explicit))),
                Some(explicit)
            );
        }
    }

    // The opted-in class finds every due time at its lane roots. Elsewhere the budgeted
    // root only orders, and the explicit deadline behind it stays in the class index.
    for queue in [&mut online, &mut offline] {
        for class_index in [0, 1] {
            push(queue, metadata(class_index, None), class_index);
            push(queue, metadata(class_index, Some(explicit)), class_index);
        }
        queue.assert_indexes_match_entries();
    }
    assert_eq!(online.next_due_at(0), Some(budget));
    assert_eq!(online.next_due_at(1), Some(explicit));
    assert_eq!(offline.next_due_at(0), Some(explicit));
    assert_eq!(offline.next_due_at(1), Some(explicit));
    let after_budget = budget + std::time::Duration::from_millis(1);
    assert_eq!(take_expired_payloads(&mut online, 0, after_budget), [0]);
    assert!(take_expired_payloads(&mut online, 1, after_budget).is_empty());
    assert!(take_expired_payloads(&mut offline, 0, after_budget).is_empty());
    let after_explicit = explicit + std::time::Duration::from_millis(1);
    for class_index in [0, 1] {
        for queue in [&mut online, &mut offline] {
            assert_eq!(
                take_expired_payloads(queue, class_index, after_explicit),
                [class_index]
            );
            queue.assert_indexes_match_entries();
        }
    }
    // What remains is each budgeted entry that nothing expires.
    assert_eq!(drain_all(&mut online), [1]);
    assert_eq!(drain_all(&mut offline).len(), 2);
}

#[test]
fn class_budget_expiry_takes_the_overdue_edf_prefix_without_spending_credit() {
    let mut queue = PolicyQueue::with_class_budget_expiry(&controlled_delay_profile());
    let now = Instant::now();
    let at = |millis| Some(now + std::time::Duration::from_millis(millis));
    push(&mut queue, timed(400, None, at(1)), "dispatched");
    push(&mut queue, timed(1, None, at(100)), "class-100");
    push(&mut queue, timed(1, at(150), at(50)), "explicit-150");
    push(&mut queue, timed(1, None, at(200)), "class-200");
    push(&mut queue, timed(1, None, None), "unbounded");
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "dispatched"
    );
    assert_eq!(queue.class_deficit(0), 600);
    // No deadline is indexed: a blocked root still carries its lane's earliest due time.
    assert!((queue.due_entry_count(0) == 0));
    assert!(queue.pop_next(|_, _, _| false).is_none());
    assert_eq!(queue.next_due_at(0), at(100));

    // A due time equal to now expires. The explicit deadline outranks its earlier class
    // due time and leaves the queue once, in order.
    assert_eq!(
        take_expired_payloads(&mut queue, 0, now + std::time::Duration::from_millis(150)),
        ["class-100", "explicit-150"]
    );
    assert_eq!(queue.pending_count(), 2);
    assert_eq!(queue.class_deficit(0), 600);
    assert_eq!(queue.next_due_at(0), at(200));
    queue.assert_indexes_match_entries();
    assert_eq!(
        take_expired_payloads(&mut queue, 0, now + std::time::Duration::from_secs(3600)),
        ["class-200"]
    );
    assert_eq!(queue.next_due_at(0), None);
    assert_eq!(drain_all(&mut queue), ["unbounded"]);
}

#[test]
fn session_lane_holds_only_its_session_until_rechecked() {
    let mut queue = PolicyQueue::with_class_budget_expiry(&controlled_delay_profile());
    let now = Instant::now();
    let at = |millis| Some(now + std::time::Duration::from_millis(millis));
    let session = |id: &str| QueueLane::Keyed(LaneKey::new(id));
    for (payload, millis, placement) in [
        ("a-10", 10, session("a")),
        ("a-40", 40, session("a")),
        ("b-20", 20, session("b")),
        ("shared-30", 30, QueueLane::Shared),
    ] {
        push_to(&mut queue, timed(1, None, at(millis)), placement, payload);
    }
    queue.assert_indexes_match_entries();

    // Session a's head is held: the gate says no. The best other head dispatches, in
    // class order, and a's second turn stays behind its held first turn.
    let hold_a = |_: usize, _: &PolicyClassConfig, payload: &&str| !payload.starts_with("a-");
    assert_eq!(queue.pop_next(hold_a).unwrap().into_payload(), "b-20");
    assert!(
        queue
            .class_lane_block_reason(0, QueueLane::Keyed(LaneKey::new("a")))
            .is_some()
    );
    assert_eq!(queue.pop_next(hold_a).unwrap().into_payload(), "shared-30");
    assert!(queue.pop_next(hold_a).is_none());
    assert_eq!(queue.pending_count(), 2);
    // A blocked lane is not re-asked until a recheck.
    assert!(queue.pop_next(|_, _, _| true).is_none());

    // A gate recheck unblocks every lane; the held head then dispatches in order.
    queue.recheck_blocked_lanes(RecheckScope::All);
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "a-10"
    );
    assert!((queue.class_blocked_lane_count(0) == 0));
    queue.assert_indexes_match_entries();

    // A lane blocked again waits for the next recheck, then dispatches in order.
    assert!(queue.pop_next(hold_a).is_none());
    queue.recheck_blocked_lanes(RecheckScope::All);
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "a-40"
    );
    assert_eq!(queue.pending_count(), 0);
}

#[test]
fn class_budget_expiry_reaches_every_lane_root_including_blocked_lanes() {
    let mut queue = PolicyQueue::with_class_budget_expiry(&controlled_delay_profile());
    let now = Instant::now();
    let at = |millis| Some(now + std::time::Duration::from_millis(millis));
    let exact = |id| QueueLane::Exact(WorkerWithDpRank::new(id, 0));
    let ranked = QueueLane::Affinity(WorkerAffinityTarget::new(1, Some(0)));
    let worker_only = QueueLane::Affinity(WorkerAffinityTarget::new(1, None));
    for (payload, millis, lane) in [
        ("shared-30", 30, QueueLane::Shared),
        ("full-10", 10, exact(1)),
        ("full-40", 40, exact(1)),
        ("idle-20", 20, exact(2)),
        ("shared-50", 50, QueueLane::Shared),
        ("soft-15", 15, ranked),
        ("soft-45", 45, ranked),
        ("worker-17", 17, worker_only),
        ("worker-60", 60, worker_only),
    ] {
        queue
            .enqueue_with_due_at(timed(1, None, at(millis)), 1, 0.0, 0, lane, payload)
            .unwrap();
    }
    let readiness = |_: usize, _: &PolicyClassConfig, payload: &&str| {
        if payload.starts_with("soft") {
            LaneCheck::Blocked(BlockReason::Fallback)
        } else if payload.starts_with("full") || payload.starts_with("worker") {
            LaneCheck::Blocked(BlockReason::Capacity)
        } else {
            LaneCheck::Dispatch
        }
    };
    let (entry, charge) = queue.extract_next(readiness).unwrap();
    queue.settle(charge, true);
    assert_eq!(entry.into_payload(), "idle-20");
    assert_eq!(queue.expiring_root_count(0), 4);
    assert_eq!(queue.class_lane_counts(0), [3, 3, 1, 0]);
    let deficit = queue.class_deficit(0);
    assert_eq!(queue.next_due_at(0), at(10));
    queue.assert_indexes_match_entries();

    assert_eq!(
        take_expired_payloads(&mut queue, 0, now + std::time::Duration::from_millis(35)),
        ["full-10", "soft-15", "worker-17", "shared-30"]
    );
    assert_eq!(queue.pending_count(), 4);
    assert_eq!(queue.class_deficit(0), deficit);
    assert_eq!(queue.class_lane_counts(0), [3, 0, 0, 3]);
    assert_eq!(queue.expiring_root_count(0), 4);
    assert_eq!(queue.next_due_at(0), at(40));
    queue.assert_indexes_match_entries();

    queue.retain(|payload| *payload != "worker-60");
    assert_eq!(queue.class_lane_counts(0)[0], 2);
    assert_eq!(queue.pending_count(), 3);
    assert_eq!(queue.expiring_root_count(0), 3);
    assert!(queue.has_affinity_backlog());
    queue.assert_indexes_match_entries();
    assert_eq!(drain_all(&mut queue), ["full-40", "soft-45", "shared-50"]);
    assert!(!queue.has_affinity_backlog());
    assert!(queue.expiring_root_count(0) == 0);
    assert_eq!(queue.class_lane_counts(0), [0, 0, 0, 0]);
    assert_eq!(queue.class_stats(0), PolicyQueueStats::default());
}

/// A removal that takes a blocked lane's root, by explicit-deadline expiry or cancellation,
/// unblocks the lane, as head expiry does, so the next dispatch checks the lane's next head.
/// A removal behind the root keeps the lane blocked.
#[test]
fn removing_a_blocked_lanes_root_unblocks_the_lane() {
    let now = Instant::now();
    let at = |millis| Some(now + std::time::Duration::from_millis(millis));
    let worker = WorkerWithDpRank::new(1, 0);
    let lane = lane_of(worker.worker_id);
    let full = |_: usize, _: &PolicyClassConfig, payload: &&str| !payload.starts_with("full");
    for policy in [
        RouterQueuePolicy::Edf,
        RouterQueuePolicy::Fcfs,
        RouterQueuePolicy::Lcfs,
        RouterQueuePolicy::Wspt,
    ] {
        let profile = PolicyProfile::synthetic(None, policy);
        for (constructor, mut queue) in [
            ("new", PolicyQueue::new(&profile)),
            (
                "with_class_budget_expiry",
                PolicyQueue::with_class_budget_expiry(&profile),
            ),
        ] {
            let case = format!("{policy} {constructor}");
            for (payload, millis) in [
                ("full-10", 10),
                ("full-20", 20),
                ("full-30", 30),
                ("full-40", 40),
            ] {
                push_to(&mut queue, timed(1, at(millis), None), lane, payload);
            }
            push(&mut queue, timed(1, None, None), "shared");
            // Worker 1 is full: its lane blocks at its root.
            assert_eq!(
                queue.pop_next(full).unwrap().into_payload(),
                "shared",
                "{case}"
            );
            assert!(
                queue
                    .class_lane_block_reason(0, lane_of(worker.worker_id))
                    .is_some(),
                "{case}"
            );

            // Cancelling behind the root leaves the lane blocked.
            queue.retain(|payload| *payload != "full-30");
            assert!(
                queue
                    .class_lane_block_reason(0, lane_of(worker.worker_id))
                    .is_some(),
                "{case}"
            );

            // Expiry takes the root: the next dispatch checks the lane's next head.
            let expiry = now + std::time::Duration::from_millis(10);
            assert_eq!(
                take_expired_payloads(&mut queue, 0, expiry),
                ["full-10"],
                "{case}"
            );
            assert!(
                queue
                    .class_lane_block_reason(0, lane_of(worker.worker_id))
                    .is_none(),
                "{case}"
            );
            queue.assert_indexes_match_entries();
            assert!(queue.pop_next(full).is_none(), "{case}");
            assert!(
                queue
                    .class_lane_block_reason(0, lane_of(worker.worker_id))
                    .is_some(),
                "{case}"
            );

            // So does cancelling the root.
            queue.retain(|payload| *payload != "full-20");
            assert!(
                queue
                    .class_lane_block_reason(0, lane_of(worker.worker_id))
                    .is_none(),
                "{case}"
            );
            queue.assert_indexes_match_entries();
            assert_eq!(drain_all(&mut queue), ["full-40"], "{case}");
        }
    }
}

#[test]
fn cancellation_keeps_the_lane_root_index_exact() {
    let mut queue = PolicyQueue::with_class_budget_expiry(&controlled_delay_profile());
    let now = Instant::now();
    let at = |millis| Some(now + std::time::Duration::from_millis(millis));
    let worker = lane_of(3);
    push_to(&mut queue, timed(1, None, at(10)), worker, "pinned-10");
    push_to(&mut queue, timed(1, None, at(20)), worker, "pinned-20");
    push(&mut queue, timed(1, None, at(15)), "shared-15");
    assert_eq!(queue.next_due_at(0), at(10));
    queue.retain(|payload| *payload != "pinned-10");
    assert_eq!(queue.next_due_at(0), at(15));
    queue.assert_indexes_match_entries();
    queue.retain(|payload| *payload != "shared-15");
    assert_eq!(queue.next_due_at(0), at(20));
    queue.assert_indexes_match_entries();
    queue.retain(|_| false);
    assert_eq!(queue.next_due_at(0), None);
    queue.assert_indexes_match_entries();
}

#[test]
fn class_budget_expiry_leaves_other_orders_to_explicit_deadlines() {
    let now = Instant::now();
    let at = |millis| Some(now + std::time::Duration::from_millis(millis));
    for policy in [
        RouterQueuePolicy::Fcfs,
        RouterQueuePolicy::Wspt,
        RouterQueuePolicy::Lcfs,
    ] {
        let mut queue =
            PolicyQueue::with_class_budget_expiry(&PolicyProfile::synthetic(None, policy));
        assert_eq!(queue.enforced_due_at(&timed(1, None, at(1))), None);
        // A strict-priority root hides the explicit deadline behind it.
        assert!(
            queue
                .enqueue_with_due_at(
                    timed(1, None, at(1)),
                    1,
                    0.0,
                    1,
                    QueueLane::Shared,
                    "strict-head"
                )
                .is_ok()
        );
        push(&mut queue, timed(1, at(100), None), "hidden-explicit");
        push(&mut queue, timed(1, None, at(2)), "class-due");
        assert_eq!(queue.next_due_at(0), at(100), "{policy}");
        assert_eq!(
            take_expired_payloads(&mut queue, 0, now + std::time::Duration::from_secs(60)),
            ["hidden-explicit"],
            "{policy}"
        );
        assert_eq!(queue.next_due_at(0), None, "{policy}");
        queue.assert_indexes_match_entries();
        assert_eq!(queue.pending_count(), 2, "{policy}");
    }

    let edf = PolicyQueue::<()>::with_class_budget_expiry(&controlled_delay_profile());
    assert_eq!(edf.enforced_due_at(&timed(1, None, at(1))), at(1));
    assert_eq!(edf.enforced_due_at(&timed(1, at(5), None)), at(5));
    assert_eq!(edf.enforced_due_at(&timed(1, None, None)), None);
    let explicit_only = PolicyQueue::<()>::new(&controlled_delay_profile());
    assert_eq!(explicit_only.enforced_due_at(&timed(1, None, at(1))), None);
}

#[test]
fn next_due_at_follows_each_class_through_insert_remove_and_refill() {
    let mut queue = PolicyQueue::with_class_budget_expiry(&profile(
        r#"
default_policy_class: first
policy_classes:
  - name: first
    quantum: 10
    controlled_delay: true
  - name: second
    quantum: 10
    controlled_delay: true
"#,
    ));
    let now = Instant::now();
    let at = |millis| Some(now + std::time::Duration::from_millis(millis));
    let in_class = |class_index, millis| QueueMetadata {
        class_index,
        ..timed(1, None, at(millis))
    };
    push(&mut queue, in_class(0, 300), "later");
    assert_eq!(queue.next_due_at(0), at(300));
    push(&mut queue, in_class(0, 100), "earlier");
    push(&mut queue, in_class(1, 200), "second");
    assert_eq!(
        (queue.next_due_at(0), queue.next_due_at(1)),
        (at(100), at(200))
    );
    queue.retain(|payload| *payload != "earlier");
    assert_eq!(queue.next_due_at(0), at(300));
    queue.retain(|payload| *payload != "later");
    assert_eq!(queue.next_due_at(0), None);
    push(&mut queue, in_class(0, 400), "refill");
    assert_eq!(
        (queue.next_due_at(0), queue.next_due_at(1)),
        (at(400), at(200))
    );
    // Expiring one class leaves the other's due time alone.
    assert_eq!(
        take_expired_payloads(&mut queue, 1, now + std::time::Duration::from_millis(500)),
        ["second"]
    );
    assert_eq!(
        (queue.next_due_at(0), queue.next_due_at(1)),
        (at(400), None)
    );
}

#[test]
fn refused_extraction_spends_no_credit_and_its_successor_runs_next() {
    let entries = [
        (0, 6, "refused"),
        (0, 6, "successor"),
        (0, 1, "last"),
        (1, 4, "other"),
    ];
    let fill = || {
        let mut queue = PolicyQueue::new(&profile(
            r#"
default_policy_class: first
policy_classes:
  - name: first
    quantum: 10
  - name: second
    quantum: 10
"#,
        ));
        for (class_index, cost, payload) in entries {
            push(
                &mut queue,
                QueueMetadata {
                    class_index,
                    ..timed(cost, None, None)
                },
                payload,
            );
        }
        queue
    };

    let mut queue = fill();
    let mut order = Vec::new();
    while let Some((entry, charge)) = queue.extract_next(|_, _, _| true) {
        let payload = entry.into_payload();
        queue.settle(charge, payload != "refused");
        if payload == "refused" {
            // The quantum that made the head affordable stays with the class.
            assert_eq!(queue.class_deficit(0), 10);
        }
        order.push(payload);
    }
    assert_eq!(order, ["refused", "successor", "last", "other"]);
    assert_eq!(queue.class_deficit(0), 0);

    // Charged, the first head would leave 4 credit, too little for its successor, and
    // hand the next turn to the other class.
    assert_eq!(
        drain_all(&mut fill()),
        ["refused", "other", "successor", "last"]
    );
}

#[test]
fn drr_charges_cost_when_the_queue_was_empty() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let mut snapshot = QueueSnapshot::new(100, 0);
    snapshot.scheduling_cost_tokens = 7;
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, snapshot, 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "idle-arrival",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 1.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "following-arrival",
        )
        .unwrap();

    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "idle-arrival"
    );
    assert_eq!(queue.class_deficit(0), 3);
}

#[test]
fn retain_removes_payload_and_rebuilds_queue_accounting() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
policy_classes:
  - name: default
    quantum: 10
"#,
    ));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(8, 4), 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "keep",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(16, 6), 1.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "remove",
        )
        .unwrap();

    queue.retain(|payload| *payload != "remove");

    assert_eq!(queue.pending_count(), 1);
    assert_eq!(queue.class_stats(0).requests, 1);
    assert_eq!(queue.class_stats(0).raw_isl_tokens, 8);
    assert_eq!(queue.class_stats(0).cached_tokens, 4);
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "keep"
    );
}

#[test]
fn blocked_lane_does_not_block_another_lane() {
    let mut queue = PolicyQueue::new(&admission_profile());
    for (worker, payload) in [(1, "blocked"), (2, "ready")] {
        queue
            .enqueue_with_due_at(
                no_due_metadata(0, QueueSnapshot::new(1, 0), worker as f64),
                2,
                0.0,
                0,
                lane_of(worker),
                payload,
            )
            .unwrap();
    }

    let candidate = queue
        .pop_next(|_, _, payload| *payload != "blocked")
        .unwrap();
    assert_eq!(candidate.into_payload(), "ready");
    assert_eq!(queue.pending_count(), 1);
}

#[test]
fn a_recheck_reasks_every_blocked_lane_in_order() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let worker_1 = WorkerWithDpRank::new(1, 0);
    let worker_2 = WorkerWithDpRank::new(2, 0);
    for (sequence, (lane, payload)) in [
        (QueueLane::Exact(worker_1), "worker-1"),
        (QueueLane::Affinity(worker_1.into()), "soft-1"),
        (
            QueueLane::Affinity(WorkerAffinityTarget::new(1, None)),
            "host-1",
        ),
        (QueueLane::Affinity(worker_2.into()), "soft-2"),
        (QueueLane::Exact(worker_2), "worker-2"),
    ]
    .into_iter()
    .enumerate()
    {
        queue
            .enqueue_with_due_at(
                QueueMetadata {
                    arrival_offset_secs: sequence as f64,
                    ..timed(1, None, None)
                },
                2,
                0.0,
                0,
                lane,
                payload,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());
    queue.recheck_worker(WorkerWithDpRank::new(1, 1));
    assert_eq!(drain_all(&mut queue), ["host-1"]);
    queue.recheck_worker(worker_1);
    assert_eq!(drain_all(&mut queue), ["worker-1", "soft-1"]);
    queue.recheck_affinity_target(worker_2.into());
    assert_eq!(drain_all(&mut queue), ["soft-2"]);
    queue.recheck_worker(worker_2);
    assert_eq!(drain_all(&mut queue), ["worker-2"]);
}

#[test]
fn lane_index_tracks_a_new_higher_priority_head() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let worker = WorkerWithDpRank::new(1, 0);
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 0.0),
            1,
            0.0,
            0,
            lane_of(worker.worker_id),
            "old",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 1.0),
            1,
            10.0,
            0,
            lane_of(worker.worker_id),
            "boosted",
        )
        .unwrap();

    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "boosted"
    );
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "old"
    );
}

#[test]
fn exact_lane_head_index_checks_each_blocked_lane_once() {
    const LANES: usize = 10_000;
    let mut queue = PolicyQueue::new(&admission_profile());
    for lane in 0..LANES {
        queue
            .enqueue_with_due_at(
                no_due_metadata(0, QueueSnapshot::new(1, 0), lane as f64),
                LANES,
                0.0,
                0,
                lane_of(lane as u64),
                lane,
            )
            .unwrap();
    }

    let mut checks = 0;
    let mut popped = 0;
    while queue
        .pop_next(|_, _, lane| {
            checks += 1;
            !lane.is_multiple_of(2)
        })
        .is_some()
    {
        popped += 1;
    }

    assert_eq!(popped, LANES / 2);
    assert_eq!(checks, LANES);
    assert_eq!(queue.pending_count(), LANES / 2);
}

#[test]
fn global_recheck_dispatches_workers_as_they_become_available() {
    let mut queue = PolicyQueue::new(&admission_profile());
    for lane in 0..3 {
        queue
            .enqueue_with_due_at(
                no_due_metadata(0, QueueSnapshot::new(1, 0), lane as f64),
                3,
                0.0,
                0,
                lane_of(lane),
                lane,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());

    let mut available = [false, true, false];
    queue.recheck_blocked_lanes(RecheckScope::All);
    let entry = queue
        .pop_next(|_, _, lane| available[*lane as usize])
        .expect("newly available worker should dispatch");
    assert_eq!(entry.into_payload(), 1);
    assert!(
        queue
            .pop_next(|_, _, lane| available[*lane as usize])
            .is_none()
    );

    available[2] = true;
    queue.recheck_blocked_lanes(RecheckScope::All);
    let entry = queue
        .pop_next(|_, _, lane| available[*lane as usize])
        .expect("later worker availability should dispatch on the next recheck");
    assert_eq!(entry.into_payload(), 2);
    assert_eq!(queue.pending_count(), 1);
}

#[test]
fn global_recheck_preserves_priority_for_multiple_available_workers() {
    let mut queue = PolicyQueue::new(&admission_profile());
    for (worker, strict_priority, payload) in [(1, 0, "low"), (2, 1, "high")] {
        queue
            .enqueue_with_due_at(
                no_due_metadata(0, QueueSnapshot::new(1, 0), 0.0),
                2,
                0.0,
                strict_priority,
                lane_of(worker),
                payload,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());

    queue.recheck_blocked_lanes(RecheckScope::All);
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "high"
    );
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "low"
    );
}

#[test]
fn pending_global_recheck_survives_retain_removing_blocked_lane() {
    let mut queue = PolicyQueue::new(&admission_profile());
    for (worker, payload) in [(1, "remove"), (2, "keep")] {
        queue
            .enqueue_with_due_at(
                no_due_metadata(0, QueueSnapshot::new(1, 0), 0.0),
                2,
                0.0,
                0,
                lane_of(worker),
                payload,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());
    queue.recheck_blocked_lanes(RecheckScope::All);
    queue.retain(|payload| *payload == "keep");

    assert_eq!(queue.pending_count(), 1);
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "keep"
    );
    assert_eq!(queue.pending_count(), 0);
}

#[test]
fn per_worker_token_caps_follow_capacity_without_evicting() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: raw
policy_classes:
  - name: raw
    quantum: 1
    raw_isl_token_queue_limit_per_worker: 10
  - name: cached
    quantum: 1
    cached_token_queue_limit_per_worker: 5
  - name: zero
    quantum: 1
    request_queue_limit_per_worker: 0
  - name: no-workers
    quantum: 1
    request_queue_limit_per_worker: 1
"#,
    ));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(11, 0), 0.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "raw-queued",
        )
        .unwrap();
    let (raw_rejection, _) = queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 1.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "raw-rejected",
        )
        .unwrap_err();
    assert_eq!(raw_rejection.limit_kind, QueueLimitKind::RawIslTokens);
    assert_eq!(raw_rejection.current, 11);
    assert_eq!(raw_rejection.limit, 10);
    assert_eq!(queue.class_stats(0).raw_isl_tokens, 11);

    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(10, 0), 2.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "raw-after-growth",
        )
        .unwrap();
    let (grown_rejection, _) = queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 3.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "raw-at-grown-cap",
        )
        .unwrap_err();
    assert_eq!(grown_rejection.current, 21);
    assert_eq!(grown_rejection.limit, 20);

    queue
        .enqueue_with_due_at(
            no_due_metadata(1, QueueSnapshot::new(8, 6), 0.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "cached-queued",
        )
        .unwrap();
    let (cached_rejection, _) = queue
        .enqueue_with_due_at(
            no_due_metadata(1, QueueSnapshot::new(1, 1), 1.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "cached-rejected",
        )
        .unwrap_err();
    assert_eq!(cached_rejection.limit_kind, QueueLimitKind::CachedTokens);
    assert_eq!(cached_rejection.current, 6);
    assert_eq!(cached_rejection.limit, 5);

    let (zero_rejection, _) = queue
        .enqueue_with_due_at(
            no_due_metadata(2, QueueSnapshot::new(1, 0), 0.0),
            4,
            0.0,
            0,
            QueueLane::Shared,
            "zero",
        )
        .unwrap_err();
    assert_eq!(zero_rejection.limit_kind, QueueLimitKind::Requests);
    assert_eq!(zero_rejection.limit, 0);

    let (no_workers_rejection, _) = queue
        .enqueue_with_due_at(
            no_due_metadata(3, QueueSnapshot::new(1, 0), 0.0),
            0,
            0.0,
            0,
            QueueLane::Shared,
            "no-workers",
        )
        .unwrap_err();
    assert_eq!(no_workers_rejection.current, 0);
    assert_eq!(no_workers_rejection.limit, 0);
}

#[test]
fn per_worker_limit_multiplication_saturates() {
    let mut queue = PolicyQueue::new(&profile(&format!(
        r#"
policy_classes:
  - name: capped
    quantum: 1
    request_queue_limit_per_worker: {}
"#,
        usize::MAX
    )));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 0.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "queued",
        )
        .unwrap();
}

#[test]
fn fcfs_and_wspt_order_only_within_each_class() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: fcfs
policy_classes:
  - name: fcfs
    queue_policy: fcfs
    quantum: 50
  - name: wspt
    queue_policy: wspt
    quantum: 50
"#,
    ));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(50, 0), 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "fcfs-long",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 1.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "fcfs-short",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(1, QueueSnapshot::new(50, 0), 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "wspt-long",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(1, QueueSnapshot::new(1, 0), 1.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "wspt-short",
        )
        .unwrap();

    let first = queue.pop_next(|_, _, _| true).unwrap();
    let second = queue.pop_next(|_, _, _| true).unwrap();
    assert_eq!(first.into_payload(), "fcfs-long");
    assert_eq!(second.into_payload(), "wspt-short");
}

#[test]
fn drr_weights_progress_and_skips_blocked_classes_without_credit() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: slow
policy_classes:
  - name: slow
    quantum: 1
  - name: fast
    quantum: 3
"#,
    ));
    for index in 0..6 {
        queue
            .enqueue_with_due_at(
                no_due_metadata(0, QueueSnapshot::new(1, 0), index as f64),
                1,
                0.0,
                0,
                QueueLane::Shared,
                "slow",
            )
            .unwrap();
        queue
            .enqueue_with_due_at(
                no_due_metadata(1, QueueSnapshot::new(1, 0), index as f64),
                1,
                0.0,
                0,
                QueueLane::Shared,
                "fast",
            )
            .unwrap();
    }

    let mut first_six = Vec::new();
    for _ in 0..6 {
        first_six.push(queue.pop_next(|_, _, _| true).unwrap().into_payload());
    }
    assert!(first_six.iter().filter(|value| **value == "fast").count() >= 3);

    let blocked_deficit = queue.class_deficit(1);
    let slow = queue.pop_next(|class, _, _| class == 0).unwrap();
    assert_eq!(slow.into_payload(), "slow");
    assert_eq!(queue.class_deficit(1), blocked_deficit);
}

#[test]
fn drr_carry_uses_next_dispatchable_lane() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: agents
policy_classes:
  - name: agents
    quantum: 10
  - name: batch
    quantum: 1
"#,
    ));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(5, 0), 0.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "agents-first",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(1, 0), 1.0),
            2,
            0.0,
            0,
            lane_of(1),
            "agents-blocked",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(10, 0), 2.0),
            2,
            0.0,
            0,
            lane_of(2),
            "agents-ready",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(1, QueueSnapshot::new(1, 0), 0.0),
            2,
            0.0,
            0,
            QueueLane::Shared,
            "batch",
        )
        .unwrap();

    let is_dispatchable =
        |_: usize, _: &PolicyClassConfig, payload: &&str| *payload != "agents-blocked";
    assert_eq!(
        queue.pop_next(is_dispatchable).unwrap().into_payload(),
        "agents-first"
    );
    assert_eq!(queue.class_deficit(0), 5);
    assert_eq!(
        queue.pop_next(is_dispatchable).unwrap().into_payload(),
        "batch"
    );
    assert_eq!(queue.class_deficit(0), 5);
}

#[test]
fn drr_serves_exact_quantum_ratio_for_equal_cost_backlogs() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: one
policy_classes:
  - name: one
    quantum: 1
  - name: three
    quantum: 3
"#,
    ));
    for index in 0..20 {
        queue
            .enqueue_with_due_at(
                no_due_metadata(0, QueueSnapshot::new(1, 0), index as f64),
                1,
                0.0,
                0,
                QueueLane::Shared,
                "one",
            )
            .unwrap();
    }
    for index in 0..60 {
        queue
            .enqueue_with_due_at(
                no_due_metadata(1, QueueSnapshot::new(1, 0), index as f64),
                1,
                0.0,
                0,
                QueueLane::Shared,
                "three",
            )
            .unwrap();
    }

    let dispatches = (0..80)
        .map(|_| queue.pop_next(|_, _, _| true).unwrap().into_payload())
        .collect::<Vec<_>>();
    for round in dispatches.chunks_exact(4) {
        assert_eq!(round, ["one", "three", "three", "three"]);
    }
}

#[test]
fn drr_quanta_split_charged_cost_not_request_count() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 800
    queue_policy: fcfs
    slo_ttft_ms: 1000
  - name: standard
    quantum: 200
    queue_policy: fcfs
    slo_ttft_ms: 3000
"#,
    ));
    // Priority requests cost 300 and 500. Standard's first three requests cost its whole
    // 200 quantum; its last costs 150 and leaves credit unspent when Standard empties.
    for round in 0..5 {
        for cost in [300, 500] {
            queue
                .enqueue_with_due_at(
                    no_due_metadata(0, QueueSnapshot::new(cost, 0), round as f64),
                    1,
                    0.0,
                    0,
                    QueueLane::Shared,
                    (),
                )
                .unwrap();
        }
    }
    for (round, cost) in [200, 200, 200, 150].into_iter().enumerate() {
        queue
            .enqueue_with_due_at(
                no_due_metadata(1, QueueSnapshot::new(cost, 0), round as f64),
                1,
                0.0,
                0,
                QueueLane::Shared,
                (),
            )
            .unwrap();
    }

    let dispatches = std::iter::from_fn(|| queue.pop_next(|_, _, _| true))
        .map(|entry| (entry.class_index(), entry.snapshot().scheduling_cost_tokens))
        .collect::<Vec<_>>();
    // While both classes are backlogged, each round charges 800 to Priority for two
    // requests and 200 to Standard for one: an 80/20 cost split at a 2:1 request count.
    for round in dispatches[..9].chunks_exact(3) {
        assert_eq!(round, [(0, 300), (0, 500), (1, 200)]);
    }
    // Standard's last request leaves 50 of its 200 credit unspent. Emptying the class
    // resets that credit to zero, and Priority then uses the idle capacity.
    assert_eq!(
        dispatches[9..],
        [(0, 300), (0, 500), (1, 150), (0, 300), (0, 500)]
    );
    assert_eq!(queue.class_deficit(1), 0);
    assert_eq!(queue.pending_count(), 0);
}

#[test]
fn fully_blocked_ring_returns_without_accruing_deficit() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: first
policy_classes:
  - name: first
    quantum: 7
  - name: second
    quantum: 11
"#,
    ));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(100, 0), 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "first",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(1, QueueSnapshot::new(100, 0), 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "second",
        )
        .unwrap();

    for _ in 0..10_000 {
        assert!(queue.pop_next(|_, _, _| false).is_none());
    }
    assert_eq!(queue.class_deficit(0), 0);
    assert_eq!(queue.class_deficit(1), 0);
}

#[test]
fn oversized_heads_bulk_add_deficit_and_make_progress() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_class: large
policy_classes:
  - name: large
    quantum: 4
  - name: blocked
    quantum: 100
"#,
    ));
    queue
        .enqueue_with_due_at(
            no_due_metadata(0, QueueSnapshot::new(101, 0), 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "large",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            no_due_metadata(1, QueueSnapshot::new(1, 0), 0.0),
            1,
            0.0,
            0,
            QueueLane::Shared,
            "blocked",
        )
        .unwrap();

    let popped = queue
        .pop_next(|class, _, _| class == 0)
        .expect("oversized request should make bounded progress");
    assert_eq!(popped.into_payload(), "large");
    assert_eq!(queue.pending_count(), 1);
}
