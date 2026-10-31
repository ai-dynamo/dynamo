// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The policy queue driven through profiles the YAML resolver built; the queue itself lives
//! in the admission stage.

use tokio::time::Instant;

use crate::protocols::WorkerWithDpRank;
use crate::scheduling::policy_config::RouterQueuePolicy;
use crate::scheduling::policy_queue::*;
use crate::scheduling::{PolicyClassConfig, PolicyProfile, RouterPolicyConfig, WorkerPlacement};

fn profile(yaml: &str) -> PolicyProfile {
    RouterPolicyConfig::from_yaml(yaml)
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
}

fn admission_profile() -> PolicyProfile {
    profile(
        r#"
default_policy_family: agents
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: agents
    cache_bucket: all
    queue_policy: fcfs
    quantum: 10
"#,
    )
}

#[test]
fn per_worker_caps_scale_and_remain_pre_add() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 10
    request_queue_limit_per_worker: 1
    raw_isl_token_queue_limit_per_worker: 5
    cached_token_queue_limit_per_worker: 3
"#,
    ));
    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(8, 4),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "first",
        )
        .unwrap();
    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(100, 100),
            1.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "overshoot",
        )
        .unwrap();
    let (rejection, payload) = queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(1, 0),
            2.0,
            0.0,
            0,
            WorkerPlacement::Any,
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
                arrival_offset_secs: 0.0,
            },
            1,
            0.0,
            0,
            WorkerPlacement::Any,
            "later-due",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(now + std::time::Duration::from_secs(10)),
                arrival_offset_secs: 1.0,
            },
            1,
            0.0,
            0,
            WorkerPlacement::Any,
            "earlier-due",
        )
        .unwrap();

    assert_eq!(
        queue.next_due_at(),
        Some(now + std::time::Duration::from_secs(10))
    );
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "earlier-due"
    );
    assert_eq!(queue.due_entry_count(), 1);
    assert_eq!(
        queue.next_due_at(),
        Some(now + std::time::Duration::from_secs(20))
    );
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "later-due"
    );
    assert_eq!(queue.due_entry_count(), 0);
    assert_eq!(queue.next_due_at(), None);
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
                arrival_offset_secs: 0.0,
            },
            1,
            0.0,
            0,
            WorkerPlacement::Exact(WorkerWithDpRank::new(1, 0)),
            "later-due-better-fcfs",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(due_at),
                arrival_offset_secs: 1.0,
            },
            1,
            0.0,
            0,
            WorkerPlacement::Exact(WorkerWithDpRank::new(2, 0)),
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
        .enqueue(
            0,
            2,
            QueueSnapshot::new(1, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "no-deadline",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(due_at),
                arrival_offset_secs: 1.0,
            },
            2,
            0.0,
            0,
            WorkerPlacement::Exact(WorkerWithDpRank::new(1, 0)),
            "deadline",
        )
        .unwrap();

    assert_eq!(queue.next_due_at(), Some(due_at));
    queue.retain(|payload| *payload != "deadline");
    assert_eq!(queue.due_entry_count(), 0);
    assert_eq!(queue.next_due_at(), None);
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
                arrival_offset_secs: 0.0,
            },
            1,
            0.0,
            0,
            WorkerPlacement::Exact(worker),
            "blocked",
        )
        .unwrap();

    assert!(queue.pop_next(|_, _, _| false).is_none());
    assert!(queue.class_blocks_worker(0, worker));
    assert_eq!(queue.next_due_at(), Some(due_at));

    let expired = queue.take_expired(due_at);
    assert_eq!(expired.len(), 1);
    assert_eq!(
        expired.into_iter().next().unwrap().into_payload(),
        "blocked"
    );
    assert_eq!(queue.next_due_at(), None);
    assert!(
        queue
            .pop_next(|_, _, _| panic!("expired lane revisited"))
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
                arrival_offset_secs: 0.0,
            },
            1,
            0.0,
            10,
            WorkerPlacement::Any,
            "high-priority-later-due",
        )
        .unwrap();
    queue
        .enqueue_with_due_at(
            QueueMetadata {
                class_index: 0,
                snapshot: QueueSnapshot::new(1, 0),
                due_at: Some(now + std::time::Duration::from_secs(10)),
                arrival_offset_secs: 1.0,
            },
            1,
            0.0,
            0,
            WorkerPlacement::Any,
            "low-priority-earlier-due",
        )
        .unwrap();

    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "high-priority-later-due"
    );
}

#[test]
fn drr_charges_cost_when_the_queue_was_empty() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let mut snapshot = QueueSnapshot::new(100, 0);
    snapshot.scheduling_cost_tokens = 7;
    queue
        .enqueue(
            0,
            1,
            snapshot,
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "idle-arrival",
        )
        .unwrap();
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(1, 0),
            1.0,
            0.0,
            0,
            WorkerPlacement::Any,
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
default_policy_family: default
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: default
    policy_family: default
    cache_bucket: all
    quantum: 10
"#,
    ));
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(8, 4),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "keep",
        )
        .unwrap();
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(16, 6),
            1.0,
            0.0,
            0,
            WorkerPlacement::Any,
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
fn blocked_worker_lane_does_not_block_another_lane() {
    let mut queue = PolicyQueue::new(&admission_profile());
    for (worker, payload) in [(1, "blocked"), (2, "ready")] {
        queue
            .enqueue(
                0,
                2,
                QueueSnapshot::new(1, 0),
                worker as f64,
                0.0,
                0,
                WorkerPlacement::Exact(WorkerWithDpRank::new(worker, 0)),
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
fn worker_update_rechecks_only_that_blocked_lane() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let worker_1 = WorkerWithDpRank::new(1, 0);
    let worker_2 = WorkerWithDpRank::new(2, 0);
    for (worker, payload) in [(worker_1, "worker-1"), (worker_2, "worker-2")] {
        queue
            .enqueue(
                0,
                2,
                QueueSnapshot::new(1, 0),
                worker.worker_id as f64,
                0.0,
                0,
                WorkerPlacement::Exact(worker),
                payload,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());
    queue.recheck_worker(worker_1);
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "worker-1"
    );
    assert!(queue.pop_next(|_, _, _| true).is_none());
    queue.recheck_worker(worker_2);
    assert_eq!(
        queue.pop_next(|_, _, _| true).unwrap().into_payload(),
        "worker-2"
    );
}

#[test]
fn exact_lane_index_tracks_a_new_higher_priority_head() {
    let mut queue = PolicyQueue::new(&admission_profile());
    let worker = WorkerWithDpRank::new(1, 0);
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(1, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Exact(worker),
            "old",
        )
        .unwrap();
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(1, 0),
            1.0,
            10.0,
            0,
            WorkerPlacement::Exact(worker),
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
            .enqueue(
                0,
                LANES,
                QueueSnapshot::new(1, 0),
                lane as f64,
                0.0,
                0,
                WorkerPlacement::Exact(WorkerWithDpRank::new(lane as u64, 0)),
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
            .enqueue(
                0,
                3,
                QueueSnapshot::new(1, 0),
                lane as f64,
                0.0,
                0,
                WorkerPlacement::Exact(WorkerWithDpRank::new(lane, 0)),
                lane,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());

    let mut available = [false, true, false];
    queue.recheck_all_workers();
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
    queue.recheck_all_workers();
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
            .enqueue(
                0,
                2,
                QueueSnapshot::new(1, 0),
                0.0,
                0.0,
                strict_priority,
                WorkerPlacement::Exact(WorkerWithDpRank::new(worker, 0)),
                payload,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());

    queue.recheck_all_workers();
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
            .enqueue(
                0,
                2,
                QueueSnapshot::new(1, 0),
                0.0,
                0.0,
                0,
                WorkerPlacement::Exact(WorkerWithDpRank::new(worker, 0)),
                payload,
            )
            .unwrap();
    }

    assert!(queue.pop_next(|_, _, _| false).is_none());
    queue.recheck_all_workers();
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
default_policy_family: raw
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: raw
    policy_family: raw
    cache_bucket: all
    quantum: 1
    raw_isl_token_queue_limit_per_worker: 10
  - name: cached
    policy_family: cached
    cache_bucket: all
    quantum: 1
    cached_token_queue_limit_per_worker: 5
  - name: zero
    policy_family: zero
    cache_bucket: all
    quantum: 1
    request_queue_limit_per_worker: 0
  - name: no-workers
    policy_family: no-workers
    cache_bucket: all
    quantum: 1
    request_queue_limit_per_worker: 1
"#,
    ));
    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(11, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "raw-queued",
        )
        .unwrap();
    let (raw_rejection, _) = queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(1, 0),
            1.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "raw-rejected",
        )
        .unwrap_err();
    assert_eq!(raw_rejection.limit_kind, QueueLimitKind::RawIslTokens);
    assert_eq!(raw_rejection.current, 11);
    assert_eq!(raw_rejection.limit, 10);
    assert_eq!(queue.class_stats(0).raw_isl_tokens, 11);

    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(10, 0),
            2.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "raw-after-growth",
        )
        .unwrap();
    let (grown_rejection, _) = queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(1, 0),
            3.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "raw-at-grown-cap",
        )
        .unwrap_err();
    assert_eq!(grown_rejection.current, 21);
    assert_eq!(grown_rejection.limit, 20);

    queue
        .enqueue(
            1,
            2,
            QueueSnapshot::new(8, 6),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "cached-queued",
        )
        .unwrap();
    let (cached_rejection, _) = queue
        .enqueue(
            1,
            1,
            QueueSnapshot::new(1, 1),
            1.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "cached-rejected",
        )
        .unwrap_err();
    assert_eq!(cached_rejection.limit_kind, QueueLimitKind::CachedTokens);
    assert_eq!(cached_rejection.current, 6);
    assert_eq!(cached_rejection.limit, 5);

    let (zero_rejection, _) = queue
        .enqueue(
            2,
            4,
            QueueSnapshot::new(1, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "zero",
        )
        .unwrap_err();
    assert_eq!(zero_rejection.limit_kind, QueueLimitKind::Requests);
    assert_eq!(zero_rejection.limit, 0);

    let (no_workers_rejection, _) = queue
        .enqueue(
            3,
            0,
            QueueSnapshot::new(1, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
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
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    request_queue_limit_per_worker: {}
"#,
        usize::MAX
    )));
    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(1, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "queued",
        )
        .unwrap();
}

#[test]
fn fcfs_and_wspt_order_only_within_each_class() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_family: fcfs
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: fcfs
    policy_family: fcfs
    cache_bucket: all
    queue_policy: fcfs
    quantum: 50
  - name: wspt
    policy_family: wspt
    cache_bucket: all
    queue_policy: wspt
    quantum: 50
"#,
    ));
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(50, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "fcfs-long",
        )
        .unwrap();
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(1, 0),
            1.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "fcfs-short",
        )
        .unwrap();
    queue
        .enqueue(
            1,
            1,
            QueueSnapshot::new(50, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "wspt-long",
        )
        .unwrap();
    queue
        .enqueue(
            1,
            1,
            QueueSnapshot::new(1, 0),
            1.0,
            0.0,
            0,
            WorkerPlacement::Any,
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
default_policy_family: slow
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: slow
    policy_family: slow
    cache_bucket: all
    quantum: 1
  - name: fast
    policy_family: fast
    cache_bucket: all
    quantum: 3
"#,
    ));
    for index in 0..6 {
        queue
            .enqueue(
                0,
                1,
                QueueSnapshot::new(1, 0),
                index as f64,
                0.0,
                0,
                WorkerPlacement::Any,
                "slow",
            )
            .unwrap();
        queue
            .enqueue(
                1,
                1,
                QueueSnapshot::new(1, 0),
                index as f64,
                0.0,
                0,
                WorkerPlacement::Any,
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
default_policy_family: agents
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: agents
    cache_bucket: all
    quantum: 10
  - name: batch
    policy_family: batch
    cache_bucket: all
    quantum: 1
"#,
    ));
    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(5, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "agents-first",
        )
        .unwrap();
    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(1, 0),
            1.0,
            0.0,
            0,
            WorkerPlacement::Exact(WorkerWithDpRank::new(1, 0)),
            "agents-blocked",
        )
        .unwrap();
    queue
        .enqueue(
            0,
            2,
            QueueSnapshot::new(10, 0),
            2.0,
            0.0,
            0,
            WorkerPlacement::Exact(WorkerWithDpRank::new(2, 0)),
            "agents-ready",
        )
        .unwrap();
    queue
        .enqueue(
            1,
            2,
            QueueSnapshot::new(1, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
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
default_policy_family: one
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: one
    policy_family: one
    cache_bucket: all
    quantum: 1
  - name: three
    policy_family: three
    cache_bucket: all
    quantum: 3
"#,
    ));
    for index in 0..20 {
        queue
            .enqueue(
                0,
                1,
                QueueSnapshot::new(1, 0),
                index as f64,
                0.0,
                0,
                WorkerPlacement::Any,
                "one",
            )
            .unwrap();
    }
    for index in 0..60 {
        queue
            .enqueue(
                1,
                1,
                QueueSnapshot::new(1, 0),
                index as f64,
                0.0,
                0,
                WorkerPlacement::Any,
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
fn fully_blocked_ring_returns_without_accruing_deficit() {
    let mut queue = PolicyQueue::new(&profile(
        r#"
default_policy_family: first
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: first
    policy_family: first
    cache_bucket: all
    quantum: 7
  - name: second
    policy_family: second
    cache_bucket: all
    quantum: 11
"#,
    ));
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(100, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "first",
        )
        .unwrap();
    queue
        .enqueue(
            1,
            1,
            QueueSnapshot::new(100, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
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
default_policy_family: large
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: large
    policy_family: large
    cache_bucket: all
    quantum: 4
  - name: blocked
    policy_family: blocked
    cache_bucket: all
    quantum: 100
"#,
    ));
    queue
        .enqueue(
            0,
            1,
            QueueSnapshot::new(101, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "large",
        )
        .unwrap();
    queue
        .enqueue(
            1,
            1,
            QueueSnapshot::new(1, 0),
            0.0,
            0.0,
            0,
            WorkerPlacement::Any,
            "blocked",
        )
        .unwrap();

    let popped = queue
        .pop_next(|class, _, _| class == 0)
        .expect("oversized request should make bounded progress");
    assert_eq!(popped.into_payload(), "large");
    assert_eq!(queue.pending_count(), 1);
}
