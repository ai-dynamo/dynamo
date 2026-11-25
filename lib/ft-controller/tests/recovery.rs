// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The controller against a fake vLLM world over real HTTP.

mod common;

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use common::{
    FakeWorld, Running, STORE_PORT_BASE, Status, Timing, file_store, ms, settings, store_host,
    wait_until,
};
use dynamo_ft_controller::controller::{ConfigError, Event, Outcome};
use dynamo_ft_controller::decision::StoreAddr;
use dynamo_ft_controller::protocol::Instruction;
use dynamo_ft_controller::round::{FileRoundStore, RoundRecord, RoundStore};
use serde_json::json;

fn assert_clean(fake: &FakeWorld) {
    assert_eq!(fake.violations(), Vec::<String>::new());
}

fn order_ranks(fake: &FakeWorld, request_id: &str) -> Vec<u32> {
    let mut ranks: Vec<u32> = fake
        .orders()
        .iter()
        .filter(|order| order.request_id == request_id)
        .map(|order| order.rank)
        .collect();
    ranks.sort_unstable();
    ranks
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dead_rank_is_scaled_down_with_one_shared_request_id() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-scale"),
        file_store(state.path(), "w-scale"),
    );
    controller.formed().await;

    fake.engine_dies(2);
    let (rounds, recovery) = controller.recovered().await;

    assert_eq!(rounds.len(), 1);
    let round = &rounds[0];
    assert_eq!(round.instruction, Instruction::ScaleDown);
    assert_eq!(round.removed, [2]);
    assert_eq!(round.survivors, [0, 1, 3]);
    assert_eq!(round.params, json!({"removed_dp_ranks": [2]}));
    assert_eq!(round.store, None);
    assert_eq!(round.next_store_port, u32::from(STORE_PORT_BASE));
    assert!(!round.escalated);
    assert!(
        round.request_id.starts_with("w-scale-e1-r1-"),
        "{}",
        round.request_id
    );
    // Every survivor got the order exactly once, with the same request id.
    assert_eq!(fake.orders().len(), 3);
    assert_eq!(order_ranks(&fake, &round.request_id), [0, 1, 3]);

    assert_eq!(recovery.removed, [2]);
    assert_eq!(recovery.survivors, [0, 1, 3]);
    assert_eq!(recovery.rounds, 1);
    assert_eq!(recovery.request_id, round.request_id);
    let line = recovery.to_string();
    assert!(
        line.starts_with("FT_CONTROLLER_RECOVERED world=w-scale "),
        "{line}"
    );
    for field in [
        "removed=[2]",
        "detect_s=",
        "round_s=",
        "total_s=",
        "request_id=w-scale-e1-r1-",
    ] {
        assert!(line.contains(field), "{field} missing from {line}");
    }

    // The round was recorded before it was sent.
    let saved = FileRoundStore::new(state.path(), "w-scale").last().unwrap();
    assert_eq!(saved.as_ref(), Some(round));

    assert_eq!(fake.members(), BTreeSet::from([0, 1, 3]));
    for rank in [0, 1, 3] {
        assert_eq!(fake.status(rank), Status::Healthy);
    }
    assert_clean(&fake);
    assert_eq!(controller.stop().await, Outcome::Stopped);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn losing_the_store_host_moves_the_store_and_later_rounds_move_it_again() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-store"),
        file_store(state.path(), "w-store"),
    );
    controller.formed().await;

    // Rank 0 hosts the store: the new store goes to the lowest survivor.
    fake.engine_dies(0);
    let (first, _) = controller.recovered().await;
    assert_eq!(first.len(), 1);
    assert_eq!(
        first[0].params,
        json!({"removed_dp_ranks": [0], "dp_master_ip": store_host(1), "dp_store_port": STORE_PORT_BASE})
    );
    assert_eq!(first[0].next_store_port, u32::from(STORE_PORT_BASE) + 1);

    // The store host survives this time, but every round after a store round
    // carries a store address again, on a new port.
    fake.engine_dies(2);
    let (second, recovery) = controller.recovered().await;
    assert_eq!(second.len(), 1);
    assert_eq!(
        second[0].store,
        Some(StoreAddr {
            host: store_host(1),
            port: STORE_PORT_BASE + 1
        })
    );
    assert_eq!(second[0].removed, [2]);
    assert_eq!(recovery.survivors, [1, 3]);
    assert_ne!(first[0].request_id, second[0].request_id);
    assert_eq!(fake.members(), BTreeSet::from([1, 3]));
    assert_clean(&fake);
    controller.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_survivors_are_woken_before_the_round() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    fake.set_idle(1, true);
    fake.set_idle(3, true);
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-wake"),
        file_store(state.path(), "w-wake"),
    );
    controller.formed().await;

    fake.engine_dies(2);
    let (rounds, _) = controller.recovered().await;
    assert_eq!(rounds.len(), 1);
    assert_eq!(rounds[0].removed, [2]);

    let wakes = fake.wakes();
    let woken: BTreeSet<u32> = wakes.iter().map(|wake| wake.rank).collect();
    assert_eq!(
        woken,
        BTreeSet::from([1, 3]),
        "only idle survivors are woken"
    );
    let first_order = fake.orders().iter().map(|order| order.at).min().unwrap();
    assert!(
        wakes.iter().all(|wake| wake.at < first_order),
        "wakes come before the round"
    );
    assert_clean(&fake);
    controller.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn all_alive_but_unhealthy_is_retried_on_every_rank() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-retry"),
        file_store(state.path(), "w-retry"),
    );
    controller.formed().await;

    fake.transient_fault();
    let (rounds, recovery) = controller.recovered().await;
    assert_eq!(rounds.len(), 1);
    assert_eq!(rounds[0].instruction, Instruction::Retry);
    assert_eq!(rounds[0].params, json!({}));
    assert!(rounds[0].removed.is_empty());
    assert_eq!(order_ranks(&fake, &rounds[0].request_id), [0, 1, 2, 3]);
    assert_eq!(recovery.instruction, Instruction::Retry);
    assert!(recovery.to_string().contains("removed=[]"));
    assert_eq!(fake.members(), BTreeSet::from([0, 1, 2, 3]));
    assert_clean(&fake);
    controller.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_survivor_dying_mid_round_escalates() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    fake.die_on_order(1);
    let state = tempfile::tempdir().unwrap();
    let controller = {
        let mut controller =
            Running::start(&fake, settings("w-mid"), file_store(state.path(), "w-mid"));
        controller.formed().await;
        controller
    };
    let escalator = controller.escalator.clone();

    fake.engine_dies(3);
    let Outcome::Escalated(reason) = controller.outcome().await else {
        panic!("expected an escalation");
    };
    assert!(reason.contains("rank 1"), "{reason}");
    // Rank 1's multi-line error stays on one line.
    assert!(!reason.contains('\n'), "{reason}");
    assert_eq!(
        *escalator.calls.lock().unwrap(),
        [("w-mid".to_string(), reason.clone())]
    );
    // One round only: the controller did not try again with a dead survivor.
    let ids: BTreeSet<String> = fake
        .orders()
        .into_iter()
        .map(|order| order.request_id)
        .collect();
    assert_eq!(ids.len(), 1);
    // The supervisor restarts the whole world, so the record says so.
    let last = FileRoundStore::new(state.path(), "w-mid")
        .last()
        .unwrap()
        .unwrap();
    assert!(last.escalated);
    assert_clean(&fake);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lost_order_is_followed_by_a_new_round_with_a_new_request_id() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-lost"),
        file_store(state.path(), "w-lost"),
    );
    controller.formed().await;

    // Rank 3 never sees the first round's order. Ranks 0 and 2 wait for it at
    // check-in, time out and stay unhealthy (vLLM 0.34.0), so the controller
    // sends a new round.
    fake.drop_next_order(3);
    fake.engine_dies(1);
    let (rounds, recovery) = controller.recovered().await;

    assert_eq!(rounds.len(), 2);
    assert_ne!(rounds[0].request_id, rounds[1].request_id);
    for round in &rounds {
        assert_eq!(round.instruction, Instruction::ScaleDown);
        assert_eq!(round.removed, [1]);
        assert_eq!(round.survivors, [0, 2, 3]);
        assert_eq!(order_ranks(&fake, &round.request_id), [0, 2, 3]);
    }
    let lost: Vec<(u32, String)> = fake
        .orders()
        .into_iter()
        .filter(|order| order.dropped)
        .map(|order| (order.rank, order.request_id))
        .collect();
    assert_eq!(lost, [(3, rounds[0].request_id.clone())]);
    assert_eq!(recovery.rounds, 2);
    assert_eq!(recovery.request_id, rounds[1].request_id);
    assert_eq!(fake.members(), BTreeSet::from([0, 2, 3]));
    assert_clean(&fake);
    controller.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_store_round_rebuilds_the_store_on_a_new_port() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    fake.refuse_store_port(STORE_PORT_BASE);
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-port"),
        file_store(state.path(), "w-port"),
    );
    controller.formed().await;

    fake.engine_dies(0);
    let (rounds, recovery) = controller.recovered().await;
    assert_eq!(rounds.len(), 2);
    let ports: Vec<u16> = rounds
        .iter()
        .map(|round| round.store.as_ref().unwrap().port)
        .collect();
    assert_eq!(ports, [STORE_PORT_BASE, STORE_PORT_BASE + 1]);
    for round in &rounds {
        assert_eq!(round.removed, [0]);
        assert_eq!(round.store.as_ref().unwrap().host, store_host(1));
        assert_eq!(order_ranks(&fake, &round.request_id), [1, 2, 3]);
    }
    assert_ne!(rounds[0].request_id, rounds[1].request_id);
    assert_eq!(recovery.rounds, 2);
    assert_eq!(recovery.request_id, rounds[1].request_id);
    assert_clean(&fake);
    controller.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_round_budget_bounds_failed_rounds() {
    let fake = FakeWorld::start(3, Timing::default()).await;
    for port in STORE_PORT_BASE..STORE_PORT_BASE + 3 {
        fake.refuse_store_port(port);
    }
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-budget"),
        file_store(state.path(), "w-budget"),
    );
    controller.formed().await;

    fake.engine_dies(0);
    let Outcome::Escalated(reason) = controller.outcome().await else {
        panic!("expected an escalation");
    };
    assert!(reason.contains("round budget of 3"), "{reason}");
    let ids: BTreeSet<String> = fake
        .orders()
        .into_iter()
        .map(|order| order.request_id)
        .collect();
    assert_eq!(ids.len(), 3);
    assert_clean(&fake);
}

/// A [`RoundStore`] whose disk is full.
struct FullDisk;

impl RoundStore for FullDisk {
    fn begin_epoch(&self) -> Result<u64> {
        Ok(1)
    }

    fn save(&self, _record: &RoundRecord) -> Result<()> {
        Err(anyhow!("no space left on device"))
    }

    fn last(&self) -> Result<Option<RoundRecord>> {
        Ok(None)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_round_that_cannot_be_recorded_is_never_sent() {
    let fake = FakeWorld::start(3, Timing::default()).await;
    let mut controller = Running::start(&fake, settings("w-disk"), Arc::new(FullDisk));
    controller.formed().await;
    let escalator = controller.escalator.clone();

    fake.engine_dies(2);
    let Outcome::Escalated(reason) = controller.outcome().await else {
        panic!("expected an escalation");
    };
    assert!(
        reason.contains("cannot record round w-disk-e1-r1-"),
        "{reason}"
    );
    assert!(reason.contains("no space left on device"), "{reason}");
    assert!(fake.orders().is_empty(), "an unrecorded round was sent");
    assert_eq!(escalator.calls.lock().unwrap().len(), 1);
    assert_clean(&fake);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unreachable_rank_counts_as_dead_after_dead_after() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let config = settings("w-gone");
    let dead_after = config.policy.dead_after;
    let mut controller = Running::start(&fake, config, file_store(state.path(), "w-gone"));
    controller.formed().await;

    let killed_at = Instant::now();
    fake.kill(2);
    let (rounds, recovery) = controller.recovered().await;
    assert_eq!(rounds.len(), 1);
    assert_eq!(rounds[0].removed, [2]);
    assert_eq!(recovery.survivors, [0, 1, 3]);
    let first_order = fake.orders().iter().map(|order| order.at).min().unwrap();
    assert!(
        first_order.duration_since(killed_at) >= dead_after,
        "scaled down {:?} after the rank vanished, before dead-after {dead_after:?}",
        first_order.duration_since(killed_at)
    );
    assert_eq!(order_ranks(&fake, &rounds[0].request_id), [0, 1, 3]);
    assert_clean(&fake);
    controller.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unreachable_rank_with_healthy_survivors_is_not_escalated() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let mut config = settings("w-partition");
    config.policy.recovery_deadline = ms(300);
    let mut controller = Running::start(&fake, config, file_store(state.path(), "w-partition"));
    controller.formed().await;
    let escalator = controller.escalator.clone();

    // The controller loses contact with rank 2, which keeps working. No
    // survivor faults, so no recovery deadline runs and nothing escalates,
    // however long it lasts.
    let lost_at = Instant::now();
    fake.partition(2);
    // Three wakes per survivor take 100 + 200 + 400 ms: past dead_after
    // (400 ms) and the 300 ms recovery deadline.
    wait_until(|| [0, 1, 3].iter().all(|&rank| fake.wakes_of(rank) >= 3)).await;
    assert!(lost_at.elapsed() >= ms(700), "{:?}", lost_at.elapsed());
    for rank in [0, 1, 3] {
        let wakes = fake.wakes_of(rank);
        assert!(wakes <= 5, "rank {rank} was woken {wakes} times");
    }
    assert_eq!(fake.wakes_of(2), 0);

    // Contact is back before any round named rank 2: the fault clears.
    fake.heal(2);
    controller.cleared().await;
    assert!(fake.orders().is_empty());
    assert!(escalator.calls.lock().unwrap().is_empty());
    assert_clean(&fake);
    assert_eq!(controller.stop().await, Outcome::Stopped);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ranks_that_never_answer_keep_reminding_while_forming() {
    let fake = FakeWorld::start(3, Timing::default()).await;
    fake.partition(2);
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-forming"),
        file_store(state.path(), "w-forming"),
    );

    // remind_every is 200 ms in the tests.
    for _ in 0..2 {
        let (silent, waited) = controller
            .next_matching(|event| match event {
                Event::StillForming { silent, waited } => Some((silent, waited)),
                Event::Formed => panic!("formed without rank 2"),
                _ => None,
            })
            .await;
        assert_eq!(silent, [2]);
        assert!(waited >= ms(200), "{waited:?}");
    }
    fake.heal(2);
    controller.formed().await;
    assert_eq!(controller.stop().await, Outcome::Stopped);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rank_the_last_round_removed_is_refused_at_start() {
    let fake = FakeWorld::start(3, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let rounds = FileRoundStore::new(state.path(), "w-stale");
    // The world kept running without rank 2 after this round.
    let mut record = RoundRecord {
        world: "w-stale".to_string(),
        epoch: 4,
        seq: 1,
        request_id: "w-stale-e4-r1-0000abcd".to_string(),
        instruction: Instruction::ScaleDown,
        params: json!({"removed_dp_ranks": [2]}),
        removed: vec![2],
        survivors: vec![0, 1],
        store: None,
        sent_at: "2026-11-25T10:00:00.000Z".to_string(),
        not_after: "2026-11-25T10:01:50.000Z".to_string(),
        next_store_port: u32::from(STORE_PORT_BASE),
        escalated: false,
    };
    rounds.save(&record).unwrap();

    let refused = Running::start(
        &fake,
        settings("w-stale"),
        file_store(state.path(), "w-stale"),
    )
    .result()
    .await
    .unwrap_err();
    assert!(refused.is::<ConfigError>(), "{refused:#}");
    let message = refused.to_string();
    assert!(
        message.contains("[2]") && message.contains("w-stale-e4-r1-0000abcd"),
        "{message}"
    );
    assert_eq!(fake.polls(0), 0, "a refused controller polls nothing");
    assert!(!state.path().join("w-stale.epoch.json").exists());

    // After an escalation the supervisor restarts the whole world, so every
    // rank is a member again.
    record.escalated = true;
    rounds.save(&record).unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-stale"),
        file_store(state.path(), "w-stale"),
    );
    controller.formed().await;
    assert_eq!(controller.stop().await, Outcome::Stopped);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dry_run_sends_nothing_writes_nothing_and_keeps_watching() {
    for idle in [false, true] {
        let fake = FakeWorld::start(4, Timing::default()).await;
        fake.set_idle(1, idle);
        let parent = tempfile::tempdir().unwrap();
        let state = parent.path().join("state");
        let mut config = settings("w-dry");
        config.dry_run = true;
        config.policy.recovery_deadline = ms(600);
        let mut controller = Running::start(&fake, config, file_store(&state, "w-dry"));
        controller.formed().await;
        let escalator = controller.escalator.clone();

        // Busy survivors make the controller decide a round; an idle one
        // makes it decide a wake. Either way nothing goes out, and where it
        // would escalate it says so and goes on.
        fake.engine_dies(3);
        let reason = controller.would_escalate().await;
        assert!(reason.contains("deadline"), "idle={idle}: {reason}");

        // The supervisor restarts the world; the dry run watches it again.
        fake.restart_world();
        controller.cleared().await;
        fake.engine_dies(2);
        let again = controller.would_escalate().await;
        assert!(again.contains("deadline"), "idle={idle}: {again}");

        assert_eq!(controller.stop().await, Outcome::Stopped);
        assert!(fake.orders().is_empty(), "idle={idle}");
        assert!(fake.wakes().is_empty(), "idle={idle}");
        assert!(escalator.calls.lock().unwrap().is_empty(), "idle={idle}");
        assert!(!state.exists(), "dry run created the state directory");
        assert_clean(&fake);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_ids_never_repeat_across_incidents_and_restarts() {
    let fake = FakeWorld::start(4, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let mut ids = Vec::new();

    let mut first = Running::start(&fake, settings("w-ids"), file_store(state.path(), "w-ids"));
    first.formed().await;
    fake.engine_dies(3);
    ids.push(first.recovered().await.1.request_id);
    fake.transient_fault();
    ids.push(first.recovered().await.1.request_id);
    assert_eq!(first.stop().await, Outcome::Stopped);

    // A restarted controller (same state directory) starts a new epoch.
    let mut second = Running::start(&fake, settings("w-ids"), file_store(state.path(), "w-ids"));
    second.formed().await;
    fake.transient_fault();
    ids.push(second.recovered().await.1.request_id);
    second.stop().await;

    for (id, prefix) in ids
        .iter()
        .zip(["w-ids-e1-r1-", "w-ids-e1-r2-", "w-ids-e2-r1-"])
    {
        assert!(id.starts_with(prefix), "{id} should start with {prefix}");
    }
    let unique: BTreeSet<&String> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len(), "{ids:?}");
    // The fake also checks every order for a reused id.
    assert_clean(&fake);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn losing_every_rank_escalates() {
    let fake = FakeWorld::start(2, Timing::default()).await;
    let state = tempfile::tempdir().unwrap();
    let mut controller = Running::start(
        &fake,
        settings("w-none"),
        file_store(state.path(), "w-none"),
    );
    controller.formed().await;
    let escalator = controller.escalator.clone();

    fake.engine_dies(0);
    fake.engine_dies(1);
    let Outcome::Escalated(reason) = controller.outcome().await else {
        panic!("expected an escalation");
    };
    assert!(reason.contains("no survivors"), "{reason}");
    assert_eq!(escalator.calls.lock().unwrap().len(), 1);
    assert!(fake.orders().is_empty());
}
