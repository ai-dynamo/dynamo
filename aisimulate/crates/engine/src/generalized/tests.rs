// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::rc::Rc;

use anyhow::{Result, bail};

use super::*;

#[derive(Clone)]
struct FakeConfig {
    ready: Vec<bool>,
    external_wait: Vec<bool>,
    pass_durations_ms: Vec<f64>,
    same_timestamp_retry: Vec<SameTimestampRetry>,
    deadlines_ms: Vec<Option<f64>>,
    fail_execute_rank: Option<u32>,
    fail_complete_rank: Option<u32>,
    fail_internal_rank: Option<u32>,
    log: Rc<RefCell<Vec<String>>>,
}

struct FakeRank {
    identity: RankIdentity,
    ready: bool,
    external_wait: bool,
    drained: bool,
    duration_ms: f64,
    same_timestamp_retry: SameTimestampRetry,
    deadlines_ms: VecDeque<f64>,
    fail_execute: bool,
    fail_complete: bool,
    fail_internal: bool,
    log: Rc<RefCell<Vec<String>>>,
}

impl RankEngine for FakeRank {
    type Config = FakeConfig;
    type Command = &'static str;
    type CommandEffects = bool;
    type PassStartEffects = &'static str;
    type PendingPass = &'static str;
    type PassCompletionEffects = &'static str;
    type InternalEffects = f64;

    fn new(identity: RankIdentity, config: &Self::Config) -> Result<Self> {
        let rank = identity.dp_rank as usize;
        Ok(Self {
            identity,
            ready: config.ready[rank],
            external_wait: config.external_wait[rank],
            drained: !config.ready[rank],
            duration_ms: config.pass_durations_ms[rank],
            same_timestamp_retry: config.same_timestamp_retry[rank],
            deadlines_ms: config.deadlines_ms[rank].into_iter().collect(),
            fail_execute: config.fail_execute_rank == Some(identity.dp_rank),
            fail_complete: config.fail_complete_rank == Some(identity.dp_rank),
            fail_internal: config.fail_internal_rank == Some(identity.dp_rank),
            log: Rc::clone(&config.log),
        })
    }

    fn apply_command_effects(
        &mut self,
        command: Self::Command,
        context: CommandContext,
        _pending_pass: Option<&mut Self::PendingPass>,
    ) -> Result<Self::CommandEffects> {
        self.log.borrow_mut().push(format!(
            "command:{}:{}:{}",
            self.identity.dp_rank, command, context.pass_in_flight
        ));
        if command == "reject" {
            bail!("injected recoverable command rejection");
        }
        if command == "wake" {
            self.ready = true;
            self.drained = false;
        }
        Ok(context.allow_immediate_admission())
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    fn waiting_for_external_command(&self) -> bool {
        self.external_wait
    }

    fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<RankPass<Self::PassStartEffects, Self::PendingPass>> {
        self.log
            .borrow_mut()
            .push(format!("execute:{}", self.identity.dp_rank));
        self.ready = false;
        if self.fail_execute {
            bail!("injected execute failure on rank {}", self.identity.dp_rank);
        }
        Ok(RankPass {
            end_ms: now_ms + self.duration_ms,
            same_timestamp_retry: self.same_timestamp_retry,
            start_effects: "started",
            pending: "pending",
        })
    }

    fn complete_pass(
        &mut self,
        _pending: Self::PendingPass,
        end_ms: f64,
    ) -> Result<Self::PassCompletionEffects> {
        self.log
            .borrow_mut()
            .push(format!("complete:{}:{end_ms}", self.identity.dp_rank));
        self.drained = true;
        if self.fail_complete {
            bail!(
                "injected completion failure on rank {}",
                self.identity.dp_rank
            );
        }
        Ok("completed")
    }

    fn complete_idle_group_pass(
        &mut self,
        _started_at_ms: f64,
        end_ms: f64,
    ) -> Result<Option<Self::PassCompletionEffects>> {
        self.log
            .borrow_mut()
            .push(format!("complete-idle:{}:{end_ms}", self.identity.dp_rank));
        Ok(Some("idle-completed"))
    }

    fn next_internal_deadline_ms(&self) -> Option<f64> {
        self.deadlines_ms.front().copied()
    }

    fn process_internal_work(
        &mut self,
        now_ms: f64,
        pass_in_flight: bool,
    ) -> Result<Self::InternalEffects> {
        self.log.borrow_mut().push(format!(
            "internal:{}:{pass_in_flight}",
            self.identity.dp_rank
        ));
        self.deadlines_ms.pop_front();
        if self.fail_internal {
            bail!(
                "injected internal-work failure on rank {}",
                self.identity.dp_rank
            );
        }
        Ok(now_ms)
    }

    fn is_drained(&self) -> bool {
        self.drained
    }
}

#[derive(Clone)]
struct CancellationConfig {
    outputs_by_rank: Vec<Vec<u64>>,
    pass_durations_ms: Vec<f64>,
}

#[derive(Clone, Copy)]
enum CancellationCommand {
    Cancel(u64),
}

struct CancellationRank {
    ready: bool,
    drained: bool,
    duration_ms: f64,
    computed_outputs: Vec<u64>,
}

#[derive(Clone)]
struct ConvergingConfig {
    hidden_passes: usize,
    observed_times: Rc<RefCell<Vec<f64>>>,
}

struct ConvergingRank {
    hidden_passes: usize,
    ready: bool,
    drained: bool,
    observed_times: Rc<RefCell<Vec<f64>>>,
}

impl RankEngine for ConvergingRank {
    type Config = ConvergingConfig;
    type Command = ();
    type CommandEffects = ();
    type PassStartEffects = bool;
    type PendingPass = bool;
    type PassCompletionEffects = bool;
    type InternalEffects = ();

    fn new(_identity: RankIdentity, config: &Self::Config) -> Result<Self> {
        Ok(Self {
            hidden_passes: config.hidden_passes,
            ready: true,
            drained: false,
            observed_times: Rc::clone(&config.observed_times),
        })
    }

    fn apply_command_effects(
        &mut self,
        _command: Self::Command,
        _context: CommandContext,
        _pending_pass: Option<&mut Self::PendingPass>,
    ) -> Result<Self::CommandEffects> {
        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<RankPass<Self::PassStartEffects, Self::PendingPass>> {
        self.observed_times.borrow_mut().push(now_ms);
        self.ready = false;
        let admitted = self.hidden_passes == 0;
        let same_timestamp_retry = if admitted {
            SameTimestampRetry::NotApplicable
        } else {
            self.hidden_passes -= 1;
            SameTimestampRetry::Retry
        };
        Ok(RankPass {
            end_ms: now_ms,
            same_timestamp_retry,
            start_effects: admitted,
            pending: admitted,
        })
    }

    fn complete_pass(
        &mut self,
        admitted: Self::PendingPass,
        _end_ms: f64,
    ) -> Result<Self::PassCompletionEffects> {
        if admitted {
            self.drained = true;
        } else {
            self.ready = true;
        }
        Ok(admitted)
    }

    fn next_internal_deadline_ms(&self) -> Option<f64> {
        None
    }

    fn process_internal_work(
        &mut self,
        _now_ms: f64,
        _pass_in_flight: bool,
    ) -> Result<Self::InternalEffects> {
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.drained
    }
}

impl RankEngine for CancellationRank {
    type Config = CancellationConfig;
    type Command = CancellationCommand;
    type CommandEffects = bool;
    type PassStartEffects = ();
    type PendingPass = Vec<u64>;
    type PassCompletionEffects = Vec<u64>;
    type InternalEffects = ();

    fn new(identity: RankIdentity, config: &Self::Config) -> Result<Self> {
        let rank = identity.dp_rank as usize;
        Ok(Self {
            ready: true,
            drained: false,
            duration_ms: config.pass_durations_ms[rank],
            computed_outputs: config.outputs_by_rank[rank].clone(),
        })
    }

    fn apply_command_effects(
        &mut self,
        command: Self::Command,
        context: CommandContext,
        pending_pass: Option<&mut Self::PendingPass>,
    ) -> Result<Self::CommandEffects> {
        let CancellationCommand::Cancel(request_id) = command;
        if !context.pass_in_flight {
            return Ok(false);
        }
        let Some(pending_pass) = pending_pass else {
            return Ok(false);
        };
        let len_before = pending_pass.len();
        pending_pass.retain(|output| *output != request_id);
        Ok(pending_pass.len() != len_before)
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<RankPass<Self::PassStartEffects, Self::PendingPass>> {
        self.ready = false;
        Ok(RankPass {
            end_ms: now_ms + self.duration_ms,
            same_timestamp_retry: SameTimestampRetry::NotApplicable,
            start_effects: (),
            pending: self.computed_outputs.clone(),
        })
    }

    fn complete_pass(
        &mut self,
        pending: Self::PendingPass,
        _end_ms: f64,
    ) -> Result<Self::PassCompletionEffects> {
        self.drained = true;
        Ok(pending)
    }

    fn next_internal_deadline_ms(&self) -> Option<f64> {
        None
    }

    fn process_internal_work(
        &mut self,
        _now_ms: f64,
        _pass_in_flight: bool,
    ) -> Result<Self::InternalEffects> {
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.drained
    }
}

fn config(
    ready: Vec<bool>,
    pass_durations_ms: Vec<f64>,
    deadlines_ms: Vec<Option<f64>>,
) -> (FakeConfig, Rc<RefCell<Vec<String>>>) {
    let log = Rc::new(RefCell::new(Vec::new()));
    (
        FakeConfig {
            external_wait: vec![false; ready.len()],
            ready,
            pass_durations_ms,
            same_timestamp_retry: vec![SameTimestampRetry::NotApplicable; deadlines_ms.len()],
            deadlines_ms,
            fail_execute_rank: None,
            fail_complete_rank: None,
            fail_internal_rank: None,
            log: Rc::clone(&log),
        },
        log,
    )
}

#[test]
fn attention_dp_external_wait_requires_every_ready_rank() -> Result<()> {
    let (mut all_waiting, _) = config(
        vec![true, true, false],
        vec![0.0, 0.0, 0.0],
        vec![None, None, None],
    );
    all_waiting.external_wait = vec![true, true, false];
    let engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(30),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(3).unwrap(), all_waiting),
    )?;
    assert!(engine.waiting_for_external_command());
    assert!(!engine.is_ready());

    let (mut mixed, _) = config(
        vec![true, true, false],
        vec![0.0, 0.0, 0.0],
        vec![None, None, None],
    );
    mixed.external_wait = vec![true, false, true];
    let engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(31),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(3).unwrap(), mixed),
    )?;
    assert!(!engine.waiting_for_external_command());
    assert!(engine.is_ready());

    let (mut idle, _) = config(vec![false], vec![0.0], vec![None]);
    idle.external_wait = vec![true];
    let engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(32),
        GeneralizedEngineConfig::single_rank(idle),
    )?;
    assert!(!engine.waiting_for_external_command());
    Ok(())
}

#[test]
fn attention_dp_skips_ready_ranks_waiting_for_external_commands() -> Result<()> {
    let (mut rank, log) = config(vec![true, true], vec![0.0, 5.0], vec![None, None]);
    rank.external_wait = vec![true, false];
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(33),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(2).unwrap(), rank),
    )?;

    let started = engine.execute_pass(10.0)?.expect("rank 1 can run");
    assert_eq!(
        started
            .by_rank
            .iter()
            .map(|rank| rank.dp_rank)
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(log.borrow().as_slice(), &["execute:1"]);

    engine.complete_pass(started.pass_id, started.end_ms)?;
    assert!(!engine.is_ready());
    assert!(engine.waiting_for_external_command());
    assert_eq!(
        log.borrow()
            .iter()
            .filter(|entry| entry.starts_with("execute:"))
            .cloned()
            .collect::<Vec<_>>(),
        vec!["execute:1"]
    );
    Ok(())
}

#[test]
fn single_rank_uses_the_generalized_contract() -> Result<()> {
    let (rank, _) = config(vec![true], vec![7.0], vec![None]);
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(11),
        GeneralizedEngineConfig::single_rank(rank),
    )?;

    assert!(engine.is_ready());
    let started = engine.execute_pass(10.0)?.expect("rank is ready");
    assert_eq!(started.pass_id.get(), 0);
    assert_eq!(started.end_ms, 17.0);
    assert_eq!(started.participating_ranks, NonZeroU32::MIN);
    assert!(!engine.is_ready());
    assert!(!engine.is_drained());

    let completed = engine.complete_pass(started.pass_id, 17.0)?;
    assert_eq!(completed.effects.by_rank.len(), 1);
    assert!(engine.is_drained());
    Ok(())
}

#[test]
fn attention_dp_aggregates_same_timestamp_retry_hints() -> Result<()> {
    let (mut rank, _) = config(
        vec![true, true, true],
        vec![0.0, 0.0, 0.0],
        vec![None, None, None],
    );
    rank.same_timestamp_retry = vec![
        SameTimestampRetry::NotApplicable,
        SameTimestampRetry::Retry,
        SameTimestampRetry::Exhausted,
    ];
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(12),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(3).unwrap(), rank),
    )?;

    let started = engine.execute_pass(10.0)?.expect("ranks are ready");
    assert_eq!(started.same_timestamp_retry, SameTimestampRetry::Retry);
    Ok(())
}

#[test]
fn same_timestamp_retry_contract_reaches_a_visible_effect_without_advancing_time() -> Result<()> {
    let observed_times = Rc::new(RefCell::new(Vec::new()));
    let mut engine = GeneralizedMockerEngine::<ConvergingRank>::new(
        EngineIdentity::new(13),
        GeneralizedEngineConfig::single_rank(ConvergingConfig {
            hidden_passes: 3,
            observed_times: Rc::clone(&observed_times),
        }),
    )?;
    let now_ms = 42.0;
    let mut hidden_passes = 0usize;

    loop {
        let started = engine.execute_pass(now_ms)?.expect("rank is ready");
        assert_eq!(started.started_at_ms, now_ms);
        assert_eq!(started.end_ms, now_ms);
        let admitted = started.by_rank[0].effects;
        let retry = started.same_timestamp_retry;
        let completed = engine.complete_pass(started.pass_id, now_ms)?;
        if admitted {
            assert!(completed.effects.by_rank[0].effects);
            break;
        }
        assert_eq!(retry, SameTimestampRetry::Retry);
        assert!(!completed.effects.by_rank[0].effects);
        hidden_passes += 1;
    }

    assert_eq!(hidden_passes, 3);
    assert_eq!(observed_times.borrow().as_slice(), &[now_ms; 4]);
    assert!(engine.is_drained());
    Ok(())
}

#[test]
fn single_rank_cancellation_suppresses_already_computed_output() -> Result<()> {
    let mut engine = GeneralizedMockerEngine::<CancellationRank>::new(
        EngineIdentity::new(21),
        GeneralizedEngineConfig::single_rank(CancellationConfig {
            outputs_by_rank: vec![vec![100, 101]],
            pass_durations_ms: vec![10.0],
        }),
    )?;

    let started = engine.execute_pass(0.0)?.expect("rank is ready");
    let cancellation = engine.apply_command_effects(
        SchedulerCommand::new(0, CancellationCommand::Cancel(100)),
        1.0,
    )?;
    assert!(cancellation.by_rank[0].effects);

    let completed = engine.complete_pass(started.pass_id, started.end_ms)?;
    assert_eq!(completed.effects.by_rank[0].effects, vec![101]);
    Ok(())
}

#[test]
fn attention_dp_uses_slowest_rank_boundary_and_holds_idle_siblings() -> Result<()> {
    let (rank, log) = config(
        vec![true, true, false],
        vec![5.0, 20.0, 0.0],
        vec![None, None, None],
    );
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(3),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(3).unwrap(), rank),
    )?;

    let started = engine.execute_pass(100.0)?.expect("two ranks are ready");
    assert_eq!(started.end_ms, 120.0);
    assert_eq!(started.participating_ranks.get(), 3);
    assert_eq!(
        started
            .by_rank
            .iter()
            .map(|rank| (rank.dp_rank, rank.rank_end_ms))
            .collect::<Vec<_>>(),
        vec![(0, 105.0), (1, 120.0)]
    );

    let command = engine.apply_command_effects(SchedulerCommand::new(2, "wake"), 101.0)?;
    assert!(!command.by_rank[0].effects);
    assert!(
        !engine.is_ready(),
        "new work on an idle sibling must wait for the group barrier"
    );
    assert!(engine.complete_pass(started.pass_id, 119.0).is_err());
    assert!(
        engine.complete_pass(PassId(99), 120.0).is_err(),
        "a mismatched pass ID must not consume the real pass"
    );

    let completed = engine.complete_pass(started.pass_id, 125.0)?;
    assert_eq!(
        completed
            .effects
            .by_rank
            .iter()
            .map(|rank| rank.dp_rank)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert!(engine.is_ready(), "rank 2 may run after sibling completion");
    assert!(log.borrow().contains(&"command:2:wake:true".to_string()));
    assert!(log.borrow().contains(&"complete:0:125".to_string()));
    assert!(log.borrow().contains(&"complete:1:125".to_string()));
    assert!(log.borrow().contains(&"complete-idle:2:120".to_string()));
    Ok(())
}

#[test]
fn attention_dp_cancellation_mutates_only_the_target_ranks_retained_pass() -> Result<()> {
    let mut engine = GeneralizedMockerEngine::<CancellationRank>::new(
        EngineIdentity::new(22),
        GeneralizedEngineConfig::attention_dp(
            NonZeroU32::new(2).unwrap(),
            CancellationConfig {
                outputs_by_rank: vec![vec![200, 201], vec![300]],
                pass_durations_ms: vec![5.0, 10.0],
            },
        ),
    )?;

    let started = engine.execute_pass(0.0)?.expect("both ranks are ready");
    let cancellation = engine.apply_command_effects(
        SchedulerCommand::new(0, CancellationCommand::Cancel(200)),
        1.0,
    )?;
    assert!(cancellation.by_rank[0].effects);

    let completed = engine.complete_pass(started.pass_id, started.end_ms)?;
    assert_eq!(
        completed
            .effects
            .by_rank
            .iter()
            .map(|rank| (rank.dp_rank, rank.effects.clone()))
            .collect::<Vec<_>>(),
        vec![(0, vec![201]), (1, vec![300])]
    );
    Ok(())
}

#[test]
fn internal_work_uses_earliest_deadline_and_group_busy_context() -> Result<()> {
    let (rank, log) = config(
        vec![true, false],
        vec![10.0, 0.0],
        vec![Some(8.0), Some(5.0)],
    );
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(7),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(2).unwrap(), rank),
    )?;
    assert_eq!(engine.next_internal_deadline_ms(), Some(5.0));

    let started = engine.execute_pass(0.0)?.expect("rank 0 is ready");
    let effects = engine.process_internal_work(5.0)?;
    assert_eq!(effects.by_rank.len(), 1);
    assert_eq!(effects.by_rank[0].dp_rank, 1);
    assert_eq!(engine.next_internal_deadline_ms(), Some(8.0));
    assert!(log.borrow().contains(&"internal:1:true".to_string()));

    engine.complete_pass(started.pass_id, 10.0)?;
    let effects = engine.process_internal_work(10.0)?;
    assert_eq!(effects.by_rank[0].dp_rank, 0);
    assert!(log.borrow().contains(&"internal:0:false".to_string()));
    Ok(())
}

#[test]
fn execute_failure_after_an_earlier_rank_commits_poisons_the_engine() -> Result<()> {
    let (mut rank, log) = config(
        vec![true, true, true],
        vec![1.0, 1.0, 1.0],
        vec![None, None, Some(10.0)],
    );
    rank.fail_execute_rank = Some(1);
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(40),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(3).unwrap(), rank),
    )?;

    let error = engine.execute_pass(0.0).unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("now poisoned"), "{error}");
    assert!(
        error.contains("injected execute failure on rank 1"),
        "{error}"
    );
    assert_eq!(
        log.borrow().as_slice(),
        &["execute:0".to_string(), "execute:1".to_string()]
    );

    assert!(!engine.is_ready());
    assert!(!engine.waiting_for_external_command());
    assert_eq!(engine.next_internal_deadline_ms(), None);
    assert!(!engine.is_drained());

    let error = engine
        .apply_command_effects(SchedulerCommand::new(2, "wake"), 1.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("is poisoned"), "{error}");
    let error = engine.execute_pass(1.0).unwrap_err().to_string();
    assert!(error.contains("is poisoned"), "{error}");
    let error = engine.process_internal_work(10.0).unwrap_err().to_string();
    assert!(error.contains("is poisoned"), "{error}");
    let error = engine
        .complete_pass(PassId(0), 1.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("is poisoned"), "{error}");
    Ok(())
}

#[test]
fn completion_failure_after_an_earlier_rank_completes_poisons_the_engine() -> Result<()> {
    let (mut rank, log) = config(vec![true, true], vec![1.0, 2.0], vec![None, None]);
    rank.fail_complete_rank = Some(1);
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(41),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(2).unwrap(), rank),
    )?;

    let started = engine.execute_pass(0.0)?.expect("both ranks are ready");
    let error = engine
        .complete_pass(started.pass_id, started.end_ms)
        .unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("now poisoned"), "{error}");
    assert!(
        error.contains("injected completion failure on rank 1"),
        "{error}"
    );
    assert!(log.borrow().contains(&"complete:0:2".to_string()));
    assert!(log.borrow().contains(&"complete:1:2".to_string()));

    // Both fake ranks marked themselves drained before rank 1 returned its
    // error, so only the generalized engine's poison state prevents a false
    // successful-drain result here.
    assert!(!engine.is_drained());
    assert!(!engine.is_ready());
    let error = engine
        .complete_pass(started.pass_id, started.end_ms)
        .unwrap_err()
        .to_string();
    assert!(error.contains("is poisoned"), "{error}");
    Ok(())
}

#[test]
fn internal_work_failure_poisons_after_due_ranks_may_mutate() -> Result<()> {
    let (mut rank, log) = config(
        vec![false, false],
        vec![0.0, 0.0],
        vec![Some(0.0), Some(0.0)],
    );
    rank.fail_internal_rank = Some(1);
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(42),
        GeneralizedEngineConfig::attention_dp(NonZeroU32::new(2).unwrap(), rank),
    )?;

    let error = engine.process_internal_work(0.0).unwrap_err();
    let error = format!("{error:#}");
    assert!(error.contains("now poisoned"), "{error}");
    assert!(
        error.contains("injected internal-work failure on rank 1"),
        "{error}"
    );
    assert_eq!(
        log.borrow().as_slice(),
        &[
            "internal:0:false".to_string(),
            "internal:1:false".to_string()
        ]
    );
    assert!(!engine.is_drained());
    Ok(())
}

#[test]
fn targeted_command_rejection_does_not_poison_the_group() -> Result<()> {
    let (rank, _) = config(vec![false], vec![0.0], vec![None]);
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(43),
        GeneralizedEngineConfig::single_rank(rank),
    )?;

    let error = engine
        .apply_command_effects(SchedulerCommand::new(0, "reject"), 0.0)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("injected recoverable command rejection"),
        "{error:#}"
    );

    engine.apply_command_effects(SchedulerCommand::new(0, "wake"), 0.0)?;
    let started = engine
        .execute_pass(0.0)?
        .expect("valid work remains executable after a command rejection");
    engine.complete_pass(started.pass_id, started.end_ms)?;
    assert!(engine.is_drained());
    Ok(())
}

#[test]
fn rejects_unknown_ranks_and_non_finite_times() -> Result<()> {
    let (rank, _) = config(vec![false], vec![0.0], vec![None]);
    let mut engine = GeneralizedMockerEngine::<FakeRank>::new(
        EngineIdentity::new(1),
        GeneralizedEngineConfig::single_rank(rank),
    )?;

    assert!(
        engine
            .apply_command_effects(SchedulerCommand::new(1, "wake"), 0.0)
            .is_err()
    );
    assert!(engine.execute_pass(f64::NAN).is_err());
    assert!(engine.execute_pass(-1.0).is_err());
    assert!(engine.process_internal_work(f64::INFINITY).is_err());
    assert!(engine.process_internal_work(-1.0).is_err());

    // Input validation happens before rank mutation and must not poison an
    // otherwise healthy engine.
    engine.apply_command_effects(SchedulerCommand::new(0, "wake"), 0.0)?;
    let started = engine
        .execute_pass(0.0)?
        .expect("valid work remains executable after validation failures");
    engine.complete_pass(started.pass_id, started.end_ms)?;
    assert!(engine.is_drained());
    Ok(())
}
