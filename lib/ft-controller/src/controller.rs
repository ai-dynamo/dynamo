// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The controller loop. It polls every rank, feeds the [decision
//! core](crate::decision), and carries out its decisions through
//! [`EngineFt`], [`RoundStore`] and [`Escalator`].
//!
//! Phases: *forming* (until every rank has answered once), *watching* (every
//! rank healthy), *deciding* (a fault is suspected), *in round* (a round was
//! sent and is being judged) and, in a dry run only, *parked* (after it would
//! have escalated, until every rank is healthy again). A recovered round, or
//! a fault that clears before any round, goes back to watching; a failed
//! round goes back to deciding; anything fatal escalates and ends the loop.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::Future;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use futures::future::join_all;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;
use tracing::{Instrument, debug, error, info, info_span, warn};

use crate::decision::{
    self, DeadCause, Decision, Incident, Observation, Policy, SentRound, StoreAddr, StoreState,
    Verdict, describe, secs, single_line,
};
use crate::engine::{ApplyError, EngineFt};
use crate::escalate::Escalator;
use crate::protocol::{ApplyRequest, Instruction, round_params};
use crate::round::{RoundRecord, RoundStore, request_id, timestamp};

/// Exit code of the binary after an escalation.
pub const ESCALATED_EXIT_CODE: u8 = 3;

/// Longest value any duration setting may take. Anything longer is a typo,
/// and the bound keeps every time computation far from overflowing.
pub const MAX_DURATION: Duration = Duration::from_secs(24 * 60 * 60);

/// The world contradicts the controller's configuration: for example a front
/// end serves another rank's engine, or a `--rank` names a rank the last
/// round removed. Restarting the world would not help; the configuration
/// must be fixed. The binary exits with code 2.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(pub String);

/// Controller settings.
#[derive(Clone, Debug)]
pub struct Settings {
    pub world: String,
    pub poll_interval: Duration,
    pub policy: Policy,
    /// Give up on one round after this. `None`: a round may run until the
    /// incident's recovery deadline.
    pub round_timeout: Option<Duration>,
    /// Send `{"abort": true}` with retry rounds. vLLM 0.34.0 ignores it.
    pub retry_abort: bool,
    /// Port of the first store rebuild; each rebuild takes the next one.
    pub store_port_base: u16,
    /// Log decisions only: send no wake or order, run no escalation command
    /// and write no state.
    pub dry_run: bool,
    /// How often a standing problem is logged again: ranks that have not
    /// answered while forming, and ranks counted dead while every survivor
    /// reports healthy.
    pub remind_every: Duration,
}

impl Settings {
    /// Checks the settings an embedding host passes in, as the command line
    /// checks its flags.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let durations = [
            ("poll_interval", Some(self.poll_interval)),
            ("policy.dead_after", Some(self.policy.dead_after)),
            ("policy.wake_max_gap", Some(self.policy.wake_max_gap)),
            ("policy.checkin_timeout", Some(self.policy.checkin_timeout)),
            (
                "policy.recovery_deadline",
                Some(self.policy.recovery_deadline),
            ),
            ("round_timeout", self.round_timeout),
            ("remind_every", Some(self.remind_every)),
        ];
        for (name, value) in durations {
            if let Some(value) = value
                && (value.is_zero() || value > MAX_DURATION)
            {
                return Err(ConfigError(format!(
                    "{name} must be positive and at most {}",
                    secs(MAX_DURATION)
                )));
            }
        }
        if self.policy.wake_after > MAX_DURATION {
            return Err(ConfigError(format!(
                "policy.wake_after must be at most {}",
                secs(MAX_DURATION)
            )));
        }
        if self.policy.max_rounds == 0 {
            return Err(ConfigError(
                "policy.max_rounds must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// One member of the world.
#[derive(Clone)]
pub struct Rank {
    pub engine: Arc<dyn EngineFt>,
    /// Address this rank would host the coordination store on
    /// (`dp_master_ip`).
    pub store_host: String,
}

/// What the controller did, for tests and embedding hosts.
#[derive(Clone, Debug)]
pub enum Event {
    /// Every rank answered once; the controller now watches for faults.
    Formed,
    /// Some ranks have still not answered. Repeated every `remind_every`,
    /// starting `remind_every` after the controller started.
    StillForming { silent: Vec<u32>, waited: Duration },
    /// Wakes were sent to these ranks.
    Woke(Vec<u32>),
    /// Every rank is healthy again without a round: the fault cleared by
    /// itself or, in a dry run, the world was restarted. The controller
    /// watches again.
    Cleared,
    /// A round was recorded and sent.
    RoundSent(RoundRecord),
    /// A round recovered the world.
    Recovered(Recovery),
    /// Dry run only: the controller would have escalated for this reason.
    /// It goes on observing.
    WouldEscalate(String),
}

/// Summary of one recovered incident. Its `Display` is the
/// `FT_CONTROLLER_RECOVERED` line the binary prints on stdout.
#[derive(Clone, Debug, PartialEq)]
pub struct Recovery {
    pub world: String,
    pub instruction: Instruction,
    pub removed: Vec<u32>,
    pub survivors: Vec<u32>,
    pub rounds: u32,
    pub request_id: String,
    /// From the first fault evidence to sending the first round.
    pub detect: Duration,
    /// From sending the successful round to every survivor healthy.
    pub round: Duration,
    /// From the first fault evidence to every survivor healthy.
    pub total: Duration,
}

impl fmt::Display for Recovery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FT_CONTROLLER_RECOVERED world={} instruction={} removed={} survivors={} rounds={} detect_s={:.3} round_s={:.3} total_s={:.3} request_id={}",
            self.world,
            self.instruction,
            rank_list(&self.removed),
            rank_list(&self.survivors),
            self.rounds,
            self.detect.as_secs_f64(),
            self.round.as_secs_f64(),
            self.total.as_secs_f64(),
            self.request_id,
        )
    }
}

fn rank_list(ranks: &[u32]) -> String {
    let ranks: Vec<String> = ranks.iter().map(u32::to_string).collect();
    format!("[{}]", ranks.join(","))
}

/// Why [`Controller::run`] returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Asked to stop.
    Stopped,
    /// Recovery through the FT API was impossible or unsafe; the supervisor
    /// must restart the world. The reason is one line. A dry run never
    /// returns this.
    Escalated(String),
}

/// The world controller.
pub struct Controller {
    settings: Settings,
    ranks: BTreeMap<u32, Rank>,
    rounds: Arc<dyn RoundStore>,
    escalator: Arc<dyn Escalator>,
    events: Option<UnboundedSender<Event>>,
}

impl Controller {
    pub fn new(
        settings: Settings,
        ranks: BTreeMap<u32, Rank>,
        rounds: Arc<dyn RoundStore>,
        escalator: Arc<dyn Escalator>,
    ) -> Result<Self, ConfigError> {
        if ranks.is_empty() {
            return Err(ConfigError("a world needs at least one rank".to_string()));
        }
        settings.validate()?;
        Ok(Self {
            settings,
            ranks,
            rounds,
            escalator,
            events: None,
        })
    }

    /// Reports what the controller does on `events`.
    pub fn with_events(mut self, events: UnboundedSender<Event>) -> Self {
        self.events = Some(events);
        self
    }

    /// Runs until `shutdown` completes (returns [`Outcome::Stopped`]) or the
    /// controller escalates. Errors are failures of the controller itself,
    /// such as an unreadable state directory, or a [`ConfigError`].
    pub async fn run(self, shutdown: impl Future<Output = ()>) -> Result<Outcome> {
        let span = info_span!("ft_controller", world = %self.settings.world);
        self.run_in_span(shutdown).instrument(span).await
    }

    async fn run_in_span(self, shutdown: impl Future<Output = ()>) -> Result<Outcome> {
        let last = self.rounds.last()?;
        let store = self.restore_store(last.as_ref())?;
        let epoch = if self.settings.dry_run {
            // A dry run sends no request ids, so its epoch is only a label.
            last.as_ref()
                .map_or(1, |record| record.epoch.saturating_add(1))
        } else {
            self.rounds.begin_epoch()?
        };
        info!(
            epoch,
            ranks = ?self.ranks.keys().collect::<Vec<_>>(),
            store_rebuilt = store.rebuilt,
            next_store_port = store.next_port,
            dry_run = self.settings.dry_run,
            "controller started"
        );

        let started = Instant::now();
        let poll_interval = self.settings.poll_interval;
        let remind_every = self.settings.remind_every;
        let mut runner = Runner {
            settings: self.settings,
            members: self.ranks,
            rounds: self.rounds,
            escalator: self.escalator,
            events: self.events,
            epoch,
            seq: 0,
            store,
            observations: BTreeMap::new(),
            answered: BTreeSet::new(),
            phase: Phase::Forming,
            started,
            forming_note: None,
            forming_reminder: Reminder::after(started, remind_every),
            wakes: JoinSet::new(),
        };

        tokio::pin!(shutdown);
        let mut ticker = tokio::time::interval(poll_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                () = &mut shutdown => {
                    info!("stopping on request");
                    return Ok(Outcome::Stopped);
                }
                _ = ticker.tick() => {}
            }
            if let ControlFlow::Break(outcome) = runner.step().await? {
                return Ok(outcome);
            }
        }
    }

    /// The store state a restarted controller continues with, after checking
    /// the membership against the last round record.
    fn restore_store(&self, last: Option<&RoundRecord>) -> Result<StoreState, ConfigError> {
        let base = u32::from(self.settings.store_port_base);
        let mut store = StoreState {
            rebuilt: false,
            next_port: base,
            hosts: self
                .ranks
                .iter()
                .map(|(&rank, member)| (rank, member.store_host.clone()))
                .collect(),
        };
        let Some(record) = last else {
            return Ok(store);
        };
        // Store ports never repeat while the round record is kept: a store
        // of a failed round may still hold its port.
        let after_last = record
            .store
            .as_ref()
            .map_or(0, |addr| u32::from(addr.port) + 1);
        store.next_port = base.max(record.next_store_port).max(after_last);
        if record.escalated {
            // The supervisor restarted the whole world after that round: it
            // runs with all its ranks and its original store again.
            return Ok(store);
        }
        let gone: Vec<u32> = self
            .ranks
            .keys()
            .copied()
            .filter(|rank| !record.survivors.contains(rank))
            .collect();
        if !gone.is_empty() {
            return Err(ConfigError(format!(
                "--rank lists ranks {gone:?}, which are not members of world {} any more: its last round {} (sent {}) left survivors {:?}. \
                 Drop them from --rank. If the world was restarted with all its ranks since, delete its round record ({}.round.json in --state-dir) first",
                record.world, record.request_id, record.sent_at, record.survivors, record.world
            )));
        }
        // After a store round, every round must carry a store address.
        store.rebuilt = record.store.is_some();
        Ok(store)
    }
}

enum Phase {
    /// Waiting for every rank to answer once.
    Forming,
    /// Every rank healthy.
    Watching,
    /// A fault is suspected and no round is in flight.
    Deciding(Episode),
    /// A round was sent and is being judged.
    InRound(Episode, Box<InFlight>),
    /// Dry run only: the controller would have escalated. It waits for every
    /// rank to be healthy again (say, after the supervisor restarted the
    /// world), then watches again.
    Parked,
}

/// Rate limit of a log line about a standing problem.
#[derive(Clone, Copy, Debug)]
struct Reminder {
    every: Duration,
    last: Option<Instant>,
}

impl Reminder {
    /// Due at once, then every `every`.
    fn immediate(every: Duration) -> Self {
        Self { every, last: None }
    }

    /// First due `every` after `start`, then every `every`.
    fn after(start: Instant, every: Duration) -> Self {
        Self {
            every,
            last: Some(start),
        }
    }

    fn due(&mut self, now: Instant) -> bool {
        if let Some(last) = self.last
            && now.saturating_duration_since(last) < self.every
        {
            return false;
        }
        self.last = Some(now);
        true
    }
}

/// One incident as the controller tracks it.
struct Episode {
    incident: Incident,
    first_round_at: Option<Instant>,
    /// The last wait reason logged, so each is logged once.
    last_note: Option<String>,
    /// Rate limit of the error logged while ranks count as dead but every
    /// survivor still reports healthy.
    stalled: Reminder,
}

impl Episode {
    fn new(started: Instant, remind_every: Duration) -> Self {
        Self {
            incident: Incident::new(started),
            first_round_at: None,
            last_note: None,
            stalled: Reminder::immediate(remind_every),
        }
    }

    fn elapsed(&self, now: Instant) -> String {
        secs(now.saturating_duration_since(self.incident.started()))
    }

    fn note(&mut self, text: String, now: Instant) {
        if self.last_note.as_deref() != Some(text.as_str()) {
            info!(phase = "decide", elapsed = %self.elapsed(now), "{text}");
            self.last_note = Some(text);
        }
    }

    /// Logs an error, at a bounded rate, while ranks count as dead but every
    /// survivor still reports healthy, once the incident has lasted `grace`:
    /// the controller lost contact with a rank, or the survivors never notice
    /// a dead peer. Before that this is the normal start of every incident
    /// (vLLM survivors notice a dead peer only after a collective timeout,
    /// 30 s in 0.34.0), so it is not an error. Returns whether it logged.
    fn alarm_if_stalled(&mut self, now: Instant, grace: Duration) -> bool {
        if self.incident.faulted_at().is_some()
            || self.incident.dead().is_empty()
            || now.saturating_duration_since(self.incident.started()) < grace
            || !self.stalled.due(now)
        {
            return false;
        }
        let dead: Vec<u32> = self.incident.dead().keys().copied().collect();
        error!(
            phase = "detect",
            dead = ?dead,
            elapsed = %self.elapsed(now),
            "ranks count as dead but every survivor still reports healthy; acting only once a survivor faults, when vLLM's recovery clock and the recovery deadline start"
        );
        true
    }
}

/// A round being judged.
struct InFlight {
    record: RoundRecord,
    survivors: BTreeSet<u32>,
    /// Survivors whose front end refused the order (HTTP 4xx).
    refused: BTreeMap<u32, String>,
    sent_at: Instant,
    timeout: Duration,
}

struct Runner {
    settings: Settings,
    members: BTreeMap<u32, Rank>,
    rounds: Arc<dyn RoundStore>,
    escalator: Arc<dyn Escalator>,
    events: Option<UnboundedSender<Event>>,
    epoch: u64,
    seq: u64,
    store: StoreState,
    /// Latest observation of every member polled so far.
    observations: BTreeMap<u32, Observation>,
    /// Members that answered at least one status poll.
    answered: BTreeSet<u32>,
    phase: Phase,
    /// When the controller started.
    started: Instant,
    forming_note: Option<Vec<u32>>,
    forming_reminder: Reminder,
    wakes: JoinSet<()>,
}

impl Runner {
    async fn step(&mut self) -> Result<ControlFlow<Outcome>, ConfigError> {
        self.reap_wakes();
        self.poll().await?;
        let now = Instant::now();
        let phase = std::mem::replace(&mut self.phase, Phase::Forming);
        let next = match phase {
            Phase::Forming => ControlFlow::Continue(self.forming(now)),
            Phase::Watching => self.watching(now).await,
            Phase::Deciding(episode) => self.deciding(episode, now).await,
            Phase::InRound(episode, round) => self.in_round(episode, *round, now).await,
            Phase::Parked => ControlFlow::Continue(self.parked()),
        };
        Ok(match next {
            ControlFlow::Continue(phase) => {
                self.phase = phase;
                ControlFlow::Continue(())
            }
            ControlFlow::Break(outcome) => ControlFlow::Break(outcome),
        })
    }

    /// Polls every member, or only the survivors while a round is in flight.
    /// A front end that serves another rank's engine is a configuration
    /// error.
    async fn poll(&mut self) -> Result<(), ConfigError> {
        let targets: Vec<u32> = match &self.phase {
            Phase::InRound(_, round) => round.survivors.iter().copied().collect(),
            _ => self.members.keys().copied().collect(),
        };
        let started = Instant::now();
        let polls = targets.iter().filter_map(|&rank| {
            let engine = self.members.get(&rank)?.engine.clone();
            Some(async move { (rank, engine.status().await) })
        });
        for (rank, result) in join_all(polls).await {
            let observation = match result {
                Ok(report) => {
                    decision::check_identity(rank, &report).map_err(ConfigError)?;
                    self.answered.insert(rank);
                    Observation::Report(report)
                }
                Err(error) => {
                    let since = match self.observations.get(&rank) {
                        Some(Observation::Unreachable { since, .. }) => *since,
                        _ => started,
                    };
                    Observation::Unreachable {
                        since,
                        error: format!("{error:#}"),
                    }
                }
            };
            let state = state_of(&observation);
            match self.observations.get(&rank).map(state_of) {
                Some(before) if before == state => {}
                before => info!(
                    rank,
                    from = before.as_deref().unwrap_or("unknown"),
                    to = %state,
                    detail = %describe(&observation),
                    "rank state changed"
                ),
            }
            self.observations.insert(rank, observation);
        }
        Ok(())
    }

    fn forming(&mut self, now: Instant) -> Phase {
        let silent: Vec<u32> = self
            .members
            .keys()
            .copied()
            .filter(|rank| !self.answered.contains(rank))
            .collect();
        if silent.is_empty() {
            info!(phase = "watch", "every rank answered; watching the world");
            self.emit(Event::Formed);
            return Phase::Watching;
        }
        if self.forming_note.as_ref() != Some(&silent) {
            info!(phase = "forming", ranks = ?silent, "waiting for ranks to answer");
            self.forming_note = Some(silent.clone());
        }
        if self.forming_reminder.due(now) {
            let waited = now.saturating_duration_since(self.started);
            warn!(
                phase = "forming",
                ranks = ?silent,
                waited = %secs(waited),
                "ranks still have not answered; the world is not protected until every rank has answered"
            );
            self.emit(Event::StillForming { silent, waited });
        }
        Phase::Forming
    }

    async fn watching(&mut self, now: Instant) -> ControlFlow<Outcome, Phase> {
        if decision::all_healthy(self.observations.values()) {
            return ControlFlow::Continue(Phase::Watching);
        }
        let faults: Vec<String> = self
            .observations
            .iter()
            .filter(|(_, observation)| !observation.is_settled_healthy())
            .map(|(rank, observation)| format!("{rank}: {}", describe(observation)))
            .collect();
        warn!(phase = "detect", faults = %faults.join("; "), "fault suspected");
        // The incident starts at the earliest evidence: a failed poll started
        // before it timed out.
        let started = self
            .observations
            .values()
            .filter_map(|observation| match observation {
                Observation::Unreachable { since, .. } => Some(*since),
                Observation::Report(_) => None,
            })
            .fold(now, Instant::min);
        let episode = Episode::new(started, self.settings.remind_every);
        self.deciding(episode, now).await
    }

    async fn deciding(
        &mut self,
        mut episode: Episode,
        now: Instant,
    ) -> ControlFlow<Outcome, Phase> {
        let dead_before = episode.incident.dead().clone();
        let next = decision::decide(
            &self.observations,
            &self.store,
            &mut episode.incident,
            &self.settings.policy,
            now,
        );
        self.log_dead_marks(&dead_before, &episode.incident);
        // A stall that outlasts a whole recovery's budget is no longer the
        // normal wait for the survivors' collective timeouts.
        let stall_grace = self.settings.policy.recovery_deadline;
        match next {
            Decision::Wait(why) => {
                episode.alarm_if_stalled(now, stall_grace);
                episode.note(why, now);
                ControlFlow::Continue(Phase::Deciding(episode))
            }
            Decision::Wake(ranks) => {
                episode.alarm_if_stalled(now, stall_grace);
                self.wake(&ranks, &episode, now);
                episode.incident.record_wakes(&ranks, now);
                ControlFlow::Continue(Phase::Deciding(episode))
            }
            Decision::Cleared => {
                info!(
                    phase = "detect",
                    elapsed = %episode.elapsed(now),
                    "fault cleared before any round; watching the world again"
                );
                self.emit(Event::Cleared);
                ControlFlow::Continue(Phase::Watching)
            }
            Decision::Retry { store } => {
                self.start_round(episode, Instruction::Retry, Vec::new(), store, now)
                    .await
            }
            Decision::ScaleDown { removed, store } => {
                self.start_round(episode, Instruction::ScaleDown, removed, store, now)
                    .await
            }
            Decision::Escalate(reason) => self.escalate(reason).await,
        }
    }

    /// Logs every rank that starts or stops counting as dead.
    fn log_dead_marks(&self, before: &BTreeMap<u32, DeadCause>, incident: &Incident) {
        for (&rank, cause) in incident.dead() {
            if before.get(&rank) != Some(cause) {
                let detail = self
                    .observations
                    .get(&rank)
                    .map(describe)
                    .unwrap_or_default();
                warn!(phase = "detect", rank, cause = ?cause, %detail, "rank counts as dead");
            }
        }
        for &rank in before.keys() {
            if !incident.dead().contains_key(&rank) {
                warn!(
                    phase = "detect",
                    rank, "rank answers again before any round named it; it is a member again"
                );
            }
        }
    }

    fn wake(&mut self, ranks: &[u32], episode: &Episode, now: Instant) {
        if self.settings.dry_run {
            info!(phase = "wake", ranks = ?ranks, elapsed = %episode.elapsed(now), "dry run: would wake idle survivors");
            return;
        }
        info!(
            phase = "wake",
            ranks = ?ranks,
            elapsed = %episode.elapsed(now),
            "waking idle survivors so they notice the fault"
        );
        for &rank in ranks {
            let Some(member) = self.members.get(&rank) else {
                continue;
            };
            let engine = member.engine.clone();
            self.wakes.spawn(
                async move {
                    if let Err(error) = engine.wake().await {
                        warn!(rank, error = %format!("{error:#}"), "wake request failed");
                    }
                }
                .in_current_span(),
            );
        }
        self.emit(Event::Woke(ranks.to_vec()));
    }

    async fn start_round(
        &mut self,
        mut episode: Episode,
        instruction: Instruction,
        removed: Vec<u32>,
        store: Option<StoreAddr>,
        now: Instant,
    ) -> ControlFlow<Outcome, Phase> {
        let survivors: BTreeSet<u32> = self
            .members
            .keys()
            .copied()
            .filter(|rank| !removed.contains(rank))
            .collect();
        let params = round_params(
            instruction,
            &removed,
            store.as_ref(),
            self.settings.retry_abort,
        );
        if self.settings.dry_run {
            episode.note(
                format!(
                    "dry run: would send {instruction} to ranks {survivors:?} with params {params}"
                ),
                now,
            );
            return ControlFlow::Continue(Phase::Deciding(episode));
        }

        self.seq += 1;
        let request_id = request_id(&self.settings.world, self.epoch, self.seq);
        // A round may run until the recovery deadline, which runs from the
        // first survivor fault, as vLLM's own recovery clock does.
        let faulted_at = episode.incident.faulted_at().unwrap_or(now);
        let left = self
            .settings
            .policy
            .recovery_deadline
            .saturating_sub(now.saturating_duration_since(faulted_at));
        let timeout = self
            .settings
            .round_timeout
            .map_or(left, |limit| limit.min(left));
        let next_store_port = if store.is_some() {
            self.store.next_port.saturating_add(1)
        } else {
            self.store.next_port
        };
        let wall = SystemTime::now();
        let Some(not_after) = wall.checked_add(timeout) else {
            let reason = format!(
                "cannot record round {request_id}, so it was not sent: its deadline is out of the clock's range"
            );
            return self.escalate(reason).await;
        };
        let record = RoundRecord {
            world: self.settings.world.clone(),
            epoch: self.epoch,
            seq: self.seq,
            request_id: request_id.clone(),
            instruction,
            params: params.clone(),
            removed: removed.clone(),
            survivors: survivors.iter().copied().collect(),
            store: store.clone(),
            sent_at: timestamp(wall),
            not_after: timestamp(not_after),
            next_store_port,
            escalated: false,
        };
        if let Err(error) = self.rounds.save(&record) {
            let reason = format!("cannot record round {request_id}, so it was not sent: {error:#}");
            return self.escalate(reason).await;
        }
        if store.is_some() {
            self.store.rebuilt = true;
        }
        self.store.next_port = next_store_port;
        episode.incident.record_round(survivors.iter().copied());
        episode.first_round_at.get_or_insert(now);
        warn!(
            phase = "round",
            %request_id,
            %instruction,
            removed = ?removed,
            survivors = ?survivors,
            store = ?store,
            round = episode.incident.rounds_sent(),
            elapsed = %episode.elapsed(now),
            timeout = %secs(timeout),
            "sending recovery round"
        );

        // Each survivor gets the order exactly once, all at the same time.
        let order = ApplyRequest {
            instruction,
            params,
            request_id: request_id.clone(),
        };
        let order = &order;
        let sends = survivors.iter().filter_map(|&rank| {
            let engine = self.members.get(&rank)?.engine.clone();
            Some(async move { (rank, engine.apply(order).await) })
        });
        let mut refused = BTreeMap::new();
        for (rank, answer) in join_all(sends).await {
            match answer {
                Ok(()) => debug!(rank, %request_id, "order accepted"),
                Err(error @ ApplyError::Refused { .. }) => {
                    warn!(rank, %request_id, %error, "order refused");
                    refused.insert(rank, error.to_string());
                }
                Err(error) => warn!(
                    rank,
                    %request_id,
                    %error,
                    "order may not have arrived; it is not resent"
                ),
            }
        }
        self.emit(Event::RoundSent(record.clone()));
        ControlFlow::Continue(Phase::InRound(
            episode,
            Box::new(InFlight {
                record,
                survivors,
                refused,
                sent_at: now,
                timeout,
            }),
        ))
    }

    async fn in_round(
        &mut self,
        episode: Episode,
        round: InFlight,
        now: Instant,
    ) -> ControlFlow<Outcome, Phase> {
        let sent = SentRound {
            request_id: &round.record.request_id,
            survivors: &round.survivors,
            refused: &round.refused,
            sent_at: round.sent_at,
            timeout: round.timeout,
        };
        match decision::judge_round(&sent, &self.observations, &self.settings.policy, now) {
            Verdict::Pending => ControlFlow::Continue(Phase::InRound(episode, Box::new(round))),
            Verdict::Recovered => {
                self.recovered(&episode, &round, now);
                ControlFlow::Continue(Phase::Watching)
            }
            Verdict::Failed(why) => {
                warn!(
                    phase = "round",
                    request_id = %round.record.request_id,
                    elapsed = %episode.elapsed(now),
                    "{why}; deciding again"
                );
                ControlFlow::Continue(Phase::Deciding(episode))
            }
            Verdict::Fatal(why) => self.escalate(why).await,
        }
    }

    fn recovered(&mut self, episode: &Episode, round: &InFlight, now: Instant) {
        let record = &round.record;
        for rank in &record.removed {
            self.members.remove(rank);
            self.observations.remove(rank);
            self.answered.remove(rank);
            self.store.hosts.remove(rank);
        }
        let started = episode.incident.started();
        let recovery = Recovery {
            world: self.settings.world.clone(),
            instruction: record.instruction,
            removed: record.removed.clone(),
            survivors: record.survivors.clone(),
            rounds: episode.incident.rounds_sent(),
            request_id: record.request_id.clone(),
            detect: episode
                .first_round_at
                .unwrap_or(round.sent_at)
                .saturating_duration_since(started),
            round: now.saturating_duration_since(round.sent_at),
            total: now.saturating_duration_since(started),
        };
        info!(
            phase = "recovered",
            request_id = %recovery.request_id,
            instruction = %recovery.instruction,
            removed = ?recovery.removed,
            rounds = recovery.rounds,
            detect = %secs(recovery.detect),
            round = %secs(recovery.round),
            total = %secs(recovery.total),
            "world recovered"
        );
        self.emit(Event::Recovered(recovery));
    }

    /// Fails closed: logs loudly, marks the last round record (the
    /// supervisor restarts the whole world), runs the escalator and ends the
    /// loop. A dry run only reports what it would do and parks.
    async fn escalate(&mut self, reason: String) -> ControlFlow<Outcome, Phase> {
        let reason = single_line(&reason);
        if self.settings.dry_run {
            warn!(
                phase = "escalate",
                %reason,
                "dry run: would escalate; waiting for every rank to be healthy again"
            );
            self.emit(Event::WouldEscalate(reason));
            return ControlFlow::Continue(Phase::Parked);
        }
        error!(phase = "escalate", %reason, "cannot recover the world through the FT API; escalating");
        self.mark_escalated();
        if let Err(error) = self.escalator.escalate(&self.settings.world, &reason).await {
            error!(phase = "escalate", error = %format!("{error:#}"), "escalation command failed");
        }
        ControlFlow::Break(Outcome::Escalated(reason))
    }

    /// Records in the last round record that the world is being restarted,
    /// so a restarted controller does not hold the new world to that round's
    /// survivors.
    fn mark_escalated(&self) {
        let marked = match self.rounds.last() {
            Ok(Some(mut record)) if !record.escalated => {
                record.escalated = true;
                self.rounds.save(&record)
            }
            Ok(_) => Ok(()),
            Err(error) => Err(error),
        };
        if let Err(error) = marked {
            error!(
                phase = "escalate",
                error = %format!("{error:#}"),
                "cannot mark the last round record as escalated; a restarted controller may refuse ranks that round removed"
            );
        }
    }

    /// Dry run after a would-be escalation: watch again once every rank is
    /// healthy.
    fn parked(&self) -> Phase {
        let healthy = self.members.keys().all(|rank| {
            self.observations
                .get(rank)
                .is_some_and(Observation::is_settled_healthy)
        });
        if !healthy {
            return Phase::Parked;
        }
        info!(
            phase = "watch",
            "every rank is healthy again; watching the world"
        );
        self.emit(Event::Cleared);
        Phase::Watching
    }

    fn reap_wakes(&mut self) {
        while let Some(done) = self.wakes.try_join_next() {
            if let Err(error) = done
                && error.is_panic()
            {
                error!(%error, "wake task panicked");
            }
        }
    }

    fn emit(&self, event: Event) {
        if let Some(events) = &self.events {
            // A host that stopped listening does not stop the controller.
            let _ = events.send(event);
        }
    }
}

/// Short state of an observation, to log transitions only.
fn state_of(observation: &Observation) -> String {
    match observation {
        Observation::Unreachable { .. } => "unreachable".to_string(),
        Observation::Report(report) => match &report.ft_state {
            Some(state) => format!("{} ({state:?})", report.status),
            None => report.status.to_string(),
        },
    }
}

/// The line the binary prints on stdout after an escalation.
pub fn escalation_line(world: &str, reason: &str) -> String {
    format!(
        "FT_CONTROLLER_ESCALATED world={world} reason={}",
        single_line(reason)
    )
}

/// The line a dry run prints on stdout where it would have escalated.
pub fn would_escalate_line(world: &str, reason: &str) -> String {
    format!(
        "FT_CONTROLLER_WOULD_ESCALATE world={world} reason={}",
        single_line(reason)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    fn settings() -> Settings {
        Settings {
            world: "w".to_string(),
            poll_interval: Duration::from_millis(500),
            policy: Policy {
                dead_after: 10 * S,
                wake_after: 2 * S,
                wake_max_gap: 30 * S,
                checkin_timeout: 10 * S,
                max_rounds: 3,
                recovery_deadline: 110 * S,
            },
            round_timeout: None,
            retry_abort: false,
            store_port_base: 29650,
            dry_run: false,
            remind_every: 30 * S,
        }
    }

    #[test]
    fn reminders_are_rate_limited() {
        let t0 = Instant::now();
        let mut at_once = Reminder::immediate(30 * S);
        assert!(at_once.due(t0));
        assert!(!at_once.due(t0 + 29 * S));
        assert!(at_once.due(t0 + 30 * S));
        assert!(!at_once.due(t0 + 31 * S));

        let mut later = Reminder::after(t0, 30 * S);
        assert!(!later.due(t0));
        assert!(!later.due(t0 + 29 * S));
        assert!(later.due(t0 + 30 * S));
        assert!(!later.due(t0 + 59 * S));
        assert!(later.due(t0 + 60 * S));
    }

    #[test]
    fn stall_alarm_waits_until_the_incident_outlasts_the_grace() {
        use crate::protocol::{EngineReport, EngineStatus};

        let t0 = Instant::now();
        let policy = settings().policy;
        let grace = policy.recovery_deadline;
        let store = StoreState {
            rebuilt: false,
            next_port: 29650,
            hosts: (0..3)
                .map(|rank| (rank, format!("10.0.0.{rank}")))
                .collect(),
        };
        let healthy = || Observation::Report(EngineReport::healthy());
        let observations: BTreeMap<u32, Observation> = [
            (0, healthy()),
            (
                1,
                Observation::Report(EngineReport::with_status(EngineStatus::Dead)),
            ),
            (2, healthy()),
        ]
        .into_iter()
        .collect();
        let mut episode = Episode::new(t0, 30 * S);
        // No rank counts as dead yet: nothing to alarm about.
        assert!(!episode.alarm_if_stalled(t0 + 200 * S, grace));

        decision::decide(&observations, &store, &mut episode.incident, &policy, t0);
        assert_eq!(
            episode.incident.dead().keys().copied().collect::<Vec<_>>(),
            [1]
        );
        // Survivors notice a dead peer only after a collective timeout (30 s
        // in vLLM 0.34.0): an error then would fire on every recovery.
        assert!(!episode.alarm_if_stalled(t0, grace));
        assert!(!episode.alarm_if_stalled(t0 + 60 * S, grace));
        // A stall that outlasts the grace is an error, repeated every 30 s.
        assert!(episode.alarm_if_stalled(t0 + grace, grace));
        assert!(!episode.alarm_if_stalled(t0 + grace + 29 * S, grace));
        assert!(episode.alarm_if_stalled(t0 + grace + 30 * S, grace));
    }

    #[test]
    fn outcome_lines_stay_on_one_line() {
        let reason = "survivor rank 1 died: Traceback (most recent call last):\n  File \"x\"";
        assert_eq!(
            escalation_line("w", reason),
            "FT_CONTROLLER_ESCALATED world=w reason=survivor rank 1 died: Traceback (most recent call last):\\n  File \"x\""
        );
        let would = would_escalate_line("w", reason);
        assert!(
            would.starts_with("FT_CONTROLLER_WOULD_ESCALATE world=w reason=survivor rank 1 died"),
            "{would}"
        );
        assert!(!would.contains('\n'));
    }

    #[test]
    fn settings_reject_zero_and_absurd_values() {
        assert!(settings().validate().is_ok());
        let mut zero = settings();
        zero.poll_interval = Duration::ZERO;
        assert!(zero.validate().unwrap_err().0.contains("poll_interval"));
        let mut huge = settings();
        huge.policy.recovery_deadline = MAX_DURATION + S;
        assert!(
            huge.validate()
                .unwrap_err()
                .0
                .contains("policy.recovery_deadline")
        );
        let mut round = settings();
        round.round_timeout = Some(Duration::ZERO);
        assert!(round.validate().unwrap_err().0.contains("round_timeout"));
        let mut budget = settings();
        budget.policy.max_rounds = 0;
        assert!(budget.validate().unwrap_err().0.contains("max_rounds"));
    }
}
