// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The decision core. It turns what the controller last observed about each
//! rank into one [`Decision`] ([`decide`]), and judges a round already sent
//! ([`judge_round`]). It does no I/O and never reads the clock: the caller
//! passes `now`, so any host (this binary, Engine Group, a Kubernetes world
//! controller) can drive it and tests can replay any timeline.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::protocol::{EngineReport, EngineStatus, FtState};

/// Knobs of the decision core.
#[derive(Clone, Debug)]
pub struct Policy {
    /// A rank whose status endpoint keeps failing this long counts as dead.
    pub dead_after: Duration,
    /// Survivors that still report healthy this long after the fault was
    /// suspected get woken with one tiny request.
    pub wake_after: Duration,
    /// Longest gap between two wakes of one rank. The gap after the n-th wake
    /// is `wake_after * 2^n`, capped at this, so wakes go on at a bounded rate
    /// for as long as the fault lasts.
    pub wake_max_gap: Duration,
    /// vLLM's round check-in timeout (`_ROUND_CHECKIN_TIMEOUT_SEC`, 10 s in
    /// 0.34.0). A sent round that no survivor has taken up after this was lost
    /// on the way, so a new round may follow.
    pub checkin_timeout: Duration,
    /// Recovery rounds per incident.
    pub max_rounds: u32,
    /// Time from the first survivor fault to giving up. vLLM's own recovery
    /// clock (`engine_recovery_timeout_sec`, 120 s by default) starts at that
    /// moment too, so keep this under it.
    pub recovery_deadline: Duration,
}

/// The latest evidence about one rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation {
    /// The front end answered the last status poll.
    Report(EngineReport),
    /// Status polls have failed since `since`.
    Unreachable { since: Instant, error: String },
}

impl Observation {
    pub fn is_settled_healthy(&self) -> bool {
        matches!(self, Self::Report(report) if report.is_settled_healthy())
    }
}

/// True when every rank is healthy and none is recovering.
pub fn all_healthy<'a>(observations: impl IntoIterator<Item = &'a Observation>) -> bool {
    observations
        .into_iter()
        .all(Observation::is_settled_healthy)
}

/// Address of the coordination store, as a round's `dp_master_ip` and
/// `dp_store_port`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreAddr {
    pub host: String,
    pub port: u16,
}

/// How rounds must address the coordination store. vLLM 0.34.0 hosts the
/// store on the lowest live rank: the lowest member of the world, and after a
/// round that removes it, the lowest survivor.
#[derive(Clone, Debug)]
pub struct StoreState {
    /// Set once a round has carried a store address. From then on every
    /// round carries one, on a new port: survivors of a failed store round
    /// may point at different stores.
    pub rebuilt: bool,
    /// Port of the next rebuild. Wider than `u16` so running out is an
    /// escalation, not an overflow.
    pub next_port: u32,
    /// Address each member would host the store on.
    pub hosts: BTreeMap<u32, String>,
}

/// One decision for the current poll.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Nothing to do yet; the text says what the controller waits for.
    Wait(String),
    /// Send one tiny request to each of these idle survivors, so they run a
    /// step and notice the dead peer.
    Wake(Vec<u32>),
    /// Every rank is healthy again and no round was sent: the fault cleared
    /// by itself, for example a rank the controller could not reach answers
    /// again.
    Cleared,
    /// Every rank is alive but unhealthy: retry on all of them.
    Retry { store: Option<StoreAddr> },
    /// Remove the dead ranks; the survivors carry on.
    ScaleDown {
        removed: Vec<u32>,
        store: Option<StoreAddr>,
    },
    /// Recovery through the FT API is impossible or unsafe.
    Escalate(String),
}

/// Why a rank counts as dead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadCause {
    /// Its front end reports `dead`: vLLM itself gave the engine up.
    Reported,
    /// Its status endpoint failed for `dead_after`. Only the controller lost
    /// contact, which a network problem can cause too.
    Unreachable,
}

/// What the core remembers about one fault, from first suspicion to recovery
/// or escalation.
#[derive(Clone, Debug)]
pub struct Incident {
    started: Instant,
    faulted_at: Option<Instant>,
    rounds_sent: u32,
    /// Ranks declared dead. A rank declared dead only because it was
    /// unreachable is a member again if it answers before any round was
    /// sent; any other dead rank that answers again is conflicting evidence.
    dead: BTreeMap<u32, DeadCause>,
    /// Every rank a round of this incident was sent to.
    sent_to: BTreeSet<u32>,
    wakes: BTreeMap<u32, WakeLog>,
}

#[derive(Clone, Copy, Debug)]
struct WakeLog {
    attempts: u32,
    last: Instant,
}

impl Incident {
    pub fn new(started: Instant) -> Self {
        Self {
            started,
            faulted_at: None,
            rounds_sent: 0,
            dead: BTreeMap::new(),
            sent_to: BTreeSet::new(),
            wakes: BTreeMap::new(),
        }
    }

    /// The first evidence of the fault.
    pub fn started(&self) -> Instant {
        self.started
    }

    /// When a survivor first reported anything but `healthy`. The recovery
    /// deadline runs from here, as vLLM's own recovery clock does. `None`
    /// while every survivor reports healthy.
    pub fn faulted_at(&self) -> Option<Instant> {
        self.faulted_at
    }

    pub fn rounds_sent(&self) -> u32 {
        self.rounds_sent
    }

    pub fn dead(&self) -> &BTreeMap<u32, DeadCause> {
        &self.dead
    }

    /// Wakes were sent to `ranks` at `now`.
    pub fn record_wakes(&mut self, ranks: &[u32], now: Instant) {
        for &rank in ranks {
            let log = self.wakes.entry(rank).or_insert(WakeLog {
                attempts: 0,
                last: now,
            });
            log.attempts = log.attempts.saturating_add(1);
            log.last = now;
        }
    }

    /// A round was sent to `survivors`.
    pub fn record_round(&mut self, survivors: impl IntoIterator<Item = u32>) {
        self.rounds_sent += 1;
        self.sent_to.extend(survivors);
    }

    fn mark_dead(&mut self, rank: u32, cause: DeadCause) {
        let known = self.dead.entry(rank).or_insert(cause);
        if cause == DeadCause::Reported {
            *known = DeadCause::Reported;
        }
    }

    fn wake_due(&self, rank: u32, policy: &Policy, now: Instant) -> bool {
        if now.saturating_duration_since(self.started) < policy.wake_after {
            return false;
        }
        match self.wakes.get(&rank) {
            None => true,
            Some(log) => {
                let gap = policy
                    .wake_after
                    .saturating_mul(2u32.saturating_pow(log.attempts))
                    .min(policy.wake_max_gap);
                now.saturating_duration_since(log.last) >= gap
            }
        }
    }
}

/// How the core classifies one rank for this poll.
enum Class {
    Dead {
        why: String,
        cause: DeadCause,
    },
    /// Unreachable, but not for `dead_after` yet.
    Suspect,
    Healthy,
    Unhealthy,
    /// Taking part in a recovery already in flight.
    Recovering,
    /// Diagnosing, or a status this controller does not know.
    NotReady(String),
}

fn classify(observation: &Observation, dead_after: Duration, now: Instant) -> Class {
    match observation {
        Observation::Unreachable { since, error } => {
            let down = now.saturating_duration_since(*since);
            if down >= dead_after {
                Class::Dead {
                    why: format!("unreachable for {}: {error}", secs(down)),
                    cause: DeadCause::Unreachable,
                }
            } else {
                Class::Suspect
            }
        }
        Observation::Report(report) => match &report.status {
            EngineStatus::Dead => Class::Dead {
                why: match &report.ft_error {
                    Some(error) => format!("reports dead: {error:?}"),
                    None => "reports dead".to_string(),
                },
                cause: DeadCause::Reported,
            },
            _ if report.ft_state == Some(FtState::Recovering) => Class::Recovering,
            EngineStatus::Healthy => Class::Healthy,
            EngineStatus::Unhealthy => Class::Unhealthy,
            other => Class::NotReady(other.to_string()),
        },
    }
}

/// Decides what to do about a suspected fault.
///
/// `observations` holds every member of the world. Sends a round only when
/// every survivor reports `unhealthy` (vLLM accepts orders from no other
/// state): a retry when nobody is dead, otherwise a scale-down of the dead.
/// The recovery deadline runs from the first survivor that reports anything
/// but `healthy`; while every survivor stays healthy (say the controller only
/// lost contact with one rank) nothing escalates.
pub fn decide(
    observations: &BTreeMap<u32, Observation>,
    store: &StoreState,
    incident: &mut Incident,
    policy: &Policy,
    now: Instant,
) -> Decision {
    let mut survivors = Vec::new();
    let mut healthy = Vec::new();
    let mut recovering = Vec::new();
    let mut suspects = Vec::new();
    let mut not_ready = Vec::new();
    let mut faulted = false;

    for (&rank, observation) in observations {
        let class = classify(observation, policy.dead_after, now);
        if let Class::Dead { why, cause } = &class {
            if incident.sent_to.contains(&rank) {
                return Decision::Escalate(format!(
                    "rank {rank} died after a recovery round was sent to it ({why})"
                ));
            }
            incident.mark_dead(rank, *cause);
            continue;
        }
        if let Some(&cause) = incident.dead.get(&rank) {
            if matches!(class, Class::Suspect) {
                // Dead, then out of reach again: consistent.
                continue;
            }
            if cause == DeadCause::Unreachable && incident.rounds_sent == 0 {
                // Only the controller had lost contact, and no round has
                // named the rank yet: it is a member again.
                incident.dead.remove(&rank);
            } else {
                return Decision::Escalate(format!(
                    "conflicting evidence: rank {rank} was declared dead but answers again ({})",
                    describe(observation)
                ));
            }
        }
        survivors.push(rank);
        match class {
            Class::Healthy => healthy.push(rank),
            Class::Unhealthy => faulted = true,
            Class::Recovering => {
                faulted = true;
                recovering.push(rank);
            }
            Class::Suspect => suspects.push(rank),
            Class::NotReady(status) => {
                faulted = true;
                not_ready.push(format!("{rank} ({status})"));
            }
            // Handled above.
            Class::Dead { .. } => {}
        }
    }

    if survivors.is_empty() {
        return Decision::Escalate("no survivors: every rank is dead".to_string());
    }
    if faulted && incident.faulted_at.is_none() {
        incident.faulted_at = Some(now);
    }
    if incident.rounds_sent == 0 && incident.dead.is_empty() && healthy.len() == survivors.len() {
        return Decision::Cleared;
    }

    let blocker = if !recovering.is_empty() {
        Some(format!(
            "ranks {recovering:?} report a recovery already in flight"
        ))
    } else if !suspects.is_empty() {
        Some(format!(
            "ranks {suspects:?} are unreachable but not yet dead (dead after {})",
            secs(policy.dead_after)
        ))
    } else if !healthy.is_empty() || !not_ready.is_empty() {
        let mut waiting = String::new();
        if !healthy.is_empty() {
            let _ = write!(waiting, "healthy {healthy:?}");
        }
        if !not_ready.is_empty() {
            if !waiting.is_empty() {
                waiting.push_str(", ");
            }
            let _ = write!(waiting, "not ready [{}]", not_ready.join(", "));
        }
        Some(format!(
            "waiting for every survivor to report unhealthy: {waiting}"
        ))
    } else {
        None
    };

    if let Some(faulted_at) = incident.faulted_at {
        let elapsed = now.saturating_duration_since(faulted_at);
        if elapsed >= policy.recovery_deadline {
            return Decision::Escalate(format!(
                "recovery deadline of {} after the first survivor fault exhausted after {} round(s); {}",
                secs(policy.recovery_deadline),
                incident.rounds_sent,
                blocker
                    .as_deref()
                    .unwrap_or("every survivor was ready for a round")
            ));
        }
    }

    if recovering.is_empty() {
        let wake: Vec<u32> = healthy
            .iter()
            .copied()
            .filter(|&rank| incident.wake_due(rank, policy, now))
            .collect();
        if !wake.is_empty() {
            return Decision::Wake(wake);
        }
    }
    if let Some(why) = blocker {
        return Decision::Wait(why);
    }

    if incident.rounds_sent >= policy.max_rounds {
        return Decision::Escalate(format!("round budget of {} exhausted", policy.max_rounds));
    }
    let removed: Vec<u32> = incident.dead.keys().copied().collect();
    // The lowest member hosts the store.
    let host_rank = observations.keys().next().copied();
    let store = match plan_store(&removed, &survivors, host_rank, store) {
        Ok(store) => store,
        Err(why) => return Decision::Escalate(why),
    };
    if removed.is_empty() {
        Decision::Retry { store }
    } else {
        Decision::ScaleDown { removed, store }
    }
}

/// The store address a round must carry, if any. Needed when the round
/// removes the store's host, and on every round after one that carried an
/// address. The new host is the lowest surviving rank.
fn plan_store(
    removed: &[u32],
    survivors: &[u32],
    host_rank: Option<u32>,
    store: &StoreState,
) -> Result<Option<StoreAddr>, String> {
    let host_removed = host_rank.is_some_and(|host| removed.contains(&host));
    if !store.rebuilt && !host_removed {
        return Ok(None);
    }
    let Some(&lowest) = survivors.iter().min() else {
        return Err("no survivor can host the coordination store".to_string());
    };
    let Some(host) = store.hosts.get(&lowest) else {
        return Err(format!("no store address is known for rank {lowest}"));
    };
    let Ok(port) = u16::try_from(store.next_port) else {
        return Err(format!(
            "store port range exhausted (next port {})",
            store.next_port
        ));
    };
    Ok(Some(StoreAddr {
        host: host.clone(),
        port,
    }))
}

/// A round already sent, as [`judge_round`] sees it.
#[derive(Clone, Debug)]
pub struct SentRound<'a> {
    pub request_id: &'a str,
    pub survivors: &'a BTreeSet<u32>,
    /// Survivors whose front end refused the order outright (HTTP 4xx).
    pub refused: &'a BTreeMap<u32, String>,
    /// When the orders went out.
    pub sent_at: Instant,
    /// How long the round may run before the controller gives up on it.
    pub timeout: Duration,
}

/// How a sent round is going.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pending,
    /// Every survivor is healthy again.
    Recovered,
    /// The round failed and left every survivor unhealthy and idle, so a new
    /// round (new request id) may be tried.
    Failed(String),
    /// The round can neither finish nor be repeated safely.
    Fatal(String),
}

/// Judges a sent round from the survivors' latest observations.
pub fn judge_round(
    round: &SentRound<'_>,
    observations: &BTreeMap<u32, Observation>,
    policy: &Policy,
    now: Instant,
) -> Verdict {
    let id = round.request_id;
    let mut recovered = Vec::new();
    let mut failed = Vec::new();
    let mut in_flight = Vec::new();
    let mut suspects = Vec::new();

    for &rank in round.survivors {
        let Some(observation) = observations.get(&rank) else {
            suspects.push(rank);
            continue;
        };
        let report = match observation {
            Observation::Unreachable { since, error } => {
                let down = now.saturating_duration_since(*since);
                if down >= policy.dead_after {
                    return Verdict::Fatal(format!(
                        "survivor rank {rank} unreachable for {} during round {id}: {error}",
                        secs(down)
                    ));
                }
                suspects.push(rank);
                continue;
            }
            Observation::Report(report) => report,
        };
        let last = report.last_ft_request_id.as_deref();
        let ours = last == Some(id);
        if report.ft_state == Some(FtState::Recovering)
            && let Some(other) = last.filter(|other| *other != id)
        {
            return Verdict::Fatal(format!(
                "conflicting evidence: rank {rank} works on round {other}, not {id}"
            ));
        }
        match &report.status {
            EngineStatus::Dead => {
                return Verdict::Fatal(format!(
                    "survivor rank {rank} died during round {id} ({})",
                    describe(observation)
                ));
            }
            EngineStatus::Healthy => match last {
                Some(other) if other != id => {
                    return Verdict::Fatal(format!(
                        "conflicting evidence: rank {rank} is healthy but its last recovery was {other}, not {id}"
                    ));
                }
                _ if ours && report.ft_state == Some(FtState::Failed) => {
                    return Verdict::Fatal(format!(
                        "conflicting evidence: rank {rank} is healthy but refused round {id} ({})",
                        describe(observation)
                    ));
                }
                // Healthy, but its front end is still finishing the round.
                _ if report.ft_state == Some(FtState::Recovering) => in_flight.push(rank),
                // vLLM 0.34.0 drops the request id once the engine is
                // healthy again, so a healthy survivor has recovered.
                _ => recovered.push(rank),
            },
            _ => {
                if let Some(why) = round.refused.get(&rank) {
                    failed.push(format!("{rank}: {why}"));
                } else if ours && report.ft_state == Some(FtState::Failed) {
                    failed.push(format!(
                        "{rank}: {}",
                        report.ft_error.as_deref().unwrap_or("no reason given")
                    ));
                } else if report.ft_state == Some(FtState::Recovering) {
                    in_flight.push(rank);
                }
                // Otherwise the rank has not taken the order up: it stays
                // unhealthy and idle.
            }
        }
    }

    if recovered.len() == round.survivors.len() {
        return Verdict::Recovered;
    }
    if !recovered.is_empty() && !failed.is_empty() {
        return Verdict::Fatal(format!(
            "conflicting evidence: round {id} recovered ranks {recovered:?} but failed on [{}]",
            failed.join("; ")
        ));
    }
    let settled = recovered.is_empty() && in_flight.is_empty() && suspects.is_empty();
    if settled && !failed.is_empty() {
        return Verdict::Failed(format!("round {id} failed on [{}]", failed.join("; ")));
    }
    let running = now.saturating_duration_since(round.sent_at);
    if settled && running >= policy.checkin_timeout {
        return Verdict::Failed(format!(
            "no survivor took round {id} up within {}; its orders were lost",
            secs(policy.checkin_timeout)
        ));
    }
    if running >= round.timeout {
        if settled {
            return Verdict::Failed(format!(
                "round {id} timed out and no survivor is working on it"
            ));
        }
        return Verdict::Fatal(format!(
            "round {id} unsettled after {}: recovered {recovered:?}, in flight {in_flight:?}, unreachable {suspects:?}",
            secs(round.timeout)
        ));
    }
    Verdict::Pending
}

/// Checks that the front end of `--rank rank` serves that rank's engine. In
/// vLLM 0.34.0 the engine id is the global data-parallel rank, so another id
/// means swapped or duplicated `--rank` URLs, and every round would name the
/// wrong ranks. A missing id cannot be checked.
pub fn check_identity(rank: u32, report: &EngineReport) -> Result<(), String> {
    match report.id {
        Some(id) if id != u64::from(rank) => Err(format!(
            "the front end of --rank {rank} reports engine {id}: its URL serves another rank (check for swapped or duplicated --rank values)"
        )),
        _ => Ok(()),
    }
}

/// One-line description of an observation, for logs and escalation reasons.
pub fn describe(observation: &Observation) -> String {
    match observation {
        Observation::Unreachable { error, .. } => {
            format!("unreachable: {}", single_line(error))
        }
        Observation::Report(report) => {
            let mut text = report.status.to_string();
            if let Some(state) = &report.ft_state {
                let _ = write!(text, ", ft_state {state:?}");
            }
            if let Some(id) = &report.last_ft_request_id {
                let _ = write!(text, ", last_ft_request_id {id}");
            }
            if let Some(error) = &report.ft_error {
                let _ = write!(text, ", ft_error {error:?}");
            }
            text
        }
    }
}

/// `text` on one line: line breaks, tabs and other control characters are
/// escaped, so a multi-line error from vLLM cannot split a summary line.
pub fn single_line(text: &str) -> String {
    let mut line = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' => line.push_str("\\n"),
            '\r' => line.push_str("\\r"),
            '\t' => line.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(line, "\\u{{{:x}}}", u32::from(c));
            }
            c => line.push(c),
        }
    }
    line
}

/// Seconds with millisecond precision, as logs and reasons print durations.
pub fn secs(duration: Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);
    const ID: &str = "w-e1-r1-0000beef";

    fn policy() -> Policy {
        Policy {
            dead_after: 10 * S,
            wake_after: 2 * S,
            wake_max_gap: 30 * S,
            checkin_timeout: 10 * S,
            max_rounds: 3,
            recovery_deadline: 110 * S,
        }
    }

    fn store(ranks: u32) -> StoreState {
        StoreState {
            rebuilt: false,
            next_port: 29650,
            hosts: (0..ranks).map(|r| (r, format!("10.0.0.{r}"))).collect(),
        }
    }

    fn report(status: EngineStatus) -> Observation {
        Observation::Report(EngineReport::with_status(status))
    }

    fn healthy() -> Observation {
        report(EngineStatus::Healthy)
    }

    fn unhealthy() -> Observation {
        report(EngineStatus::Unhealthy)
    }

    fn dead() -> Observation {
        report(EngineStatus::Dead)
    }

    fn unreachable(since: Instant) -> Observation {
        Observation::Unreachable {
            since,
            error: "connection refused".to_string(),
        }
    }

    fn world(observations: Vec<Observation>) -> BTreeMap<u32, Observation> {
        (0u32..).zip(observations).collect()
    }

    fn after_round(
        request_id: &str,
        status: EngineStatus,
        ft_state: Option<FtState>,
    ) -> Observation {
        Observation::Report(EngineReport {
            id: None,
            status,
            ft_state,
            last_ft_request_id: Some(request_id.to_string()),
            ft_error: Some("boom".to_string()),
        })
    }

    #[test]
    fn scale_down_removes_only_the_dead_ranks() {
        let t0 = Instant::now();
        let obs = world(vec![unhealthy(), unhealthy(), dead(), unhealthy()]);
        let mut incident = Incident::new(t0);
        let decision = decide(&obs, &store(4), &mut incident, &policy(), t0 + S);
        assert_eq!(
            decision,
            Decision::ScaleDown {
                removed: vec![2],
                store: None
            }
        );
        assert_eq!(incident.faulted_at(), Some(t0 + S));
    }

    #[test]
    fn losing_the_store_host_moves_the_store_to_the_lowest_survivor() {
        let t0 = Instant::now();
        let obs = world(vec![dead(), dead(), unhealthy(), unhealthy()]);
        let mut incident = Incident::new(t0);
        let decision = decide(&obs, &store(4), &mut incident, &policy(), t0 + S);
        assert_eq!(
            decision,
            Decision::ScaleDown {
                removed: vec![0, 1],
                store: Some(StoreAddr {
                    host: "10.0.0.2".to_string(),
                    port: 29650
                }),
            }
        );
    }

    #[test]
    fn after_a_store_round_every_round_carries_a_store() {
        let t0 = Instant::now();
        let mut state = store(4);
        state.rebuilt = true;
        state.next_port = 29652;
        state.hosts.remove(&0);
        let obs: BTreeMap<u32, Observation> =
            [(1, unhealthy()), (2, unhealthy()), (3, unhealthy())].into();
        let mut incident = Incident::new(t0);
        let decision = decide(&obs, &state, &mut incident, &policy(), t0 + S);
        assert_eq!(
            decision,
            Decision::Retry {
                store: Some(StoreAddr {
                    host: "10.0.0.1".to_string(),
                    port: 29652
                })
            }
        );
    }

    #[test]
    fn all_alive_but_unhealthy_retries() {
        let t0 = Instant::now();
        let obs = world(vec![unhealthy(); 4]);
        let mut incident = Incident::new(t0);
        let decision = decide(&obs, &store(4), &mut incident, &policy(), t0);
        assert_eq!(decision, Decision::Retry { store: None });
    }

    #[test]
    fn an_unreachable_rank_is_dead_only_after_dead_after() {
        let t0 = Instant::now();
        let obs = world(vec![unhealthy(), unreachable(t0), unhealthy()]);
        let mut incident = Incident::new(t0);
        let early = decide(&obs, &store(3), &mut incident, &policy(), t0 + 9 * S);
        assert!(
            matches!(early, Decision::Wait(ref why) if why.contains("[1]")),
            "{early:?}"
        );
        let late = decide(&obs, &store(3), &mut incident, &policy(), t0 + 10 * S);
        assert_eq!(
            late,
            Decision::ScaleDown {
                removed: vec![1],
                store: None
            }
        );
        assert_eq!(incident.dead().get(&1), Some(&DeadCause::Unreachable));
    }

    #[test]
    fn healthy_survivors_are_woken_with_backoff_up_to_the_max_gap() {
        let t0 = Instant::now();
        let p = Policy {
            wake_max_gap: 10 * S,
            ..policy()
        };
        let obs = world(vec![unhealthy(), healthy(), dead(), healthy()]);
        let mut incident = Incident::new(t0);
        let state = store(4);

        let before = decide(&obs, &state, &mut incident, &p, t0 + S);
        assert!(matches!(before, Decision::Wait(_)), "{before:?}");
        let expect_wake_at = |at: u32, quiet_before: bool, incident: &mut Incident| {
            if quiet_before {
                let early = decide(&obs, &state, incident, &p, t0 + (at - 1) * S);
                assert!(matches!(early, Decision::Wait(_)), "at {at}: {early:?}");
            }
            assert_eq!(
                decide(&obs, &state, incident, &p, t0 + at * S),
                Decision::Wake(vec![1, 3]),
                "at {at}"
            );
            incident.record_wakes(&[1, 3], t0 + at * S);
        };
        // The gap doubles after each wake (4 s, 8 s), then stays at the
        // 10 s maximum instead of growing to 16 s.
        expect_wake_at(2, false, &mut incident);
        expect_wake_at(6, true, &mut incident);
        expect_wake_at(14, true, &mut incident);
        expect_wake_at(24, true, &mut incident);
        expect_wake_at(34, true, &mut incident);

        // The deadline runs from rank 0's fault, seen at t0 + 1 s.
        let last_chance = decide(&obs, &state, &mut incident, &p, t0 + 110 * S);
        assert_eq!(last_chance, Decision::Wake(vec![1, 3]));
        let gave_up = decide(&obs, &state, &mut incident, &p, t0 + 111 * S);
        assert!(
            matches!(gave_up, Decision::Escalate(ref why) if why.contains("deadline") && why.contains("healthy [1, 3]")),
            "{gave_up:?}"
        );
    }

    #[test]
    fn an_unreachable_rank_with_healthy_survivors_never_escalates() {
        let t0 = Instant::now();
        let p = policy();
        let obs = world(vec![healthy(), unreachable(t0), healthy()]);
        let state = store(3);
        let mut incident = Incident::new(t0);
        let mut wakes = Vec::new();
        for second in 0..=1000u64 {
            let now = t0 + Duration::from_secs(second);
            match decide(&obs, &state, &mut incident, &p, now) {
                Decision::Wake(ranks) => {
                    assert_eq!(ranks, [0, 2]);
                    incident.record_wakes(&ranks, now);
                    wakes.push(second);
                }
                Decision::Wait(_) => {}
                other => panic!("at {second} s: {other:?}"),
            }
        }
        assert_eq!(incident.dead().get(&1), Some(&DeadCause::Unreachable));
        assert_eq!(incident.faulted_at(), None);
        // Wakes go on at a bounded rate: 2, 6, 14, 30 s, then every 30 s.
        assert_eq!(wakes[..5], [2, 6, 14, 30, 60]);
        assert!(wakes.windows(2).all(|gap| gap[1] - gap[0] <= 30));
        assert_eq!(wakes.len(), 36);
    }

    #[test]
    fn an_unreachable_rank_that_answers_before_any_round_is_cleared() {
        let t0 = Instant::now();
        let gone = world(vec![healthy(), unreachable(t0), healthy()]);

        let mut incident = Incident::new(t0);
        decide(&gone, &store(3), &mut incident, &policy(), t0 + 10 * S);
        assert_eq!(incident.dead().get(&1), Some(&DeadCause::Unreachable));
        let back = world(vec![healthy(); 3]);
        let decision = decide(&back, &store(3), &mut incident, &policy(), t0 + 11 * S);
        assert_eq!(decision, Decision::Cleared);
        assert!(incident.dead().is_empty());

        // Back but faulted, like every other rank: a retry, not a
        // scale-down of a live rank.
        let mut incident = Incident::new(t0);
        decide(&gone, &store(3), &mut incident, &policy(), t0 + 10 * S);
        let faulted = world(vec![unhealthy(); 3]);
        let decision = decide(&faulted, &store(3), &mut incident, &policy(), t0 + 11 * S);
        assert_eq!(decision, Decision::Retry { store: None });
    }

    #[test]
    fn an_unreachable_rank_that_answers_after_a_round_is_conflicting_evidence() {
        let t0 = Instant::now();
        let mut incident = Incident::new(t0);
        let gone = world(vec![unhealthy(), unreachable(t0), unhealthy()]);
        assert!(matches!(
            decide(&gone, &store(3), &mut incident, &policy(), t0 + 10 * S),
            Decision::ScaleDown { .. }
        ));
        incident.record_round([0, 2]);
        let back = world(vec![unhealthy(), healthy(), unhealthy()]);
        let decision = decide(&back, &store(3), &mut incident, &policy(), t0 + 11 * S);
        assert!(
            matches!(decision, Decision::Escalate(ref why) if why.contains("conflicting")),
            "{decision:?}"
        );
    }

    #[test]
    fn the_deadline_starts_at_the_first_survivor_fault() {
        let t0 = Instant::now();
        let p = policy();
        let state = store(3);
        let mut incident = Incident::new(t0);
        // Rank 1 is out of reach from t0 while the survivors stay healthy.
        let quiet = world(vec![healthy(), unreachable(t0), healthy()]);
        for at in [10, 200, 500] {
            let decision = decide(&quiet, &state, &mut incident, &p, t0 + at * S);
            assert!(
                !matches!(decision, Decision::Escalate(_)),
                "at {at}: {decision:?}"
            );
        }
        assert_eq!(incident.faulted_at(), None);
        // Rank 0 faults at 500 s: vLLM's clock and the deadline start now.
        let faulted = world(vec![unhealthy(), unreachable(t0), healthy()]);
        decide(&faulted, &state, &mut incident, &p, t0 + 500 * S);
        assert_eq!(incident.faulted_at(), Some(t0 + 500 * S));
        let before = decide(&faulted, &state, &mut incident, &p, t0 + 609 * S);
        assert!(!matches!(before, Decision::Escalate(_)), "{before:?}");
        let after = decide(&faulted, &state, &mut incident, &p, t0 + 610 * S);
        assert!(
            matches!(after, Decision::Escalate(ref why) if why.contains("deadline")),
            "{after:?}"
        );
    }

    #[test]
    fn diagnosing_or_unknown_ranks_hold_the_round() {
        let t0 = Instant::now();
        let obs = world(vec![
            unhealthy(),
            report(EngineStatus::Diagnosing),
            report(EngineStatus::Unknown("paused".to_string())),
        ]);
        let mut incident = Incident::new(t0);
        let decision = decide(&obs, &store(3), &mut incident, &policy(), t0 + 5 * S);
        assert!(
            matches!(decision, Decision::Wait(ref why) if why.contains("1 (diagnosing)") && why.contains("paused")),
            "{decision:?}"
        );
    }

    #[test]
    fn a_recovery_in_flight_is_left_alone() {
        let t0 = Instant::now();
        let busy = after_round(ID, EngineStatus::Unhealthy, Some(FtState::Recovering));
        let obs = world(vec![busy, unhealthy(), healthy()]);
        let mut incident = Incident::new(t0);
        let decision = decide(&obs, &store(3), &mut incident, &policy(), t0 + 5 * S);
        assert!(
            matches!(decision, Decision::Wait(ref why) if why.contains("in flight")),
            "{decision:?}"
        );
    }

    #[test]
    fn escalates_without_survivors_deadline_or_budget() {
        let t0 = Instant::now();
        let p = policy();
        let mut incident = Incident::new(t0);
        let none = decide(
            &world(vec![dead(), dead()]),
            &store(2),
            &mut incident,
            &p,
            t0,
        );
        assert!(matches!(none, Decision::Escalate(ref why) if why.contains("no survivors")));

        let obs = world(vec![unhealthy(), unhealthy()]);
        let mut incident = Incident::new(t0);
        assert_eq!(
            decide(&obs, &store(2), &mut incident, &p, t0),
            Decision::Retry { store: None }
        );
        let late = decide(&obs, &store(2), &mut incident, &p, t0 + 110 * S);
        assert!(matches!(late, Decision::Escalate(ref why) if why.contains("deadline")));

        let mut incident = Incident::new(t0);
        for _ in 0..3 {
            incident.record_round([0, 1]);
        }
        let spent = decide(&obs, &store(2), &mut incident, &p, t0 + S);
        assert!(matches!(spent, Decision::Escalate(ref why) if why.contains("budget")));
    }

    #[test]
    fn a_dead_round_survivor_escalates() {
        let t0 = Instant::now();
        let mut incident = Incident::new(t0);
        incident.record_round([0, 1, 3]);
        let obs = world(vec![unhealthy(), dead(), dead(), unhealthy()]);
        let decision = decide(&obs, &store(4), &mut incident, &policy(), t0 + S);
        assert!(
            matches!(decision, Decision::Escalate(ref why) if why.contains("rank 1")),
            "{decision:?}"
        );
    }

    #[test]
    fn a_rank_reported_dead_that_answers_again_is_conflicting_evidence() {
        let t0 = Instant::now();
        let mut incident = Incident::new(t0);
        let first = world(vec![unhealthy(), dead(), healthy()]);
        assert!(matches!(
            decide(&first, &store(3), &mut incident, &policy(), t0),
            Decision::Wait(_)
        ));
        assert_eq!(incident.dead().get(&1), Some(&DeadCause::Reported));
        // Gone for good is consistent...
        let gone = world(vec![unhealthy(), unreachable(t0), healthy()]);
        assert!(matches!(
            decide(&gone, &store(3), &mut incident, &policy(), t0 + S),
            Decision::Wait(_)
        ));
        // ...a restarted process is not, even before any round.
        let back = world(vec![unhealthy(), healthy(), healthy()]);
        let decision = decide(&back, &store(3), &mut incident, &policy(), t0 + S);
        assert!(
            matches!(decision, Decision::Escalate(ref why) if why.contains("conflicting")),
            "{decision:?}"
        );
    }

    #[test]
    fn running_out_of_store_ports_escalates() {
        let t0 = Instant::now();
        let mut state = store(2);
        state.next_port = u32::from(u16::MAX) + 1;
        let obs = world(vec![dead(), unhealthy()]);
        let mut incident = Incident::new(t0);
        let decision = decide(&obs, &state, &mut incident, &policy(), t0);
        assert!(matches!(decision, Decision::Escalate(ref why) if why.contains("port range")));
    }

    /// Judges round [`ID`] (survivors 0, 1, 2; sent at `t0`; may run 60 s)
    /// at `t0 + at`.
    fn judge_at(
        t0: Instant,
        observations: &BTreeMap<u32, Observation>,
        refused: &BTreeMap<u32, String>,
        at: Duration,
    ) -> Verdict {
        let survivors: BTreeSet<u32> = [0, 1, 2].into();
        let round = SentRound {
            request_id: ID,
            survivors: &survivors,
            refused,
            sent_at: t0,
            timeout: 60 * S,
        };
        judge_round(&round, observations, &policy(), t0 + at)
    }

    fn judge(
        observations: &BTreeMap<u32, Observation>,
        refused: &BTreeMap<u32, String>,
        at: Duration,
    ) -> Verdict {
        judge_at(Instant::now(), observations, refused, at)
    }

    #[test]
    fn a_round_recovers_when_every_survivor_is_healthy() {
        let none = BTreeMap::new();
        let recovering = after_round(ID, EngineStatus::Unhealthy, Some(FtState::Recovering));
        // vLLM 0.34.0 drops the request id once healthy; a front end that
        // keeps it is fine too.
        let kept = after_round(ID, EngineStatus::Healthy, None);
        let mut obs: BTreeMap<u32, Observation> =
            [(0, healthy()), (1, recovering), (2, kept)].into();
        assert_eq!(judge(&obs, &none, S), Verdict::Pending);
        // Healthy while the front end still reports the round in flight is
        // not recovered yet.
        let finishing = after_round(ID, EngineStatus::Healthy, Some(FtState::Recovering));
        obs.insert(1, finishing);
        assert_eq!(judge(&obs, &none, S), Verdict::Pending);
        obs.insert(1, healthy());
        assert_eq!(judge(&obs, &none, S), Verdict::Recovered);
    }

    #[test]
    fn a_settled_failure_allows_another_round() {
        let failed = after_round(ID, EngineStatus::Unhealthy, Some(FtState::Failed));
        let recovering = after_round(ID, EngineStatus::Unhealthy, Some(FtState::Recovering));
        // Rank 2 never got the order: it stays unhealthy and idle.
        let mut obs: BTreeMap<u32, Observation> =
            [(0, failed.clone()), (1, recovering), (2, unhealthy())].into();
        assert_eq!(judge(&obs, &BTreeMap::new(), S), Verdict::Pending);
        obs.insert(1, failed);
        assert!(
            matches!(judge(&obs, &BTreeMap::new(), S), Verdict::Failed(ref why) if why.contains("boom"))
        );

        let refused: BTreeMap<u32, String> = [(2, "HTTP 400".to_string())].into();
        let obs: BTreeMap<u32, Observation> =
            [(0, unhealthy()), (1, unhealthy()), (2, unhealthy())].into();
        assert!(
            matches!(judge(&obs, &refused, S), Verdict::Failed(ref why) if why.contains("HTTP 400"))
        );
    }

    #[test]
    fn a_round_nobody_took_up_fails_after_the_checkin_timeout() {
        let t0 = Instant::now();
        let none = BTreeMap::new();
        // An earlier round's failure is not this round's.
        let stale = after_round(
            "w-e1-r0-00000000",
            EngineStatus::Unhealthy,
            Some(FtState::Failed),
        );
        let idle: BTreeMap<u32, Observation> =
            [(0, unhealthy()), (1, stale), (2, unhealthy())].into();
        assert_eq!(judge_at(t0, &idle, &none, 9 * S), Verdict::Pending);
        assert!(
            matches!(judge_at(t0, &idle, &none, 10 * S), Verdict::Failed(ref why) if why.contains("lost")),
        );
    }

    #[test]
    fn deaths_splits_and_foreign_rounds_are_fatal() {
        let none = BTreeMap::new();
        let healthy_after = after_round(ID, EngineStatus::Healthy, None);
        let failed = after_round(ID, EngineStatus::Unhealthy, Some(FtState::Failed));

        let died: BTreeMap<u32, Observation> =
            [(0, healthy_after.clone()), (1, dead()), (2, unhealthy())].into();
        assert!(
            matches!(judge(&died, &none, S), Verdict::Fatal(ref why) if why.contains("rank 1 died"))
        );

        let t0 = Instant::now();
        let gone: BTreeMap<u32, Observation> =
            [(0, unhealthy()), (1, unreachable(t0)), (2, unhealthy())].into();
        assert_eq!(judge_at(t0, &gone, &none, 9 * S), Verdict::Pending);
        assert!(
            matches!(judge_at(t0, &gone, &none, 10 * S), Verdict::Fatal(ref why) if why.contains("unreachable"))
        );

        let split: BTreeMap<u32, Observation> =
            [(0, healthy_after), (1, failed), (2, unhealthy())].into();
        assert!(
            matches!(judge(&split, &none, S), Verdict::Fatal(ref why) if why.contains("conflicting"))
        );

        let foreign = after_round("w-e1-r9-00000000", EngineStatus::Healthy, None);
        let stale: BTreeMap<u32, Observation> =
            [(0, foreign), (1, unhealthy()), (2, unhealthy())].into();
        assert!(
            matches!(judge(&stale, &none, S), Verdict::Fatal(ref why) if why.contains("w-e1-r9"))
        );

        let elsewhere = after_round(
            "w-e1-r9-00000000",
            EngineStatus::Unhealthy,
            Some(FtState::Recovering),
        );
        let busy: BTreeMap<u32, Observation> =
            [(0, unhealthy()), (1, elsewhere), (2, unhealthy())].into();
        assert!(
            matches!(judge(&busy, &none, S), Verdict::Fatal(ref why) if why.contains("works on round w-e1-r9"))
        );
    }

    #[test]
    fn a_round_past_its_timeout_is_fatal_while_a_survivor_works_on_it() {
        let none = BTreeMap::new();
        let busy: BTreeMap<u32, Observation> = [
            (0, unhealthy()),
            (
                1,
                after_round(ID, EngineStatus::Unhealthy, Some(FtState::Recovering)),
            ),
            (2, unhealthy()),
        ]
        .into();
        assert_eq!(judge(&busy, &none, 59 * S), Verdict::Pending);
        assert!(
            matches!(judge(&busy, &none, 60 * S), Verdict::Fatal(ref why) if why.contains("in flight [1]"))
        );
    }

    #[test]
    fn a_front_end_must_serve_its_own_rank() {
        let mut report = EngineReport::healthy();
        assert_eq!(
            check_identity(2, &report),
            Ok(()),
            "no id: nothing to check"
        );
        report.id = Some(2);
        assert_eq!(check_identity(2, &report), Ok(()));
        report.id = Some(1);
        let error = check_identity(2, &report).unwrap_err();
        assert!(error.contains("--rank 2 reports engine 1"), "{error}");
    }

    #[test]
    fn single_line_escapes_line_breaks_and_controls() {
        assert_eq!(single_line("plain text"), "plain text");
        assert_eq!(
            single_line("Traceback:\n  File \"x\"\r\n\tboom\u{7}"),
            "Traceback:\\n  File \"x\"\\r\\n\\tboom\\u{7}"
        );
    }
}
