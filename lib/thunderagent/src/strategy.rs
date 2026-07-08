// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Reverse;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::{
    AdmissionAction, AdmissionDecision, AdmissionEvent, AdmissionId, AdmissionRequest,
    PolicyClassAdmissionStrategy, WorkerEligibility, WorkerEligibilitySnapshot, WorkerPlacement,
};
use indexmap::{IndexMap, IndexSet};

use crate::{ConfigError, ThunderAgentConfig, WorkerCapacity, WorkerCapacityProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgramStatus {
    Reasoning,
    Acting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgramLifecycle {
    Active,
    Paused,
}

struct Program {
    status: ProgramStatus,
    lifecycle: ProgramLifecycle,
    assigned_worker: Option<WorkerWithDpRank>,
    token_total: usize,
    step_count: usize,
    marked_for_pause: bool,
    acting_since: Option<Instant>,
    deferred: IndexMap<AdmissionId, Instant>,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            status: ProgramStatus::Reasoning,
            lifecycle: ProgramLifecycle::Active,
            assigned_worker: None,
            token_total: 0,
            step_count: 0,
            marked_for_pause: false,
            acting_since: None,
            deferred: IndexMap::new(),
        }
    }
}

#[derive(Clone)]
struct ProgramSnapshot {
    status: ProgramStatus,
    lifecycle: ProgramLifecycle,
    assigned_worker: Option<WorkerWithDpRank>,
    token_total: usize,
    step_count: usize,
    marked_for_pause: bool,
    acting_since: Option<Instant>,
}

impl From<&Program> for ProgramSnapshot {
    fn from(program: &Program) -> Self {
        Self {
            status: program.status,
            lifecycle: program.lifecycle,
            assigned_worker: program.assigned_worker,
            token_total: program.token_total,
            step_count: program.step_count,
            marked_for_pause: program.marked_for_pause,
            acting_since: program.acting_since,
        }
    }
}

impl ProgramSnapshot {
    fn restore(self, program: &mut Program) {
        program.status = self.status;
        program.lifecycle = self.lifecycle;
        program.assigned_worker = self.assigned_worker;
        program.token_total = self.token_total;
        program.step_count = self.step_count;
        program.marked_for_pause = self.marked_for_pause;
        program.acting_since = self.acting_since;
    }
}

struct RequestState {
    session_id: String,
    input_tokens: usize,
    worker_eligibility: Option<WorkerEligibility>,
    dispatched: bool,
    prior: Option<ProgramSnapshot>,
}

#[derive(Debug, Default, Clone, Copy)]
struct WorkerUsage {
    used: usize,
    decayed: usize,
}

pub struct ThunderAgent<P> {
    capacity: P,
    config: ThunderAgentConfig,
    programs: IndexMap<String, Program>,
    paused: IndexSet<String>,
    requests: HashMap<AdmissionId, RequestState>,
    next_tick: Instant,
}

impl<P: WorkerCapacityProvider> ThunderAgent<P> {
    pub fn new(capacity: P, config: ThunderAgentConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let next_tick = Instant::now() + Duration::from_secs_f64(config.scheduler_interval_seconds);
        Ok(Self {
            capacity,
            config,
            programs: IndexMap::new(),
            paused: IndexSet::new(),
            requests: HashMap::new(),
            next_tick,
        })
    }

    fn admit_request(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
        let Some(session_id) = request.session_id() else {
            return AdmissionDecision::Ready(WorkerPlacement::Any);
        };

        let now = Instant::now();
        let capacities = self.capacity.snapshot();
        let capacity_known = !capacities.is_empty();
        let worker_eligibility = request.worker_eligibility().cloned();
        let eligibility = worker_eligibility.as_ref().map(WorkerEligibility::snapshot);
        let allows_worker = |worker| {
            eligibility
                .as_ref()
                .is_none_or(|eligibility| eligibility.allows(worker))
        };
        let session_id = session_id.to_owned();
        let prior = self.programs.get(&session_id).map(ProgramSnapshot::from);
        let was_new = prior.is_none();
        let program = self.programs.entry(session_id.clone()).or_default();
        program.step_count = program.step_count.saturating_add(1);
        if request.input_tokens() > 0 {
            program.token_total = request.input_tokens();
        }
        program.status = ProgramStatus::Reasoning;
        program.acting_since = None;
        let lifecycle = program.lifecycle;
        let assigned_worker = program.assigned_worker;

        self.requests.insert(
            request.id(),
            RequestState {
                session_id: session_id.clone(),
                input_tokens: request.input_tokens(),
                worker_eligibility,
                dispatched: false,
                prior,
            },
        );

        if lifecycle == ProgramLifecycle::Paused {
            self.defer_request(&session_id, request.id(), now);
            return AdmissionDecision::Defer;
        }

        if !capacity_known {
            if let Some(worker) = assigned_worker {
                if allows_worker(worker) {
                    return AdmissionDecision::Ready(WorkerPlacement::Exact(worker));
                }
                if let Some(program) = self.programs.get_mut(&session_id) {
                    program.assigned_worker = None;
                }
            }
            return AdmissionDecision::Ready(WorkerPlacement::Any);
        }
        if !capacities
            .iter()
            .any(|capacity| allows_worker(capacity.worker))
        {
            if let Some(program) = self.programs.get_mut(&session_id) {
                program.assigned_worker = None;
            }
            return AdmissionDecision::Ready(WorkerPlacement::Any);
        }

        if !was_new
            && assigned_worker.is_some_and(|worker| {
                allows_worker(worker) && capacities.iter().any(|capacity| capacity.worker == worker)
            })
        {
            return AdmissionDecision::Ready(WorkerPlacement::Exact(
                assigned_worker.expect("assigned worker was checked"),
            ));
        }
        if !was_new && let Some(program) = self.programs.get_mut(&session_id) {
            program.assigned_worker = None;
        }

        // Preserve source fairness: a new program cannot bypass an already-paused one.
        if was_new && !self.paused.is_empty() {
            self.defer_request(&session_id, request.id(), now);
            return AdmissionDecision::Defer;
        }

        let required = request
            .input_tokens()
            .saturating_add(self.config.buffer_per_program);
        let usage = self.worker_usage(now);
        let selected = capacities
            .iter()
            .filter(|capacity| allows_worker(capacity.worker))
            .filter_map(|capacity| {
                let used = usage.get(&capacity.worker).map_or(0, |usage| usage.used);
                capacity
                    .tokens
                    .checked_sub(used)
                    .is_some_and(|remaining| remaining >= required)
                    .then_some((capacity.worker, used))
            })
            .min_by_key(|(worker, used)| (*used, *worker))
            .map(|(worker, _)| worker);

        if let Some(worker) = selected {
            if let Some(program) = self.programs.get_mut(&session_id) {
                program.assigned_worker = Some(worker);
            }
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker))
        } else {
            self.defer_request(&session_id, request.id(), now);
            AdmissionDecision::Defer
        }
    }

    fn defer_request(&mut self, session_id: &str, id: AdmissionId, now: Instant) {
        let Some(program) = self.programs.get_mut(session_id) else {
            return;
        };
        program.lifecycle = ProgramLifecycle::Paused;
        program.assigned_worker = None;
        program.deferred.insert(id, now);
        self.paused.insert(session_id.to_owned());
    }

    fn dispatched(&mut self, id: AdmissionId, worker: WorkerWithDpRank) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        request.dispatched = true;
        if let Some(program) = self.programs.get_mut(&request.session_id) {
            program.assigned_worker = Some(worker);
        }
    }

    fn progress(&mut self, id: AdmissionId, output_tokens: usize) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        if !request.dispatched {
            return;
        }
        if let Some(program) = self.programs.get_mut(&request.session_id) {
            program.token_total = request.input_tokens.saturating_add(output_tokens);
        }
    }

    fn finished(&mut self, id: AdmissionId, total_tokens: usize) {
        let Some(request) = self.requests.remove(&id) else {
            return;
        };
        let Some(program) = self.programs.get_mut(&request.session_id) else {
            return;
        };
        program.deferred.shift_remove(&id);

        if !request.dispatched {
            let has_other_request = self
                .requests
                .values()
                .any(|other| other.session_id == request.session_id);
            if has_other_request {
                return;
            }
            match request.prior {
                Some(prior) => {
                    prior.restore(program);
                    if program.lifecycle == ProgramLifecycle::Paused {
                        self.paused.insert(request.session_id);
                    } else {
                        self.paused.shift_remove(&request.session_id);
                    }
                }
                None => {
                    self.programs.shift_remove(&request.session_id);
                    self.paused.shift_remove(&request.session_id);
                }
            }
            return;
        }

        program.token_total = total_tokens;
        program.status = ProgramStatus::Acting;
        program.acting_since = Some(Instant::now());
        let pause = std::mem::take(&mut program.marked_for_pause);
        if pause {
            self.pause_acting(&request.session_id);
        }
    }

    fn reconcile(&mut self) -> Vec<AdmissionAction> {
        let now = Instant::now();
        let tick_due = now >= self.next_tick;
        let timeout_due = self
            .next_resume_deadline()
            .is_some_and(|deadline| deadline <= now);
        if !tick_due && !timeout_due {
            return Vec::new();
        }
        if tick_due {
            self.next_tick = now + Duration::from_secs_f64(self.config.scheduler_interval_seconds);
        }
        let capacities = self.capacity.snapshot();
        let mut usage = self.worker_usage(now);
        let eligibility = self.deferred_eligibility_snapshots();
        let mut actions = if capacities.is_empty() {
            Vec::new()
        } else {
            self.greedy_resume(&capacities, &mut usage, &eligibility, now)
        };
        actions.extend(self.force_timed_out(&capacities, &mut usage, &eligibility, now));
        if !capacities.is_empty() {
            self.pause_until_safe(&capacities, &mut usage, now);
        }
        actions
    }

    fn program_tokens(&self, program: &Program, decayed: bool, now: Instant) -> usize {
        if program.status != ProgramStatus::Acting {
            return program.token_total;
        }
        if !decayed {
            return scale_tokens(program.token_total, self.config.acting_token_weight);
        }
        let idle = program
            .acting_since
            .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
        let weight = 2.0_f64.powf(-idle.as_secs_f64() / self.config.acting_decay_tau_seconds);
        scale_tokens(program.token_total, weight)
    }

    fn worker_usage(&self, now: Instant) -> HashMap<WorkerWithDpRank, WorkerUsage> {
        let mut usage = HashMap::<WorkerWithDpRank, WorkerUsage>::new();
        for program in self.programs.values() {
            if program.lifecycle == ProgramLifecycle::Active
                && let Some(worker) = program.assigned_worker
            {
                let worker_usage = usage.entry(worker).or_default();
                worker_usage.used = worker_usage
                    .used
                    .saturating_add(self.program_tokens(program, false, now))
                    .saturating_add(self.config.buffer_per_program);
                worker_usage.decayed = worker_usage
                    .decayed
                    .saturating_add(self.program_tokens(program, true, now))
                    .saturating_add(self.config.buffer_per_program);
            }
        }
        usage
    }

    fn greedy_resume(
        &mut self,
        capacities: &[WorkerCapacity],
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
        now: Instant,
    ) -> Vec<AdmissionAction> {
        if self.paused.is_empty() {
            return Vec::new();
        }

        let mut paused: Vec<String> = self
            .paused
            .iter()
            .filter(|session_id| self.programs.contains_key(*session_id))
            .cloned()
            .collect();
        paused.sort_by_key(|session_id| {
            let program = &self.programs[session_id];
            let group = if program.step_count <= 1 {
                1
            } else if program.status == ProgramStatus::Reasoning {
                0
            } else {
                2
            };
            (group, program.token_total)
        });

        let ceiling = (self.config.pause_threshold - self.config.resume_hysteresis).max(0.0);
        let mut backend_caps: Vec<(WorkerWithDpRank, usize)> = capacities
            .iter()
            .filter_map(|capacity| {
                let limit = scale_tokens(capacity.tokens, ceiling);
                let remaining =
                    limit.saturating_sub(usage.get(&capacity.worker).map_or(0, |usage| usage.used));
                (remaining > self.config.buffer_per_program).then_some((capacity.worker, remaining))
            })
            .collect();
        sort_backend_caps(&mut backend_caps);
        if backend_caps.is_empty() {
            return Vec::new();
        }

        let total_capacity = backend_caps.iter().fold(0usize, |total, (_, remaining)| {
            total.saturating_add(*remaining)
        });
        let mut cumulative = 0usize;
        let mut resumable = Vec::new();
        for session_id in paused {
            let required = self.programs[&session_id]
                .token_total
                .saturating_add(self.config.buffer_per_program);
            if !backend_caps.iter().any(|(worker, remaining)| {
                self.session_allows_worker(&session_id, *worker, eligibility)
                    && required <= *remaining
            }) {
                continue;
            }
            if cumulative.saturating_add(required) <= total_capacity {
                cumulative = cumulative.saturating_add(required);
                resumable.push(session_id);
            }
        }
        resumable.sort_by_key(|session_id| Reverse(self.programs[session_id].token_total));
        let mut actions = Vec::new();
        for session_id in resumable {
            let Some((position, &(worker, remaining))) =
                backend_caps
                    .iter()
                    .enumerate()
                    .find(|(_, (worker, remaining))| {
                        self.session_allows_worker(&session_id, *worker, eligibility)
                            && self.programs[&session_id]
                                .token_total
                                .saturating_add(self.config.buffer_per_program)
                                <= *remaining
                    })
            else {
                continue;
            };
            let required = self.programs[&session_id]
                .token_total
                .saturating_add(self.config.buffer_per_program);
            actions.extend(self.resume_program(&session_id, Some(worker)));
            let program = &self.programs[&session_id];
            let worker_usage = usage.entry(worker).or_default();
            worker_usage.used = worker_usage
                .used
                .saturating_add(self.program_tokens(program, false, now))
                .saturating_add(self.config.buffer_per_program);
            worker_usage.decayed = worker_usage
                .decayed
                .saturating_add(self.program_tokens(program, true, now))
                .saturating_add(self.config.buffer_per_program);
            let updated = remaining - required;
            if updated > self.config.buffer_per_program {
                backend_caps[position] = (worker, updated);
                sort_backend_caps(&mut backend_caps);
            } else {
                backend_caps.remove(position);
            }
        }
        actions
    }

    fn force_timed_out(
        &mut self,
        capacities: &[WorkerCapacity],
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
        now: Instant,
    ) -> Vec<AdmissionAction> {
        let timeout = Duration::from_secs_f64(self.config.resume_timeout_seconds);
        let timed_out: Vec<String> = self
            .paused
            .iter()
            .filter(|session_id| {
                self.programs
                    .get(*session_id)
                    .and_then(|program| program.deferred.first())
                    .is_some_and(|(_, since)| now.saturating_duration_since(*since) >= timeout)
            })
            .cloned()
            .collect();

        let mut actions = Vec::new();
        for session_id in timed_out {
            if self.programs[&session_id].lifecycle != ProgramLifecycle::Paused {
                continue;
            }
            let target = capacities
                .iter()
                .filter(|capacity| {
                    self.session_allows_worker(&session_id, capacity.worker, eligibility)
                })
                .max_by_key(|capacity| {
                    (
                        capacity.tokens as i128
                            - usage.get(&capacity.worker).map_or(0, |usage| usage.decayed) as i128,
                        Reverse(capacity.worker),
                    )
                })
                .map(|capacity| capacity.worker);
            actions.extend(self.resume_program(&session_id, target));
            if let Some(worker) = target {
                let program = &self.programs[&session_id];
                let worker_usage = usage.entry(worker).or_default();
                worker_usage.used = worker_usage
                    .used
                    .saturating_add(self.program_tokens(program, false, now))
                    .saturating_add(self.config.buffer_per_program);
                worker_usage.decayed = worker_usage
                    .decayed
                    .saturating_add(self.program_tokens(program, true, now))
                    .saturating_add(self.config.buffer_per_program);
            }
        }
        actions
    }

    fn pause_until_safe(
        &mut self,
        capacities: &[WorkerCapacity],
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        now: Instant,
    ) {
        let mut acting = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        let mut reasoning = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        for (session_id, program) in &self.programs {
            if program.lifecycle != ProgramLifecycle::Active || program.marked_for_pause {
                continue;
            }
            let Some(worker) = program.assigned_worker else {
                continue;
            };
            let candidates = match program.status {
                ProgramStatus::Acting => &mut acting,
                ProgramStatus::Reasoning => &mut reasoning,
            };
            candidates
                .entry(worker)
                .or_default()
                .push((program.token_total, session_id.clone()));
        }
        for candidates in acting.values_mut().chain(reasoning.values_mut()) {
            candidates.sort_by_key(|(tokens, _)| *tokens);
        }

        let pause_target = self.config.pause_target;
        for capacity in capacities {
            let threshold = scale_tokens(capacity.tokens, self.config.pause_threshold);
            let worker_used = usage.get(&capacity.worker).map_or(0, |usage| usage.used);
            if worker_used <= threshold {
                continue;
            }
            let target = scale_tokens(capacity.tokens, pause_target);
            if let Some(candidates) = acting.get(&capacity.worker) {
                for (_, session_id) in candidates {
                    if usage.get(&capacity.worker).map_or(0, |usage| usage.used) <= target {
                        break;
                    }
                    let Some(program) = self.programs.get(session_id) else {
                        continue;
                    };
                    let used = self
                        .program_tokens(program, false, now)
                        .saturating_add(self.config.buffer_per_program);
                    let decayed = self
                        .program_tokens(program, true, now)
                        .saturating_add(self.config.buffer_per_program);
                    self.pause_acting(session_id);
                    let worker_usage = usage.entry(capacity.worker).or_default();
                    worker_usage.used = worker_usage.used.saturating_sub(used);
                    worker_usage.decayed = worker_usage.decayed.saturating_sub(decayed);
                }
            }
            if usage.get(&capacity.worker).map_or(0, |usage| usage.used) <= target {
                continue;
            }
            if let Some(candidates) = reasoning.get(&capacity.worker) {
                for (_, session_id) in candidates {
                    if let Some(program) = self.programs.get_mut(session_id) {
                        program.marked_for_pause = true;
                    }
                }
            }
        }
    }

    fn deferred_eligibility_snapshots(&self) -> HashMap<AdmissionId, WorkerEligibilitySnapshot> {
        self.programs
            .values()
            .flat_map(|program| program.deferred.keys())
            .filter_map(|id| {
                self.requests
                    .get(id)?
                    .worker_eligibility
                    .as_ref()
                    .map(|eligibility| (*id, eligibility.snapshot()))
            })
            .collect()
    }

    fn session_allows_worker(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
    ) -> bool {
        let Some(program) = self.programs.get(session_id) else {
            return false;
        };
        program.deferred.keys().all(|id| {
            eligibility
                .get(id)
                .is_none_or(|eligibility| eligibility.allows(worker))
        })
    }

    fn next_resume_deadline(&self) -> Option<Instant> {
        let timeout = Duration::from_secs_f64(self.config.resume_timeout_seconds);
        self.paused
            .iter()
            .filter_map(|session_id| {
                self.programs
                    .get(session_id)
                    .and_then(|program| program.deferred.first())
                    .map(|(_, since)| *since + timeout)
            })
            .min()
    }

    fn pause_acting(&mut self, session_id: &str) {
        let Some(program) = self.programs.get_mut(session_id) else {
            return;
        };
        if program.lifecycle != ProgramLifecycle::Active || program.status != ProgramStatus::Acting
        {
            return;
        }
        program.lifecycle = ProgramLifecycle::Paused;
        program.assigned_worker = None;
        self.paused.insert(session_id.to_owned());
    }

    fn resume_program(
        &mut self,
        session_id: &str,
        worker: Option<WorkerWithDpRank>,
    ) -> Vec<AdmissionAction> {
        let Some(program) = self.programs.get_mut(session_id) else {
            return Vec::new();
        };
        if program.lifecycle != ProgramLifecycle::Paused {
            return Vec::new();
        }
        program.lifecycle = ProgramLifecycle::Active;
        program.assigned_worker = worker;
        let deferred = std::mem::take(&mut program.deferred);
        self.paused.shift_remove(session_id);
        deferred
            .into_keys()
            .map(|id| AdmissionAction::MakeReady {
                id,
                placement: worker.map_or(WorkerPlacement::Any, WorkerPlacement::Exact),
            })
            .collect()
    }
}

impl<P: WorkerCapacityProvider> PolicyClassAdmissionStrategy for ThunderAgent<P> {
    fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
        self.admit_request(request)
    }

    fn on_event(&mut self, event: AdmissionEvent) -> Vec<AdmissionAction> {
        match event {
            AdmissionEvent::Dispatched { id, worker } => {
                self.dispatched(id, worker);
                Vec::new()
            }
            AdmissionEvent::Progress { id, output_tokens } => {
                self.progress(id, output_tokens);
                Vec::new()
            }
            AdmissionEvent::Finished { id, total_tokens } => {
                self.finished(id, total_tokens);
                Vec::new()
            }
            AdmissionEvent::Reconcile => self.reconcile(),
            _ => Vec::new(),
        }
    }

    fn next_reconcile_at(&self) -> Option<Instant> {
        Some(
            self.next_resume_deadline()
                .map_or(self.next_tick, |deadline| deadline.min(self.next_tick)),
        )
    }
}

fn scale_tokens(tokens: usize, factor: f64) -> usize {
    ((tokens as f64) * factor).clamp(0.0, usize::MAX as f64) as usize
}

fn sort_backend_caps(capacities: &mut [(WorkerWithDpRank, usize)]) {
    capacities.sort_unstable_by_key(|(worker, remaining)| (Reverse(*remaining), *worker));
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn request(id: u64, session_id: Option<&str>, input_tokens: usize) -> AdmissionRequest<'_> {
        AdmissionRequest::new(AdmissionId::new(id), session_id, input_tokens)
    }

    fn filtered_request(
        id: u64,
        session_id: Option<&str>,
        input_tokens: usize,
        eligible_workers: impl Fn() -> Vec<WorkerWithDpRank> + Send + Sync + 'static,
    ) -> AdmissionRequest<'_> {
        AdmissionRequest::with_worker_eligibility(
            AdmissionId::new(id),
            session_id,
            input_tokens,
            WorkerEligibility::new(move || WorkerEligibilitySnapshot::new(eligible_workers())),
        )
    }

    fn capacities(values: &[(u64, usize)]) -> Vec<WorkerCapacity> {
        values
            .iter()
            .map(|&(id, tokens)| WorkerCapacity {
                worker: worker(id),
                tokens,
            })
            .collect()
    }

    #[test]
    fn sessionless_requests_do_not_create_program_state() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        assert_eq!(
            strategy.admit(request(1, None, 100)),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        assert!(strategy.programs.is_empty());
    }

    #[test]
    fn new_session_uses_least_loaded_worker_and_sticks() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000), (2, 1_000)]), Default::default())
                .unwrap();
        assert_eq!(
            strategy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Finished {
            id: AdmissionId::new(1),
            total_tokens: 150,
        });
        assert_eq!(strategy.programs["a"].token_total, 150);
        assert_eq!(
            strategy.admit(request(2, Some("a"), 120)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
    }

    #[test]
    fn placement_respects_request_worker_eligibility() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000), (2, 1_000)]), Default::default())
                .unwrap();
        assert_eq!(
            strategy.admit(filtered_request(1, Some("a"), 100, || vec![worker(2)])),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(2)))
        );
    }

    #[test]
    fn deferred_request_rechecks_live_worker_eligibility() {
        let allowed = Arc::new(AtomicBool::new(false));
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        strategy.programs.insert(
            "paused".to_owned(),
            Program {
                lifecycle: ProgramLifecycle::Paused,
                ..Program::default()
            },
        );
        strategy.paused.insert("paused".to_owned());
        let live_allowed = Arc::clone(&allowed);
        assert_eq!(
            strategy.admit(filtered_request(1, Some("paused"), 100, move || {
                live_allowed
                    .load(Ordering::Relaxed)
                    .then(|| worker(1))
                    .into_iter()
                    .collect()
            },)),
            AdmissionDecision::Defer
        );

        strategy.next_tick = Instant::now();
        assert!(strategy.on_event(AdmissionEvent::Reconcile).is_empty());

        allowed.store(true, Ordering::Relaxed);
        strategy.next_tick = Instant::now();
        assert_eq!(
            strategy.on_event(AdmissionEvent::Reconcile),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
    }

    #[test]
    fn sticky_session_reassigns_after_worker_removal() {
        let current = Arc::new(Mutex::new(capacities(&[(1, 1_000), (2, 1_000)])));
        let provider = {
            let current = Arc::clone(&current);
            move || current.lock().unwrap().clone()
        };
        let mut strategy = ThunderAgent::new(provider, Default::default()).unwrap();
        assert_eq!(
            strategy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Finished {
            id: AdmissionId::new(1),
            total_tokens: 150,
        });

        *current.lock().unwrap() = capacities(&[(2, 1_000)]);
        assert_eq!(
            strategy.admit(request(2, Some("a"), 120)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(2)))
        );
    }

    #[test]
    fn sticky_session_drops_disallowed_worker_without_capacity_metadata() {
        let current = Arc::new(Mutex::new(capacities(&[(1, 1_000)])));
        let provider = {
            let current = Arc::clone(&current);
            move || current.lock().unwrap().clone()
        };
        let mut strategy = ThunderAgent::new(provider, Default::default()).unwrap();
        strategy.admit(request(1, Some("a"), 100));
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Finished {
            id: AdmissionId::new(1),
            total_tokens: 150,
        });

        current.lock().unwrap().clear();
        assert_eq!(
            strategy.admit(filtered_request(2, Some("a"), 120, || vec![worker(2)])),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        assert_eq!(strategy.programs["a"].assigned_worker, None);
    }

    #[test]
    fn progress_updates_reasoning_tokens_and_finish_uses_authoritative_total() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        strategy.admit(request(1, Some("a"), 100));
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Progress {
            id: AdmissionId::new(1),
            output_tokens: 25,
        });
        assert_eq!(strategy.programs["a"].token_total, 125);
        strategy.on_event(AdmissionEvent::Finished {
            id: AdmissionId::new(1),
            total_tokens: 140,
        });
        assert_eq!(strategy.programs["a"].token_total, 140);
    }

    #[test]
    fn pressure_pauses_smallest_acting_program_then_resumes_deferred_turn() {
        let current = Arc::new(Mutex::new(capacities(&[(1, 300)])));
        let provider = {
            let current = Arc::clone(&current);
            move || current.lock().unwrap().clone()
        };
        let config = ThunderAgentConfig {
            pause_threshold: 0.8,
            pause_target: 0.5,
            buffer_per_program: 0,
            ..Default::default()
        };
        let mut strategy = ThunderAgent::new(provider, config).unwrap();

        for (id, session, input, output) in [(1, "small", 80, 20), (2, "large", 150, 30)] {
            assert!(matches!(
                strategy.admit(request(id, Some(session), input)),
                AdmissionDecision::Ready(_)
            ));
            strategy.on_event(AdmissionEvent::Dispatched {
                id: AdmissionId::new(id),
                worker: worker(1),
            });
            strategy.on_event(AdmissionEvent::Finished {
                id: AdmissionId::new(id),
                total_tokens: input + output,
            });
        }

        strategy.next_tick = Instant::now();
        assert!(strategy.on_event(AdmissionEvent::Reconcile).is_empty());
        assert_eq!(
            strategy.programs["small"].lifecycle,
            ProgramLifecycle::Paused
        );
        assert_eq!(
            strategy.admit(request(3, Some("small"), 110)),
            AdmissionDecision::Defer
        );

        *current.lock().unwrap() = capacities(&[(1, 600)]);
        strategy.next_tick = Instant::now();
        assert_eq!(
            strategy.on_event(AdmissionEvent::Reconcile),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(3),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
    }

    #[test]
    fn timeout_forces_deferred_request_without_capacity_metadata() {
        let config = ThunderAgentConfig {
            resume_timeout_seconds: 1.0,
            ..Default::default()
        };
        let mut strategy = ThunderAgent::new(Vec::<WorkerCapacity>::new, config).unwrap();
        strategy.programs.insert(
            "paused".to_owned(),
            Program {
                lifecycle: ProgramLifecycle::Paused,
                ..Program::default()
            },
        );
        strategy.paused.insert("paused".to_owned());
        assert_eq!(
            strategy.admit(request(1, Some("paused"), 100)),
            AdmissionDecision::Defer
        );
        *strategy.programs["paused"]
            .deferred
            .get_mut(&AdmissionId::new(1))
            .unwrap() = Instant::now() - Duration::from_secs(2);
        assert_eq!(
            strategy.on_event(AdmissionEvent::Reconcile),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Any,
            }]
        );
    }

    #[test]
    fn forced_resume_chooses_the_least_overfull_worker() {
        let config = ThunderAgentConfig {
            resume_timeout_seconds: 1.0,
            buffer_per_program: 0,
            ..Default::default()
        };
        let mut strategy = ThunderAgent::new(|| capacities(&[(1, 100), (2, 100)]), config).unwrap();
        for (session_id, assigned_worker, token_total) in
            [("worker-1", worker(1), 300), ("worker-2", worker(2), 200)]
        {
            strategy.programs.insert(
                session_id.to_owned(),
                Program {
                    status: ProgramStatus::Acting,
                    assigned_worker: Some(assigned_worker),
                    token_total,
                    ..Program::default()
                },
            );
        }
        strategy.programs.insert(
            "paused".to_owned(),
            Program {
                lifecycle: ProgramLifecycle::Paused,
                ..Program::default()
            },
        );
        strategy.paused.insert("paused".to_owned());
        assert_eq!(
            strategy.admit(request(1, Some("paused"), 50)),
            AdmissionDecision::Defer
        );
        *strategy.programs["paused"]
            .deferred
            .get_mut(&AdmissionId::new(1))
            .unwrap() = Instant::now() - Duration::from_secs(2);

        assert_eq!(
            strategy.on_event(AdmissionEvent::Reconcile),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Exact(worker(2)),
            }]
        );
    }

    #[test]
    fn cancellation_before_dispatch_restores_prior_program_state() {
        let mut strategy =
            ThunderAgent::new(Vec::<WorkerCapacity>::new, Default::default()).unwrap();
        strategy.programs.insert(
            "existing".to_owned(),
            Program {
                status: ProgramStatus::Acting,
                token_total: 200,
                step_count: 3,
                ..Program::default()
            },
        );
        assert!(matches!(
            strategy.admit(request(1, Some("existing"), 300)),
            AdmissionDecision::Ready(_)
        ));
        strategy.on_event(AdmissionEvent::Finished {
            id: AdmissionId::new(1),
            total_tokens: 0,
        });
        let program = &strategy.programs["existing"];
        assert_eq!(program.status, ProgramStatus::Acting);
        assert_eq!(program.token_total, 200);
        assert_eq!(program.step_count, 3);
    }
}
