// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Reverse;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::{
    AdmissionAction, AdmissionDecision, AdmissionEvent, AdmissionId, AdmissionRequest,
    PolicyClassAdmissionStrategy, RequestProgress, WorkerEligibility, WorkerEligibilitySnapshot,
    WorkerPlacement,
};
use indexmap::{IndexMap, IndexSet};

use super::{ConfigError, ThunderAgentConfig, WorkerCapacity, WorkerCapacityProvider};

#[derive(Clone)]
enum ProgramState {
    Running {
        progress: RequestProgress,
        pause_after_completion: bool,
    },
    IdleResident {
        footprint: usize,
        last_activity: Instant,
    },
    Suspended {
        footprint: usize,
        since: Instant,
    },
}

#[derive(Clone)]
struct Program {
    state: ProgramState,
    assigned_worker: Option<WorkerWithDpRank>,
    step_count: usize,
}

impl Program {
    fn running(
        progress: RequestProgress,
        assigned_worker: Option<WorkerWithDpRank>,
        step_count: usize,
    ) -> Self {
        Self {
            state: ProgramState::Running {
                progress,
                pause_after_completion: false,
            },
            assigned_worker,
            step_count,
        }
    }

    fn footprint(&self) -> usize {
        match &self.state {
            ProgramState::Running { progress, .. } => progress.context_tokens(),
            ProgramState::IdleResident { footprint, .. }
            | ProgramState::Suspended { footprint, .. } => *footprint,
        }
    }

    #[cfg(test)]
    fn is_idle_resident(&self) -> bool {
        matches!(self.state, ProgramState::IdleResident { .. })
    }

    fn is_suspended(&self) -> bool {
        matches!(self.state, ProgramState::Suspended { .. })
    }

    fn pause_after_completion(&self) -> bool {
        matches!(
            self.state,
            ProgramState::Running {
                pause_after_completion: true,
                ..
            }
        )
    }

    fn mark_for_pause(&mut self) -> bool {
        let ProgramState::Running {
            pause_after_completion,
            ..
        } = &mut self.state
        else {
            return false;
        };
        !std::mem::replace(pause_after_completion, true)
    }

    fn retained_since(&self) -> Option<Instant> {
        match self.state {
            ProgramState::IdleResident { last_activity, .. } => Some(last_activity),
            ProgramState::Suspended { since, .. } => Some(since),
            ProgramState::Running { .. } => None,
        }
    }

    fn suspended_since(&self) -> Option<Instant> {
        match self.state {
            ProgramState::Suspended { since, .. } => Some(since),
            _ => None,
        }
    }
}

struct RequestState {
    session_id: String,
    progress: RequestProgress,
    worker_eligibility: WorkerEligibility,
    prior: Option<Program>,
}

#[derive(Default)]
struct SessionRequests {
    current: Option<AdmissionId>,
    waiting: IndexSet<AdmissionId>,
}

#[derive(Debug, Default, Clone, Copy)]
struct WorkerUsage {
    used: usize,
}

impl WorkerUsage {
    fn add_program(&mut self, tokens: usize, buffer: usize) {
        self.used = self.used.saturating_add(tokens).saturating_add(buffer);
    }

    fn remove_program(&mut self, tokens: usize, buffer: usize) {
        self.used = self.used.saturating_sub(tokens).saturating_sub(buffer);
    }
}

pub struct ThunderAgent<P> {
    capacity: P,
    config: ThunderAgentConfig,
    programs: IndexMap<String, Program>,
    requests: HashMap<AdmissionId, RequestState>,
    sessions: HashMap<String, SessionRequests>,
    next_tick: Instant,
}

impl<P: WorkerCapacityProvider> ThunderAgent<P> {
    pub fn new(capacity: P, config: ThunderAgentConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        tracing::info!(
            pause_threshold = config.pause_threshold,
            pause_target = config.pause_target,
            resume_hysteresis = config.resume_hysteresis,
            resume_timeout_seconds = config.resume_timeout_seconds,
            session_retention_seconds = config.session_retention_seconds,
            scheduler_interval_seconds = config.scheduler_interval_seconds,
            buffer_per_program = config.buffer_per_program,
            "ThunderAgent admission strategy configured"
        );
        let next_tick = Instant::now() + Duration::from_secs_f64(config.scheduler_interval_seconds);
        Ok(Self {
            capacity,
            config,
            programs: IndexMap::new(),
            requests: HashMap::new(),
            sessions: HashMap::new(),
            next_tick,
        })
    }

    fn admit_request(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
        let Some(session_id) = request.session_id() else {
            return AdmissionDecision::Bypass;
        };

        let now = Instant::now();
        let id = request.id();
        let session_is_busy = self
            .sessions
            .get(session_id)
            .is_some_and(|requests| requests.current.is_some());
        self.requests.insert(
            id,
            RequestState {
                session_id: session_id.to_owned(),
                progress: request.progress().clone(),
                worker_eligibility: request.worker_eligibility().clone(),
                prior: None,
            },
        );
        if session_is_busy {
            self.sessions
                .get_mut(session_id)
                .expect("busy session must exist")
                .waiting
                .insert(id);
            return AdmissionDecision::Defer;
        }

        self.begin_request(id, session_id, now)
    }

    fn begin_request(
        &mut self,
        id: AdmissionId,
        session_id: &str,
        now: Instant,
    ) -> AdmissionDecision {
        let Some(request) = self.requests.get(&id) else {
            return AdmissionDecision::Defer;
        };
        let context_tokens = request.progress.context_tokens();
        let progress = request.progress.clone();
        let worker_eligibility = request.worker_eligibility.clone();
        let capacities = self.capacity.snapshot();
        let capacity_known = !capacities.is_empty();
        let eligibility = worker_eligibility.snapshot();
        let worker_is_available = |worker| eligibility.allows(worker);
        let worker_is_structurally_allowed = |worker| eligibility.structurally_allows(worker);
        let prior = self.programs.get(session_id).cloned();
        let was_new = prior.is_none();
        let was_suspended = prior.as_ref().is_some_and(Program::is_suspended);
        let assigned_worker = prior.as_ref().and_then(|program| program.assigned_worker);
        let step_count = prior
            .as_ref()
            .map_or(1, |program| program.step_count.saturating_add(1));
        let Some(request) = self.requests.get_mut(&id) else {
            return AdmissionDecision::Defer;
        };
        request.prior = prior;
        if let Some(requests) = self.sessions.get_mut(session_id) {
            requests.current = Some(id);
        } else {
            self.sessions.insert(
                session_id.to_owned(),
                SessionRequests {
                    current: Some(id),
                    ..Default::default()
                },
            );
        }
        self.programs.insert(
            session_id.to_owned(),
            Program::running(progress, assigned_worker, step_count),
        );

        if was_suspended {
            self.defer_request(session_id, id, now, true);
            return AdmissionDecision::Defer;
        }

        if let Some(worker) = assigned_worker {
            if worker_is_structurally_allowed(worker) {
                if worker_is_available(worker) {
                    return AdmissionDecision::Ready(WorkerPlacement::Exact(worker));
                }
                self.defer_request(session_id, id, now, true);
                return AdmissionDecision::Defer;
            }
            if let Some(program) = self.programs.get_mut(session_id) {
                program.assigned_worker = None;
            }
        }

        // Do not migrate a session just because every structurally valid worker
        // is temporarily overloaded.
        if eligibility.has_structural_worker() && !eligibility.has_available_worker() {
            self.defer_request(session_id, id, now, false);
            return AdmissionDecision::Defer;
        }

        // Preserve source fairness: a new program cannot bypass an already-paused one.
        if was_new && self.programs.values().any(Program::is_suspended) {
            self.defer_request(session_id, id, now, false);
            return AdmissionDecision::Defer;
        }

        if !capacity_known
            || !capacities
                .iter()
                .any(|capacity| worker_is_available(capacity.worker))
        {
            return AdmissionDecision::Ready(WorkerPlacement::Any);
        }

        let required = context_tokens.saturating_add(self.config.buffer_per_program);
        let usage = self.worker_usage();
        let selected = capacities
            .iter()
            .filter(|capacity| worker_is_available(capacity.worker))
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
            if let Some(program) = self.programs.get_mut(session_id) {
                program.assigned_worker = Some(worker);
            }
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker))
        } else {
            self.defer_request(session_id, id, now, false);
            AdmissionDecision::Defer
        }
    }

    fn defer_request(
        &mut self,
        session_id: &str,
        id: AdmissionId,
        now: Instant,
        preserve_assignment: bool,
    ) {
        let Some(program) = self.programs.get_mut(session_id) else {
            return;
        };
        let footprint = program.footprint();
        program.state = ProgramState::Suspended {
            footprint,
            since: now,
        };
        if !preserve_assignment {
            program.assigned_worker = None;
        }
        debug_assert_eq!(
            self.sessions
                .get(session_id)
                .and_then(|requests| requests.current),
            Some(id)
        );
    }

    fn dispatched(&mut self, id: AdmissionId, worker: WorkerWithDpRank) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        if self
            .sessions
            .get(&request.session_id)
            .and_then(|requests| requests.current)
            != Some(id)
        {
            return;
        }
        if let Some(program) = self.programs.get_mut(&request.session_id) {
            program.assigned_worker = Some(worker);
        }
    }

    fn completed(&mut self, id: AdmissionId, context_tokens: usize) -> Vec<AdmissionAction> {
        self.finish_request(id, Some(context_tokens))
    }

    fn aborted(&mut self, id: AdmissionId) -> Vec<AdmissionAction> {
        self.finish_request(id, None)
    }

    fn finish_request(
        &mut self,
        id: AdmissionId,
        completed_context_tokens: Option<usize>,
    ) -> Vec<AdmissionAction> {
        let Some(request) = self.requests.remove(&id) else {
            return Vec::new();
        };
        if self
            .sessions
            .get(&request.session_id)
            .and_then(|requests| requests.current)
            != Some(id)
        {
            if let Some(requests) = self.sessions.get_mut(&request.session_id) {
                requests.waiting.shift_remove(&id);
            }
            return Vec::new();
        }
        if let Some(requests) = self.sessions.get_mut(&request.session_id) {
            requests.current = None;
        }
        if completed_context_tokens.is_none() {
            match request.prior {
                Some(prior) => {
                    self.programs.insert(request.session_id.clone(), prior);
                }
                None => {
                    self.programs.shift_remove(&request.session_id);
                }
            }
        } else if let Some(context_tokens) = completed_context_tokens
            && let Some(program) = self.programs.get_mut(&request.session_id)
        {
            let pause = program.pause_after_completion();
            program.state = ProgramState::IdleResident {
                footprint: context_tokens,
                last_activity: Instant::now(),
            };
            if pause {
                self.suspend_idle(&request.session_id);
            }
        }

        self.promote_next(&request.session_id)
    }

    fn promote_next(&mut self, session_id: &str) -> Vec<AdmissionAction> {
        let next = self
            .sessions
            .get_mut(session_id)
            .and_then(|requests| requests.waiting.shift_remove_index(0));
        let Some(id) = next else {
            self.sessions.remove(session_id);
            return Vec::new();
        };
        match self.begin_request(id, session_id, Instant::now()) {
            AdmissionDecision::Ready(placement) => {
                vec![AdmissionAction::MakeReady { id, placement }]
            }
            _ => Vec::new(),
        }
    }

    fn reconcile(&mut self) -> Vec<AdmissionAction> {
        let now = Instant::now();
        if now < self.next_tick {
            return Vec::new();
        }
        self.next_tick = now + Duration::from_secs_f64(self.config.scheduler_interval_seconds);
        let expired_programs = self.expire_retained_programs(now);
        let paused_before = self
            .programs
            .values()
            .filter(|program| program.is_suspended())
            .count();
        let marked_before = self
            .programs
            .values()
            .filter(|program| program.pause_after_completion())
            .count();
        let capacities = self.capacity.snapshot();
        let mut usage = self.worker_usage();
        let eligibility = self.deferred_eligibility_snapshots();
        let (mut actions, greedy_resumes) = if capacities.is_empty() {
            (Vec::new(), 0)
        } else {
            self.greedy_resume(&capacities, &mut usage, &eligibility)
        };
        let (forced_actions, forced_resumes) = self.force_timed_out(&eligibility, now);
        actions.extend(forced_actions);
        let (paused_now, marked_now) = if capacities.is_empty() {
            (0, 0)
        } else {
            self.pause_until_safe(&capacities, &mut usage)
        };
        let marked = self
            .programs
            .values()
            .filter(|program| program.pause_after_completion())
            .count();
        let paused = self
            .programs
            .values()
            .filter(|program| program.is_suspended())
            .count();
        if greedy_resumes > 0
            || forced_resumes > 0
            || paused_now > 0
            || marked_now > 0
            || expired_programs > 0
            || paused != paused_before
            || marked != marked_before
        {
            tracing::info!(
                programs = self.programs.len(),
                active = self.programs.len().saturating_sub(paused),
                paused,
                marked,
                greedy_resumed_programs = greedy_resumes,
                forced_resumed_programs = forced_resumes,
                paused_programs = paused_now,
                marked_programs = marked_now,
                expired_programs,
                released_requests = actions.len(),
                capacity_workers = capacities.len(),
                "ThunderAgent admission state changed"
            );
        }
        actions
    }

    fn expire_retained_programs(&mut self, now: Instant) -> usize {
        let retention = Duration::from_secs_f64(self.config.session_retention_seconds);
        let expired: Vec<String> = self
            .programs
            .iter()
            .filter(|(session_id, program)| {
                !self.sessions.contains_key(session_id.as_str())
                    && program
                        .retained_since()
                        .is_some_and(|since| now.saturating_duration_since(since) >= retention)
            })
            .map(|(session_id, _)| session_id.clone())
            .collect();

        for session_id in &expired {
            self.programs.shift_remove(session_id);
        }
        expired.len()
    }

    fn worker_usage(&self) -> HashMap<WorkerWithDpRank, WorkerUsage> {
        let mut usage = HashMap::<WorkerWithDpRank, WorkerUsage>::new();
        for program in self.programs.values() {
            if !program.is_suspended()
                && let Some(worker) = program.assigned_worker
            {
                let worker_usage = usage.entry(worker).or_default();
                worker_usage.add_program(program.footprint(), self.config.buffer_per_program);
            }
        }
        usage
    }

    fn greedy_resume(
        &mut self,
        capacities: &[WorkerCapacity],
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
    ) -> (Vec<AdmissionAction>, usize) {
        let mut paused: Vec<String> = self
            .programs
            .iter()
            .filter(|(_, program)| program.is_suspended())
            .map(|(session_id, _)| session_id.clone())
            .collect();
        if paused.is_empty() {
            return (Vec::new(), 0);
        }
        paused.sort_by_key(|session_id| {
            let program = &self.programs[session_id];
            let group = if program.step_count <= 1 {
                1
            } else if self
                .sessions
                .get(session_id)
                .is_some_and(|requests| requests.current.is_some())
            {
                0
            } else {
                2
            };
            (group, program.footprint())
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
            return (Vec::new(), 0);
        }

        let total_capacity = backend_caps.iter().fold(0usize, |total, (_, remaining)| {
            total.saturating_add(*remaining)
        });
        let mut cumulative = 0usize;
        let mut resumable = Vec::new();
        for session_id in paused {
            let required = self.programs[&session_id]
                .footprint()
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
        resumable.sort_by_key(|session_id| Reverse(self.programs[session_id].footprint()));
        let mut actions = Vec::new();
        let mut resumed = 0;
        for session_id in resumable {
            let Some((position, &(worker, remaining))) =
                backend_caps
                    .iter()
                    .enumerate()
                    .find(|(_, (worker, remaining))| {
                        self.session_allows_worker(&session_id, *worker, eligibility)
                            && self.programs[&session_id]
                                .footprint()
                                .saturating_add(self.config.buffer_per_program)
                                <= *remaining
                    })
            else {
                continue;
            };
            let required = self.programs[&session_id]
                .footprint()
                .saturating_add(self.config.buffer_per_program);
            actions.extend(self.resume_program(&session_id, Some(worker)));
            resumed += 1;
            let program = &self.programs[&session_id];
            let worker_usage = usage.entry(worker).or_default();
            worker_usage.add_program(program.footprint(), self.config.buffer_per_program);
            let updated = remaining - required;
            if updated > self.config.buffer_per_program {
                backend_caps[position] = (worker, updated);
                sort_backend_caps(&mut backend_caps);
            } else {
                backend_caps.remove(position);
            }
        }
        (actions, resumed)
    }

    fn force_timed_out(
        &mut self,
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
        now: Instant,
    ) -> (Vec<AdmissionAction>, usize) {
        let timeout = Duration::from_secs_f64(self.config.resume_timeout_seconds);
        let timed_out: Vec<String> = self
            .programs
            .iter()
            .filter(|(session_id, program)| {
                self.sessions
                    .get(session_id.as_str())
                    .is_some_and(|requests| requests.current.is_some())
                    && program
                        .suspended_since()
                        .is_some_and(|since| now.saturating_duration_since(since) >= timeout)
            })
            .map(|(session_id, _)| session_id.clone())
            .collect();

        let mut actions = Vec::new();
        let mut resumed = 0;
        for session_id in timed_out {
            if self.session_waits_for_available_worker(&session_id, eligibility) {
                continue;
            }
            actions.extend(self.resume_program(&session_id, None));
            resumed += 1;
        }
        (actions, resumed)
    }

    fn pause_until_safe(
        &mut self,
        capacities: &[WorkerCapacity],
        usage: &mut HashMap<WorkerWithDpRank, WorkerUsage>,
    ) -> (usize, usize) {
        let mut acting = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        let mut reasoning = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        for (session_id, program) in &self.programs {
            if program.is_suspended() || program.pause_after_completion() {
                continue;
            }
            let Some(worker) = program.assigned_worker else {
                continue;
            };
            let candidates = match program.state {
                ProgramState::IdleResident { .. } => &mut acting,
                ProgramState::Running { .. } => &mut reasoning,
                ProgramState::Suspended { .. } => continue,
            };
            candidates
                .entry(worker)
                .or_default()
                .push((program.footprint(), session_id.clone()));
        }
        for candidates in acting.values_mut().chain(reasoning.values_mut()) {
            candidates.sort_by_key(|(tokens, _)| *tokens);
        }

        let pause_target = self.config.pause_target;
        let mut paused_total = 0;
        let mut marked_total = 0;
        for capacity in capacities {
            let threshold = scale_tokens(capacity.tokens, self.config.pause_threshold);
            let worker_used = usage.get(&capacity.worker).map_or(0, |usage| usage.used);
            if worker_used <= threshold {
                continue;
            }
            let target = scale_tokens(capacity.tokens, pause_target);
            let mut paused = 0;
            let mut marked = 0;
            if let Some(candidates) = acting.get(&capacity.worker) {
                for (_, session_id) in candidates {
                    if usage.get(&capacity.worker).map_or(0, |usage| usage.used) <= target {
                        break;
                    }
                    let Some(program) = self.programs.get(session_id) else {
                        continue;
                    };
                    let used = program.footprint();
                    self.suspend_idle(session_id);
                    paused += 1;
                    let worker_usage = usage.entry(capacity.worker).or_default();
                    worker_usage.remove_program(used, self.config.buffer_per_program);
                }
            }
            if usage.get(&capacity.worker).map_or(0, |usage| usage.used) > target
                && let Some(candidates) = reasoning.get(&capacity.worker)
            {
                for (_, session_id) in candidates {
                    if let Some(program) = self.programs.get_mut(session_id) {
                        marked += usize::from(program.mark_for_pause());
                    }
                }
            }
            let used_after = usage.get(&capacity.worker).map_or(0, |usage| usage.used);
            paused_total += paused;
            marked_total += marked;
            tracing::info!(
                worker_id = capacity.worker.worker_id,
                dp_rank = capacity.worker.dp_rank,
                capacity_tokens = capacity.tokens,
                used_before = worker_used,
                used_after,
                threshold_tokens = threshold,
                target_tokens = target,
                paused_programs = paused,
                marked_programs = marked,
                "ThunderAgent worker pressure handled"
            );
        }
        (paused_total, marked_total)
    }

    fn deferred_eligibility_snapshots(&self) -> HashMap<AdmissionId, WorkerEligibilitySnapshot> {
        self.programs
            .iter()
            .filter(|(_, program)| program.is_suspended())
            .filter_map(|(session_id, _)| {
                let id = self.sessions.get(session_id)?.current?;
                let request = self.requests.get(&id)?;
                Some((id, request.worker_eligibility.snapshot()))
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
        if program
            .assigned_worker
            .is_some_and(|assigned| assigned != worker)
        {
            return false;
        }
        if !program.is_suspended()
            || self
                .sessions
                .get(session_id)
                .is_none_or(|requests| requests.current.is_none())
        {
            return true;
        }
        self.sessions
            .get(session_id)
            .and_then(|requests| requests.current)
            .and_then(|id| eligibility.get(&id))
            .is_none_or(|eligibility| eligibility.allows(worker))
    }

    fn session_waits_for_available_worker(
        &self,
        session_id: &str,
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
    ) -> bool {
        let Some(program) = self.programs.get(session_id) else {
            return false;
        };
        if !program.is_suspended() {
            return false;
        }
        self.sessions
            .get(session_id)
            .and_then(|requests| requests.current)
            .and_then(|id| eligibility.get(&id))
            .is_some_and(|snapshot| {
                snapshot.has_structural_worker() && !snapshot.has_available_worker()
            })
    }

    fn suspend_idle(&mut self, session_id: &str) {
        let Some(program) = self.programs.get_mut(session_id) else {
            return;
        };
        let ProgramState::IdleResident {
            footprint,
            last_activity,
        } = program.state
        else {
            return;
        };
        program.state = ProgramState::Suspended {
            footprint,
            since: last_activity,
        };
    }

    fn resume_program(
        &mut self,
        session_id: &str,
        worker: Option<WorkerWithDpRank>,
    ) -> Vec<AdmissionAction> {
        let deferred_id = self
            .sessions
            .get(session_id)
            .and_then(|requests| requests.current);
        let deferred_progress = match deferred_id {
            Some(id) => {
                let Some(request) = self.requests.get(&id) else {
                    return Vec::new();
                };
                Some(request.progress.clone())
            }
            None => None,
        };
        let Some(program) = self.programs.get_mut(session_id) else {
            return Vec::new();
        };
        let ProgramState::Suspended { footprint, since } = program.state else {
            return Vec::new();
        };
        program.state = match deferred_progress {
            Some(progress) => ProgramState::Running {
                progress,
                pause_after_completion: false,
            },
            None => ProgramState::IdleResident {
                footprint,
                last_activity: since,
            },
        };
        program.assigned_worker = worker;
        match deferred_id {
            Some(id) => vec![AdmissionAction::MakeReady {
                id,
                placement: worker.map_or(WorkerPlacement::Any, WorkerPlacement::Exact),
            }],
            None => Vec::new(),
        }
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
            AdmissionEvent::Completed { id, context_tokens } => self.completed(id, context_tokens),
            AdmissionEvent::Aborted { id } => self.aborted(id),
            AdmissionEvent::Reconcile => self.reconcile(),
            _ => Vec::new(),
        }
    }

    fn reconcile_interval(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            self.config.scheduler_interval_seconds,
        ))
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

    fn request(id: u64, session_id: Option<&str>, context_tokens: usize) -> AdmissionRequest<'_> {
        AdmissionRequest::new(
            AdmissionId::new(id),
            session_id,
            context_tokens,
            WorkerEligibility::new(|| WorkerEligibilitySnapshot::new([worker(1), worker(2)])),
        )
    }

    fn filtered_request(
        id: u64,
        session_id: Option<&str>,
        context_tokens: usize,
        eligible_workers: impl Fn() -> Vec<WorkerWithDpRank> + Send + Sync + 'static,
    ) -> AdmissionRequest<'_> {
        AdmissionRequest::new(
            AdmissionId::new(id),
            session_id,
            context_tokens,
            WorkerEligibility::new(move || WorkerEligibilitySnapshot::new(eligible_workers())),
        )
    }

    fn availability_request(
        id: u64,
        session_id: Option<&str>,
        context_tokens: usize,
        workers: impl Fn() -> (Vec<WorkerWithDpRank>, Vec<WorkerWithDpRank>) + Send + Sync + 'static,
    ) -> AdmissionRequest<'_> {
        AdmissionRequest::new(
            AdmissionId::new(id),
            session_id,
            context_tokens,
            WorkerEligibility::new(move || {
                let (structural, available) = workers();
                WorkerEligibilitySnapshot::with_availability(
                    structural.into_iter().collect(),
                    available.into_iter().collect(),
                )
            }),
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

    fn idle_program(
        assigned_worker: Option<WorkerWithDpRank>,
        footprint: usize,
        last_activity: Instant,
        step_count: usize,
    ) -> Program {
        Program {
            state: ProgramState::IdleResident {
                footprint,
                last_activity,
            },
            assigned_worker,
            step_count,
        }
    }

    fn suspended_program(footprint: usize, since: Instant, step_count: usize) -> Program {
        Program {
            state: ProgramState::Suspended { footprint, since },
            assigned_worker: None,
            step_count,
        }
    }

    fn running_program(
        assigned_worker: Option<WorkerWithDpRank>,
        footprint: usize,
        step_count: usize,
    ) -> Program {
        let (progress, _) = RequestProgress::new(footprint);
        Program::running(progress, assigned_worker, step_count)
    }

    fn set_suspended_since(program: &mut Program, value: Instant) {
        let ProgramState::Suspended { since, .. } = &mut program.state else {
            panic!("program must be suspended");
        };
        *since = value;
    }

    fn reconcile_now<P: WorkerCapacityProvider>(
        strategy: &mut ThunderAgent<P>,
    ) -> Vec<AdmissionAction> {
        strategy.next_tick = Instant::now();
        strategy.on_event(AdmissionEvent::Reconcile)
    }

    #[test]
    fn sessionless_requests_do_not_create_program_state() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        assert_eq!(
            strategy.admit(request(1, None, 100)),
            AdmissionDecision::Bypass
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
        strategy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });
        assert!(strategy.programs["a"].is_idle_resident());
        assert_eq!(strategy.programs["a"].footprint(), 150);
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
        strategy
            .programs
            .insert("paused".to_owned(), suspended_program(0, Instant::now(), 0));
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

        assert!(reconcile_now(&mut strategy).is_empty());

        allowed.store(true, Ordering::Relaxed);
        assert_eq!(
            reconcile_now(&mut strategy),
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
        strategy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });

        *current.lock().unwrap() = capacities(&[(2, 1_000)]);
        assert_eq!(
            strategy.admit(filtered_request(2, Some("a"), 120, || vec![worker(2)])),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(2)))
        );
    }

    #[test]
    fn sticky_session_waits_for_transiently_unavailable_worker() {
        let available = Arc::new(AtomicBool::new(true));
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000), (2, 1_000)]), Default::default())
                .unwrap();
        let snapshot = {
            let available = Arc::clone(&available);
            move || {
                let structural = vec![worker(1), worker(2)];
                let available = if available.load(Ordering::Relaxed) {
                    structural.clone()
                } else {
                    vec![worker(2)]
                };
                (structural, available)
            }
        };
        assert_eq!(
            strategy.admit(availability_request(1, Some("a"), 100, snapshot)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });

        available.store(false, Ordering::Relaxed);
        let snapshot = {
            let available = Arc::clone(&available);
            move || {
                let structural = vec![worker(1), worker(2)];
                let available = if available.load(Ordering::Relaxed) {
                    structural.clone()
                } else {
                    vec![worker(2)]
                };
                (structural, available)
            }
        };
        assert_eq!(
            strategy.admit(availability_request(2, Some("a"), 160, snapshot)),
            AdmissionDecision::Defer
        );
        assert_eq!(strategy.programs["a"].assigned_worker, Some(worker(1)));
        assert!(reconcile_now(&mut strategy).is_empty());

        available.store(true, Ordering::Relaxed);
        assert_eq!(
            reconcile_now(&mut strategy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(2),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
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
        strategy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });

        current.lock().unwrap().clear();
        assert_eq!(
            strategy.admit(filtered_request(2, Some("a"), 120, || vec![worker(2)])),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        assert_eq!(strategy.programs["a"].assigned_worker, None);
    }

    #[test]
    fn completion_commits_authoritative_context_tokens() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        strategy.admit(request(1, Some("a"), 100));
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 140,
        });
        assert!(strategy.programs["a"].is_idle_resident());
        assert_eq!(strategy.programs["a"].footprint(), 140);
    }

    #[test]
    fn live_progress_updates_inflight_reasoning_context_tokens() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        strategy.admit(request(1, Some("a"), 100));
        let (progress, updater) = RequestProgress::new(100);
        strategy.programs["a"].state = ProgramState::Running {
            progress,
            pause_after_completion: false,
        };
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        updater.update_context_tokens(128);

        let program = &strategy.programs["a"];
        assert!(matches!(program.state, ProgramState::Running { .. }));
        assert_eq!(program.footprint(), 128);

        strategy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 140,
        });
        assert!(strategy.programs["a"].is_idle_resident());
        assert_eq!(strategy.programs["a"].footprint(), 140);
    }

    #[test]
    fn retention_expiry_removes_only_quiescent_acting_programs() {
        let config = ThunderAgentConfig {
            session_retention_seconds: 900.0,
            buffer_per_program: 0,
            ..Default::default()
        };
        let mut strategy = ThunderAgent::new(Vec::<WorkerCapacity>::new, config).unwrap();
        let now = Instant::now();
        let expired_at = now - Duration::from_secs(900);

        for (session_id, program) in [
            (
                "expired-active",
                idle_program(Some(worker(1)), 100, expired_at, 0),
            ),
            ("expired-paused", suspended_program(200, expired_at, 0)),
            (
                "fresh",
                idle_program(
                    Some(worker(1)),
                    300,
                    expired_at + Duration::from_nanos(1),
                    0,
                ),
            ),
            ("reasoning", running_program(Some(worker(1)), 400, 0)),
            ("busy", idle_program(Some(worker(1)), 500, expired_at, 0)),
        ] {
            strategy.programs.insert(session_id.to_owned(), program);
        }
        strategy.sessions.insert(
            "busy".to_owned(),
            SessionRequests {
                current: Some(AdmissionId::new(99)),
                ..Default::default()
            },
        );

        assert_eq!(strategy.expire_retained_programs(now), 2);
        assert!(!strategy.programs.contains_key("expired-active"));
        assert!(!strategy.programs.contains_key("expired-paused"));
        assert!(strategy.programs.contains_key("fresh"));
        assert!(strategy.programs.contains_key("reasoning"));
        assert!(strategy.programs.contains_key("busy"));

        let usage = strategy.worker_usage();
        assert_eq!(usage[&worker(1)].used, 1_200);
    }

    #[test]
    fn expired_session_returns_as_a_new_program() {
        let config = ThunderAgentConfig {
            session_retention_seconds: 900.0,
            ..Default::default()
        };
        let mut strategy = ThunderAgent::new(|| capacities(&[(1, 1_000)]), config).unwrap();
        strategy.programs.insert(
            "expired".to_owned(),
            idle_program(
                Some(worker(2)),
                700,
                Instant::now() - Duration::from_secs(901),
                4,
            ),
        );

        assert!(reconcile_now(&mut strategy).is_empty());
        assert!(!strategy.programs.contains_key("expired"));
        assert_eq!(
            strategy.admit(request(100, Some("expired"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(strategy.programs["expired"].step_count, 1);
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
            strategy.on_event(AdmissionEvent::Completed {
                id: AdmissionId::new(id),
                context_tokens: input + output,
            });
        }

        assert!(strategy.on_event(AdmissionEvent::Reconcile).is_empty());
        assert!(strategy.programs["small"].is_idle_resident());
        assert!(reconcile_now(&mut strategy).is_empty());
        assert!(strategy.programs["small"].is_suspended());
        assert_eq!(strategy.programs["small"].assigned_worker, Some(worker(1)));
        assert_eq!(
            strategy.admit(request(3, Some("small"), 110)),
            AdmissionDecision::Defer
        );
        assert_eq!(strategy.programs["small"].assigned_worker, Some(worker(1)));

        *current.lock().unwrap() = capacities(&[(1, 600)]);
        assert_eq!(
            reconcile_now(&mut strategy),
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
        strategy
            .programs
            .insert("paused".to_owned(), suspended_program(0, Instant::now(), 0));
        assert_eq!(
            strategy.admit(request(1, Some("paused"), 100)),
            AdmissionDecision::Defer
        );
        set_suspended_since(
            &mut strategy.programs["paused"],
            Instant::now() - Duration::from_secs(2),
        );
        assert_eq!(
            reconcile_now(&mut strategy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Any,
            }]
        );
    }

    #[test]
    fn forced_resume_delegates_worker_selection() {
        let config = ThunderAgentConfig {
            resume_timeout_seconds: 1.0,
            buffer_per_program: 0,
            ..Default::default()
        };
        let mut strategy = ThunderAgent::new(|| capacities(&[(1, 100), (2, 100)]), config).unwrap();
        for (session_id, assigned_worker, footprint) in
            [("worker-1", worker(1), 300), ("worker-2", worker(2), 200)]
        {
            strategy.programs.insert(
                session_id.to_owned(),
                idle_program(Some(assigned_worker), footprint, Instant::now(), 0),
            );
        }
        strategy
            .programs
            .insert("paused".to_owned(), suspended_program(0, Instant::now(), 0));
        assert_eq!(
            strategy.admit(request(1, Some("paused"), 50)),
            AdmissionDecision::Defer
        );
        set_suspended_since(
            &mut strategy.programs["paused"],
            Instant::now() - Duration::from_secs(2),
        );

        assert_eq!(
            reconcile_now(&mut strategy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Any,
            }]
        );
    }

    #[test]
    fn cancellation_before_dispatch_restores_prior_program_state() {
        let mut strategy =
            ThunderAgent::new(Vec::<WorkerCapacity>::new, Default::default()).unwrap();
        strategy.programs.insert(
            "existing".to_owned(),
            idle_program(None, 200, Instant::now(), 3),
        );
        assert!(matches!(
            strategy.admit(request(1, Some("existing"), 300)),
            AdmissionDecision::Ready(_)
        ));
        strategy.on_event(AdmissionEvent::Aborted {
            id: AdmissionId::new(1),
        });
        let program = &strategy.programs["existing"];
        assert!(program.is_idle_resident());
        assert_eq!(program.footprint(), 200);
        assert_eq!(program.step_count, 3);
    }

    #[test]
    fn cancellation_after_dispatch_restores_prior_program_state() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        strategy.programs.insert(
            "existing".to_owned(),
            idle_program(Some(worker(1)), 200, Instant::now(), 3),
        );
        strategy.admit(request(1, Some("existing"), 300));
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Aborted {
            id: AdmissionId::new(1),
        });

        let program = &strategy.programs["existing"];
        assert!(program.is_idle_resident());
        assert_eq!(program.footprint(), 200);
        assert_eq!(program.step_count, 3);
    }

    #[test]
    fn concurrent_session_request_waits_without_mutating_program() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        assert_eq!(
            strategy.admit(request(1, Some("same"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(
            strategy.admit(request(2, Some("same"), 900)),
            AdmissionDecision::Defer
        );
        assert!(matches!(
            strategy.programs["same"].state,
            ProgramState::Running { .. }
        ));
        assert_eq!(strategy.programs["same"].footprint(), 100);
        assert_eq!(strategy.programs["same"].step_count, 1);

        assert!(
            strategy
                .on_event(AdmissionEvent::Aborted {
                    id: AdmissionId::new(2),
                })
                .is_empty()
        );
        strategy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        strategy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });
        assert!(strategy.programs["same"].is_idle_resident());
        assert_eq!(strategy.programs["same"].footprint(), 150);
        assert_eq!(strategy.programs["same"].step_count, 1);
    }

    #[test]
    fn cancelling_current_session_request_promotes_one_waiter() {
        let mut strategy =
            ThunderAgent::new(|| capacities(&[(1, 1_000)]), Default::default()).unwrap();
        assert!(matches!(
            strategy.admit(request(1, Some("same"), 100)),
            AdmissionDecision::Ready(_)
        ));
        assert_eq!(
            strategy.admit(request(2, Some("same"), 120)),
            AdmissionDecision::Defer
        );

        assert_eq!(
            strategy.on_event(AdmissionEvent::Aborted {
                id: AdmissionId::new(1),
            }),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(2),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
        assert!(matches!(
            strategy.programs["same"].state,
            ProgramState::Running { .. }
        ));
        assert_eq!(strategy.programs["same"].footprint(), 120);
        assert_eq!(strategy.programs["same"].step_count, 1);
        assert_eq!(strategy.sessions["same"].current, Some(AdmissionId::new(2)));
    }
}
