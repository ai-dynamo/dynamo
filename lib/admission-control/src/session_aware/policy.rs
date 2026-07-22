// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Reverse;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::{
    AdmissionAction, AdmissionDecision, AdmissionEvent, AdmissionId, AdmissionRequest,
    PolicyClassAdmissionPolicy, RequestProgress, WorkerEligibility, WorkerEligibilitySnapshot,
    WorkerPlacement,
};
use indexmap::{IndexMap, IndexSet};

use super::{
    ConfigError, SessionAwareAdmissionControlConfig, WorkerCapacity, WorkerCapacityProvider,
};

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

    fn buffered_tokens(&self, buffer_per_program: usize) -> usize {
        self.footprint().saturating_add(buffer_per_program)
    }

    fn active_accounted_tokens(
        &self,
        acting_token_weight: f64,
        buffer_per_program: usize,
    ) -> usize {
        let tokens = match &self.state {
            ProgramState::IdleResident { footprint, .. } => {
                (*footprint as f64 * acting_token_weight) as usize
            }
            _ => self.footprint(),
        };
        tokens.saturating_add(buffer_per_program)
    }

    fn decayed_accounted_tokens(
        &self,
        now: Instant,
        acting_decay_tau_seconds: f64,
        buffer_per_program: usize,
    ) -> usize {
        let tokens = match &self.state {
            ProgramState::IdleResident {
                footprint,
                last_activity,
            } => {
                let idle = now.saturating_duration_since(*last_activity).as_secs_f64();
                let tau = acting_decay_tau_seconds.max(1e-3);
                (*footprint as f64 * 2.0_f64.powf(-(idle / tau))) as usize
            }
            _ => self.footprint(),
        };
        tokens.saturating_add(buffer_per_program)
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
    session_final: bool,
    progress: RequestProgress,
    worker_eligibility: WorkerEligibility,
    prior: Option<Program>,
}

#[derive(Default)]
struct SessionRequests {
    current: Option<AdmissionId>,
    waiting: IndexSet<AdmissionId>,
}

pub struct SessionAwareAdmissionControl<P> {
    capacity: P,
    config: SessionAwareAdmissionControlConfig,
    programs: IndexMap<String, Program>,
    requests: HashMap<AdmissionId, RequestState>,
    sessions: HashMap<String, SessionRequests>,
    next_tick: Instant,
}

impl<P: WorkerCapacityProvider> SessionAwareAdmissionControl<P> {
    pub fn new(
        capacity: P,
        config: SessionAwareAdmissionControlConfig,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        tracing::info!(
            pause_threshold = config.pause_threshold,
            pause_target = config.pause_target,
            resume_hysteresis = config.resume_hysteresis,
            resume_timeout_seconds = config.resume_timeout_seconds,
            session_retention_seconds = config.session_retention_seconds,
            scheduler_interval_seconds = config.scheduler_interval_seconds,
            acting_token_weight = config.acting_token_weight,
            acting_decay_tau_seconds = config.acting_decay_tau_seconds,
            buffer_per_program = config.buffer_per_program,
            "Session-aware admission control configured"
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

    fn admit_request(&mut self, mut request: AdmissionRequest) -> AdmissionDecision {
        let Some(session) = request.take_session() else {
            return AdmissionDecision::Bypass;
        };
        let session_final = session.is_final_request();
        let session_id = session.into_session_id();

        let now = Instant::now();
        let id = request.id();
        let session_is_busy = self
            .sessions
            .get(session_id.as_str())
            .is_some_and(|requests| requests.current.is_some());
        self.requests.insert(
            id,
            RequestState {
                session_id: session_id.clone(),
                session_final,
                progress: request.progress().clone(),
                worker_eligibility: request.worker_eligibility().clone(),
                prior: None,
            },
        );
        if session_is_busy {
            self.sessions
                .get_mut(session_id.as_str())
                .expect("busy session must exist")
                .waiting
                .insert(id);
            return AdmissionDecision::Defer;
        }

        self.begin_request(id, &session_id, now)
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
        if request.session_final {
            return self.begin_session_final(id, session_id, worker_eligibility);
        }
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

        // Existing active continuations are the common path. Match the source
        // scheduler's immediate admission without taking a capacity snapshot or
        // scanning the full program table on every turn.
        let eligibility = worker_eligibility.snapshot();
        let worker_is_available = |worker| eligibility.allows(worker);
        let worker_is_structurally_allowed = |worker| eligibility.structurally_allows(worker);

        if let Some(worker) = assigned_worker {
            if worker_is_structurally_allowed(worker) {
                if worker_is_available(worker) {
                    // An assigned session already belongs to the active working set.
                    // Match ThunderAgent: dispatch its next turn and let periodic
                    // pressure handling choose victims after admission.
                    return AdmissionDecision::Ready(WorkerPlacement::Exact(worker));
                }
                self.defer_request(session_id, id, now, true);
                return AdmissionDecision::Defer;
            }
            if let Some(program) = self.programs.get_mut(session_id) {
                program.assigned_worker = None;
            }
        }

        let capacities = self.capacity.snapshot();
        let usage = self.worker_usages();

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

        if !capacities
            .iter()
            .any(|capacity| worker_is_available(capacity.worker))
        {
            return AdmissionDecision::Ready(WorkerPlacement::Any);
        }

        let selected = capacities
            .iter()
            .filter(|capacity| worker_is_available(capacity.worker))
            .filter_map(|capacity| {
                let used = usage.get(&capacity.worker).copied().unwrap_or(0);
                fits_worker_capacity(
                    capacity,
                    used,
                    context_tokens,
                    self.config.buffer_per_program,
                )
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

    fn begin_session_final(
        &mut self,
        id: AdmissionId,
        session_id: &str,
        worker_eligibility: WorkerEligibility,
    ) -> AdmissionDecision {
        let assigned_worker = self
            .programs
            .get(session_id)
            .and_then(|program| program.assigned_worker);
        self.programs.shift_remove(session_id);
        if let Some(request) = self.requests.get_mut(&id) {
            // A terminal request releases the prior program at admission time.
            // Completion or abort must not restore it.
            request.prior = None;
        }
        self.sessions
            .entry(session_id.to_owned())
            .or_default()
            .current = Some(id);
        tracing::info!(%session_id, "Session-aware admission released final session");

        if let Some(worker) = assigned_worker
            && worker_eligibility.snapshot().structurally_allows(worker)
        {
            return AdmissionDecision::Ready(WorkerPlacement::Exact(worker));
        }
        AdmissionDecision::Ready(WorkerPlacement::Any)
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
        if request.session_final {
            self.programs.shift_remove(&request.session_id);
            return self.promote_next(&request.session_id);
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
        let eligibility = self.deferred_eligibility_snapshots();
        let cleared_stale_assignments = self.clear_stale_assignments(&eligibility);
        let (mut actions, unmetered_resumes) = self.resume_unmetered(&capacities, &eligibility);
        let mut usage = self.worker_usages();
        let (greedy_actions, greedy_resumes) = if capacities.is_empty() {
            (Vec::new(), 0)
        } else {
            self.greedy_resume(&capacities, &mut usage, &eligibility)
        };
        actions.extend(greedy_actions);
        let (forced_actions, forced_resumes) = self.force_timed_out(&capacities, &eligibility, now);
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
            || unmetered_resumes > 0
            || forced_resumes > 0
            || paused_now > 0
            || marked_now > 0
            || expired_programs > 0
            || cleared_stale_assignments > 0
            || paused != paused_before
            || marked != marked_before
        {
            tracing::info!(
                programs = self.programs.len(),
                active = self.programs.len().saturating_sub(paused),
                paused,
                marked,
                greedy_resumed_programs = greedy_resumes,
                unmetered_resumed_programs = unmetered_resumes,
                forced_resumed_programs = forced_resumes,
                paused_programs = paused_now,
                marked_programs = marked_now,
                expired_programs,
                cleared_stale_assignments,
                released_requests = actions.len(),
                capacity_workers = capacities.len(),
                "Session-aware admission-control state changed"
            );
        }
        actions
    }

    fn expire_retained_programs(&mut self, now: Instant) -> usize {
        let retention = Duration::from_secs_f64(self.config.session_retention_seconds);
        let sessions = &self.sessions;
        let before = self.programs.len();
        self.programs.retain(|session_id, program| {
            sessions.contains_key(session_id.as_str())
                || program
                    .retained_since()
                    .is_none_or(|since| now.saturating_duration_since(since) < retention)
        });
        before - self.programs.len()
    }

    fn worker_usages(&self) -> HashMap<WorkerWithDpRank, usize> {
        let mut total = HashMap::<WorkerWithDpRank, usize>::new();
        for program in self.programs.values() {
            if !program.is_suspended()
                && let Some(worker) = program.assigned_worker
            {
                let used = total.entry(worker).or_default();
                *used = used.saturating_add(program.active_accounted_tokens(
                    self.config.acting_token_weight,
                    self.config.buffer_per_program,
                ));
            }
        }
        total
    }

    fn resume_group(&self, session_id: &str) -> usize {
        let program = &self.programs[session_id];
        if program.step_count <= 1 {
            1
        } else if self
            .sessions
            .get(session_id)
            .is_some_and(|requests| requests.current.is_some())
        {
            0
        } else {
            2
        }
    }

    fn greedy_resume(
        &mut self,
        capacities: &[WorkerCapacity],
        usage: &mut HashMap<WorkerWithDpRank, usize>,
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
        let ceiling = self.config.pause_threshold - self.config.resume_hysteresis;
        let mut backend_caps: Vec<(WorkerWithDpRank, usize)> = capacities
            .iter()
            .filter_map(|capacity| {
                let limit = scale_tokens(capacity.tokens, ceiling);
                let remaining =
                    limit.saturating_sub(usage.get(&capacity.worker).copied().unwrap_or(0));
                (remaining > self.config.buffer_per_program).then_some((capacity.worker, remaining))
            })
            .collect();
        sort_backend_caps(&mut backend_caps);
        if backend_caps.is_empty() {
            return (Vec::new(), 0);
        }
        let original_backend_caps = backend_caps.clone();
        let all_flexible = paused.iter().all(|session_id| {
            backend_caps
                .iter()
                .all(|(worker, _)| self.session_allows_worker(session_id, *worker, eligibility))
        });

        let mut fallback_assignments = None;
        let mut selected = if all_flexible {
            // ThunderAgent first chooses a candidate set in group/smallest order
            // against one scalar cluster-wide budget.
            paused.sort_by_key(|session_id| {
                (
                    self.resume_group(session_id),
                    self.programs[session_id].footprint(),
                )
            });
            let total_capacity = backend_caps
                .iter()
                .map(|(_, remaining)| *remaining)
                .fold(0usize, usize::saturating_add);
            let mut cumulative = 0usize;
            paused
                .into_iter()
                .filter_map(|session_id| {
                    let required =
                        self.programs[&session_id].buffered_tokens(self.config.buffer_per_program);
                    (cumulative.saturating_add(required) <= total_capacity).then(|| {
                        cumulative = cumulative.saturating_add(required);
                        (session_id, required)
                    })
                })
                .collect::<Vec<_>>()
        } else {
            // Preserve request eligibility by reserving a concrete worker while
            // selecting whenever the candidate set is not fully flexible.
            paused.sort_by_key(|session_id| {
                let required =
                    self.programs[session_id].buffered_tokens(self.config.buffer_per_program);
                let eligible_workers = backend_caps
                    .iter()
                    .filter(|(worker, remaining)| {
                        self.session_allows_worker(session_id, *worker, eligibility)
                            && required <= *remaining
                    })
                    .count();
                (
                    self.resume_group(session_id),
                    eligible_workers,
                    self.programs[session_id].footprint(),
                )
            });
            let mut selection_caps = backend_caps.clone();
            let mut assignments = HashMap::new();
            let mut selected = Vec::new();
            for session_id in paused {
                let required =
                    self.programs[&session_id].buffered_tokens(self.config.buffer_per_program);
                let Some((position, &(worker, _))) =
                    selection_caps
                        .iter()
                        .enumerate()
                        .find(|(_, (worker, remaining))| {
                            self.session_allows_worker(&session_id, *worker, eligibility)
                                && required <= *remaining
                        })
                else {
                    continue;
                };
                reserve_backend_capacity(
                    &mut selection_caps,
                    position,
                    required,
                    self.config.buffer_per_program,
                );
                assignments.insert(session_id.clone(), worker);
                selected.push((session_id, required));
            }
            fallback_assignments = Some(assignments);
            selected
        };

        // ThunderAgent drops group ordering after selection and packs the
        // chosen programs largest-first onto the worker with most room.
        selected.sort_by_key(|(session_id, _)| Reverse(self.programs[session_id].footprint()));
        let mut packed_backend_caps = original_backend_caps;
        let mut assignments = HashMap::with_capacity(selected.len());
        let mut repack_succeeded = true;
        for (session_id, required) in &selected {
            let Some((position, &(worker, _))) =
                packed_backend_caps
                    .iter()
                    .enumerate()
                    .find(|(_, (worker, remaining))| {
                        self.session_allows_worker(session_id, *worker, eligibility)
                            && *required <= *remaining
                    })
            else {
                if fallback_assignments.is_some() {
                    repack_succeeded = false;
                    break;
                }
                continue;
            };
            assignments.insert(session_id.clone(), worker);
            reserve_backend_capacity(
                &mut packed_backend_caps,
                position,
                *required,
                self.config.buffer_per_program,
            );
        }
        if !repack_succeeded {
            assignments = fallback_assignments.expect("constrained selection has assignments");
        }

        let mut actions = Vec::new();
        let mut resumed = 0;
        for (session_id, required) in selected {
            let Some(&worker) = assignments.get(&session_id) else {
                continue;
            };
            actions.extend(self.resume_program(&session_id, Some(worker)));
            resumed += 1;
            let used = usage.entry(worker).or_default();
            *used = used.saturating_add(required);
        }
        (actions, resumed)
    }

    fn resume_unmetered(
        &mut self,
        capacities: &[WorkerCapacity],
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
    ) -> (Vec<AdmissionAction>, usize) {
        let suspended: Vec<String> = self
            .programs
            .iter()
            .filter(|(_, program)| program.is_suspended())
            .map(|(session_id, _)| session_id.clone())
            .collect();
        let mut actions = Vec::new();
        let mut resumed = 0;
        for session_id in suspended {
            let current = self
                .sessions
                .get(&session_id)
                .and_then(|requests| requests.current);
            let assigned_worker = self.programs[&session_id].assigned_worker;
            let worker = match current.and_then(|id| eligibility.get(&id)) {
                Some(snapshot) => {
                    if !snapshot.has_available_worker() {
                        continue;
                    }
                    match assigned_worker {
                        Some(worker) if snapshot.structurally_allows(worker) => {
                            if !snapshot.allows(worker) {
                                continue;
                            }
                            if capacities.iter().any(|capacity| capacity.worker == worker) {
                                continue;
                            }
                            Some(worker)
                        }
                        _ => {
                            if capacities
                                .iter()
                                .any(|capacity| snapshot.allows(capacity.worker))
                            {
                                continue;
                            }
                            None
                        }
                    }
                }
                None => {
                    if assigned_worker.is_some_and(|worker| {
                        capacities.iter().any(|capacity| capacity.worker == worker)
                    }) || assigned_worker.is_none() && !capacities.is_empty()
                    {
                        continue;
                    }
                    assigned_worker
                }
            };
            actions.extend(self.resume_program(&session_id, worker));
            resumed += 1;
        }
        (actions, resumed)
    }

    fn force_timed_out(
        &mut self,
        capacities: &[WorkerCapacity],
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
            let worker = self.least_loaded_worker(&session_id, capacities, eligibility, now);
            if worker.is_none() && !capacities.is_empty() {
                continue;
            }
            actions.extend(self.resume_program(&session_id, worker));
            resumed += 1;
        }
        (actions, resumed)
    }

    fn least_loaded_worker(
        &self,
        session_id: &str,
        capacities: &[WorkerCapacity],
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
        now: Instant,
    ) -> Option<WorkerWithDpRank> {
        capacities
            .iter()
            .filter(|capacity| self.session_allows_worker(session_id, capacity.worker, eligibility))
            .max_by_key(|capacity| {
                let used = self
                    .programs
                    .values()
                    .filter(|program| {
                        !program.is_suspended() && program.assigned_worker == Some(capacity.worker)
                    })
                    .fold(0usize, |used, program| {
                        used.saturating_add(program.decayed_accounted_tokens(
                            now,
                            self.config.acting_decay_tau_seconds,
                            self.config.buffer_per_program,
                        ))
                    });
                (
                    capacity.tokens as i128 - used as i128,
                    Reverse(capacity.worker),
                )
            })
            .map(|capacity| capacity.worker)
    }

    fn pause_until_safe(
        &mut self,
        capacities: &[WorkerCapacity],
        usage: &mut HashMap<WorkerWithDpRank, usize>,
    ) -> (usize, usize) {
        let mut acting = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        let mut reasoning = HashMap::<WorkerWithDpRank, Vec<(usize, String)>>::new();
        for (session_id, program) in &self.programs {
            if program.is_suspended() {
                continue;
            }
            let Some(worker) = program.assigned_worker else {
                continue;
            };
            if program.pause_after_completion() {
                continue;
            }
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
            let worker_used = usage.get(&capacity.worker).copied().unwrap_or(0);
            if worker_used <= threshold {
                continue;
            }
            let target = scale_tokens(capacity.tokens, pause_target);
            let mut paused = 0;
            let mut marked = 0;
            if let Some(candidates) = acting.get(&capacity.worker) {
                for (_, session_id) in candidates {
                    if usage.get(&capacity.worker).copied().unwrap_or(0) <= target {
                        break;
                    }
                    let Some(program) = self.programs.get(session_id) else {
                        continue;
                    };
                    let used = program.active_accounted_tokens(
                        self.config.acting_token_weight,
                        self.config.buffer_per_program,
                    );
                    self.suspend_idle(session_id);
                    paused += 1;
                    let worker_used = usage.entry(capacity.worker).or_default();
                    *worker_used = worker_used.saturating_sub(used);
                }
            }
            if usage.get(&capacity.worker).copied().unwrap_or(0) > target
                && let Some(candidates) = reasoning.get(&capacity.worker)
            {
                // Marking a running program does not lower current utilization.
                // Match ThunderAgent's loop: once idle pauses are exhausted, every
                // remaining running program on the overloaded worker is marked.
                for (_, session_id) in candidates {
                    if let Some(program) = self.programs.get_mut(session_id)
                        && program.mark_for_pause()
                    {
                        marked += 1;
                    }
                }
            }
            let used_after = usage.get(&capacity.worker).copied().unwrap_or(0);
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
                "Session-aware admission-control worker pressure handled"
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

    fn clear_stale_assignments(
        &mut self,
        eligibility: &HashMap<AdmissionId, WorkerEligibilitySnapshot>,
    ) -> usize {
        let mut cleared = 0;
        for (session_id, program) in &mut self.programs {
            let Some(worker) = program.assigned_worker else {
                continue;
            };
            let Some(id) = self
                .sessions
                .get(session_id.as_str())
                .and_then(|requests| requests.current)
            else {
                continue;
            };
            if eligibility
                .get(&id)
                .is_some_and(|snapshot| !snapshot.structurally_allows(worker))
            {
                program.assigned_worker = None;
                cleared += 1;
            }
        }
        cleared
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
        program.assigned_worker = None;
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

impl<P: WorkerCapacityProvider> PolicyClassAdmissionPolicy for SessionAwareAdmissionControl<P> {
    fn admit(&mut self, request: AdmissionRequest) -> AdmissionDecision {
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

fn fits_worker_capacity(
    capacity: &WorkerCapacity,
    total_used: usize,
    request_tokens: usize,
    buffer_per_program: usize,
) -> bool {
    capacity
        .tokens
        .checked_sub(total_used)
        .is_some_and(|remaining| remaining >= request_tokens.saturating_add(buffer_per_program))
}

fn sort_backend_caps(capacities: &mut [(WorkerWithDpRank, usize)]) {
    capacities.sort_unstable_by_key(|(worker, remaining)| (Reverse(*remaining), *worker));
}

fn reserve_backend_capacity(
    capacities: &mut Vec<(WorkerWithDpRank, usize)>,
    position: usize,
    required: usize,
    buffer_per_program: usize,
) {
    let (worker, remaining) = capacities[position];
    debug_assert!(required <= remaining);
    let updated = remaining - required;
    if updated <= buffer_per_program {
        capacities.remove(position);
    } else {
        capacities[position] = (worker, updated);
        sort_backend_caps(capacities);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use dynamo_kv_router::scheduling::AdmissionSession;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn request(id: u64, session_id: Option<&str>, context_tokens: usize) -> AdmissionRequest {
        AdmissionRequest::new(
            AdmissionId::new(id),
            session_id.map(|id| AdmissionSession::new(id, false)),
            context_tokens,
            WorkerEligibility::new(|| WorkerEligibilitySnapshot::new([worker(1), worker(2)])),
        )
    }

    fn final_request(id: u64, session_id: Option<&str>, context_tokens: usize) -> AdmissionRequest {
        AdmissionRequest::new(
            AdmissionId::new(id),
            session_id.map(|id| AdmissionSession::new(id, true)),
            context_tokens,
            WorkerEligibility::new(|| WorkerEligibilitySnapshot::new([worker(1), worker(2)])),
        )
    }

    fn filtered_request(
        id: u64,
        session_id: Option<&str>,
        context_tokens: usize,
        eligible_workers: impl Fn() -> Vec<WorkerWithDpRank> + Send + Sync + 'static,
    ) -> AdmissionRequest {
        AdmissionRequest::new(
            AdmissionId::new(id),
            session_id.map(|id| AdmissionSession::new(id, false)),
            context_tokens,
            WorkerEligibility::new(move || WorkerEligibilitySnapshot::new(eligible_workers())),
        )
    }

    fn availability_request(
        id: u64,
        session_id: Option<&str>,
        context_tokens: usize,
        workers: impl Fn() -> (Vec<WorkerWithDpRank>, Vec<WorkerWithDpRank>) + Send + Sync + 'static,
    ) -> AdmissionRequest {
        AdmissionRequest::new(
            AdmissionId::new(id),
            session_id.map(|id| AdmissionSession::new(id, false)),
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
                device_tokens: tokens,
                tokens,
            })
            .collect()
    }

    fn tiered_capacities(values: &[(u64, usize, usize)]) -> Vec<WorkerCapacity> {
        values
            .iter()
            .map(|&(id, device_tokens, tokens)| WorkerCapacity {
                worker: worker(id),
                device_tokens,
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
        policy: &mut SessionAwareAdmissionControl<P>,
    ) -> Vec<AdmissionAction> {
        policy.next_tick = Instant::now();
        policy.on_event(AdmissionEvent::Reconcile)
    }

    #[test]
    fn sessionless_requests_do_not_create_program_state() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        assert_eq!(
            policy.admit(request(1, None, 100)),
            AdmissionDecision::Bypass
        );
        assert!(policy.programs.is_empty());
    }

    #[test]
    fn new_session_uses_least_loaded_worker_and_sticks() {
        let mut policy = SessionAwareAdmissionControl::new(
            || capacities(&[(1, 1_000), (2, 1_000)]),
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });
        assert!(policy.programs["a"].is_idle_resident());
        assert_eq!(policy.programs["a"].footprint(), 150);
        assert_eq!(
            policy.admit(request(2, Some("a"), 120)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
    }

    #[test]
    fn active_session_continuation_bypasses_retained_capacity() {
        let mut policy = SessionAwareAdmissionControl::new(
            || tiered_capacities(&[(1, 300, 300)]),
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 70)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 70,
        });
        assert_eq!(
            policy.admit(request(2, Some("b"), 30)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(2),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(2),
            context_tokens: 30,
        });

        assert_eq!(
            policy.admit(request(3, Some("a"), 80)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(
            policy.admit(request(4, Some("c"), 1)),
            AdmissionDecision::Defer
        );
    }

    #[test]
    fn active_session_continuation_skips_capacity_snapshot() {
        let snapshots = Arc::new(AtomicUsize::new(0));
        let observed_snapshots = Arc::clone(&snapshots);
        let mut policy = SessionAwareAdmissionControl::new(
            move || {
                observed_snapshots.fetch_add(1, Ordering::Relaxed);
                capacities(&[(1, 1_000)])
            },
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 100,
        });
        let before_continuation = snapshots.load(Ordering::Relaxed);

        assert_eq!(
            policy.admit(request(2, Some("a"), 120)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(snapshots.load(Ordering::Relaxed), before_continuation);
    }

    #[test]
    fn active_session_continuation_bypasses_device_capacity() {
        let mut policy = SessionAwareAdmissionControl::new(
            || tiered_capacities(&[(1, 100, 1_000)]),
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 40)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 40,
        });
        assert_eq!(
            policy.admit(request(2, Some("b"), 80)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );

        assert_eq!(
            policy.admit(request(3, Some("a"), 30)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(policy.programs["a"].assigned_worker, Some(worker(1)));
        assert!(!policy.programs["a"].is_suspended());
    }

    #[test]
    fn new_backend_work_uses_combined_retention_capacity() {
        let mut policy = SessionAwareAdmissionControl::new(
            || tiered_capacities(&[(1, 100, 1_000)]),
            Default::default(),
        )
        .unwrap();

        assert_eq!(
            policy.admit(request(1, Some("running"), 80)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(
            policy.admit(request(2, Some("waiting"), 30)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert!(!policy.programs["waiting"].is_suspended());
    }

    #[test]
    fn placement_respects_request_worker_eligibility() {
        let mut policy = SessionAwareAdmissionControl::new(
            || capacities(&[(1, 1_000), (2, 1_000)]),
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            policy.admit(filtered_request(1, Some("a"), 100, || vec![worker(2)])),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(2)))
        );
    }

    #[test]
    fn new_program_reserves_the_configured_token_buffer() {
        let mut full =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 199)]), Default::default())
                .unwrap();
        assert_eq!(
            full.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Defer
        );

        let mut fits =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 200)]), Default::default())
                .unwrap();
        assert_eq!(
            fits.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
    }

    #[test]
    fn acting_weight_scales_only_idle_working_set_tokens() {
        let config = SessionAwareAdmissionControlConfig {
            acting_token_weight: 0.5,
            buffer_per_program: 10,
            ..Default::default()
        };
        let mut policy =
            SessionAwareAdmissionControl::new(Vec::<WorkerCapacity>::new, config).unwrap();
        policy.programs.insert(
            "acting".to_owned(),
            idle_program(Some(worker(1)), 100, Instant::now(), 1),
        );
        policy.programs.insert(
            "reasoning".to_owned(),
            running_program(Some(worker(1)), 100, 1),
        );

        let usage = policy.worker_usages();

        assert_eq!(usage[&worker(1)], 170);
    }

    #[test]
    fn deferred_request_rechecks_live_worker_eligibility() {
        let allowed = Arc::new(AtomicBool::new(false));
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        policy
            .programs
            .insert("paused".to_owned(), suspended_program(0, Instant::now(), 0));
        let live_allowed = Arc::clone(&allowed);
        assert_eq!(
            policy.admit(filtered_request(1, Some("paused"), 100, move || {
                live_allowed
                    .load(Ordering::Relaxed)
                    .then(|| worker(1))
                    .into_iter()
                    .collect()
            },)),
            AdmissionDecision::Defer
        );

        assert!(reconcile_now(&mut policy).is_empty());

        allowed.store(true, Ordering::Relaxed);
        assert_eq!(
            reconcile_now(&mut policy),
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
        let mut policy = SessionAwareAdmissionControl::new(provider, Default::default()).unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });

        *current.lock().unwrap() = capacities(&[(2, 1_000)]);
        assert_eq!(
            policy.admit(filtered_request(2, Some("a"), 120, || vec![worker(2)])),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(2)))
        );
    }

    #[test]
    fn sticky_session_waits_for_transiently_unavailable_worker() {
        let available = Arc::new(AtomicBool::new(true));
        let mut policy = SessionAwareAdmissionControl::new(
            || capacities(&[(1, 1_000), (2, 1_000)]),
            Default::default(),
        )
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
            policy.admit(availability_request(1, Some("a"), 100, snapshot)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
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
            policy.admit(availability_request(2, Some("a"), 160, snapshot)),
            AdmissionDecision::Defer
        );
        assert_eq!(policy.programs["a"].assigned_worker, Some(worker(1)));
        assert!(reconcile_now(&mut policy).is_empty());

        available.store(true, Ordering::Relaxed);
        assert_eq!(
            reconcile_now(&mut policy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(2),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
    }

    #[test]
    fn deferred_sticky_session_reassigns_after_worker_removal() {
        let mut policy = SessionAwareAdmissionControl::new(
            || capacities(&[(1, 1_000), (2, 1_000)]),
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });

        let workers = Arc::new(Mutex::new((vec![worker(1), worker(2)], vec![worker(2)])));
        let snapshot = {
            let workers = Arc::clone(&workers);
            move || workers.lock().unwrap().clone()
        };
        assert_eq!(
            policy.admit(availability_request(2, Some("a"), 160, snapshot)),
            AdmissionDecision::Defer
        );
        assert_eq!(policy.programs["a"].assigned_worker, Some(worker(1)));

        *workers.lock().unwrap() = (vec![worker(2)], vec![worker(2)]);
        assert_eq!(
            reconcile_now(&mut policy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(2),
                placement: WorkerPlacement::Exact(worker(2)),
            }]
        );
        assert_eq!(policy.programs["a"].assigned_worker, Some(worker(2)));
    }

    #[test]
    fn sticky_session_drops_disallowed_worker_without_capacity_metadata() {
        let current = Arc::new(Mutex::new(capacities(&[(1, 1_000)])));
        let provider = {
            let current = Arc::clone(&current);
            move || current.lock().unwrap().clone()
        };
        let mut policy = SessionAwareAdmissionControl::new(provider, Default::default()).unwrap();
        policy.admit(request(1, Some("a"), 100));
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });

        current.lock().unwrap().clear();
        assert_eq!(
            policy.admit(filtered_request(2, Some("a"), 120, || vec![worker(2)])),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        assert_eq!(policy.programs["a"].assigned_worker, None);
    }

    #[test]
    fn completion_commits_authoritative_context_tokens() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        policy.admit(request(1, Some("a"), 100));
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 140,
        });
        assert!(policy.programs["a"].is_idle_resident());
        assert_eq!(policy.programs["a"].footprint(), 140);
    }

    #[test]
    fn session_final_releases_program_and_routes_normally() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 140,
        });

        assert_eq!(
            policy.admit(final_request(2, Some("a"), 1)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert!(!policy.programs.contains_key("a"));

        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(2),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(2),
            context_tokens: 1,
        });
        assert!(!policy.programs.contains_key("a"));

        assert_eq!(
            policy.admit(final_request(3, Some("a"), 1)),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        policy.on_event(AdmissionEvent::Aborted {
            id: AdmissionId::new(3),
        });
        assert!(!policy.programs.contains_key("a"));
    }

    #[test]
    fn session_final_waits_for_inflight_turn_before_release() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("a"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });

        assert_eq!(
            policy.admit(final_request(2, Some("a"), 1)),
            AdmissionDecision::Defer
        );
        assert!(policy.programs.contains_key("a"));

        assert_eq!(
            policy.on_event(AdmissionEvent::Completed {
                id: AdmissionId::new(1),
                context_tokens: 140,
            }),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(2),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
        assert!(!policy.programs.contains_key("a"));

        policy.on_event(AdmissionEvent::Aborted {
            id: AdmissionId::new(2),
        });
        assert!(!policy.programs.contains_key("a"));
    }

    #[test]
    fn live_progress_updates_inflight_reasoning_context_tokens() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        policy.admit(request(1, Some("a"), 100));
        let (progress, updater) = RequestProgress::new(100);
        policy.programs["a"].state = ProgramState::Running {
            progress,
            pause_after_completion: false,
        };
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        updater.update_context_tokens(128);

        let program = &policy.programs["a"];
        assert!(matches!(program.state, ProgramState::Running { .. }));
        assert_eq!(program.footprint(), 128);

        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 140,
        });
        assert!(policy.programs["a"].is_idle_resident());
        assert_eq!(policy.programs["a"].footprint(), 140);
    }

    #[test]
    fn retention_expiry_removes_only_quiescent_acting_programs() {
        let config = SessionAwareAdmissionControlConfig {
            session_retention_seconds: 900.0,
            ..Default::default()
        };
        let mut policy =
            SessionAwareAdmissionControl::new(Vec::<WorkerCapacity>::new, config).unwrap();
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
            policy.programs.insert(session_id.to_owned(), program);
        }
        policy.sessions.insert(
            "busy".to_owned(),
            SessionRequests {
                current: Some(AdmissionId::new(99)),
                ..Default::default()
            },
        );

        assert_eq!(policy.expire_retained_programs(now), 2);
        assert!(!policy.programs.contains_key("expired-active"));
        assert!(!policy.programs.contains_key("expired-paused"));
        assert!(policy.programs.contains_key("fresh"));
        assert!(policy.programs.contains_key("reasoning"));
        assert!(policy.programs.contains_key("busy"));

        let usage = policy.worker_usages();
        assert_eq!(usage[&worker(1)], 1_500);
    }

    #[test]
    fn expired_session_returns_as_a_new_program() {
        let config = SessionAwareAdmissionControlConfig {
            session_retention_seconds: 900.0,
            ..Default::default()
        };
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), config).unwrap();
        policy.programs.insert(
            "expired".to_owned(),
            idle_program(
                Some(worker(2)),
                700,
                Instant::now() - Duration::from_secs(901),
                4,
            ),
        );

        assert!(reconcile_now(&mut policy).is_empty());
        assert!(!policy.programs.contains_key("expired"));
        assert_eq!(
            policy.admit(request(100, Some("expired"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(policy.programs["expired"].step_count, 1);
    }

    #[test]
    fn pressure_pause_clears_affinity_and_repacks_deferred_turn() {
        let current = Arc::new(Mutex::new(capacities(&[(1, 500)])));
        let provider = {
            let current = Arc::clone(&current);
            move || current.lock().unwrap().clone()
        };
        let config = SessionAwareAdmissionControlConfig {
            pause_threshold: 0.8,
            pause_target: 0.5,
            ..Default::default()
        };
        let mut policy = SessionAwareAdmissionControl::new(provider, config).unwrap();

        for (id, session, input, output) in [(1, "small", 80, 20), (2, "large", 150, 30)] {
            assert!(matches!(
                policy.admit(request(id, Some(session), input)),
                AdmissionDecision::Ready(_)
            ));
            policy.on_event(AdmissionEvent::Dispatched {
                id: AdmissionId::new(id),
                worker: worker(1),
            });
            policy.on_event(AdmissionEvent::Completed {
                id: AdmissionId::new(id),
                context_tokens: input + output,
            });
        }

        assert!(policy.on_event(AdmissionEvent::Reconcile).is_empty());
        assert!(policy.programs["small"].is_idle_resident());
        assert!(reconcile_now(&mut policy).is_empty());
        assert!(policy.programs["small"].is_suspended());
        assert_eq!(policy.programs["small"].assigned_worker, None);
        assert_eq!(
            policy.admit(request(3, Some("small"), 110)),
            AdmissionDecision::Defer
        );
        assert_eq!(policy.programs["small"].assigned_worker, None);

        *current.lock().unwrap() = capacities(&[(1, 200), (2, 430)]);
        assert_eq!(
            reconcile_now(&mut policy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(3),
                placement: WorkerPlacement::Exact(worker(2)),
            }]
        );
    }

    #[test]
    fn resume_selection_respects_per_worker_eligibility() {
        let mut policy = SessionAwareAdmissionControl::new(
            || capacities(&[(1, 300), (2, 300)]),
            Default::default(),
        )
        .unwrap();
        for session_id in ["a", "b", "c"] {
            policy.programs.insert(
                session_id.to_owned(),
                suspended_program(0, Instant::now(), 2),
            );
        }
        assert_eq!(
            policy.admit(filtered_request(1, Some("a"), 80, || vec![worker(1)])),
            AdmissionDecision::Defer
        );
        assert_eq!(
            policy.admit(filtered_request(2, Some("b"), 80, || vec![worker(1)])),
            AdmissionDecision::Defer
        );
        assert_eq!(
            policy.admit(filtered_request(3, Some("c"), 80, || vec![worker(2)])),
            AdmissionDecision::Defer
        );

        assert_eq!(
            reconcile_now(&mut policy),
            vec![
                AdmissionAction::MakeReady {
                    id: AdmissionId::new(1),
                    placement: WorkerPlacement::Exact(worker(1)),
                },
                AdmissionAction::MakeReady {
                    id: AdmissionId::new(3),
                    placement: WorkerPlacement::Exact(worker(2)),
                },
            ]
        );
        assert!(policy.programs["b"].is_suspended());
    }

    #[test]
    fn resume_selection_reassigns_flexible_work_for_a_constrained_candidate() {
        let mut policy = SessionAwareAdmissionControl::new(
            || capacities(&[(1, 300), (2, 300)]),
            Default::default(),
        )
        .unwrap();
        for session_id in ["flexible", "constrained"] {
            policy.programs.insert(
                session_id.to_owned(),
                suspended_program(0, Instant::now(), 2),
            );
        }
        assert_eq!(
            policy.admit(request(1, Some("flexible"), 80)),
            AdmissionDecision::Defer
        );
        assert_eq!(
            policy.admit(filtered_request(2, Some("constrained"), 80, || {
                vec![worker(1)]
            })),
            AdmissionDecision::Defer
        );

        assert_eq!(
            reconcile_now(&mut policy),
            vec![
                AdmissionAction::MakeReady {
                    id: AdmissionId::new(2),
                    placement: WorkerPlacement::Exact(worker(1)),
                },
                AdmissionAction::MakeReady {
                    id: AdmissionId::new(1),
                    placement: WorkerPlacement::Exact(worker(2)),
                },
            ]
        );
    }

    #[test]
    fn greedy_resume_selects_scalar_prefix_then_packs_largest_first() {
        let config = SessionAwareAdmissionControlConfig {
            pause_threshold: 1.0,
            resume_hysteresis: 0.0,
            ..Default::default()
        };
        let mut policy =
            SessionAwareAdmissionControl::new(Vec::<WorkerCapacity>::new, config).unwrap();
        for (session_id, footprint) in [("small", 1), ("medium", 100), ("large", 201)] {
            policy.programs.insert(
                session_id.to_owned(),
                suspended_program(footprint, Instant::now(), 2),
            );
        }
        let capacity = capacities(&[(1, 301), (2, 301)]);

        let (actions, resumed) =
            policy.greedy_resume(&capacity, &mut HashMap::new(), &HashMap::new());

        assert!(actions.is_empty());
        assert_eq!(resumed, 3);
        assert_eq!(policy.programs["large"].assigned_worker, Some(worker(1)));
        assert_eq!(policy.programs["medium"].assigned_worker, Some(worker(2)));
        assert_eq!(policy.programs["small"].assigned_worker, Some(worker(2)));
    }

    #[test]
    fn pressure_marks_every_running_context_once_idle_pauses_are_exhausted() {
        let config = SessionAwareAdmissionControlConfig {
            pause_threshold: 0.8,
            pause_target: 0.8,
            ..Default::default()
        };
        let mut policy =
            SessionAwareAdmissionControl::new(Vec::<WorkerCapacity>::new, config).unwrap();
        policy
            .programs
            .insert("large".to_owned(), running_program(Some(worker(1)), 600, 1));
        policy
            .programs
            .insert("small".to_owned(), running_program(Some(worker(1)), 300, 1));
        let capacity = [WorkerCapacity {
            worker: worker(1),
            device_tokens: 1_000,
            tokens: 1_000,
        }];

        let mut usage = policy.worker_usages();
        assert_eq!(policy.pause_until_safe(&capacity, &mut usage), (0, 2));
        assert!(policy.programs["small"].pause_after_completion());
        assert!(policy.programs["large"].pause_after_completion());

        let mut usage = policy.worker_usages();
        assert_eq!(policy.pause_until_safe(&capacity, &mut usage), (0, 0));
        assert!(policy.programs["large"].pause_after_completion());
    }

    #[test]
    fn missing_capacity_metadata_immediately_releases_deferred_request() {
        let mut policy =
            SessionAwareAdmissionControl::new(Vec::<WorkerCapacity>::new, Default::default())
                .unwrap();
        policy
            .programs
            .insert("paused".to_owned(), suspended_program(0, Instant::now(), 0));
        assert_eq!(
            policy.admit(request(1, Some("paused"), 100)),
            AdmissionDecision::Defer
        );
        assert_eq!(
            reconcile_now(&mut policy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Any,
            }]
        );
    }

    #[test]
    fn missing_capacity_metadata_does_not_bypass_transient_overload() {
        let mut policy =
            SessionAwareAdmissionControl::new(Vec::<WorkerCapacity>::new, Default::default())
                .unwrap();
        policy
            .programs
            .insert("paused".to_owned(), suspended_program(0, Instant::now(), 0));
        assert_eq!(
            policy.admit(availability_request(1, Some("paused"), 100, || {
                (vec![worker(1)], Vec::new())
            })),
            AdmissionDecision::Defer
        );

        assert!(reconcile_now(&mut policy).is_empty());
        assert!(policy.programs["paused"].is_suspended());
    }

    #[test]
    fn missing_capacity_for_sticky_worker_releases_exact_with_mixed_metadata() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(2, 1_000)]), Default::default())
                .unwrap();
        let mut paused = suspended_program(0, Instant::now(), 2);
        paused.assigned_worker = Some(worker(1));
        policy.programs.insert("paused".to_owned(), paused);
        assert_eq!(
            policy.admit(request(1, Some("paused"), 100)),
            AdmissionDecision::Defer
        );

        assert_eq!(
            reconcile_now(&mut policy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
    }

    #[test]
    fn forced_resume_pins_to_least_loaded_worker_with_acting_decay() {
        let config = SessionAwareAdmissionControlConfig {
            resume_timeout_seconds: 1.0,
            ..Default::default()
        };
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 100), (2, 100)]), config)
                .unwrap();
        let now = Instant::now();
        for (session_id, assigned_worker, footprint, last_activity) in [
            ("worker-1", worker(1), 300, now - Duration::from_secs(10)),
            ("worker-2", worker(2), 200, now),
        ] {
            policy.programs.insert(
                session_id.to_owned(),
                idle_program(Some(assigned_worker), footprint, last_activity, 0),
            );
        }
        policy
            .programs
            .insert("paused".to_owned(), suspended_program(0, Instant::now(), 0));
        assert_eq!(
            policy.admit(request(1, Some("paused"), 50)),
            AdmissionDecision::Defer
        );
        set_suspended_since(
            &mut policy.programs["paused"],
            Instant::now() - Duration::from_secs(2),
        );

        assert_eq!(
            reconcile_now(&mut policy),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(1),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
    }

    #[test]
    fn cancellation_before_dispatch_restores_prior_program_state() {
        let mut policy =
            SessionAwareAdmissionControl::new(Vec::<WorkerCapacity>::new, Default::default())
                .unwrap();
        policy.programs.insert(
            "existing".to_owned(),
            idle_program(None, 200, Instant::now(), 3),
        );
        assert!(matches!(
            policy.admit(request(1, Some("existing"), 300)),
            AdmissionDecision::Ready(_)
        ));
        policy.on_event(AdmissionEvent::Aborted {
            id: AdmissionId::new(1),
        });
        let program = &policy.programs["existing"];
        assert!(program.is_idle_resident());
        assert_eq!(program.footprint(), 200);
        assert_eq!(program.step_count, 3);
    }

    #[test]
    fn cancellation_after_dispatch_restores_prior_program_state() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        policy.programs.insert(
            "existing".to_owned(),
            idle_program(Some(worker(1)), 200, Instant::now(), 3),
        );
        policy.admit(request(1, Some("existing"), 300));
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Aborted {
            id: AdmissionId::new(1),
        });

        let program = &policy.programs["existing"];
        assert!(program.is_idle_resident());
        assert_eq!(program.footprint(), 200);
        assert_eq!(program.step_count, 3);
    }

    #[test]
    fn concurrent_session_request_waits_without_mutating_program() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        assert_eq!(
            policy.admit(request(1, Some("same"), 100)),
            AdmissionDecision::Ready(WorkerPlacement::Exact(worker(1)))
        );
        assert_eq!(
            policy.admit(request(2, Some("same"), 900)),
            AdmissionDecision::Defer
        );
        assert!(matches!(
            policy.programs["same"].state,
            ProgramState::Running { .. }
        ));
        assert_eq!(policy.programs["same"].footprint(), 100);
        assert_eq!(policy.programs["same"].step_count, 1);

        assert!(
            policy
                .on_event(AdmissionEvent::Aborted {
                    id: AdmissionId::new(2),
                })
                .is_empty()
        );
        policy.on_event(AdmissionEvent::Dispatched {
            id: AdmissionId::new(1),
            worker: worker(1),
        });
        policy.on_event(AdmissionEvent::Completed {
            id: AdmissionId::new(1),
            context_tokens: 150,
        });
        assert!(policy.programs["same"].is_idle_resident());
        assert_eq!(policy.programs["same"].footprint(), 150);
        assert_eq!(policy.programs["same"].step_count, 1);
    }

    #[test]
    fn cancelling_current_session_request_promotes_one_waiter() {
        let mut policy =
            SessionAwareAdmissionControl::new(|| capacities(&[(1, 1_000)]), Default::default())
                .unwrap();
        assert!(matches!(
            policy.admit(request(1, Some("same"), 100)),
            AdmissionDecision::Ready(_)
        ));
        assert_eq!(
            policy.admit(request(2, Some("same"), 120)),
            AdmissionDecision::Defer
        );

        assert_eq!(
            policy.on_event(AdmissionEvent::Aborted {
                id: AdmissionId::new(1),
            }),
            vec![AdmissionAction::MakeReady {
                id: AdmissionId::new(2),
                placement: WorkerPlacement::Exact(worker(1)),
            }]
        );
        assert!(matches!(
            policy.programs["same"].state,
            ProgramState::Running { .. }
        ));
        assert_eq!(policy.programs["same"].footprint(), 120);
        assert_eq!(policy.programs["same"].step_count, 1);
        assert_eq!(policy.sessions["same"].current, Some(AdmissionId::new(2)));
    }
}
