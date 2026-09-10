// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Co-locates the subagents of one parent session.
//!
//! Dynamo binds session affinity to each request's own session id. A subagent carries its own
//! session id, so the subagents of one parent receive independent bindings and scatter across the
//! worker pool even though they all replay the parent's system prompt, tool definitions, and
//! context. This policy keeps a second binding keyed on the *parent* session id and steers any
//! request that carries one by it, so siblings share a worker and reuse that prefix.
//!
//! A new group is placed on the least-loaded eligible worker, not on the parent's own worker: a
//! parent typically holds its worker for the whole session, so sending its fan-out there adds a
//! burst of siblings to a worker that is already busy. A bound group moves only when its worker
//! exceeds `max_active_requests` *and* a strictly less loaded worker exists. The strict comparison
//! is load-bearing: moving to an equally loaded worker would relocate the group on every sibling
//! once the threshold is crossed, so it would oscillate across the pool and lose the prefix it
//! exists to reuse.
//!
//! Requests without a parent session id keep Dynamo's own session target under the same threshold.
//!
//! # Operating requirements
//!
//! Run with `--router-session-affinity-mode soft`. The default `hard` mode passes a bound session
//! as a pinned target, which narrows the candidate set to one worker before any policy runs, so a
//! returning subagent could never be free to join its parent's group.
//!
//! # Known limits
//!
//! The group binding is recorded during selection, because a picker has no post-dispatch callback.
//! A request that is later cancelled, fails scheduler booking, or fails to dispatch can therefore
//! leave the group pointing at a worker that never received the prefix, until the idle TTL clears
//! it. A request whose candidate set holds a single worker does not rebind the group at all, since
//! the host rather than this policy narrowed that choice.
//!
//! Bindings live in the policy instance, so they are per frontend process and do not survive a
//! restart or coordinate across router replicas.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::{WorkerAffinityTarget, WorkerWithDpRank};
use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, ScoredWorkerCandidate, SessionContext, WorkerInputView, WorkerInputs,
    WorkerLoadInput, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-subagent-group-affinity";

/// `active_requests` counts everything a worker is serving, not just this group, so the threshold
/// has to sit above ordinary per-worker concurrency or a group moves on nearly every sibling. This
/// default is chosen to clear typical steady-state concurrency rather than derived from a model or
/// topology; a deployment that runs hotter should raise it.
const DEFAULT_MAX_ACTIVE_REQUESTS: usize = 32;
const DEFAULT_GROUP_IDLE_TTL_SECS: u64 = 300;

/// Mirrors Dynamo's own session-affinity bounds, so a client cannot grow this map without limit by
/// varying the parent session header, and a TTL cannot be set so long that the map never reclaims.
const MAX_GROUPS: usize = 65_536;
const MAX_GROUP_ID_BYTES: usize = 256;
const MAX_GROUP_IDLE_TTL_SECS: u64 = 31_536_000;

/// Tunables for [`POLICY_TYPE`].
///
/// Every field is optional. Unknown keys are rejected at startup rather than ignored, so a
/// misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Active requests the group's worker may already hold before the group becomes eligible to
    /// move. Counts every request that worker is serving, not just this group's, so it must sit
    /// above ordinary per-worker concurrency for grouping to hold. Compared inclusively. `0` moves
    /// the group as soon as its worker has any in-flight request and a strictly less loaded worker
    /// exists, which effectively disables grouping under concurrency.
    max_active_requests: usize,
    /// Seconds a group binding survives without being used.
    group_idle_ttl_secs: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            max_active_requests: DEFAULT_MAX_ACTIVE_REQUESTS,
            group_idle_ttl_secs: DEFAULT_GROUP_IDLE_TTL_SECS,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !(1..=MAX_GROUP_IDLE_TTL_SECS).contains(&self.group_idle_ttl_secs) {
            return Err(WorkerSelectionPolicyProviderError::new(format!(
                "group_idle_ttl_secs must be between 1 and {MAX_GROUP_IDLE_TTL_SECS}"
            )));
        }
        Ok(())
    }
}

struct GroupBinding {
    worker: WorkerWithDpRank,
    last_used: Instant,
}

struct SubagentGroupAffinityPicker {
    parameters: Parameters,
    group_idle_ttl: Duration,
    groups: HashMap<String, GroupBinding>,
    last_sweep: Instant,
}

impl SubagentGroupAffinityPicker {
    fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            group_idle_ttl: Duration::from_secs(parameters.group_idle_ttl_secs),
            groups: HashMap::new(),
            last_sweep: Instant::now(),
        }
    }

    fn matches_target(candidate: &ScoredWorkerCandidate, target: WorkerAffinityTarget) -> bool {
        let worker = candidate.worker();
        worker.worker_id == target.worker_id
            && target.dp_rank.is_none_or(|rank| worker.dp_rank == rank)
    }

    fn least_loaded_row(
        candidates: &[ScoredWorkerCandidate],
        loads: &[WorkerLoadInput],
        excluded_target: Option<WorkerAffinityTarget>,
    ) -> Option<usize> {
        candidates
            .iter()
            .zip(loads)
            .enumerate()
            .filter(|(_, (candidate, _))| {
                excluded_target.is_none_or(|target| !Self::matches_target(candidate, target))
            })
            .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
            .map(|(row, _)| row)
    }

    fn target_row(
        candidates: &[ScoredWorkerCandidate],
        loads: &[WorkerLoadInput],
        target: WorkerAffinityTarget,
    ) -> Option<usize> {
        candidates
            .iter()
            .zip(loads)
            .enumerate()
            .filter(|(_, (candidate, _))| Self::matches_target(candidate, target))
            .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
            .map(|(row, _)| row)
    }

    fn select_row(
        &self,
        candidates: &[ScoredWorkerCandidate],
        loads: &[WorkerLoadInput],
        preferred: Option<WorkerAffinityTarget>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if let Some(target) = preferred
            && let Some(target_row) = Self::target_row(candidates, loads, target)
        {
            let target_load = loads[target_row].active_requests();
            if target_load <= self.parameters.max_active_requests {
                return Ok(target_row);
            }
            return Ok(Self::least_loaded_row(candidates, loads, Some(target))
                .filter(|&row| loads[row].active_requests() < target_load)
                .unwrap_or(target_row));
        }

        Self::least_loaded_row(candidates, loads, None)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }

    fn sweep_expired(&mut self, now: Instant) {
        if now.duration_since(self.last_sweep) < self.group_idle_ttl {
            return;
        }
        let group_idle_ttl = self.group_idle_ttl;
        self.groups
            .retain(|_, binding| now.duration_since(binding.last_used) < group_idle_ttl);
        self.last_sweep = now;
    }

    fn touch_group(&mut self, group_id: &str, now: Instant) {
        if let Some(binding) = self.groups.get_mut(group_id) {
            binding.last_used = now;
        }
    }

    fn bind_group(&mut self, group_id: &str, worker: WorkerWithDpRank, now: Instant) {
        if let Some(binding) = self.groups.get_mut(group_id) {
            binding.worker = worker;
            binding.last_used = now;
            return;
        }
        if group_id.len() > MAX_GROUP_ID_BYTES || self.groups.len() >= MAX_GROUPS {
            return;
        }
        self.groups.insert(
            group_id.to_owned(),
            GroupBinding {
                worker,
                last_used: now,
            },
        );
    }
}

impl WorkerPicker for SubagentGroupAffinityPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let loads = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;

        let group_id = context
            .session_context()
            .and_then(SessionContext::parent_session_id);

        let Some(group_id) = group_id else {
            return self.select_row(candidates, loads, context.affinity_target());
        };

        let now = Instant::now();
        self.sweep_expired(now);

        // Only the group binding steers a subagent. Falling back to this subagent's own session
        // target is what scatters siblings, so an unbound group is placed purely by load.
        let preferred = self
            .groups
            .get(group_id)
            .map(|binding| WorkerAffinityTarget::from(binding.worker));

        let row = self.select_row(candidates, loads, preferred)?;

        // Keep an actively used group alive even on requests that cannot rebind it, otherwise a
        // busy group expires mid-use and its siblings scatter.
        self.touch_group(group_id, now);

        // One candidate means the host narrowed the choice rather than the policy making it, most
        // often a pinned session target. Recording it would move every sibling onto a worker this
        // policy never selected. This is a heuristic: it cannot see why the set was narrowed.
        if candidates.len() > 1
            && let Some(candidate) = candidates.get(row)
        {
            self.bind_group(group_id, candidate.worker(), now);
        }
        Ok(row)
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(SubagentGroupAffinityPicker::new(parameters)),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1024)
        }
    }

    fn request(
        session_id: &str,
        parent_session_id: Option<&str>,
        affinity_target: Option<WorkerAffinityTarget>,
    ) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: Some(SessionContext::new(
                session_id.to_owned(),
                parent_session_id.map(str::to_owned),
                None,
                None,
            )),
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        }
    }

    fn set_active_requests(
        request: &mut SchedulingRequest,
        worker: WorkerWithDpRank,
        active_requests: usize,
    ) {
        request.worker_loads.insert(
            worker,
            WorkerLoadProjection {
                active_requests,
                ..Default::default()
            },
        );
    }

    fn policy(max_active_requests: usize) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(SubagentGroupAffinityPicker::new(Parameters {
                max_active_requests,
                group_idle_ttl_secs: DEFAULT_GROUP_IDLE_TTL_SECS,
            })),
        )
    }

    fn select(
        policy: &WorkerSelectionPolicy,
        workers: &HashMap<u64, TestWorker>,
        request: &SchedulingRequest,
    ) -> WorkerWithDpRank {
        policy
            .select_worker(WorkerSelectionInput::configured(
                workers,
                request,
                request.eligibility(),
                16,
            ))
            .unwrap()
            .worker
    }

    // Worker 29 wins every load tie because selection breaks ties on the lower worker id, so a
    // selection of worker 41 under a tie proves the group binding was used.
    fn workers() -> HashMap<u64, TestWorker> {
        HashMap::from([(29, TestWorker), (41, TestWorker)])
    }

    #[test]
    fn places_a_new_group_on_the_least_loaded_worker() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);

        assert_eq!(select(&policy(0), &workers(), &first), worker_b);
    }

    #[test]
    fn co_locates_later_subagents_of_the_same_parent() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);
        let workers = workers();

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers, &first), worker_b);

        // A sibling with its own session id and no affinity target of its own still joins the
        // group, even though the tie-break would otherwise send it to worker 29.
        let mut sibling = request("child-2", Some("parent-1"), None);
        set_active_requests(&mut sibling, worker_a, 0);
        set_active_requests(&mut sibling, worker_b, 0);
        assert_eq!(select(&policy, &workers, &sibling), worker_b);
    }

    #[test]
    fn group_binding_outranks_a_subagents_own_affinity_target() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);
        let workers = workers();

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers, &first), worker_b);

        let mut sibling = request("child-2", Some("parent-1"), Some(worker_a.into()));
        set_active_requests(&mut sibling, worker_a, 0);
        set_active_requests(&mut sibling, worker_b, 0);
        assert_eq!(select(&policy, &workers, &sibling), worker_b);
    }

    #[test]
    fn an_unbound_group_ignores_the_subagents_own_affinity_target() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);

        // This subagent's own session is bound to the busy worker 41. Honoring that would place
        // the whole new group on a loaded worker instead of by load.
        let mut first = request("child-1", Some("parent-1"), Some(worker_b.into()));
        set_active_requests(&mut first, worker_a, 0);
        set_active_requests(&mut first, worker_b, 3);

        assert_eq!(select(&policy(0), &workers(), &first), worker_a);
    }

    #[test]
    fn keeps_a_group_together_below_a_nonzero_threshold() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(4);
        let workers = workers();

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers, &first), worker_b);

        // Four in-flight siblings are at the inclusive threshold, so the group holds worker 41
        // even though worker 29 is idle.
        let mut sibling = request("child-2", Some("parent-1"), None);
        set_active_requests(&mut sibling, worker_a, 0);
        set_active_requests(&mut sibling, worker_b, 4);
        assert_eq!(select(&policy, &workers, &sibling), worker_b);

        let mut over = request("child-3", Some("parent-1"), None);
        set_active_requests(&mut over, worker_a, 0);
        set_active_requests(&mut over, worker_b, 5);
        assert_eq!(select(&policy, &workers, &over), worker_a);
    }

    #[test]
    fn does_not_move_a_group_to_an_equally_loaded_worker() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);
        let workers = workers();

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers, &first), worker_b);

        // Worker 41 is over the threshold, but worker 29 is no better. Moving here would make the
        // group oscillate between the two workers on every sibling.
        let mut sibling = request("child-2", Some("parent-1"), None);
        set_active_requests(&mut sibling, worker_a, 1);
        set_active_requests(&mut sibling, worker_b, 1);
        assert_eq!(select(&policy, &workers, &sibling), worker_b);
    }

    #[test]
    fn moves_a_group_off_a_worker_that_exceeds_the_threshold() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);
        let workers = workers();

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers, &first), worker_b);

        let mut overloaded = request("child-2", Some("parent-1"), None);
        set_active_requests(&mut overloaded, worker_a, 0);
        set_active_requests(&mut overloaded, worker_b, 1);
        assert_eq!(select(&policy, &workers, &overloaded), worker_a);

        // The group followed the move, so a later sibling joins worker 29 rather than the worker
        // the group started on.
        let mut later = request("child-3", Some("parent-1"), None);
        set_active_requests(&mut later, worker_a, 0);
        set_active_requests(&mut later, worker_b, 0);
        assert_eq!(select(&policy, &workers, &later), worker_a);
    }

    #[test]
    fn keeps_separate_parents_in_separate_groups() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);
        let workers = workers();

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers, &first), worker_b);

        let mut other_parent = request("child-2", Some("parent-2"), None);
        set_active_requests(&mut other_parent, worker_a, 0);
        set_active_requests(&mut other_parent, worker_b, 0);
        assert_eq!(select(&policy, &workers, &other_parent), worker_a);
    }

    #[test]
    fn requests_without_a_parent_keep_session_affinity() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);
        let workers = workers();

        let mut unbound = request("parent-1", None, None);
        set_active_requests(&mut unbound, worker_a, 2);
        set_active_requests(&mut unbound, worker_b, 0);
        assert_eq!(select(&policy, &workers, &unbound), worker_b);

        let mut bound = request("parent-1", None, Some(worker_a.into()));
        set_active_requests(&mut bound, worker_a, 0);
        set_active_requests(&mut bound, worker_b, 0);
        assert_eq!(select(&policy, &workers, &bound), worker_a);
    }

    #[test]
    fn rebinds_when_the_group_worker_is_not_eligible() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers(), &first), worker_b);

        // Worker 41 left the pool, so the group falls back to the least-loaded eligible worker.
        let remaining = HashMap::from([(29, TestWorker), (57, TestWorker)]);
        let worker_c = WorkerWithDpRank::from_worker_id(57);
        let mut sibling = request("child-2", Some("parent-1"), None);
        set_active_requests(&mut sibling, worker_a, 3);
        set_active_requests(&mut sibling, worker_c, 0);
        assert_eq!(select(&policy, &remaining, &sibling), worker_c);
    }

    #[test]
    fn a_constrained_candidate_set_does_not_rebind_the_group() {
        let worker_a = WorkerWithDpRank::from_worker_id(29);
        let worker_b = WorkerWithDpRank::from_worker_id(41);
        let policy = policy(0);

        let mut first = request("child-1", Some("parent-1"), None);
        set_active_requests(&mut first, worker_a, 2);
        set_active_requests(&mut first, worker_b, 0);
        assert_eq!(select(&policy, &workers(), &first), worker_b);

        // A sibling the host pinned elsewhere sees one candidate. It must not drag the group onto
        // a worker this policy never chose.
        let only_worker_a = HashMap::from([(29, TestWorker)]);
        let mut pinned = request("child-2", Some("parent-1"), None);
        set_active_requests(&mut pinned, worker_a, 0);
        assert_eq!(select(&policy, &only_worker_a, &pinned), worker_a);

        let mut sibling = request("child-3", Some("parent-1"), None);
        set_active_requests(&mut sibling, worker_a, 0);
        set_active_requests(&mut sibling, worker_b, 0);
        assert_eq!(select(&policy, &workers(), &sibling), worker_b);
    }

    #[test]
    fn a_used_group_survives_the_idle_sweep() {
        let worker = WorkerWithDpRank::from_worker_id(29);
        let ttl = Duration::from_secs(60);
        let mut picker = SubagentGroupAffinityPicker::new(Parameters {
            max_active_requests: 0,
            group_idle_ttl_secs: ttl.as_secs(),
        });
        let start = Instant::now();
        picker.bind_group("busy", worker, start);
        picker.bind_group("idle", worker, start);

        // "busy" keeps arriving on requests that cannot rebind it, so it must stay alive.
        let later = start + Duration::from_secs(90);
        picker.touch_group("busy", later);
        picker.sweep_expired(later);

        assert!(picker.groups.contains_key("busy"));
        assert!(!picker.groups.contains_key("idle"));
    }

    #[test]
    fn group_bindings_are_capped() {
        let worker = WorkerWithDpRank::from_worker_id(29);
        let mut picker = SubagentGroupAffinityPicker::new(Parameters::default());
        let oversized = "x".repeat(MAX_GROUP_ID_BYTES + 1);

        picker.bind_group(&oversized, worker, Instant::now());
        assert!(picker.groups.is_empty());

        for index in 0..MAX_GROUPS {
            picker.bind_group(&format!("parent-{index}"), worker, Instant::now());
        }
        assert_eq!(picker.groups.len(), MAX_GROUPS);

        picker.bind_group("one-too-many", worker, Instant::now());
        assert_eq!(picker.groups.len(), MAX_GROUPS);
    }

    #[test]
    fn validates_group_idle_ttl_secs() {
        assert!(Parameters::default().validate().is_ok());
        for group_idle_ttl_secs in [0, MAX_GROUP_IDLE_TTL_SECS + 1] {
            assert!(
                Parameters {
                    group_idle_ttl_secs,
                    ..Parameters::default()
                }
                .validate()
                .is_err()
            );
        }
    }
}
