// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! External-host contract tests: only public Dynamo APIs, with no runtime,
//! transport, engine, or simulator dependency. Selection does not own admission.

#![cfg(feature = "standalone-selection")]

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};
use dynamo_kv_router::protocols::{WorkerSelectionResult, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
use dynamo_kv_router::services::selection::affinity::{
    AcquireStep, AffinityTarget, Hold, SessionAffinity, SessionAffinityConfig, SessionAffinityMode,
    subagent_group_affinity_id,
};
use dynamo_kv_router::{
    KvRouterConfig, KvSchedulerError, SchedulingRequest, WorkerConfigLike, WorkerSelectionInput,
    WorkerSelector, WorkerType,
};
use futures_util::FutureExt;
use tokio::time::Instant;

const BLOCK_SIZE: u32 = 16;
const TTL: Duration = Duration::from_secs(10);

struct HostWorker {
    first_rank: u32,
}

impl WorkerConfigLike for HostWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        self.first_rank
    }

    fn data_parallel_size(&self) -> u32 {
        2
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(1024)
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        Some(128)
    }
}

fn workers() -> HashMap<u64, HostWorker> {
    HashMap::from([
        (7, HostWorker { first_rank: 4 }),
        (8, HostWorker { first_rank: 0 }),
    ])
}

fn request(cached: WorkerWithDpRank) -> SchedulingRequest {
    let mut overlap = OverlapSignals::default();
    overlap.tier_overlap_blocks.device.insert(cached, 3);
    overlap.effective_overlap_blocks.insert(cached, 3.0);
    overlap.effective_cached_tokens.insert(cached, 48);
    SchedulingRequest {
        mode: ScheduleMode::QueryOnly {
            request_id: Some("host-request".into()),
        },
        token_seq: None,
        isl_tokens: 64,
        lora_name: None,
        expected_output_tokens: Some(8),
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: Default::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap,
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads: Default::default(),
        resp_tx: None,
    }
}

fn select(
    role: WorkerType,
    workers: &HashMap<u64, HostWorker>,
    request: &SchedulingRequest,
) -> WorkerSelectionResult {
    select_with_policy(
        &WorkerSelectionPolicy::default(KvRouterConfig::default(), role.default_selector_label()),
        workers,
        request,
    )
}

fn select_with_policy(
    policy: &impl WorkerSelector<HostWorker>,
    workers: &HashMap<u64, HostWorker>,
    request: &SchedulingRequest,
) -> WorkerSelectionResult {
    let mut eligibility = request.eligibility();
    if policy.uses_exclusive_affinity_target()
        && let Some(target) = request.affinity_target
    {
        eligibility = eligibility.with_eligible_affinity_target(workers, target);
    }
    policy
        .select_worker(WorkerSelectionInput::configured(
            workers,
            request,
            eligibility,
            BLOCK_SIZE,
        ))
        .unwrap()
}

fn hold(table: &SessionAffinity, key: &str) -> Hold {
    match table.try_acquire(key, None).unwrap() {
        AcquireStep::Held(hold) => hold,
        AcquireStep::Wait(_) => panic!("expected an immediately available affinity hold"),
    }
}

fn table(now: Instant) -> SessionAffinity {
    SessionAffinity::with_manual_clock(SessionAffinityConfig::new(TTL), now).unwrap()
}

#[test]
fn cache_selection_and_session_affinity_preserve_the_global_dp_rank() {
    let table = table(Instant::now());
    let workers = workers();
    let cached = WorkerWithDpRank::new(7, 5);
    let first = hold(&table, "conversation");
    assert!(first.target().is_none());
    let selected = select(WorkerType::Aggregated, &workers, &request(cached));
    assert_eq!(selected.worker, cached);
    assert_eq!(selected.cached_tokens, 48);
    assert_eq!(selected.effective_overlap_blocks, 3.0);
    // Computing a route has not admitted work or installed a binding.
    assert_eq!(table.query_target("conversation", None).unwrap(), None);
    let admitted = table.commit(first, selected.worker.into()).unwrap();

    let next = hold(&table, "conversation");
    let mut moved_cache = request(WorkerWithDpRank::new(8, 1));
    moved_cache.affinity_target = next.target();
    assert_eq!(
        select(WorkerType::Aggregated, &workers, &moved_cache).worker,
        cached
    );
    let second = table.commit(next, cached.into()).unwrap();
    drop((admitted, second));
}

#[test]
fn worker_only_affinity_leaves_dp_rank_selection_to_current_cache_inputs() {
    let table = table(Instant::now());
    let workers = workers();
    let lease = table
        .commit(hold(&table, "session"), AffinityTarget::new(7, None))
        .unwrap();
    for rank in [4, 5] {
        let held = hold(&table, "session");
        let mut request = request(WorkerWithDpRank::new(7, rank));
        request.affinity_target = held.target();
        assert_eq!(
            select(WorkerType::Aggregated, &workers, &request).worker,
            WorkerWithDpRank::new(7, rank)
        );
    }
    drop(lease);
}

#[test]
fn ineligible_affinity_falls_back_without_relaxing_host_constraints() {
    let workers = workers();
    let fallback = WorkerWithDpRank::new(8, 1);
    let available = HashSet::from([8]);
    let overloaded = HashSet::from([7]);
    for cause in [
        "missing worker",
        "rank",
        "availability",
        "allowlist",
        "overload",
    ] {
        let target = match cause {
            "missing worker" => AffinityTarget::new(99, None),
            "rank" => AffinityTarget::new(7, Some(99)),
            _ => AffinityTarget::new(7, Some(5)),
        };
        let mut request = request(fallback);
        request.affinity_target = Some(target);
        if cause == "allowlist" {
            request.allowed_worker_ids = Some(HashSet::from([8]));
        }
        let eligibility = request
            .eligibility_with_overloaded((cause == "overload").then_some(&overloaded))
            .with_available_workers((cause == "availability").then_some(&available))
            .with_eligible_affinity_target(&workers, target);
        let selected = WorkerSelectionPolicy::default(KvRouterConfig::default(), "decode")
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                eligibility,
                BLOCK_SIZE,
            ))
            .unwrap();
        assert_eq!(selected.worker, fallback, "ineligible because of {cause}");
    }
}

#[test]
fn affinity_fallback_does_not_override_an_invalid_explicit_pin() {
    let workers = workers();
    let mut request = request(WorkerWithDpRank::new(8, 1));
    request.allowed_worker_ids = Some(HashSet::from([8]));
    request.pinned_worker = Some(WorkerWithDpRank::new(7, 5));
    let eligibility = request
        .eligibility()
        .with_eligible_affinity_target(&workers, AffinityTarget::new(7, Some(5)));
    let result = WorkerSelectionPolicy::default(KvRouterConfig::default(), "decode").select_worker(
        WorkerSelectionInput::configured(&workers, &request, eligibility, BLOCK_SIZE),
    );
    assert!(matches!(
        result,
        Err(KvSchedulerError::PinnedWorkerNotAllowed { worker_id: 7 })
    ));
}

#[test]
fn sibling_groups_share_a_binding_without_inheriting_the_parent_binding() {
    let table = table(Instant::now());
    let workers = workers();
    let parent_worker = WorkerWithDpRank::new(7, 4);
    let child_worker = WorkerWithDpRank::new(8, 1);
    let parent = table
        .commit(hold(&table, "parent"), parent_worker.into())
        .unwrap();
    let group = subagent_group_affinity_id("parent");
    assert_ne!(group, "parent");
    assert_ne!(group, subagent_group_affinity_id("other-parent"));
    assert_ne!(group, subagent_group_affinity_id("child-a"));
    assert_eq!(group, subagent_group_affinity_id("parent"));
    assert_eq!(
        subagent_group_affinity_id(&"p".repeat(10_000)).len(),
        group.len()
    );

    let child_a = hold(&table, &group);
    let selected = select(WorkerType::Aggregated, &workers, &request(child_worker));
    assert_eq!(selected.worker, child_worker);
    let first_child = table.commit(child_a, selected.worker.into()).unwrap();
    // A different conversation with this immediate parent uses the same key.
    let child_b = hold(&table, &subagent_group_affinity_id("parent"));
    let mut sibling_request = request(parent_worker);
    sibling_request.affinity_target = child_b.target();
    assert_eq!(
        select(WorkerType::Aggregated, &workers, &sibling_request).worker,
        child_worker
    );
    assert_eq!(
        table.query_target("parent", None).unwrap(),
        Some(parent_worker.into())
    );
    drop((child_b, first_child, parent));
}

#[test]
fn abort_and_commit_wake_waiters_and_cancellation_starts_virtual_idle_ttl() {
    let epoch = Instant::now();
    let table = table(epoch);
    let tentative = hold(&table, "session");
    let AcquireStep::Wait(mut aborted_waiter) = table.try_acquire("session", None).unwrap() else {
        panic!("a second request must wait for initialization");
    };
    assert!(aborted_waiter.as_mut().now_or_never().is_none());
    drop(tentative); // Admission failed; dropping the hold is an abort.
    assert!(aborted_waiter.as_mut().now_or_never().is_some());
    assert_eq!(table.query_target("session", None).unwrap(), None);

    let tentative = hold(&table, "session");
    let AcquireStep::Wait(mut committed_waiter) = table.try_acquire("session", None).unwrap()
    else {
        panic!("a second request must still wait before admission");
    };
    let target = AffinityTarget::new(7, Some(5));
    let active = table.commit(tentative, target).unwrap();
    assert!(committed_waiter.as_mut().now_or_never().is_some());
    let overlapping = table.commit(hold(&table, "session"), target).unwrap();
    table.advance_clock(epoch + TTL * 2).unwrap();
    drop(active);
    table.advance_clock(epoch + TTL * 4).unwrap();
    assert_eq!(table.query_target("session", None).unwrap(), Some(target));

    // Cancellation and normal completion both release the host's active lease.
    drop(overlapping);
    table
        .advance_clock(epoch + TTL * 5 - Duration::from_nanos(1))
        .unwrap();
    assert_eq!(table.query_target("session", None).unwrap(), Some(target));
    table.advance_clock(epoch + TTL * 5).unwrap();
    assert_eq!(table.query_target("session", None).unwrap(), None);
    assert!(table.advance_clock(epoch).is_err());
    assert!(matches!(hold(&table, "session"), Hold::Initialize(_)));
}

#[test]
fn removed_worker_invalidation_allows_reselection_and_fences_old_leases() {
    let table = table(Instant::now());
    let mut workers = workers();
    let old_target = WorkerWithDpRank::new(7, 5);
    let old = table
        .commit(hold(&table, "session"), old_target.into())
        .unwrap();
    let invalid = hold(&table, "session");
    workers.remove(&7);
    invalid.invalidate();
    assert_eq!(table.query_target("session", None).unwrap(), None);

    let replacement = hold(&table, "session");
    let new_target = WorkerWithDpRank::new(8, 1);
    let selected = select(WorkerType::Aggregated, &workers, &request(new_target));
    assert_eq!(selected.worker, new_target);
    let active = table.commit(replacement, selected.worker.into()).unwrap();
    drop(old);
    assert_eq!(
        table.query_target("session", None).unwrap(),
        Some(new_target.into())
    );
    drop(active);
}

#[test]
fn prefill_and_decode_pools_own_independent_bindings_even_with_shared_worker_ids() {
    let epoch = Instant::now();
    let prefill = table(epoch);
    let decode = table(epoch);
    let workers = workers();
    let key = subagent_group_affinity_id("parent");
    let prefill_worker = WorkerWithDpRank::new(7, 4);
    let decode_worker = WorkerWithDpRank::new(7, 5);
    let p = select(WorkerType::Prefill, &workers, &request(prefill_worker));
    let d = select(WorkerType::Decode, &workers, &request(decode_worker));
    assert_eq!(p.worker, prefill_worker);
    assert_eq!(d.worker, decode_worker);
    let p_lease = prefill
        .commit(hold(&prefill, &key), p.worker.into())
        .unwrap();
    let d_lease = decode.commit(hold(&decode, &key), d.worker.into()).unwrap();
    drop(p_lease);
    prefill.advance_clock(epoch + TTL).unwrap();
    decode.advance_clock(epoch + TTL).unwrap();
    assert_eq!(prefill.query_target(&key, None).unwrap(), None);
    assert_eq!(
        decode.query_target(&key, None).unwrap(),
        Some(decode_worker.into())
    );
    drop(d_lease);
}

struct CacheScorer;

impl WorkerScorer for CacheScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        assert_eq!(
            context.affinity_target(),
            Some(AffinityTarget::new(7, None))
        );
        Ok(-candidate.cache().unwrap().device_overlap_blocks())
    }
}

struct LowestCost;

impl WorkerPicker for LowestCost {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        Ok(input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.cost()
                    .total_cmp(&b.cost())
                    .then_with(|| a.worker().cmp(&b.worker()))
            })
            .expect("host supplies eligible candidates")
            .0)
    }
}

#[test]
fn selection_policy_and_soft_affinity_are_independent_public_contracts() {
    let table = SessionAffinity::with_manual_clock(
        SessionAffinityConfig::new(TTL).with_mode(SessionAffinityMode::Soft),
        Instant::now(),
    )
    .unwrap();
    let old = table
        .commit(hold(&table, "session"), AffinityTarget::new(7, None))
        .unwrap();
    let next = hold(&table, "session");
    let workers = workers();
    let better_cache = WorkerWithDpRank::new(8, 1);
    let mut request = request(better_cache);
    request.affinity_target = next.target();
    let policy = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        WorkerType::Aggregated.as_str(),
        vec![Box::new(CacheScorer)],
        Box::new(LowestCost),
    );
    let selected = select_with_policy(&policy, &workers, &request);
    assert_eq!(selected.worker, better_cache);
    assert_eq!(
        table.query_target("session", None).unwrap(),
        Some(AffinityTarget::new(7, None))
    );
    let rebound = table.commit(next, selected.worker.into()).unwrap();
    assert_eq!(
        table.query_target("session", None).unwrap(),
        Some(AffinityTarget::new(8, None))
    );
    drop((old, rebound));
}
