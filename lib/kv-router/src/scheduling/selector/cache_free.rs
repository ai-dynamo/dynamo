// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache-free worker-selection host contracts and generic policy adapter.

use std::collections::HashSet;

use parking_lot::Mutex;

use super::{
    CustomWorkerSelectionState, LogitWeights, ScoredWorkerCandidate, WorkerCacheInput,
    WorkerCandidate, WorkerInputView, WorkerLoadInput, WorkerRoutingInput, WorkerSelectionContext,
    WorkerSelectionRequirements,
};
use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::scheduling::types::{SessionContext, WorkerSelectionPolicyError};

/// Request metadata made available to cache-free policy selection.
///
/// A cache-free host has no KV block size, cache index, or scheduler load projections. It
/// therefore reports prompt tokens as one-token blocks and exposes only the request metadata
/// already available at the frontend routing boundary.
pub struct CacheFreeRequestContext<'a> {
    request_id: &'a str,
    request_tokens: usize,
    session_context: Option<&'a SessionContext>,
    expected_output_tokens: Option<u32>,
    priority_jump: f64,
    strict_priority: u32,
    advisory: bool,
}

impl<'a> CacheFreeRequestContext<'a> {
    pub fn new(request_id: &'a str, request_tokens: usize, advisory: bool) -> Self {
        Self {
            request_id,
            request_tokens,
            session_context: None,
            expected_output_tokens: None,
            priority_jump: 0.0,
            strict_priority: 0,
            advisory,
        }
    }

    pub fn with_session_context(mut self, session_context: Option<&'a SessionContext>) -> Self {
        self.session_context = session_context;
        self
    }

    pub fn with_expected_output_tokens(mut self, expected_output_tokens: Option<u32>) -> Self {
        self.expected_output_tokens = expected_output_tokens;
        self
    }

    pub fn with_priority(mut self, priority_jump: f64, strict_priority: u32) -> Self {
        self.priority_jump = priority_jump;
        self.strict_priority = strict_priority;
        self
    }

    /// Whether this is a preview that must not commit policy-local state.
    pub fn is_advisory(&self) -> bool {
        self.advisory
    }
}

/// Borrowed, host-owned candidate table for cache-free selection.
///
/// The host has already applied discovery, namespace, health, and admission eligibility before
/// exposing rows here. A cache-free policy may read worker identity and the frontend-local active
/// request count only; it cannot request KV cache, taint, or scheduler projection inputs.
///
/// The table is a stable snapshot: its row count and worker identity must remain unchanged from
/// policy selection through the host's admission of the returned row. Hosts that expose
/// [`Self::least_loaded_index`] must maintain that minimum as an O(1) lookup; policies must not
/// rescan the table to reconstruct it. All table methods used by cache-free pickers must also be
/// O(1), including [`Self::len`] and [`Self::active_requests`].
pub trait CacheFreeCandidateTable {
    /// Return the number of eligible rows in O(1).
    fn len(&self) -> usize;

    /// Return one row in O(1).
    fn worker(&self, index: usize) -> WorkerWithDpRank;

    /// Return the frontend-local active-request count for one row in O(1).
    fn active_requests(&self, index: usize) -> usize;

    /// Return the host-maintained least-loaded row, if this host supports that policy.
    ///
    /// Implementations must return in O(1). A host that does not maintain a load index returns
    /// `None`, making a least-loaded policy unavailable rather than falling back to an O(N) scan.
    fn least_loaded_index(&self) -> Option<usize> {
        None
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One cache-free policy decision and, when filters narrowed the candidate table, the set that
/// transport fallback must remain within.
#[derive(Debug)]
pub struct CacheFreePolicyDecision {
    pub index: usize,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
}

impl CacheFreePolicyDecision {
    /// Select one row without narrowing the host-provided eligible set.
    pub const fn unfiltered(index: usize) -> Self {
        Self {
            index,
            allowed_worker_ids: None,
        }
    }
}

/// An O(1) cache-free policy implementation supplied by a linked policy crate.
///
/// This bypasses generic filter/score/pick materialization. It is intended for policies whose
/// selection algorithm samples a constant number of rows, such as round-robin, random, and P2C.
/// A host-provided least-loaded index also keeps least-loaded selection O(1).
pub trait CacheFreeWorkerPicker: Send + Sync {
    /// Return the infrastructure signals needed by this picker.
    fn requirements(&self) -> WorkerSelectionRequirements;

    /// Select one row from a stable host-owned candidate table.
    fn select(
        &self,
        request: &CacheFreeRequestContext<'_>,
        candidate_table: &dyn CacheFreeCandidateTable,
    ) -> Result<CacheFreePolicyDecision, WorkerSelectionPolicyError>;
}

/// Thread-safe host adapter for custom filters, scorers, and pickers that do not require KV
/// routing state.
///
/// The mutex serializes policy-local state such as a round-robin cursor. The routing host must
/// hold its own discovery/admission lock around a committed call so observing active request
/// counts and reserving the selected worker stay atomic.
pub struct CacheFreeWorkerSelectionPolicy {
    kind: CacheFreeWorkerSelectionPolicyKind,
}

enum CacheFreeWorkerSelectionPolicyKind {
    Generic(Box<Mutex<CacheFreeWorkerSelectionState>>),
    Direct(Box<dyn CacheFreeWorkerPicker>),
}

struct CacheFreeWorkerSelectionState {
    policy: CustomWorkerSelectionState,
    source_indices: Vec<usize>,
}

impl CacheFreeWorkerSelectionPolicy {
    pub(super) fn direct(picker: Box<dyn CacheFreeWorkerPicker>) -> Self {
        Self {
            kind: CacheFreeWorkerSelectionPolicyKind::Direct(picker),
        }
    }

    pub(super) fn generic(policy: CustomWorkerSelectionState) -> Self {
        Self {
            kind: CacheFreeWorkerSelectionPolicyKind::Generic(Box::new(Mutex::new(
                CacheFreeWorkerSelectionState {
                    policy,
                    source_indices: Vec::new(),
                },
            ))),
        }
    }

    /// Run the custom policy against a borrowed table of host-eligible workers.
    pub fn select(
        &self,
        request: &CacheFreeRequestContext<'_>,
        candidate_table: &dyn CacheFreeCandidateTable,
    ) -> Result<CacheFreePolicyDecision, WorkerSelectionPolicyError> {
        let state = match &self.kind {
            CacheFreeWorkerSelectionPolicyKind::Direct(picker) => {
                return picker.select(request, candidate_table);
            }
            CacheFreeWorkerSelectionPolicyKind::Generic(state) => state,
        };
        let mut state = state.lock();
        let CacheFreeWorkerSelectionState {
            policy,
            source_indices,
        } = &mut *state;
        let CustomWorkerSelectionState {
            filters,
            scorers,
            filter_inputs,
            scorer_picker_inputs,
            picker_inputs,
            picker,
            unscored_candidates,
            candidates,
            load_inputs,
            ..
        } = policy;

        unscored_candidates.clear();
        candidates.clear();
        source_indices.clear();
        load_inputs.clear();

        let context = WorkerSelectionContext {
            request_id: request.request_id,
            request_blocks: request.request_tokens as u64,
            block_size: 1,
            track_prefill_tokens: false,
            weights: LogitWeights {
                overlap_score_credit: 0.0,
                overlap_score_credit_decay: 0.0,
                prefill_load_scale: 0.0,
                shared_cache_multiplier: 0.0,
            },
            min_active_prefill_tokens: 0,
            router_temperature_override: None,
            session_context: request.session_context,
            expected_output_tokens: request.expected_output_tokens,
            priority_jump: request.priority_jump,
            strict_priority: request.strict_priority,
            advisory: request.advisory,
        };

        let mut filters_narrowed = false;
        for index in 0..candidate_table.len() {
            let candidate_for = |inputs| WorkerCandidate {
                worker: candidate_table.worker(index),
                inputs,
                cache: WorkerCacheInput::default(),
                load: if inputs.needs_worker_load() {
                    WorkerLoadInput {
                        active_requests: candidate_table.active_requests(index),
                        ..Default::default()
                    }
                } else {
                    WorkerLoadInput::default()
                },
                routing: WorkerRoutingInput::default(),
            };
            let filter_candidate = candidate_for(*filter_inputs);
            let mut keep = true;
            for filter in filters.iter_mut() {
                if !filter.keep(&context, &filter_candidate)? {
                    keep = false;
                    break;
                }
            }
            if !keep {
                filters_narrowed = true;
                continue;
            }
            unscored_candidates.push(candidate_for(*scorer_picker_inputs));
            source_indices.push(index);
        }

        for candidate in unscored_candidates.iter() {
            let mut cost = 0.0;
            for (scorer_index, scorer) in scorers.iter_mut().enumerate() {
                let contribution = scorer.score(&context, candidate)?;
                cost += contribution;
                if !contribution.is_finite() || !cost.is_finite() {
                    return Err(WorkerSelectionPolicyError::NonFiniteCost {
                        scorer_index,
                        row: candidates.len(),
                    });
                }
            }
            candidates.push(ScoredWorkerCandidate {
                worker: candidate.worker,
                cost,
            });
            if picker_inputs.needs_worker_load() {
                load_inputs.push(candidate.load);
            }
        }

        if candidates.is_empty() {
            let message = if candidate_table.is_empty() {
                "no eligible workers"
            } else {
                "all eligible workers were rejected by policy filters"
            };
            return Err(WorkerSelectionPolicyError::failed(message));
        }
        let input = WorkerInputView {
            candidates,
            cache: None,
            load: picker_inputs
                .needs_worker_load()
                .then_some(load_inputs.as_slice()),
            routing: None,
        };
        let row = picker.pick(&context, input)?;
        let index = source_indices.get(row).copied().ok_or(
            WorkerSelectionPolicyError::InvalidPickerRow {
                row,
                candidate_count: candidates.len(),
            },
        )?;
        Ok(CacheFreePolicyDecision {
            index,
            allowed_worker_ids: filters_narrowed.then(|| {
                candidates
                    .iter()
                    .map(|candidate| candidate.worker.worker_id)
                    .collect()
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::scheduling::config::KvRouterConfig;
    use crate::scheduling::selector::{
        WorkerCandidate, WorkerFilter, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
        WorkerSelectionContext, WorkerSelectionPolicy,
    };

    struct CacheFreeRows(Vec<(WorkerWithDpRank, usize)>);

    impl CacheFreeCandidateTable for CacheFreeRows {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn worker(&self, index: usize) -> WorkerWithDpRank {
            self.0[index].0
        }

        fn active_requests(&self, index: usize) -> usize {
            self.0[index].1
        }
    }

    struct StaticCacheFreeRows(Vec<WorkerWithDpRank>);

    impl CacheFreeCandidateTable for StaticCacheFreeRows {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn worker(&self, index: usize) -> WorkerWithDpRank {
            self.0[index]
        }

        fn active_requests(&self, _index: usize) -> usize {
            panic!("static policy must not request active-load state")
        }
    }

    struct LowestActiveRequestScorer;

    impl WorkerScorer for LowestActiveRequestScorer {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::ACTIVE_REQUEST_LOAD
        }

        fn score(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            candidate: &WorkerCandidate,
        ) -> Result<f64, WorkerSelectionPolicyError> {
            Ok(candidate
                .load()
                .expect("active-request load was requested")
                .active_requests() as f64)
        }
    }

    struct LowestCostPicker;

    impl WorkerPicker for LowestCostPicker {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            input
                .candidates()
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
                .map(|(row, _)| row)
                .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible workers"))
        }
    }

    #[test]
    fn uses_active_request_load_without_kv_inputs() {
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            vec![Box::new(LowestActiveRequestScorer)],
            Box::new(LowestCostPicker),
        )
        .into_cache_free()
        .unwrap();
        let rows = CacheFreeRows(vec![
            (WorkerWithDpRank::from_worker_id(41), 9),
            (WorkerWithDpRank::from_worker_id(42), 2),
            (WorkerWithDpRank::from_worker_id(43), 6),
        ]);

        let selected = policy
            .select(&CacheFreeRequestContext::new("request", 32, false), &rows)
            .unwrap();
        assert_eq!(selected.index, 1);
        assert!(selected.allowed_worker_ids.is_none());
    }

    #[test]
    fn static_policy_never_reads_active_request_load() {
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            Vec::new(),
            Box::new(LowestCostPicker),
        )
        .into_cache_free()
        .unwrap();
        let rows = StaticCacheFreeRows(vec![
            WorkerWithDpRank::from_worker_id(41),
            WorkerWithDpRank::from_worker_id(42),
        ]);

        assert_eq!(
            policy
                .select(&CacheFreeRequestContext::new("request", 32, false), &rows)
                .unwrap()
                .index,
            0
        );
    }

    #[test]
    fn returns_filtered_fallback_set() {
        struct RejectWorker(WorkerId);

        impl WorkerFilter for RejectWorker {
            fn keep(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                candidate: &WorkerCandidate,
            ) -> Result<bool, WorkerSelectionPolicyError> {
                Ok(candidate.worker().worker_id != self.0)
            }
        }

        let policy = WorkerSelectionPolicy::new_with_filters(
            KvRouterConfig::default(),
            "decode",
            vec![Box::new(RejectWorker(41))],
            Vec::new(),
            Box::new(LowestCostPicker),
        )
        .into_cache_free()
        .unwrap();
        let rows = CacheFreeRows(vec![
            (WorkerWithDpRank::from_worker_id(41), 0),
            (WorkerWithDpRank::from_worker_id(42), 0),
        ]);

        let selected = policy
            .select(&CacheFreeRequestContext::new("request", 32, false), &rows)
            .unwrap();
        assert_eq!(selected.index, 1);
        assert_eq!(selected.allowed_worker_ids, Some(HashSet::from([42])));
    }

    #[test]
    fn serializes_state_and_keeps_advisory_selection_read_only() {
        struct StatefulPicker {
            next: usize,
        }

        impl WorkerPicker for StatefulPicker {
            fn pick(
                &mut self,
                context: &WorkerSelectionContext<'_>,
                input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                let row = self.next % input.candidates().len();
                if !context.is_advisory() {
                    self.next += 1;
                }
                Ok(row)
            }
        }

        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<CacheFreeWorkerSelectionPolicy>();
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            Vec::new(),
            Box::new(StatefulPicker { next: 0 }),
        )
        .into_cache_free()
        .unwrap();
        let rows = CacheFreeRows(vec![
            (WorkerWithDpRank::from_worker_id(7), 0),
            (WorkerWithDpRank::from_worker_id(8), 0),
        ]);
        let advisory = CacheFreeRequestContext::new("query", 8, true);
        let committed = CacheFreeRequestContext::new("request", 8, false);

        assert_eq!(policy.select(&advisory, &rows).unwrap().index, 0);
        assert_eq!(policy.select(&advisory, &rows).unwrap().index, 0);
        assert_eq!(policy.select(&committed, &rows).unwrap().index, 0);
        assert_eq!(policy.select(&committed, &rows).unwrap().index, 1);
    }

    #[test]
    fn rejects_kv_and_full_load_inputs() {
        struct CachePicker;

        impl WorkerPicker for CachePicker {
            fn required_worker_inputs(&self) -> WorkerInputs {
                WorkerInputs::CACHE
            }

            fn pick(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                _input: WorkerInputView<'_>,
            ) -> Result<usize, WorkerSelectionPolicyError> {
                Ok(0)
            }
        }

        struct FullLoadScorer;

        impl WorkerScorer for FullLoadScorer {
            fn required_worker_inputs(&self) -> WorkerInputs {
                WorkerInputs::LOAD
            }

            fn score(
                &mut self,
                _context: &WorkerSelectionContext<'_>,
                _candidate: &WorkerCandidate,
            ) -> Result<f64, WorkerSelectionPolicyError> {
                Ok(0.0)
            }
        }

        let cache_error = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            Vec::new(),
            Box::new(CachePicker),
        )
        .into_cache_free()
        .err()
        .expect("cache inputs must be rejected");
        assert!(cache_error.to_string().contains("active-request load"));

        let full_load_policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "decode",
            vec![Box::new(FullLoadScorer)],
            Box::new(LowestCostPicker),
        );
        assert!(full_load_policy.requirements().needs_cache_index());
        let full_load_error = full_load_policy
            .into_cache_free()
            .err()
            .expect("full load inputs must be rejected");
        assert!(full_load_error.to_string().contains("active-request load"));
    }
}
