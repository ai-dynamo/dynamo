// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local, attempt-fenced observations for opt-in decoder-progress policies.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::AttemptId;
use crate::{RoutingPartitionId, RoutingPartitionRef};
type Key = (Arc<RoutingPartitionId>, String);
use crate::plugins::worker_selection::ScoredWorkerCandidate;
use crate::protocols::WorkerWithDpRank;
use rustc_hash::FxHashSet;

// A full table rejects new observations; policy completeness checks must then fall back.
const MAX_ENTRIES: usize = 4096;
static REGISTRY: OnceLock<Arc<Registry>> = OnceLock::new();

/// One frontend-observed active attempt, borrowed through a selection snapshot.
/// Request identifiers and mutable lifecycle state are deliberately not exposed.
#[derive(Clone, Debug)]
pub struct DecodeProgressEntry {
    partition: Arc<RoutingPartitionId>,
    worker: WorkerWithDpRank,
    max_output_tokens: Option<u32>,
    ignore_eos: bool,
    generated_tokens: usize,
    observed_age: Duration,
}

impl DecodeProgressEntry {
    /// Canonical logical model and routing group of the observation producer.
    pub fn partition(&self) -> RoutingPartitionRef<'_> {
        RoutingPartitionRef::new(&self.partition.model_name, &self.partition.routing_group)
    }
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }
    /// The request's declared token limit, not predicted future output.
    pub fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }
    pub fn ignores_eos(&self) -> bool {
        self.ignore_eos
    }
    /// Original output token IDs observed by this frontend for this attempt.
    pub fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }
    /// Age at snapshot capture; it is not updated during the picker callback.
    pub fn observed_age(&self) -> Duration {
        self.observed_age
    }
    /// Declared remaining budget. Early stopping can make actual future work smaller.
    pub fn remaining_token_budget(&self) -> Option<usize> {
        usize::try_from(self.max_output_tokens?)
            .ok()?
            .checked_sub(self.generated_tokens)
    }
}

/// Read-only facts captured under one lock immediately before a picker callback.
/// This process-local snapshot is not a backend/GPU clock or a completion-time estimate.
pub struct DecodeProgressSnapshot {
    entries: Vec<DecodeProgressEntry>,
}

impl DecodeProgressSnapshot {
    pub fn entries(&self) -> &[DecodeProgressEntry] {
        &self.entries
    }
}

struct Record {
    attempt: AttemptId,
    worker: WorkerWithDpRank,
    max_output_tokens: Option<u32>,
    ignore_eos: bool,
    generated_tokens: usize,
    observed_at: Instant,
    consistent: bool,
}

struct Registry {
    records: Mutex<HashMap<Key, Record>>,
    capacity: usize,
}

impl Registry {
    fn new(capacity: usize) -> Self {
        Self {
            records: Mutex::new(HashMap::new()),
            capacity,
        }
    }

    fn admit(&self, id: &Key, record: Record) -> bool {
        let Ok(mut records) = self.records.lock() else {
            return false;
        };
        // A new scheduler attempt may enter only after the old observation is retired.
        if records.len() >= self.capacity || records.contains_key(id) {
            return false;
        }
        records.insert(id.clone(), record);
        true
    }

    fn advance(&self, id: &Key, attempt: AttemptId, generated: usize, now: Instant) {
        let Ok(mut records) = self.records.lock() else {
            return;
        };
        let Some(record) = records.get_mut(id) else {
            return;
        };
        if record.attempt != attempt {
            return;
        }
        if generated < record.generated_tokens
            || now < record.observed_at
            || record
                .max_output_tokens
                .is_some_and(|limit| generated > limit as usize)
        {
            record.consistent = false;
            return;
        }
        // Empty/duplicate chunks do not make an old observation appear fresh.
        if generated == record.generated_tokens {
            return;
        }
        record.generated_tokens = generated;
        record.observed_at = now;
    }

    fn retire(&self, id: &Key, attempt: AttemptId) {
        let Ok(mut records) = self.records.lock() else {
            return;
        };
        if records
            .get(id)
            .is_some_and(|record| record.attempt == attempt)
        {
            records.remove(id);
        }
    }

    fn snapshot_matching(
        &self,
        include: impl Fn(WorkerWithDpRank) -> bool,
        now: impl FnOnce() -> Instant,
    ) -> Option<DecodeProgressSnapshot> {
        let records = self.records.lock().ok()?;
        let now = now();
        let mut entries = Vec::new();
        for (key, record) in records.iter() {
            if !include(record.worker) {
                continue;
            }
            if !record.consistent {
                return None;
            }
            entries.push(DecodeProgressEntry {
                partition: Arc::clone(&key.0),
                worker: record.worker,
                max_output_tokens: record.max_output_tokens,
                ignore_eos: record.ignore_eos,
                generated_tokens: record.generated_tokens,
                observed_age: now.checked_duration_since(record.observed_at)?,
            });
        }
        Some(DecodeProgressSnapshot { entries })
    }

    #[cfg(test)]
    fn snapshot(&self, now: impl FnOnce() -> Instant) -> Option<DecodeProgressSnapshot> {
        self.snapshot_matching(|_| true, now)
    }
}

/// Internal frontend lifecycle hook. The frontend must explicitly enable collection.
/// Admission failure is observation absence, never a request/booking failure.
#[doc(hidden)]
pub struct DecodeProgressGuard {
    registry: Arc<Registry>,
    key: Key,
    attempt: AttemptId,
}

impl DecodeProgressGuard {
    pub fn start(
        partition: RoutingPartitionRef<'_>,
        id: &str,
        attempt: AttemptId,
        worker: WorkerWithDpRank,
        max_output_tokens: Option<u32>,
        ignore_eos: bool,
    ) -> Option<Self> {
        let registry = REGISTRY.get_or_init(|| Arc::new(Registry::new(MAX_ENTRIES)));
        Self::start_in(
            registry,
            partition,
            id,
            attempt,
            worker,
            max_output_tokens,
            ignore_eos,
        )
    }

    fn start_in(
        registry: &Arc<Registry>,
        partition: RoutingPartitionRef<'_>,
        id: &str,
        attempt: AttemptId,
        worker: WorkerWithDpRank,
        max_output_tokens: Option<u32>,
        ignore_eos: bool,
    ) -> Option<Self> {
        let key = (Arc::new(partition.into_owned()), id.to_owned());
        if !registry.admit(
            &key,
            Record {
                attempt,
                worker,
                max_output_tokens,
                ignore_eos,
                generated_tokens: 0,
                observed_at: Instant::now(),
                consistent: true,
            },
        ) {
            return None;
        }
        Some(Self {
            registry: Arc::clone(registry),
            key,
            attempt,
        })
    }

    pub fn advance(&self, generated_tokens: usize) {
        self.registry
            .advance(&self.key, self.attempt, generated_tokens, Instant::now());
    }
}

impl Drop for DecodeProgressGuard {
    fn drop(&mut self) {
        self.registry.retire(&self.key, self.attempt);
    }
}

pub(crate) fn snapshot_now(candidates: &[ScoredWorkerCandidate]) -> Option<DecodeProgressSnapshot> {
    // No allocation or table initialization when the producer has never been enabled.
    let registry = REGISTRY.get()?;
    let workers: FxHashSet<_> = candidates.iter().map(|row| row.worker()).collect();
    registry.snapshot_matching(|worker| workers.contains(&worker), Instant::now)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: &str) -> Key {
        (
            Arc::new(RoutingPartitionId::new("model", "default")),
            id.to_owned(),
        )
    }

    fn record(attempt: u64, now: Instant) -> Record {
        Record {
            attempt: AttemptId::new(attempt),
            worker: WorkerWithDpRank::new(7, 0),
            max_output_tokens: Some(128),
            ignore_eos: true,
            generated_tokens: 0,
            observed_at: now,
            consistent: true,
        }
    }

    #[test]
    fn decode_progress_old_attempt_cannot_update_or_retire_reused_id() {
        let registry = Registry::new(4);
        let now = Instant::now();
        assert!(registry.admit(&key("r"), record(1, now)));
        registry.retire(&key("r"), AttemptId::new(1));
        assert!(registry.admit(&key("r"), record(2, now)));
        registry.advance(&key("r"), AttemptId::new(1), 90, now);
        registry.retire(&key("r"), AttemptId::new(1));
        assert_eq!(
            registry.snapshot(|| now).unwrap().entries()[0].generated_tokens(),
            0
        );
    }

    #[test]
    fn decode_progress_duplicate_admission_does_not_replace_live_owner() {
        let registry = Registry::new(2);
        let now = Instant::now();
        assert!(registry.admit(&key("r"), record(2, now)));
        assert!(!registry.admit(&key("r"), record(1, now)));
        registry.advance(&key("r"), AttemptId::new(2), 10, now);
        assert_eq!(
            registry.snapshot(|| now).unwrap().entries()[0].generated_tokens(),
            10
        );
    }

    #[test]
    fn decode_progress_capacity_rejects_without_evicting_live_entries() {
        let registry = Registry::new(1);
        let now = Instant::now();
        assert!(registry.admit(&key("a"), record(1, now)));
        assert!(!registry.admit(&key("b"), record(2, now)));
        assert_eq!(registry.snapshot(|| now).unwrap().entries().len(), 1);
        registry.retire(&key("a"), AttemptId::new(1));
        assert!(registry.admit(&key("b"), record(2, now)));
    }

    #[test]
    fn decode_progress_empty_update_does_not_refresh_age() {
        let registry = Registry::new(1);
        let now = Instant::now();
        assert!(registry.admit(&key("r"), record(1, now)));
        registry.advance(
            &key("r"),
            AttemptId::new(1),
            0,
            now + Duration::from_secs(2),
        );
        assert_eq!(
            registry
                .snapshot(|| now + Duration::from_secs(3))
                .unwrap()
                .entries()[0]
                .observed_age(),
            Duration::from_secs(3)
        );
    }

    #[test]
    fn decode_progress_regression_invalidates_snapshot_until_retirement() {
        let registry = Registry::new(1);
        let now = Instant::now();
        assert!(registry.admit(&key("r"), record(1, now)));
        registry.advance(&key("r"), AttemptId::new(1), 20, now);
        registry.advance(&key("r"), AttemptId::new(1), 19, now);
        assert!(registry.snapshot(|| now).is_none());
        registry.retire(&key("r"), AttemptId::new(1));
        assert!(registry.snapshot(|| now).unwrap().entries().is_empty());
    }

    #[test]
    fn decode_progress_over_budget_is_not_saturated_to_zero() {
        let registry = Registry::new(1);
        let now = Instant::now();
        assert!(registry.admit(&key("r"), record(1, now)));
        registry.advance(&key("r"), AttemptId::new(1), 129, now);
        assert!(registry.snapshot(|| now).is_none());
    }

    #[test]
    fn decode_progress_unknown_budget_remains_absent() {
        let registry = Registry::new(1);
        let now = Instant::now();
        let mut row = record(1, now);
        row.max_output_tokens = None;
        assert!(registry.admit(&key("r"), row));
        let view = registry.snapshot(|| now).unwrap();
        assert_eq!(view.entries()[0].max_output_tokens(), None);
        assert_eq!(view.entries()[0].remaining_token_budget(), None);
    }

    #[test]
    fn decode_progress_clock_regression_is_not_treated_as_fresh() {
        let registry = Registry::new(1);
        let now = Instant::now();
        assert!(registry.admit(&key("r"), record(1, now)));
        assert!(registry.snapshot(|| now - Duration::from_secs(1)).is_none());
    }

    #[test]
    fn decode_progress_guard_drop_retires_without_logging() {
        let registry = Arc::new(Registry::new(1));
        let guard = DecodeProgressGuard::start_in(
            &registry,
            RoutingPartitionRef::new("model", "default"),
            "r",
            AttemptId::new(1),
            WorkerWithDpRank::new(7, 0),
            Some(128),
            true,
        )
        .unwrap();
        guard.advance(12);
        assert_eq!(
            registry.snapshot(Instant::now).unwrap().entries()[0].remaining_token_budget(),
            Some(116)
        );
        drop(guard);
        assert!(
            registry
                .snapshot(Instant::now)
                .unwrap()
                .entries()
                .is_empty()
        );
    }

    #[test]
    fn decode_progress_same_request_and_attempt_are_isolated_by_partition() {
        let registry = Registry::new(3);
        let now = Instant::now();
        let left = (
            Arc::new(RoutingPartitionId::new("left", "default")),
            "same-id".into(),
        );
        let right = (
            Arc::new(RoutingPartitionId::new("right", "default")),
            "same-id".into(),
        );
        assert!(registry.admit(&left, record(1, now)));
        assert!(registry.admit(&right, record(1, now)));
        registry.advance(&left, AttemptId::new(1), 64, now);
        let snapshot = registry.snapshot(|| now).unwrap();
        assert_eq!(
            snapshot
                .entries()
                .iter()
                .find(|r| r.partition().model_name == "left")
                .unwrap()
                .generated_tokens(),
            64
        );
        assert_eq!(
            snapshot
                .entries()
                .iter()
                .find(|r| r.partition().model_name == "right")
                .unwrap()
                .generated_tokens(),
            0
        );
        registry.retire(&left, AttemptId::new(1));
        assert_eq!(
            registry.snapshot(|| now).unwrap().entries()[0]
                .partition()
                .model_name,
            "right"
        );
    }
}
