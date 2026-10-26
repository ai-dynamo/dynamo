// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional raw output facts owned by the native slot-tracker request lifecycle.

use std::collections::BTreeMap;
use std::time::Duration;
use tokio::time::Instant;

use crate::protocols::WorkerWithDpRank;

const MAX_OBSERVATIONS_PER_WORKER: usize = 4096;

/// Host-only metadata accompanying the native attempt-fenced output update.
/// A supported producer supplies its explicit output bound and original token count.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct DecodeProgressUpdate {
    observation: Option<Observation>,
}

impl DecodeProgressUpdate {
    pub fn observed(output_limit: u32, generated_tokens: usize) -> Self {
        Self::observed_at(output_limit, generated_tokens, Instant::now())
    }

    pub fn unavailable() -> Self {
        Self { observation: None }
    }

    fn observed_at(output_limit: u32, generated_tokens: usize, at: Instant) -> Self {
        Self {
            observation: Some(Observation {
                output_limit,
                generated_tokens,
                at,
                valid: output_limit > 0
                    && generated_tokens > 0
                    && generated_tokens <= output_limit as usize,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Observation {
    output_limit: u32,
    generated_tokens: usize,
    at: Instant,
    valid: bool,
}

impl Observation {
    fn remaining(self) -> usize {
        (self.output_limit as usize).saturating_sub(self.generated_tokens)
    }
}

/// Internal derived worker snapshot supplied by the native scheduler host.
/// All observations belong to its current worker-local request owners.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeDecodeProgress {
    observed_requests: usize,
    remaining_token_budget: usize,
    oldest_observed_at: Instant,
}

/// Host-only point lookup into the owning scheduler's derived read model.
#[doc(hidden)]
pub type DecodeProgressProvider<'a> = dyn Fn(WorkerWithDpRank) -> Option<NativeDecodeProgress> + 'a;

/// Additional bookkeeping within one native `ActiveSequences`, without request IDs,
/// separate ownership, lifecycle registration, locks, or global state.
#[derive(Debug, Default)]
pub(crate) struct DecodeProgressAggregate {
    tracked: usize,
    invalid: usize,
    remaining: usize,
    // A counted timestamp index keeps oldest-observation reads O(1) and updates O(log N).
    times: BTreeMap<Instant, usize>,
}

impl DecodeProgressAggregate {
    fn remove_valid(&mut self, observation: Observation) {
        if observation.valid {
            self.remaining -= observation.remaining();
            let count = self.times.get_mut(&observation.at).unwrap();
            *count -= 1;
            if *count == 0 {
                self.times.remove(&observation.at);
            }
        } else {
            self.invalid -= 1;
        }
    }

    pub(crate) fn retire(&mut self, observation: Option<Box<Observation>>) {
        if let Some(observation) = observation {
            self.remove_valid(*observation);
            self.tracked -= 1;
        }
    }

    pub(crate) fn update(
        &mut self,
        owner: &mut Option<Box<Observation>>,
        update: DecodeProgressUpdate,
    ) -> bool {
        if owner.is_none() && self.tracked == MAX_OBSERVATIONS_PER_WORKER {
            return false;
        }
        let mut observation = update.observation.unwrap_or(Observation {
            output_limit: 0,
            generated_tokens: 0,
            at: Instant::now(),
            valid: false,
        });
        if let Some(previous) = owner.as_deref().copied() {
            // Duplicated/empty output is not evidence of a fresh observation.
            if previous.valid
                && observation.valid
                && previous.output_limit == observation.output_limit
                && previous.generated_tokens == observation.generated_tokens
            {
                return false;
            }
            observation.valid &= previous.valid
                && previous.output_limit == observation.output_limit
                && previous.generated_tokens <= observation.generated_tokens
                && previous.at <= observation.at;
            self.remove_valid(previous);
        } else {
            self.tracked += 1;
        }
        if observation.valid {
            if let Some(remaining) = self.remaining.checked_add(observation.remaining()) {
                self.remaining = remaining;
                *self.times.entry(observation.at).or_default() += 1;
            } else {
                observation.valid = false;
            }
        }
        if !observation.valid {
            self.invalid += 1;
        }
        match owner {
            Some(slot) => **slot = observation,
            None => *owner = Some(Box::new(observation)),
        }
        true
    }

    pub(crate) fn snapshot(&self, active_requests: usize) -> Option<NativeDecodeProgress> {
        if active_requests == 0 || self.invalid != 0 || self.tracked != active_requests {
            return None;
        }
        Some(NativeDecodeProgress {
            observed_requests: self.tracked,
            remaining_token_budget: self.remaining,
            oldest_observed_at: *self.times.first_key_value()?.0,
        })
    }
}

/// Callback-scoped raw summary for one eligible worker in the owning scheduler.
/// Remaining declared budget is a bound, not a prediction of future tokens or GPU time.
#[derive(Clone, Copy, Debug)]
pub struct DecodeProgressEntry {
    worker: WorkerWithDpRank,
    observed_requests: usize,
    remaining_token_budget: usize,
    observed_age: Duration,
}

impl DecodeProgressEntry {
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }
    pub fn observed_requests(&self) -> usize {
        self.observed_requests
    }
    /// Sum of explicit output limits minus original tokens observed for each native owner.
    /// Units are raw tokens; unsupported, partial or overflowed sets have no entry.
    pub fn remaining_token_budget(&self) -> usize {
        self.remaining_token_budget
    }
    /// Age of the oldest nonempty original-token observation in this worker summary.
    pub fn observed_age(&self) -> Duration {
        self.observed_age
    }
}

/// Read-only summaries materialized only for an explicitly opted-in picker.
/// The native scheduler supplies the routing scope and owns all underlying lifetimes.
#[derive(Debug)]
pub struct DecodeProgressSnapshot {
    entries: Vec<DecodeProgressEntry>,
}

impl DecodeProgressSnapshot {
    pub fn entries(&self) -> &[DecodeProgressEntry] {
        &self.entries
    }

    pub(crate) fn capture(
        workers: impl Iterator<Item = WorkerWithDpRank>,
        provider: Option<&DecodeProgressProvider<'_>>,
    ) -> Option<Self> {
        let provider = provider?;
        let observed: Vec<_> = workers
            .filter_map(|worker| Some((worker, provider(worker)?)))
            .collect();
        // Capture after every lookup so lock waits cannot make old data appear fresh.
        let now = Instant::now();
        let entries = observed
            .into_iter()
            .filter_map(|(worker, source)| {
                Some(DecodeProgressEntry {
                    worker,
                    observed_requests: source.observed_requests,
                    remaining_token_budget: source.remaining_token_budget,
                    observed_age: now.checked_duration_since(source.oldest_observed_at)?,
                })
            })
            .collect::<Vec<_>>();
        (!entries.is_empty()).then_some(Self { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_age_includes_native_projection_lookup_time() {
        let mut aggregate = DecodeProgressAggregate::default();
        let mut owner = None;
        aggregate.update(&mut owner, DecodeProgressUpdate::observed(128, 8));
        let value = aggregate.snapshot(1).unwrap();
        let provider = |worker: WorkerWithDpRank| {
            if worker.worker_id == 2 {
                std::thread::sleep(Duration::from_millis(5));
                None
            } else {
                Some(value)
            }
        };
        let view = DecodeProgressSnapshot::capture(
            [WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)].into_iter(),
            Some(&provider),
        )
        .unwrap();
        assert!(view.entries()[0].observed_age() >= Duration::from_millis(5));
    }

    #[test]
    fn missing_native_projection_stays_absent_in_the_public_view() {
        let worker = WorkerWithDpRank::new(1, 0);
        assert!(DecodeProgressSnapshot::capture([worker].into_iter(), None).is_none());
        let provider = |_| None;
        assert!(DecodeProgressSnapshot::capture([worker].into_iter(), Some(&provider)).is_none());
    }

    #[test]
    fn complete_owner_set_and_native_retirement_determine_availability() {
        let mut aggregate = DecodeProgressAggregate::default();
        let (mut a, mut b) = (None, None);
        aggregate.update(&mut a, DecodeProgressUpdate::observed(128, 8));
        assert!(aggregate.snapshot(2).is_none());
        aggregate.update(&mut b, DecodeProgressUpdate::observed(24, 8));
        assert_eq!(aggregate.snapshot(2).unwrap().remaining_token_budget, 136);
        aggregate.retire(a.take());
        assert_eq!(aggregate.snapshot(1).unwrap().remaining_token_budget, 16);
        aggregate.retire(b.take());
        assert!(aggregate.snapshot(0).is_none());
        assert!(aggregate.times.is_empty());
    }

    #[test]
    fn duplicate_output_does_not_refresh_oldest_observation() {
        let mut aggregate = DecodeProgressAggregate::default();
        let mut a = None;
        let now = Instant::now();
        aggregate.update(&mut a, DecodeProgressUpdate::observed_at(128, 8, now));
        assert!(!aggregate.update(
            &mut a,
            DecodeProgressUpdate::observed_at(128, 8, now + Duration::from_secs(1))
        ));
        assert_eq!(aggregate.snapshot(1).unwrap().oldest_observed_at, now);
    }

    #[test]
    fn invalid_progress_stays_unavailable_until_native_owner_is_retired() {
        for broken in [
            DecodeProgressUpdate::observed(128, 7),
            DecodeProgressUpdate::observed(128, 129),
            DecodeProgressUpdate::observed(64, 9),
            DecodeProgressUpdate::unavailable(),
        ] {
            let mut aggregate = DecodeProgressAggregate::default();
            let mut a = None;
            aggregate.update(&mut a, DecodeProgressUpdate::observed(128, 8));
            aggregate.update(&mut a, broken);
            aggregate.update(&mut a, DecodeProgressUpdate::observed(128, 10));
            assert!(aggregate.snapshot(1).is_none());
            aggregate.retire(a.take());
            aggregate.update(&mut a, DecodeProgressUpdate::observed(128, 10));
            assert_eq!(aggregate.snapshot(1).unwrap().remaining_token_budget, 118);
        }
    }

    #[test]
    fn oldest_timestamp_tracks_updates_and_tied_owner_removal() {
        let mut aggregate = DecodeProgressAggregate::default();
        let (mut a, mut b) = (None, None);
        let now = Instant::now();
        aggregate.update(&mut a, DecodeProgressUpdate::observed_at(128, 8, now));
        aggregate.update(&mut b, DecodeProgressUpdate::observed_at(128, 8, now));
        aggregate.update(
            &mut a,
            DecodeProgressUpdate::observed_at(128, 9, now + Duration::from_millis(1)),
        );
        assert_eq!(aggregate.snapshot(2).unwrap().oldest_observed_at, now);
        aggregate.retire(b.take());
        assert_eq!(
            aggregate.snapshot(1).unwrap().oldest_observed_at,
            now + Duration::from_millis(1)
        );
    }

    #[test]
    fn capped_native_metadata_stays_unavailable_for_a_partial_worker() {
        let mut aggregate = DecodeProgressAggregate::default();
        let mut owners = vec![None; MAX_OBSERVATIONS_PER_WORKER];
        let now = Instant::now();
        for owner in &mut owners {
            assert!(aggregate.update(owner, DecodeProgressUpdate::observed_at(128, 8, now)));
        }
        let mut extra = None;
        assert!(!aggregate.update(&mut extra, DecodeProgressUpdate::observed(128, 8)));
        assert!(
            aggregate
                .snapshot(MAX_OBSERVATIONS_PER_WORKER + 1)
                .is_none()
        );
        aggregate.retire(owners[0].take());
        assert!(aggregate.update(&mut extra, DecodeProgressUpdate::observed(128, 8)));
        assert_eq!(
            aggregate
                .snapshot(MAX_OBSERVATIONS_PER_WORKER)
                .unwrap()
                .remaining_token_budget,
            MAX_OBSERVATIONS_PER_WORKER * 120
        );
    }
}
