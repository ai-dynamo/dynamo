// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared correctness harness for [`SyncIndexer`] backends.
//!
//! Every backend is checked the same way:
//! - `differential`: randomized event streams checked against a clean-room
//!   set-semantics reference (`reference.rs`), on one serial event lane and through
//!   [`ThreadPoolIndexer`](super::ThreadPoolIndexer) with 1 and 4 lanes. Overcounts fail
//!   the run. Undercounts are split into hole undercounts, which Dynamo semantics allow
//!   (`test_remove_mid_chain_block`), and unexplained undercounts, which fail it too.
//!   The earliest overcounts are shrunk to a minimal event list.
//! - `known_grouped_removal_overcount`: the shrunk CRTC grouped-removal repro, replayed
//!   through all three drivers.
//! - `soak`: the adversarial race soak from #15614, driving the backend through its own
//!   `SyncIndexer::worker` lanes instead of CRTC internals.
//! - `selftest`: the differential and the strict soak must both fail on a backend that
//!   drops removals.
//!
//! The memory harness needs its own global allocator, so it is the `indexer_memory`
//! bench in `lib/bench/kv_router/indexer_memory.rs`.
//!
//! On ConcurrentRadixTreeCompressed at 22f7f2a103 the differential and the known repro
//! fail until the grouped-removal overcount is fixed. Gate a branch on its own suite,
//! `harness::<name>::`.
//!
//! # Plugging in a backend
//!
//! 1. Implement [`HarnessBackend`] for the backend, in its module if its constructor is
//!    private. Only `harness_new` is required; the other hooks feed optional checks.
//!    ```ignore
//!    impl HarnessBackend for ArenaIndex {
//!        fn harness_new() -> Self {
//!            Self::new()
//!        }
//!    }
//!    ```
//! 2. Add `harness_suite!(name, path::to::Backend);` at the bottom of this file. It
//!    generates `harness::name::{differential_serial, differential_pool1,
//!    differential_pool4, known_grouped_removal_overcount, race_soak_strict,
//!    race_soak_chaos, race_soak}`; the soaks are `#[ignore]`d.
//! 3. For memory, add an arm to `run_backend` in `indexer_memory.rs`.
//!
//! # Commands
//!
//! ```text
//! # Differential, 40 seeds per mode, plus the known repro:
//! cargo test -p dynamo-kv-router --release --lib harness::crtc:: -- --nocapture
//! # 1000 seeds, or one seed with its whole stream printed:
//! HARNESS_SEEDS=1000 cargo test -p dynamo-kv-router --release --lib harness::crtc::differential -- --nocapture
//! HARNESS_SEED=20 HARNESS_TRACE=1 cargo test -p dynamo-kv-router --release --lib harness::crtc::differential_serial -- --nocapture
//! # Race soak (opt-in), strict and chaos, with churn and recycled overflow slots:
//! SOAK_SECS=60 SOAK_CHURN=20 SOAK_SLOT_OFFSET=300 \
//!   cargo test -p dynamo-kv-router --release --lib harness::crtc::race_soak_strict -- --ignored --nocapture
//! SOAK_SECS=60 SOAK_CHURN=20 SOAK_SLOT_OFFSET=300 \
//!   cargo test -p dynamo-kv-router --release --lib harness::crtc::race_soak_chaos -- --ignored --nocapture
//! # Harness self-test:
//! cargo test -p dynamo-kv-router --release --lib harness::selftest -- --nocapture
//! ```
//!
//! Differential knobs (environment variables, default in parentheses):
//! - `HARNESS_SEEDS` (40): seeds per mode, starting at `HARNESS_SEED_BASE` (1).
//! - `HARNESS_SEED`: run only this seed.
//! - `HARNESS_STEPS` (2000): operations per seed.
//! - `HARNESS_DOC_LEN` (6): maximum document length in blocks; every fourth document is
//!   four times longer, so compressed edges also exceed 16 blocks.
//! - `HARNESS_SHRINK` (1): shrink the two earliest failing seeds and print the smaller
//!   repro. Replays apply one operation at a time on the mode's lane count, or, when the
//!   overcount needs concurrency, the run's own pipelined batches.
//! - `HARNESS_STRICT` (1): also fail on unexplained undercounts and on valid operations the
//!   backend rejects; `0` only reports them.
//! - `HARNESS_TRACE` (0): print every generated operation, with checkpoints marked.
//!
//! Soak knobs keep #15614's names; see `soak.rs`.

use std::sync::Arc;

use super::{MatchDetails, SyncIndexer};
use crate::protocols::{LocalBlockHash, WorkerWithDpRank};

mod differential;
mod driver;
mod reference;
mod selftest;
mod soak;
mod workload;

/// A [`SyncIndexer`] backend the harness can build and check.
pub(crate) trait HarnessBackend: SyncIndexer + Sized {
    /// Builds an empty backend.
    fn harness_new() -> Self;

    /// Scores plus each scored rank's last matched sequence hash, for the hash-mismatch
    /// check. Backends without detailed lookups return `None`.
    fn harness_match_details(&self, _sequence: &[LocalBlockHash]) -> Option<MatchDetails> {
        None
    }

    /// The dense coverage slot `rank` holds, if the backend has one. The soak counts slots
    /// handed to a different rank than their previous owner.
    fn harness_rank_slot(&self, _rank: WorkerWithDpRank) -> Option<usize> {
        None
    }

    /// Structural size after a final cleanup (child edges, runs, ...), for the soak report.
    fn harness_structure_size(&self) -> Option<usize> {
        None
    }
}

pub(crate) fn env_u64(name: &str, default: u64) -> u64 {
    let Ok(value) = std::env::var(name) else {
        return default;
    };
    value
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be an unsigned integer, got {value:?}"))
}

fn new_backend<T: HarnessBackend>() -> Arc<T> {
    Arc::new(T::harness_new())
}

/// Generates the shared suite for one backend.
macro_rules! harness_suite {
    ($name:ident, $backend:ty) => {
        mod $name {
            use super::differential::{self, Mode};
            use super::soak;

            #[test]
            fn differential_serial() {
                differential::run_suite::<$backend>(stringify!($name), Mode::Serial);
            }

            #[test]
            fn differential_pool1() {
                differential::run_suite::<$backend>(stringify!($name), Mode::Pool { lanes: 1 });
            }

            #[test]
            fn differential_pool4() {
                differential::run_suite::<$backend>(stringify!($name), Mode::Pool { lanes: 4 });
            }

            #[test]
            fn known_grouped_removal_overcount() {
                differential::known_grouped_removal_overcount::<$backend>(stringify!($name));
            }

            #[test]
            #[ignore = "long-running race soak; run explicitly"]
            fn race_soak_strict() {
                soak::run::<$backend>(stringify!($name), Some(false));
            }

            #[test]
            #[ignore = "long-running race soak; run explicitly"]
            fn race_soak_chaos() {
                soak::run::<$backend>(stringify!($name), Some(true));
            }

            /// Mode from `SOAK_MODE`, as in #15614.
            #[test]
            #[ignore = "long-running race soak; run explicitly"]
            fn race_soak() {
                soak::run::<$backend>(stringify!($name), None);
            }
        }
    };
}

harness_suite!(
    crtc,
    crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed
);
