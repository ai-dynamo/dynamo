// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loom models of the CRTC volume trigger (`trigger.rs`), composed with the one-in-flight
//! scheduling flag the way `ConcurrentRadixTreeCompressed::try_schedule_cleanup` and
//! `CleanupGuard` use `CleanupState`'s.
//!
//! Every model has a negative control that must fail, which shows loom explores the
//! interleaving the model guards against.

#![cfg(loom)]

#[path = "../../src/indexer/concurrent_radix_tree_compressed/trigger.rs"]
#[allow(dead_code)]
mod trigger;

#[cfg(test)]
mod models {
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use loom::thread;

    use super::trigger::VolumeTrigger;

    /// A fixed clock: the models are about orderings, not time.
    const NOW_US: u64 = 1_000;

    /// Bounded preemption keeps the four-thread models tractable; three preemptions
    /// cover every interleaving of the two schedulers' claim with each other and with
    /// the sweep's release.
    fn model(f: impl Fn() + Sync + Send + 'static) {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.check(f);
    }

    /// The volume path of `try_schedule_cleanup` plus a sweep that `CleanupGuard` ends.
    struct Scheduler {
        trigger: VolumeTrigger,
        /// `CleanupState::scheduled`.
        scheduled: AtomicBool,
        in_flight: AtomicUsize,
        sweeps: AtomicUsize,
        /// The negative control schedules with a load and a store instead of a CAS.
        racy: bool,
    }

    impl Scheduler {
        fn new(racy: bool) -> Self {
            Self {
                trigger: VolumeTrigger::new(true, 1, 0),
                scheduled: AtomicBool::new(false),
                in_flight: AtomicUsize::new(0),
                sweeps: AtomicUsize::new(0),
                racy,
            }
        }

        /// `CleanupState::try_claim`.
        fn claim(&self) -> bool {
            if self.racy {
                if self.scheduled.load(Ordering::Acquire) {
                    return false;
                }
                self.scheduled.store(true, Ordering::Relaxed);
                return true;
            }
            self.scheduled
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        }

        fn try_schedule(&self) -> bool {
            self.trigger.due(|| NOW_US) && self.claim()
        }

        /// The sweep task, then `CleanupGuard::drop`.
        fn sweep(&self) {
            assert_eq!(
                self.in_flight.fetch_add(1, Ordering::Relaxed),
                0,
                "two sweeps in flight"
            );
            self.sweeps.fetch_add(1, Ordering::Relaxed);
            // The recount found everything dead and unlinked it.
            self.trigger.finish(NOW_US, 0, 0, 0);
            self.in_flight.fetch_sub(1, Ordering::Relaxed);
            self.scheduled.store(false, Ordering::Release);
        }
    }

    /// One lane flushes dead volume while two producers race to schedule. At most one
    /// sweep is ever in flight, and the flag is never left set: once everything has
    /// joined, a due trigger can always schedule again.
    fn one_sweep_in_flight(racy: bool) {
        model(move || {
            let scheduler = Arc::new(Scheduler::new(racy));
            scheduler.trigger.add(4, 4);
            let lane = {
                let scheduler = scheduler.clone();
                thread::spawn(move || scheduler.trigger.add(2, 2))
            };
            let producers: Vec<_> = (0..2)
                .map(|_| {
                    let scheduler = scheduler.clone();
                    thread::spawn(move || {
                        if scheduler.try_schedule() {
                            scheduler.sweep();
                        }
                    })
                })
                .collect();
            lane.join().unwrap();
            for producer in producers {
                producer.join().unwrap();
            }

            assert!(!scheduler.scheduled.load(Ordering::Relaxed));
            if scheduler.trigger.due(|| NOW_US) {
                assert!(scheduler.try_schedule(), "flag stuck while a sweep is due");
                scheduler.sweep();
            }
            assert!(scheduler.sweeps.load(Ordering::Relaxed) >= 1);
        });
    }

    #[test]
    fn at_most_one_sweep_in_flight() {
        one_sweep_in_flight(false);
    }

    #[test]
    #[should_panic(expected = "two sweeps in flight")]
    fn negative_control_racy_claim_runs_two_sweeps() {
        one_sweep_in_flight(true);
    }

    /// Lanes flushing while a sweep overwrites the estimates may lose the deltas the
    /// overwrite raced with (the estimates drift until the next recount), but never tear:
    /// the values stay within what the flushes and the overwrite can explain, and a flush
    /// ordered after the sweep always counts.
    ///
    /// Loom lets a relaxed fetch-add that raced a store come last in the modification
    /// order while reading the value before the store, which hardware coherence rules
    /// out; the bounds below hold under either.
    #[test]
    fn flushes_racing_an_overwrite_drift_but_never_tear() {
        model(|| {
            let trigger = Arc::new(VolumeTrigger::new(true, 1, 0));
            trigger.add(10, 10);
            let lanes: Vec<_> = (0..2)
                .map(|_| {
                    let trigger = trigger.clone();
                    thread::spawn(move || trigger.add(1, 1))
                })
                .collect();
            let sweep = {
                let trigger = trigger.clone();
                thread::spawn(move || trigger.finish(NOW_US, 0, 3, 0))
            };
            sweep.join().unwrap();
            for lane in lanes {
                lane.join().unwrap();
            }
            let (dead, linked) = trigger.estimates();
            assert!(dead <= 12, "dead {dead}");
            assert!((3..=12).contains(&linked), "linked {linked}");
            trigger.add(1, 1);
            assert_eq!(trigger.estimates(), (dead + 1, linked + 1));
        });
    }

    /// The negative control for the overwrite model: an overwrite is not a flush, so
    /// asserting that every flush survives it must fail in some interleaving.
    #[test]
    #[should_panic(expected = "flush lost")]
    fn negative_control_overwrite_can_absorb_a_racing_flush() {
        model(|| {
            let trigger = Arc::new(VolumeTrigger::new(true, 1, 0));
            let lane = {
                let trigger = trigger.clone();
                thread::spawn(move || trigger.add(1, 0))
            };
            trigger.finish(NOW_US, 0, 0, 0);
            lane.join().unwrap();
            assert_eq!(trigger.estimates().0, 1, "flush lost");
        });
    }

    /// The gap rule: a sweep that took `d` blocks volume sweeps for `max(min_gap, 10 d)`.
    #[test]
    fn gap_rule_spaces_volume_sweeps() {
        model(|| {
            let trigger = VolumeTrigger::new(true, 1, 50);
            trigger.add(5, 5);
            trigger.finish(1_000, 20, 5, 5);
            assert!(!trigger.due(|| 1_000 + 199));
            assert!(trigger.due(|| 1_000 + 200));
            trigger.finish(2_000, 1, 5, 5);
            assert!(!trigger.due(|| 2_000 + 49));
            assert!(trigger.due(|| 2_000 + 50));
        });
    }
}
