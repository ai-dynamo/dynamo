// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One tick of the control loop: load snapshot in, an `Update` out (or `None` when nothing changes).
//! Owns the smoothed imbalance, the engaged/disengaged state (hysteresis) and the dwell counter.

use std::time::Instant;

use crate::Parameters;
use crate::math::{imbalance, sigmoid, smooth, weights_for};
use crate::state::{LoadSnapshot, Weights};

/// A change of the engaged state during one tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transition {
    Engaged,
    Disengaged,
}

/// What one tick publishes: the new weights and, when the engaged state flipped, which way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Update {
    pub weights: Weights,
    pub transition: Option<Transition>,
}

pub struct Controller {
    params: Parameters,
    baseline: Weights,
    smoothed: f64,
    engaged: bool,
    cycles_in_state: u32,
    last: Weights,
}

impl Controller {
    pub fn new(params: &Parameters, baseline: Weights) -> Self {
        Self {
            params: params.clone(),
            baseline,
            smoothed: 0.0,
            engaged: false,
            // Start "long settled" so the first engage is not delayed by the dwell; only flips back are.
            cycles_in_state: u32::MAX,
            last: baseline,
        }
    }

    /// Advance one cycle at time `now`. Returns `Some` only when the published set should change.
    /// A missing snapshot, or one older than `stale_after_ms`, counts as an idle pool, so the loop
    /// drains back to baseline. The scorer only refreshes the snapshot when it scores a request, so
    /// the window must cover the longest gap between requests the loop should ride through.
    pub fn tick(&mut self, snapshot: &LoadSnapshot, now: Instant) -> Option<Update> {
        let fresh = snapshot
            .taken_at
            .is_some_and(|t| now.saturating_duration_since(t) <= self.params.stale_after());
        let raw = if fresh {
            if snapshot.active_requests.len() < 2 {
                return None; // not enough data this cycle, keep what we have
            }
            imbalance(&snapshot.active_requests)
        } else {
            0.0
        };
        self.smoothed = smooth(self.smoothed, raw, self.params.smoothing);
        self.cycles_in_state = self.cycles_in_state.saturating_add(1);

        // Hysteresis with dwell: enter above enter_threshold, leave below exit_threshold,
        // and never flip before dwell_cycles have passed in the current state.
        let can_flip = self.cycles_in_state >= self.params.dwell_cycles;
        let mut transition = None;
        if !self.engaged && can_flip && self.smoothed > self.params.enter_threshold {
            self.engaged = true;
            self.cycles_in_state = 0;
            transition = Some(Transition::Engaged);
        } else if self.engaged && can_flip && self.smoothed < self.params.exit_threshold {
            self.engaged = false;
            self.cycles_in_state = 0;
            transition = Some(Transition::Disengaged);
        }

        // The split jumps from 0 to the sigmoid value on engage and back to 0 on disengage, on purpose:
        // the hysteresis band and the dwell bound how often that step can happen.
        let split = if self.engaged {
            sigmoid(
                self.smoothed,
                self.params.sigmoid_midpoint,
                self.params.sigmoid_slope,
            )
        } else {
            0.0
        };
        let next = Weights {
            set: weights_for(self.baseline.set, split, self.params.gain),
            temperature: self.last.temperature,
            imbalance: self.smoothed,
            split,
        };
        // The set and the temperature are deterministic in `split`, so an exact compare is enough.
        // A transition always moves the split between 0 and a positive sigmoid value, so it is never dropped here.
        if next.split == self.last.split {
            return None;
        }
        self.last = next;
        Some(Update {
            weights: next,
            transition,
        })
    }

    #[cfg(test)]
    pub fn engaged(&self) -> bool {
        self.engaged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::WeightSet;

    fn baseline() -> Weights {
        Weights {
            set: WeightSet {
                overlap_score_credit: 1.0,
                prefill_load_scale: 1.0,
                decode_active_request_weight: 0.5,
            },
            temperature: 0.0,
            imbalance: 0.0,
            split: 0.0,
        }
    }
    fn snap(v: &[f64], taken_at: Instant) -> LoadSnapshot {
        LoadSnapshot {
            active_requests: v.to_vec(),
            taken_at: Some(taken_at),
        }
    }
    /// One tick on a snapshot taken at the same instant the loop reads it.
    fn tick_fresh(c: &mut Controller, v: &[f64]) -> Option<Update> {
        let now = Instant::now();
        c.tick(&snap(v, now), now)
    }
    fn params() -> Parameters {
        Parameters {
            smoothing: 0.3,
            sigmoid_midpoint: 0.5,
            sigmoid_slope: 10.0,
            gain: 4.0,
            dwell_cycles: 2,
            enter_threshold: 0.6,
            exit_threshold: 0.4,
            ..Parameters::default()
        }
    }

    #[test]
    fn even_load_never_engages_or_publishes() {
        let mut c = Controller::new(&params(), baseline());
        for _ in 0..20 {
            assert!(tick_fresh(&mut c, &[5.0, 4.0, 6.0, 5.0]).is_none());
        }
        assert!(!c.engaged());
    }

    #[test]
    fn hotspot_engages_and_lowers_credit_then_drains_back() {
        let mut c = Controller::new(&params(), baseline());
        let mut last = None;
        let mut transitions = Vec::new();
        for _ in 0..8 {
            if let Some(u) = tick_fresh(&mut c, &[30.0, 4.0, 2.0, 5.0]) {
                transitions.extend(u.transition);
                last = Some(u.weights);
            }
        }
        let hot = last.expect("hotspot must publish new weights");
        assert!(c.engaged());
        assert_eq!(
            transitions,
            vec![Transition::Engaged],
            "engaging is reported once; later updates while engaged carry no transition"
        );
        assert!(
            hot.set.overlap_score_credit < 0.2,
            "credit={}",
            hot.set.overlap_score_credit
        );
        assert!(hot.set.prefill_load_scale > 3.0);

        let mut back = None;
        for _ in 0..12 {
            if let Some(u) = tick_fresh(&mut c, &[9.0, 9.0, 8.0, 9.0]) {
                back = Some(u);
            }
        }
        let calm = back.expect("draining must publish again");
        assert!(!c.engaged());
        assert_eq!(calm.weights.set, baseline().set);
        assert_eq!(calm.transition, Some(Transition::Disengaged));
    }

    #[test]
    fn dwell_blocks_an_immediate_flip_back() {
        let p = Parameters {
            dwell_cycles: 3,
            ..params()
        };
        let mut c = Controller::new(&p, baseline());
        // one hot tick: smoothed 0.7 > enter 0.6, engages and starts the dwell clock
        let engaged = tick_fresh(&mut c, &[30.0, 4.0, 2.0, 5.0]);
        assert!(c.engaged());
        assert_eq!(
            engaged.and_then(|u| u.transition),
            Some(Transition::Engaged)
        );
        // calm cycles 1 and 2 are below dwell_cycles: still engaged although smoothed < exit
        for _ in 0..2 {
            let held = tick_fresh(&mut c, &[5.0, 5.0, 5.0, 5.0]);
            assert!(c.engaged());
            assert_eq!(held.and_then(|u| u.transition), None);
        }
        // calm cycle 3 reaches the dwell: disengages
        let released = tick_fresh(&mut c, &[5.0, 5.0, 5.0, 5.0]);
        assert!(!c.engaged());
        assert_eq!(
            released.and_then(|u| u.transition),
            Some(Transition::Disengaged)
        );
    }

    #[test]
    fn single_worker_snapshot_is_ignored() {
        let mut c = Controller::new(&params(), baseline());
        assert!(tick_fresh(&mut c, &[30.0]).is_none());
        assert!(tick_fresh(&mut c, &[]).is_none());
    }

    #[test]
    fn stale_hot_snapshot_drains_back_to_baseline() {
        let p = params();
        let mut c = Controller::new(&p, baseline());
        for _ in 0..8 {
            tick_fresh(&mut c, &[30.0, 4.0, 2.0, 5.0]);
        }
        assert!(c.engaged());

        // The request path went quiet: the same hot sample keeps being read, but it ages past the limit.
        let taken_at = Instant::now();
        let hot = snap(&[30.0, 4.0, 2.0, 5.0], taken_at);
        let mut back = None;
        for i in 1..=12 {
            let now = taken_at + p.stale_after() + p.interval() * i;
            if let Some(u) = c.tick(&hot, now) {
                back = Some(u.weights);
            }
        }
        let calm = back.expect("a stale sample must drain the weights");
        assert!(!c.engaged());
        assert_eq!(calm.set, baseline().set);
        assert_eq!(calm.split, 0.0);
    }

    #[test]
    fn gap_shorter_than_stale_after_keeps_the_loop_engaged() {
        let p = params();
        let mut c = Controller::new(&p, baseline());
        for _ in 0..8 {
            tick_fresh(&mut c, &[30.0, 4.0, 2.0, 5.0]);
        }
        assert!(c.engaged());

        // No request for eight intervals, but still inside the staleness window: the loop holds.
        let taken_at = Instant::now();
        let hot = snap(&[30.0, 4.0, 2.0, 5.0], taken_at);
        for i in 1..=8 {
            let now = taken_at + p.interval() * i;
            assert!(
                now - taken_at < p.stale_after(),
                "test premise: inside the window"
            );
            c.tick(&hot, now);
        }
        assert!(c.engaged());
    }

    #[test]
    fn missing_snapshot_counts_as_idle() {
        let mut c = Controller::new(&params(), baseline());
        let empty = LoadSnapshot::default();
        for _ in 0..10 {
            assert!(c.tick(&empty, Instant::now()).is_none());
        }
        assert!(!c.engaged());
    }
}
