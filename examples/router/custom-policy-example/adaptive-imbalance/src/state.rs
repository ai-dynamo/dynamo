// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State shared between the request path (scorer, picker) and the control loop.
//! Readers take one `load()` guard; writers build a new value and `store()` it. No locks on the request path.

use std::sync::Arc;
use std::time::Instant;

use arc_swap::{ArcSwap, Guard};
use dynamo_kv_router::KvRouterConfig;

use crate::math::WeightSet;

/// The values the request path reads. Immutable once published.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weights {
    pub set: WeightSet,
    pub temperature: f64,
    /// The signals that produced this set, for logs.
    pub imbalance: f64,
    pub split: f64,
}

impl Weights {
    pub fn baseline(config: &KvRouterConfig) -> Self {
        Self {
            set: WeightSet {
                overlap_score_credit: config.overlap_score_credit,
                prefill_load_scale: config.prefill_load_scale,
                decode_active_request_weight: config.decode_active_request_weight,
            },
            temperature: config.router_temperature,
            imbalance: 0.0,
            split: 0.0,
        }
    }
}

/// Per-worker load as last observed by the scorer. One entry per candidate row.
#[derive(Clone, Debug, Default)]
pub struct LoadSnapshot {
    pub active_requests: Vec<f64>,
    pub taken_at: Option<Instant>,
}

pub struct SharedState {
    weights: ArcSwap<Weights>,
    load: ArcSwap<LoadSnapshot>,
}

impl SharedState {
    pub fn new(initial: Weights) -> Self {
        Self {
            weights: ArcSwap::from_pointee(initial),
            load: ArcSwap::from_pointee(LoadSnapshot::default()),
        }
    }

    /// Cheap read for the request path. Use the value and drop the guard; do not hold it across an await.
    pub fn weights(&self) -> Guard<Arc<Weights>> {
        self.weights.load()
    }
    pub fn publish_weights(&self, w: Weights) {
        self.weights.store(Arc::new(w));
    }

    pub fn load_snapshot(&self) -> Arc<LoadSnapshot> {
        self.load.load_full()
    }
    pub fn publish_load(&self, active_requests: Vec<f64>) {
        self.load.store(Arc::new(LoadSnapshot {
            active_requests,
            taken_at: Some(Instant::now()),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(split: f64) -> Weights {
        Weights {
            set: WeightSet {
                overlap_score_credit: 1.0 - split,
                prefill_load_scale: 1.0,
                decode_active_request_weight: 0.5,
            },
            temperature: 0.0,
            imbalance: 0.0,
            split,
        }
    }

    #[test]
    fn readers_see_whole_sets_only() {
        let s = SharedState::new(w(0.0));
        assert_eq!(s.weights().split, 0.0);
        s.publish_weights(w(0.9));
        let seen = s.weights();
        assert_eq!(seen.split, 0.9);
        assert!((seen.set.overlap_score_credit - 0.1).abs() < 1e-9);
    }

    #[test]
    fn load_snapshot_starts_empty_and_records_time() {
        let s = SharedState::new(w(0.0));
        assert!(s.load_snapshot().taken_at.is_none());
        s.publish_load(vec![30.0, 4.0, 2.0, 5.0]);
        let snap = s.load_snapshot();
        assert_eq!(snap.active_requests, vec![30.0, 4.0, 2.0, 5.0]);
        assert!(snap.taken_at.is_some());
    }
}
