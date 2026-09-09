// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_custom_policy_example_adaptive::controller::{
    AdaptiveRouter, Algorithm, Candidate, Parameters,
};
use std::time::Duration;

fn rows(a: usize, b: usize) -> [Candidate; 2] {
    [
        Candidate {
            affinity: 1.0,
            active_requests: a,
        },
        Candidate {
            affinity: 0.0,
            active_requests: b,
        },
    ]
}

fn router(algorithm: Algorithm) -> AdaptiveRouter {
    AdaptiveRouter::new(Parameters {
        algorithm,
        seed: Some(42),
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn hotspot_shifts_budget_and_selection_then_recovers() {
    for algorithm in [Algorithm::Sigmoid, Algorithm::Aimd] {
        let mut router = router(algorithm);
        assert_eq!(
            router.select(Duration::ZERO, rows(1, 0).into_iter()),
            Some(0)
        );
        for tick in 1..100 {
            router.select(Duration::from_millis(tick * 100), rows(64, 0).into_iter());
        }
        assert!(router.snapshot().distribution_weight > 0.7);
        assert_eq!(
            router.select(Duration::from_secs(10), rows(64, 0).into_iter()),
            Some(1)
        );
        for tick in 101..301 {
            router.select(Duration::from_millis(tick * 100), rows(0, 0).into_iter());
        }
        assert!(router.snapshot().distribution_weight < 0.101);
        assert_eq!(
            router.select(Duration::from_secs(31), rows(0, 0).into_iter()),
            Some(0)
        );
    }
}

#[test]
fn update_cadence_and_slew_are_bounded() {
    let mut router = router(Algorithm::Sigmoid);
    router.select(Duration::ZERO, rows(1000, 0).into_iter());
    let first = router.snapshot();
    for _ in 0..1000 {
        router.select(Duration::from_millis(99), rows(1000, 0).into_iter());
    }
    assert_eq!(router.snapshot().updates, first.updates);
    router.select(Duration::from_secs(1000), rows(1000, 0).into_iter());
    assert_eq!(router.snapshot().updates, first.updates + 1);
    assert!(
        (router.snapshot().distribution_weight - first.distribution_weight).abs() <= 0.1 + 1e-12
    );
    router.select(Duration::ZERO, rows(1000, 0).into_iter());
    assert_eq!(router.snapshot().updates, first.updates + 1);
}

#[test]
fn empty_singleton_and_topology_changes_need_no_identity_state() {
    let mut router = router(Algorithm::Sigmoid);
    assert_eq!(router.select(Duration::ZERO, [].into_iter()), None);
    assert_eq!(router.snapshot().updates, 0);
    assert_eq!(
        router.select(Duration::ZERO, rows(100, 0)[..1].iter().copied()),
        Some(0)
    );
    assert_eq!(router.snapshot().imbalance, 0.0);
    assert_eq!(
        router.select(Duration::from_secs(1), rows(0, 1).into_iter()),
        Some(0)
    );
}

#[test]
fn low_absolute_load_does_not_trigger_full_distribution() {
    for algorithm in [Algorithm::Sigmoid, Algorithm::Aimd] {
        let mut router = router(algorithm);
        for tick in 0..100 {
            assert_eq!(
                router.select(Duration::from_millis(tick * 100), rows(1, 0).into_iter()),
                Some(0)
            );
        }
        assert!(router.snapshot().distribution_weight < 0.2);
    }
}

#[test]
fn pressure_caps_distribution_only_when_every_candidate_is_busy() {
    let p = Parameters {
        smoothing: 1.0,
        max_step: 1.0,
        ..Default::default()
    };
    let mut idle_alternative = AdaptiveRouter::new(p.clone()).unwrap();
    let mut busy_alternative = AdaptiveRouter::new(p).unwrap();
    idle_alternative.select(Duration::ZERO, rows(1_000_000, 0).into_iter());
    busy_alternative.select(Duration::ZERO, rows(1_000_000, 1000).into_iter());
    assert!(idle_alternative.snapshot().distribution_weight > 0.89);
    assert!(busy_alternative.snapshot().distribution_weight < 0.51);
    assert!(busy_alternative.snapshot().pressure > 0.99);
}

#[test]
fn nonfinite_cache_is_cold_and_no_cache_chooses_least_load() {
    for affinity in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        let mut router = router(Algorithm::Sigmoid);
        assert_eq!(
            router.select(
                Duration::ZERO,
                [
                    Candidate {
                        affinity,
                        active_requests: 1
                    },
                    Candidate {
                        affinity: 0.0,
                        active_requests: 0
                    }
                ]
                .into_iter()
            ),
            Some(1)
        );
    }
}

#[test]
fn tied_candidates_all_receive_traffic_and_seed_replays() {
    let candidates = [Candidate {
        affinity: 0.0,
        active_requests: 0,
    }; 4];
    let mut a = router(Algorithm::Sigmoid);
    let mut b = router(Algorithm::Sigmoid);
    let mut counts = [0; 4];
    for _ in 0..10_000 {
        let selected = a.select(Duration::ZERO, candidates.into_iter()).unwrap();
        assert_eq!(
            b.select(Duration::ZERO, candidates.into_iter()),
            Some(selected)
        );
        counts[selected] += 1;
    }
    assert!(counts.into_iter().all(|n| (2300..2700).contains(&n)));
}

#[test]
fn random_inputs_preserve_finite_budgets_and_valid_rows() {
    let mut rng = fastrand::Rng::with_seed(81);
    for algorithm in [Algorithm::Sigmoid, Algorithm::Aimd] {
        let mut router = router(algorithm);
        for tick in 0..10_000 {
            let candidates: Vec<_> = (0..rng.usize(1..64))
                .map(|_| Candidate {
                    affinity: rng.f64() * 2.0,
                    active_requests: rng.usize(..),
                })
                .collect();
            let before = router.snapshot().distribution_weight;
            let row = router
                .select(
                    Duration::from_millis(tick * 100),
                    candidates.iter().copied(),
                )
                .unwrap();
            let s = router.snapshot();
            assert!(row < candidates.len());
            assert!((0.1..=0.9).contains(&s.distribution_weight));
            assert!((s.distribution_weight - before).abs() <= 0.1 + 1e-12);
            assert!((0.0..=1.0).contains(&s.imbalance));
            assert!((0.0..=1.0).contains(&s.pressure));
        }
    }
}

#[test]
fn parameters_reject_nonfinite_invalid_and_unknown_fields() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        for field in [
            "smoothing",
            "max_step",
            "distribution_min",
            "distribution_max",
            "pressure_max",
            "midpoint",
            "slope",
            "load_scale",
        ] {
            let mut p = Parameters::default();
            match field {
                "smoothing" => p.smoothing = value,
                "max_step" => p.max_step = value,
                "distribution_min" => p.distribution_min = value,
                "distribution_max" => p.distribution_max = value,
                "pressure_max" => p.pressure_max = value,
                "midpoint" => p.midpoint = value,
                "slope" => p.slope = value,
                "load_scale" => p.load_scale = value,
                _ => unreachable!(),
            }
            assert!(p.validate().is_err(), "{field}: {value}");
        }
    }
    assert!(
        Parameters {
            update_interval_ms: 0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        Parameters {
            distribution_min: 0.8,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(serde_json::from_str::<Parameters>(r#"{"unknown":1}"#).is_err());
    assert!(serde_json::from_str::<Parameters>(r#"{"algorithm":"ucb"}"#).is_err());
}
