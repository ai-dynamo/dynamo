// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_custom_policy_example_adaptive::controller::{
    AdaptiveRouter, Algorithm, Candidate, Parameters, pick_weighted,
};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn main() {
    println!("workers,policy,median_ns,min_ns,max_ns");
    for n in [8, 64, 1024, 4096] {
        let candidates: Vec<_> = (0..n)
            .map(|i| Candidate {
                affinity: (i % 10) as f64 / 10.0,
                active_requests: i % 32,
            })
            .collect();
        let iterations = 10_000_000 / n;
        for policy in ["static", "sigmoid", "sigmoid_tick", "aimd", "aimd_tick"] {
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut router = AdaptiveRouter::new(Parameters {
                    algorithm: if policy.starts_with("aimd") {
                        Algorithm::Aimd
                    } else {
                        Algorithm::Sigmoid
                    },
                    seed: Some(1),
                    ..Default::default()
                })
                .unwrap();
                let mut rng = fastrand::Rng::with_seed(1);
                for _ in 0..100 {
                    black_box(router.select(Duration::ZERO, candidates.iter().copied()));
                }
                let start = Instant::now();
                for i in 0..iterations {
                    let rows = black_box(&candidates).iter().copied();
                    let selected = if policy == "static" {
                        let min = rows.clone().map(|c| c.active_requests).min().unwrap();
                        let max = rows.clone().map(|c| c.active_requests).max().unwrap();
                        pick_weighted(rows, min, max, 0.5, 8.0, &mut rng)
                    } else {
                        let now = Duration::from_millis(if policy.ends_with("_tick") {
                            i as u64 * 100
                        } else {
                            0
                        });
                        router.select(now, rows)
                    };
                    black_box(selected);
                }
                samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{n},{policy},{:.1},{:.1},{:.1}",
                samples[2], samples[0], samples[4]
            );
        }
    }
}
