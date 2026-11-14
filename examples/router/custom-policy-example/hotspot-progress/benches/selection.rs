// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[path = "../tests/support/mod.rs"]
mod support;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;
use tokio::time::Instant;

fn selection(c: &mut Criterion) {
    let workers = support::workers();
    let (slots, _bookings) = support::backgrounds(128, 8);
    // Static observations use a longer age only in this selection microbenchmark.
    let policy = support::policy_with_age(60_000);
    let mut rejected = support::request();
    rejected.session_context = None;
    let mut qualified = support::request();
    for request in [&mut rejected, &mut qualified] {
        slots.project_worker_loads_into(
            request.token_seq.as_deref(),
            Instant::now(),
            &mut request.worker_loads,
            None,
        );
    }
    let lookup = |worker| slots.decode_progress_for(worker);
    let mut group = c.benchmark_group("hotspot_progress_selection");
    group
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(30)
        .nresamples(1_000);
    for (name, request, expected) in [
        ("structural_reject", &rejected, support::SOURCE),
        ("qualified", &qualified, support::ALTERNATIVE),
    ] {
        assert_eq!(
            support::select_with_provider(&policy, &workers, request, Some(&lookup)),
            expected,
        );
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(support::select_with_provider(
                    &policy,
                    &workers,
                    request,
                    Some(&lookup),
                ))
            })
        });
        assert_eq!(
            support::select_with_provider(&policy, &workers, request, Some(&lookup)),
            expected,
        );
    }
    group.finish();
}

criterion_group!(benches, selection);
criterion_main!(benches);
