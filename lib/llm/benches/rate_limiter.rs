use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dynamo_llm::http::service::rate_limiter::{
    RateLimiter, RateLimiterConfig, TimeWeightedAverageTracker,
};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// Benchmark configurations
const SAMPLE_SIZES: &[usize] = &[10, 100, 1000, 10000];
const TIME_CONSTANTS: &[f64] = &[1.0, 10.0, 30.0, 60.0];
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

/// Benchmark recording single values to the tracker
fn bench_record_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_value");

    for &sample_size in SAMPLE_SIZES {
        group.throughput(Throughput::Elements(sample_size as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential", sample_size),
            &sample_size,
            |b, &size| {
                b.iter(|| {
                    let mut tracker = TimeWeightedAverageTracker::new(10.0);
                    for i in 0..size {
                        tracker.record_value(black_box(i as f64));
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark computing time-weighted averages
fn bench_time_weighted_average(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_weighted_average");

    for &sample_size in SAMPLE_SIZES {
        group.throughput(Throughput::Elements(1)); // One calculation per iteration

        group.bench_with_input(
            BenchmarkId::new("computation", sample_size),
            &sample_size,
            |b, &size| {
                // Pre-populate tracker with samples
                let mut tracker = TimeWeightedAverageTracker::new(10.0);
                for i in 0..size {
                    tracker.record_value(i as f64);
                    if i % 100 == 0 {
                        // Add some time variance
                        thread::sleep(Duration::from_nanos(1));
                    }
                }

                b.iter(|| {
                    black_box(tracker.get_decayed_time_weighted_average());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark different time constants impact on performance
fn bench_time_constants(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_constants");

    const SAMPLE_SIZE: usize = 1000;

    for &time_constant in TIME_CONSTANTS {
        group.bench_with_input(
            BenchmarkId::new("record_and_compute", time_constant),
            &time_constant,
            |b, &tc| {
                b.iter(|| {
                    let mut tracker = TimeWeightedAverageTracker::new(tc);

                    // Record samples
                    for i in 0..SAMPLE_SIZE {
                        tracker.record_value(black_box(i as f64));
                    }

                    // Compute average
                    black_box(tracker.get_decayed_time_weighted_average());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark rate limiter decision making
fn bench_rate_limiter_decisions(c: &mut Criterion) {
    let mut group = c.benchmark_group("rate_limiter_decisions");

    let config = RateLimiterConfig::new(100.0, 10.0, 10.0, false);

    group.bench_function("should_reject_with_data", |b| {
        let rate_limiter = RateLimiter::new(Some(config.clone()));

        // Pre-populate with samples
        for i in 0..100 {
            rate_limiter.record_ttft("test-model", 50.0 + i as f64);
            rate_limiter.record_itl("test-model", 5.0 + (i as f64 / 10.0));
        }

        b.iter(|| {
            black_box(rate_limiter.should_reject(black_box("test-model")));
        });
    });

    group.bench_function("record_ttft", |b| {
        let rate_limiter = RateLimiter::new(Some(config.clone()));
        let mut counter = 0;

        b.iter(|| {
            rate_limiter.record_ttft(black_box("test-model"), black_box(counter as f64));
            counter += 1;
        });
    });

    group.bench_function("record_itl", |b| {
        let rate_limiter = RateLimiter::new(Some(config.clone()));
        let mut counter = 0;

        b.iter(|| {
            rate_limiter.record_itl(black_box("test-model"), black_box(counter as f64));
            counter += 1;
        });
    });

    group.finish();
}

/// Benchmark concurrent access patterns
fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");

    for &thread_count in THREAD_COUNTS {
        group.throughput(Throughput::Elements(thread_count as u64 * 100));

        group.bench_with_input(
            BenchmarkId::new("multi_thread_records", thread_count),
            &thread_count,
            |b, &num_threads| {
                b.iter(|| {
                    let config = RateLimiterConfig::new(1000.0, 10.0, 30.0, false);
                    let rate_limiter = Arc::new(RateLimiter::new(Some(config)));

                    let handles: Vec<_> = (0..num_threads)
                        .map(|thread_id| {
                            let limiter = rate_limiter.clone();
                            thread::spawn(move || {
                                for i in 0..100 {
                                    let value = (thread_id * 100 + i) as f64;
                                    limiter.record_ttft("test-model", value);
                                    limiter.record_itl("test-model", value / 10.0);

                                    // Some threads check rejection status
                                    if i % 10 == 0 {
                                        black_box(limiter.should_reject("test-model"));
                                    }
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    black_box(rate_limiter);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    group.bench_function("memory_bounded_growth", |b| {
        b.iter(|| {
            let mut tracker = TimeWeightedAverageTracker::new(10.0);

            // Add way more samples than max_samples to test memory bounds
            for i in 0..1000 {
                tracker.record_value(black_box(i as f64));

                // Occasionally compute average to trigger cleanup
                if i % 50 == 0 {
                    black_box(tracker.get_decayed_time_weighted_average());
                }
            }

            black_box(tracker);
        });
    });

    group.bench_function("per_model_isolation", |b| {
        let config = RateLimiterConfig::new(1000.0, 10.0, 30.0, true);

        b.iter(|| {
            let rate_limiter = RateLimiter::new(Some(config.clone()));

            // Simulate multiple models
            for model_id in 0..10 {
                let model_name = format!("model-{}", model_id);
                for i in 0..50 {
                    rate_limiter.record_ttft(&model_name, i as f64);
                    rate_limiter.record_itl(&model_name, (i as f64) / 10.0);
                }
                black_box(rate_limiter.should_reject(&model_name));
            }

            black_box(rate_limiter);
        });
    });

    group.finish();
}

/// Benchmark edge cases and stress scenarios
fn bench_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");

    group.bench_function("rapid_fire_records", |b| {
        b.iter(|| {
            let mut tracker = TimeWeightedAverageTracker::new(1.0);

            // Rapid fire recording without any delays
            for i in 0..5000 {
                tracker.record_value(black_box(i as f64));
            }

            black_box(tracker.get_decayed_time_weighted_average());
        });
    });

    group.bench_function("alternating_high_low_values", |b| {
        b.iter(|| {
            let mut tracker = TimeWeightedAverageTracker::new(5.0);

            // Alternating between very high and very low values
            for i in 0..500 {
                let value = if i % 2 == 0 { 1000000.0 } else { 0.001 };
                tracker.record_value(black_box(value));
            }

            black_box(tracker.get_decayed_time_weighted_average());
        });
    });

    group.bench_function("very_old_samples", |b| {
        b.iter(|| {
            let mut tracker = TimeWeightedAverageTracker::new(0.1); // Very short time constant

            // Add some samples
            for i in 0..100 {
                tracker.record_value(black_box(i as f64));
            }

            // Sleep to make them very old
            thread::sleep(Duration::from_millis(100));

            // Add fresh samples
            for i in 100..200 {
                tracker.record_value(black_box(i as f64));
            }

            black_box(tracker.get_decayed_time_weighted_average());
        });
    });

    group.finish();
}

/// Comprehensive benchmark comparing different configurations
fn bench_configuration_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("configuration_comparison");

    let configs = vec![
        (
            "aggressive",
            RateLimiterConfig::new(1000.0, 10.0, 1.0, false),
        ),
        (
            "balanced",
            RateLimiterConfig::new(1000.0, 10.0, 10.0, false),
        ),
        (
            "conservative",
            RateLimiterConfig::new(1000.0, 10.0, 60.0, false),
        ),
    ];

    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::new("full_workflow", name),
            &config,
            |b, config| {
                b.iter(|| {
                    let rate_limiter = RateLimiter::new(Some(config.clone()));

                    // Simulate realistic usage pattern
                    for i in 0..200 {
                        rate_limiter.record_ttft("model", black_box(50.0 + (i as f64)));
                        rate_limiter.record_itl("model", black_box(5.0 + (i as f64 / 10.0)));

                        if i % 20 == 0 {
                            black_box(rate_limiter.should_reject("model"));
                        }
                    }

                    black_box(rate_limiter);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_record_value,
    bench_time_weighted_average,
    bench_time_constants,
    bench_rate_limiter_decisions,
    bench_concurrent_access,
    bench_memory_patterns,
    bench_edge_cases,
    bench_configuration_comparison
);

criterion_main!(benches);
