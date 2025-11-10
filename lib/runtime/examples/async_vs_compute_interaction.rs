// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark demonstrating async I/O vs compute workload interaction
//!
//! This example measures how different types of compute workloads interfere with
//! async I/O latency by comparing actual elapsed time vs expected sleep time.
//!
//! Key measurements:
//! - Baseline async overhead with no compute load
//! - Interference from small (<100μs), medium (~500μs), and large (~2-5ms) compute tasks
//! - Comparison between all-async (4 Tokio threads) vs hybrid (2 Tokio + 2 Rayon)
//! - Impact of offloading compute work to dedicated Rayon threads
//!
//! The benchmark spawns many lightweight async tasks doing timed sleeps, then runs
//! a fixed compute workload while measuring how much the compute work delays the
//! async tasks from being revisited after their sleeps complete.
//!
//! Two configurations are tested with EXACTLY 4 total threads:
//! 1. All-Async: 4 Tokio threads (compute runs inline, blocking async work)
//! 2. Hybrid: 2 Tokio threads + 2 Rayon threads (compute offloaded, async stays responsive)

use anyhow::Result;
use dynamo_runtime::{
    Runtime,
    compute::{ComputeConfig, ComputePool},
    compute_large, compute_medium, compute_small,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

/// Sleep latency measurement
#[derive(Debug, Clone)]
struct SleepMeasurement {
    _expected_ms: f64,
    _actual_ms: f64,
    overhead_ms: f64, // actual - expected
}

/// Statistics for latency measurements
#[derive(Debug, Clone)]
struct LatencyStats {
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
    count: usize,
}

/// Test results for a single configuration
#[derive(Debug)]
struct TestResults {
    baseline_overhead: LatencyStats,
    compute_overhead: Option<LatencyStats>,
    compute_duration: Option<Duration>,
    _total_sleep_measurements: usize,
}

/// Type of workload to run
#[derive(Debug, Clone, Copy, PartialEq)]
enum WorkloadType {
    None,   // No compute (baseline)
    Small,  // 100% small tasks
    Medium, // 100% medium tasks
    Large,  // 100% large tasks
    Mixed,  // 33/33/33 mix
}

/// Individual task type
#[derive(Debug, Clone, Copy)]
enum TaskType {
    Small,
    Medium,
    Large,
}

/// Compute-intensive function: sum of all primes up to n
fn compute_primes_sum(n: u64) -> u64 {
    let mut sum = 0u64;
    for candidate in 2..=n {
        if is_prime(candidate) {
            sum += candidate;
        }
    }
    sum
}

fn is_prime(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as u64;
    for i in (5..=sqrt_n).step_by(6) {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
    }
    true
}

// Global tuned values (set during calibration)
static mut SMALL_N: u64 = 1_500;
static mut MEDIUM_N: u64 = 20_000;
static mut LARGE_N: u64 = 120_000;

/// Small compute task (~10μs)
fn small_compute() -> u64 {
    unsafe { compute_primes_sum(SMALL_N) }
}

/// Medium compute task (~500μs)
fn medium_compute() -> u64 {
    unsafe { compute_primes_sum(MEDIUM_N) }
}

/// Large compute task (~2-5ms)
fn large_compute() -> u64 {
    unsafe { compute_primes_sum(LARGE_N) }
}

/// Dynamically tune a compute function to hit target time
fn tune_compute_n(target_us: f64, initial_n: u64, name: &str) -> u64 {
    let mut n = initial_n;
    let mut best_n = n;
    let mut best_diff = f64::MAX;

    // Binary search for the right value
    let mut low = 10u64;
    let mut high = 1_000_000u64;

    for _ in 0..20 {
        // Max 20 iterations
        n = (low + high) / 2;

        // Measure this n value (average of 3 runs for stability)
        let mut total_time = Duration::ZERO;
        for _ in 0..3 {
            let start = Instant::now();
            let _ = compute_primes_sum(n);
            total_time += start.elapsed();
        }
        let elapsed_us = total_time.as_secs_f64() * 1_000_000.0 / 3.0;

        let diff = (elapsed_us - target_us).abs();
        if diff < best_diff {
            best_diff = diff;
            best_n = n;
        }

        // Check if we're close enough (within 20%)
        if diff / target_us < 0.20 {
            println!(
                "  ✓ {} tuned to n={} ({:.1}μs, target {:.0}μs)",
                name, n, elapsed_us, target_us
            );
            return n;
        }

        // Adjust search range
        if elapsed_us < target_us {
            low = n + 1;
        } else {
            high = n - 1;
        }

        if low > high {
            break;
        }
    }

    // Use best found value
    let start = Instant::now();
    let _ = compute_primes_sum(best_n);
    let final_time = start.elapsed().as_secs_f64() * 1_000_000.0;
    println!(
        "  ✓ {} tuned to n={} ({:.1}μs, target {:.0}μs)",
        name, best_n, final_time, target_us
    );
    best_n
}

/// Calibrate compute functions to measure actual execution times
fn calibrate_compute_functions() {
    println!("\n Dynamically calibrating compute functions for this machine...");
    println!("{:-<60}", "");

    // Tune each function
    unsafe {
        SMALL_N = tune_compute_n(10.0, SMALL_N, "Small");
        MEDIUM_N = tune_compute_n(500.0, MEDIUM_N, "Medium");
        LARGE_N = tune_compute_n(3000.0, LARGE_N, "Large"); // Target 3ms (middle of 2-5ms range)
    }

    println!();
    println!("For future runs on this machine, you can use:");
    println!("     SMALL_N  = {}", unsafe { SMALL_N });
    println!("     MEDIUM_N = {}", unsafe { MEDIUM_N });
    println!("     LARGE_N  = {}", unsafe { LARGE_N });
}

/// Worker that repeatedly sleeps and measures latency
async fn sleep_worker(
    sleep_duration: Duration,
    results: Arc<Mutex<Vec<SleepMeasurement>>>,
    cancel: CancellationToken,
) {
    while !cancel.is_cancelled() {
        let start = Instant::now();
        tokio::select! {
            _ = sleep(sleep_duration) => {
                let elapsed = start.elapsed();
                let measurement = SleepMeasurement {
                    _expected_ms: sleep_duration.as_secs_f64() * 1000.0,
                    _actual_ms: elapsed.as_secs_f64() * 1000.0,
                    overhead_ms: (elapsed.as_secs_f64() - sleep_duration.as_secs_f64()) * 1000.0,
                };
                results.lock().unwrap().push(measurement);
            }
            _ = cancel.cancelled() => break,
        }
    }
}

/// Execute a single compute task based on type
async fn execute_compute_task(task_type: TaskType, pool: Option<Arc<ComputePool>>) -> Result<u64> {
    match task_type {
        TaskType::Small => {
            // Small tasks always run inline
            Ok(compute_small!(small_compute()))
        }
        TaskType::Medium => {
            // Medium tasks: offload if pool available, else run inline (blocking)
            if let Some(pool) = pool.clone() {
                pool.execute(medium_compute).await
            } else {
                // No pool - run inline on Tokio thread (will block!)
                Ok(medium_compute())
            }
        }
        TaskType::Large => {
            // Large tasks: offload if pool available, else run inline (severely blocking)
            if let Some(pool) = pool {
                pool.execute(large_compute).await
            } else {
                // No pool - run inline on Tokio thread (will severely block!)
                Ok(large_compute())
            }
        }
    }
}

/// Execute a batch of compute tasks with concurrency limiting
async fn execute_compute_batch(
    workload_type: WorkloadType,
    num_tasks: usize,
    concurrency_limit: Arc<Semaphore>,
    pool: Option<Arc<ComputePool>>,
) -> Duration {
    if workload_type == WorkloadType::None {
        return Duration::from_secs(0);
    }

    let start = Instant::now();
    let mut handles = Vec::new();

    for i in 0..num_tasks {
        let permit = concurrency_limit.clone().acquire_owned().await.unwrap();
        let pool = pool.clone();

        let task_type = match workload_type {
            WorkloadType::Small => TaskType::Small,
            WorkloadType::Medium => TaskType::Medium,
            WorkloadType::Large => TaskType::Large,
            WorkloadType::Mixed => {
                // Round-robin: 33% small, 33% medium, 33% large
                match i % 3 {
                    0 => TaskType::Small,
                    1 => TaskType::Medium,
                    _ => TaskType::Large,
                }
            }
            WorkloadType::None => unreachable!(),
        };

        let handle = tokio::spawn(async move {
            let _permit = permit; // Hold permit until task completes
            execute_compute_task(task_type, pool).await
        });
        handles.push(handle);
    }

    // Wait for all compute tasks
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    start.elapsed()
}

/// Calculate statistics from measurements
fn calculate_stats(measurements: &[SleepMeasurement]) -> LatencyStats {
    if measurements.is_empty() {
        return LatencyStats {
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            max: 0.0,
            mean: 0.0,
            count: 0,
        };
    }

    let mut overheads: Vec<f64> = measurements.iter().map(|m| m.overhead_ms).collect();
    overheads.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = overheads.len();
    LatencyStats {
        p50: overheads[len / 2],
        p95: overheads[len * 95 / 100],
        p99: overheads[len * 99 / 100],
        max: *overheads.last().unwrap(),
        mean: overheads.iter().sum::<f64>() / len as f64,
        count: len,
    }
}

/// Run a single interference test
async fn run_interference_test(
    _runtime: Arc<Runtime>,
    workload_type: WorkloadType,
    pool: Option<Arc<ComputePool>>,
) -> TestResults {
    // Configuration
    const NUM_SLEEP_TASKS: usize = 100;
    const SLEEP_DURATION_MS: u64 = 1;
    const NUM_COMPUTE_TASKS: usize = 2000; // Increased for longer workload
    const CONCURRENCY_LIMIT: usize = 8; // Allow more parallel work
    const BASELINE_DURATION_SECS: u64 = 1;

    // 1. Start async load (100 tasks doing 1ms sleeps)
    let results = Arc::new(Mutex::new(Vec::new()));
    let cancel = CancellationToken::new();
    let mut handles = Vec::new();

    for _ in 0..NUM_SLEEP_TASKS {
        let r = results.clone();
        let c = cancel.clone();
        handles.push(tokio::spawn(sleep_worker(
            Duration::from_millis(SLEEP_DURATION_MS),
            r,
            c,
        )));
    }

    // 2. Collect baseline measurements
    sleep(Duration::from_secs(BASELINE_DURATION_SECS)).await;
    let baseline_count = results.lock().unwrap().len();

    // 3. Run compute workload (if not baseline)
    let compute_duration = if workload_type != WorkloadType::None {
        let semaphore = Arc::new(Semaphore::new(CONCURRENCY_LIMIT));
        Some(execute_compute_batch(workload_type, NUM_COMPUTE_TASKS, semaphore, pool).await)
    } else {
        // For baseline, just wait another second
        sleep(Duration::from_secs(1)).await;
        None
    };

    // 4. Stop async load
    cancel.cancel();
    for handle in handles {
        handle.await.unwrap();
    }

    // 5. Analyze results
    let all_measurements = results.lock().unwrap().clone();
    let baseline = &all_measurements[..baseline_count.min(all_measurements.len())];
    let during_compute = if baseline_count < all_measurements.len() {
        Some(&all_measurements[baseline_count..])
    } else {
        None
    };

    TestResults {
        baseline_overhead: calculate_stats(baseline),
        compute_overhead: during_compute.map(calculate_stats),
        compute_duration,
        _total_sleep_measurements: all_measurements.len(),
    }
}

/// Run all test workloads for a given configuration
async fn run_all_tests(pool: Option<Arc<ComputePool>>) -> Result<()> {
    let workload_types = vec![
        ("Baseline (no compute)", WorkloadType::None),
        ("100% Small (~10μs each)", WorkloadType::Small),
        ("100% Medium (~500μs each)", WorkloadType::Medium),
        ("100% Large (~2-5ms each)", WorkloadType::Large),
        ("Mixed 33/33/33", WorkloadType::Mixed),
    ];

    // Create dummy runtime for the test functions
    let runtime = Arc::new(Runtime::from_current()?);

    for (name, workload) in &workload_types {
        println!("\n Workload: {}", name);
        println!("{:-<50}", "");

        let results = run_interference_test(runtime.clone(), *workload, pool.clone()).await;

        // Always show baseline overhead
        println!("  Baseline async overhead (first {}s):", 1);
        println!(
            "    Mean: {:.3}ms, P50: {:.3}ms, P95: {:.3}ms, P99: {:.3}ms",
            results.baseline_overhead.mean,
            results.baseline_overhead.p50,
            results.baseline_overhead.p95,
            results.baseline_overhead.p99
        );
        println!("    Measurements: {}", results.baseline_overhead.count);

        // Show compute interference if applicable
        if let Some(compute_overhead) = results.compute_overhead {
            println!("\n  During compute workload:");
            println!(
                "    Mean: {:.3}ms, P50: {:.3}ms, P95: {:.3}ms, P99: {:.3}ms",
                compute_overhead.mean,
                compute_overhead.p50,
                compute_overhead.p95,
                compute_overhead.p99
            );
            println!(
                "    Max: {:.3}ms, Measurements: {}",
                compute_overhead.max, compute_overhead.count
            );

            // Calculate interference factor
            if results.baseline_overhead.mean > 0.0 {
                let interference_factor = compute_overhead.mean / results.baseline_overhead.mean;
                println!(
                    "\n   Interference factor: {:.1}x slower",
                    interference_factor
                );

                // Provide interpretation
                let impact = if interference_factor < 2.0 {
                    "Minimal - async remains responsive"
                } else if interference_factor < 10.0 {
                    "Moderate - noticeable async delays"
                } else {
                    "SEVERE - async tasks are heavily blocked!"
                };
                println!("     Impact: {}", impact);
            }
        }

        // Show compute duration
        if let Some(duration) = results.compute_duration {
            println!(
                "\n  Compute workload completed in: {:.2}s",
                duration.as_secs_f64()
            );
            println!(
                "  Throughput: {:.0} tasks/sec",
                1000.0 / duration.as_secs_f64()
            );
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    println!(" Async vs Compute Interaction Benchmark");
    println!("==========================================");
    println!();
    println!("This benchmark measures how compute workloads interfere with async I/O latency.");
    println!("We test with EXACTLY 4 total threads in two configurations:");
    println!("  1. All-Async: 4 Tokio threads (compute blocks async work)");
    println!("  2. Hybrid: 2 Tokio + 2 Rayon threads (compute offloaded)");
    println!("  3. Bonus: Thread-local macro demonstration");
    println!();
    println!("Lower overhead numbers mean better async responsiveness.");

    // Calibrate compute functions
    calibrate_compute_functions();

    // Test 1: All Async (4 Tokio threads, no Rayon)
    println!("\n{:=<70}", "");
    println!("Configuration 1: All-Async (4 Tokio threads, no Rayon)");
    println!("{:=<70}", "");
    println!("  Compute tasks run INLINE on Tokio threads, blocking async work!");

    let all_async_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("tokio-worker")
        .enable_all()
        .build()?;

    all_async_rt.block_on(async {
        // No compute pool for all-async mode
        // All compute work will run inline on Tokio threads
        run_all_tests(None).await
    })?;

    // Test 2: Hybrid (2 Tokio + 2 Rayon)
    println!("\n{:=<70}", "");
    println!("Configuration 2: Hybrid (2 Tokio + 2 Rayon threads)");
    println!("{:=<70}", "");
    println!(" Compute tasks offloaded to Rayon, keeping async threads free!");

    let hybrid_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("tokio-worker")
        .enable_all()
        .build()?;

    // Create Rayon pool with 2 threads
    let compute_pool = Arc::new(ComputePool::new(ComputeConfig {
        num_threads: Some(2),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "rayon".to_string(),
        pin_threads: false,
    })?);

    hybrid_rt.block_on(async { run_all_tests(Some(compute_pool)).await })?;

    // Summary
    println!("\n{:=<70}", "");
    println!(" Key Takeaway");
    println!("{:=<70}", "");
    println!();
    println!("The benchmark demonstrates that offloading compute work to dedicated");
    println!("threads becomes increasingly important as task duration increases.");
    println!("Look at the interference factors above to see the actual impact.");

    // Test 3: Demonstrate thread-local macros
    println!("\n{:=<70}", "");
    println!("Configuration 3: Thread-Local Macro Demonstration");
    println!("{:=<70}", "");
    println!("Testing thread-local compute context initialization...");

    // Create a runtime with thread-local setup
    let macro_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("macro-demo")
        .enable_all()
        .build()?;

    // Create compute pool for macros
    let macro_pool = Arc::new(ComputePool::new(ComputeConfig {
        num_threads: Some(2),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "macro-compute".to_string(),
        pin_threads: false,
    })?);

    macro_rt.block_on(async {
        // We can't directly create a Runtime with private fields, so we'll
        // initialize thread-local context manually using a barrier approach

        // Set up semaphore permits
        let permits = Arc::new(tokio::sync::Semaphore::new(1)); // 2 workers - 1

        // Detect number of worker threads
        use parking_lot::Mutex;
        use std::collections::HashSet;

        let thread_ids = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();

        // Probe to find worker thread count
        for _ in 0..50 {
            let ids = Arc::clone(&thread_ids);
            let handle = tokio::task::spawn_blocking(move || {
                let thread_id = std::thread::current().id();
                ids.lock().insert(thread_id);
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.await;
        }

        let num_workers = thread_ids.lock().len();
        println!("  Detected {} worker threads", num_workers);

        // Now initialize thread-local on all workers using a barrier
        let barrier = Arc::new(std::sync::Barrier::new(num_workers));
        let mut init_handles = Vec::new();

        for i in 0..num_workers {
            let barrier_clone = Arc::clone(&barrier);
            let pool_clone = Arc::clone(&macro_pool);
            let permits_clone = Arc::clone(&permits);

            let handle = tokio::task::spawn_blocking(move || {
                // Wait at barrier
                barrier_clone.wait();

                // Initialize thread-local
                dynamo_runtime::compute::thread_local::initialize_context(
                    pool_clone,
                    permits_clone,
                );
                println!("  Initialized thread-local on worker {}", i);
            });
            init_handles.push(handle);
        }

        for handle in init_handles {
            handle.await?;
        }

        // Test if macros work
        println!("\n Testing thread-local macros:");

        if dynamo_runtime::compute::thread_local::has_compute_context() {
            println!("  Thread-local context is available!");

            // Test compute_small! macro
            println!("\n  Testing compute_small! (inline):");
            let start = std::time::Instant::now();
            let result = compute_small!(small_compute());
            println!("    Result: {}, Time: {:?}", result, start.elapsed());

            // Test compute_medium! macro (would use thread-local context)
            println!("\n  Testing compute_medium! (block_in_place or offload):");
            let start = std::time::Instant::now();
            let result = compute_medium!(medium_compute());
            println!("    Result: {}, Time: {:?}", result, start.elapsed());

            // Test compute_large! macro (would use thread-local context)
            println!("\n  Testing compute_large! (always offload):");
            let start = std::time::Instant::now();
            let result = compute_large!(large_compute());
            println!("    Result: {}, Time: {:?}", result, start.elapsed());

            println!("\n  All macros work with thread-local context!");
        } else {
            println!("  Thread-local context NOT available - macros would fail");
        }

        Ok::<_, anyhow::Error>(())
    })?;

    println!();
    println!(" Benchmark complete!");

    Ok(())
}
