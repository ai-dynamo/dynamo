// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::compute::{ComputeConfig, ComputePool};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::time::sleep;

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

/// Simulated async task that does both I/O and compute
async fn async_task_inline(id: usize, n: u64, io_delay: Duration) -> (usize, Duration) {
    let start = Instant::now();

    // Simulate async I/O operation
    sleep(io_delay).await;

    // CPU-intensive work (blocks the async runtime!)
    let _result = compute_primes_sum(n);

    // More async I/O
    sleep(io_delay).await;

    (id, start.elapsed())
}

/// Async task that offloads compute to Rayon pool
async fn async_task_rayon(
    id: usize,
    n: u64,
    io_delay: Duration,
    pool: Arc<ComputePool>,
) -> (usize, Duration) {
    let start = Instant::now();

    // Simulate async I/O operation
    sleep(io_delay).await;

    // CPU-intensive work (offloaded, doesn't block runtime)
    let _result = pool.execute(move || compute_primes_sum(n)).await.unwrap();

    // More async I/O
    sleep(io_delay).await;

    (id, start.elapsed())
}

/// Async task using spawn_blocking
async fn async_task_spawn_blocking(id: usize, n: u64, io_delay: Duration) -> (usize, Duration) {
    let start = Instant::now();

    // Simulate async I/O operation
    sleep(io_delay).await;

    // CPU-intensive work (offloaded to blocking pool)
    let _result = tokio::task::spawn_blocking(move || compute_primes_sum(n))
        .await
        .unwrap();

    // More async I/O
    sleep(io_delay).await;

    (id, start.elapsed())
}

async fn run_throughput_test(
    name: &str,
    num_tasks: usize,
    n: u64,
    io_delay: Duration,
    pool: Option<Arc<ComputePool>>,
    mode: &str,
) -> (Duration, Vec<Duration>) {
    println!("\n Running: {} (n={}, tasks={})", name, n, num_tasks);

    let completed = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let tasks: Vec<_> = (0..num_tasks)
        .map(|id| {
            let pool = pool.clone();
            let completed = completed.clone();
            let mode = mode.to_string();

            tokio::spawn(async move {
                let result = match mode.as_str() {
                    "inline" => async_task_inline(id, n, io_delay).await,
                    "rayon" => async_task_rayon(id, n, io_delay, pool.unwrap()).await,
                    "spawn_blocking" => async_task_spawn_blocking(id, n, io_delay).await,
                    _ => panic!("Unknown mode"),
                };

                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_multiple_of(10) {
                    print!(".");
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();
                }

                result
            })
        })
        .collect();

    let mut latencies = Vec::new();
    for task in tasks {
        let (_id, latency) = task.await.unwrap();
        latencies.push(latency);
    }

    let total_time = start.elapsed();
    println!(" Done in {:.2}s", total_time.as_secs_f64());

    (total_time, latencies)
}

fn calculate_percentiles(latencies: &mut [Duration]) -> (Duration, Duration, Duration) {
    latencies.sort();
    let len = latencies.len();
    let p50 = latencies[len / 2];
    let p95 = latencies[len * 95 / 100];
    let p99 = latencies[len * 99 / 100];
    (p50, p95, p99)
}

fn print_results(_name: &str, total: Duration, latencies: &mut [Duration]) {
    let (p50, p95, p99) = calculate_percentiles(latencies);
    let throughput = latencies.len() as f64 / total.as_secs_f64();

    println!("  Total time:     {:.2}s", total.as_secs_f64());
    println!("  Throughput:     {:.1} tasks/s", throughput);
    println!("  Latency p50:    {:.2}ms", p50.as_secs_f64() * 1000.0);
    println!("  Latency p95:    {:.2}ms", p95.as_secs_f64() * 1000.0);
    println!("  Latency p99:    {:.2}ms", p99.as_secs_f64() * 1000.0);
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Async Throughput Demonstration");
    println!("==================================\n");
    println!("This demo shows how compute-intensive work affects async task throughput.\n");

    // Create compute pool directly
    let compute_config = ComputeConfig {
        num_threads: Some(4),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "demo".to_string(),
        pin_threads: false,
    };
    let pool = Arc::new(ComputePool::new(compute_config)?);

    println!("Configuration:");
    println!("  Rayon threads: {}", pool.num_threads());

    // Test parameters
    let num_tasks = 100;
    let io_delay = Duration::from_millis(10);

    println!("\nTest: {} concurrent async tasks", num_tasks);
    println!("Each task: 10ms I/O → compute → 10ms I/O");
    println!("Expected minimum time: ~20ms (if no blocking)");

    // Test with different compute loads
    for n in [10, 1_000, 100_000] {
        println!("\n{:=<60}", "");
        println!("Compute load: n={} (prime sum)", n);
        println!("{:=<60}", "");

        // Measure compute time alone
        let compute_start = Instant::now();
        let _ = compute_primes_sum(n);
        let compute_time = compute_start.elapsed();
        println!(
            "Pure compute time: {:.2}ms",
            compute_time.as_secs_f64() * 1000.0
        );

        // Test 1: Inline execution (blocks async runtime)
        let (total1, mut latencies1) = run_throughput_test(
            "Inline (blocks runtime)",
            num_tasks,
            n,
            io_delay,
            None,
            "inline",
        )
        .await;
        print_results("Inline", total1, &mut latencies1);

        // Test 2: Rayon offload
        let (total2, mut latencies2) = run_throughput_test(
            "Rayon offload",
            num_tasks,
            n,
            io_delay,
            Some(pool.clone()),
            "rayon",
        )
        .await;
        print_results("Rayon", total2, &mut latencies2);

        // Test 3: spawn_blocking
        let (total3, mut latencies3) = run_throughput_test(
            "spawn_blocking",
            num_tasks,
            n,
            io_delay,
            None,
            "spawn_blocking",
        )
        .await;
        print_results("spawn_blocking", total3, &mut latencies3);

        // Analysis
        println!("\n Impact Analysis:");
        let speedup_rayon = total1.as_secs_f64() / total2.as_secs_f64();
        let speedup_spawn = total1.as_secs_f64() / total3.as_secs_f64();

        println!(
            "  Rayon vs Inline:          {:.2}x throughput",
            speedup_rayon
        );
        println!(
            "  spawn_blocking vs Inline: {:.2}x throughput",
            speedup_spawn
        );

        if compute_time.as_millis() > 1 {
            let blocking_factor = compute_time.as_secs_f64() / io_delay.as_secs_f64();
            println!(
                "\n  Compute time ({:.1}ms) is {:.1}x the I/O time",
                compute_time.as_secs_f64() * 1000.0,
                blocking_factor
            );
            println!("     This severely impacts async concurrency when run inline!");
        }
    }

    // Show pool metrics
    println!("\n Compute Pool Metrics:");
    println!("========================");
    println!("{}", pool.metrics());

    println!("\n Conclusion:");
    println!("==============");
    println!("• Small compute (n=10): Overhead may not justify offloading");
    println!("• Medium compute (n=1000): Offloading preserves async throughput");
    println!("• Large compute (n=100000): Offloading is essential for responsiveness");
    println!("\nKey insight: Even small amounts of blocking compute can destroy");
    println!("async throughput when you have many concurrent tasks!");

    Ok(())
}
