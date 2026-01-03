// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::compute::{ComputeConfig, ComputePool};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

async fn measure_direct(n: u64) -> Duration {
    let start = Instant::now();
    let _ = compute_primes_sum(n);
    start.elapsed()
}

async fn measure_rayon(pool: &ComputePool, n: u64) -> Duration {
    let start = Instant::now();
    let _ = pool.execute(move || compute_primes_sum(n)).await.unwrap();
    start.elapsed()
}

async fn measure_spawn_blocking(n: u64) -> Duration {
    let start = Instant::now();
    let _ = tokio::task::spawn_blocking(move || compute_primes_sum(n))
        .await
        .unwrap();
    start.elapsed()
}

fn format_duration(d: Duration) -> String {
    if d.as_secs() > 0 {
        format!("{:.2}s", d.as_secs_f64())
    } else if d.as_millis() > 0 {
        format!("{:.2}ms", d.as_secs_f64() * 1000.0)
    } else if d.as_micros() > 0 {
        format!("{:.2}μs", d.as_secs_f64() * 1_000_000.0)
    } else {
        format!("{}ns", d.as_nanos())
    }
}

fn print_table_header() {
    println!("\n{:=<120}", "");
    println!(
        "{:>10} | {:>15} | {:>15} | {:>15} | {:>12} | {:>12} | {:>20}",
        "n", "Direct", "Rayon", "spawn_blocking", "Rayon Ratio", "Spawn Ratio", "Winner"
    );
    println!("{:-<120}", "");
}

fn print_row(
    n: u64,
    direct: Duration,
    rayon: Duration,
    spawn_blocking: Duration,
    highlight_crossover: bool,
) {
    let rayon_ratio = rayon.as_secs_f64() / direct.as_secs_f64();
    let spawn_ratio = spawn_blocking.as_secs_f64() / direct.as_secs_f64();

    let winner = if rayon_ratio < 1.0 && rayon_ratio < spawn_ratio {
        "Rayon ✓"
    } else if spawn_ratio < 1.0 && spawn_ratio < rayon_ratio {
        "spawn_blocking"
    } else {
        "Direct"
    };

    let row = format!(
        "{:>10} | {:>15} | {:>15} | {:>15} | {:>12.2}x | {:>12.2}x | {:>20}",
        n,
        format_duration(direct),
        format_duration(rayon),
        format_duration(spawn_blocking),
        rayon_ratio,
        spawn_ratio,
        winner
    );

    if highlight_crossover {
        println!(">>> {} <<<", row);
    } else {
        println!("{}", row);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔬 Compute Pool Overhead Demonstration");
    println!("=====================================\n");

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
    println!();

    // Warm up all execution paths
    println!("Warming up...");
    for _ in 0..5 {
        let _ = measure_direct(100).await;
        let _ = measure_rayon(&pool, 100).await;
        let _ = measure_spawn_blocking(100).await;
    }

    print_table_header();

    // Dynamic scanning with exponential growth
    let mut n = 10u64;
    let mut results = Vec::new();
    let mut found_crossover = false;
    let mut last_rayon_ratio = f64::MAX;

    while n <= 1_000_000 {
        // Measure each approach multiple times and take the minimum
        let mut direct_times = Vec::new();
        for _ in 0..3 {
            direct_times.push(measure_direct(n).await);
        }
        let direct = direct_times.into_iter().min().unwrap();

        let mut rayon_times = Vec::new();
        for _ in 0..3 {
            rayon_times.push(measure_rayon(&pool, n).await);
        }
        let rayon = rayon_times.into_iter().min().unwrap();

        let mut spawn_times = Vec::new();
        for _ in 0..3 {
            spawn_times.push(measure_spawn_blocking(n).await);
        }
        let spawn_blocking = spawn_times.into_iter().min().unwrap();

        let rayon_ratio = rayon.as_secs_f64() / direct.as_secs_f64();

        // Detect crossover point
        let is_crossover = !found_crossover && rayon_ratio < 1.0 && last_rayon_ratio >= 1.0;
        if is_crossover {
            found_crossover = true;
        }

        print_row(n, direct, rayon, spawn_blocking, is_crossover);
        results.push((n, direct, rayon, spawn_blocking));

        last_rayon_ratio = rayon_ratio;

        // Adaptive step size
        if n < 100 {
            n = (n as f64 * 2.0) as u64;
        } else if n < 10_000 {
            n = (n as f64 * 3.16) as u64; // ~10x every 2 steps
        } else {
            n *= 10;
        }
    }

    println!("{:=<120}", "");

    // Analysis
    println!("\n Analysis:");
    println!("============\n");

    if found_crossover {
        let crossover_point = results
            .iter()
            .find(|(_, d, r, _)| r.as_secs_f64() < d.as_secs_f64())
            .map(|(n, _, _, _)| *n);

        if let Some(n) = crossover_point {
            println!("✓ Rayon becomes beneficial at n ≈ {}", n);
            println!(
                "  Below n={}: Overhead dominates, direct execution is faster",
                n
            );
            println!(
                "  Above n={}: Compute dominates, Rayon offload is faster",
                n
            );
        }
    } else {
        println!("✗ No crossover found in tested range");
        println!("  Direct execution was always faster (overhead too high)");
    }

    // Find where spawn_blocking becomes beneficial
    let spawn_crossover = results
        .iter()
        .find(|(_, d, _, s)| s.as_secs_f64() < d.as_secs_f64())
        .map(|(n, _, _, _)| *n);

    if let Some(n) = spawn_crossover {
        println!("\n✓ spawn_blocking becomes beneficial at n ≈ {}", n);
    }

    // Show overhead at minimum
    if let Some((n, direct, rayon, spawn)) = results.first() {
        let rayon_overhead = rayon.as_secs_f64() - direct.as_secs_f64();
        let spawn_overhead = spawn.as_secs_f64() - direct.as_secs_f64();
        println!("\nOverhead at n={}:", n);
        println!(
            "  Rayon:          +{}",
            format_duration(Duration::from_secs_f64(rayon_overhead))
        );
        println!(
            "  spawn_blocking: +{}",
            format_duration(Duration::from_secs_f64(spawn_overhead))
        );
    }

    // Show benefit at maximum
    if let Some((n, direct, rayon, spawn)) = results.last() {
        let rayon_speedup = direct.as_secs_f64() / rayon.as_secs_f64();
        let spawn_speedup = direct.as_secs_f64() / spawn.as_secs_f64();
        println!("\nSpeedup at n={}:", n);
        println!("  Rayon:          {:.2}x faster", rayon_speedup);
        println!("  spawn_blocking: {:.2}x faster", spawn_speedup);
    }

    // Print pool metrics
    println!("\n Compute Pool Metrics:");
    println!("========================");
    println!("{}", pool.metrics());

    Ok(())
}
