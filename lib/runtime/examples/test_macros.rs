// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::compute::{ComputeConfig, ComputePool};
use dynamo_runtime::{compute_large, compute_medium, compute_small};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing compute macros...\n");

    // Create compute pool
    let compute_config = ComputeConfig {
        num_threads: Some(4),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "test".to_string(),
        pin_threads: false,
    };
    let pool = Arc::new(ComputePool::new(compute_config)?);

    // Test small macro (direct execution)
    println!("Testing compute_small!...");
    let result = compute_small!(2 + 2);
    println!("  Result: {}", result);

    // Test medium macro (block_in_place with fallback)
    println!("\nTesting compute_medium!...");
    let result = compute_medium!(pool, {
        let mut sum = 0u64;
        for i in 0..1000 {
            sum += i;
        }
        sum
    });
    println!("  Result: {}", result);

    // Test large macro (always Rayon)
    println!("\nTesting compute_large!...");
    let result = compute_large!(pool, {
        let mut sum = 0u64;
        for i in 0..1_000_000 {
            sum += i;
        }
        sum
    });
    println!("  Result: {}", result);

    println!("\n All macros working!");
    Ok(())
}
