// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example showing how to integrate ComputePool with tokenization workloads
//!
//! This demonstrates the pattern that could be used in lib/llm/src/preprocessor.rs
//! to leverage the compute pool for batch tokenization operations.

use anyhow::Result;
use dynamo_runtime::{Worker, compute::ComputePool};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Mock tokenizer for demonstration
struct MockTokenizer;

impl MockTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // Simulate tokenization work
        let mut tokens = Vec::new();
        for (i, word) in text.split_whitespace().enumerate() {
            // Simulate expensive computation
            let hash = word
                .bytes()
                .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            tokens.push(hash.wrapping_add(i as u32));
        }
        tokens
    }

    fn decode(&self, tokens: &[u32]) -> String {
        // Simulate detokenization
        tokens
            .iter()
            .map(|t| format!("token_{}", t % 1000))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Pattern 1: Direct replacement for par_iter in preprocessor
///
/// This shows how the existing code in lib/llm/src/preprocessor.rs:330
/// could be enhanced with explicit compute pool control
async fn tokenize_batch_with_pool(
    pool: &ComputePool,
    tokenizer: Arc<MockTokenizer>,
    texts: Vec<String>,
) -> Result<Vec<Vec<u32>>> {
    println!(
        "\n=== Tokenizing {} texts with compute pool ===",
        texts.len()
    );
    let start = Instant::now();

    // Option 1: Using scope for fine control
    let token_batches = pool
        .execute_scoped(move |scope| {
            let results = Arc::new(Mutex::new(vec![Vec::new(); texts.len()]));

            for (i, text) in texts.iter().enumerate() {
                let tokenizer = tokenizer.clone();
                let text = text.clone();
                let results = results.clone();
                scope.spawn(move |_| {
                    let tokens = tokenizer.encode(&text);
                    let mut r = results.lock().unwrap();
                    r[i] = tokens;
                });
            }

            Arc::try_unwrap(results).unwrap().into_inner().unwrap()
        })
        .await?;

    let total_tokens: usize = token_batches.iter().map(|v| v.len()).sum();
    println!(
        "Tokenized in {:?}, total tokens: {}",
        start.elapsed(),
        total_tokens
    );

    Ok(token_batches)
}

/// Pattern 2: Using rayon's par_iter within the compute pool
///
/// This maintains compatibility with existing code patterns
async fn tokenize_batch_par_iter(
    pool: &ComputePool,
    tokenizer: Arc<MockTokenizer>,
    texts: Vec<String>,
) -> Result<Vec<Vec<u32>>> {
    use rayon::prelude::*;

    println!("\n=== Tokenizing with par_iter in compute pool ===");
    let start = Instant::now();

    // This is how the existing preprocessor code could work
    let token_batches: Vec<Vec<u32>> = pool
        .install(move || {
            texts
                .par_iter()
                .map(|text| tokenizer.encode(text))
                .collect()
        })
        .await?;

    let total_tokens: usize = token_batches.iter().map(|v| v.len()).sum();
    println!(
        "Tokenized in {:?}, total tokens: {}",
        start.elapsed(),
        total_tokens
    );

    Ok(token_batches)
}

/// Pattern 3: Mixed async/sync processing
///
/// This shows how to handle a stream of requests where each request
/// contains a batch that needs parallel processing
async fn process_request_stream(pool: &ComputePool, tokenizer: Arc<MockTokenizer>) -> Result<()> {
    println!("\n=== Processing request stream ===");

    // Simulate incoming requests
    let requests = vec![
        vec![
            "Request 1 text 1".to_string(),
            "Request 1 text 2".to_string(),
        ],
        vec![
            "Request 2 text 1".to_string(),
            "Request 2 text 2".to_string(),
            "Request 2 text 3".to_string(),
        ],
        vec!["Request 3 text 1".to_string()],
    ];

    for (i, batch) in requests.into_iter().enumerate() {
        println!("Processing request {}", i + 1);

        // Each request gets processed in parallel
        let tokens = tokenize_batch_with_pool(pool, tokenizer.clone(), batch).await?;

        // Simulate async I/O between requests
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        println!(
            "Request {} completed with {} token batches",
            i + 1,
            tokens.len()
        );
    }

    Ok(())
}

/// Pattern 4: Encode/Decode pipeline
///
/// Shows how to chain multiple compute operations
async fn encode_decode_pipeline(
    pool: &ComputePool,
    tokenizer: Arc<MockTokenizer>,
    texts: Vec<String>,
) -> Result<Vec<String>> {
    println!("\n=== Encode/Decode Pipeline ===");
    let start = Instant::now();

    // Step 1: Encode all texts in parallel
    let tokenizer_clone = tokenizer.clone();
    let encoded = pool
        .execute_scoped(move |scope| {
            let results = Arc::new(Mutex::new(vec![Vec::new(); texts.len()]));

            for (i, text) in texts.iter().enumerate() {
                let tokenizer = tokenizer_clone.clone();
                let text = text.clone();
                let results = results.clone();
                scope.spawn(move |_| {
                    let tokens = tokenizer.encode(&text);
                    let mut r = results.lock().unwrap();
                    r[i] = tokens;
                });
            }

            Arc::try_unwrap(results).unwrap().into_inner().unwrap()
        })
        .await?;

    println!("Encoding complete in {:?}", start.elapsed());

    // Step 2: Decode all token sequences in parallel
    let decoded_start = Instant::now();
    let decoded = pool
        .execute_scoped(move |scope| {
            let results = Arc::new(Mutex::new(vec![String::new(); encoded.len()]));

            for (i, tokens) in encoded.iter().enumerate() {
                let tokenizer = tokenizer.clone();
                let tokens = tokens.clone();
                let results = results.clone();
                scope.spawn(move |_| {
                    let text = tokenizer.decode(&tokens);
                    let mut r = results.lock().unwrap();
                    r[i] = text;
                });
            }

            Arc::try_unwrap(results).unwrap().into_inner().unwrap()
        })
        .await?;

    println!("Decoding complete in {:?}", decoded_start.elapsed());
    println!("Total pipeline time: {:?}", start.elapsed());

    Ok(decoded)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Set compute pool configuration via environment
    unsafe {
        std::env::set_var("DYN_COMPUTE_THREADS", "4");
    }

    // Create worker and runtime
    let worker = Worker::from_settings()?;
    let runtime = worker.runtime().clone();

    // Get compute pool
    let pool = runtime
        .compute_pool()
        .ok_or_else(|| anyhow::anyhow!("Compute pool not initialized"))?
        .clone();

    println!(
        "Compute pool initialized with {} threads",
        pool.num_threads()
    );

    // Create mock tokenizer
    let tokenizer = Arc::new(MockTokenizer);

    // Generate test data
    let texts: Vec<String> = (0..50)
        .map(|i| {
            format!(
                "This is sample text number {} with some words to tokenize. \
             The quick brown fox jumps over the lazy dog.",
                i
            )
        })
        .collect();

    // Run examples
    let _ = tokenize_batch_with_pool(&pool, tokenizer.clone(), texts.clone()).await?;
    let _ = tokenize_batch_par_iter(&pool, tokenizer.clone(), texts.clone()).await?;
    process_request_stream(&pool, tokenizer.clone()).await?;
    let decoded = encode_decode_pipeline(&pool, tokenizer.clone(), texts.clone()).await?;

    println!("\n=== Results ===");
    println!("Processed {} texts", texts.len());
    println!("First decoded text: {}", &decoded[0]);

    // Print metrics
    let metrics = pool.metrics();
    println!("\n=== Compute Pool Metrics ===");
    println!("{}", metrics);

    Ok(())
}
