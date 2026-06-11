// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dataset-driven benchmarks
//!
//! Downloads a real dataset (LongBench-v2) from HuggingFace Hub and benchmarks
//! per-sample encode throughput with correctness verification.
//!
//! This benchmark is opt-in: set RUN_BENCH=1 to run it. Without that variable
//! it exits immediately so that `cargo test --all-targets` in CI is not affected.
//!
//! Run:
//!   RUN_BENCH=1 cargo bench --bench tokenizer_dataset
//!
//! Override tokenizer (default: Qwen/Qwen3-0.6B):
//!   RUN_BENCH=1 TOKENIZER_PATH=deepseek-ai/DeepSeek-V3 cargo bench --bench tokenizer_dataset
//!
//! Override dataset and sample count (default: 503):
//!   RUN_BENCH=1 DATASET=RyokoAI/ShareGPT52K MAX_SAMPLES=50 cargo bench --bench tokenizer_dataset
//!
//! Batch benchmark (default: sequential):
//!   RUN_BENCH=1 BATCH_SIZE=64 cargo bench --bench tokenizer_dataset

use std::path::Path;
use std::time::{Duration, Instant};

use dynamo_llm::tokenizers::{FastTokenizer, HuggingFaceTokenizer, traits::Encoder};

/// Default HuggingFace model for the tokenizer.
const DEFAULT_HF_MODEL: &str = "Qwen/Qwen3-0.6B";

/// Default dataset on HuggingFace Hub.
const DEFAULT_DATASET: &str = "zai-org/LongBench-v2";

/// Default number of samples when MAX_SAMPLES is not set.
const DEFAULT_MAX_SAMPLES: usize = 503;

/// Resolve tokenizer path: local file, HF model name, or default.
fn resolve_tokenizer_path() -> String {
    let input = std::env::var("TOKENIZER_PATH").ok();

    if let Some(ref p) = input
        && Path::new(p).is_file()
    {
        eprintln!("[setup] Using local tokenizer: {p}");
        return p.clone();
    }

    let model_name = input.as_deref().unwrap_or(DEFAULT_HF_MODEL);
    eprintln!("[setup] Downloading tokenizer for {model_name}...");

    let cache = hf_hub::Cache::default();
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .expect("Failed to create HuggingFace API client");

    let repo = api.model(model_name.to_string());
    let path = repo
        .get("tokenizer.json")
        .expect("Failed to download tokenizer.json");

    let path_str = path.display().to_string();
    eprintln!("[setup] Tokenizer: {path_str}");
    path_str
}

/// Return the JSON filename for a known HuggingFace Hub dataset.
fn dataset_json_file(dataset: &str) -> &'static str {
    match dataset {
        "RyokoAI/ShareGPT52K" => "sg_90k_part1.json",
        "zai-org/LongBench-v2" => "data.json",
        _ => panic!(
            "Unknown dataset: {dataset}. Supported: zai-org/LongBench-v2, RyokoAI/ShareGPT52K"
        ),
    }
}

/// Extract a text sample from a single JSON item.
fn extract_text(dataset: &str, item: &serde_json::Value) -> Option<String> {
    match dataset {
        "RyokoAI/ShareGPT52K" => {
            let messages = item.get("conversations")?.as_array()?;
            let parts: Vec<String> = messages
                .iter()
                .filter_map(|msg| {
                    let role = msg.get("from")?.as_str()?;
                    let value = msg.get("value")?.as_str()?;
                    if value.is_empty() {
                        return None;
                    }
                    Some(format!("[{role}]: {value}"))
                })
                .collect();
            if parts.is_empty() {
                return None;
            }
            Some(parts.join("\n\n"))
        }
        "zai-org/LongBench-v2" => {
            let context = item.get("context")?.as_str()?;
            if context.is_empty() {
                return None;
            }
            Some(context.to_string())
        }
        _ => None,
    }
}

/// Load text samples from a HuggingFace Hub dataset.
fn load_dataset(dataset: &str, max_items: usize) -> Vec<String> {
    let json_file = dataset_json_file(dataset);

    eprintln!("[setup] Downloading dataset {dataset}...");
    let api = hf_hub::api::sync::Api::new().expect("Failed to create HuggingFace API client");
    let repo = api.dataset(dataset.to_string());
    let json_path = repo.get(json_file).expect("Failed to download dataset");

    let text = std::fs::read_to_string(&json_path).expect("Failed to read dataset JSON");
    let data: Vec<serde_json::Value> =
        serde_json::from_str(&text).expect("Failed to parse dataset JSON");

    let samples: Vec<String> = data
        .iter()
        .take(max_items)
        .filter_map(|item| extract_text(dataset, item))
        .collect();

    eprintln!("[setup] Loaded {} samples", samples.len());
    samples
}

fn print_summary(
    label: &str,
    n: usize,
    total_chars: u64,
    total_tokens: u64,
    total_hf: Duration,
    total_ft: Duration,
) {
    let hf_ms = total_hf.as_secs_f64() * 1000.0;
    let ft_ms = total_ft.as_secs_f64() * 1000.0;
    let speedup = hf_ms / ft_ms;
    let nf = n as f64;

    println!();
    println!("===  {label} ({n} samples)  ===");
    println!("  Total chars:        {total_chars}");
    println!("  Total tokens:       {total_tokens}");
    println!("  ---");
    println!("  HF total:           {hf_ms:>10.2} ms");
    println!("  fastokens total:   {ft_ms:>10.2} ms");
    println!("  Speedup:            {speedup:>10.2}x");
    println!("  ---");
    println!("  HF avg/sample:      {:>10.3} ms", hf_ms / nf);
    println!("  ft avg/sample:      {:>10.3} ms", ft_ms / nf);
    println!(
        "  HF throughput:      {:>10.2} MB/s",
        total_chars as f64 / total_hf.as_secs_f64() / 1_000_000.0
    );
    println!(
        "  ft throughput:      {:>10.2} MB/s",
        total_chars as f64 / total_ft.as_secs_f64() / 1_000_000.0
    );
}

fn bench_sequential(samples: &[String], hf: &HuggingFaceTokenizer, fast: &FastTokenizer) {
    let mut total_hf = Duration::ZERO;
    let mut total_ft = Duration::ZERO;
    let mut total_tokens: u64 = 0;
    let mut total_chars: u64 = 0;
    let mut mismatches = 0u64;

    for (i, text) in samples.iter().enumerate() {
        let t0 = Instant::now();
        let hf_enc = hf.encode(text).expect("HF encode failed");
        let t1 = Instant::now();
        let ft_enc = fast.encode(text).expect("fastokens encode failed");
        let t2 = Instant::now();

        let dt_hf = t1 - t0;
        let dt_ft = t2 - t1;

        if hf_enc.token_ids() != ft_enc.token_ids() {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!(
                    "[MISMATCH] sample {i}: hf={} tokens, ft={} tokens",
                    hf_enc.token_ids().len(),
                    ft_enc.token_ids().len()
                );
            }
        }

        total_hf += dt_hf;
        total_ft += dt_ft;
        total_tokens += ft_enc.token_ids().len() as u64;
        total_chars += text.len() as u64;

        if (i + 1) % 20 == 0 {
            eprintln!("[progress] {}/{}", i + 1, samples.len());
        }
    }

    if mismatches > 0 {
        eprintln!("[WARNING] {mismatches} samples had mismatched token IDs");
    } else {
        eprintln!("[OK] All samples produced identical token IDs");
    }

    print_summary(
        "Sequential Benchmark",
        samples.len(),
        total_chars,
        total_tokens,
        total_hf,
        total_ft,
    );
}

fn bench_batched(
    samples: &[String],
    hf: &HuggingFaceTokenizer,
    fast: &FastTokenizer,
    batch_size: usize,
) {
    let mut total_hf = Duration::ZERO;
    let mut total_ft = Duration::ZERO;
    let mut total_tokens: u64 = 0;
    let mut total_chars: u64 = 0;
    let mut mismatches = 0u64;

    let num_batches = samples.len().div_ceil(batch_size);

    for (batch_idx, batch) in samples.chunks(batch_size).enumerate() {
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let batch_chars: u64 = batch.iter().map(|s| s.len() as u64).sum();

        let t0 = Instant::now();
        let hf_results = hf
            .encode_batch(&batch_refs)
            .expect("HF encode_batch failed");
        let t1 = Instant::now();
        let ft_results = fast
            .encode_batch(&batch_refs)
            .expect("fastokens encode_batch failed");
        let t2 = Instant::now();

        // Verify correctness per sample within the batch
        if hf_results.len() != ft_results.len() {
            eprintln!(
                "[LENGTH MISMATCH] batch {batch_idx}: hf returned {} results, ft returned {} results (expected {})",
                hf_results.len(),
                ft_results.len(),
                batch.len()
            );
        }
        let max_len = hf_results.len().max(ft_results.len());
        for j in 0..max_len {
            let global_idx = batch_idx * batch_size + j;
            match (hf_results.get(j), ft_results.get(j)) {
                (Some(hf_enc), Some(ft_enc)) => {
                    if hf_enc.token_ids() != ft_enc.token_ids() {
                        mismatches += 1;
                        if mismatches <= 3 {
                            eprintln!(
                                "[MISMATCH] sample {global_idx}: hf={} tokens, ft={} tokens",
                                hf_enc.token_ids().len(),
                                ft_enc.token_ids().len()
                            );
                        }
                    }
                }
                (Some(_), None) => {
                    mismatches += 1;
                    if mismatches <= 3 {
                        eprintln!("[MISMATCH] sample {global_idx}: hf produced result, ft missing");
                    }
                }
                (None, Some(_)) => {
                    mismatches += 1;
                    if mismatches <= 3 {
                        eprintln!("[MISMATCH] sample {global_idx}: ft produced result, hf missing");
                    }
                }
                (None, None) => unreachable!(),
            }
        }

        let batch_tokens: u64 = ft_results
            .iter()
            .map(|enc| enc.token_ids().len() as u64)
            .sum();

        total_hf += t1 - t0;
        total_ft += t2 - t1;
        total_tokens += batch_tokens;
        total_chars += batch_chars;

        if (batch_idx + 1) % 5 == 0 {
            eprintln!("[progress] batch {}/{num_batches}", batch_idx + 1);
        }
    }

    if mismatches > 0 {
        eprintln!("[WARNING] {mismatches} samples had mismatched token IDs");
    } else {
        eprintln!("[OK] All samples produced identical token IDs");
    }

    print_summary(
        &format!("Batched Benchmark (batch_size={batch_size})"),
        samples.len(),
        total_chars,
        total_tokens,
        total_hf,
        total_ft,
    );
}

fn main() {
    // This benchmark downloads a large dataset and takes several minutes.
    // It is opt-in to avoid blocking `cargo test --all-targets` in CI.
    if std::env::var("RUN_BENCH").is_err() {
        eprintln!("[skip] tokenizer_dataset benchmark skipped. Set RUN_BENCH=1 to run it.");
        eprintln!("[skip] See lib/llm/benches/README.md for usage.");
        return;
    }

    let tokenizer_path = resolve_tokenizer_path();
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| DEFAULT_DATASET.to_string());
    let max_samples: usize = std::env::var("MAX_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_SAMPLES);
    let batch_size: Option<usize> = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok());

    let samples = load_dataset(&dataset, max_samples);

    let hf = HuggingFaceTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load HuggingFace tokenizer");
    let fast =
        FastTokenizer::from_file(&tokenizer_path).expect("Failed to load fastokens tokenizer");

    // Warmup
    if let Some(s) = samples.first() {
        let _ = hf.encode(s.as_str());
        let _ = fast.encode(s.as_str());
    }

    if let Some(bs) = batch_size {
        bench_batched(&samples, &hf, &fast, bs);
    } else {
        bench_sequential(&samples, &hf, &fast);
    }
}
