// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multiturn chat benchmark.
//!
//! Simulates N concurrent users, each running a multiturn conversation against
//! an OpenAI-compatible `/v1/chat/completions` endpoint. Measures TTFT (time to
//! first token) and total request latency per turn, with configurable inter-turn
//! exponential delay.
//!
//! Run with: cargo run --package dynamo-bench --bin multiturn_bench -- --help

use anyhow::{Context, Result};
use clap::Parser;
use dynamo_bench::common::{ChatMessage, LatencyStats, fetch_model_name};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "multiturn_bench")]
#[command(about = "Multiturn chat completion benchmark with concurrent users")]
struct Args {
    /// Frontend HTTP endpoint
    #[arg(long, default_value = "http://localhost:8000")]
    url: String,

    /// Model name (auto-detected from /v1/models if omitted)
    #[arg(long)]
    model: Option<String>,

    /// Number of concurrent simulated users
    #[arg(long, default_value = "10")]
    num_users: usize,

    /// Number of conversation turns per user
    #[arg(long, default_value = "5")]
    num_turns: usize,

    /// Approximate user-prompt token count per turn (lorem ipsum)
    #[arg(long, default_value = "128")]
    num_user_tokens: usize,

    /// Maximum completion tokens (output sequence length cap)
    #[arg(long, default_value = "1000")]
    max_completion_tokens: u32,

    /// Force generation to max tokens (ignore end-of-sequence)
    #[arg(long, default_value = "true")]
    ignore_eos: bool,

    /// Mean inter-turn delay in milliseconds (exponential distribution)
    #[arg(long, default_value = "5000")]
    mean_delay_ms: u64,

    /// Enable speculative prefill via nvext
    #[arg(long)]
    speculative_prefill: bool,

    /// Write results to JSON file
    #[arg(long)]
    output: Option<String>,

    /// Print per-turn logging
    #[arg(short, long)]
    verbose: bool,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Smoke-test mode: 1 user, 1 turn, ~50 tokens ISL/OSL, no delay
    #[arg(long)]
    ping: bool,

    /// Ignored -- passed by cargo bench
    #[arg(long, hide = true)]
    bench: bool,
}

// ---------------------------------------------------------------------------
// Request body (richer than the common ChatCompletionRequest)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct MultiturnRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_completion_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    ignore_eos: Option<bool>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    nvext: Option<NvExtBody>,
}

#[derive(Debug, Serialize)]
struct NvExtBody {
    agent_hints: AgentHintsBody,
}

#[derive(Debug, Serialize)]
struct AgentHintsBody {
    speculative_prefill: bool,
}

// ---------------------------------------------------------------------------
// Turn result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct TurnResult {
    user_id: usize,
    turn: usize,
    ttft_us: u64,
    total_latency_us: u64,
    success: bool,
}

// ---------------------------------------------------------------------------
// Lorem ipsum generator
// ---------------------------------------------------------------------------

const LOREM_WORDS: &[&str] = &[
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "do",
    "eiusmod",
    "tempor",
    "incididunt",
    "ut",
    "labore",
    "et",
    "dolore",
    "magna",
    "aliqua",
    "enim",
    "ad",
    "minim",
    "veniam",
    "quis",
    "nostrud",
    "exercitation",
    "ullamco",
    "laboris",
    "nisi",
    "aliquip",
    "ex",
    "ea",
    "commodo",
    "consequat",
    "duis",
    "aute",
    "irure",
    "in",
    "reprehenderit",
    "voluptate",
    "velit",
    "esse",
    "cillum",
    "fugiat",
    "nulla",
    "pariatur",
    "excepteur",
    "sint",
    "occaecat",
    "cupidatat",
    "non",
    "proident",
    "sunt",
    "culpa",
    "qui",
    "officia",
    "deserunt",
    "mollit",
    "anim",
    "id",
    "est",
    "laborum",
];

fn generate_lorem(rng: &mut StdRng, approx_tokens: usize) -> String {
    // ~1.3 tokens per word on average for English text
    let word_count = (approx_tokens as f64 * 0.8) as usize;
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        let idx = rng.random_range(0..LOREM_WORDS.len());
        words.push(LOREM_WORDS[idx]);
    }
    words.join(" ")
}

fn generate_system_prompt(user_id: usize) -> String {
    format!(
        "You are a helpful assistant. You are assisting user {}. \
         Answer questions thoroughly and accurately.",
        user_id
    )
}

// ---------------------------------------------------------------------------
// SSE stream consumer
// ---------------------------------------------------------------------------

async fn consume_sse_stream(response: reqwest::Response) -> Result<(Duration, String)> {
    let start = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut accumulated = String::new();
    let mut buffer = String::new();

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream read error")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE events (delimited by double newlines)
        while let Some(boundary) = buffer.find("\n\n") {
            let event_block = buffer[..boundary].to_string();
            buffer = buffer[boundary + 2..].to_string();

            for line in event_block.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };

                if data.trim() == "[DONE]" {
                    let ttft = first_token_time
                        .map(|t| t.duration_since(start))
                        .unwrap_or_else(|| start.elapsed());
                    return Ok((ttft, accumulated));
                }

                let Ok(val) = serde_json::from_str::<serde_json::Value>(data) else {
                    continue;
                };

                if let Some(text) = val
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                    && !text.is_empty()
                {
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }
                    accumulated.push_str(text);
                }
            }
        }
    }

    // Stream ended without [DONE]
    let ttft = first_token_time
        .map(|t| t.duration_since(start))
        .unwrap_or_else(|| start.elapsed());
    Ok((ttft, accumulated))
}

// ---------------------------------------------------------------------------
// Per-user conversation loop
// ---------------------------------------------------------------------------

async fn run_user(
    client: reqwest::Client,
    url: String,
    model: String,
    args: Arc<Args>,
    user_id: usize,
    progress: ProgressBar,
) -> Vec<TurnResult> {
    let mut rng = StdRng::seed_from_u64(args.seed.wrapping_add(user_id as u64));
    let mean_delay = args.mean_delay_ms as f64;

    let system_prompt = generate_system_prompt(user_id);
    let mut messages = vec![ChatMessage {
        role: "system".to_string(),
        content: system_prompt,
    }];

    let mut results = Vec::with_capacity(args.num_turns);

    for turn in 0..args.num_turns {
        // Generate user prompt
        let user_text = generate_lorem(&mut rng, args.num_user_tokens);
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_text,
        });

        let body = MultiturnRequest {
            model: model.clone(),
            messages: messages.clone(),
            max_completion_tokens: args.max_completion_tokens,
            ignore_eos: if args.ignore_eos { Some(true) } else { None },
            stream: true,
            nvext: if args.speculative_prefill {
                Some(NvExtBody {
                    agent_hints: AgentHintsBody {
                        speculative_prefill: true,
                    },
                })
            } else {
                None
            },
        };

        let req_start = Instant::now();
        let response = client.post(&url).json(&body).send().await;

        let result = match response {
            Ok(resp) if resp.status().is_success() => match consume_sse_stream(resp).await {
                Ok((ttft, text)) => {
                    let total = req_start.elapsed();
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    });
                    TurnResult {
                        user_id,
                        turn,
                        ttft_us: ttft.as_micros() as u64,
                        total_latency_us: total.as_micros() as u64,
                        success: true,
                    }
                }
                Err(e) => {
                    let total = req_start.elapsed();
                    if args.verbose {
                        eprintln!("  [user {}][turn {}] stream error: {}", user_id, turn, e);
                    }
                    TurnResult {
                        user_id,
                        turn,
                        ttft_us: 0,
                        total_latency_us: total.as_micros() as u64,
                        success: false,
                    }
                }
            },
            Ok(resp) => {
                let total = req_start.elapsed();
                let status = resp.status();
                if args.verbose {
                    let body = resp.text().await.unwrap_or_default();
                    eprintln!(
                        "  [user {}][turn {}] HTTP {}: {}",
                        user_id, turn, status, body
                    );
                }
                TurnResult {
                    user_id,
                    turn,
                    ttft_us: 0,
                    total_latency_us: total.as_micros() as u64,
                    success: false,
                }
            }
            Err(e) => {
                let total = req_start.elapsed();
                if args.verbose {
                    eprintln!("  [user {}][turn {}] request error: {}", user_id, turn, e);
                }
                TurnResult {
                    user_id,
                    turn,
                    ttft_us: 0,
                    total_latency_us: total.as_micros() as u64,
                    success: false,
                }
            }
        };

        if args.verbose {
            progress.println(format!(
                "  [user {}][turn {}/{}] ttft={:.1}ms  total={:.1}s  ok={}",
                user_id,
                turn + 1,
                args.num_turns,
                result.ttft_us as f64 / 1000.0,
                result.total_latency_us as f64 / 1_000_000.0,
                result.success,
            ));
        }

        results.push(result);
        progress.inc(1);

        // Exponential inter-turn delay (skip after last turn)
        // Exp(1/mean) = -mean * ln(U), U ~ Uniform(0,1)
        if turn + 1 < args.num_turns {
            let u: f64 = rng.random();
            let delay_ms = (-mean_delay * u.ln()).max(0.0);
            tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Statistics & reporting
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct BenchmarkResults {
    num_users: usize,
    num_turns: usize,
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    speculative_prefill: bool,

    aggregate_ttft: Option<StatsReport>,
    aggregate_latency: Option<StatsReport>,
    per_turn_ttft: Vec<PerTurnStats>,
    per_turn_latency: Vec<PerTurnStats>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    raw_results: Vec<TurnResult>,
}

#[derive(Debug, Clone, Serialize)]
struct StatsReport {
    min_us: u64,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    max_us: u64,
}

impl StatsReport {
    fn from_latency_stats(s: &LatencyStats) -> Self {
        Self {
            min_us: s.min.as_micros() as u64,
            p50_us: s.p50.as_micros() as u64,
            p95_us: s.p95.as_micros() as u64,
            p99_us: s.p99.as_micros() as u64,
            max_us: s.max.as_micros() as u64,
        }
    }

    fn print(&self, label: &str) {
        println!("{}:", label);
        println!("  min:  {:>12.1}ms", self.min_us as f64 / 1000.0);
        println!("  p50:  {:>12.1}ms", self.p50_us as f64 / 1000.0);
        println!("  p95:  {:>12.1}ms", self.p95_us as f64 / 1000.0);
        println!("  p99:  {:>12.1}ms", self.p99_us as f64 / 1000.0);
        println!("  max:  {:>12.1}ms", self.max_us as f64 / 1000.0);
    }
}

#[derive(Debug, Clone, Serialize)]
struct PerTurnStats {
    turn: usize,
    count: usize,
    min_us: u64,
    p50_us: u64,
    p95_us: u64,
    max_us: u64,
}

fn compute_per_turn_stats(
    results: &[TurnResult],
    num_turns: usize,
    extract: fn(&TurnResult) -> u64,
) -> Vec<PerTurnStats> {
    (0..num_turns)
        .filter_map(|turn| {
            let durations: Vec<Duration> = results
                .iter()
                .filter(|r| r.turn == turn && r.success)
                .map(|r| Duration::from_micros(extract(r)))
                .collect();

            let stats = LatencyStats::from_durations(&durations)?;
            Some(PerTurnStats {
                turn,
                count: durations.len(),
                min_us: stats.min.as_micros() as u64,
                p50_us: stats.p50.as_micros() as u64,
                p95_us: stats.p95.as_micros() as u64,
                max_us: stats.max.as_micros() as u64,
            })
        })
        .collect()
}

fn print_per_turn_table(label: &str, stats: &[PerTurnStats]) {
    println!("{}:", label);
    println!(
        "  {:>6} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Turn", "Count", "Min(ms)", "P50(ms)", "P95(ms)", "Max(ms)"
    );
    println!("  {}", "-".repeat(66));
    for s in stats {
        println!(
            "  {:>6} {:>8} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
            s.turn + 1,
            s.count,
            s.min_us as f64 / 1000.0,
            s.p50_us as f64 / 1000.0,
            s.p95_us as f64 / 1000.0,
            s.max_us as f64 / 1000.0,
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();

    if args.ping {
        args.num_users = 1;
        args.num_turns = 1;
        args.num_user_tokens = 50;
        args.max_completion_tokens = 50;
        args.mean_delay_ms = 0;
        args.verbose = true;
    }

    println!("Multiturn Chat Benchmark");
    println!("========================\n");

    // Resolve model name
    let model = match args.model.clone() {
        Some(m) => m,
        None => {
            println!("Model Detection:");
            fetch_model_name(&args.url).await?
        }
    };

    println!("Configuration:");
    println!("  URL: {}", args.url);
    println!("  Model: {}", model);
    println!("  Users: {}", args.num_users);
    println!("  Turns: {}", args.num_turns);
    println!("  User tokens: ~{}", args.num_user_tokens);
    println!("  Max completion tokens: {}", args.max_completion_tokens);
    println!("  Ignore EOS: {}", args.ignore_eos);
    println!("  Mean delay: {}ms", args.mean_delay_ms);
    println!("  Speculative prefill: {}", args.speculative_prefill);
    println!("  Seed: {}", args.seed);
    println!();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .context("Failed to create HTTP client")?;

    let args = Arc::new(args);
    let chat_url = format!("{}/v1/chat/completions", args.url);

    // Spawn concurrent user tasks
    println!(
        "Running {} users x {} turns = {} total requests...\n",
        args.num_users,
        args.num_turns,
        args.num_users * args.num_turns,
    );

    let bench_start = Instant::now();

    let total_turns = (args.num_users * args.num_turns) as u64;
    let progress = ProgressBar::new(total_turns);
    progress.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} turns ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let handles: Vec<_> = (0..args.num_users)
        .map(|user_id| {
            let client = client.clone();
            let url = chat_url.clone();
            let model = model.clone();
            let args = args.clone();
            let progress = progress.clone();
            tokio::spawn(async move { run_user(client, url, model, args, user_id, progress).await })
        })
        .collect();

    // Collect results from all users
    let mut all_results = Vec::new();
    for handle in handles {
        let user_results = handle.await.context("user task panicked")?;
        all_results.extend(user_results);
    }
    progress.finish_and_clear();

    let bench_elapsed = bench_start.elapsed();
    println!(
        "Benchmark completed in {:.1}s\n",
        bench_elapsed.as_secs_f64()
    );

    // Compute statistics
    let successful: Vec<&TurnResult> = all_results.iter().filter(|r| r.success).collect();
    let failed_count = all_results.len() - successful.len();

    let ttft_durations: Vec<Duration> = successful
        .iter()
        .map(|r| Duration::from_micros(r.ttft_us))
        .collect();
    let latency_durations: Vec<Duration> = successful
        .iter()
        .map(|r| Duration::from_micros(r.total_latency_us))
        .collect();

    let agg_ttft =
        LatencyStats::from_durations(&ttft_durations).map(|s| StatsReport::from_latency_stats(&s));
    let agg_latency = LatencyStats::from_durations(&latency_durations)
        .map(|s| StatsReport::from_latency_stats(&s));
    let per_turn_ttft = compute_per_turn_stats(&all_results, args.num_turns, |r| r.ttft_us);
    let per_turn_latency =
        compute_per_turn_stats(&all_results, args.num_turns, |r| r.total_latency_us);

    // Print report
    println!("========================================");
    println!("Multiturn Benchmark Results");
    println!("========================================\n");

    println!(
        "Users: {}, Turns: {}, Completed: {}/{}\n",
        args.num_users,
        args.num_turns,
        successful.len(),
        all_results.len(),
    );

    if failed_count > 0 {
        println!("Failed requests: {}\n", failed_count);
    }

    if let Some(ref s) = agg_ttft {
        s.print("Aggregate TTFT");
        println!();
    }

    if let Some(ref s) = agg_latency {
        s.print("Aggregate Request Latency");
        println!();
    }

    if !per_turn_ttft.is_empty() {
        print_per_turn_table("Per-Turn TTFT", &per_turn_ttft);
        println!();
    }

    if !per_turn_latency.is_empty() {
        print_per_turn_table("Per-Turn Request Latency", &per_turn_latency);
        println!();
    }

    // JSON output
    if let Some(ref output_path) = args.output {
        let report = BenchmarkResults {
            num_users: args.num_users,
            num_turns: args.num_turns,
            total_requests: all_results.len(),
            successful_requests: successful.len(),
            failed_requests: failed_count,
            speculative_prefill: args.speculative_prefill,
            aggregate_ttft: agg_ttft,
            aggregate_latency: agg_latency,
            per_turn_ttft,
            per_turn_latency,
            raw_results: all_results,
        };
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(output_path, json)?;
        println!("Results written to: {}", output_path);
    }

    Ok(())
}
