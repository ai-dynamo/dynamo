//! Benchmark: serde_json parse of a ~40MB multimodal request body
//!
//! Usage: cargo run --example json_parse_bench --release -p dynamo-llm

use std::time::Instant;

fn main() {
    let body = std::fs::read("/tmp/bench_request_body.json").expect(
        "Failed to read /tmp/bench_request_body.json. Generate it first with the Python script.",
    );
    println!("Body size: {:.1} MB", body.len() as f64 / 1024.0 / 1024.0);
    println!();

    // Benchmark 1: serde_json::Value (generic parse, allocates all strings)
    let mut times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let elapsed = start.elapsed().as_millis();
        times.push(elapsed);
        drop(_v);
    }
    println!(
        "serde_json::Value (owned):     avg={}ms  min={}ms",
        times.iter().sum::<u128>() / times.len() as u128,
        times.iter().min().unwrap()
    );

    // Benchmark 2: serde_json::Value with borrowed strings (zero-copy where possible)
    let mut times2 = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        // Use from_str which can borrow string slices
        let body_str = std::str::from_utf8(&body).unwrap();
        let _v: serde_json::Value = serde_json::from_str(body_str).unwrap();
        let elapsed = start.elapsed().as_millis();
        times2.push(elapsed);
        drop(_v);
    }
    println!(
        "serde_json::Value (from_str):   avg={}ms  min={}ms",
        times2.iter().sum::<u128>() / times2.len() as u128,
        times2.iter().min().unwrap()
    );

    // Benchmark 3: Just scan body length (baseline IO)
    let mut times3 = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let mut count = 0u64;
        for &b in &body {
            if b == b'"' {
                count += 1;
            }
        }
        let elapsed = start.elapsed().as_millis();
        times3.push(elapsed);
        println!("  (quotes found: {})", count);
    }
    println!(
        "raw byte scan:                  avg={}ms  min={}ms",
        times3.iter().sum::<u128>() / times3.len() as u128,
        times3.iter().min().unwrap()
    );

    // Benchmark 4: Parse as the actual NvCreateChatCompletionRequest type
    // This requires the full type, which we import from the protocols crate
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    let mut times4 = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _v: NvCreateChatCompletionRequest = serde_json::from_slice(&body).unwrap();
        let elapsed = start.elapsed().as_millis();
        times4.push(elapsed);
        drop(_v);
    }
    println!(
        "NvCreateChatCompletionRequest:  avg={}ms  min={}ms",
        times4.iter().sum::<u128>() / times4.len() as u128,
        times4.iter().min().unwrap()
    );
}
