// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full stress test for the KV Router.
//!
//! Stress tests the full `KvRouter` frontend without worker backends:
//! - Phase 1: Build tree by publishing KV events to NATS (with computed hashes matching tokenized requests)
//! - Phase 2: Send HTTP requests and measure routing decision latency
//!
//! The key feature is that tree construction uses the same hash computation as the frontend,
//! ensuring that HTTP requests will match the pre-populated tree entries.
//!
//! Run with: cargo bench --package dynamo-llm --bench kv_router_bench --features kv-router-stress -- --help

use anyhow::{Context, Result};
use bytes::Bytes;
use clap::Parser;
use dynamo_runtime::transports::event_plane::EventEnvelope;
use hf_hub;
use indicatif::{ProgressBar, ProgressStyle};
use minijinja::{Environment, context, value::Value};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, Semaphore};

use dynamo_llm::kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerId, compute_hash,
    compute_seq_hash_for_block,
};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::prompt::{
    ChatTemplate, ContextMixins, OAIChatLikeRequest, PromptFormatter,
};

/// KV Router event subject suffix (appended to Component.subject())
/// Full subject format: namespace.{namespace}.component.{component}.kv-events
const KV_EVENT_SUBJECT: &str = "kv-events";

/// Unique publisher ID for this benchmark instance
static PUBLISHER_ID: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
});

/// Sequence counter for envelope ordering
static ENVELOPE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Encode an event into Msgpack format with EventEnvelope wrapper.
/// This matches the format expected by the event plane subscriber.
fn encode_event_with_envelope<T: Serialize>(event: &T, topic: &str) -> Result<Vec<u8>> {
    // Encode the payload with msgpack
    let payload = rmp_serde::to_vec_named(event).context("Failed to encode event payload")?;

    // Create the envelope
    let envelope = EventEnvelope {
        publisher_id: *PUBLISHER_ID,
        sequence: ENVELOPE_SEQUENCE.fetch_add(1, Ordering::SeqCst),
        published_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        topic: topic.to_string(),
        payload: Bytes::from(payload),
    };

    // Encode the envelope with msgpack
    rmp_serde::to_vec_named(&envelope).context("Failed to encode envelope")
}

#[derive(Parser, Debug)]
#[command(name = "kv_router_bench")]
#[command(about = "Full stress test for the KV Router via NATS events and HTTP requests")]
struct Args {
    // Tree construction parameters
    /// Target tree size in total (worker, block) pairs
    #[arg(long, default_value = "500000")]
    tree_size: usize,

    /// Sequence depth in blocks (blocks per sequence)
    #[arg(long, default_value = "512")]
    depth: usize,

    /// Number of workers to distribute blocks across
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// Portion of sequence that is shared prefix (0.0 to 1.0)
    #[arg(long, default_value = "0.25")]
    prefix_prompt_ratio: f64,

    /// Number of unique prefix groups
    #[arg(long, default_value = "20")]
    num_prefix_prompts: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    // Stress test parameters
    /// KV events per second during stress test (0 = no ongoing events)
    #[arg(long, default_value = "0")]
    event_rate: f64,

    /// HTTP requests per second
    #[arg(long, default_value = "100")]
    request_rate: f64,

    /// Test duration in seconds
    #[arg(long, default_value = "30")]
    duration: u64,

    /// Warmup duration before measurement in seconds
    #[arg(long, default_value = "5")]
    warmup: u64,

    /// Maximum concurrent HTTP requests
    #[arg(long, default_value = "50")]
    concurrency: usize,

    // Infrastructure
    /// NATS server URL
    #[arg(long, default_value = "nats://localhost:4222")]
    nats_url: String,

    /// Frontend HTTP URL
    #[arg(long, default_value = "http://localhost:8000")]
    frontend_url: String,

    /// NATS namespace (used to construct subject)
    #[arg(long, default_value = "dynamo")]
    namespace: String,

    /// Component name (used to construct subject)
    #[arg(long, default_value = "backend")]
    component: String,

    // Output
    /// Write results to JSON file
    #[arg(long)]
    output: Option<String>,

    /// Print per-request timings
    #[arg(short, long)]
    verbose: bool,

    /// Skip tree construction via NATS (use when kv_stress_worker handles it)
    #[arg(long)]
    skip_tree_construction: bool,

    /// Time bucket size in seconds for latency-over-time tracking (0 to disable)
    #[arg(long, default_value = "5")]
    bucket_size: u64,

    /// Include raw latency samples in JSON output (for graphing)
    #[arg(long)]
    include_raw_samples: bool,

    /// Model name to use in requests (should match the registered model).
    /// If not specified, auto-detects from /v1/models when exactly one model is available.
    #[arg(long)]
    model: Option<String>,

    /// KV block size in tokens (must match frontend configuration)
    #[arg(long, default_value = "16")]
    kv_block_size: u32,

    /// Path to tokenizer (HuggingFace model ID or local path). Defaults to --model value.
    #[arg(long)]
    tokenizer_path: Option<String>,

    /// Ignored - passed by cargo bench
    #[arg(long, hide = true)]
    bench: bool,
}

/// Compute LocalBlockHash (tokens_hash) from a slice of token IDs.
/// Uses the same algorithm as lib/llm/src/kv_router/protocols.rs::compute_block_hash_for_seq
fn compute_block_hashes(tokens: &[u32], kv_block_size: u32) -> Vec<LocalBlockHash> {
    tokens
        .chunks_exact(kv_block_size as usize)
        .map(|chunk| {
            let bytes: Vec<u8> = chunk.iter().flat_map(|&num| num.to_le_bytes()).collect();
            LocalBlockHash(compute_hash(&bytes))
        })
        .collect()
}

/// Compute ExternalSequenceBlockHash (block_hash) from LocalBlockHash values.
/// Uses the router's compute_seq_hash_for_block to ensure identical computation.
fn compute_sequence_hashes(block_hashes: &[LocalBlockHash]) -> Vec<ExternalSequenceBlockHash> {
    // Use the router's sequence hash computation to ensure consistency
    let seq_hashes = compute_seq_hash_for_block(block_hashes);
    seq_hashes
        .into_iter()
        .map(ExternalSequenceBlockHash::from)
        .collect()
}

fn compute_hashes_for_content(
    content: &str,
    tokenizer: &Tokenizer,
    kv_block_size: u32,
    prompt_renderer: Option<&PromptRenderer>,
) -> Result<(Vec<LocalBlockHash>, Vec<ExternalSequenceBlockHash>)> {
    let formatted_text = if let Some(renderer) = prompt_renderer {
        renderer.render_user_message(content)?
    } else {
        content.to_string()
    };

    let encoding = tokenizer
        .encode(formatted_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize request content: {}", e))?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let local_hashes = compute_block_hashes(&token_ids, kv_block_size);
    let external_hashes = compute_sequence_hashes(&local_hashes);

    Ok((local_hashes, external_hashes))
}

/// Tokenizer config from tokenizer_config.json
#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<String>,
    bos_token: Option<serde_json::Value>,
    eos_token: Option<serde_json::Value>,
}

impl TokenizerConfig {
    /// Extract bos_token as a string (handles both string and object formats)
    fn bos_token_str(&self) -> Option<String> {
        self.bos_token.as_ref().and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else if let Some(obj) = v.as_object() {
                obj.get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
    }

    /// Extract eos_token as a string (handles both string and object formats)
    fn eos_token_str(&self) -> Option<String> {
        self.eos_token.as_ref().and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else if let Some(obj) = v.as_object() {
                obj.get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
    }
}

/// Load tokenizer_config.json to get the chat template and special tokens.
fn load_tokenizer_config(model_or_path: &str) -> Result<Option<TokenizerConfig>> {
    use std::path::Path;

    let path = Path::new(model_or_path);

    // If it's a directory, look for tokenizer_config.json inside
    if path.is_dir() {
        let config_path = path.join("tokenizer_config.json");
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .context("Failed to read tokenizer_config.json")?;
            let config: TokenizerConfig =
                serde_json::from_str(&content).context("Failed to parse tokenizer_config.json")?;
            return Ok(Some(config));
        }
        return Ok(None);
    }

    // Try to download from HuggingFace
    let cache = hf_hub::Cache::default();
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(false)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create HuggingFace API client: {}", e))?;

    let repo = api.model(model_or_path.to_string());
    match repo.get("tokenizer_config.json") {
        Ok(config_path) => {
            let content = std::fs::read_to_string(&config_path)
                .context("Failed to read tokenizer_config.json")?;
            let config: TokenizerConfig =
                serde_json::from_str(&content).context("Failed to parse tokenizer_config.json")?;
            Ok(Some(config))
        }
        Err(_) => Ok(None),
    }
}

fn try_load_prompt_renderer(model_or_path: &str) -> Option<PromptRenderer> {
    use std::path::Path;

    let path = Path::new(model_or_path);
    if !path.is_dir() {
        return None;
    }

    let card = ModelDeploymentCard::load_from_disk(path, None).ok()?;
    let formatter = PromptFormatter::from_mdc(&card).ok()?;
    Some(PromptRenderer::Formatter(formatter))
}

/// Chat template renderer using minijinja.
#[derive(Clone)]
struct ChatTemplateRenderer {
    template: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplateRenderer {
    fn new(template: String, bos_token: Option<String>, eos_token: Option<String>) -> Self {
        Self {
            template,
            bos_token: bos_token.unwrap_or_else(|| "<s>".to_string()),
            eos_token: eos_token.unwrap_or_else(|| "</s>".to_string()),
        }
    }

    /// Apply the chat template to a list of messages.
    /// Returns the formatted prompt string.
    fn apply(&self, messages: &[ChatTemplateMessage]) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("chat", &self.template)
            .context("Failed to compile chat template")?;

        let tmpl = env.get_template("chat").unwrap();

        // Render with add_generation_prompt=true to match frontend behavior
        let result = tmpl
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
            })
            .context("Failed to render chat template")?;

        Ok(result)
    }
}

/// Minimal chat request to reuse the frontend prompt formatter.
struct SimpleChatRequest {
    messages: Vec<ChatTemplateMessage>,
}

impl OAIChatLikeRequest for SimpleChatRequest {
    fn model(&self) -> String {
        "kv_router_bench".to_string()
    }

    fn messages(&self) -> Value {
        Value::from_serialize(&self.messages)
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }
}

/// Prompt renderer that mirrors the frontend prompt formatting pipeline.
#[derive(Clone)]
enum PromptRenderer {
    Formatter(PromptFormatter),
    Simple(ChatTemplateRenderer),
}

impl PromptRenderer {
    fn render_user_message(&self, content: &str) -> Result<String> {
        match self {
            PromptRenderer::Formatter(formatter) => {
                let req = SimpleChatRequest {
                    messages: vec![ChatTemplateMessage {
                        role: "user".to_string(),
                        content: content.to_string(),
                    }],
                };
                match formatter {
                    PromptFormatter::OAI(inner) => inner.render(&req),
                }
            }
            PromptRenderer::Simple(renderer) => renderer.apply(&[ChatTemplateMessage {
                role: "user".to_string(),
                content: content.to_string(),
            }]),
        }
    }
}

/// Message format for chat template rendering
#[derive(Debug, Clone, Serialize)]
struct ChatTemplateMessage {
    role: String,
    content: String,
}

/// Load a tokenizer from a local path or HuggingFace model ID.
///
/// Tries in order:
/// 1. Direct file path (tokenizer.json)
/// 2. Directory containing tokenizer.json
/// 3. HuggingFace model ID (downloads tokenizer.json)
fn load_tokenizer(model_or_path: &str) -> Result<Tokenizer> {
    use std::path::Path;

    let path = Path::new(model_or_path);

    // If it's a file, load directly
    if path.is_file() {
        return Tokenizer::from_file(path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load tokenizer from file '{}': {}",
                model_or_path,
                e
            )
        });
    }

    // If it's a directory, look for tokenizer.json inside
    if path.is_dir() {
        let tokenizer_path = path.join("tokenizer.json");
        if tokenizer_path.exists() {
            return Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load tokenizer from '{}': {}",
                    tokenizer_path.display(),
                    e
                )
            });
        }
        return Err(anyhow::anyhow!(
            "Directory '{}' does not contain tokenizer.json",
            model_or_path
        ));
    }

    // Try to download from HuggingFace
    println!(
        "  Downloading tokenizer from HuggingFace: {}...",
        model_or_path
    );

    let cache = hf_hub::Cache::default();
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create HuggingFace API client: {}", e))?;

    let repo = api.model(model_or_path.to_string());
    let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
        anyhow::anyhow!(
            "Failed to download tokenizer.json from '{}': {}",
            model_or_path,
            e
        )
    })?;

    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load downloaded tokenizer: {}", e))
}

/// Pre-computed prefix data with text and corresponding hashes.
#[derive(Clone, Debug)]
struct PrefixData {
    /// The raw text content of this prefix (before chat template)
    text: String,
    /// The formatted text after applying chat template
    formatted_text: String,
    /// Token IDs from tokenizing the formatted text
    token_ids: Vec<u32>,
    /// LocalBlockHash values (tokens_hash) for each complete block
    local_hashes: Vec<LocalBlockHash>,
}

impl PrefixData {
    /// Create a new PrefixData by applying prompt formatting (if provided), tokenizing, and computing hashes.
    fn from_text(
        text: String,
        tokenizer: &Tokenizer,
        kv_block_size: u32,
        prompt_renderer: Option<&PromptRenderer>,
    ) -> Result<Self> {
        // Apply prompt formatting if provided
        let formatted_text = if let Some(renderer) = prompt_renderer {
            renderer.render_user_message(&text)?
        } else {
            text.clone()
        };

        let encoding = tokenizer
            .encode(formatted_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prefix: {}", e))?;

        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let local_hashes = compute_block_hashes(&token_ids, kv_block_size);

        Ok(Self {
            text,
            formatted_text,
            token_ids,
            local_hashes,
        })
    }

    /// Number of complete blocks in this prefix
    fn num_blocks(&self) -> usize {
        self.local_hashes.len()
    }
}

/// Pre-generated sequence data for benchmarking
#[derive(Clone)]
struct SequenceData {
    worker_id: WorkerId,
    local_hashes: Vec<LocalBlockHash>,
    external_hashes: Vec<ExternalSequenceBlockHash>,
}

impl SequenceData {
    /// Create a sequence from the exact request content.
    fn from_request_content(
        content: &str,
        worker_id: WorkerId,
        kv_block_size: u32,
        tokenizer: &Tokenizer,
        prompt_renderer: Option<&PromptRenderer>,
    ) -> Result<Self> {
        let (local_hashes, external_hashes) =
            compute_hashes_for_content(content, tokenizer, kv_block_size, prompt_renderer)?;

        Ok(Self {
            worker_id,
            local_hashes,
            external_hashes,
        })
    }

    fn to_router_event(&self, event_id: u64) -> RouterEvent {
        let kv_event = KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: self
                    .local_hashes
                    .iter()
                    .zip(self.external_hashes.iter())
                    .map(|(local, ext)| KvCacheStoredBlockData {
                        block_hash: *ext,
                        tokens_hash: *local,
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
            dp_rank: 0,
        };
        RouterEvent::new(self.worker_id, kv_event)
    }
}

/// Response from the frontend's /health endpoint
#[derive(Debug, Deserialize)]
struct HealthResponse {
    #[allow(dead_code)]
    status: String,
    instances: Vec<HealthInstance>,
}

/// Instance info from health endpoint
#[derive(Debug, Deserialize)]
struct HealthInstance {
    instance_id: u64,
    #[allow(dead_code)]
    endpoint: String,
}

/// Response from the frontend's /v1/models endpoint
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

/// Model info from /v1/models endpoint
#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// Fetch the model name from the frontend's /v1/models endpoint.
///
/// Returns the model ID if exactly one model is available.
/// Returns an error if zero or multiple models are found (requiring explicit --model).
async fn fetch_model_name(frontend_url: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/models", frontend_url);

    println!("  Auto-detecting model from {}...", url);

    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to connect to frontend /v1/models endpoint")?;

    if !response.status().is_success() {
        anyhow::bail!("Models endpoint returned status: {}", response.status());
    }

    let models: ModelsResponse = response
        .json()
        .await
        .context("Failed to parse models response")?;

    match models.data.len() {
        0 => anyhow::bail!("No models found at endpoint. Is a backend running?"),
        1 => {
            let model_id = models.data[0].id.clone();
            println!("  Auto-detected model: {}", model_id);
            Ok(model_id)
        }
        n => {
            println!("  Multiple models available ({}):", n);
            for m in &models.data {
                println!("    - {}", m.id);
            }
            anyhow::bail!("Multiple models available. Please specify --model explicitly.")
        }
    }
}

/// Discover worker IDs from the frontend's /health endpoint.
///
/// Returns a list of instance_ids (worker_ids) that are currently registered.
async fn discover_worker_ids(frontend_url: &str) -> Result<Vec<WorkerId>> {
    let client = reqwest::Client::new();
    let url = format!("{}/health", frontend_url);

    println!("  Discovering workers from {}...", url);

    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to connect to frontend /health endpoint")?;

    if !response.status().is_success() {
        anyhow::bail!("Health endpoint returned status: {}", response.status());
    }

    let health: HealthResponse = response
        .json()
        .await
        .context("Failed to parse health response")?;

    let worker_ids: Vec<WorkerId> = health.instances.iter().map(|i| i.instance_id).collect();

    // Deduplicate (in case of multiple endpoints per worker)
    let mut unique_ids: Vec<WorkerId> = worker_ids.clone();
    unique_ids.sort_unstable();
    unique_ids.dedup();

    println!("  Discovered {} workers", unique_ids.len());

    if unique_ids.is_empty() {
        anyhow::bail!("No workers discovered from frontend. Are kv_stress_workers running?");
    }

    Ok(unique_ids)
}

/// Generate sequences with shared prefix prompts using computed hashes.
///
/// This function:
/// 1. Takes pre-computed PrefixData with real token hashes
/// 2. Creates sequences that share these prefix hashes
/// 3. Adds unique suffix blocks for each sequence
///
/// The prefix hashes are computed from actual tokenized text, so HTTP requests
/// with the same prefix text will produce matching hashes in the frontend.
///
/// Worker IDs are taken from the provided list (discovered from frontend).
/// Uses parallel processing for tokenization to speed up generation.
fn generate_sequences_for_requests(
    num_sequences: usize,
    worker_ids: &[WorkerId],
    prefix_prompts: &[String],
    num_prefix_prompts: usize,
    kv_block_size: u32,
    tokenizer: &Tokenizer,
    prompt_renderer: Option<&PromptRenderer>,
    seed: u64,
    show_progress: bool,
) -> Result<Vec<SequenceData>> {
    if prefix_prompts.is_empty() || num_prefix_prompts == 0 {
        anyhow::bail!("No prefix prompts available for request-aligned sequence generation");
    }

    let progress = if show_progress {
        let pb = ProgressBar::new(num_sequences as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  [{bar:40.cyan/blue}] {pos}/{len} sequences ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );
        Some(pb)
    } else {
        None
    };

    // Clone tokenizer and prompt_renderer for parallel access
    let tokenizer = tokenizer.clone();
    let prompt_renderer_clone = prompt_renderer.cloned();
    let progress_clone = progress.clone();

    // Generate sequences in parallel
    let results: Result<Vec<SequenceData>> = (0..num_sequences as u64)
        .into_par_iter()
        .map(|request_id| {
            let worker_id = worker_ids[request_id as usize % worker_ids.len()];
            let (_prefix_idx, content) = build_request_content_with_prefix(
                request_id,
                prefix_prompts,
                num_prefix_prompts,
                seed,
            );
            let seq = SequenceData::from_request_content(
                &content,
                worker_id,
                kv_block_size,
                &tokenizer,
                prompt_renderer_clone.as_ref(),
            )?;
            if let Some(ref pb) = progress_clone {
                pb.inc(1);
            }
            Ok(seq)
        })
        .collect();

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    results
}

/// Build tree by publishing events to NATS
async fn build_tree_via_nats(
    nats_client: &async_nats::Client,
    namespace: &str,
    component: &str,
    sequences: &[SequenceData],
    verbose: bool,
) -> Result<Duration> {
    // Subject format must match Component.subject() from lib/runtime/src/component/component.rs
    // which returns: namespace.{namespace_name}.component.{component_name}
    let subject = format!(
        "namespace.{}.component.{}.{}",
        namespace, component, KV_EVENT_SUBJECT
    );

    println!(
        "Building tree: {} sequences to subject {}...",
        sequences.len(),
        subject
    );
    let start = Instant::now();

    let progress = if !verbose {
        let pb = ProgressBar::new(sequences.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );
        Some(pb)
    } else {
        None
    };

    for (event_id, seq) in sequences.iter().enumerate() {
        let event = seq.to_router_event(event_id as u64);
        let data = encode_event_with_envelope(&event, KV_EVENT_SUBJECT)?;
        nats_client
            .publish(subject.clone(), data.into())
            .await
            .context("Failed to publish to NATS")?;

        if let Some(ref pb) = progress {
            pb.set_position((event_id + 1) as u64);
        } else if verbose && (event_id + 1) % 100 == 0 {
            println!("  Published {}/{} events", event_id + 1, sequences.len());
        }
    }

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    nats_client.flush().await.context("Failed to flush NATS")?;

    // Wait for events to be processed by the frontend
    println!("  Waiting for event processing...");
    tokio::time::sleep(Duration::from_secs(2)).await;

    let elapsed = start.elapsed();
    println!("Tree construction: {:.2?}", elapsed);

    Ok(elapsed)
}

/// Result of a single HTTP request
#[derive(Debug, Clone)]
struct RequestResult {
    latency: Duration,
    /// Time when request completed, relative to measurement start
    completion_time: Duration,
    success: bool,
}

/// Individual latency sample for raw data export
#[derive(Debug, Clone, Serialize)]
struct LatencySample {
    /// Latency in microseconds
    latency_us: u64,
    /// Completion time in milliseconds from measurement start
    completion_time_ms: u64,
    /// Whether the request succeeded
    success: bool,
}

/// OpenAI-style chat completion request
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Generate prefix text content.
/// These are long enough to span multiple KV blocks when tokenized.
/// Each prefix is designed to be distinct and consistent across requests.
///
fn generate_prefix_text(prefix_id: usize, target_tokens: usize) -> String {
    // Each word is roughly 1-2 tokens. We generate enough words to hit target_tokens.
    // Using deterministic content so the same prefix_id always produces the same text.
    let words_per_prefix = target_tokens * 2; // Conservative estimate

    // Generate a deterministic "document" for each prefix
    // This simulates a system prompt or context that would be cached
    let mut content = format!(
        "System configuration document version {} revision {}. ",
        prefix_id,
        prefix_id * 17 + 3
    );

    // Add filler content to reach target token count
    for i in 0..words_per_prefix {
        let word_idx = (prefix_id * 1000 + i) % 100;
        let words = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was",
            "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new",
            "now", "old", "see", "two", "way", "who", "boy", "did", "oil", "sit", "set", "run",
            "top", "got", "let", "put", "say", "she", "too", "use", "dad", "mom", "end", "big",
            "ask", "own", "why", "men", "read", "need", "land", "same", "here", "must", "home",
            "hand", "high", "year", "come", "made", "find", "long", "down", "look", "write", "go",
            "word", "call", "first", "water", "been", "number", "people", "over", "such", "make",
            "time", "very", "when", "would", "more", "some", "into", "them", "than", "only",
            "have", "from", "this", "that", "with", "they", "will", "each", "about", "which",
        ];
        content.push_str(words[word_idx]);
        content.push(' ');
    }

    // Trim trailing space and end with punctuation + newline for clean token boundary.
    // This ensures the prefix tokenization is stable regardless of what follows.
    content = content.trim_end().to_string();
    content.push_str(".\n");

    content
}

/// Generate prefix prompts with pre-computed hashes.
///
/// This tokenizes each prefix and computes the block hashes that the frontend
/// will produce when it tokenizes the same text. This ensures that NATS events
/// and HTTP requests will have matching hashes.
///
/// If a chat template is provided, it will be applied to the messages before tokenizing,
/// matching the frontend's behavior for /v1/chat/completions requests.
///
/// Uses parallel processing for tokenization to speed up generation.
fn generate_prefix_data(
    num_prefixes: usize,
    target_tokens: usize,
    tokenizer: &Tokenizer,
    kv_block_size: u32,
    prompt_renderer: Option<&PromptRenderer>,
    show_progress: bool,
) -> Result<Vec<PrefixData>> {
    let progress = if show_progress {
        let pb = ProgressBar::new(num_prefixes as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  [{bar:40.cyan/blue}] {pos}/{len} prefixes ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );
        Some(pb)
    } else {
        None
    };

    // Generate prefix texts in parallel
    let texts: Vec<String> = (0..num_prefixes)
        .into_par_iter()
        .map(|prefix_id| generate_prefix_text(prefix_id, target_tokens))
        .collect();

    // Tokenize and compute hashes in parallel
    // The tokenizer is thread-safe (Send + Sync), and prompt rendering creates
    // a new Environment each time, so this is safe to parallelize.
    let tokenizer = tokenizer.clone();
    let prompt_renderer_clone = prompt_renderer.cloned();
    let progress_clone = progress.clone();
    let results: Result<Vec<PrefixData>> = texts
        .into_par_iter()
        .map(|text| {
            let result = PrefixData::from_text(
                text,
                &tokenizer,
                kv_block_size,
                prompt_renderer_clone.as_ref(),
            );
            if let Some(ref pb) = progress_clone {
                pb.inc(1);
            }
            result
        })
        .collect();

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    // Results are already collected, just return them
    results
}

/// Build an HTTP request body that will exercise routing with cache-friendly prefixes.
///
/// Uses a shared prefix prompt (based on group_id) plus a unique suffix.
/// This allows the warmup phase to populate the cache, and measurement phase
/// requests with the same prefix will get cache hits.
///
/// IMPORTANT: The suffix is appended with a newline separator to ensure clean token
/// boundaries. This prevents the suffix from affecting how the prefix tokens are
/// split by BPE tokenizers, ensuring that pre-computed prefix hashes match what
/// the frontend computes for the full request.
fn build_request_content_with_prefix(
    request_id: u64,
    prefix_prompts: &[String],
    num_prefix_prompts: usize,
    seed: u64,
) -> (usize, String) {
    // Deterministically select a prefix based on request_id and seed
    let prefix_idx = ((request_id ^ seed) as usize) % num_prefix_prompts.min(prefix_prompts.len());
    let prefix = &prefix_prompts[prefix_idx];

    // Add a unique suffix so each request is distinct but shares the prefix.
    // Use a newline separator to create a clean token boundary between prefix and suffix.
    // This ensures the prefix tokens remain identical whether tokenized alone or with suffix,
    // which is critical for hash matching between pre-computed NATS events and HTTP requests.
    let suffix = format!(
        "\n\nRequest {} query: What is the answer to question number {}?",
        request_id,
        request_id % 1000
    );

    let content = format!("{}{}", prefix, suffix);

    (prefix_idx, content)
}

fn build_routing_request_with_prefix(
    request_id: u64,
    prefix_prompts: &[String],
    num_prefix_prompts: usize,
    model: &str,
    seed: u64,
) -> ChatCompletionRequest {
    let (_prefix_idx, content) =
        build_request_content_with_prefix(request_id, prefix_prompts, num_prefix_prompts, seed);
    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content,
        }],
        max_tokens: Some(1),
    }
}

/// Send HTTP requests at a specified rate.
/// Returns the Unix timestamp (seconds since epoch) when warmup ended.
async fn send_requests_at_rate(
    client: reqwest::Client,
    frontend_url: String,
    prefix_prompts: Arc<Vec<String>>,
    num_prefix_prompts: usize,
    model: String,
    seed: u64,
    rate: f64,
    duration_secs: u64,
    warmup_secs: u64,
    max_concurrency: usize,
    results: Arc<Mutex<Vec<RequestResult>>>,
    in_flight: Arc<AtomicU64>,
    max_in_flight: Arc<AtomicU64>,
    verbose: bool,
) -> f64 {
    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    let interval = Duration::from_secs_f64(1.0 / rate);
    let start = Instant::now();
    let warmup_duration = Duration::from_secs(warmup_secs);
    let total_duration = Duration::from_secs(duration_secs + warmup_secs);
    let measurement_start = Arc::new(Mutex::new(None::<Instant>));
    let mut request_id = 0u64;
    let mut warmup_end_timestamp: f64 = 0.0;
    let mut warmup_ended = false;

    println!(
        "  Running for {}s ({}s warmup + {}s measurement) at {} req/sec...",
        warmup_secs + duration_secs,
        warmup_secs,
        duration_secs,
        rate
    );
    println!(
        "  Using {} prefix prompts for cache sharing (warmup populates cache)...",
        num_prefix_prompts
    );

    // Counters for completed requests (updated by spawned tasks)
    let success_count = Arc::new(AtomicU64::new(0));
    let failure_count = Arc::new(AtomicU64::new(0));

    // Monitor in-flight count and throughput every second
    let in_flight_monitor = in_flight.clone();
    let success_monitor = success_count.clone();
    let failure_monitor = failure_count.clone();
    let monitor_start = start;
    let monitor_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.tick().await; // Skip first immediate tick
        let mut prev_success = 0u64;
        let mut prev_failure = 0u64;
        loop {
            interval.tick().await;
            let in_flight_now = in_flight_monitor.load(Ordering::Relaxed);
            let success_now = success_monitor.load(Ordering::Relaxed);
            let failure_now = failure_monitor.load(Ordering::Relaxed);
            let success_delta = success_now - prev_success;
            let failure_delta = failure_now - prev_failure;
            prev_success = success_now;
            prev_failure = failure_now;
            eprintln!(
                "  [t={:>3}s] in-flight: {:>4}, completed: {:>4} ok / {:>3} err",
                monitor_start.elapsed().as_secs(),
                in_flight_now,
                success_delta,
                failure_delta
            );
        }
    });

    while start.elapsed() < total_duration {
        let is_warmup = start.elapsed() < warmup_duration;

        // Detect transition from warmup to measurement phase
        if !is_warmup && !warmup_ended {
            warmup_ended = true;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            warmup_end_timestamp = now.as_secs_f64();
            println!();
            println!("  *** WARMUP COMPLETE ***");
            println!("  WARMUP_END_TIMESTAMP={:.6}", warmup_end_timestamp);
            println!();
        }
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let body = build_routing_request_with_prefix(
            request_id,
            &prefix_prompts,
            num_prefix_prompts,
            &model,
            seed,
        );

        let client = client.clone();
        let url = format!("{}/v1/chat/completions", frontend_url);
        let results = results.clone();
        let in_flight_clone = in_flight.clone();
        let max_in_flight_clone = max_in_flight.clone();
        let success_clone = success_count.clone();
        let failure_clone = failure_count.clone();
        let measurement_start_clone = measurement_start.clone();
        let req_id = request_id;

        // Track in-flight
        let current = in_flight_clone.fetch_add(1, Ordering::Relaxed) + 1;
        max_in_flight_clone.fetch_max(current, Ordering::Relaxed);

        tokio::spawn(async move {
            let submit_time = Instant::now();

            let response = client.post(&url).json(&body).send().await;

            let complete_time = Instant::now();
            in_flight_clone.fetch_sub(1, Ordering::Relaxed);
            drop(permit);

            // Determine success/failure and update counters
            let success = match &response {
                Ok(resp) => resp.status().is_success(),
                Err(_) => false,
            };
            if success {
                success_clone.fetch_add(1, Ordering::Relaxed);
            } else {
                failure_clone.fetch_add(1, Ordering::Relaxed);
            }

            // Only record results after warmup
            if !is_warmup {
                // Initialize measurement start on first non-warmup completion
                let mut ms_guard = measurement_start_clone.lock().await;
                if ms_guard.is_none() {
                    *ms_guard = Some(complete_time);
                }
                let measurement_base = ms_guard.unwrap();
                drop(ms_guard);

                let result = RequestResult {
                    latency: complete_time.duration_since(submit_time),
                    completion_time: complete_time.duration_since(measurement_base),
                    success,
                };

                if verbose {
                    println!(
                        "    Request {} completed in {:?} (success: {})",
                        req_id, result.latency, result.success
                    );
                }
                results.lock().await.push(result);
            }
        });

        request_id += 1;
        tokio::time::sleep(interval).await;
    }

    println!("  Submitted {} requests", request_id);

    // Wait for in-flight requests
    println!("  Waiting for in-flight requests...");
    let drain_start = Instant::now();
    while in_flight.load(Ordering::Relaxed) > 0 && drain_start.elapsed() < Duration::from_secs(30) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let remaining = in_flight.load(Ordering::Relaxed);
    if remaining > 0 {
        println!("  {} requests still in-flight after timeout", remaining);
    }

    // Stop the in-flight monitor
    monitor_handle.abort();

    warmup_end_timestamp
}

/// Publish events at a specified rate during stress test
async fn publish_events_at_rate(
    nats_client: async_nats::Client,
    namespace: String,
    component: String,
    sequences: Vec<SequenceData>,
    rate: f64,
    duration_secs: u64,
) {
    // Subject format must match Component.subject() from lib/runtime/src/component/component.rs
    let subject = format!(
        "namespace.{}.component.{}.{}",
        namespace, component, KV_EVENT_SUBJECT
    );
    let interval = Duration::from_secs_f64(1.0 / rate);
    let start = Instant::now();
    let duration = Duration::from_secs(duration_secs);
    let start_id = 1000000u64; // Start high to avoid collision with tree construction
    let mut event_id = start_id;

    // Failure tracking
    let mut publish_failures: u64 = 0;
    let mut encode_failures: u64 = 0;
    let mut last_publish_error: Option<String> = None;
    let mut last_encode_error: Option<String> = None;

    // Periodic reporting interval (every 10 seconds)
    let report_interval = Duration::from_secs(10);
    let mut last_report = Instant::now();

    while start.elapsed() < duration {
        let seq = &sequences[(event_id as usize) % sequences.len()];
        let event = seq.to_router_event(event_id);

        match encode_event_with_envelope(&event, KV_EVENT_SUBJECT) {
            Ok(data) => {
                if let Err(e) = nats_client.publish(subject.clone(), data.into()).await {
                    publish_failures += 1;
                    last_publish_error = Some(format!("{:?}", e));
                }
            }
            Err(e) => {
                encode_failures += 1;
                last_encode_error = Some(format!("{:?}", e));
            }
        }

        event_id += 1;

        // Periodic failure report
        if last_report.elapsed() >= report_interval {
            let total_attempts = event_id - start_id;
            let total_failures = publish_failures + encode_failures;
            if total_failures > 0 {
                eprintln!(
                    "  [publish_events] Periodic report: {} failures / {} attempts ({} publish, {} encode)",
                    total_failures, total_attempts, publish_failures, encode_failures
                );
            }
            last_report = Instant::now();
        }

        tokio::time::sleep(interval).await;
    }

    // Final failure report
    let total_attempts = event_id - start_id;
    let total_failures = publish_failures + encode_failures;
    if total_failures > 0 {
        eprintln!(
            "  [publish_events] Final report: {} failures / {} attempts ({:.2}% failure rate)",
            total_failures,
            total_attempts,
            (total_failures as f64 / total_attempts as f64) * 100.0
        );
        eprintln!(
            "    Publish failures: {}, Encode failures: {}",
            publish_failures, encode_failures
        );
        if let Some(ref err) = last_publish_error {
            eprintln!("    Last publish error: {}", err);
        }
        if let Some(ref err) = last_encode_error {
            eprintln!("    Last encode error: {}", err);
        }
    } else {
        println!(
            "  [publish_events] Completed: {} events published with no failures",
            total_attempts
        );
    }
}

/// Latency statistics
struct LatencyStats {
    min: Duration,
    max: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
}

impl LatencyStats {
    fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();
        let n = sorted.len();

        Some(Self {
            min: sorted[0],
            max: sorted[n - 1],
            p50: sorted[n / 2],
            p95: sorted[n * 95 / 100],
            p99: sorted[n * 99 / 100],
        })
    }
}

/// Time-bucketed latency statistics for tracking latency over time
#[derive(Debug, Clone, Serialize)]
struct TimeBucketStats {
    /// Bucket start time in seconds from measurement start
    bucket_start_sec: u64,
    /// Bucket end time in seconds
    bucket_end_sec: u64,
    /// Number of requests completed in this bucket
    count: usize,
    /// Latency stats for this bucket (in microseconds)
    latency_min_us: u64,
    latency_p50_us: u64,
    latency_p95_us: u64,
    latency_max_us: u64,
}

/// Compute per-bucket latency statistics
fn compute_time_bucket_stats(
    results: &[RequestResult],
    bucket_size_secs: u64,
) -> Vec<TimeBucketStats> {
    if results.is_empty() {
        return Vec::new();
    }

    // Find the max completion time to determine bucket count
    let max_completion = results
        .iter()
        .map(|r| r.completion_time)
        .max()
        .unwrap_or(Duration::ZERO);

    let num_buckets = (max_completion.as_secs() / bucket_size_secs) + 1;

    let mut bucket_latencies: Vec<Vec<Duration>> = vec![Vec::new(); num_buckets as usize];

    // Group latencies by completion time bucket
    for result in results {
        let bucket_idx = (result.completion_time.as_secs() / bucket_size_secs) as usize;
        if bucket_idx < bucket_latencies.len() {
            bucket_latencies[bucket_idx].push(result.latency);
        }
    }

    // Compute stats for each bucket
    bucket_latencies
        .iter()
        .enumerate()
        .filter_map(|(idx, latencies)| {
            if latencies.is_empty() {
                return None;
            }

            let stats = LatencyStats::from_durations(latencies)?;
            Some(TimeBucketStats {
                bucket_start_sec: idx as u64 * bucket_size_secs,
                bucket_end_sec: (idx as u64 + 1) * bucket_size_secs,
                count: latencies.len(),
                latency_min_us: stats.min.as_micros() as u64,
                latency_p50_us: stats.p50.as_micros() as u64,
                latency_p95_us: stats.p95.as_micros() as u64,
                latency_max_us: stats.max.as_micros() as u64,
            })
        })
        .collect()
}

/// Print time-bucket latency report
fn print_time_bucket_report(buckets: &[TimeBucketStats]) {
    if buckets.is_empty() {
        println!("  No time bucket data available");
        return;
    }

    println!(
        "  {:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Time(s)", "Count", "Min(ms)", "P50(ms)", "P95(ms)", "Max(ms)"
    );
    println!("  {}", "-".repeat(68));

    for bucket in buckets {
        println!(
            "  {:>3}-{:<4} {:>8} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
            bucket.bucket_start_sec,
            bucket.bucket_end_sec,
            bucket.count,
            bucket.latency_min_us as f64 / 1000.0,
            bucket.latency_p50_us as f64 / 1000.0,
            bucket.latency_p95_us as f64 / 1000.0,
            bucket.latency_max_us as f64 / 1000.0,
        );
    }
}

/// Stress test results
#[derive(Debug, Serialize)]
struct StressResults {
    // Configuration
    tree_size: usize,
    num_sequences: usize,
    depth: usize,
    num_workers: usize,

    // Tree construction
    tree_construction_time_ms: u64,

    // Request metrics
    requests_submitted: u64,
    requests_completed: u64,
    requests_failed: u64,

    // Latency stats (in microseconds)
    latency_min_us: u64,
    latency_p50_us: u64,
    latency_p95_us: u64,
    latency_p99_us: u64,
    latency_max_us: u64,

    // Throughput
    achieved_request_rate: f64,

    // Saturation
    max_in_flight: u64,

    // Time-bucketed latency stats for tracking latency over time
    #[serde(skip_serializing_if = "Vec::is_empty")]
    time_buckets: Vec<TimeBucketStats>,

    // Raw latency samples for detailed graphing
    #[serde(skip_serializing_if = "Vec::is_empty")]
    raw_samples: Vec<LatencySample>,
}

impl StressResults {
    fn print_report(&self) {
        println!("\n========================================");
        println!("KV Router Full Stress Test Results");
        println!("========================================\n");

        println!("Tree Construction:");
        println!("  Sequences: {}", self.num_sequences);
        println!("  Blocks: {}", self.num_sequences * self.depth);
        println!("  Time: {}ms", self.tree_construction_time_ms);
        println!();

        println!("Request Statistics:");
        println!("  Submitted: {}", self.requests_submitted);
        println!("  Completed: {}", self.requests_completed);
        println!("  Failed: {}", self.requests_failed);
        println!("  Throughput: {:.1} req/sec", self.achieved_request_rate);
        println!();

        println!("End-to-End Latency (includes HTTP overhead):");
        println!("  min:  {:>12}us", self.latency_min_us);
        println!("  p50:  {:>12}us", self.latency_p50_us);
        println!("  p95:  {:>12}us", self.latency_p95_us);
        println!("  p99:  {:>12}us", self.latency_p99_us);
        println!("  max:  {:>12}us", self.latency_max_us);
        println!();

        if !self.time_buckets.is_empty() {
            println!("Latency Over Time:");
            print_time_bucket_report(&self.time_buckets);
            println!();
        }

        println!("Saturation:");
        println!("  Max in-flight: {}", self.max_in_flight);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let num_sequences = args.tree_size / args.depth;

    println!("KV Router Full Stress Test");
    println!("==========================\n");

    // Resolve model name: use provided value or auto-detect from /v1/models
    let model = match args.model {
        Some(m) => m,
        None => {
            println!("Model Detection:");
            fetch_model_name(&args.frontend_url).await?
        }
    };

    // Tokenizer path defaults to model if not specified
    let tokenizer_path = args.tokenizer_path.as_ref().unwrap_or(&model);

    println!("Configuration:");
    println!(
        "  Tree size: {} blocks ({} sequences x {} depth)",
        args.tree_size, num_sequences, args.depth
    );
    println!("  Workers: {}", args.num_workers);
    println!(
        "  Prefix prompt ratio: {:.1}%",
        args.prefix_prompt_ratio * 100.0
    );
    println!("  Prefix prompts: {}", args.num_prefix_prompts);
    println!("  Seed: {}", args.seed);
    println!("  Request rate: {:.1} req/sec", args.request_rate);
    println!("  Event rate: {:.1} events/sec", args.event_rate);
    println!("  Duration: {}s", args.duration);
    println!("  Warmup: {}s", args.warmup);
    println!("  Concurrency: {}", args.concurrency);
    println!("  Model: {}", model);
    println!("  KV block size: {}", args.kv_block_size);
    println!("  Tokenizer: {}", tokenizer_path);
    println!("  Namespace: {}", args.namespace);
    println!("  Component: {}", args.component);
    println!(
        "  NATS subject: namespace.{}.component.{}.kv-events",
        args.namespace, args.component
    );
    if args.skip_tree_construction {
        println!("  Tree construction: SKIPPED (using external kv_stress_worker)");
    }
    println!();

    // Create HTTP client
    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("Failed to create HTTP client")?;

    // Phase 0: Load tokenizer and prompt formatter
    println!("Phase 0: Loading Tokenizer and Prompt Formatter");
    println!("  Loading tokenizer from {}...", tokenizer_path);
    let tokenizer = load_tokenizer(tokenizer_path)?;
    println!("  Tokenizer loaded successfully");

    let mut prompt_renderer =
        try_load_prompt_renderer(&model).or_else(|| try_load_prompt_renderer(tokenizer_path));

    if prompt_renderer.is_some() {
        println!("  Prompt formatter loaded from ModelDeploymentCard");
    } else {
        // Fallback to tokenizer_config.json - use the same HfTokenizerConfigJsonFormatter as the frontend
        // Try local path first, then download from HuggingFace
        let contents: Option<String> = {
            let config_path = std::path::Path::new(tokenizer_path).join("tokenizer_config.json");
            if config_path.exists() {
                std::fs::read_to_string(&config_path).ok()
            } else if !std::path::Path::new(tokenizer_path).exists() {
                // Might be a HuggingFace model ID - try to download tokenizer_config.json
                println!(
                    "  Downloading tokenizer_config.json from HuggingFace: {}...",
                    tokenizer_path
                );
                if let Ok(api) = hf_hub::api::sync::Api::new() {
                    let repo = api.model(tokenizer_path.to_string());
                    repo.get("tokenizer_config.json")
                        .ok()
                        .and_then(|path| std::fs::read_to_string(&path).ok())
                } else {
                    None
                }
            } else {
                None
            }
        };

        let try_simple_fallback = |path: &str| -> Option<PromptRenderer> {
            let config = load_tokenizer_config(path).ok()??;
            let template = config.chat_template.clone()?;
            Some(PromptRenderer::Simple(ChatTemplateRenderer::new(
                template,
                config.bos_token_str(),
                config.eos_token_str(),
            )))
        };

        if let Some(contents) = contents {
            match serde_json::from_str::<ChatTemplate>(&contents) {
                Ok(chat_template) => {
                    match PromptFormatter::from_parts(chat_template, ContextMixins::new(&[])) {
                        Ok(formatter) => {
                            println!(
                                "  Prompt formatter loaded from tokenizer_config.json (using frontend-compatible renderer)"
                            );
                            prompt_renderer = Some(PromptRenderer::Formatter(formatter));
                        }
                        Err(e) => {
                            println!("  WARNING: Failed to create prompt formatter: {}", e);
                            println!(
                                "           Using fallback Simple renderer (may not match frontend)"
                            );
                            prompt_renderer = try_simple_fallback(tokenizer_path);
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "  WARNING: Failed to parse tokenizer_config.json as ChatTemplate: {}",
                        e
                    );
                    println!("           Using fallback Simple renderer (may not match frontend)");
                    prompt_renderer = try_simple_fallback(tokenizer_path);
                }
            }
        } else {
            println!("  WARNING: No tokenizer_config.json found. Hashes may not match frontend!");
            println!(
                "           The frontend applies chat templates to /v1/chat/completions requests."
            );
        }
    }

    // Target tokens for prefix (block_size tokens per block)
    let target_prefix_tokens = (args.depth as f64 * args.prefix_prompt_ratio).round() as usize
        * args.kv_block_size as usize;

    // Phase 1: Generate prefix data with computed hashes
    println!("\nPhase 1: Generating Prefix Data");
    println!(
        "  Generating {} prefixes (~{} tokens each)...",
        args.num_prefix_prompts, target_prefix_tokens
    );

    let prefix_data = generate_prefix_data(
        args.num_prefix_prompts,
        target_prefix_tokens,
        &tokenizer,
        args.kv_block_size,
        prompt_renderer.as_ref(),
        true, // show_progress
    )?;

    // Print prefix stats
    for (i, prefix) in prefix_data.iter().enumerate() {
        if i < 3 || args.verbose {
            // Show a preview of the formatted text (first 80 chars, escape newlines)
            let preview: String = prefix
                .formatted_text
                .chars()
                .take(80)
                .map(|c| if c == '\n' { ' ' } else { c })
                .collect();
            println!(
                "    Prefix {}: {} tokens, {} blocks, first hash: {:016x}",
                i,
                prefix.token_ids.len(),
                prefix.num_blocks(),
                prefix.local_hashes.first().map(|h| h.0).unwrap_or(0)
            );
            if args.verbose {
                println!("      Formatted: {}...", preview);
            }
        }
    }
    if prefix_data.len() > 3 && !args.verbose {
        println!("    ... ({} more prefixes)", prefix_data.len() - 3);
    }

    // Show first prefix's formatted text sample if chat template was applied
    if prompt_renderer.is_some() && !prefix_data.is_empty() {
        let sample: String = prefix_data[0].formatted_text.chars().take(200).collect();
        println!("  Sample formatted prefix (first 200 chars):");
        for line in sample.lines().take(5) {
            println!("    | {}", line);
        }
        if prefix_data[0].formatted_text.len() > 200 {
            println!("    | ...");
        }
    }

    // Extract prefix texts for HTTP requests and request-aligned hashing
    let prefix_prompts: Vec<String> = prefix_data.iter().map(|p| p.text.clone()).collect();

    // Phase 2: Discover workers and generate sequences
    println!("\nPhase 2: Discover Workers & Generate Sequences");

    // Discover actual worker IDs from the frontend
    let discovered_worker_ids = discover_worker_ids(&args.frontend_url).await?;

    if discovered_worker_ids.len() != args.num_workers {
        println!(
            "  NOTE: Discovered {} workers but --num-workers was set to {}. Using discovered workers.",
            discovered_worker_ids.len(),
            args.num_workers
        );
    }

    println!(
        "  Generating {} sequences with shared prefixes...",
        num_sequences
    );

    let sequences = generate_sequences_for_requests(
        num_sequences,
        &discovered_worker_ids,
        &prefix_prompts,
        args.num_prefix_prompts,
        args.kv_block_size,
        &tokenizer,
        prompt_renderer.as_ref(),
        args.seed,
        true, // show_progress
    )?;
    println!(
        "  Generated {} sequences distributed across {} workers",
        sequences.len(),
        discovered_worker_ids.len()
    );

    // Phase 3: Build tree via NATS (unless skipped)
    let tree_construction_time = if args.skip_tree_construction {
        println!("\nPhase 3: Tree Construction via NATS - SKIPPED");
        println!("  Using external kv_stress_worker for tree construction");
        Duration::ZERO
    } else {
        println!("\nPhase 3: Tree Construction via NATS");
        // Connect to NATS
        println!("  Connecting to NATS at {}...", args.nats_url);
        let nats_client = async_nats::connect(&args.nats_url)
            .await
            .context("Failed to connect to NATS")?;
        println!("  Connected to NATS");

        build_tree_via_nats(
            &nats_client,
            &args.namespace,
            &args.component,
            &sequences,
            args.verbose,
        )
        .await?
    };

    // Phase 4: Stress Test
    println!("\nPhase 4: Stress Test");

    println!(
        "  Using {} prefix prompts for HTTP requests (hashes pre-computed)...",
        prefix_prompts.len()
    );
    let prefix_prompts = Arc::new(prefix_prompts);

    let results = Arc::new(Mutex::new(Vec::new()));
    let in_flight = Arc::new(AtomicU64::new(0));
    let max_in_flight = Arc::new(AtomicU64::new(0));

    // Spawn event publisher if rate > 0 and tree construction wasn't skipped
    let event_handle = if args.event_rate > 0.0 && !args.skip_tree_construction {
        // Need to connect to NATS for ongoing events
        let nats = async_nats::connect(&args.nats_url)
            .await
            .context("Failed to connect to NATS for event publishing")?;
        let ns = args.namespace.clone();
        let comp = args.component.clone();
        let seqs = sequences.clone();
        let rate = args.event_rate;
        let dur = args.duration + args.warmup;
        Some(tokio::spawn(async move {
            publish_events_at_rate(nats, ns, comp, seqs, rate, dur).await;
        }))
    } else {
        None
    };

    // Run request generator
    // During warmup, requests populate the cache via mocker engine.
    // During measurement, requests with the same prefixes get cache hits.
    let warmup_end_ts = send_requests_at_rate(
        http_client,
        args.frontend_url.clone(),
        prefix_prompts,
        args.num_prefix_prompts,
        model.clone(),
        args.seed,
        args.request_rate,
        args.duration,
        args.warmup,
        args.concurrency,
        results.clone(),
        in_flight.clone(),
        max_in_flight.clone(),
        args.verbose,
    )
    .await;

    // Print the timestamp again at the end for easy copy-paste
    println!();
    println!("To filter FE logs for post-warmup only:");
    println!(
        "  python analyze_frontend_log.py frontend.log --after-warmup {:.6}",
        warmup_end_ts
    );

    // Wait for event publisher
    if let Some(h) = event_handle {
        let _ = h.await;
    }

    // Collect results
    let results = results.lock().await;
    let latencies: Vec<Duration> = results.iter().map(|r| r.latency).collect();
    let successful_results: Vec<&RequestResult> = results.iter().filter(|r| r.success).collect();
    let failed_count = results.len() - successful_results.len();

    // Compute actual measurement duration from completion times of successful requests.
    // This accounts for the drain phase where in-flight requests complete after submission stops.
    let actual_duration_secs = successful_results
        .iter()
        .map(|r| r.completion_time.as_secs_f64())
        .fold(0.0_f64, |a, b| a.max(b))
        .max(1.0); // Avoid division by zero

    let stats = LatencyStats::from_durations(&latencies);

    // Compute time-bucketed stats for latency-over-time tracking
    let time_buckets = if args.bucket_size > 0 {
        compute_time_bucket_stats(&results, args.bucket_size)
    } else {
        Vec::new()
    };

    // Collect raw latency samples if requested
    let raw_samples: Vec<LatencySample> = if args.include_raw_samples {
        results
            .iter()
            .map(|r| LatencySample {
                latency_us: r.latency.as_micros() as u64,
                completion_time_ms: r.completion_time.as_millis() as u64,
                success: r.success,
            })
            .collect()
    } else {
        Vec::new()
    };

    let stress_results = StressResults {
        tree_size: args.tree_size,
        num_sequences,
        depth: args.depth,
        num_workers: discovered_worker_ids.len(),
        tree_construction_time_ms: tree_construction_time.as_millis() as u64,
        requests_submitted: results.len() as u64,
        requests_completed: successful_results.len() as u64,
        requests_failed: failed_count as u64,
        latency_min_us: stats
            .as_ref()
            .map(|s| s.min.as_micros() as u64)
            .unwrap_or(0),
        latency_p50_us: stats
            .as_ref()
            .map(|s| s.p50.as_micros() as u64)
            .unwrap_or(0),
        latency_p95_us: stats
            .as_ref()
            .map(|s| s.p95.as_micros() as u64)
            .unwrap_or(0),
        latency_p99_us: stats
            .as_ref()
            .map(|s| s.p99.as_micros() as u64)
            .unwrap_or(0),
        latency_max_us: stats
            .as_ref()
            .map(|s| s.max.as_micros() as u64)
            .unwrap_or(0),
        achieved_request_rate: successful_results.len() as f64 / actual_duration_secs,
        max_in_flight: max_in_flight.load(Ordering::Relaxed),
        time_buckets,
        raw_samples,
    };

    stress_results.print_report();

    // Write JSON output if requested
    if let Some(output_path) = args.output {
        let json = serde_json::to_string_pretty(&stress_results)?;
        std::fs::write(&output_path, json)?;
        println!("\nResults written to: {}", output_path);
    }

    Ok(())
}
