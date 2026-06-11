// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline-replay driver for blended (non-prefix) KV reuse simulation.
//!
//! Reads a scenario JSON (engine knobs + explicit token-level requests with
//! optional `SemanticReusePlan`s), runs the SGLang mocker personality through
//! the offline replay harness, and emits a JSON report: the standard summary,
//! a semantic aggregate, and per-request records.
//!
//! Usage: `cargo run --example semantic_replay -- scenario.json > report.json`

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use dynamo_mocker::common::protocols::{DirectRequest, EngineType, MockEngineArgsBuilder};
use dynamo_mocker::common::semantic::{
    FallbackReason, SemanticOutcome, SemanticReusePlan, SemanticSimConfig,
};
use dynamo_mocker::replay::ReplayRouterMode;
use dynamo_mocker::replay::simulate_trace_requests_with_router_mode_and_options;

#[derive(Deserialize)]
struct Scenario {
    #[serde(default = "default_num_workers")]
    num_workers: usize,
    #[serde(default)]
    router_mode: RouterModeArg,
    block_size: usize,
    num_gpu_blocks: usize,
    #[serde(default = "default_speedup")]
    speedup_ratio: f64,
    #[serde(default)]
    max_prefill_tokens: Option<usize>,
    #[serde(default)]
    chunked_prefill_size: Option<usize>,
    #[serde(default)]
    semantic: Option<SemanticSimConfig>,
    requests: Vec<ScenarioRequest>,
}

fn default_num_workers() -> usize {
    1
}
fn default_speedup() -> f64 {
    1.0
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum RouterModeArg {
    #[default]
    RoundRobin,
    KvRouter,
}

#[derive(Deserialize)]
struct ScenarioRequest {
    uuid: Uuid,
    tokens: Vec<u32>,
    max_output_tokens: usize,
    arrival_ms: f64,
    #[serde(default)]
    plan: Option<SemanticReusePlan>,
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: semantic_replay <scenario.json>")?;
    let scenario: Scenario = serde_json::from_str(
        &std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?,
    )
    .context("parsing scenario JSON")?;

    let mut plans: HashMap<Uuid, SemanticReusePlan> = HashMap::new();
    let mut requests = Vec::with_capacity(scenario.requests.len());
    for r in &scenario.requests {
        if let Some(plan) = &r.plan {
            plans.insert(r.uuid, plan.clone());
        }
        requests.push(DirectRequest {
            tokens: r.tokens.clone(),
            max_output_tokens: r.max_output_tokens,
            uuid: Some(r.uuid),
            dp_rank: 0,
            arrival_timestamp_ms: Some(r.arrival_ms),
            ..Default::default()
        });
    }

    let semantic_sim = scenario.semantic.map(|mut sem| {
        sem.plans = Arc::new(plans);
        sem
    });

    let mut builder = MockEngineArgsBuilder::default()
        .engine_type(EngineType::Sglang)
        .block_size(scenario.block_size)
        .num_gpu_blocks(scenario.num_gpu_blocks)
        .speedup_ratio(scenario.speedup_ratio)
        .semantic_sim(semantic_sim);
    if scenario.max_prefill_tokens.is_some() || scenario.chunked_prefill_size.is_some() {
        builder = builder.sglang(Some(dynamo_mocker::common::protocols::SglangArgs {
            schedule_policy: None,
            page_size: None,
            max_prefill_tokens: scenario.max_prefill_tokens,
            chunked_prefill_size: scenario.chunked_prefill_size,
            clip_max_new_tokens: None,
            schedule_conservativeness: None,
        }));
    }
    let args = builder.build().context("building MockEngineArgs")?;

    let router_mode = match scenario.router_mode {
        RouterModeArg::RoundRobin => ReplayRouterMode::RoundRobin,
        RouterModeArg::KvRouter => ReplayRouterMode::KvRouter,
    };

    let report = simulate_trace_requests_with_router_mode_and_options(
        args,
        None,
        None,
        requests,
        scenario.num_workers,
        1.0,
        router_mode,
        true,
        None,
    )?;

    // Aggregate the semantic outcomes from per-request records so the
    // report's custom summary serializer stays untouched.
    let mut copied = 0usize;
    let mut repaired = 0usize;
    let mut halo = 0usize;
    let mut accepted = 0usize;
    let mut fallbacks: HashMap<String, usize> = HashMap::new();
    let mut accepted_ttfts: Vec<f64> = Vec::new();
    let mut other_ttfts: Vec<f64> = Vec::new();
    for rec in &report.per_request {
        match rec.semantic_outcome {
            Some(SemanticOutcome::Accepted {
                copied_tokens,
                repaired_tokens,
                recomputed_halo_tokens,
            }) => {
                copied += copied_tokens;
                repaired += repaired_tokens;
                halo += recomputed_halo_tokens;
                accepted += 1;
                if let Some(t) = rec.ttft_ms {
                    accepted_ttfts.push(t);
                }
            }
            Some(SemanticOutcome::Fallback { reason }) => {
                *fallbacks.entry(fallback_name(reason)).or_default() += 1;
                if let Some(t) = rec.ttft_ms {
                    other_ttfts.push(t);
                }
            }
            None => {
                if let Some(t) = rec.ttft_ms {
                    other_ttfts.push(t);
                }
            }
        }
    }

    let out = json!({
        "summary": &report,
        "semantic": {
            "accepted_plans": accepted,
            "copied_tokens": copied,
            "repaired_tokens": repaired,
            "recomputed_halo_tokens": halo,
            "fallbacks": fallbacks,
            "ttft_ms_mean_accepted": mean(&accepted_ttfts),
            "ttft_ms_mean_other": mean(&other_ttfts),
        },
        "per_request": &report.per_request,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn fallback_name(reason: FallbackReason) -> String {
    serde_json::to_value(reason)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| format!("{reason:?}"))
}
