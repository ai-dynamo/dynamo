// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs the same requests through a sidecar connected to a live native engine
//! and through one connected to the matching mocker, then compares what the
//! sidecar returns. Token values, logprob values and timing differ by design, so
//! each scenario compares only counts, finish reasons, field presence and types.

// Each mocker's test uses a subset of these helpers.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{
    AsyncEngineContext, EngineConfig, ErrorType, GenerateContext, LLMEngine, LLMEngineOutput,
    OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::StreamExt;
use serde_json::Value;

const TIMEOUT: Duration = Duration::from_secs(60);
const PROMPT: [u32; 6] = [9707, 11, 1246, 525, 498, 30];

/// Environment set by `lib/mocker/servers/parity/run.py`.
pub struct LiveEngine {
    pub endpoint: String,
    pub model: String,
    pub version: String,
}

impl LiveEngine {
    pub fn from_env() -> Self {
        let var = |name: &str| {
            std::env::var(name).unwrap_or_else(|_| panic!("{name} is required; see parity/run.py"))
        };
        Self {
            endpoint: var("MOCKER_PARITY_ENDPOINT"),
            model: var("MOCKER_PARITY_MODEL"),
            version: var("MOCKER_PARITY_ENGINE_VERSION"),
        }
    }
}

impl LiveEngine {
    pub fn assert_version(&self, mocker_version: &str) {
        assert_eq!(
            mocker_version, self.version,
            "update ENGINE_VERSION after checking the mocker against this engine"
        );
    }
}

/// The model name a request must carry for this registration.
pub fn served(config: &EngineConfig) -> String {
    config
        .served_model_name
        .clone()
        .unwrap_or_else(|| config.model.clone())
}

/// Mocker engine arguments matching what the live engine registered.
pub fn mocker_engine_args(backend: &str, config: &EngineConfig) -> serde_json::Value {
    let llm = config.llm.as_ref().expect("live engine registers an LLM");
    serde_json::json!({
        "dp_size": 1,
        "engine": {
            "backend": backend,
            "block_size": llm.kv_cache_block_size.expect("live engine reports a block size"),
            "num_gpu_blocks": llm.total_kv_blocks.expect("live engine reports its KV blocks"),
            "max_num_seqs": llm.max_num_seqs,
            "max_num_batched_tokens": llm.max_num_batched_tokens,
            "max_model_len": llm.context_length,
            "speedup_ratio": 1.0
        }
    })
}

pub fn request(model: &str, max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(PROMPT.to_vec())
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions::default())
        .build()
        .unwrap()
}

fn context(id: &str) -> Arc<dyn AsyncEngineContext> {
    Context::with_id_and_metadata((), id.to_string(), BTreeMap::new()).context()
}

/// A response reduced to what both backends must agree on.
#[derive(Debug, PartialEq)]
pub struct Completed {
    tokens: usize,
    finish_reason: Option<String>,
    stop_reason: Option<String>,
    usage: Option<Value>,
    logprobs: Vec<Option<(usize, bool)>>,
    handoff: Option<Value>,
}

pub type Outcome = Result<Completed, ErrorType>;

async fn generate<E: LLMEngine>(
    engine: &E,
    request: PreprocessedRequest,
    ctx: Arc<dyn AsyncEngineContext>,
    cancel_after_first: bool,
) -> Result<Vec<LLMEngineOutput>, ErrorType> {
    tokio::time::timeout(TIMEOUT, async {
        let mut stream = engine
            .generate(request, GenerateContext::new(Arc::clone(&ctx), None))
            .await
            .map_err(|error| error.error_type())?;
        let mut outputs = Vec::new();
        while let Some(item) = stream.next().await {
            let output = item.map_err(|error| error.error_type())?;
            if cancel_after_first && outputs.is_empty() {
                ctx.stop_generating();
                engine.abort(Arc::clone(&ctx)).await;
            }
            outputs.push(output);
        }
        Ok(outputs)
    })
    .await
    .expect("request did not finish")
}

fn outcome(result: Result<Vec<LLMEngineOutput>, ErrorType>) -> Outcome {
    let outputs = result?;
    let terminal = outputs.last().expect("a terminal chunk");
    let json = |value: Option<Value>| value.filter(|value| !value.is_null());
    Ok(Completed {
        tokens: outputs.iter().map(|output| output.token_ids.len()).sum(),
        finish_reason: json(
            terminal
                .finish_reason
                .as_ref()
                .map(|r| serde_json::to_value(r).unwrap()),
        )
        .map(|value| shape_name(&value)),
        stop_reason: json(
            terminal
                .stop_reason
                .as_ref()
                .map(|r| serde_json::to_value(r).unwrap()),
        )
        .map(|value| shape(&value).to_string()),
        usage: terminal
            .completion_usage
            .as_ref()
            .map(|usage| usage_shape(serde_json::to_value(usage).unwrap())),
        logprobs: outputs
            .iter()
            .flat_map(|output| {
                (0..output.token_ids.len()).map(move |index| {
                    output.log_probs.as_ref()?.get(index)?;
                    let top = output
                        .top_logprobs
                        .as_ref()
                        .and_then(|top| top.get(index))
                        .map_or(&[][..], Vec::as_slice);
                    let chosen = output.token_ids[index];
                    Some((top.len(), top.iter().any(|entry| entry.token_id == chosen)))
                })
            })
            .collect(),
        handoff: outputs
            .iter()
            .find_map(|output| output.disaggregated_params.as_deref())
            .map(|params| without_mocker_keys(shape(params))),
    })
}

/// `Error(message)` serializes as an object; only its kind is comparable.
fn shape_name(value: &Value) -> String {
    match value {
        Value::String(name) => name.clone(),
        Value::Object(fields) => fields.keys().cloned().collect::<Vec<_>>().join(","),
        other => other.to_string(),
    }
}

/// Usage keeps exact counts and the shape of the details.
fn usage_shape(mut usage: Value) -> Value {
    if let Value::Object(fields) = &mut usage {
        for (key, value) in fields.iter_mut() {
            if !matches!(
                key.as_str(),
                "prompt_tokens" | "completion_tokens" | "total_tokens"
            ) {
                *value = shape(value);
            }
        }
        fields.retain(|_, value| !value.is_null());
    }
    usage
}

/// Replaces leaf values with their JSON type, keeping object keys.
pub fn shape(value: &Value) -> Value {
    match value {
        Value::Null => Value::Null,
        Value::Bool(_) => "bool".into(),
        Value::Number(_) => "number".into(),
        Value::String(_) => "string".into(),
        Value::Array(items) => {
            let shapes: BTreeSet<String> =
                items.iter().map(|item| shape(item).to_string()).collect();
            Value::Array(
                shapes
                    .into_iter()
                    .map(|item| serde_json::from_str(&item).unwrap())
                    .collect(),
            )
        }
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key.clone(), shape(value)))
                .collect(),
        ),
    }
}

/// The mockers add `mocker_*` sentinels to prove the sidecar forwards opaque
/// handoff fields; real engines do not send them.
fn without_mocker_keys(mut value: Value) -> Value {
    if let Value::Object(fields) = &mut value {
        fields.retain(|key, _| !key.starts_with("mocker_"));
        for field in fields.values_mut() {
            *field = without_mocker_keys(field.take());
        }
    }
    value
}

/// Registration fields the router, planner and frontend read.
pub fn registration(config: &EngineConfig) -> Value {
    let llm = config.llm.as_ref().expect("an LLM registration");
    serde_json::json!({
        "served_model_name": config.served_model_name.is_some(),
        "model_aliases": config.model_aliases.len(),
        "runtime_data": config.runtime_data.keys().collect::<BTreeSet<_>>(),
        "context_length": llm.context_length,
        "kv_cache_block_size": llm.kv_cache_block_size,
        "total_kv_blocks": llm.total_kv_blocks.is_some(),
        "max_num_seqs": llm.max_num_seqs,
        "max_num_batched_tokens": llm.max_num_batched_tokens,
        "max_gpu_lora_count": llm.max_gpu_lora_count,
        "data_parallel_size": llm.data_parallel_size,
        "data_parallel_start_rank": llm.data_parallel_start_rank,
        "enable_eagle": llm.enable_eagle,
        "bootstrap_host": llm.bootstrap_host.is_some(),
        "bootstrap_port": llm.bootstrap_port.is_some(),
    })
}

/// Aggregated scenarios: length, stop tokens, logprobs, abort and rejection.
pub async fn aggregated<E: LLMEngine>(
    model: &str,
    engine: &E,
    context_length: u32,
) -> Vec<(&'static str, Outcome)> {
    let mut outcomes = Vec::new();

    let length = generate(engine, request(model, 8), context("parity-length"), false).await;
    let tokens: Vec<u32> = length
        .as_ref()
        .expect("a length-limited request succeeds")
        .iter()
        .flat_map(|output| output.token_ids.clone())
        .collect();
    outcomes.push(("length", outcome(length)));

    // Both backends stop at position 2, so their token counts are comparable.
    let stop_token = tokens[2];
    assert!(
        !tokens[..2].contains(&stop_token),
        "the backend repeats token {stop_token} before position 2; change PROMPT"
    );
    for (name, visible) in [("stop_token_hidden", false), ("stop_token_visible", true)] {
        // SGLang ignores stop tokens along with EOS when `ignore_eos` is set.
        let mut stop = request(model, 8);
        stop.stop_conditions.ignore_eos = None;
        if visible {
            stop.stop_conditions.stop_token_ids_visible = Some(vec![stop_token]);
        } else {
            stop.stop_conditions.stop_token_ids_hidden = Some(vec![stop_token]);
        }
        // Reuse the probe's ID: the mocker derives its tokens from it.
        let stopped = outcome(generate(engine, stop, context("parity-length"), false).await);
        assert!(
            matches!(&stopped, Ok(Completed { finish_reason: Some(reason), .. }) if reason == "stop"),
            "{name} did not stop on token {stop_token}: {stopped:?}"
        );
        outcomes.push((name, stopped));
    }

    let mut logprobs = request(model, 4);
    logprobs.output_options.logprobs = Some(2);
    outcomes.push((
        "logprobs",
        outcome(generate(engine, logprobs, context("parity-logprobs"), false).await),
    ));

    let long = request(model, context_length - PROMPT.len() as u32);
    outcomes.push((
        "abort",
        outcome(generate(engine, long, context("parity-abort"), true).await),
    ));

    let mut too_long = request(model, 1);
    too_long.token_ids = vec![PROMPT[0]; context_length as usize + 1].into();
    outcomes.push((
        "too_long",
        outcome(generate(engine, too_long, context("parity-too-long"), false).await),
    ));

    outcomes.push((
        "after_errors",
        outcome(generate(engine, request(model, 2), context("parity-after"), false).await),
    ));
    for (name, outcome) in &outcomes {
        if !matches!(*name, "abort" | "too_long") {
            assert!(outcome.is_ok(), "{name} failed: {outcome:?}");
        }
    }
    outcomes
}

/// A prefill-only request and the handoff it returns.
pub async fn prefill<E: LLMEngine>(model: &str, engine: &E) -> Outcome {
    let outcome =
        outcome(generate(engine, request(model, 8), context("parity-prefill"), false).await);
    assert!(
        matches!(
            &outcome,
            Ok(Completed {
                handoff: Some(_),
                ..
            })
        ),
        "prefill returned no handoff: {outcome:?}"
    );
    outcome
}

/// Fails with a readable diff of every scenario that differs.
pub fn assert_same(engine: &[(&str, Outcome)], mocker: &[(&str, Outcome)]) {
    let differences: Vec<String> = engine
        .iter()
        .zip(mocker)
        .filter(|(engine, mocker)| engine != mocker)
        .map(|((name, engine), (_, mocker))| {
            format!("{name}:\n  engine: {engine:?}\n  mocker: {mocker:?}")
        })
        .collect();
    assert!(
        differences.is_empty(),
        "mocker differs from the engine:\n{}",
        differences.join("\n")
    );
}
