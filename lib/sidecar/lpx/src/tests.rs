// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixtures shared by the test modules in `tests/`, which are grouped by the
//! surface they cover.

use std::num::NonZeroUsize;
use std::time::Duration;

use dynamo_backend_common::{
    BackendError, DisaggregationMode, DynamoError, ErrorType, GenerateContext, LLMEngine,
    LLMEngineOutput, OutputOptions, PrefillResult, PreprocessedRequest, PromptTokensDetails,
    SamplingOptions, StopConditions, WorkerConfig,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::StreamExt;
use serde_json::{Value, json};

use crate::engine::LpxSidecarEngine;
use crate::model::DiscoveredEngine;

mod convert;
mod engine;
mod fake;
mod handoff;
mod model;

use fake::{FakeEngine, FakeServer};

const AGG: DisaggregationMode = DisaggregationMode::Aggregated;
const DECODE: DisaggregationMode = DisaggregationMode::Decode;

/// The served name the fake engine reports.
const SERVED: &str = "llama32-1b";
const MODEL_PATH: &str = "/models/Llama-3.2-1B-Instruct";

/// An aggregated request with every field the sidecar accepts set, stop
/// strings included.
fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model(SERVED.to_string())
        .token_ids(vec![128000, 9906, 1917])
        .stop_conditions(StopConditions {
            max_tokens: Some(16),
            min_tokens: Some(1),
            stop: Some(vec!["done".to_string()]),
            stop_token_ids: Some(vec![128009, 7]),
            stop_token_ids_hidden: Some(vec![128009, 128001]),
            ignore_eos: Some(false),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.5),
            top_p: Some(0.75),
            top_k: Some(4),
            min_p: Some(0.25),
            seed: Some(123),
            presence_penalty: Some(0.125),
            frequency_penalty: Some(0.375),
            repetition_penalty: Some(1.5),
            ..Default::default()
        })
        .output_options(OutputOptions::default())
        .build()
        .expect("request")
}

/// The vLLM 0.30 `kv_transfer_params` a prefill worker hands to decode
/// (golden vector G1).
fn vllm_params() -> Value {
    json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_block_ids": [[3, 4]],
        "remote_engine_id": "engine",
        "remote_request_id": "request",
        "remote_host": "10.0.0.2",
        "remote_port": 5600,
        "tp_size": 1,
        "pp_size": 1,
        "dcp_size": 1,
        "remote_num_tokens": 40,
        "remote_blocks_expiry_time": null,
        "transfer_mode": "pull"
    })
}

/// The wrapped shape the Python `dynamo.vllm` prefill worker emits.
fn wrapped(params: Value) -> Value {
    json!({ "kv_transfer_params": params })
}

fn prefill_result(disaggregated_params: Value, cached_tokens: Option<u32>) -> PrefillResult {
    PrefillResult {
        disaggregated_params,
        prompt_tokens_details: cached_tokens.map(|cached| PromptTokensDetails {
            cached_tokens: Some(cached),
            ..Default::default()
        }),
    }
}

/// A decode leg carrying the G1 handoff in the wrapped shape.
fn decode_request() -> PreprocessedRequest {
    let mut request = request();
    request.prefill_result = Some(prefill_result(wrapped(vllm_params()), Some(1)));
    request
}

/// One connection, retried at an interval the CLI cannot express.
fn transport() -> GrpcTransportConfig {
    GrpcTransportConfig {
        connections: NonZeroUsize::MIN,
        retry_interval: Duration::from_millis(10),
        startup_deadline: Duration::from_secs(5),
        ..Default::default()
    }
}

/// The fake engine in `mode`, as the sidecar identifies it.
fn discovered(mode: DisaggregationMode) -> DiscoveredEngine {
    DiscoveredEngine::check(fake::server_info(mode), fake::model_info(), mode)
        .expect("the fake is accepted")
}

/// An engine in `mode` for `server` that skipped the bootstrap, as if it
/// had passed, and started.
async fn started(server: &FakeServer, mode: DisaggregationMode) -> LpxSidecarEngine {
    let discovered = discovered(mode);
    let engine_config = discovered.engine_config();
    let engine = LpxSidecarEngine::new(
        GrpcEndpoint::parse(&server.endpoint, "--grpc-endpoint").expect("valid test endpoint"),
        transport(),
        discovered,
        engine_config,
    );
    engine.start(0).await.expect("start");
    engine
}

/// Runs one request to the end of its stream.
async fn collect(
    engine: &LpxSidecarEngine,
    request: PreprocessedRequest,
) -> Vec<Result<LLMEngineOutput, DynamoError>> {
    let context = dynamo_backend_common::testing::mock_context();
    engine
        .generate(request, GenerateContext::new(context, None))
        .await
        .expect("generate opens a stream")
        .collect()
        .await
}

/// Command-line arguments for a sidecar pointed at `endpoint`.
fn argv(endpoint: &str, extra: &[&str]) -> Vec<String> {
    let mut argv = vec![
        "dynamo-lpx-sidecar".to_string(),
        "--grpc-endpoint".to_string(),
        endpoint.to_string(),
        "--model-path".to_string(),
        MODEL_PATH.to_string(),
        "--grpc-retry-interval-secs".to_string(),
        "1".to_string(),
    ];
    argv.extend(extra.iter().map(|arg| arg.to_string()));
    argv
}

/// Parses `argv` as the executable would and runs the bootstrap it defers.
async fn discover(argv: Vec<String>) -> Result<(LpxSidecarEngine, WorkerConfig), DynamoError> {
    let Ok((_, bootstrap)) = LpxSidecarEngine::try_from_args_async(argv) else {
        panic!("the arguments must parse");
    };
    bootstrap.await
}

fn assert_backend(error: &DynamoError, expected: BackendError) {
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(expected),
        "unexpected error: {error}"
    );
}
