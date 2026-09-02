// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, ValueEnum};
use dynamo_backend_common::{
    BackendError, CommonArgs, DisaggregationMode, DynamoError, WorkerConfig,
};

use super::backend_error;
use super::queue::TokenMode;

pub const DEFAULT_MODEL_NAME: &str = "mock-specdec-model";

#[derive(Parser, Debug)]
#[command(
    name = "mock-target-specdec-worker",
    about = "CPU-only Dynamo speculative-decoding target worker"
)]
pub(crate) struct TargetArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Public model name advertised to the frontend.
    #[arg(long, default_value = DEFAULT_MODEL_NAME)]
    pub model_name: String,

    /// Local tokenizer directory used by the frontend model card.
    #[arg(long, default_value = "")]
    pub model_path: String,

    /// Discovery endpoint of the speculative draft worker.
    #[arg(long, default_value = "specdec/draft/generate")]
    pub draft_endpoint: String,

    /// Maximum context length advertised by the mock worker.
    #[arg(long, default_value_t = 4096)]
    pub context_length: u32,

    /// Simulated target-prefill duration.
    #[arg(long, default_value_t = 5)]
    pub target_prefill_ms: u64,

    /// Delay between accepted target tokens.
    #[arg(long, default_value_t = 1)]
    pub target_token_interval_ms: u64,

    /// Maximum proposal prefix accepted by fake verification.
    #[arg(long, default_value_t = 8)]
    pub accepted_proposal_tokens: u32,

    /// Fail after draft START when the prompt contains this token.
    #[arg(long)]
    pub fail_after_draft_start_prompt_token: Option<u32>,

    /// ZMQ send and receive high-water mark for target DEALER sockets.
    #[arg(long, default_value_t = 64)]
    pub transport_hwm: i32,

    /// Bounded target-side transport send queue.
    #[arg(long, default_value_t = 64)]
    pub transport_queue_capacity: usize,

    /// Bounded target-side per-request response queue.
    #[arg(long, default_value_t = 16)]
    pub session_queue_capacity: usize,

    /// Maximum target-side requests admitted across the draft connection pool.
    #[arg(long, default_value_t = 64)]
    pub max_inflight_sessions: usize,

    /// Maximum simultaneously leased DEALER connections in the draft pool.
    #[arg(long, default_value_t = 8)]
    pub draft_connection_pool_size: usize,

    /// HELLO handshake timeout.
    #[arg(long, default_value_t = 2_000)]
    pub handshake_timeout_ms: u64,

    /// START acknowledgement timeout.
    #[arg(long, default_value_t = 2_000)]
    pub start_timeout_ms: u64,

    /// Proposal inactivity timeout.
    #[arg(long, default_value_t = 2_000)]
    pub inactivity_timeout_ms: u64,

    /// CLEANUP acknowledgement timeout.
    #[arg(long, default_value_t = 2_000)]
    pub cleanup_timeout_ms: u64,
}

#[derive(Parser, Debug)]
#[command(
    name = "mock-draft-spec-dec-worker",
    about = "CPU-only Dynamo speculative-decoding draft worker"
)]
pub(crate) struct DraftArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Logical model name used to compose this draft with a target.
    #[arg(long, default_value = DEFAULT_MODEL_NAME)]
    pub model_name: String,

    /// ZMQ ROUTER bind address. Use 0.0.0.0 in containers.
    #[arg(long, default_value = "tcp://0.0.0.0:5560")]
    pub draft_bind_address: String,

    /// Reachable ZMQ address advertised to targets.
    #[arg(long, default_value = "tcp://127.0.0.1:5560")]
    pub draft_advertise_address: String,

    /// Maximum disconnect-to-session-cleanup interval advertised to targets.
    #[arg(long, default_value_t = 1_000)]
    pub orphan_cleanup_timeout_ms: u32,

    /// ZMQ send and receive high-water mark.
    #[arg(long, default_value_t = 64)]
    pub transport_hwm: i32,

    /// Bounded draft transport send queue.
    #[arg(long, default_value_t = 64)]
    pub transport_queue_capacity: usize,

    /// Maximum queued fake-inference jobs.
    #[arg(long, default_value_t = 32)]
    pub inference_queue_capacity: usize,

    /// Maximum concurrently running fake-inference jobs.
    #[arg(long, default_value_t = 4)]
    pub inference_concurrency: usize,

    /// Bounded per-job phase/token event queue.
    #[arg(long, default_value_t = 8)]
    pub inference_output_capacity: usize,

    /// Simulated draft-prefill duration.
    #[arg(long, default_value_t = 5)]
    pub draft_prefill_ms: u64,

    /// Delay between deterministic proposal tokens.
    #[arg(long, default_value_t = 1)]
    pub draft_token_interval_ms: u64,

    /// Deterministic proposal-token mode.
    #[arg(long, value_enum, default_value_t = TokenModeArg::Echo)]
    pub draft_token_mode: TokenModeArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum TokenModeArg {
    Echo,
    Counter,
}

impl From<TokenModeArg> for TokenMode {
    fn from(value: TokenModeArg) -> Self {
        match value {
            TokenModeArg::Echo => Self::Echo,
            TokenModeArg::Counter => Self::Counter,
        }
    }
}

pub(crate) fn parse_target(argv: Option<Vec<String>>) -> Result<TargetArgs, DynamoError> {
    let args = match argv {
        Some(argv) => {
            TargetArgs::try_parse_from(argv).map_err(|error| invalid_argument(error.to_string()))?
        }
        None => TargetArgs::parse(),
    };
    validate_common(&args.common)?;
    if args.model_name.is_empty() {
        return Err(invalid_argument("model name must not be empty"));
    }
    if args.context_length == 0 {
        return Err(invalid_argument("context length must be positive"));
    }
    if args.accepted_proposal_tokens == 0
        || args.transport_hwm <= 0
        || args.transport_queue_capacity == 0
        || args.session_queue_capacity == 0
        || args.max_inflight_sessions == 0
        || args.draft_connection_pool_size == 0
        || args.handshake_timeout_ms == 0
        || args.start_timeout_ms == 0
        || args.inactivity_timeout_ms == 0
        || args.cleanup_timeout_ms == 0
    {
        return Err(invalid_argument(
            "target transport bounds, timeouts, and accepted proposal count must be positive",
        ));
    }
    if args.transport_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
        || args.session_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
        || args.max_inflight_sessions > tokio::sync::Semaphore::MAX_PERMITS
        || args.draft_connection_pool_size > tokio::sync::Semaphore::MAX_PERMITS
    {
        return Err(invalid_argument(
            "target transport capacity exceeds the Tokio primitive limit",
        ));
    }
    if args.draft_connection_pool_size > args.max_inflight_sessions {
        return Err(invalid_argument(
            "draft connection pool size exceeds the target session limit",
        ));
    }
    Ok(args)
}

pub(crate) fn parse_draft(argv: Option<Vec<String>>) -> Result<DraftArgs, DynamoError> {
    let args = match argv {
        Some(argv) => {
            DraftArgs::try_parse_from(argv).map_err(|error| invalid_argument(error.to_string()))?
        }
        None => DraftArgs::parse(),
    };
    validate_common(&args.common)?;
    if args.model_name.is_empty() {
        return Err(invalid_argument("model name must not be empty"));
    }
    if args.transport_hwm <= 0 {
        return Err(invalid_argument("transport HWM must be positive"));
    }
    if args.transport_queue_capacity == 0
        || args.inference_queue_capacity == 0
        || args.inference_concurrency == 0
        || args.inference_output_capacity < 2
    {
        return Err(invalid_argument(
            "draft transport and inference queue bounds are invalid",
        ));
    }
    if args.transport_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
        || args.inference_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
        || args.inference_concurrency > tokio::sync::Semaphore::MAX_PERMITS
        || args.inference_output_capacity > tokio::sync::Semaphore::MAX_PERMITS
    {
        return Err(invalid_argument(
            "draft transport capacity exceeds the Tokio primitive limit",
        ));
    }
    Ok(args)
}

pub(crate) fn worker_config(
    common: CommonArgs,
    model_name: String,
    model_path: String,
) -> WorkerConfig {
    WorkerConfig {
        namespace: common.namespace,
        component: common.component,
        endpoint: common.endpoint,
        endpoint_types: common.endpoint_types,
        custom_jinja_template: common.custom_jinja_template,
        tool_call_parser: common.dyn_tool_call_parser,
        reasoning_parser: common.dyn_reasoning_parser,
        exclude_tools_when_tool_choice_none: common.exclude_tools_when_tool_choice_none,
        disaggregation_mode: common.disaggregation_mode,
        route_to_encoder: common.route_to_encoder,
        enable_rl: common.enable_rl,
        model_name: model_path,
        served_model_name: Some(model_name),
        ..WorkerConfig::default()
    }
}

fn validate_common(common: &CommonArgs) -> Result<(), DynamoError> {
    if common.disaggregation_mode != DisaggregationMode::Aggregated {
        return Err(invalid_argument(
            "speculative target and draft workers require --disaggregation-mode aggregated",
        ));
    }
    if common.route_to_encoder {
        return Err(invalid_argument(
            "speculative target and draft workers do not support --route-to-encoder",
        ));
    }
    Ok(())
}

fn invalid_argument(message: impl Into<String>) -> DynamoError {
    backend_error(BackendError::InvalidArgument, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_cli_exposes_local_runtime_and_model_configuration() {
        let args = parse_target(Some(vec![
            "target".into(),
            "--namespace".into(),
            "test-ns".into(),
            "--component".into(),
            "target".into(),
            "--draft-endpoint".into(),
            "test-ns/draft/generate".into(),
            "--model-path".into(),
            "/tmp/tokenizer".into(),
        ]))
        .unwrap();

        assert_eq!(args.common.namespace, "test-ns");
        assert_eq!(args.common.component, "target");
        assert_eq!(args.draft_endpoint, "test-ns/draft/generate");
        assert_eq!(args.model_path, "/tmp/tokenizer");
    }

    #[test]
    fn draft_cli_rejects_unbounded_or_wrong_stage_configuration() {
        assert!(
            parse_draft(Some(vec![
                "draft".into(),
                "--transport-hwm".into(),
                "0".into(),
            ]))
            .is_err()
        );
        assert!(
            parse_draft(Some(vec![
                "draft".into(),
                "--disaggregation-mode".into(),
                "decode".into(),
            ]))
            .is_err()
        );
        let unsupported_capacity = tokio::sync::Semaphore::MAX_PERMITS
            .checked_add(1)
            .expect("Tokio capacity limit leaves headroom");
        assert!(
            parse_target(Some(vec![
                "target".into(),
                "--max-inflight-sessions".into(),
                unsupported_capacity.to_string(),
            ]))
            .is_err()
        );
        assert!(
            parse_draft(Some(vec![
                "draft".into(),
                "--inference-concurrency".into(),
                unsupported_capacity.to_string(),
            ]))
            .is_err()
        );
    }
}
