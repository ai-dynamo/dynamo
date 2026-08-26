// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone EPP mode configuration.
//!
//! `DYN_EPP_MODE=dynamo` (default) uses the Dynamo runtime. `standalone` parses
//! the selector-only config used when the EPP fronts raw OpenAI-compatible
//! workers without a Dynamo runtime.
//!
//! [`EppStandaloneConfig::from_env`] reads envs, applies defaults, and calls
//! [`EppStandaloneConfig::validate_config`] for field and cross-field checks.

use validator::Validate;
use validator::ValidationError;

use crate::vllm_render_client::parse_tokenizer_service_base_url;

const DEFAULT_KV_EVENT_PORT: u16 = 5557;
const DEFAULT_REPLICA_SYNC_PORT: u16 = 9092;
const DEFAULT_SELECTOR_THREADS: usize = 4;
const DEFAULT_TOKENIZATION_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_TOKENIZER_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
/// Safety ceiling on concurrent in-flight `pick()`s (tunable), NOT a throughput
/// throttle: it caps concurrent tokenizer/render calls and buffered request
/// bodies so a burst can't exhaust memory. HTTP/2 stream multiplexing means the
/// TCP-connection cap does not bound requests, so this is the actual guardrail.
const DEFAULT_MAX_INFLIGHT_REQUESTS: usize = 1024;

/// Environment variable that selects the EPP operating mode.
pub const DYN_EPP_MODE: &str = "DYN_EPP_MODE";
/// `DYN_EPP_MODE` value selecting standalone mode.
pub const STANDALONE_MODE: &str = "standalone";
/// `DYN_EPP_MODE` value selecting the Dynamo runtime.
pub const DYNAMO_RUNTIME_MODE: &str = "dynamo";

/// Reads an environment variable, matching the injectable getter used in tests.
type EnvGet<'a> = dyn Fn(&str) -> Option<String> + 'a;

/// Top-level EPP operating mode from `DYN_EPP_MODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EppMode {
    // Connects to the Dynamo runtime and constructs a KvRouter. Requires Dynamo workers
    // to be connected to the runtime. (default)
    DynamoRuntime,
    // No runtime connection. Constructs a ServiceSelector for tracking workers, kv state and selecting best worker.
    Standalone,
}

impl EppMode {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(&|k| std::env::var(k).ok())
    }

    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        match trimmed(get(DYN_EPP_MODE)).as_deref() {
            None | Some(DYNAMO_RUNTIME_MODE) => Ok(Self::DynamoRuntime),
            Some(STANDALONE_MODE) => Ok(Self::Standalone),
            Some(other) => anyhow::bail!(
                "{DYN_EPP_MODE} has invalid value {other:?}; \
                 expected {STANDALONE_MODE:?} or {DYNAMO_RUNTIME_MODE:?}"
            ),
        }
    }
}

/// Wire protocol exposed by the configured tokenizer service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerProtocol {
    VllmRender,
}

impl std::str::FromStr for TokenizerProtocol {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "vllm-render" => Ok(Self::VllmRender),
            other => anyhow::bail!(
                "DYN_EPP_TOKENIZER_PROTOCOL has invalid value {other:?}; \
                 expected \"vllm-render\""
            ),
        }
    }
}

#[derive(Debug, Clone, Validate)]
pub struct EppStandaloneConfig {
    /// KV indexer thread-pool size for the in-process selector.
    #[validate(range(min = 1))]
    pub selector_threads: usize,
    /// EPP Service for peer discovery and state synchronization.
    pub peer_service: Option<String>,
    /// Local EPP Pod IP (from `POD_IP`, downward API). Required when peer
    /// discovery is enabled so this replica can exclude itself from peers.
    pub pod_ip: Option<String>,
    /// ZMQ listener and peer dial port for replica synchronization. Every EPP
    /// selected by `peer_service` must use the same port.
    #[validate(range(
        min = 1,
        message = "DYN_EPP_REPLICA_SYNC_PORT must be greater than zero"
    ))]
    pub replica_sync_port: u16,
    /// `InferencePool` this EPP backs; its selector + target port drive discovery.
    #[validate(length(min = 1, message = "DYN_EPP_INFERENCE_POOL_NAME is required"))]
    pub inference_pool_name: String,
    /// Kubernetes namespace the EPP runs in (from `POD_NAMESPACE`, downward API).
    #[validate(length(min = 1, message = "POD_NAMESPACE is required"))]
    pub namespace: String,
    /// Served/catalog model identity used to group discovered workers.
    #[validate(length(min = 1, message = "DYN_MODEL_NAME is required"))]
    pub model_name: String,
    /// Base URL of the tokenizer service.
    #[validate(length(min = 1, message = "DYN_EPP_TOKENIZER_SERVICE_URL is required"))]
    #[validate(custom(function = "validate_tokenizer_service_url"))]
    pub tokenizer_service_url: String,
    /// Protocol spoken by the configured tokenizer service.
    pub tokenizer_protocol: TokenizerProtocol,
    /// Deadline for calls to the configured tokenization provider.
    #[validate(range(min = 1, message = "DYN_EPP_TOKENIZATION_TIMEOUT_MS must be >= 1"))]
    pub tokenization_timeout_ms: u64,
    /// Maximum successful response body accepted from the tokenizer service.
    #[validate(range(min = 1, message = "DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES must be >= 1"))]
    pub tokenizer_max_response_bytes: usize,
    /// KV-cache block size; MUST equal the inference engine block size.
    #[validate(range(min = 1, message = "DYN_KV_CACHE_BLOCK_SIZE must be >= 1"))]
    pub block_size: u32,
    /// KV zmq event port.
    #[validate(range(min = 1))]
    pub kv_event_port: u16,
    /// Optional ZMQ port the selector uses for live-stream gap replay. This
    /// must match the worker's explicitly configured replay endpoint.
    #[validate(range(
        min = 1,
        message = "DYN_EPP_KV_EVENT_REPLAY_PORT must be greater than zero when set"
    ))]
    pub replay_port: Option<u16>,
    /// Optional per-worker total KV blocks.
    pub total_kv_blocks: Option<u64>,
    /// Optional per-worker max batched tokens.
    #[validate(range(
        min = 1,
        message = "DYN_EPP_MAX_NUM_BATCHED_TOKENS must be greater than zero when set"
    ))]
    pub max_num_batched_tokens: Option<u64>,
    /// Safety ceiling on concurrent in-flight `pick()`s: bounds concurrent
    /// tokenizer/render calls and buffered bodies (a load-shed guardrail, not a
    /// throughput throttle). Excess requests are shed with a 503, not queued.
    #[validate(range(min = 1, message = "DYN_EPP_MAX_INFLIGHT_REQUESTS must be >= 1"))]
    pub max_inflight_requests: usize,
}

impl EppStandaloneConfig {
    /// Build and validate the standalone contract from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        let config = Self::parse(&|k| std::env::var(k).ok())?;
        config.validate_config()?;
        Ok(config)
    }

    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        let tokenizer_protocol = trimmed(get("DYN_EPP_TOKENIZER_PROTOCOL"))
            .ok_or_else(|| anyhow::anyhow!("DYN_EPP_TOKENIZER_PROTOCOL is required"))?
            .parse()?;

        Ok(Self {
            selector_threads: opt_parse::<usize>(get, "DYN_EPP_SELECTION_INDEXER_THREADS")?
                .unwrap_or(DEFAULT_SELECTOR_THREADS),
            peer_service: trimmed(get("DYN_EPP_PEER_SERVICE")),
            pod_ip: trimmed(get("POD_IP")),
            replica_sync_port: opt_parse::<u16>(get, "DYN_EPP_REPLICA_SYNC_PORT")?
                .unwrap_or(DEFAULT_REPLICA_SYNC_PORT),
            inference_pool_name: trimmed(get("DYN_EPP_INFERENCE_POOL_NAME")).unwrap_or_default(),
            namespace: trimmed(get("POD_NAMESPACE")).unwrap_or_default(),
            model_name: trimmed(get("DYN_MODEL_NAME")).unwrap_or_default(),
            tokenizer_service_url: trimmed(get("DYN_EPP_TOKENIZER_SERVICE_URL"))
                .unwrap_or_default(),
            tokenizer_protocol,
            tokenization_timeout_ms: opt_parse::<u64>(get, "DYN_EPP_TOKENIZATION_TIMEOUT_MS")?
                .unwrap_or(DEFAULT_TOKENIZATION_TIMEOUT_MS),
            tokenizer_max_response_bytes: opt_parse::<usize>(
                get,
                "DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES",
            )?
            .unwrap_or(DEFAULT_TOKENIZER_MAX_RESPONSE_BYTES),
            block_size: opt_parse::<u32>(get, "DYN_KV_CACHE_BLOCK_SIZE")?.unwrap_or(0),
            kv_event_port: opt_parse::<u16>(get, "DYN_EPP_KV_EVENT_PORT")?
                .unwrap_or(DEFAULT_KV_EVENT_PORT),
            replay_port: opt_parse::<u16>(get, "DYN_EPP_KV_EVENT_REPLAY_PORT")?,
            total_kv_blocks: opt_parse::<u64>(get, "DYN_EPP_TOTAL_KV_BLOCKS")?,
            max_num_batched_tokens: opt_parse::<u64>(get, "DYN_EPP_MAX_NUM_BATCHED_TOKENS")?,
            max_inflight_requests: opt_parse::<usize>(get, "DYN_EPP_MAX_INFLIGHT_REQUESTS")?
                .unwrap_or(DEFAULT_MAX_INFLIGHT_REQUESTS),
        })
    }

    /// Enforce the `validator` constraints, mapping the failure to `anyhow`.
    pub fn validate_config(&self) -> anyhow::Result<()> {
        self.validate()
            .map_err(|e| anyhow::anyhow!("invalid {STANDALONE_MODE} EPP config: {e}"))?;
        if self.peer_service.is_some() && self.pod_ip.is_none() {
            anyhow::bail!(
                "invalid {STANDALONE_MODE} EPP config: DYN_EPP_PEER_SERVICE is set but POD_IP is unavailable; \
                 inject POD_IP via the downward API (fieldRef status.podIP)"
            );
        }
        Ok(())
    }
}

fn validate_tokenizer_service_url(value: &str) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Ok(());
    }

    parse_tokenizer_service_base_url(value)
        .map(|_| ())
        .map_err(|_| {
            let mut error = ValidationError::new("tokenizer_service_url_invalid");
            error.message =
                Some("DYN_EPP_TOKENIZER_SERVICE_URL must be an absolute HTTP(S) URL".into());
            error
        })
}

/// Trim a raw value and treat empty as absent.
fn trimmed(v: Option<String>) -> Option<String> {
    v.and_then(|v| {
        let t = v.trim();
        if t.is_empty() {
            None
        } else {
            Some(t.to_string())
        }
    })
}

fn opt_parse<T>(get: &EnvGet, key: &str) -> anyhow::Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match trimmed(get(key)) {
        None => Ok(None),
        Some(raw) => raw
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("{key} has an invalid value {raw:?}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an injectable env getter from key/value pairs — no process-global
    /// env mutation, so these tests are isolated and parallel-safe.
    fn getter(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    fn parse_mode(pairs: &[(&str, &str)]) -> anyhow::Result<EppMode> {
        EppMode::parse(&getter(pairs))
    }

    /// Mirror `from_env`: resolve (parse) then validate.
    fn parse_cfg(pairs: &[(&str, &str)]) -> anyhow::Result<EppStandaloneConfig> {
        let cfg = EppStandaloneConfig::parse(&getter(pairs))?;
        cfg.validate_config()?;
        Ok(cfg)
    }

    #[test]
    fn mode_defaults_to_dynamo_when_unset() {
        assert_eq!(parse_mode(&[]).unwrap(), EppMode::DynamoRuntime);
    }

    #[test]
    fn mode_parses_known_values() {
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "standalone")]).unwrap(),
            EppMode::Standalone
        );
        assert_eq!(
            parse_mode(&[(DYN_EPP_MODE, DYNAMO_RUNTIME_MODE)]).unwrap(),
            EppMode::DynamoRuntime
        );
    }

    #[test]
    fn mode_rejects_unknown_value() {
        // An unknown value must fail fast, not silently boot full-dynamo mode.
        assert!(parse_mode(&[("DYN_EPP_MODE", "nonsense-mode")]).is_err());
    }

    #[test]
    fn parses_required_and_defaults() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("config should parse");
        assert_eq!(cfg.selector_threads, DEFAULT_SELECTOR_THREADS);
        // No peer service => single-replica (replica sync off).
        assert!(cfg.peer_service.is_none());
        assert!(cfg.pod_ip.is_none());
        assert_eq!(cfg.replica_sync_port, DEFAULT_REPLICA_SYNC_PORT);
        assert_eq!(cfg.inference_pool_name, "vllm-qwen-pool");
        assert_eq!(cfg.namespace, "inference");
        assert_eq!(cfg.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(cfg.tokenizer_service_url, "http://vllm-render:8000");
        assert_eq!(cfg.tokenizer_protocol, TokenizerProtocol::VllmRender);
        assert_eq!(cfg.tokenization_timeout_ms, DEFAULT_TOKENIZATION_TIMEOUT_MS);
        assert_eq!(
            cfg.tokenizer_max_response_bytes,
            DEFAULT_TOKENIZER_MAX_RESPONSE_BYTES
        );
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.kv_event_port, DEFAULT_KV_EVENT_PORT);
        assert!(cfg.replay_port.is_none());
        assert!(cfg.total_kv_blocks.is_none());
        assert_eq!(cfg.max_inflight_requests, DEFAULT_MAX_INFLIGHT_REQUESTS);
    }

    #[test]
    fn missing_pod_namespace_fails() {
        // POD_NAMESPACE is the single namespace source (downward API); without
        // it the EPP can't watch its pool, pods, or peers.
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn peer_service_config_parsed() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
            ("POD_IP", "10.0.0.10"),
            ("DYN_EPP_SELECTION_INDEXER_THREADS", "8"),
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("peer service config should parse");
        assert_eq!(cfg.peer_service.as_deref(), Some("dynamo-epp"));
        assert_eq!(cfg.pod_ip.as_deref(), Some("10.0.0.10"));
        assert_eq!(cfg.replica_sync_port, DEFAULT_REPLICA_SYNC_PORT);
        assert_eq!(cfg.selector_threads, 8);
        assert_eq!(cfg.namespace, "inference");
    }

    #[test]
    fn replica_sync_port_can_be_overridden() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
            ("POD_IP", "10.0.0.10"),
            ("DYN_EPP_REPLICA_SYNC_PORT", "9192"),
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("replica sync port override should parse");

        assert_eq!(cfg.replica_sync_port, 9192);
    }

    #[test]
    fn peer_service_requires_pod_ip() {
        let error = parse_cfg(&[
            ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect_err("peer service without POD_IP must fail");

        assert!(error.to_string().contains("POD_IP"));
    }

    #[test]
    fn zero_replica_sync_port_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
                ("DYN_EPP_REPLICA_SYNC_PORT", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn missing_inference_pool_name_fails() {
        assert!(
            parse_cfg(&[
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn zero_block_size_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn replay_port_is_optional_and_uses_the_kv_event_name() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ("DYN_EPP_KV_EVENT_REPLAY_PORT", "5558"),
        ])
        .unwrap();
        assert_eq!(cfg.replay_port, Some(5558));
    }

    #[test]
    fn zero_replay_port_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
                ("DYN_EPP_KV_EVENT_REPLAY_PORT", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn zero_max_num_batched_tokens_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
                ("DYN_EPP_MAX_NUM_BATCHED_TOKENS", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn max_inflight_requests_can_be_overridden() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ("DYN_EPP_MAX_INFLIGHT_REQUESTS", "256"),
        ])
        .unwrap();
        assert_eq!(cfg.max_inflight_requests, 256);
    }

    #[test]
    fn zero_max_inflight_requests_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
                ("DYN_EPP_MAX_INFLIGHT_REQUESTS", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_service_url_is_required() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_service_url_must_be_http() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "unix:///tmp/vllm.sock"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenization_timeout_must_be_positive() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_EPP_TOKENIZATION_TIMEOUT_MS", "0"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_max_response_bytes_can_be_overridden() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
            ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
            ("DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES", "33554432"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .unwrap();

        assert_eq!(cfg.tokenizer_max_response_bytes, 32 * 1024 * 1024);
    }

    #[test]
    fn tokenizer_max_response_bytes_must_be_positive() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "vllm-render"),
                ("DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES", "0"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn tokenizer_protocol_is_required() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_EPP_TOKENIZER_SERVICE_URL", "http://vllm-render:8000"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn unsupported_tokenizer_protocol_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                (
                    "DYN_EPP_TOKENIZER_SERVICE_URL",
                    "http://sglang-tokenizer:30000"
                ),
                ("DYN_EPP_TOKENIZER_PROTOCOL", "sglang-tokenize"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }
}
