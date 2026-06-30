// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-only mode configuration.
//!
//! `DYN_EPP_MODE` selects the EPP mode: `full-dynamo-stack` (default) or
//! `router-only` (fronts raw `vllm serve` pods with no Dynamo runtime and
//! delegates KV-aware selection to the standalone selection service). The pod
//! selector and target port come from the `InferencePool`; the rest of the
//! router-only contract comes from the environment and is parsed here.

const DEFAULT_KV_EVENT_PORT: u16 = 5557;
const DEFAULT_DATA_PARALLEL_SIZE: u32 = 1;
const DEFAULT_SELECTOR_HTTP_PORT: u16 = 8092;
const DEFAULT_SELECTOR_REPLICA_SYNC_PORT: u16 = 9092;

/// Environment variable that selects the EPP operating mode.
pub const DYN_EPP_MODE: &str = "DYN_EPP_MODE";
/// `DYN_EPP_MODE` value selecting router-only mode.
pub const ROUTER_ONLY_MODE: &str = "router-only";
/// `DYN_EPP_MODE` value selecting the default full Dynamo stack.
pub const FULL_DYNAMO_STACK_MODE: &str = "full-dynamo-stack";

/// Reads an environment variable, matching the injectable getter used in tests.
type EnvGet<'a> = dyn Fn(&str) -> Option<String> + 'a;

/// Top-level EPP operating mode from `DYN_EPP_MODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EppMode {
    /// Default: attach to a Dynamo deployment (etcd/NATS + embedded KV router).
    FullDynamoStack,
    /// Front raw `vllm serve` pods with no Dynamo runtime.
    RouterOnly,
}

impl EppMode {
    /// Parse `DYN_EPP_MODE` from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(&|k| std::env::var(k).ok())
    }

    /// Parse the mode, failing fast on an unrecognized value so a typo (e.g.
    /// `router_only`) can never silently fall through to full-dynamo mode.
    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        match trimmed(get(DYN_EPP_MODE)).as_deref() {
            None | Some(FULL_DYNAMO_STACK_MODE) => Ok(Self::FullDynamoStack),
            Some(ROUTER_ONLY_MODE) => Ok(Self::RouterOnly),
            Some(other) => anyhow::bail!(
                "{DYN_EPP_MODE} has invalid value {other:?}; \
                 expected {ROUTER_ONLY_MODE:?} or {FULL_DYNAMO_STACK_MODE:?}"
            ),
        }
    }
}

/// Router-only configuration, built from the environment with fail-fast
/// validation. The pod selector and target port are NOT here — they come from
/// the `InferencePool` named by `pool_name` at runtime.
#[derive(Debug, Clone)]
pub struct EppConfig {
    /// Selection-service `Service` whose EndpointSlices the EPP watches.
    pub selector_service: String,
    /// Namespace of the selection-service `Service` (default `POD_NAMESPACE`).
    pub selector_service_namespace: Option<String>,
    /// HTTP port each selection-service replica serves on.
    pub selector_http_port: u16,
    /// ZMQ replica-sync PUB port each selection-service replica binds.
    pub selector_replica_sync_port: u16,
    /// `InferencePool` this EPP backs; its selector + target port drive discovery.
    pub pool_name: String,
    /// Namespace of the `InferencePool` (default `POD_NAMESPACE`).
    pub pool_namespace: Option<String>,
    /// Model id used to build the offline tokenizer (no model card in this mode).
    pub model_name: String,
    /// KV-cache block size; MUST equal the vLLM `--block-size`.
    pub block_size: u32,
    /// vLLM `--kv-events-config` PUB port the selector subscribes to.
    pub kv_event_port: u16,
    /// vLLM `--kv-events-config` topic (default empty).
    pub kv_event_topic: String,
    /// Optional ZMQ REQ port the selector uses for live-stream gap replay.
    pub replay_port: Option<u16>,
    /// Engine data-parallel size (default 1; aggregated V1 supports DP=1).
    pub data_parallel_size: u32,
    /// Optional per-worker total KV blocks hint for the selector's load model.
    pub total_kv_blocks: Option<u64>,
    /// Optional per-worker max batched tokens (needed only when queueing is on).
    pub max_num_batched_tokens: Option<u64>,
}

impl EppConfig {
    /// Parse and validate the router-only contract from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(&|k| std::env::var(k).ok())
    }

    /// Parse from an injectable getter. Keeps parsing pure so tests supply a map
    /// instead of mutating the process-global environment.
    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        let selector_service = require(get, "DYN_EPP_SELECTOR_SERVICE")?;
        let selector_service_namespace = trimmed(get("DYN_EPP_SELECTOR_SERVICE_NAMESPACE"));
        let selector_http_port =
            opt_parse::<u16>(get, "DYN_EPP_SELECTOR_HTTP_PORT")?.unwrap_or(DEFAULT_SELECTOR_HTTP_PORT);
        let selector_replica_sync_port = opt_parse::<u16>(get, "DYN_EPP_SELECTOR_REPLICA_SYNC_PORT")?
            .unwrap_or(DEFAULT_SELECTOR_REPLICA_SYNC_PORT);

        let pool_name = require(get, "DYN_EPP_POOL_NAME")?;
        let pool_namespace = trimmed(get("DYN_EPP_POOL_NAMESPACE"));
        let model_name = require(get, "DYN_MODEL_NAME")?;
        let block_size = require_parse::<u32>(get, "DYN_KV_CACHE_BLOCK_SIZE")?;
        if block_size == 0 {
            anyhow::bail!("DYN_KV_CACHE_BLOCK_SIZE must be greater than zero");
        }

        let kv_event_port =
            opt_parse::<u16>(get, "DYN_EPP_KV_EVENT_PORT")?.unwrap_or(DEFAULT_KV_EVENT_PORT);
        let kv_event_topic = trimmed(get("DYN_EPP_KV_EVENT_TOPIC")).unwrap_or_default();
        let replay_port = opt_parse::<u16>(get, "DYN_EPP_REPLAY_PORT")?;
        let data_parallel_size =
            opt_parse::<u32>(get, "DYN_DATA_PARALLEL_SIZE")?.unwrap_or(DEFAULT_DATA_PARALLEL_SIZE);
        if data_parallel_size == 0 {
            anyhow::bail!("DYN_DATA_PARALLEL_SIZE must be greater than zero");
        }
        let total_kv_blocks = opt_parse::<u64>(get, "DYN_TOTAL_KV_BLOCKS")?;
        let max_num_batched_tokens = opt_parse::<u64>(get, "DYN_MAX_NUM_BATCHED_TOKENS")?;

        Ok(Self {
            selector_service,
            selector_service_namespace,
            selector_http_port,
            selector_replica_sync_port,
            pool_name,
            pool_namespace,
            model_name,
            block_size,
            kv_event_port,
            kv_event_topic,
            replay_port,
            data_parallel_size,
            total_kv_blocks,
            max_num_batched_tokens,
        })
    }
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

fn require(get: &EnvGet, key: &str) -> anyhow::Result<String> {
    trimmed(get(key)).ok_or_else(|| anyhow::anyhow!("{key} is required in {ROUTER_ONLY_MODE} mode"))
}

fn require_parse<T>(get: &EnvGet, key: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = require(get, key)?;
    raw.parse::<T>()
        .map_err(|e| anyhow::anyhow!("{key} has an invalid value {raw:?}: {e}"))
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
        let map: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| map.get(k).cloned()
    }

    fn parse_mode(pairs: &[(&str, &str)]) -> anyhow::Result<EppMode> {
        EppMode::parse(&getter(pairs))
    }

    fn parse_cfg(pairs: &[(&str, &str)]) -> anyhow::Result<EppConfig> {
        EppConfig::parse(&getter(pairs))
    }

    #[test]
    fn mode_defaults_to_full_when_unset() {
        assert_eq!(parse_mode(&[]).unwrap(), EppMode::FullDynamoStack);
    }

    #[test]
    fn mode_parses_known_values() {
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "router-only")]).unwrap(),
            EppMode::RouterOnly
        );
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "full-dynamo-stack")]).unwrap(),
            EppMode::FullDynamoStack
        );
    }

    #[test]
    fn mode_rejects_unknown_value() {
        // A typo must fail fast, not silently boot full-dynamo mode.
        assert!(parse_mode(&[("DYN_EPP_MODE", "router_only")]).is_err());
    }

    #[test]
    fn parses_required_and_defaults() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_SELECTOR_SERVICE", "dynamo-selector"),
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("config should parse");
        assert_eq!(cfg.selector_service, "dynamo-selector");
        assert!(cfg.selector_service_namespace.is_none());
        assert_eq!(cfg.selector_http_port, DEFAULT_SELECTOR_HTTP_PORT);
        assert_eq!(
            cfg.selector_replica_sync_port,
            DEFAULT_SELECTOR_REPLICA_SYNC_PORT
        );
        assert_eq!(cfg.pool_name, "vllm-qwen-pool");
        assert!(cfg.pool_namespace.is_none());
        assert_eq!(cfg.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.kv_event_port, DEFAULT_KV_EVENT_PORT);
        assert_eq!(cfg.data_parallel_size, 1);
        assert!(cfg.replay_port.is_none());
        assert!(cfg.total_kv_blocks.is_none());
    }

    #[test]
    fn pool_namespace_is_parsed_when_set() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_SELECTOR_SERVICE", "dynamo-selector"),
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
            ("DYN_EPP_POOL_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("config should parse");
        assert_eq!(cfg.pool_namespace.as_deref(), Some("inference"));
    }

    #[test]
    fn missing_pool_name_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_SELECTOR_SERVICE", "dynamo-selector"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn missing_selector_service_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn zero_block_size_fails() {
        assert!(
            parse_cfg(&[
                ("DYN_EPP_SELECTOR_SERVICE", "dynamo-selector"),
                ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "0"),
            ])
            .is_err()
        );
    }
}
