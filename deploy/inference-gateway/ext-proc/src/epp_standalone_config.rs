// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone mode configuration.
//!
//! `DYN_EPP_MODE` selects the EPP mode: `full-dynamo-stack` (default) or
//! `standalone` (fronts raw `vllm serve` pods with no Dynamo runtime and runs an
//! in-process, runtime-free KV-aware selector). The pod selector and target port
//! come from the `InferencePool`; the rest of the standalone contract comes from
//! the environment.
//!
//! [`EppConfig::from_env`] separates the two concerns: [`EppConfig::parse`] does
//! env-read + default resolution only, and [`EppConfig::validate_config`] enforces
//! the constraints declaratively via the `validator` derive.

use validator::Validate;

const DEFAULT_KV_EVENT_PORT: u16 = 5557;
const DEFAULT_DATA_PARALLEL_SIZE: u32 = 1;
const DEFAULT_SELECTOR_THREADS: usize = 4;
const DEFAULT_PEER_SYNC_PORT: u16 = 9092;

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
    // Workers can be raw vllm serve pods or other OpenAI-compatible backends.
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

#[derive(Debug, Clone, Validate)]
pub struct EppStandaloneConfig {
    /// KV indexer thread-pool size for the in-process selector.
    #[validate(range(min = 1))]
    pub selector_threads: usize,
    // EPP Service for peer discovery and state synchronization.
    // `None` = single-replica (no cross-replica sync).
    pub peer_service: Option<String>,
    // Peer replication sync port.
    // Required when `peer_service` is set.
    #[validate(range(min = 1))]
    pub peer_sync_port: u16,
    /// `InferencePool` this EPP backs; its selector + target port drive discovery.
    #[validate(length(min = 1, message = "DYN_EPP_INFERENCE_POOL_NAME is required"))]
    pub inference_pool_name: String,
    // Kubernetes namespace the EPP runs in (from `POD_NAMESPACE`, downward API).
    #[validate(length(min = 1, message = "POD_NAMESPACE is required"))]
    pub namespace: String,
    /// Model id used to build the offline tokenizer and registerd with the selector.
    #[validate(length(min = 1, message = "DYN_MODEL_NAME is required"))]
    pub model_name: String,
    /// KV-cache block size; MUST equal the inference engine block size.
    #[validate(range(min = 1, message = "DYN_KV_CACHE_BLOCK_SIZE must be >= 1"))]
    pub block_size: u32,
    /// KV zmq event port
    #[validate(range(min = 1))]
    pub kv_event_port: u16,
    // KV zmq event topic
    pub kv_event_topic: String,
    /// Optional zmq port the selector uses for live-stream gap replay.
    pub replay_port: Option<u16>,
    /// Engine data-parallel size. Currently only support DP=1.
    #[validate(range(min = 1, max = 1))]
    pub data_parallel_size: u32,
    /// Optional per-worker total KV blocks.
    pub total_kv_blocks: Option<u64>,
    /// Optional per-worker max batched tokens (required when queuing or policy config is used).
    pub max_num_batched_tokens: Option<u64>,
}

impl EppStandaloneConfig {
    /// Build and validate the standalone contract from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        let config = Self::parse(&|k| std::env::var(k).ok())?;
        config.validate_config()?;
        Ok(config)
    }

    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        Ok(Self {
            selector_threads: opt_parse::<usize>(get, "DYN_EPP_SELECTOR_THREADS")?
                .unwrap_or(DEFAULT_SELECTOR_THREADS),
            peer_service: trimmed(get("DYN_EPP_PEER_SERVICE")),
            peer_sync_port: opt_parse::<u16>(get, "DYN_EPP_PEER_SYNC_PORT")?
                .unwrap_or(DEFAULT_PEER_SYNC_PORT),
            pool_name: trimmed(get("DYN_EPP_INFERENCE_POOL_NAME")).unwrap_or_default(),
            namespace: trimmed(get("POD_NAMESPACE")).unwrap_or_default(),
            model_name: trimmed(get("DYN_MODEL_NAME")).unwrap_or_default(),
            block_size: opt_parse::<u32>(get, "DYN_KV_CACHE_BLOCK_SIZE")?.unwrap_or(0),
            kv_event_port: opt_parse::<u16>(get, "DYN_EPP_KV_EVENT_PORT")?
                .unwrap_or(DEFAULT_KV_EVENT_PORT),
            kv_event_topic: trimmed(get("DYN_EPP_KV_EVENT_TOPIC")).unwrap_or_default(),
            replay_port: opt_parse::<u16>(get, "DYN_EPP_REPLAY_PORT")?,
            data_parallel_size: opt_parse::<u32>(get, "DYN_DATA_PARALLEL_SIZE")?
                .unwrap_or(DEFAULT_DATA_PARALLEL_SIZE),
            total_kv_blocks: opt_parse::<u64>(get, "DYN_TOTAL_KV_BLOCKS")?,
            max_num_batched_tokens: opt_parse::<u64>(get, "DYN_MAX_NUM_BATCHED_TOKENS")?,
        })
    }

    /// Enforce the `validator` constraints, mapping the failure to `anyhow`.
    pub fn validate_config(&self) -> anyhow::Result<()> {
        self.validate()
            .map_err(|e| anyhow::anyhow!("invalid {STANDALONE_MODE} EPP config: {e}"))
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

    /// Mirror `from_env`: resolve (parse) then validate.
    fn parse_cfg(pairs: &[(&str, &str)]) -> anyhow::Result<EppStandaloneConfig> {
        let cfg = EppStandaloneConfig::parse(&getter(pairs))?;
        cfg.validate_config()?;
        Ok(cfg)
    }

    #[test]
    fn mode_defaults_to_full_when_unset() {
        assert_eq!(parse_mode(&[]).unwrap(), EppMode::DynamoRuntime);
    }

    #[test]
    fn mode_parses_known_values() {
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "standalone")]).unwrap(),
            EppMode::Standalone
        );
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "full-dynamo-stack")]).unwrap(),
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
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("config should parse");
        assert_eq!(cfg.selector_threads, DEFAULT_SELECTOR_THREADS);
        // No peer service => single-replica (replica sync off).
        assert!(cfg.peer_service.is_none());
        assert_eq!(cfg.peer_sync_port, DEFAULT_PEER_SYNC_PORT);
        assert_eq!(cfg.pool_name, "vllm-qwen-pool");
        assert_eq!(cfg.namespace, "inference");
        assert_eq!(cfg.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.kv_event_port, DEFAULT_KV_EVENT_PORT);
        assert_eq!(cfg.data_parallel_size, 1);
        assert!(cfg.replay_port.is_none());
        assert!(cfg.total_kv_blocks.is_none());
    }

    #[test]
    fn missing_pod_namespace_fails() {
        // POD_NAMESPACE is the single namespace source (downward API); without
        // it the EPP can't watch its pool, pods, or peers.
        assert!(
            parse_cfg(&[
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
            ])
            .is_err()
        );
    }

    #[test]
    fn replication_config_parsed() {
        let cfg = parse_cfg(&[
            ("DYN_EPP_PEER_SERVICE", "dynamo-epp"),
            ("DYN_EPP_PEER_SYNC_PORT", "9191"),
            ("DYN_EPP_SELECTOR_THREADS", "8"),
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ])
        .expect("replication config should parse");
        assert_eq!(cfg.peer_service.as_deref(), Some("dynamo-epp"));
        assert_eq!(cfg.peer_sync_port, 9191);
        assert_eq!(cfg.selector_threads, 8);
        assert_eq!(cfg.namespace, "inference");
    }

    #[test]
    fn missing_pool_name_fails() {
        assert!(
            parse_cfg(&[
                ("POD_NAMESPACE", "inference"),
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
                ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
                ("POD_NAMESPACE", "inference"),
                ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
                ("DYN_KV_CACHE_BLOCK_SIZE", "0"),
            ])
            .is_err()
        );
    }

    #[test]
    fn data_parallel_size_must_be_one() {
        // Data parallelism is not supported in this release: only DP=1 is valid.
        let base = [
            ("DYN_EPP_INFERENCE_POOL_NAME", "vllm-qwen-pool"),
            ("POD_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ];
        let with_dp = |dp: &str| {
            let mut pairs = base.to_vec();
            pairs.push(("DYN_DATA_PARALLEL_SIZE", dp));
            parse_cfg(&pairs)
        };
        assert!(with_dp("0").is_err());
        assert!(with_dp("2").is_err());
        assert_eq!(with_dp("1").unwrap().data_parallel_size, 1);
    }
}
