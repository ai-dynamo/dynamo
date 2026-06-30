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
    Standalone,
}

impl EppMode {
    /// Parse `DYN_EPP_MODE` from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(&|k| std::env::var(k).ok())
    }

    /// Parse the mode, failing fast on an unrecognized value so a typo (e.g.
    /// `standalone`) can never silently fall through to full-dynamo mode.
    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        match trimmed(get(DYN_EPP_MODE)).as_deref() {
            None | Some(FULL_DYNAMO_STACK_MODE) => Ok(Self::FullDynamoStack),
            Some(STANDALONE_MODE) => Ok(Self::Standalone),
            Some(other) => anyhow::bail!(
                "{DYN_EPP_MODE} has invalid value {other:?}; \
                 expected {STANDALONE_MODE:?} or {FULL_DYNAMO_STACK_MODE:?}"
            ),
        }
    }
}

/// Standalone configuration. The pod selector and target port are NOT here —
/// they come from the `InferencePool` named by `pool_name` at runtime.
///
/// Constraints are declared with `validator` attributes and checked by
/// [`EppConfig::validate_config`]; `parse` only resolves env + defaults.
#[derive(Debug, Clone, Validate)]
pub struct EppConfig {
    /// KV indexer thread-pool size for the in-process selector.
    #[validate(range(min = 1))]
    pub selector_threads: usize,
    /// Embedded replication: the EPP's OWN `Service` whose EndpointSlices the
    /// EPP watches to discover sibling replicas and sync active load with them
    /// over ZMQ. `None` = single-replica (no cross-replica sync).
    pub peer_service: Option<String>,
    /// Port each EPP replica binds for peer replica-sync (ZMQ) and dials peers
    /// on; all replicas must agree. Used only when `peer_service` is set.
    #[validate(range(min = 1))]
    pub peer_sync_port: u16,
    /// `InferencePool` this EPP backs; its selector + target port drive discovery.
    #[validate(length(min = 1, message = "DYN_EPP_POOL_NAME is required"))]
    pub pool_name: String,
    /// Kubernetes namespace the EPP runs in (from `POD_NAMESPACE`, downward API).
    /// The EPP, its `InferencePool`, the worker pods, and sibling EPP replicas
    /// all live here — the EPP and the pool it backs are always co-located.
    #[validate(length(min = 1, message = "POD_NAMESPACE is required (downward API metadata.namespace)"))]
    pub namespace: String,
    /// Model id used to build the offline tokenizer (no model card in this mode).
    #[validate(length(min = 1, message = "DYN_MODEL_NAME is required"))]
    pub model_name: String,
    /// KV-cache block size; MUST equal the vLLM `--block-size`.
    #[validate(range(min = 1, message = "DYN_KV_CACHE_BLOCK_SIZE must be >= 1"))]
    pub block_size: u32,
    /// vLLM `--kv-events-config` PUB port the selector subscribes to.
    #[validate(range(min = 1))]
    pub kv_event_port: u16,
    /// vLLM `--kv-events-config` topic (default empty).
    pub kv_event_topic: String,
    /// Optional ZMQ REQ port the selector uses for live-stream gap replay.
    pub replay_port: Option<u16>,
    /// Engine data-parallel size. Data parallelism is not supported in this
    /// release, so this must be exactly 1.
    #[validate(range(min = 1, max = 1))]
    pub data_parallel_size: u32,
    /// Optional per-worker total KV blocks hint for the selector's load model.
    pub total_kv_blocks: Option<u64>,
    /// Optional per-worker max batched tokens (needed only when queueing is on).
    pub max_num_batched_tokens: Option<u64>,
}

impl EppConfig {
    /// Build and validate the standalone contract from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        let config = Self::parse(&|k| std::env::var(k).ok())?;
        config.validate_config()?;
        Ok(config)
    }

    /// Resolve env vars + defaults into the struct. No constraint checking — that
    /// lives in [`Self::validate_config`]. Only fails when a value that IS set
    /// cannot be parsed into its type (e.g. a non-numeric port). Keeps parsing
    /// pure so tests supply a map instead of mutating the process environment.
    fn parse(get: &EnvGet) -> anyhow::Result<Self> {
        Ok(Self {
            selector_threads: opt_parse::<usize>(get, "DYN_EPP_SELECTOR_THREADS")?
                .unwrap_or(DEFAULT_SELECTOR_THREADS),
            peer_service: trimmed(get("DYN_EPP_PEER_SERVICE")),
            peer_sync_port: opt_parse::<u16>(get, "DYN_EPP_PEER_SYNC_PORT")?
                .unwrap_or(DEFAULT_PEER_SYNC_PORT),
            pool_name: trimmed(get("DYN_EPP_POOL_NAME")).unwrap_or_default(),
            // The EPP and its InferencePool are always co-located, so the pod's
            // own namespace (downward API) is the single source of truth — used
            // to watch the pool, the worker pods, and sibling EPP replicas.
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
    fn parse_cfg(pairs: &[(&str, &str)]) -> anyhow::Result<EppConfig> {
        let cfg = EppConfig::parse(&getter(pairs))?;
        cfg.validate_config()?;
        Ok(cfg)
    }

    #[test]
    fn mode_defaults_to_full_when_unset() {
        assert_eq!(parse_mode(&[]).unwrap(), EppMode::FullDynamoStack);
    }

    #[test]
    fn mode_parses_known_values() {
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "standalone")]).unwrap(),
            EppMode::Standalone
        );
        assert_eq!(
            parse_mode(&[("DYN_EPP_MODE", "full-dynamo-stack")]).unwrap(),
            EppMode::FullDynamoStack
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
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
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
                ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
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
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
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
                ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
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
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
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
