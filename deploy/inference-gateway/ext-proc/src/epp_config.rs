// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-only (selector) mode configuration.
//!
//! The EPP runs in one of two modes, selected at startup by `DYN_EPP_MODE`:
//!
//! - `full-dynamo-stack` (default): the EPP attaches to a Dynamo deployment over
//!   etcd/NATS and uses an embedded KV router. See [`crate::epp::Router::from_discovery`].
//! - `router-only` (`DYN_EPP_MODE=router-only`): the EPP fronts a fleet of raw
//!   `vllm serve` pods with **no Dynamo control plane** and **no Dynamo runtime**.
//!   Worker discovery reads the GAIE `InferencePool` this EPP backs — the same
//!   object the gateway routes to — and watches the pods it selects. KV-aware
//!   selection is delegated to the standalone selection service
//!   (`python -m dynamo.select_service`) over HTTP.
//!
//! The `InferencePool` is the single source of truth for the pod label selector
//! and the OpenAI HTTP target port, so the EPP and the gateway can never disagree
//! about pool membership. Metadata the pool cannot carry — the KV-event/replay
//! ZMQ ports, model id, and block size — comes from the environment.

const DEFAULT_KV_EVENT_PORT: u16 = 5557;
const DEFAULT_DATA_PARALLEL_SIZE: u32 = 1;

/// Value of `DYN_EPP_MODE` that selects router-only (selector) mode.
pub const ROUTER_ONLY_MODE: &str = "router-only";

/// Environment variable that selects the EPP operating mode.
pub const DYN_EPP_MODE: &str = "DYN_EPP_MODE";

/// Configuration for router-only (selector) mode.
///
/// Built from the environment with fail-fast validation. The pod label selector
/// and HTTP target port are NOT here — they are read from the `InferencePool`
/// named by `pool_name` at runtime.
#[derive(Debug, Clone)]
pub struct EppConfig {
    /// Comma-separated HTTP base URLs of the selection-service replicas, e.g.
    /// `http://selector-0:8092,http://selector-1:8092`. Catalog writes fan out
    /// to all replicas; selection reads target one replica.
    pub selector_urls: Vec<String>,
    /// Name of the `InferencePool` this EPP backs. Its selector and target port
    /// drive pod discovery.
    pub pool_name: String,
    /// Namespace of the `InferencePool`. Defaults to the EPP's own namespace
    /// (`POD_NAMESPACE`) when unset.
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
    /// Engine data-parallel size (default 1). Aggregated V1 supports DP=1.
    pub data_parallel_size: u32,
    /// Optional per-worker total KV blocks hint for the selector's load model.
    pub total_kv_blocks: Option<u64>,
    /// Optional per-worker max batched tokens (required by the selector only
    /// when queueing is enabled).
    pub max_num_batched_tokens: Option<u64>,
}

impl EppConfig {
    /// Returns `true` iff `DYN_EPP_MODE=router-only`.
    pub fn is_enabled() -> bool {
        matches!(env_trimmed(DYN_EPP_MODE).as_deref(), Some(ROUTER_ONLY_MODE))
    }

    /// Parse and validate the router-only environment contract. Fails fast on a
    /// missing required field or a malformed numeric/boolean value.
    pub fn from_env() -> anyhow::Result<Self> {
        let selector_urls = env_trimmed("DYN_EPP_SELECTOR_URLS")
            .map(|raw| {
                raw.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if selector_urls.is_empty() {
            anyhow::bail!(
                "DYN_EPP_SELECTOR_URLS is required in {ROUTER_ONLY_MODE} mode \
                 (comma-separated selection-service base URLs)"
            );
        }

        let pool_name = require("DYN_EPP_POOL_NAME")?;
        let pool_namespace = env_trimmed("DYN_EPP_POOL_NAMESPACE");
        let model_name = require("DYN_MODEL_NAME")?;
        let block_size = require_parse::<u32>("DYN_KV_CACHE_BLOCK_SIZE")?;
        if block_size == 0 {
            anyhow::bail!("DYN_KV_CACHE_BLOCK_SIZE must be greater than zero");
        }

        let kv_event_port =
            opt_parse::<u16>("DYN_EPP_KV_EVENT_PORT")?.unwrap_or(DEFAULT_KV_EVENT_PORT);
        let kv_event_topic = env_trimmed("DYN_EPP_KV_EVENT_TOPIC").unwrap_or_default();
        let replay_port = opt_parse::<u16>("DYN_EPP_REPLAY_PORT")?;
        let data_parallel_size =
            opt_parse::<u32>("DYN_DATA_PARALLEL_SIZE")?.unwrap_or(DEFAULT_DATA_PARALLEL_SIZE);
        if data_parallel_size == 0 {
            anyhow::bail!("DYN_DATA_PARALLEL_SIZE must be greater than zero");
        }
        let total_kv_blocks = opt_parse::<u64>("DYN_TOTAL_KV_BLOCKS")?;
        let max_num_batched_tokens = opt_parse::<u64>("DYN_MAX_NUM_BATCHED_TOKENS")?;

        Ok(Self {
            selector_urls,
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

fn env_trimmed(key: &str) -> Option<String> {
    std::env::var(key).ok().and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn require(key: &str) -> anyhow::Result<String> {
    env_trimmed(key).ok_or_else(|| anyhow::anyhow!("{key} is required in {ROUTER_ONLY_MODE} mode"))
}

fn require_parse<T>(key: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = require(key)?;
    raw.parse::<T>()
        .map_err(|e| anyhow::anyhow!("{key} has an invalid value {raw:?}: {e}"))
}

fn opt_parse<T>(key: &str) -> anyhow::Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env_trimmed(key) {
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
    use std::sync::Mutex;

    // Env mutation is process-global; serialize the tests that touch it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard(Vec<&'static str>);
    impl EnvGuard {
        fn set(pairs: &[(&'static str, &str)]) -> Self {
            let keys = pairs.iter().map(|(k, _)| *k).collect();
            for (k, v) in pairs {
                unsafe { std::env::set_var(k, v) };
            }
            EnvGuard(keys)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for k in &self.0 {
                unsafe { std::env::remove_var(k) };
            }
        }
    }

    fn clear_all() {
        for k in [
            "DYN_EPP_MODE",
            "DYN_EPP_SELECTOR_URLS",
            "DYN_EPP_POOL_NAME",
            "DYN_EPP_POOL_NAMESPACE",
            "DYN_MODEL_NAME",
            "DYN_KV_CACHE_BLOCK_SIZE",
            "DYN_EPP_KV_EVENT_PORT",
            "DYN_EPP_KV_EVENT_TOPIC",
            "DYN_EPP_REPLAY_PORT",
            "DYN_DATA_PARALLEL_SIZE",
            "DYN_TOTAL_KV_BLOCKS",
            "DYN_MAX_NUM_BATCHED_TOKENS",
        ] {
            unsafe { std::env::remove_var(k) };
        }
    }

    #[test]
    fn mode_detection() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_all();
        assert!(!EppConfig::is_enabled());
        let _g = EnvGuard::set(&[("DYN_EPP_MODE", "router-only")]);
        assert!(EppConfig::is_enabled());
    }

    #[test]
    fn parses_required_and_defaults() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_all();
        let _g = EnvGuard::set(&[
            ("DYN_EPP_SELECTOR_URLS", "http://a:8092, http://b:8092"),
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ]);
        let cfg = EppConfig::from_env().expect("config should parse");
        assert_eq!(cfg.selector_urls, vec!["http://a:8092", "http://b:8092"]);
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
        let _lock = ENV_LOCK.lock().unwrap();
        clear_all();
        let _g = EnvGuard::set(&[
            ("DYN_EPP_SELECTOR_URLS", "http://a:8092"),
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
            ("DYN_EPP_POOL_NAMESPACE", "inference"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ]);
        let cfg = EppConfig::from_env().expect("config should parse");
        assert_eq!(cfg.pool_namespace.as_deref(), Some("inference"));
    }

    #[test]
    fn missing_pool_name_fails() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_all();
        let _g = EnvGuard::set(&[
            ("DYN_EPP_SELECTOR_URLS", "http://a:8092"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ]);
        assert!(EppConfig::from_env().is_err());
    }

    #[test]
    fn empty_selector_urls_fails() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_all();
        let _g = EnvGuard::set(&[
            ("DYN_EPP_SELECTOR_URLS", " , "),
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "16"),
        ]);
        assert!(EppConfig::from_env().is_err());
    }

    #[test]
    fn zero_block_size_fails() {
        let _lock = ENV_LOCK.lock().unwrap();
        clear_all();
        let _g = EnvGuard::set(&[
            ("DYN_EPP_SELECTOR_URLS", "http://a:8092"),
            ("DYN_EPP_POOL_NAME", "vllm-qwen-pool"),
            ("DYN_MODEL_NAME", "Qwen/Qwen3-0.6B"),
            ("DYN_KV_CACHE_BLOCK_SIZE", "0"),
        ]);
        assert!(EppConfig::from_env().is_err());
    }
}
