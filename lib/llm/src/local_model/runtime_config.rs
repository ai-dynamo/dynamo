// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::protocols::tensor;
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DisaggregatedEndpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_host: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRuntimeConfig {
    pub total_kv_blocks: Option<u64>,

    pub max_num_seqs: Option<u64>,

    pub max_num_batched_tokens: Option<u64>,

    pub tool_call_parser: Option<String>,

    pub reasoning_parser: Option<String>,

    /// When true, strip tool definitions from the chat template when tool_choice is "none".
    #[serde(default = "default_exclude_tools_when_tool_choice_none")]
    pub exclude_tools_when_tool_choice_none: bool,

    /// Starting rank of data parallel ranks for this worker (0 if DP not enabled)
    #[serde(default = "default_data_parallel_start_rank")]
    pub data_parallel_start_rank: u32,

    /// Total number of data parallel ranks for this worker (1 if DP not enabled)
    #[serde(default = "default_data_parallel_size")]
    pub data_parallel_size: u32,

    /// Enable worker-local KV indexer for tracking this worker's own KV cache state (default: true)
    #[serde(default = "default_local_indexer")]
    pub enable_local_indexer: bool,

    /// Mapping of engine-specific runtime configs
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub runtime_data: HashMap<String, serde_json::Value>,

    // Provide tensor model config in the case where the model type is Tensor.
    // Currently use JSON object for convinence, the programmatic way is to
    // define the model config struct as part of the tensor protocol and
    // import it here.
    // [gluo TODO] switch to ModelConfig if desired and workout a way to
    // prepare it in a convinent way, the protobuf library used by tonic
    // doesn't provide JSON parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_model_config: Option<tensor::TensorModelConfig>,

    /// Bootstrap endpoint for disaggregated serving (prefill workers publish this)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disaggregated_endpoint: Option<DisaggregatedEndpoint>,

    #[serde(default = "default_eagle")]
    pub enable_eagle: bool,

    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub taints: HashSet<String>,

    /// Stable routing identifier preserved across worker restarts.
    ///
    /// Populated from the `HOSTNAME` environment variable when the worker is launched in a
    /// Kubernetes StatefulSet (`worker-0`, `worker-1`, …). Distributed caching layers (KV
    /// router, KVBM, multi-model dispatch) read this through the discovery watch and use it
    /// as the keying input for rendezvous (HRW) hashing so cache assignments survive a pod
    /// restart that gives the worker a new ephemeral `worker_id`. `None` means no stable id
    /// was published; callers fall back to `worker_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_routing_id: Option<String>,
}

const fn default_data_parallel_start_rank() -> u32 {
    0
}

const fn default_data_parallel_size() -> u32 {
    1
}

const fn default_local_indexer() -> bool {
    true
}

const fn default_exclude_tools_when_tool_choice_none() -> bool {
    true
}

const fn default_eagle() -> bool {
    false
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            total_kv_blocks: None,
            max_num_seqs: None,
            max_num_batched_tokens: None,
            tool_call_parser: None,
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: default_exclude_tools_when_tool_choice_none(),
            data_parallel_start_rank: default_data_parallel_start_rank(),
            data_parallel_size: default_data_parallel_size(),
            enable_local_indexer: true,
            runtime_data: HashMap::new(),
            tensor_model_config: None,
            disaggregated_endpoint: None,
            enable_eagle: false,
            taints: HashSet::new(),
            stable_routing_id: None,
        }
    }
}

impl dynamo_kv_router::WorkerConfigLike for ModelRuntimeConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }

    fn stable_routing_id(&self) -> Option<&str> {
        self.stable_routing_id.as_deref()
    }
}

impl ModelRuntimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_engine_specific<T: Serialize>(&mut self, key: &str, value: T) -> anyhow::Result<()> {
        self.runtime_data
            .insert(key.to_string(), serde_json::to_value(value)?);
        Ok(())
    }

    pub fn get_engine_specific<T: DeserializeOwned>(&self, key: &str) -> anyhow::Result<Option<T>> {
        if let Some(value) = self.runtime_data.get(key) {
            Ok(Some(serde_json::from_value(value.clone())?))
        } else {
            Ok(None)
        }
    }

    /// Populate `stable_routing_id` from the `HOSTNAME` environment variable.
    ///
    /// On Kubernetes StatefulSets the pod hostname is stable across pod restarts
    /// (`worker-0`, `worker-1`, …), which makes it a good rendezvous-hash key for caching
    /// layers that want a worker's logical identity to outlive its process. The override
    /// `DYN_STABLE_ROUTING_ID` is honoured first for operators who want to supply a value
    /// explicitly (deployments where hostname is not a meaningful identity).
    ///
    /// Sets the field only if it is currently unset; returns `&mut self` for chaining. If
    /// neither `DYN_STABLE_ROUTING_ID` nor `HOSTNAME` is set (or both are empty) the field
    /// is left as `None` and rendezvous-aware consumers should fall back to `worker_id`.
    pub fn populate_stable_routing_id_from_env(&mut self) -> &mut Self {
        if self.stable_routing_id.is_some() {
            return self;
        }
        let candidate = std::env::var("DYN_STABLE_ROUTING_ID")
            .ok()
            .or_else(|| std::env::var("HOSTNAME").ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if let Some(value) = candidate {
            tracing::info!(stable_routing_id = %value, "populated stable_routing_id from environment");
            self.stable_routing_id = Some(value);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test runs in its own thread, so flip both env vars to a known state to avoid
    /// inheriting state from a previously-run test or the surrounding process.
    fn clear_env() {
        // SAFETY: tests in this module are not run concurrently for the same env vars;
        // see `serialize_env_test` mutex.
        unsafe {
            std::env::remove_var("DYN_STABLE_ROUTING_ID");
            std::env::remove_var("HOSTNAME");
        }
    }

    fn set_env(key: &str, value: Option<&str>) {
        unsafe {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }

    /// Tests in this module mutate process-wide env vars. Serialize them so they don't race
    /// each other when `cargo test` uses multiple threads.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn populates_from_hostname_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        set_env("HOSTNAME", Some("worker-3"));

        let mut cfg = ModelRuntimeConfig::default();
        cfg.populate_stable_routing_id_from_env();
        assert_eq!(cfg.stable_routing_id.as_deref(), Some("worker-3"));
    }

    #[test]
    fn dyn_override_wins_over_hostname() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        set_env("HOSTNAME", Some("worker-3"));
        set_env("DYN_STABLE_ROUTING_ID", Some("router-shard-7"));

        let mut cfg = ModelRuntimeConfig::default();
        cfg.populate_stable_routing_id_from_env();
        assert_eq!(cfg.stable_routing_id.as_deref(), Some("router-shard-7"));
    }

    #[test]
    fn preserves_caller_supplied_value() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        set_env("HOSTNAME", Some("worker-3"));

        let mut cfg = ModelRuntimeConfig {
            stable_routing_id: Some("explicit".to_string()),
            ..Default::default()
        };
        cfg.populate_stable_routing_id_from_env();
        assert_eq!(cfg.stable_routing_id.as_deref(), Some("explicit"));
    }

    #[test]
    fn empty_value_treated_as_unset() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        set_env("HOSTNAME", Some("   "));

        let mut cfg = ModelRuntimeConfig::default();
        cfg.populate_stable_routing_id_from_env();
        assert!(cfg.stable_routing_id.is_none());
    }

    #[test]
    fn no_env_leaves_field_none() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        let mut cfg = ModelRuntimeConfig::default();
        cfg.populate_stable_routing_id_from_env();
        assert!(cfg.stable_routing_id.is_none());
    }

    #[test]
    fn roundtrips_through_serde_json() {
        let cfg = ModelRuntimeConfig {
            stable_routing_id: Some("worker-7".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"stable_routing_id\":\"worker-7\""));
        let parsed: ModelRuntimeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.stable_routing_id.as_deref(), Some("worker-7"));
    }

    #[test]
    fn serde_skips_when_none() {
        let cfg = ModelRuntimeConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(!json.contains("stable_routing_id"));
    }
}
