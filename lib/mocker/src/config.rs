// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo integration options around AISimulate's canonical engine configuration.

use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use aisimulate_core::engine::{EngineLaunchConfig, TimingModelConfig};
use anyhow::{Result, ensure};
use dynamo_kv_router::config::RouterQueuePolicy;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use validator::Validate;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::ReasoningConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MockerRuntimeOptions {
    pub enable_local_indexer: bool,
    pub bootstrap_port: Option<u16>,
    pub handoff_session_timeout_ms: u64,
    pub reasoning: Option<ReasoningConfig>,
    pub response_replay_trace_path: Option<PathBuf>,
    pub zmq_kv_events_port: Option<u16>,
    pub zmq_replay_port: Option<u16>,
    pub router_queue_policy: Option<RouterQueuePolicy>,
}

impl Default for MockerRuntimeOptions {
    fn default() -> Self {
        Self {
            enable_local_indexer: false,
            bootstrap_port: None,
            handoff_session_timeout_ms: 300_000,
            reasoning: None,
            response_replay_trace_path: None,
            zmq_kv_events_port: None,
            zmq_replay_port: None,
            router_queue_policy: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MockerConfig {
    pub engine: EngineLaunchConfig,
    pub runtime: MockerRuntimeOptions,
    pub perf_model: Arc<PerfModel>,
}

impl Default for MockerConfig {
    fn default() -> Self {
        Self {
            engine: EngineLaunchConfig::default(),
            runtime: MockerRuntimeOptions::default(),
            perf_model: Arc::new(PerfModel::default()),
        }
    }
}

impl Deref for MockerConfig {
    type Target = EngineLaunchConfig;
    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}
impl DerefMut for MockerConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}

impl Serialize for MockerConfig {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.engine).map_err(serde::ser::Error::custom)?;
        value["dynamo"] = serde_json::to_value(&self.runtime).map_err(serde::ser::Error::custom)?;
        value.serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for MockerConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_value(Value::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl MockerConfig {
    pub fn from_value(value: Value) -> Result<Self> {
        let Value::Object(mut engine) = value else {
            anyhow::bail!("Mocker config must be an object");
        };
        let runtime: MockerRuntimeOptions = match engine.remove("dynamo") {
            Some(options) => serde_path_to_error::deserialize(options)?,
            None => MockerRuntimeOptions::default(),
        };
        let engine = EngineLaunchConfig::from_value(Value::Object(engine))?;
        let perf_model = match &engine.timing_model {
            TimingModelConfig::Polynomial => Arc::new(PerfModel::default()),
            TimingModelConfig::Fixed {
                prefill_ms,
                decode_ms,
            } => Arc::new(PerfModel::Fixed {
                prefill_ms: *prefill_ms,
                decode_ms: *decode_ms,
            }),
            TimingModelConfig::External { provider, config } if provider == "dynamo_profile" => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Profile {
                    path: PathBuf,
                }
                let profile: Profile = serde_path_to_error::deserialize(config.clone())?;
                Arc::new(PerfModel::from_npz(&profile.path)?)
            }
            TimingModelConfig::External { provider, .. } => {
                ensure!(
                    matches!(provider.as_str(), "ais" | "aic"),
                    "unsupported external timing provider {provider:?}"
                );
                // Python supplies the process-local AIS callback before execution.
                Arc::new(PerfModel::default())
            }
        };
        Self {
            engine,
            runtime,
            perf_model,
        }
        .normalized()
    }

    pub fn normalized(mut self) -> Result<Self> {
        self.engine = self.engine.normalized()?;
        ensure!(
            self.runtime.handoff_session_timeout_ms > 0,
            "handoff_session_timeout_ms must be positive"
        );
        if let Some(reasoning) = &self.runtime.reasoning {
            reasoning.validate()?;
        }
        Ok(self)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        Self::from_value(serde_json::from_str(content)?)
    }
    pub fn from_json_file(path: &Path) -> Result<Self> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }
    pub fn is_prefill(&self) -> bool {
        self.engine.is_prefill()
    }
    pub fn is_decode(&self) -> bool {
        self.engine.is_decode()
    }
    pub fn needs_kv_publisher(&self) -> bool {
        self.enable_prefix_caching && !self.is_decode()
    }
    pub fn ais_gpus_per_worker(&self) -> usize {
        self.tensor_parallel_size
            .saturating_mul(self.dp_size as usize)
    }
    pub fn effective_handoff_capacity(&self) -> usize {
        if self.max_num_seqs == usize::MAX {
            self.num_gpu_blocks.max(1)
        } else {
            self.max_num_seqs.max(1)
        }
    }

    pub fn ais_perf_config(&self) -> Option<&Value> {
        match &self.timing_model {
            TimingModelConfig::External { provider, config }
                if matches!(provider.as_str(), "ais" | "aic") =>
            {
                Some(config)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_fields_and_dynamo_options_round_trip_without_reconstruction() {
        let input = json!({
            "engine": {
                "backend": "vllm",
                "max_model_len": 64,
                "kv_cache_bytes_per_token": 128,
                "prefill_schedule_interval": 2,
                "timing_model": {
                    "type": "fixed",
                    "prefill_ms": 1.0,
                    "decode_ms": 2.0
                }
            },
            "dynamo": {
                "enable_local_indexer": true,
                "bootstrap_port": 9001
            }
        });
        let args = MockerConfig::from_value(input).unwrap();
        let roundtrip = MockerConfig::from_value(serde_json::to_value(&args).unwrap()).unwrap();
        assert_eq!(roundtrip.engine, args.engine);
        assert_eq!(roundtrip.runtime.bootstrap_port, Some(9001));
        assert!(roundtrip.runtime.enable_local_indexer);
        let rank = crate::engine_adapter::engine_components(roundtrip, false, false)
            .unwrap()
            .rank;
        assert_eq!(rank, args.engine.engine);
    }

    #[test]
    fn boundary_rejects_unknown_runtime_options_and_conflicting_sources() {
        for input in [
            json!({"dynamo":{"bootstrap_prot":9001}}),
            json!({"bootstrap_port":9001}),
            json!({"dynamo":{"handoff_session_timeout_ms":0}}),
        ] {
            assert!(MockerConfig::from_value(input).is_err());
        }
        let args = MockerConfig::from_value(json!({
            "engine": {
                "max_num_seqs": usize::MAX,
                "num_gpu_blocks": 16
            }
        }))
        .unwrap();
        assert_eq!(args.effective_handoff_capacity(), 16);
        let args = MockerConfig::from_value(json!({
            "engine": {
                "max_num_seqs": 32,
                "num_gpu_blocks": 16
            }
        }))
        .unwrap();
        assert_eq!(args.effective_handoff_capacity(), 32);
    }
}
