// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON-only PyO3 boundary for one materialized AISimulate replay execution.

use std::sync::Arc;

use aisimulate_core::engine::{Backend, TimingModel, TimingModelConfig};
use aisimulate_core::replay::{
    ReplayEngineConfig, ReplayRoleConfig, ReplaySpec, ReplayTopology, run_engine_replay,
    run_engine_replay_with_optional_role_timing, run_engine_replay_with_timing,
};
use anyhow::{Context, Result, anyhow, ensure};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AicTimingConfig {
    model: String,
    backend: String,
    system: String,
    #[serde(alias = "tp_size")]
    tp: u32,
    #[serde(default)]
    backend_version: Option<String>,
    #[serde(default = "one")]
    pp: u32,
    #[serde(default = "one")]
    attention_dp: u32,
    #[serde(default)]
    moe_tp_size: Option<u32>,
    #[serde(default)]
    moe_ep_size: Option<u32>,
    #[serde(default, alias = "gemm_quant_mode")]
    gemm_dtype: Option<String>,
    #[serde(default, alias = "moe_quant_mode")]
    moe_dtype: Option<String>,
    #[serde(default, alias = "fmha_quant_mode")]
    fmha_dtype: Option<String>,
    #[serde(default, alias = "kvcache_quant_mode")]
    kv_cache_dtype: Option<String>,
    #[serde(default, alias = "comm_quant_mode")]
    comm_dtype: Option<String>,
    #[serde(default)]
    nextn: u32,
    #[serde(default)]
    kv_block_size: Option<u32>,
    #[serde(default)]
    gpu_memory_utilization: Option<f64>,
    #[serde(default)]
    mem_fraction_static: Option<f64>,
    #[serde(default)]
    free_gpu_memory_fraction: Option<f64>,
    #[serde(default)]
    systems_path: Option<String>,
}

const fn one() -> u32 {
    1
}

impl AicTimingConfig {
    fn resolved_backend_version(&self) -> &str {
        self.backend_version
            .as_deref()
            .unwrap_or(match self.backend.as_str() {
                "vllm" => "0.19.0",
                "sglang" => "0.5.10",
                "trtllm" => "1.3.0rc10",
                _ => "",
            })
    }

    fn resolved_memory_fraction(&self) -> Result<(&'static str, f64)> {
        for (name, value) in [
            ("gpu_memory_utilization", self.gpu_memory_utilization),
            ("mem_fraction_static", self.mem_fraction_static),
            ("free_gpu_memory_fraction", self.free_gpu_memory_fraction),
        ] {
            ensure!(
                value.is_none_or(|fraction| {
                    fraction.is_finite() && (0.0..=1.0).contains(&fraction)
                }),
                "{name} must be finite and between 0 and 1"
            );
        }
        match self.backend.as_str() {
            "vllm" => Ok(("of_total", self.gpu_memory_utilization.unwrap_or(0.9))),
            "sglang" => Ok(("of_total", self.mem_fraction_static.unwrap_or(0.88))),
            "trtllm" => Ok(("of_free", self.free_gpu_memory_fraction.unwrap_or(0.9))),
            _ => Err(anyhow!(
                "unsupported AIC backend {:?}; expected vllm, sglang, or trtllm",
                self.backend
            )),
        }
    }

    fn validate_parallel_shape(&self) -> Result<()> {
        ensure!(
            self.tp > 0
                && self.pp > 0
                && self.attention_dp > 0
                && self.moe_tp_size != Some(0)
                && self.moe_ep_size != Some(0),
            "AIC timing parallel sizes tp, pp, attention_dp, moe_tp_size, and \
             moe_ep_size must be positive"
        );
        ensure!(self.nextn <= 5, "AIC nextn must be in 0..=5");
        ensure!(
            self.moe_tp_size.is_some() == self.moe_ep_size.is_some(),
            "AIC moe_tp_size and moe_ep_size must be configured together"
        );
        if let (Some(moe_tp), Some(moe_ep)) = (self.moe_tp_size, self.moe_ep_size) {
            ensure!(
                u64::from(self.tp) * u64::from(self.attention_dp)
                    == u64::from(moe_tp) * u64::from(moe_ep),
                "AIC topology requires tp * attention_dp == moe_tp_size * moe_ep_size"
            );
        }
        Ok(())
    }
}

struct AicTimingModel {
    engine: Py<PyAny>,
}

impl AicTimingModel {
    fn build(config: AicTimingConfig) -> Result<Self> {
        ensure!(
            !config.model.trim().is_empty(),
            "AIC timing config field \"model\" cannot be empty"
        );
        ensure!(
            !config.system.trim().is_empty(),
            "AIC timing config field \"system\" cannot be empty"
        );
        config.validate_parallel_shape()?;
        ensure!(
            matches!(config.backend.as_str(), "vllm" | "sglang" | "trtllm"),
            "unsupported AIC backend {:?}; expected vllm, sglang, or trtllm",
            config.backend
        );
        ensure!(
            !config.resolved_backend_version().is_empty(),
            "AIC backend version cannot be empty"
        );
        config.resolved_memory_fraction()?;

        let engine = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
            let sdk = PyModule::import(py, "aiconfigurator_core.sdk.engine")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("backend_version", config.resolved_backend_version())?;
            kwargs.set_item("tp_size", config.tp)?;
            kwargs.set_item("pp_size", config.pp)?;
            kwargs.set_item("attention_dp_size", config.attention_dp)?;
            kwargs.set_item("moe_tp_size", config.moe_tp_size)?;
            kwargs.set_item("moe_ep_size", config.moe_ep_size)?;
            kwargs.set_item("gemm_quant_mode", config.gemm_dtype.as_deref())?;
            kwargs.set_item("moe_quant_mode", config.moe_dtype.as_deref())?;
            kwargs.set_item("fmha_quant_mode", config.fmha_dtype.as_deref())?;
            kwargs.set_item("kvcache_quant_mode", config.kv_cache_dtype.as_deref())?;
            kwargs.set_item("comm_quant_mode", config.comm_dtype.as_deref())?;
            kwargs.set_item("nextn", config.nextn)?;
            kwargs.set_item("kv_block_size", config.kv_block_size)?;
            kwargs.set_item("systems_path", config.systems_path.as_deref())?;
            let spec = sdk.getattr("compile_engine")?.call(
                (
                    config.model.as_str(),
                    config.system.as_str(),
                    config.backend.as_str(),
                ),
                Some(&kwargs),
            )?;
            let aic = PyModule::import(py, "aiconfigurator_core")?
                .getattr("AicEngine")?
                .call_method1("from_spec", (spec, config.systems_path.as_deref()))?;
            Ok(aic.unbind())
        })
        .map_err(|error| {
            anyhow!("AIC timing provider could not compile the requested engine: {error}")
        })?;
        Ok(Self { engine })
    }
}

impl TimingModel for AicTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        let batch_size = checked_u32(batch_size, "prefill batch size")?;
        let mean_isl = checked_u32(mean_isl, "mean input length")?;
        let mean_prefix = checked_u32(mean_prefix, "mean prefix length")?;
        Python::with_gil(|py| {
            self.engine
                .bind(py)
                .call_method1(
                    "predict_prefill_latency",
                    (batch_size, mean_isl, mean_prefix),
                )?
                .extract::<f64>()
        })
        .map_err(|error| anyhow!("AIC prefill prediction failed: {error}"))
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        _active_kv_tokens: usize,
        mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        let batch_size = checked_u32(batch_size, "decode batch size")?;
        let mean_context_length = checked_u32(mean_context_length, "mean context length")?;
        Python::with_gil(|py| {
            self.engine
                .bind(py)
                .call_method1(
                    "predict_decode_latency",
                    (batch_size, mean_context_length, 2),
                )?
                .extract::<f64>()
        })
        .map_err(|error| anyhow!("AIC decode prediction failed: {error}"))
    }
}

fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} {value} exceeds AIC's u32 limit"))
}

fn estimate_aic_num_gpu_blocks(config: &AicTimingConfig, role: &ReplayRoleConfig) -> Result<usize> {
    let (memory_fraction_kind, memory_fraction_value) = config.resolved_memory_fraction()?;
    Python::with_gil(|py| -> PyResult<usize> {
        let memory = PyModule::import(py, "aiconfigurator_core.sdk.memory")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("backend_version", config.resolved_backend_version())?;
        kwargs.set_item("scheduler_block_size", role.rank.block_size)?;
        kwargs.set_item("max_num_tokens", role.rank.max_num_batched_tokens)?;
        kwargs.set_item("max_batch_size", 1)?;
        kwargs.set_item("memory_fraction_kind", memory_fraction_kind)?;
        kwargs.set_item("memory_fraction_value", memory_fraction_value)?;
        kwargs.set_item("tp_size", config.tp)?;
        kwargs.set_item("pp_size", config.pp)?;
        kwargs.set_item("attention_dp_size", config.attention_dp)?;
        kwargs.set_item("moe_tp_size", config.moe_tp_size)?;
        kwargs.set_item("moe_ep_size", config.moe_ep_size)?;
        kwargs.set_item("gemm_quant_mode", config.gemm_dtype.as_deref())?;
        kwargs.set_item("moe_quant_mode", config.moe_dtype.as_deref())?;
        kwargs.set_item("fmha_quant_mode", config.fmha_dtype.as_deref())?;
        kwargs.set_item("kvcache_quant_mode", config.kv_cache_dtype.as_deref())?;
        kwargs.set_item("comm_quant_mode", config.comm_dtype.as_deref())?;
        // Capacity intentionally omits NextN until AIC's Eagle memory model no
        // longer returns negative KV capacity. Timing compilation still uses it.
        kwargs.set_item("systems_path", config.systems_path.as_deref())?;
        memory
            .getattr("estimate_num_gpu_blocks")?
            .call(
                (
                    config.model.as_str(),
                    config.system.as_str(),
                    config.backend.as_str(),
                ),
                Some(&kwargs),
            )?
            .extract()
    })
    .map_err(|error| anyhow!("AIC KV-cache capacity estimation failed: {error}"))
}

fn materialize_aic_capacity(
    config: &AicTimingConfig,
    role: &mut ReplayRoleConfig,
    capacity_is_explicit: bool,
    estimate: impl FnOnce(&AicTimingConfig, &ReplayRoleConfig) -> Result<usize>,
) -> Result<()> {
    config.validate_parallel_shape()?;
    let engine_backend = match role.rank.backend {
        Backend::Vllm => "vllm",
        Backend::Sglang => "sglang",
        Backend::Trtllm => "trtllm",
    };
    ensure!(
        config.backend == engine_backend,
        "AIC backend {:?} does not match engine backend {engine_backend:?}",
        config.backend
    );
    ensure!(
        config.tp == role.tensor_parallel_size,
        "AIC tp={} does not match engine tensor_parallel_size={}",
        config.tp,
        role.tensor_parallel_size
    );
    ensure!(
        config.attention_dp == role.dp_size,
        "AIC attention_dp={} does not match engine dp_size={}",
        config.attention_dp,
        role.dp_size
    );
    ensure!(
        config
            .kv_block_size
            .is_none_or(|block_size| block_size as usize == role.rank.block_size),
        "AIC kv_block_size does not match engine block_size={}",
        role.rank.block_size
    );
    let engine_nextn = role.rank.aic_nextn.unwrap_or(0);
    ensure!(
        config.nextn as usize == engine_nextn,
        "AIC nextn={} does not match engine aic_nextn={engine_nextn}",
        config.nextn
    );
    if capacity_is_explicit {
        return Ok(());
    }
    let blocks = estimate(config, role)?;
    ensure!(blocks > 0, "AIC estimated zero KV-cache blocks");
    role.rank.num_gpu_blocks = blocks;
    Ok(())
}

fn aggregated_role(engine: &ReplayEngineConfig) -> ReplayRoleConfig {
    ReplayRoleConfig {
        dp_size: engine.dp_size,
        tensor_parallel_size: engine.tensor_parallel_size,
        rank: engine.rank.clone(),
    }
}

fn role_capacity_is_explicit(engine_value: &serde_json::Value, role: Option<&str>) -> bool {
    let role_rank = role
        .and_then(|role| engine_value.get(role))
        .and_then(|role| role.get("rank"));
    role_rank
        .or_else(|| engine_value.get("rank"))
        .and_then(serde_json::Value::as_object)
        .is_some_and(|rank| rank.contains_key("num_gpu_blocks"))
}

fn resolve_role_timing(
    role: &mut ReplayRoleConfig,
    capacity_is_explicit: bool,
) -> Result<Option<Arc<dyn TimingModel>>> {
    let TimingModelConfig::External { provider, config } = role.rank.timing_model.clone() else {
        return Ok(None);
    };
    ensure!(
        provider == "aic",
        "native timing provider {provider:?} is not installed; only \"aic\" is \
         available in the AISimulate runtime"
    );
    let config: AicTimingConfig =
        serde_json::from_value(config).context("invalid AIC timing provider configuration")?;
    materialize_aic_capacity(
        &config,
        role,
        capacity_is_explicit,
        estimate_aic_num_gpu_blocks,
    )?;
    Ok(Some(Arc::new(AicTimingModel::build(config)?)))
}

fn execute_json(payload: &str) -> Result<String> {
    let mut spec: ReplaySpec =
        serde_json::from_str(payload).context("invalid AISimulate execution ReplaySpec")?;
    let serialized_engine = spec.engine.clone();
    let mut engine_config: ReplayEngineConfig = if spec.engine.is_null() {
        ReplayEngineConfig::default()
    } else {
        serde_json::from_value(spec.engine.clone())
            .context("invalid native engine descriptor in execution ReplaySpec")?
    };

    let report = match spec.topology.clone() {
        ReplayTopology::Aggregated { .. } => {
            let mut role = aggregated_role(&engine_config);
            let timing = resolve_role_timing(
                &mut role,
                role_capacity_is_explicit(&serialized_engine, None),
            )?;
            engine_config.dp_size = role.dp_size;
            engine_config.tensor_parallel_size = role.tensor_parallel_size;
            engine_config.rank = role.rank;
            spec.engine = serde_json::to_value(&engine_config)
                .context("serializing materialized native engine descriptor")?;
            match timing {
                Some(timing) => run_engine_replay_with_timing(spec, timing),
                None => run_engine_replay(spec),
            }
        }
        ReplayTopology::Disaggregated { .. } => {
            let mut prefill = engine_config
                .prefill
                .clone()
                .unwrap_or_else(|| aggregated_role(&engine_config));
            let mut decode = engine_config
                .decode
                .clone()
                .unwrap_or_else(|| aggregated_role(&engine_config));
            let prefill_timing = resolve_role_timing(
                &mut prefill,
                role_capacity_is_explicit(&serialized_engine, Some("prefill")),
            )?;
            let decode_timing = resolve_role_timing(
                &mut decode,
                role_capacity_is_explicit(&serialized_engine, Some("decode")),
            )?;
            engine_config.prefill = Some(prefill);
            engine_config.decode = Some(decode);
            spec.engine = serde_json::to_value(&engine_config)
                .context("serializing materialized native engine descriptor")?;
            run_engine_replay_with_optional_role_timing(spec, prefill_timing, decode_timing)
        }
    }
    .context("AISimulate replay failed")?;
    let mut serialized =
        serde_json::to_value(&report).context("serializing AISimulate replay report summary")?;
    if !report.per_request.is_empty() {
        let object = serialized
            .as_object_mut()
            .context("AISimulate replay report did not serialize as an object")?;
        object.insert(
            "per_request".to_string(),
            serde_json::to_value(&report.per_request)
                .context("serializing AISimulate per-request report records")?,
        );
    }
    serde_json::to_string(&serialized).context("serializing AISimulate replay report")
}

/// Execute one canonical serialized ReplaySpec and return serialized report JSON.
#[pyfunction]
fn run_replay_json(py: Python<'_>, payload: &str) -> PyResult<String> {
    py.allow_threads(|| execute_json(payload))
        .map_err(|error| PyRuntimeError::new_err(format!("{error:#}")))
}

/// AISimulate native runtime module.
#[pymodule]
fn _runtime(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_replay_json, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use aisimulate_core::engine::{EngineConfig, TimingModelConfig};
    use aisimulate_core::replay::{
        ProviderSpec, ReplayAdapters, ReplayEngineConfig, ReplayRequest, ReplaySpec,
        ReplayTopology, WorkerPoolSpec,
    };

    use super::*;

    fn aic_config() -> AicTimingConfig {
        AicTimingConfig {
            model: "test-model".into(),
            backend: "vllm".into(),
            system: "test-system".into(),
            tp: 1,
            backend_version: None,
            pp: 1,
            attention_dp: 1,
            moe_tp_size: None,
            moe_ep_size: None,
            gemm_dtype: None,
            moe_dtype: None,
            fmha_dtype: None,
            kv_cache_dtype: None,
            comm_dtype: None,
            nextn: 0,
            kv_block_size: None,
            gpu_memory_utilization: None,
            mem_fraction_static: None,
            free_gpu_memory_fraction: None,
            systems_path: None,
        }
    }

    #[test]
    fn capacity_is_estimated_only_when_not_explicit() {
        let mut role = aggregated_role(&ReplayEngineConfig::default());
        materialize_aic_capacity(&aic_config(), &mut role, false, |_config, rank| {
            assert_eq!(rank.rank.block_size, EngineConfig::default().block_size);
            Ok(321)
        })
        .unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 321);

        role.rank.num_gpu_blocks = 17;
        materialize_aic_capacity(
            &aic_config(),
            &mut role,
            true,
            |_config, _rank| -> Result<usize> {
                panic!("explicit capacity must not invoke the estimator")
            },
        )
        .unwrap();
        assert_eq!(role.rank.num_gpu_blocks, 17);
    }

    #[test]
    fn capacity_is_detected_independently_per_role() {
        let engine = serde_json::json!({
            "prefill": {"rank": {"num_gpu_blocks": 17}},
            "decode": {"rank": {}}
        });
        assert!(role_capacity_is_explicit(&engine, Some("prefill")));
        assert!(!role_capacity_is_explicit(&engine, Some("decode")));
    }

    #[test]
    fn invalid_memory_fraction_is_rejected() {
        let mut config = aic_config();
        config.gpu_memory_utilization = Some(1.1);
        assert!(
            config
                .resolved_memory_fraction()
                .unwrap_err()
                .to_string()
                .contains("gpu_memory_utilization")
        );
    }

    #[test]
    fn json_binding_retains_authored_per_request_correlation() {
        let spec = ReplaySpec {
            version: 1,
            topology: ReplayTopology::Aggregated {
                workers: WorkerPoolSpec::default(),
            },
            engine: serde_json::to_value(ReplayEngineConfig {
                rank: EngineConfig {
                    num_gpu_blocks: 16,
                    block_size: 4,
                    max_num_seqs: 4,
                    max_num_batched_tokens: 64,
                    timing_model: TimingModelConfig::Fixed {
                        prefill_ms: 1.0,
                        decode_ms: 1.0,
                    },
                    ..EngineConfig::default()
                },
                ..ReplayEngineConfig::default()
            })
            .unwrap(),
            adapters: ReplayAdapters {
                placement: ProviderSpec::round_robin(),
                scaling: ProviderSpec::no_scaling(),
            },
            max_sim_time_ms: None,
            max_in_flight: None,
            record_per_request: true,
            sla: Default::default(),
            requests: vec![ReplayRequest {
                id: "authored-id".into(),
                arrival_time_ms: 0.0,
                input_tokens: 4,
                input_token_ids: Some(vec![1, 2, 3, 4]),
                output_tokens: 1,
                output_token_ids: None,
                dp_rank: None,
                session_id: Some("session-a".into()),
                turn_index: Some(2),
                metadata: serde_json::json!({"caller_tag": "binding"}),
            }],
        };

        let output = execute_json(&serde_json::to_string(&spec).unwrap()).unwrap();
        let report: serde_json::Value = serde_json::from_str(&output).unwrap();
        let record = &report["per_request"][0];
        assert_eq!(record["request_id"], "authored-id");
        assert_eq!(record["session_id"], "session-a");
        assert_eq!(record["turn_index"], 2);
        assert_eq!(record["metadata"]["caller_tag"], "binding");
    }
}
