// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical AISimulate model construction and pure Rust scheduler adapters.
#[cfg(feature = "ais-forward-pass")]
use aisimulate_core::{
    ForwardPassMetrics, ForwardPassPerfModel, ForwardPassPerfModelConfig, ForwardPassPerfReadiness,
    ScheduledRequestMetrics,
};
use dynamo_mocker::common::perf_model::AisCallback;
use dynamo_scheduler::host::PrefillLoadEstimator;
use pyo3::prelude::*;
use std::sync::Arc;
#[cfg(feature = "ais-forward-pass")]
use std::time::Duration;

#[cfg(feature = "ais-forward-pass")]
pub(super) struct RustAisCallback {
    model: ForwardPassPerfModel,
    nextn: u32,
}

/// A sweep varies a handful of engine configurations; router settings do not change the model.
#[cfg(feature = "ais-forward-pass")]
const REPLAY_MODEL_CAPACITY: usize = 8;

#[cfg(feature = "ais-forward-pass")]
static REPLAY_MODELS: parking_lot::Mutex<RecentCache<Arc<RustAisCallback>>> =
    parking_lot::Mutex::new(RecentCache::new(REPLAY_MODEL_CAPACITY));

/// Most recently used first; the oldest entry is dropped at capacity.
#[cfg(any(test, feature = "ais-forward-pass"))]
struct RecentCache<T> {
    entries: Vec<(String, T)>,
    capacity: usize,
}

#[cfg(any(test, feature = "ais-forward-pass"))]
impl<T: Clone> RecentCache<T> {
    const fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
        }
    }

    fn get(&mut self, key: &str) -> Option<T> {
        let index = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(index);
        let value = entry.1.clone();
        self.entries.insert(0, entry);
        Some(value)
    }

    fn insert(&mut self, key: String, value: T) {
        self.entries.insert(0, (key, value));
        self.entries.truncate(self.capacity);
    }
}

/// The model for `config`: a kept one, or the one `build` returns.
#[cfg(any(test, feature = "ais-forward-pass"))]
fn kept_or_built<T: Clone, E>(
    models: &parking_lot::Mutex<RecentCache<T>>,
    config: &serde_json::Value,
    build: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    if !is_replay_cacheable(config) {
        return build();
    }
    let key = config.to_string();
    if let Some(model) = models.lock().get(&key) {
        return Ok(model);
    }
    // Built outside the lock, because a build parses perf tables. Callers that miss together
    // each build, and the first to finish is kept.
    let model = build()?;
    let mut models = models.lock();
    if let Some(kept) = models.get(&key) {
        return Ok(kept);
    }
    models.insert(key, model.clone());
    Ok(model)
}

/// A profile, snapshot or other estimator input can be rewritten in place between replays, so
/// only a model that reads nothing but its systems root is kept.
#[cfg(any(test, feature = "ais-forward-pass"))]
fn is_replay_cacheable(config: &serde_json::Value) -> bool {
    let is_unset = |key: &str| match config.get(key) {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::Object(map)) => map.is_empty(),
        Some(_) => false,
    };
    is_unset("fpm_profile") && is_unset("estimator_config")
}

#[cfg(feature = "ais-forward-pass")]
fn checked_count(value: usize, field: &str) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("AIS {field} exceeds u32"))
}

#[cfg(feature = "ais-forward-pass")]
fn prefill_metrics(
    batch_size: usize,
    effective_isl: usize,
    prefix: usize,
) -> anyhow::Result<ForwardPassMetrics> {
    let batch = checked_count(batch_size, "batch size")?;
    let tokens = checked_count(effective_isl, "prefill tokens")?;
    let prefix = checked_count(prefix, "prefix tokens")?;
    Ok(ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: batch,
            sum_prefill_tokens: batch
                .checked_mul(tokens)
                .ok_or_else(|| anyhow::anyhow!("AIS prefill total exceeds u32"))?,
            sum_prefill_kv_tokens: batch
                .checked_mul(prefix)
                .ok_or_else(|| anyhow::anyhow!("AIS prefix total exceeds u32"))?,
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(feature = "ais-forward-pass")]
fn decode_metrics(
    batch_size: usize,
    context: usize,
    nextn: u32,
) -> anyhow::Result<ForwardPassMetrics> {
    // FPM counts are already packed. Preserve the former static-engine adapter's
    // verification width explicitly, without applying attention-DP a second time.
    let batch = checked_count(batch_size, "batch size")?
        .checked_mul(nextn + 1)
        .ok_or_else(|| anyhow::anyhow!("AIS verification batch exceeds u32"))?;
    let context = checked_count(context, "decode context")?;
    Ok(ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: batch,
            sum_decode_kv_tokens: batch
                .checked_mul(context)
                .ok_or_else(|| anyhow::anyhow!("AIS decode KV total exceeds u32"))?,
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(feature = "ais-forward-pass")]
impl RustAisCallback {
    fn estimate(&self, metrics: ForwardPassMetrics) -> anyhow::Result<f64> {
        let value = self
            .model
            .estimate_forward_pass_time_ms(&[metrics])?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "AIS estimator is not ready; Router/Mocker has no FPM training source"
                )
            })?;
        anyhow::ensure!(
            value.is_finite() && value >= 0.0,
            "AIS returned invalid latency {value}"
        );
        Ok(value)
    }
}

#[cfg(feature = "ais-forward-pass")]
impl AisCallback for RustAisCallback {
    fn predict_prefill(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<f64> {
        self.estimate(prefill_metrics(batch_size, effective_isl, prefix)?)
    }
    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> anyhow::Result<f64> {
        // Legacy API returns a trajectory sum. Mocker uses osl=2, i.e. one
        // generation step at isl+1; preserve this exact context convention.
        let mut total = 0.0;
        let stride = aisimulate_core::perfmodel::engine::DEFAULT_STATIC_STRIDE as usize;
        for step in (1..osl).step_by(stride) {
            let context = isl
                .checked_add(step)
                .ok_or_else(|| anyhow::anyhow!("AIS decode context overflow"))?;
            total += self.estimate(decode_metrics(batch_size, context, self.nextn)?)?
                * (osl - step).min(stride) as f64;
        }
        Ok(total)
    }
}

#[cfg(feature = "ais-forward-pass")]
impl PrefillLoadEstimator for RustAisCallback {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<Duration> {
        Ok(Duration::try_from_secs_f64(
            self.predict_prefill(batch_size, effective_isl, prefix)? / 1000.0,
        )?)
    }
}

#[cfg(feature = "ais-forward-pass")]
fn build_model(config: &serde_json::Value) -> PyResult<RustAisCallback> {
    let config: ForwardPassPerfModelConfig =
        serde_json::from_value(config.clone()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid AIS perf config: {e}"))
        })?;
    let nextn = match config.speculation.as_ref() {
        None => config.nextn,
        Some(spec) if spec.kind == "ngram" => spec
            .params
            .get("num_speculative_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|depth| u32::try_from(depth).ok())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "ngram num_speculative_tokens must be a nonnegative u32",
                )
            })?,
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Router/Mocker supports only ngram speculation or legacy nextn",
            ));
        }
    };
    let model = ForwardPassPerfModel::best_available(config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("AIS model construction failed: {e}"))
    })?;
    if model.diagnostics().readiness != ForwardPassPerfReadiness::Ready {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "AIS estimator is not ready; Router/Mocker requires a ready model because it has no FPM training source",
        ));
    }
    if nextn > 0
        && model.provenance().is_some_and(|p| {
            p.selected_estimation_mode == aisimulate_core::EstimationMode::FpmInterpolation
        })
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "AIS canonical FPM queries do not yet support speculative FPM interpolation; use op_level",
        ));
    }
    Ok(RustAisCallback { model, nextn })
}

#[cfg_attr(not(feature = "ais-forward-pass"), allow(unused_variables))]
pub(super) fn create_ais_callback(
    _py: Python<'_>,
    config: &serde_json::Value,
) -> PyResult<Arc<dyn AisCallback>> {
    #[cfg(feature = "ais-forward-pass")]
    {
        Ok(Arc::new(build_model(config)?))
    }
    #[cfg(not(feature = "ais-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIS perf model requires the ais-forward-pass feature",
    ))
}

/// [`create_ais_callback`] for replay. A sweep replays the same few engine configurations many
/// times and building a model parses its perf tables each time, so replays share the built
/// model. Data under a systems root is taken as fixed for the life of the process.
#[cfg_attr(not(feature = "ais-forward-pass"), allow(unused_variables))]
pub(super) fn create_replay_ais_callback(
    _py: Python<'_>,
    config: &serde_json::Value,
) -> PyResult<Arc<dyn AisCallback>> {
    #[cfg(feature = "ais-forward-pass")]
    {
        let model = kept_or_built(&REPLAY_MODELS, config, || build_model(config).map(Arc::new))?;
        Ok(model)
    }
    #[cfg(not(feature = "ais-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIS perf model requires the ais-forward-pass feature",
    ))
}

#[cfg_attr(not(feature = "ais-forward-pass"), allow(unused_variables))]
pub(super) fn create_ais_prefill_load_estimator(
    _py: Python<'_>,
    config: &serde_json::Value,
) -> PyResult<Arc<dyn PrefillLoadEstimator>> {
    #[cfg(feature = "ais-forward-pass")]
    {
        Ok(Arc::new(build_model(config)?))
    }
    #[cfg(not(feature = "ais-forward-pass"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "AIS perf model requires the ais-forward-pass feature",
    ))
}

#[cfg(test)]
mod replay_cache_tests {
    use super::{RecentCache, kept_or_built};
    use parking_lot::Mutex;
    use serde_json::{Value, json};

    /// The value is the number of the build that produced it.
    fn model(models: &Mutex<RecentCache<u32>>, config: &Value, builds: &mut u32) -> u32 {
        kept_or_built(models, config, || {
            *builds += 1;
            Ok::<_, ()>(*builds)
        })
        .unwrap()
    }

    #[test]
    fn a_config_is_built_once_and_each_config_separately() {
        let models = Mutex::new(RecentCache::new(4));
        let a = json!({"model": "a", "estimator_config": {}});
        let b = json!({"model": "b", "fpm_profile": null});
        let mut builds = 0;
        assert_eq!(model(&models, &a, &mut builds), 1);
        assert_eq!(model(&models, &a, &mut builds), 1);
        assert_eq!(model(&models, &b, &mut builds), 2);
        assert_eq!(model(&models, &a, &mut builds), 1);
    }

    #[test]
    fn a_config_with_caller_owned_inputs_is_built_every_time() {
        for config in [
            json!({"model": "a", "fpm_profile": {"path": "/tmp/profile"}}),
            json!({"model": "a", "estimator_config": {"state_snapshot": {"path": "/tmp/s"}}}),
        ] {
            let models = Mutex::new(RecentCache::new(4));
            let mut builds = 0;
            model(&models, &config, &mut builds);
            model(&models, &config, &mut builds);
            assert_eq!(builds, 2, "{config}");
        }
    }

    #[test]
    fn the_least_recently_used_model_is_dropped_and_a_failed_build_is_not_kept() {
        let models = Mutex::new(RecentCache::new(2));
        let config = |name: &str| json!({ "model": name });
        let mut builds = 0;
        for name in ["a", "b", "a", "c"] {
            model(&models, &config(name), &mut builds);
        }
        assert_eq!(model(&models, &config("a"), &mut builds), 1);
        assert_eq!(
            model(&models, &config("b"), &mut builds),
            4,
            "c pushed b out"
        );

        let failed: Result<u32, &str> = kept_or_built(&models, &config("d"), || Err("no data"));
        assert_eq!(failed, Err("no data"));
        assert_eq!(model(&models, &config("d"), &mut builds), 5);
    }
}

#[cfg(all(test, feature = "ais-forward-pass"))]
mod tests {
    use super::{decode_metrics, prefill_metrics};

    #[test]
    fn scheduler_queries_preserve_prefix_and_local_attention_dp_batch() {
        let prefill = prefill_metrics(7, 96, 32).unwrap().scheduled_requests;
        assert_eq!(prefill.num_prefill_requests, 7);
        assert_eq!(prefill.sum_prefill_tokens, 672);
        assert_eq!(prefill.sum_prefill_kv_tokens, 224);
        let decode = decode_metrics(7, 129, 2).unwrap().scheduled_requests;
        assert_eq!(decode.num_decode_requests, 21);
        assert_eq!(decode.sum_decode_kv_tokens, 2709);
    }

    #[test]
    fn oversized_scheduler_work_is_rejected_instead_of_wrapping() {
        assert!(prefill_metrics(u32::MAX as usize, 2, 0).is_err());
        assert!(prefill_metrics(u32::MAX as usize, 1, 2).is_err());
        assert!(decode_metrics(u32::MAX as usize, 1, 1).is_err());
        assert!(decode_metrics(u32::MAX as usize, 2, 0).is_err());
    }
}
