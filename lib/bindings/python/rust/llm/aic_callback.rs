// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python↔Rust bridge for the AIC (AI Configurator) perf model.
//!
//! [`PyAicCallback`] wraps a Python `AicSession` object and implements the
//! [`AicCallback`] trait so the Rust mocker scheduler can call AIC latency
//! predictions without knowing about PyO3.

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;

use dynamo_kv_router::PrefillLoadEstimator;
use dynamo_mocker::common::perf_model::AicCallback;

/// Wraps a Python AIC InferenceSession for direct calls from Rust.
///
/// The Python object must expose:
/// - `predict_prefill(batch_size, effective_isl, prefix) -> float`
/// - `predict_decode(batch_size, isl, osl) -> float`
pub(super) struct PyAicCallback {
    pub(super) session: PyObject,
}

// Safety: PyAicCallback is only called via Python::with_gil which acquires the GIL.
unsafe impl Send for PyAicCallback {}
unsafe impl Sync for PyAicCallback {}

impl PyAicCallback {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> PyResult<f64> {
        Python::with_gil(|py| {
            self.session
                .call_method1(py, "predict_prefill", (batch_size, effective_isl, prefix))
                .and_then(|result| result.extract::<f64>(py))
        })
    }
}

impl AicCallback for PyAicCallback {
    fn predict_prefill(&self, batch_size: usize, effective_isl: usize, prefix: usize) -> f64 {
        self.predict_prefill_ms(batch_size, effective_isl, prefix)
            .unwrap_or_else(|e| panic!("AIC predict_prefill failed: {e}"))
    }

    fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> f64 {
        Python::with_gil(|py| {
            self.session
                .call_method1(py, "predict_decode", (batch_size, isl, osl))
                .and_then(|r| r.extract::<f64>(py))
                .unwrap_or_else(|e| panic!("AIC predict_decode failed: {e}"))
        })
    }
}

impl PrefillLoadEstimator for PyAicCallback {
    fn predict_prefill_duration(
        &self,
        batch_size: usize,
        effective_isl: usize,
        prefix: usize,
    ) -> anyhow::Result<Duration> {
        let latency_ms = self.predict_prefill_ms(batch_size, effective_isl, prefix)?;
        Ok(Duration::from_secs_f64(latency_ms / 1000.0))
    }
}

/// Initialize an AIC callback by importing and calling the Python setup function.
///
/// Called once at mocker startup when `--aic-perf-model` is requested.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_aic_callback(
    py: Python<'_>,
    backend_name: &str,
    system: &str,
    model_path: &str,
    tp_size: usize,
    backend_version: Option<&str>,
    moe_tp_size: Option<usize>,
    moe_ep_size: Option<usize>,
    attention_dp_size: Option<usize>,
) -> PyResult<Arc<dyn AicCallback>> {
    let module = py.import("dynamo._internal.aic")?;
    let session = module.call_method1(
        "create_session",
        (
            backend_name,
            system,
            model_path,
            tp_size,
            backend_version,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
        ),
    )?;
    Ok(Arc::new(PyAicCallback {
        session: session.into(),
    }))
}

pub(super) fn create_aic_prefill_load_estimator(
    py: Python<'_>,
    backend_name: &str,
    system: &str,
    model_path: &str,
    tp_size: usize,
    backend_version: Option<&str>,
) -> PyResult<Arc<dyn PrefillLoadEstimator>> {
    let module = py.import("dynamo._internal.aic")?;
    let session = module.call_method1(
        "create_session",
        (backend_name, system, model_path, tp_size, backend_version),
    )?;
    Ok(Arc::new(PyAicCallback {
        session: session.into(),
    }))
}
