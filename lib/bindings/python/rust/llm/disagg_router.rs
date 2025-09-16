// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[pyclass]
pub struct DisaggregatedRouter {
    inner: Arc<dynamo_llm::disagg_router::DisaggregatedRouter>,
}

#[pymethods]
impl DisaggregatedRouter {
    #[new]
    #[pyo3(signature = (drt, model_name, default_max_local_prefill_length))]
    fn new(
        drt: PyObject,
        model_name: String,
        default_max_local_prefill_length: i32,
    ) -> PyResult<Self> {
        let drt_arc = Python::with_gil(|py| {
            let drt_ref = drt.extract::<DistributedRuntime>(py)?;
            Ok::<_, PyErr>(Arc::new(drt_ref.inner))
        })?;

        // Create the runtime directly with the correct import
        let runtime = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create tokio runtime: {}", e))
        })?;

        let router = runtime.block_on(async {
            dynamo_llm::disagg_router::DisaggregatedRouter::new_with_etcd_and_default(
                drt_arc,
                model_name,
                default_max_local_prefill_length,
            )
            .await
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create DisaggregatedRouter: {}", e))
            })
        })?;

        Ok(DisaggregatedRouter {
            inner: Arc::new(router),
        })
    }

    fn prefill_remote(&self, prefill_length: i32, prefix_hit_length: i32) -> bool {
        self.inner.prefill_remote(prefill_length, prefix_hit_length)
    }

    fn get_model_name(&self) -> &str {
        self.inner.get_model_name()
    }
}
