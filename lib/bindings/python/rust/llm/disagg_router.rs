// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[pyclass]
pub struct DisaggregatedRouter {
    inner: Arc<dynemo_llm::disagg_router::DisaggregatedRouter>,
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
            dynemo_llm::disagg_router::DisaggregatedRouter::new_with_etcd_and_default(
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
