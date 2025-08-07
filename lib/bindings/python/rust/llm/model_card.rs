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
use llm_rs::model_card::model::ModelDeploymentCard as RsModelDeploymentCard;
use llm_rs::model_card::runtime_config::ModelRuntimeConfig as RsModelRuntimeConfig;

#[pyclass]
#[derive(Clone)]
pub(crate) struct ModelDeploymentCard {
    pub(crate) inner: RsModelDeploymentCard,
}

impl ModelDeploymentCard {}

#[pymethods]
impl ModelDeploymentCard {
    // Previously called "from_local_path"
    #[staticmethod]
    fn load(path: String, model_name: String, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut card = RsModelDeploymentCard::load(&path).await.map_err(to_pyerr)?;
            card.set_name(&model_name);
            Ok(ModelDeploymentCard { inner: card })
        })
    }

    #[staticmethod]
    fn from_json_str(json: String) -> PyResult<ModelDeploymentCard> {
        let card = RsModelDeploymentCard::load_from_json_str(&json).map_err(to_pyerr)?;
        Ok(ModelDeploymentCard { inner: card })
    }

    fn to_json_str(&self) -> PyResult<String> {
        let json = self.inner.to_json().map_err(to_pyerr)?;
        Ok(json)
    }

    #[setter]
    fn register_runtime_config(&mut self, runtime_config: ModelRuntimeConfig) {
        self.inner.runtime_config = Some(runtime_config.inner);
    }

    #[getter]
    fn runtime_config(&self) -> Option<ModelRuntimeConfig> {
        self.inner
            .runtime_config
            .as_ref()
            .map(|config| ModelRuntimeConfig {
                inner: config.clone(),
            })
    }

    #[getter]
    fn total_kv_blocks(&self) -> Option<u64> {
        self.inner
            .runtime_config
            .as_ref()
            .map(|config| config.total_kv_blocks)
            .flatten()
    }

    #[getter]
    fn max_num_seqs(&self) -> Option<u64> {
        self.inner
            .runtime_config
            .as_ref()
            .map(|config| config.max_num_seqs)
            .flatten()
    }

    #[getter]
    fn gpu_memory_utilization(&self) -> Option<u64> {
        self.inner
            .runtime_config
            .as_ref()
            .map(|config| config.gpu_memory_utilization)
            .flatten()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ModelRuntimeConfig {
    pub(crate) inner: RsModelRuntimeConfig,
}

#[pymethods]
impl ModelRuntimeConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsModelRuntimeConfig::new(),
        }
    }

    #[setter]
    fn set_total_kv_blocks(&mut self, total_kv_blocks: u64) {
        self.inner.total_kv_blocks = Some(total_kv_blocks);
    }

    #[setter]
    fn set_max_num_seqs(&mut self, max_num_seqs: u64) {
        self.inner.max_num_seqs = Some(max_num_seqs);
    }

    #[setter]
    fn set_gpu_memory_utilization(&mut self, gpu_memory_utilization: u64) {
        self.inner.gpu_memory_utilization = Some(gpu_memory_utilization);
    }

    fn set_engine_specific(&mut self, key: &str, value: String) -> PyResult<()> {
        let value: serde_json::Value = serde_json::from_str(&value).map_err(to_pyerr)?;
        self.inner
            .set_engine_specific(key, value)
            .map_err(to_pyerr)?;
        Ok(())
    }

    #[getter]
    fn total_kv_blocks(&self) -> Option<u64> {
        self.inner.total_kv_blocks
    }

    #[getter]
    fn max_num_seqs(&self) -> Option<u64> {
        self.inner.max_num_seqs
    }

    #[getter]
    fn gpu_memory_utilization(&self) -> Option<u64> {
        self.inner.gpu_memory_utilization
    }

    #[getter]
    fn runtime_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in self.inner.runtime_data.clone() {
            dict.set_item(key, value.to_string())?;
        }
        Ok(dict.into())
    }

    fn get_engine_specific(&self, key: &str) -> PyResult<Option<String>> {
        self.inner.get_engine_specific(key).map_err(to_pyerr)
    }
}
