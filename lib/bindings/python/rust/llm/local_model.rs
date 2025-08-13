// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use llm_rs::local_model::runtime_config::ModelRuntimeConfig as RsModelRuntimeConfig;

#[pyclass]
#[derive(Clone, Default)]
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
    fn set_max_num_batched_tokens(&mut self, max_num_batched_tokens: u64) {
        self.inner.max_num_batched_tokens = Some(max_num_batched_tokens);
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
    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.inner.max_num_batched_tokens
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
