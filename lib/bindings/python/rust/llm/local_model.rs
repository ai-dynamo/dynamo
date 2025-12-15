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

    #[setter]
    fn set_tool_call_parser(&mut self, tool_call_parser: Option<String>) {
        self.inner.tool_call_parser = tool_call_parser;
    }

    #[setter]
    fn set_reasoning_parser(&mut self, reasoning_parser: Option<String>) {
        self.inner.reasoning_parser = reasoning_parser;
    }

    #[setter]
    fn set_data_parallel_size(&mut self, data_parallel_size: u32) {
        self.inner.data_parallel_size = data_parallel_size;
    }

    #[setter]
    fn set_enable_local_indexer(&mut self, enable_local_indexer: bool) {
        self.inner.enable_local_indexer = enable_local_indexer;
    }

    fn set_engine_specific(&mut self, key: &str, value: String) -> PyResult<()> {
        let value: serde_json::Value = serde_json::from_str(&value).map_err(to_pyerr)?;
        self.inner
            .set_engine_specific(key, value)
            .map_err(to_pyerr)?;
        Ok(())
    }

    fn set_tensor_model_config(
        &mut self,
        _py: Python<'_>,
        tensor_model_config: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let tensor_model_config = pythonize::depythonize(tensor_model_config).map_err(|err| {
            PyErr::new::<PyException, _>(format!("Failed to convert tensor_model_config: {}", err))
        })?;
        self.inner.tensor_model_config = Some(tensor_model_config);
        Ok(())
    }

    fn get_tensor_model_config(&self, _py: Python<'_>) -> PyResult<Option<PyObject>> {
        if let Some(tensor_model_config) = &self.inner.tensor_model_config {
            let py_obj = pythonize::pythonize(_py, tensor_model_config).map_err(to_pyerr)?;
            Ok(Some(py_obj.unbind()))
        } else {
            Ok(None)
        }
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
    fn tool_call_parser(&self) -> Option<String> {
        self.inner.tool_call_parser.clone()
    }

    #[getter]
    fn reasoning_parser(&self) -> Option<String> {
        self.inner.reasoning_parser.clone()
    }

    #[getter]
    fn enable_local_indexer(&self) -> bool {
        self.inner.enable_local_indexer
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
