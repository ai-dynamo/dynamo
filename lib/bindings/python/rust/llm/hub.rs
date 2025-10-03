// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::PathBuf;

/// Download a model from HuggingFace using ModelExpress client.
/// 
/// This function attempts to download the model through the ModelExpress server first.
/// If the server is unavailable or fails, it falls back to direct download.
/// 
/// Args:
///     model_name: The HuggingFace model identifier (e.g., "meta-llama/Llama-3.3-70B-Instruct")
///     ignore_weights: If True, skip downloading model weight files
/// 
/// Returns:
///     The local filesystem path to the downloaded model
#[pyfunction]
#[pyo3(signature = (model_name, ignore_weights=false))]
fn from_hf<'p>(
    py: Python<'p>,
    model_name: String,
    ignore_weights: bool,
) -> PyResult<Bound<'p, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let path = dynamo_llm::hub::from_hf(&model_name, ignore_weights)
            .await
            .map_err(to_pyerr)?;
        
        Ok(path.to_string_lossy().to_string())
    })
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_hf, m)?)?;
    Ok(())
}

