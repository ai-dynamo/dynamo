// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;

use llm_rs::postprocessor::media::encoders;
use llm_rs::postprocessor::media::encoders::Encoder;

/// Encode raw pixel data into an image format (e.g., PNG).
///
/// Releases the GIL during encoding so other Python threads can run.
#[pyfunction]
#[pyo3(text_signature = "(data, width, height, channels, format)")]
pub fn encode_image<'py>(
    py: Python<'py>,
    data: &[u8],
    width: u32,
    height: u32,
    channels: u8,
    format: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    match format {
        "png" => {
            let encoder = encoders::ImageEncoder;
            let result = py
                .allow_threads(|| encoder.encode(data, width, height, channels))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(PyBytes::new(py, &result))
        }
        other => Err(PyValueError::new_err(format!(
            "Unsupported image format: '{}'. Supported: 'png'",
            other
        ))),
    }
}

/// Encode arbitrary bytes as base64 string.
///
/// Releases the GIL during encoding for large payloads.
#[pyfunction]
#[pyo3(text_signature = "(data)")]
pub fn encode_base64(py: Python<'_>, data: &[u8]) -> String {
    py.allow_threads(|| encoders::encode_base64(data))
}
