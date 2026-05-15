// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use kvbm_connector::RequestMetadata;

#[derive(Clone)]
#[pyclass(name = "KvbmRequest")]
pub struct PyRequest {
    pub(crate) inner: Request,
}

#[pymethods]
impl PyRequest {
    #[new]
    #[pyo3(signature = (
        request_id,
        tokens,
        lora_name=None,
        salt_hash=None,
        max_tokens=None,
        kv_transfer_params_json=None,
    ))]
    pub fn new(
        request_id: String,
        tokens: Vec<usize>,
        lora_name: Option<String>,
        salt_hash: Option<String>,
        max_tokens: Option<usize>,
        kv_transfer_params_json: Option<String>,
    ) -> PyResult<Self> {
        if max_tokens.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_tokens is required",
            ));
        }

        let metadata = match kv_transfer_params_json {
            Some(s) => {
                let value: serde_json::Value = serde_json::from_str(&s).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "invalid kv_transfer_params JSON: {e}"
                    ))
                })?;
                Some(RequestMetadata::with_kv_transfer_params(value))
            }
            None => None,
        };

        let inner = Request::with_token_limits(
            request_id, tokens, lora_name, salt_hash, None, max_tokens, metadata,
        );

        Ok(Self { inner })
    }

    #[getter]
    pub fn request_id(&self) -> &str {
        &self.inner.request_id
    }
}

impl From<&PyRequest> for Request {
    fn from(value: &PyRequest) -> Self {
        value.inner.clone()
    }
}
