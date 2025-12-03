// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[pyclass(name = "KvbmRequest")]
pub struct PyRequest {
    pub(crate) inner: Request,
}

#[pymethods]
impl PyRequest {
    #[new]
    #[pyo3(signature = (request_id, tokens, lora_name=None, salt_hash=None, max_tokens=None))]
    pub fn new(
        request_id: String,
        tokens: Vec<usize>,
        lora_name: Option<String>,
        salt_hash: Option<String>,
        max_tokens: Option<usize>,
    ) -> PyResult<Self> {
        if max_tokens.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_tokens is required",
            ));
        }
        Ok(Self {
            inner: Request::new(request_id, tokens, lora_name, salt_hash, max_tokens),
        })
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
