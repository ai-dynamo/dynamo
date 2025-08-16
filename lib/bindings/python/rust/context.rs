// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PyContext is a wrapper around the AsyncEngineContext to allow for Python bindings.

pub use dynamo_runtime::pipeline::AsyncEngineContext;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

// PyContext is a wrapper around the AsyncEngineContext to allow for Python bindings.
// Not all methods of the AsyncEngineContext are exposed, jsut the primary ones for tracing + cancellation.
// Kept as class, to allow for future expansion if needed.
#[pyclass]
pub struct PyContext {
    pub inner: Arc<dyn AsyncEngineContext>,
}

impl PyContext {
    pub fn new(inner: Arc<dyn AsyncEngineContext>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyContext {
    // sync method of `await async_is_stopped()`
    fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    // sync method of `await async_is_killed()`
    fn is_killed(&self) -> bool {
        self.inner.is_killed()
    }
    // issues a stop generating
    fn stop_generating(&self) {
        self.inner.stop_generating();
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    // allows building a async callback.
    // since async tasks in python get canceled, but memory is not freed in rust.
    // allow for up to 360 seconds for the async task to cycle and free memory.
    // however, calling `is_stopped()` would take a long time, therefore its preferable to have a async method
    #[pyo3(signature = (wait_for=60))]
    fn async_is_stopped<'a>(&self, py: Python<'a>, wait_for: u16) -> PyResult<Bound<'a, PyAny>> {
        let inner = self.inner.clone();
        // allow wait_for to be 360 seconds max
        if !(1..=360).contains(&wait_for) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "wait_for must be between 1 and 360 seconds to allow for async task to cycle.",
            ));
        }

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Wait up to `wait_for` seconds for inner.stopped() to complete.
            if inner.is_stopped() {
                return Ok(true);
            }
            let _ = timeout(Duration::from_secs(wait_for as u64), inner.stopped()).await;

            Ok(inner.is_stopped() || inner.is_killed())
        })
    }
}
