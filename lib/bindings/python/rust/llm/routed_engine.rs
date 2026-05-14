// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use tokio_stream::StreamExt;

use dynamo_llm::entrypoint::PrefillRoutedEngine;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;

use crate::to_pyerr;

#[pyclass]
pub struct RoutedEngine {
    inner: PrefillRoutedEngine,
}

impl RoutedEngine {
    pub fn new(inner: PrefillRoutedEngine) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl RoutedEngine {
    /// Send a preprocessed request through the Rust prefill-routed pipeline.
    #[pyo3(signature = (preprocessed, context=None))]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        preprocessed: PyObject,
        context: Option<crate::context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let request: PreprocessedRequest = depythonize(preprocessed.bind(py)).map_err(to_pyerr)?;
        let request_context = if let Some(parent_context) = context.as_ref() {
            let parent_context = parent_context.inner();
            let child_context = SingleIn::with_id(request, parent_context.id().to_string());
            let child_controller = child_context.context();
            parent_context.link_child(child_controller.clone());
            if parent_context.is_killed() {
                child_controller.kill();
            } else if parent_context.is_stopped() {
                child_controller.stop_generating();
            }
            child_context
        } else {
            SingleIn::new(request)
        };
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut stream = inner.generate(request_context).await.map_err(to_pyerr)?;
            let task_context = stream.context();
            let (tx, rx) = tokio::sync::mpsc::channel::<RsAnnotated<PyObject>>(32);

            tokio::spawn(async move {
                loop {
                    let response = tokio::select! {
                        _ = tx.closed() => {
                            task_context.stop_generating();
                            break;
                        }
                        response = stream.next() => response,
                    };

                    let Some(response) = response else {
                        break;
                    };

                    let py_response = Python::with_gil(|py| {
                        response.map_data(|data| {
                            pythonize(py, &data)
                                .map(|obj| obj.unbind())
                                .map_err(|e| format!("pythonize failed: {e}"))
                        })
                    });

                    if tx.send(py_response).await.is_err() {
                        task_context.stop_generating();
                        break;
                    }
                }
            });

            Ok(crate::AsyncResponseStream::new(rx, true))
        })
    }
}
