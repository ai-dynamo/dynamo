// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::Stream;
use pyo3::prelude::*;
use pyo3_async_runtimes::{into_future_with_locals, TaskLocals};
use tokio::sync::mpsc;

pub fn into_stream(
    locals: TaskLocals,
    gen_: Bound<'_, PyAny>,
) -> PyResult<impl Stream<Item = PyResult<PyObject>> + 'static> {
    let (tx, rx) = mpsc::channel(1);
    let anext = PyObject::from(gen_.getattr("__anext__")?);
    let py_gen = PyObject::from(gen_); // Store the generator for athrow call

    tokio::spawn(async move {
        loop {
            let fut = Python::with_gil(|py| -> PyResult<_> {
                into_future_with_locals(&locals, anext.bind(py).call0()?)
            });
            let item = match fut {
                Ok(fut) => match fut.await {
                    Ok(item) => Ok(item),
                    Err(e) => {
                        let stop_iter = Python::with_gil(|py| {
                            e.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py)
                        });

                        if stop_iter {
                            // end the iteration
                            break;
                        } else {
                            Err(e)
                        }
                    }
                },
                Err(e) => Err(e),
            };

            if tx.send(item).await.is_err() {
                tracing::debug!("Stream receiver dropped");

                // Cancel the Python async generator object
                // https://peps.python.org/pep-0525/#asynchronous-generator-object
                let athrow = Python::with_gil(|py| -> PyResult<_> {
                    let cancelled_error = pyo3::exceptions::asyncio::CancelledError::new_err(());
                    let athrow_coro = py_gen.call_method1(py, "athrow", (cancelled_error,))?;
                    into_future_with_locals(&locals, athrow_coro.into_bound(py))
                });
                match athrow {
                    Ok(fut) => match fut.await {
                        Ok(_) => {
                            tracing::debug!("Generator cancelled: no further exception raised")
                        }
                        Err(err) => tracing::debug!("Generator cancelled: {}", err),
                    },
                    Err(err) => tracing::error!("Failed to call athrow on the generator: {}", err),
                }

                break;
            }
        }
    });

    Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
}
