// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep GIL acquisition off the Tokio runtime workers.
//!
//! A Tokio worker that calls `Python::with_gil` while another thread holds the GIL
//! blocks, and every task queued on that worker waits with it. On a frontend that runs a
//! Python chat processor this happens on every streamed chunk, so a single long GIL hold
//! (a C extension that never releases it, a long regex, a slow tokenizer call) parks every
//! worker that is streaming. Once streams in flight >= worker threads, the HTTP server,
//! `/health` and `/live` stop answering too, although they never touch Python.
//!
//! The helpers here move that GIL work to the blocking pool: a long GIL hold then delays
//! the Python work it contends with, as it must, but the runtime workers stay free.

use futures::FutureExt;
use pyo3::BoundObject;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use tokio::sync::oneshot;

/// Run `f` under the GIL on the Tokio blocking pool and await it.
///
/// For per-item Python work on an async path (for example converting one streamed
/// response). Ordering is preserved because the caller awaits each call.
pub(crate) async fn with_gil_offloaded<F, R>(f: F) -> Result<R, tokio::task::JoinError>
where
    F: for<'py> FnOnce(Python<'py>) -> R + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(move || Python::with_gil(f)).await
}

/// Done callback on the asyncio future: cancels the Rust future when Python cancels.
#[pyclass]
struct CancelOnDone {
    cancel_tx: Option<oneshot::Sender<()>>,
}

#[pymethods]
impl CancelOnDone {
    fn __call__(&mut self, fut: &Bound<PyAny>) -> PyResult<()> {
        if fut.call_method0("cancelled")?.is_truthy()?
            && let Some(tx) = self.cancel_tx.take()
        {
            let _ = tx.send(());
        }
        Ok(())
    }
}

/// Scheduled with `call_soon_threadsafe`: completes the future unless it was cancelled
/// in the meantime (same contract as pyo3-async-runtimes' completor).
#[pyclass]
struct CompleteUnlessCancelled;

#[pymethods]
impl CompleteUnlessCancelled {
    fn __call__(
        &self,
        fut: &Bound<PyAny>,
        complete: &Bound<PyAny>,
        value: &Bound<PyAny>,
    ) -> PyResult<()> {
        if fut.call_method0("cancelled")?.is_truthy()? {
            return Ok(());
        }
        complete.call1((value,))?;
        Ok(())
    }
}

/// Convert a Rust future into an asyncio awaitable without taking the GIL on a Tokio
/// worker.
///
/// Same contract as `pyo3_async_runtimes::tokio::future_into_py_with_locals`: the future
/// runs on the bridge runtime, cancelling the asyncio future drops it, a panic surfaces as
/// `RustPanic`, and the result is delivered on `locals`' event loop. The difference is
/// where the GIL is taken: pyo3-async-runtimes takes it on the runtime worker that drove the
/// future (three times: two `clone_ref`s and the result delivery); here the locals are
/// cloned while the caller already holds the GIL, and the result is converted and
/// delivered from the blocking pool.
///
/// The future is not wrapped in pyo3-async-runtimes' task-local scope (that scope is
/// private to the crate). Dynamo code passes `TaskLocals` explicitly wherever a bridged
/// future calls back into Python, so nothing reads it.
pub(crate) fn future_into_py_with_locals<F, T>(
    py: Python<'_>,
    locals: TaskLocals,
    fut: F,
) -> PyResult<Bound<'_, PyAny>>
where
    F: std::future::Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    let event_loop = locals.event_loop(py);
    let py_fut = event_loop.call_method0("create_future")?;
    let (cancel_tx, mut cancel_rx) = oneshot::channel::<()>();
    py_fut.call_method1(
        "add_done_callback",
        (CancelOnDone {
            cancel_tx: Some(cancel_tx),
        },),
    )?;

    let py_fut_handle = py_fut.clone().unbind();
    let event_loop = event_loop.unbind();

    crate::bridge_runtime().spawn(async move {
        let fut = std::panic::AssertUnwindSafe(fut).catch_unwind();
        tokio::pin!(fut);
        let mut cancel_armed = true;
        let outcome = loop {
            tokio::select! {
                biased;
                outcome = &mut fut => break Some(outcome),
                // Ok = Python cancelled the future: drop the Rust future. Err = the done
                // callback was dropped without cancelling; keep waiting for the result.
                cancelled = &mut cancel_rx, if cancel_armed => {
                    if cancelled.is_ok() {
                        break None;
                    }
                    cancel_armed = false;
                }
            }
        };
        let Some(outcome) = outcome else {
            return;
        };

        // Await the blocking step so this task stays alive until the result is delivered:
        // `wait_for_bridge_tasks_at_exit` drains by counting live runtime tasks.
        let _ = with_gil_offloaded(move |py| {
            let fut = py_fut_handle.bind(py);
            if fut
                .call_method0("cancelled")
                .and_then(|c| c.is_truthy())
                .unwrap_or(false)
            {
                return;
            }
            let result: PyResult<PyObject> = match outcome {
                Ok(result) => result.and_then(|value| {
                    value
                        .into_pyobject(py)
                        .map(|obj| obj.into_any().unbind())
                        .map_err(Into::into)
                }),
                Err(panic) => Err(pyo3_async_runtimes::err::RustPanic::new_err(format!(
                    "rust future panicked: {}",
                    panic_message(panic.as_ref())
                ))),
            };
            let delivered = match result {
                Ok(value) => fut
                    .getattr("set_result")
                    .map(|complete| (complete, value.into_bound(py))),
                Err(err) => fut
                    .getattr("set_exception")
                    .map(|complete| (complete, err.into_value(py).into_bound(py).into_any())),
            }
            .and_then(|(complete, value)| {
                event_loop.bind(py).call_method1(
                    "call_soon_threadsafe",
                    (CompleteUnlessCancelled, fut, complete, value),
                )
            });
            if let Err(err) = delivered {
                // The event loop is closed or the future is gone; nothing awaits it.
                err.print_and_set_sys_last_vars(py);
            }
        })
        .await;
    });

    Ok(py_fut)
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = panic.downcast_ref::<&str>() {
        message
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.as_str()
    } else {
        "unknown error"
    }
}
