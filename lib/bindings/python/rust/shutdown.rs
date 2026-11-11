// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::mpsc::{self, Sender};
use std::time::{Duration, Instant};

use pyo3::prelude::*;

/// Last-resort exit independent of both the Python GIL and the async runtime.
#[pyclass]
pub struct ShutdownWatchdog {
    cancel: Option<Sender<()>>,
}

#[pymethods]
impl ShutdownWatchdog {
    #[new]
    fn new(timeout_secs: f64) -> PyResult<Self> {
        if !timeout_secs.is_finite() || !(0.0..=630_720_005.0).contains(&timeout_secs) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid shutdown deadline",
            ));
        }
        let (cancel, receiver) = mpsc::channel();
        let origin = Instant::now();
        let timeout = Duration::from_secs_f64(timeout_secs);
        std::thread::Builder::new()
            .name("python-shutdown-watchdog".into())
            .spawn(move || {
                if receiver.recv_timeout(timeout).is_err() {
                    // Dropping the Python handle must not disarm or advance the deadline.
                    std::thread::sleep(timeout.saturating_sub(origin.elapsed()));
                    // No Python, tracing subscriber, or async executor is needed here.
                    eprintln!("graceful shutdown deadline exceeded; force-exiting 70");
                    std::process::exit(70);
                }
            })
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        Ok(Self {
            cancel: Some(cancel),
        })
    }

    /// Disarm only after engine cleanup and runtime teardown have finished.
    fn finish(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

#[pyfunction]
fn worker_shutdown_timeout_secs() -> u64 {
    dynamo_runtime::worker::graceful_shutdown_timeout_secs()
}

pub fn add_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ShutdownWatchdog>()?;
    module.add_function(wrap_pyfunction!(worker_shutdown_timeout_secs, module)?)?;
    Ok(())
}
