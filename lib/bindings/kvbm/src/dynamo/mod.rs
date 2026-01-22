// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Dynamo Runtime.

use std::sync::{Arc, OnceLock, Weak};

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

use dynamo_runtime::{self as rs, CancellationToken, RuntimeConfig, logging};

pub fn add_to_module(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tokio runtime first to avoid panics when OTEL_EXPORT_ENABLED=1
    init_pyo3_tokio_rt();

    if std::env::var("OTEL_EXPORT_ENABLED")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        // OTLP batch exporter needs runtime context to spawn background tasks
        let handle = get_current_tokio_handle();
        let _guard = handle.enter();
        logging::init();
    } else {
        // OTEL disabled: no runtime context needed
        logging::init();
    }
    Ok(())
}

static PYO3_TOKIO_INIT: OnceLock<()> = OnceLock::new();
static PYO3_TOKIO_RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
static PYO3_TOKIO_CANCEL_TOKEN: OnceLock<CancellationToken> = OnceLock::new();

// The runtime's threads do not survive when passing DistributedRuntime across bindings,
// so we need to reinitialize the runtime thread pool.
// This is also required in environments without a DistributedRuntime.
fn init_pyo3_tokio_rt() {
    PYO3_TOKIO_INIT.get_or_init(|| {
        let cfg =
            RuntimeConfig::from_settings().expect("failed to build runtime config from settings");

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(
                cfg.num_worker_threads
                    .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get()),
            )
            .max_blocking_threads(cfg.max_blocking_threads)
            .enable_all()
            .build()
            .expect("failed to build fallback tokio runtime for pyo3_async_runtimes");

        let _ = PYO3_TOKIO_RT.set(rt);
        let rt_ref = PYO3_TOKIO_RT.get().expect("runtime missing after set");

        // Initialize the shared cancellation token
        let cancel_token = CancellationToken::new();
        let _ = PYO3_TOKIO_CANCEL_TOKEN.set(cancel_token);

        // Initialize pyo3-async runtimes with this runtime
        let _ = pyo3_async_runtimes::tokio::init_with_runtime(rt_ref);
    });
}

pub fn get_current_tokio_handle() -> tokio::runtime::Handle {
    PYO3_TOKIO_RT
        .get()
        .expect("Tokio runtime not initialized!")
        .handle()
        .clone()
}

pub fn get_current_cancel_token() -> CancellationToken {
    PYO3_TOKIO_CANCEL_TOKEN
        .get()
        .expect("Cancellation token not initialized!")
        .clone()
}

#[pyclass]
#[derive(Clone)]
pub struct Component {
    pub inner: rs::component::Component,
}

pub fn extract_distributed_runtime_from_obj(
    py: Python<'_>,
    drt_obj: PyObject,
) -> PyResult<Option<Arc<rs::DistributedRuntime>>> {
    if drt_obj.is_none(py) {
        return Ok(None);
    }

    let obj = drt_obj.bind(py);

    let cls = py.import("dynamo._core")?.getattr("DistributedRuntime")?;
    if !obj.is_instance(&cls)? {
        return Err(PyTypeError::new_err(
            "expected dynamo._core.DistributedRuntime",
        ));
    }

    let cap_any = obj.call_method0("to_capsule")?;
    let cap: &Bound<'_, PyCapsule> = cap_any.downcast()?;
    let weak: &Weak<rs::DistributedRuntime> = unsafe { cap.reference::<Weak<_>>() };

    let strong = weak.upgrade().ok_or_else(|| {
        PyRuntimeError::new_err("runtime is no longer alive (weak ref upgrade failed)")
    })?;

    Ok(Some(strong))
}
