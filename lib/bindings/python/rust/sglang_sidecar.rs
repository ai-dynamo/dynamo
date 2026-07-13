// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 entry point for the SGLang native gRPC sidecar.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
pub fn run_sglang_sidecar(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("python -m dynamo.sglang_sidecar".to_string());
    argv.extend(args);

    let (engine, config) = dynamo_sglang_sidecar::SglangSidecarEngine::from_args(Some(argv))
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    py.allow_threads(|| {
        dynamo_backend_common::run(Arc::new(engine), config)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    })
}
