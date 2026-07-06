// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 entry point for the SGLang backend worker.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
pub fn run_sglang_backend(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("dynamo-sglang-rs-backend".to_string());
    argv.extend(args);

    let (engine, config) = dynamo_sglang::SglangBackend::from_argv(argv)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    py.allow_threads(|| {
        dynamo_backend_common::run(Arc::new(engine), config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}
