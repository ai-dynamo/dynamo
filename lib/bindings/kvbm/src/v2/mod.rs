// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framework integration modules for Python bindings.

pub mod connector;
pub mod runtime;
pub mod scheduler;
pub mod torch;
pub mod vllm;

use pyo3::prelude::*;

/// Check if the v2 feature is available.
///
/// This function always returns `true` when the v2 module is compiled.
#[pyfunction]
fn is_available() -> bool {
    true
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_class::<vllm::PyKvbmVllmConfig>()?;
    m.add_class::<torch::PyTensor>()?;
    m.add_class::<runtime::PyKvbmRuntime>()?;

    // Connector classes
    m.add_class::<connector::leader::PyConnectorLeader>()?;
    m.add_class::<connector::worker::PyConnectorWorker>()?;

    m.add_class::<connector::leader::PyRequest>()?;
    m.add_class::<connector::leader::PySchedulerOutput>()?;

    // Scheduler classes
    m.add_class::<scheduler::PyScheduler>()?;
    m.add_class::<scheduler::PySchedulerConfig>()?;
    m.add_class::<scheduler::PyRequestStatus>()?;

    // // vLLM specific classes
    // // Leader connector classes for v2 vLLM integration
    // m.add_class::<vllm::PyKvbmRequest>()?;
    // m.add_class::<vllm::PyConnectorMetadataBuilder>()?;
    // m.add_class::<vllm::PyRustSchedulerOutput>()?;
    Ok(())
}
