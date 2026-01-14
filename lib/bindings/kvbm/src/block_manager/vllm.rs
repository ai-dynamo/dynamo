// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, wrap_pymodule};

mod connector;
mod request;

pub use request::KvbmRequest;

#[pymodule]
fn _vllm_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvbmRequest>()?;

    m.add_class::<connector::worker::PyKvConnectorWorker>()?;
    m.add_class::<connector::leader::PyKvConnectorLeader>()?;
    m.add_class::<connector::SchedulerOutput>()?;
    // TODO: use TRTLLM own integration module
    m.add_class::<connector::trtllm_worker::PyTrtllmKvConnectorWorker>()?;
    m.add_class::<connector::trtllm_leader::PyTrtllmKvConnectorLeader>()?;
    Ok(())
}

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(_vllm_integration))?;
    Ok(())
}
