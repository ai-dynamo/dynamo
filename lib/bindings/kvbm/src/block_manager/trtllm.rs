// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, wrap_pymodule};

use super::vllm::{
    BlockState, BlockStates, KvbmBlockList, KvbmRequest, PyTrtllmKvConnectorLeader,
    PyTrtllmKvConnectorWorker, SchedulerOutput, SlotUpdate,
};

#[pymodule]
fn _trtllm_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvbmRequest>()?;
    m.add_class::<KvbmBlockList>()?;
    m.add_class::<BlockState>()?;
    m.add_class::<BlockStates>()?;
    m.add_class::<SlotUpdate>()?;
    m.add_class::<PyTrtllmKvConnectorWorker>()?;
    m.add_class::<PyTrtllmKvConnectorLeader>()?;
    m.add_class::<SchedulerOutput>()?;
    Ok(())
}

/// Add bindings from this crate to the provided module.
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(_trtllm_integration))?;
    Ok(())
}
