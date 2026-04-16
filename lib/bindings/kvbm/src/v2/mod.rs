// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framework integration modules for Python bindings.

pub mod connector;
pub mod runtime;
// pub mod scheduler;
//
// DEFERRED — the on-disk `scheduler/{mod,config,status}.rs` files reference
// `dynamo_kvbm::v2::integrations::scheduler::{Scheduler, KVCacheManager}`,
// `dynamo_kvbm::v2::logical::{BlockRegistry, manager::BlockManager}`,
// `dynamo_kvbm::v2::utils::tinylfu::TinyLFUTracker`, and `dynamo_kvbm::G1`
// — all from the pre-decomposition `dynamo_kvbm` crate. Re-enabling the
// module requires porting those types into the new decomposed kvbm-* crates
// (likely `kvbm-engine` + `kvbm-logical`).
//
// The Python layer (`kvbm/v2/vllm/schedulers/dynamo.py:39-55`) already has
// a `_RUST_SCHEDULER_AVAILABLE = False` fallback that degrades to a vLLM
// passthrough — KV transfer offload still routes through the v2
// ConnectorLeader/Worker, which IS exported below. Phase 5 of the
// ACTIVE_PLAN does not need the Rust scheduler; restoring it is shadow-mode
// divergence work and can wait for an explicit driver.
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
    m.add_class::<vllm::PyRustKvCacheManager>()?;
    m.add_class::<vllm::PyG1BlockManagerHandle>()?;
    m.add_class::<torch::PyTensor>()?;
    m.add_class::<runtime::PyKvbmRuntime>()?;

    // Connector classes
    m.add_class::<connector::leader::PyConnectorLeader>()?;
    m.add_class::<connector::worker::PyConnectorWorker>()?;

    m.add_class::<connector::leader::PyRequest>()?;
    m.add_class::<connector::leader::PySchedulerOutput>()?;

    // // vLLM specific classes
    // // Leader connector classes for v2 vLLM integration
    // m.add_class::<vllm::PyKvbmRequest>()?;
    // m.add_class::<vllm::PyConnectorMetadataBuilder>()?;
    // m.add_class::<vllm::PyRustSchedulerOutput>()?;
    Ok(())
}
