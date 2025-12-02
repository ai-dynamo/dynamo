// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the v2 connector worker.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use dynamo_kvbm::v2::integrations::connector::worker::{ConnectorWorker, ConnectorWorkerInterface};
use dynamo_memory::TensorDescriptor;

use crate::to_pyerr;
use crate::v2::torch::Tensor;

/// Python wrapper for the v2 ConnectorWorker.
///
/// This class wraps the Rust ConnectorWorker, which handles:
/// - KV cache registration with NIXL for RDMA transfers
/// - Handshake metadata export
/// - Graceful shutdown
#[pyclass(name = "ConnectorWorker")]
pub struct PyConnectorWorker {
    inner: ConnectorWorker,
}

#[pymethods]
impl PyConnectorWorker {
    /// Create a new ConnectorWorker from a KvbmRuntime.
    ///
    /// Args:
    ///     runtime: The KvbmRuntime instance (provides Nova for communication)
    ///
    /// Returns:
    ///     ConnectorWorker: The worker instance ready for KV cache registration.
    #[new]
    pub fn new(runtime: &crate::v2::runtime::PyKvbmRuntime) -> PyResult<Self> {
        let nova = runtime.nova().map_err(to_pyerr)?;
        let inner = ConnectorWorker::new(nova);
        Ok(Self { inner })
    }

    /// Register KV cache tensors with NIXL for RDMA transfers.
    ///
    /// This is the critical step that enables remote GPU-to-GPU transfers.
    /// Each tensor is registered with NIXL via the UCX backend.
    ///
    /// Args:
    ///     tensors: List of PyTorch CUDA tensors representing KV cache layers.
    ///              Must be in layer order and all on the same CUDA device.
    ///     num_device_blocks: Number of device blocks (from vLLM's cache config)
    ///     page_size: Block/page size for the KV cache
    ///     dtype_width_bytes: Data type width in bytes (e.g., 2 for fp16)
    ///
    /// Raises:
    ///     RuntimeError: If registration fails (e.g., UCX backend not available,
    ///                   tensors on different devices, or already registered).
    #[pyo3(signature = (tensors, num_device_blocks, page_size, dtype_width_bytes))]
    pub fn register_kv_caches(
        &self,
        tensors: Vec<Py<PyAny>>,
        num_device_blocks: usize,
        page_size: usize,
        dtype_width_bytes: usize,
    ) -> PyResult<()> {
        // Convert Python tensors to Rust TensorDescriptor
        let rust_tensors: Vec<Arc<dyn TensorDescriptor>> = tensors
            .into_iter()
            .map(|py_tensor| {
                let tensor = Tensor::new(py_tensor).map_err(to_pyerr)?;
                Ok(Arc::new(tensor) as Arc<dyn TensorDescriptor>)
            })
            .collect::<PyResult<Vec<_>>>()?;

        self.inner
            .register_kv_caches(
                rust_tensors,
                num_device_blocks,
                page_size,
                dtype_width_bytes,
            )
            .map_err(to_pyerr)
    }

    /// Get handshake metadata to send to the leader.
    ///
    /// Returns metadata bytes that can be all-gathered to the leader
    /// for peer discovery and RDMA setup.
    ///
    /// Returns:
    ///     bytes: Serialized metadata (includes NIXL registration info if available)
    pub fn handshake_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let metadata = self.inner.handshake_metadata().map_err(to_pyerr)?;
        Ok(PyBytes::new(py, &metadata))
    }

    /// Check if initialization has been completed.
    ///
    /// Returns:
    ///     bool: True if NIXL registration is complete, False if still pending.
    pub fn is_initialized(&self) -> bool {
        self.inner.is_initialized()
    }

    /// Gracefully shutdown the connector worker.
    ///
    /// This ensures proper cleanup of NIXL registrations.
    pub fn shutdown(&self) -> PyResult<()> {
        self.inner.shutdown().map_err(to_pyerr)
    }

    /// Get completed transfer request IDs (drains the sets).
    ///
    /// Called by the worker executor (vLLM) to check which requests have
    /// completed onboarding or offloading. The leader populates this state
    /// via Nova messages after detecting all workers have completed transfers.
    ///
    /// Returns:
    ///     tuple: (Optional[set[str]], Optional[set[str]]) for (offload_ids, onboard_ids)
    ///            Returns None for each set if there are no completed requests of that type.
    #[pyo3(name = "get_finished")]
    pub fn py_get_finished(
        &self,
    ) -> PyResult<(
        Option<std::collections::HashSet<String>>,
        Option<std::collections::HashSet<String>>,
    )> {
        let (offload_ids, onboard_ids) = self.inner.get_finished();

        let offload = if offload_ids.is_empty() {
            None
        } else {
            Some(offload_ids)
        };
        let onboard = if onboard_ids.is_empty() {
            None
        } else {
            Some(onboard_ids)
        };

        Ok((offload, onboard))
    }
}
