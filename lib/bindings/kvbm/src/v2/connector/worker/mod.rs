// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the v2 connector worker.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use dynamo_kvbm::v2::integrations::connector::leader::scheduler::KvConnectorMetadata;
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
        let runtime = runtime.inner();
        let inner = ConnectorWorker::new(runtime);
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

    /// Bind connector metadata from the leader.
    ///
    /// Args:
    ///     data: The connector metadata bytes
    pub fn bind_connector_metadata(&self, data: Vec<u8>) -> PyResult<bool> {
        let metadata: KvConnectorMetadata = serde_json::from_slice(&data).map_err(to_pyerr)?;

        // todo: add a method on KvConnectorMetadata to check if it should be bound
        // todo: change the return type to bool
        // todo: binding if should_bind is true and return true; otherwise return false
        if metadata.should_bind() {
            self.inner
                .bind_connector_metadata(metadata)
                .map_err(to_pyerr)?;
            return Ok(true);
        }
        return Ok(false);
    }

    /// Clear connector metadata.
    ///
    /// This function should be called by the model runner every time
    /// after the model execution.
    pub fn clear_connector_metadata(&self) -> PyResult<()> {
        self.inner.clear_connector_metadata().map_err(to_pyerr)
    }

    /// Save KV layer and trigger forward pass completion on last layer.
    ///
    /// Always callable - returns immediately if no action is needed for this layer.
    /// On the last layer with a pending forward pass event, records a CUDA event
    /// on the provided stream and spawns an async task that waits for the event
    /// before triggering the Nova forward pass event.
    ///
    /// Args:
    ///     layer_index: The layer index being saved
    ///     stream_handle: Raw CUDA stream handle (u64) from Python's current stream
    ///                   Obtained via: torch.cuda.current_stream().cuda_stream
    pub fn save_kv_layer(&self, layer_index: usize, stream_handle: u64) -> PyResult<()> {
        self.inner
            .save_kv_layer(layer_index, stream_handle)
            .map_err(to_pyerr)
    }

    /// Start loading KV cache.
    ///
    /// If the bound metadata dictates that we should start loading KV cache,
    /// this function will trigger the loading of the KV cache.
    pub fn start_load_kv(&self) -> PyResult<()> {
        self.inner.start_load_kv().map_err(to_pyerr)
    }

    /// Wait for a specific layer's KV cache load to complete.
    ///
    /// If intra-pass onboarding was triggered in start_load_kv, this method
    /// inserts a cudaStreamWaitEvent on the provided torch stream to synchronize
    /// with the layer's onboard completion.
    ///
    /// Args:
    ///     layer_index: The layer index to wait for
    ///     stream_handle: Raw CUDA stream handle (u64) from Python's current torch stream
    pub fn wait_for_layer_load(&self, layer_index: usize, stream_handle: u64) -> PyResult<()> {
        self.inner
            .wait_for_layer_load(layer_index, stream_handle)
            .map_err(to_pyerr)
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
    #[allow(clippy::type_complexity)]
    pub fn get_finished(&self) -> PyResult<(Option<HashSet<String>>, Option<HashSet<String>>)> {
        let (offload_ids, onboard_ids) = self.inner.get_finished().dissolve();

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

    pub fn get_failed_onboarding(&self) -> PyResult<HashSet<usize>> {
        Ok(self.inner.get_failed_onboarding())
    }
}
