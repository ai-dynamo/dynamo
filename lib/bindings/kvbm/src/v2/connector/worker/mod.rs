// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the v2 connector worker.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use kvbm_connector::TensorDescriptor;
use kvbm_connector::connector::leader::scheduler::KvConnectorMetadata;
use kvbm_connector::connector::worker::{ConnectorWorker, ConnectorWorkerInterface};

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
    ///     runtime: The KvbmRuntime instance (provides Velo for communication)
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
    /// Args:
    ///     tensors: List of PyTorch CUDA tensors representing KV cache layers.
    ///              Must be in layer order (matching `kv_cache_config.kv_cache_tensors`)
    ///              and all on the same CUDA device.
    ///     num_device_blocks: Number of device blocks (from vLLM's per-rank cache config).
    ///     dtype_width_bytes: Data type width in bytes (e.g., 2 for fp16).
    ///     dim_labels: Per-axis labels as strings — one of `"Block"`, `"Layer"`,
    ///                 `"Outer"`, `"Page"`, `"HeadCount"`, `"HeadSize"`, `"Payload"`.
    ///                 Length must match the rank of every tensor.
    ///     dim_sizes: Per-axis sizes (matching `dim_labels` index-for-index). Each
    ///                tensor's `shape()` must equal `dim_sizes` exactly.
    ///     block_layout: Per-block dim ordering as a string — one of
    ///                   `"OperationalNHD"`, `"OperationalHND"`, `"Universal"`,
    ///                   `"Unknown"`. Derived in Python from
    ///                   `attn_backend.get_kv_cache_stride_order(False)`.
    ///
    /// Raises:
    ///     RuntimeError: If registration fails (UCX backend missing, tensor
    ///                   shape disagreement, unknown labels, etc.).
    #[pyo3(signature = (tensors, num_device_blocks, dtype_width_bytes, dim_labels, dim_sizes, block_layout))]
    pub fn register_kv_caches(
        &self,
        tensors: Vec<Py<PyAny>>,
        num_device_blocks: usize,
        dtype_width_bytes: usize,
        dim_labels: Vec<String>,
        dim_sizes: Vec<usize>,
        block_layout: String,
    ) -> PyResult<()> {
        let rust_tensors: Vec<Arc<dyn TensorDescriptor>> = tensors
            .into_iter()
            .map(|py_tensor| {
                let tensor = Tensor::new(py_tensor).map_err(to_pyerr)?;
                Ok(Arc::new(tensor) as Arc<dyn TensorDescriptor>)
            })
            .collect::<PyResult<Vec<_>>>()?;

        let dims: Vec<kvbm_common::KvDim> = dim_labels
            .iter()
            .map(|s| parse_kv_dim(s))
            .collect::<PyResult<_>>()?;
        let dim_layout = kvbm_common::KvDimLayout::new(dims, dim_sizes).map_err(to_pyerr)?;
        let block_layout = parse_kv_block_layout(&block_layout)?;

        self.inner
            .register_kv_caches(
                rust_tensors,
                num_device_blocks,
                dtype_width_bytes,
                dim_layout,
                block_layout,
            )
            .map_err(to_pyerr)
    }

    /// Register a single cross-layer KV cache tensor with NIXL.
    ///
    /// Used when vLLM allocates a uniform cross-layer KV cache (a single
    /// allocation whose physical byte layout is
    /// `[num_blocks, num_layers, outer_dim, page_size, num_kv_heads, head_size]`).
    /// The caller must validate the layout against the attention backend's
    /// stride order before invoking this; the Rust side trusts the dims.
    ///
    /// Args:
    ///     tensor: A single PyTorch CUDA tensor covering all layers.
    ///     num_device_blocks: Number of device blocks from vLLM's cache config.
    ///     num_layers: Number of attention layers fused into the tensor.
    ///     outer_dim: K/V split dimension (typically 2 for standard attention).
    ///     page_size: Block/page size (tokens per block).
    ///     num_kv_heads: Per-rank KV head count. Must evenly divide `inner_dim`;
    ///         used by the planner to split the inner axis into
    ///         `(HeadCount, HeadSize)` for stride-aware projection.
    ///     inner_dim: Flattened head dimension (`num_kv_heads * head_size`).
    ///     dtype_width_bytes: Data type width in bytes (e.g. 2 for fp16).
    #[pyo3(signature = (tensor, num_device_blocks, num_layers, outer_dim, page_size, num_kv_heads, inner_dim, dtype_width_bytes))]
    #[allow(clippy::too_many_arguments)]
    pub fn register_cross_layers_kv_cache(
        &self,
        tensor: Py<PyAny>,
        num_device_blocks: usize,
        num_layers: usize,
        outer_dim: usize,
        page_size: usize,
        num_kv_heads: usize,
        inner_dim: usize,
        dtype_width_bytes: usize,
    ) -> PyResult<()> {
        let rust_tensor = Tensor::new(tensor).map_err(to_pyerr)?;
        let rust_tensor: Arc<dyn TensorDescriptor> = Arc::new(rust_tensor);

        self.inner
            .register_cross_layers_kv_cache(
                rust_tensor,
                num_device_blocks,
                num_layers,
                outer_dim,
                page_size,
                num_kv_heads,
                inner_dim,
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
        Ok(false)
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
    /// before triggering the Velo forward pass event.
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

    /// Wait for the intra-pass offload to complete.
    ///
    /// This is a blocking call; however, we might choose to make it non-blocking
    /// in the future.
    ///
    /// To make it non-blocking, we would have to put an stream wait event on both the torch stream and intra-pass onboard stream
    /// to ensure that no cuda stream operations are allowed to modify the kv blocks being offloaded while the offload is in progress.
    ///
    /// The CUDA coordination would require that we correctly synchronize any stream, so the intergration with the LLM framework
    /// needs to be carefully aligned.
    ///
    /// Args:
    ///     stream_handle: Raw CUDA stream handle (u64) from Python's current stream
    ///                    Obtained via: torch.cuda.current_stream().cuda_stream
    pub fn wait_for_save(&self) -> PyResult<()> {
        self.inner.wait_for_save().map_err(to_pyerr)
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
    /// via Velo messages after detecting all workers have completed transfers.
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

/// Parse a `KvDim` axis label string. The accepted values mirror the
/// `kvbm_common::KvDim` variants exactly.
fn parse_kv_dim(s: &str) -> PyResult<kvbm_common::KvDim> {
    use kvbm_common::KvDim;
    match s {
        "Block" => Ok(KvDim::Block),
        "Layer" => Ok(KvDim::Layer),
        "Outer" => Ok(KvDim::Outer),
        "Page" => Ok(KvDim::Page),
        "HeadCount" => Ok(KvDim::HeadCount),
        "HeadSize" => Ok(KvDim::HeadSize),
        "Payload" => Ok(KvDim::Payload),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown KvDim label '{other}'; expected one of \
             Block, Layer, Outer, Page, HeadCount, HeadSize, Payload"
        ))),
    }
}

/// Parse a `KvBlockLayout` enum string. Custom block layouts are not
/// reachable from Python today; if needed, add a serde-via-JSON path.
fn parse_kv_block_layout(s: &str) -> PyResult<kvbm_common::KvBlockLayout> {
    use kvbm_common::KvBlockLayout;
    match s {
        "Universal" => Ok(KvBlockLayout::Universal),
        "OperationalHND" => Ok(KvBlockLayout::OperationalHND),
        "OperationalNHD" => Ok(KvBlockLayout::OperationalNHD),
        "Unknown" => Ok(KvBlockLayout::Unknown),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown KvBlockLayout '{other}'; expected one of \
             Universal, OperationalHND, OperationalNHD, Unknown"
        ))),
    }
}
