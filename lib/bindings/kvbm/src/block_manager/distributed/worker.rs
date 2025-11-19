// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use std::sync::Arc;
use utils::{get_leader_zmq_ack_url, get_leader_zmq_pub_url};

use llm_rs::block_manager::distributed::{
    BlockTransferHandler as RustBlockTransferHandler, KvbmWorker as KvbmWorkerImpl,
    KvbmWorkerConfig,
};
use llm_rs::block_manager::layout::LayoutType;
use llm_rs::block_manager::storage::torch::{TorchDevice, TorchTensor};

/// A wrapper around a layout type.
/// This is used to convert between the Python and Rust layout types.
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Eq)]
pub enum PyLayoutType {
    FullyContiguous,
    LayerSeparate,
}

#[pymethods]
impl PyLayoutType {
    /// String representation of the layout type
    fn __str__(&self) -> &'static str {
        match self {
            PyLayoutType::FullyContiguous => "FullyContiguous",
            PyLayoutType::LayerSeparate => "LayerSeparate",
        }
    }

    /// Representation for debugging
    fn __repr__(&self) -> String {
        format!("PyLayoutType.{}", self.__str__())
    }
}

impl From<PyLayoutType> for LayoutType {
    fn from(py_layout: PyLayoutType) -> Self {
        match py_layout {
            PyLayoutType::FullyContiguous => LayoutType::FullyContiguous,
            // Layout (outer_contiguous vs block_contiguous) is auto-detected from tensor shapes
            PyLayoutType::LayerSeparate => LayoutType::layer_separate_auto_default(),
        }
    }
}

/// A wrapper around a Torch tensor.
/// We hold onto the py object to ensure it doesn't get GCed.
#[derive(Clone, Debug)]
pub struct VllmTensor {
    _py_tensor: Py<PyAny>,
    device: TorchDevice,
    data_ptr: u64,
    size_bytes: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl VllmTensor {
    pub fn new(py_tensor: Py<PyAny>) -> anyhow::Result<Self> {
        Python::with_gil(|py| {
            let device = py_tensor.getattr(py, "device")?;
            let device_type = device.getattr(py, "type")?.extract::<String>(py)?;

            let device = if device_type == "cuda" {
                TorchDevice::Cuda(device.getattr(py, "index")?.extract::<usize>(py)?)
            } else {
                TorchDevice::Other(device_type)
            };

            let data_ptr = py_tensor.call_method0(py, "data_ptr")?.extract::<u64>(py)?;
            let size_bytes = py_tensor.getattr(py, "nbytes")?.extract::<usize>(py)?;
            let shape = py_tensor.getattr(py, "shape")?.extract::<Vec<usize>>(py)?;
            let stride = py_tensor
                .call_method0(py, "stride")?
                .extract::<Vec<usize>>(py)?;

            tracing::trace!("VllmTensor: {data_ptr}, {size_bytes}, {shape:?}, {stride:?}");

            Ok(Self {
                _py_tensor: py_tensor,
                device,
                data_ptr,
                size_bytes,
                shape,
                stride,
            })
        })
    }
}

impl TorchTensor for VllmTensor {
    fn device(&self) -> TorchDevice {
        self.device.clone()
    }

    fn data_ptr(&self) -> u64 {
        self.data_ptr
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn stride(&self) -> Vec<usize> {
        self.stride.clone()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BlockTransferHandler {
    _impl: Arc<dyn RustBlockTransferHandler>,
}

impl BlockTransferHandler {
    pub fn get_handler(&self) -> Arc<dyn RustBlockTransferHandler> {
        self._impl.clone()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct KvbmWorker {
    inner: Arc<Mutex<KvbmWorkerImpl>>,
    _drt: Option<Arc<rs::DistributedRuntime>>,
}

impl KvbmWorker {
    pub fn get_inner(&self) -> Arc<Mutex<KvbmWorkerImpl>> {
        self.inner.clone()
    }
}

#[pymethods]
impl KvbmWorker {
    #[new]
    #[pyo3(signature = (num_device_blocks, page_size, tensors, device_id=0, dtype_width_bytes=2, drt=None, layout_blocking=false, device_layout_type=None, host_layout_type=None, disk_layout_type=None))]
    fn new(
        num_device_blocks: usize,
        page_size: usize,
        tensors: Vec<Py<PyAny>>,
        device_id: usize,
        dtype_width_bytes: usize,
        drt: Option<PyObject>,
        layout_blocking: bool,
        device_layout_type: Option<PyLayoutType>,
        host_layout_type: Option<PyLayoutType>,
        disk_layout_type: Option<PyLayoutType>,
    ) -> PyResult<Self> {
        let drt: Option<Arc<rs::DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = drt {
                extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let rt = get_current_tokio_handle();

        let mut vllm_tensors: Vec<Arc<dyn TorchTensor>> = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            let vllm_tensor = VllmTensor::new(tensor.clone()).map_err(to_pyerr)?;
            vllm_tensors.push(Arc::new(vllm_tensor));
        }

        let config = KvbmWorkerConfig::builder()
            .cancel_token(get_current_cancel_token())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(vllm_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .device_layout_type(
                device_layout_type
                    .map(|py_layout| py_layout.into())
                    .unwrap_or(LayoutType::FullyContiguous),
            )
            .host_layout_type(
                host_layout_type
                    .map(|py_layout| py_layout.into())
                    .unwrap_or(LayoutType::FullyContiguous),
            )
            .disk_layout_type(
                disk_layout_type
                    .map(|py_layout| py_layout.into())
                    .unwrap_or(LayoutType::FullyContiguous),
            )
            .leader_pub_url(get_leader_zmq_pub_url())
            .leader_ack_url(get_leader_zmq_ack_url())
            .build()
            .map_err(to_pyerr)?;

        let worker = rt
            .block_on(async move {
                let kvbm_worker = KvbmWorkerImpl::new(config, layout_blocking).await?;
                anyhow::Ok(kvbm_worker)
            })
            .map_err(to_pyerr)?;

        Ok(Self {
            inner: Arc::new(Mutex::new(worker)),
            _drt: drt,
        })
    }
}
