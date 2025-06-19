// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use llm_rs::block_manager::distributed::{KvbmWorker as KvbmWorkerImpl, KvbmWorkerConfig};
use llm_rs::block_manager::storage::torch::{TorchDevice, TorchTensor};

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
pub struct KvbmWorker {
    _impl: Arc<KvbmWorkerImpl>,
}

#[pymethods]
impl KvbmWorker {
    #[new]
    #[pyo3(signature = (num_device_blocks, page_size, tensors, device_id=0, worker_id=0, dtype=None, barrier_id="kvbm".to_string()))]
    fn new(
        num_device_blocks: usize,
        page_size: usize,
        tensors: Vec<Py<PyAny>>,
        device_id: usize,
        worker_id: usize,
        dtype: Option<String>,
        barrier_id: String,
    ) -> PyResult<Self> {
        let dtype = map_dtype(&dtype.unwrap_or("fp16".to_string())).map_err(to_pyerr)?;

        let mut vllm_tensors: Vec<Box<dyn TorchTensor>> = Vec::with_capacity(tensors.len());

        for tensor in tensors {
            let vllm_tensor = VllmTensor::new(tensor.clone()).map_err(to_pyerr)?;
            vllm_tensors.push(Box::new(vllm_tensor));
        }

        let config = KvbmWorkerConfig::builder()
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(vllm_tensors)
            .device_id(device_id)
            .worker_id(worker_id)
            .dtype(dtype)
            .barrier_id(barrier_id)
            .build()
            .map_err(to_pyerr)?;

        let worker = KvbmWorkerImpl::new(config).map_err(to_pyerr)?;

        Ok(Self {
            _impl: Arc::new(worker),
        })
    }
}
