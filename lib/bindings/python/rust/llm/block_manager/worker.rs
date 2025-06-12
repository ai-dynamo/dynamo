// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use llm_rs::block_manager::{
    block::{layout_to_blocks, Block},
    layout::LayoutType,
    storage::{
        torch::{TorchDevice, TorchTensor},
        DeviceAllocator, DeviceStorage,
    },
    BasicMetadata, LayoutConfig, NixlLayout,
};

use nixl_sys::Agent as NixlAgent;

#[derive(Clone, Debug)]
struct VllmTensor {
    _py_tensor: Py<PyAny>,
    device: TorchDevice,
    data_ptr: u64,
    size_bytes: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl VllmTensor {
    fn new(py_tensor: Py<PyAny>) -> anyhow::Result<Self> {
        Python::with_gil(|py| {
            let device = py_tensor.getattr(py, "device")?;
            let device_type = device.getattr(py, "type")?.extract::<String>(py)?;

            let device = if device_type == "cuda" {
                TorchDevice::Cuda(device.getattr(py, "index")?.extract::<usize>(py)?)
            } else {
                TorchDevice::Other(device_type)
            };

            let data_ptr = py_tensor.call_method0(py, "data_ptr")?.extract::<u64>(py)?;
            let size_bytes = py_tensor
                .getattr(py, "nbytes")?
                .extract::<usize>(py)?;
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

fn load_and_validate_tensors(
    tensors: Vec<Py<PyAny>>,
    device_id: usize,
) -> anyhow::Result<(Vec<DeviceStorage>, Vec<usize>)> {
    let mut device_tensors = Vec::with_capacity(tensors.len());

    let allocator = DeviceAllocator::new(device_id)?;

    let mut shape = None;
    for tensor in tensors {
        let vllm_tensor = VllmTensor::new(tensor)?;

        // Check the stride, and ensure our tensor is contiguous.
        // TODO: We eventually need to be able to handle this.
        let stride = vllm_tensor.stride();
        for i in 1..stride.len() {
            if stride[i] >= stride[i - 1] {
                return Err(anyhow::anyhow!(
                    "Tensor strides must be monotonically decreasing"
                ));
            }
        }

        // Check that all layer tensors have the same shape.
        // TODO: We eventually need to support the weirder models with heterogenous layers.
        if let Some(shape) = shape.as_ref() {
            if *shape != vllm_tensor.shape() {
                return Err(anyhow::anyhow!("All tensors must have the same shape"));
            }
        } else {
            shape = Some(vllm_tensor.shape());
        }

        let device_tensor = DeviceStorage::new_from_torch(allocator.ctx(), Box::new(vllm_tensor))?;

        device_tensors.push(device_tensor);
    }

    Ok((device_tensors, shape.unwrap()))
}

#[pyclass]
pub struct KvbmWorker {
    _device_blocks: Vec<Block<DeviceStorage, BasicMetadata>>,
}

#[pymethods]
impl KvbmWorker {
    #[new]
    #[pyo3(signature = (num_layers, num_blocks, outer_dim, page_size, inner_dim, tensors, device_id=0, worker_id=0, dtype=None))]
    fn new(
        num_layers: usize,
        num_blocks: usize,
        outer_dim: usize,
        page_size: usize,
        inner_dim: usize,
        tensors: Vec<Py<PyAny>>,
        device_id: usize,
        worker_id: usize,
        dtype: Option<String>,
    ) -> PyResult<Self> {
        if num_layers == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_layers must be greater than 0",
            ));
        }

        if num_layers != tensors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_layers must match the number of tensors",
            ));
        }

        let dtype = dtype.unwrap_or("fp16".to_string());

        tracing::info!("Initializing KvbmWorker with params: num_layers={}, num_blocks={}, outer_dim={}, page_size={}, inner_dim={}, dtype={}", num_layers, num_blocks, outer_dim, page_size, inner_dim, dtype);

        let (device_tensors, shape) =
            load_and_validate_tensors(tensors, device_id).map_err(to_pyerr)?;

        let outer_contiguous = shape[0] == outer_dim;

        let agent = NixlAgent::new(&format!("kvbm-worker-{}", worker_id)).map_err(to_pyerr)?;

        let layout_builder = LayoutConfig::builder()
            .num_layers(num_layers)
            .num_blocks(num_blocks)
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype(map_dtype(&dtype).map_err(to_pyerr)?)
            .build()
            .map_err(to_pyerr)?;

        let mut layout = layout_builder
            .create_layout(
                LayoutType::LayerSeparate { outer_contiguous },
                device_tensors,
                true,
            )
            .map_err(to_pyerr)?;

        layout.nixl_register(&agent, None).map_err(to_pyerr)?;

        let layout: Arc<dyn NixlLayout<StorageType = DeviceStorage>> = Arc::from(layout);

        let device_blocks =
            layout_to_blocks::<_, BasicMetadata>(layout, 0, worker_id as u64).map_err(to_pyerr)?;

        Ok(Self {
            _device_blocks: device_blocks,
        })
    }
}
