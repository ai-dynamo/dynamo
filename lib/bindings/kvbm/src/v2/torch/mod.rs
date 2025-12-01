// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyTorch tensor wrapper implementing TensorDescriptor.

use std::any::Any;

use dynamo_memory::nixl::NixlDescriptor;
use dynamo_memory::{MemoryDescriptor, StorageKind, TensorDescriptor};
use pyo3::prelude::*;

/// A wrapper around a PyTorch tensor that implements TensorDescriptor.
///
/// This struct holds a reference to the Python tensor object to prevent
/// garbage collection while the tensor is in use.
pub struct Tensor {
    /// Python reference to the tensor (keeps it alive)
    _py_tensor: Py<PyAny>,
    /// Storage kind (device type and index)
    storage_kind: StorageKind,
    /// Raw pointer to tensor data
    data_ptr: u64,
    /// Total size in bytes
    size_bytes: usize,
    /// Shape (number of elements per dimension)
    shape: Vec<usize>,
    /// Stride (elements to skip per dimension)
    stride: Vec<usize>,
    /// Bytes per element
    element_size: usize,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("storage_kind", &self.storage_kind)
            .field("data_ptr", &format_args!("0x{:x}", self.data_ptr))
            .field("size_bytes", &self.size_bytes)
            .field("shape", &self.shape)
            .field("stride", &self.stride)
            .field("element_size", &self.element_size)
            .finish()
    }
}

impl Tensor {
    /// Create a new Tensor wrapper from a PyTorch tensor.
    ///
    /// Extracts device, data pointer, size, shape, stride, and element size
    /// from the Python tensor object.
    pub fn new(py_tensor: Py<PyAny>) -> anyhow::Result<Self> {
        Python::with_gil(|py| {
            let device = py_tensor.getattr(py, "device")?;
            let device_type = device.getattr(py, "type")?.extract::<String>(py)?;

            let storage_kind = if device_type == "cuda" {
                let index = device.getattr(py, "index")?.extract::<u32>(py)?;
                StorageKind::Device(index)
            } else {
                anyhow::bail!(
                    "Only CUDA tensors are supported, got device type: {}",
                    device_type
                );
            };

            let data_ptr = py_tensor.call_method0(py, "data_ptr")?.extract::<u64>(py)?;
            let size_bytes = py_tensor.getattr(py, "nbytes")?.extract::<usize>(py)?;
            let shape = py_tensor.getattr(py, "shape")?.extract::<Vec<usize>>(py)?;
            let stride = py_tensor
                .call_method0(py, "stride")?
                .extract::<Vec<usize>>(py)?;

            // element_size() returns bytes per element
            let element_size = py_tensor
                .call_method0(py, "element_size")?
                .extract::<usize>(py)?;

            tracing::trace!(
                "Tensor: addr=0x{:x}, size={}, shape={:?}, stride={:?}, element_size={}",
                data_ptr,
                size_bytes,
                shape,
                stride,
                element_size
            );

            Ok(Self {
                _py_tensor: py_tensor,
                storage_kind,
                data_ptr,
                size_bytes,
                shape,
                stride,
                element_size,
            })
        })
    }
}

impl MemoryDescriptor for Tensor {
    fn addr(&self) -> usize {
        self.data_ptr as usize
    }

    fn size(&self) -> usize {
        self.size_bytes
    }

    fn storage_kind(&self) -> StorageKind {
        self.storage_kind
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl TensorDescriptor for Tensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn stride(&self) -> &[usize] {
        &self.stride
    }

    fn element_size(&self) -> usize {
        self.element_size
    }
}

/// Python wrapper for Tensor that exposes it to Python.
///
/// This class wraps a PyTorch tensor and validates that it is a CUDA tensor.
/// Non-CUDA tensors will raise an error.
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    /// Create a new Tensor from a PyTorch tensor.
    ///
    /// Args:
    ///     tensor: A PyTorch tensor (must be on a CUDA device)
    ///
    /// Raises:
    ///     RuntimeError: If the tensor is not on a CUDA device
    #[new]
    pub fn new(tensor: Py<PyAny>) -> PyResult<Self> {
        let inner = Tensor::new(tensor).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;
        Ok(Self { inner })
    }

    /// Get the data pointer address.
    #[getter]
    pub fn data_ptr(&self) -> u64 {
        self.inner.data_ptr
    }

    /// Get the total size in bytes.
    #[getter]
    pub fn size_bytes(&self) -> usize {
        self.inner.size_bytes
    }

    /// Get the shape of the tensor.
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    /// Get the stride of the tensor.
    #[getter]
    pub fn stride(&self) -> Vec<usize> {
        self.inner.stride.clone()
    }

    /// Get the element size in bytes.
    #[getter]
    pub fn element_size(&self) -> usize {
        self.inner.element_size
    }

    /// Get the CUDA device index.
    #[getter]
    pub fn device_index(&self) -> Option<u32> {
        self.inner.storage_kind.cuda_device_index()
    }
}
