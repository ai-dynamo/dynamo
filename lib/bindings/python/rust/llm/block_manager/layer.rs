// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "block-manager")]

use super::*;
use dynamo_llm::block_manager::block::BlockDataExt;
use dynamo_llm::block_manager::block::BlockDataProviderMut;
use pyo3::{types::PyTuple, PyObject, PyResult, Python};
use std::sync::{Arc, Mutex};

// Layer struct that represents a layer within a block
#[pyclass]
pub struct Layer {
    inner: Arc<Mutex<block::BlockType>>,
    layer_idx: usize,
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

impl Layer {
    pub fn from_rust(
        block: Arc<Mutex<block::BlockType>>,
        layer_idx: usize,
        dtype: dynamo_llm::common::dtype::DType,
        device_id: usize,
    ) -> Self {
        Self {
            inner: block,
            layer_idx,
            dtype,
            device_id,
        }
    }
}

#[pymethods]
impl Layer {
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        // Return error if any arguments are provided
        if stream.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "stream argument is not supported",
            ));
        }
        if max_version.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "max_version argument is not supported",
            ));
        }
        if dl_device.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "dl_device argument is not supported",
            ));
        }
        if copy.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "copy argument is not supported",
            ));
        }

        // Extract all necessary data for dlpack
        let ptr: *mut std::ffi::c_void;
        let num_outer_dims: i64;
        let page_size: i64;
        let inner_dim: i64;
        {
            let mut mutable_block = self.inner.lock().unwrap();
            ptr = match &mut *mutable_block {
                block::BlockType::Pinned(block) => {
                    use dynamo_llm::block_manager::block::private::PrivateToken;
                    let block_data = block.block_data_mut(PrivateToken);
                    let mut layer_view_mut =
                        block_data.layer_view_mut(self.layer_idx, 0).map_err(to_pyerr)?;
                    (unsafe { layer_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
                }
                block::BlockType::Device(block) => {
                    use dynamo_llm::block_manager::block::private::PrivateToken;
                    let block_data = block.block_data_mut(PrivateToken);
                    let mut layer_view_mut =
                        block_data.layer_view_mut(self.layer_idx, 0).map_err(to_pyerr)?;
                    (unsafe { layer_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
                }
            };
            (num_outer_dims, page_size, inner_dim) = match &*mutable_block {
                block::BlockType::Pinned(block) => (
                    block.num_outer_dims() as i64,
                    block.page_size() as i64,
                    block.inner_dim() as i64,
                ),
                block::BlockType::Device(block) => (
                    block.num_outer_dims() as i64,
                    block.page_size() as i64,
                    block.inner_dim() as i64,
                ),
            };
        }

        // Create the DLPack tensor
        dlpack::dlpack(
            py,
            self.inner.clone(),
            ptr,
            vec![1, 1, num_outer_dims, page_size, inner_dim],
            self.dtype,
            self.device_id,
        )
    }

    #[pyo3(signature = ())]
    fn __dlpack_device__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        dlpack::dlpack_device(py, self.inner.clone(), self.device_id)
    }
}
