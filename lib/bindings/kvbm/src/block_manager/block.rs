// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "block-manager")]

use super::*;
use dynamo_llm::block_manager::block::BlockDataExt;
use dynamo_llm::block_manager::block::BlockDataProviderMut;
use pyo3::{
    types::{PyList, PyTuple},
    PyObject, PyResult, Python,
};
use std::sync::{Arc, Mutex};

pub enum BlockType {
    Pinned(
        dynamo_llm::block_manager::block::MutableBlock<
            dynamo_llm::block_manager::storage::PinnedStorage,
            dynamo_llm::block_manager::block::locality::Local,
            dynamo_llm::block_manager::block::BasicMetadata,
        >,
    ),
    Device(
        dynamo_llm::block_manager::block::MutableBlock<
            dynamo_llm::block_manager::storage::DeviceStorage,
            dynamo_llm::block_manager::block::locality::Local,
            dynamo_llm::block_manager::block::BasicMetadata,
        >,
    ),
}

#[pyclass]
pub struct Block {
    inner: Arc<Mutex<BlockType>>,
    // TODO: Metadata should be stored in the block manager?
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
    // Python iterator state
    py_itr_idx: usize,
}

impl Block {
    pub fn from_rust(
        block: Arc<Mutex<BlockType>>,
        dtype: dynamo_llm::common::dtype::DType,
        device_id: usize,
    ) -> Self {
        Self {
            inner: block,
            dtype,
            device_id,
            py_itr_idx: 0,
        }
    }

    fn num_layers(&self) -> usize {
        let mutable_block = self.inner.lock().unwrap();
        match &*mutable_block {
            BlockType::Pinned(block) => block.num_layers(),
            BlockType::Device(block) => block.num_layers(),
        }
    }
}

#[pymethods]
impl Block {
    #[pyo3(signature = ())]
    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let layers: Vec<layer::Layer> = (0..self.num_layers())
            .map(|layer_idx| {
                layer::Layer::from_rust(self.inner.clone(), layer_idx, self.dtype, self.device_id)
            })
            .collect();
        PyList::new(py, layers)
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.num_layers())
    }

    fn __getitem__(&self, index: usize) -> PyResult<layer::Layer> {
        let num_layers = self.num_layers();
        if index >= num_layers {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range for Block with {} layers",
                index, num_layers
            )));
        }
        let layer = layer::Layer::from_rust(self.inner.clone(), index, self.dtype, self.device_id);
        Ok(layer)
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        // Reset iterator index at the beginning of each iteration
        // Use to_list() for iterating concurrently
        slf.py_itr_idx = 0;
        Ok(slf)
    }

    fn __next__(&mut self) -> PyResult<layer::Layer> {
        if self.py_itr_idx >= self.num_layers() {
            return Err(pyo3::exceptions::PyStopIteration::new_err(
                "No more items in Block",
            ));
        }
        let layer = layer::Layer::from_rust(
            self.inner.clone(),
            self.py_itr_idx,
            self.dtype,
            self.device_id,
        );
        self.py_itr_idx += 1;
        Ok(layer)
    }

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
        let num_blocks: i64;
        let num_layers: i64;
        let num_outer_dims: i64;
        let page_size: i64;
        let inner_dim: i64;
        {
            let mut mutable_block = self.inner.lock().unwrap();
            ptr = match &mut *mutable_block {
                BlockType::Pinned(block) => {
                    use dynamo_llm::block_manager::block::private::PrivateToken;
                    let block_data = block.block_data_mut(PrivateToken);
                    let mut block_view_mut = block_data.block_view_mut().map_err(to_pyerr)?;
                    (unsafe { block_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
                }
                BlockType::Device(block) => {
                    use dynamo_llm::block_manager::block::private::PrivateToken;
                    let block_data = block.block_data_mut(PrivateToken);
                    let mut block_view_mut = block_data.block_view_mut().map_err(to_pyerr)?;
                    (unsafe { block_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
                }
            };
            (num_blocks, num_layers, num_outer_dims, page_size, inner_dim) = match &*mutable_block {
                BlockType::Pinned(block) => (
                    block.num_blocks() as i64,
                    block.num_layers() as i64,
                    block.num_outer_dims() as i64,
                    block.page_size() as i64,
                    block.inner_dim() as i64,
                ),
                BlockType::Device(block) => (
                    block.num_blocks() as i64,
                    block.num_layers() as i64,
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
            vec![num_blocks, num_layers, num_outer_dims, page_size, inner_dim],
            self.dtype,
            self.device_id,
        )
    }

    #[pyo3(signature = ())]
    fn __dlpack_device__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        dlpack::dlpack_device(py, self.inner.clone(), self.device_id)
    }
}
