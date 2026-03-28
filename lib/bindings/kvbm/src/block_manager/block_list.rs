// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "block-manager")]

use super::*;
use dynamo_llm::block_manager::block::BlockDataExt;
use dynamo_llm::block_manager::block::BlockDataProviderMut;
use pyo3::{types::{PyList, PyTuple}, PyObject, PyResult, Python};
use std::sync::{Arc, Mutex};

fn primary_pool_shape(
    num_blocks: i64,
    num_layers: i64,
    num_outer_dims: i64,
    page_size: i64,
    inner_dim: i64,
) -> Vec<i64> {
    vec![num_blocks, num_layers, num_outer_dims, page_size, inner_dim]
}

fn layer_pool_shape_and_strides(
    num_blocks: i64,
    num_layers: i64,
    num_outer_dims: i64,
    page_size: i64,
    inner_dim: i64,
) -> (Vec<i64>, Vec<i64>) {
    let block_stride = num_layers * num_outer_dims * page_size * inner_dim;
    let layer_stride = num_outer_dims * page_size * inner_dim;
    let outer_stride = page_size * inner_dim;
    let page_stride = inner_dim;

    (
        vec![num_blocks, 1, num_outer_dims, page_size, inner_dim],
        vec![block_stride, layer_stride, outer_stride, page_stride, 1],
    )
}

#[pyclass]
pub struct BlockList {
    inner: Vec<Arc<Mutex<block::BlockType>>>,
    // TODO: Metadata should be stored in the block manager?
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
    // Python iterator state
    py_itr_idx: usize,
}

impl BlockList {
    pub fn from_rust(
        block_list: Vec<block::BlockType>,
        dtype: dynamo_llm::common::dtype::DType,
        device_id: usize,
    ) -> Self {
        Self {
            inner: block_list
                .into_iter()
                .map(|b| Arc::new(Mutex::new(b)))
                .collect(),
            dtype,
            device_id,
            py_itr_idx: 0,
        }
    }

    fn first_block_dims(
        &self,
    ) -> PyResult<(i64, i64, i64, i64)> {
        let block = self
            .inner
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("BlockList is empty"))?;
        let mutable_block = block.lock().unwrap();
        let dims = match &*mutable_block {
            block::BlockType::Pinned(block) => (
                block.num_layers() as i64,
                block.num_outer_dims() as i64,
                block.page_size() as i64,
                block.inner_dim() as i64,
            ),
            block::BlockType::Device(block) => (
                block.num_layers() as i64,
                block.num_outer_dims() as i64,
                block.page_size() as i64,
                block.inner_dim() as i64,
            ),
        };
        Ok(dims)
    }

    fn first_block_ptr(&self) -> PyResult<*mut std::ffi::c_void> {
        let block = self
            .inner
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("BlockList is empty"))?;
        let mut mutable_block = block.lock().unwrap();
            let ptr = match &mut *mutable_block {
                block::BlockType::Pinned(block) => {
                    let block_data = block.block_data_mut();
                    let mut block_view_mut = block_data.block_view_mut().map_err(to_pyerr)?;
                    (unsafe { block_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
                }
                block::BlockType::Device(block) => {
                    let block_data = block.block_data_mut();
                    let mut block_view_mut = block_data.block_view_mut().map_err(to_pyerr)?;
                    (unsafe { block_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
                }
            };
        Ok(ptr)
    }
}

#[pymethods]
impl BlockList {
    #[pyo3(signature = ())]
    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let blocks: Vec<block::Block> = self
            .inner
            .iter()
            .map(|b| block::Block::from_rust(b.clone(), self.dtype, self.device_id))
            .collect();
        PyList::new(py, blocks)
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    fn __getitem__(&self, index: usize) -> PyResult<block::Block> {
        if index >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range for BlockList of length {}",
                index,
                self.inner.len()
            )));
        }
        let block = block::Block::from_rust(self.inner[index].clone(), self.dtype, self.device_id);
        Ok(block)
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        // Reset iterator index at the beginning of each iteration
        // Use to_list() for iterating concurrently
        slf.py_itr_idx = 0;
        Ok(slf)
    }

    fn __next__(&mut self) -> PyResult<block::Block> {
        if self.py_itr_idx >= self.inner.len() {
            return Err(pyo3::exceptions::PyStopIteration::new_err(
                "No more items in BlockList",
            ));
        }
        let block = block::Block::from_rust(
            self.inner[self.py_itr_idx].clone(),
            self.dtype,
            self.device_id,
        );
        self.py_itr_idx += 1;
        Ok(block)
    }

    fn layer_view(&self, layer_idx: usize) -> PyResult<BlockListLayerView> {
        let (num_layers, _, _, _) = self.first_block_dims()?;
        if layer_idx >= num_layers as usize {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Layer index {} out of range for BlockList with {} layers",
                layer_idx, num_layers
            )));
        }
        Ok(BlockListLayerView {
            inner: self.inner.clone(),
            layer_idx,
            dtype: self.dtype,
            device_id: self.device_id,
        })
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

        let (num_layers, num_outer_dims, page_size, inner_dim) = self.first_block_dims()?;
        let ptr = self.first_block_ptr()?;
        dlpack::dlpack_many(
            py,
            self.inner.clone(),
            ptr,
            primary_pool_shape(
                self.inner.len() as i64,
                num_layers,
                num_outer_dims,
                page_size,
                inner_dim,
            ),
            None,
            self.dtype,
            self.device_id,
        )
    }

    #[pyo3(signature = ())]
    fn __dlpack_device__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let first_block = self
            .inner
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("BlockList is empty"))?;
        dlpack::dlpack_device(py, first_block.clone(), self.device_id)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BlockListLayerView {
    inner: Vec<Arc<Mutex<block::BlockType>>>,
    layer_idx: usize,
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

impl BlockListLayerView {
    fn layer_ptr_and_dims(&self) -> PyResult<(*mut std::ffi::c_void, i64, i64, i64, i64)> {
        let block = self
            .inner
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("BlockListLayerView is empty"))?;
        let mut mutable_block = block.lock().unwrap();
        let ptr = match &mut *mutable_block {
            block::BlockType::Pinned(block) => {
                let block_data = block.block_data_mut();
                let mut layer_view_mut = block_data
                    .layer_view_mut(self.layer_idx, 0)
                    .map_err(to_pyerr)?;
                (unsafe { layer_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
            }
            block::BlockType::Device(block) => {
                let block_data = block.block_data_mut();
                let mut layer_view_mut = block_data
                    .layer_view_mut(self.layer_idx, 0)
                    .map_err(to_pyerr)?;
                (unsafe { layer_view_mut.as_mut_ptr() }) as *mut std::ffi::c_void
            }
        };
        let dims = match &*mutable_block {
            block::BlockType::Pinned(block) => (
                block.num_layers() as i64,
                block.num_outer_dims() as i64,
                block.page_size() as i64,
                block.inner_dim() as i64,
            ),
            block::BlockType::Device(block) => (
                block.num_layers() as i64,
                block.num_outer_dims() as i64,
                block.page_size() as i64,
                block.inner_dim() as i64,
            ),
        };
        Ok((ptr, dims.0, dims.1, dims.2, dims.3))
    }
}

#[pymethods]
impl BlockListLayerView {
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
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

        let (ptr, num_layers, num_outer_dims, page_size, inner_dim) = self.layer_ptr_and_dims()?;
        let (shape, strides) = layer_pool_shape_and_strides(
            self.inner.len() as i64,
            num_layers,
            num_outer_dims,
            page_size,
            inner_dim,
        );

        dlpack::dlpack_many(
            py,
            self.inner.clone(),
            ptr,
            shape,
            Some(strides),
            self.dtype,
            self.device_id,
        )
    }

    #[pyo3(signature = ())]
    fn __dlpack_device__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let first_block = self
            .inner
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("BlockListLayerView is empty"))?;
        dlpack::dlpack_device(py, first_block.clone(), self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_pool_shape_matches_block_first_layout() {
        assert_eq!(primary_pool_shape(4, 2, 3, 16, 8), vec![4, 2, 3, 16, 8]);
    }

    #[test]
    fn layer_pool_shape_and_strides_skip_other_layers() {
        let (shape, strides) = layer_pool_shape_and_strides(4, 2, 3, 16, 8);
        assert_eq!(shape, vec![4, 1, 3, 16, 8]);
        assert_eq!(strides, vec![768, 384, 128, 8, 1]);
    }
}
