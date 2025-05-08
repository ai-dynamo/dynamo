// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Silence warnings about deprecated features (like pyo3::IntoPy::into_py)
#![allow(deprecated)]

use super::*;

use pyo3::{Python, PyObject, PyResult, ffi::c_str, prelude::IntoPy, types::PyTuple};
use dlpark::prelude::{Device, DataType, ManagerCtx, ShapeAndStrides, ToTensor};
use std::sync::{Arc, Mutex};

use dynamo_llm::block_manager::block::{BlockDataProviderMut, BlockDataExt};
// TODO: Remove this
use dynamo_llm::block_manager::block::private;

// TODO: Different storage types?
pub type BlockType = dynamo_llm::block_manager::block::MutableBlock<dynamo_llm::block_manager::storage::PinnedStorage, dynamo_llm::block_manager::block::BasicMetadata>;

struct DlPackTensor {
    block: Arc<Mutex<BlockType>>,
}

impl ToTensor for DlPackTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        let mut mutable_block = self.block.lock().unwrap();
        // TODO: How to access without private::PrivateToken?
        let block_data_mut = mutable_block.block_data_mut(private::PrivateToken);
        let mut block_view_mut = block_data_mut.block_view_mut().expect("Failed to get mutable block view");
        unsafe {
            block_view_mut.as_mut_ptr() as *mut std::ffi::c_void
        }
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        // TODO: Could there be different devices?
        // Why torch does not support CPU_PINNED here?
        /*Device {
            device_type: DeviceType::CudaHost,
            device_id: 0,
        }*/
        Device::CPU
    }

    fn dtype(&self) -> DataType {
        // TODO: Could there be different dtypes?
        DataType::F16
    }

    fn shape_and_strides(&self) -> ShapeAndStrides {
        let mutable_block = self.block.lock().unwrap();

        let num_blocks = mutable_block.num_blocks();
        let num_layers = mutable_block.num_layers();
        let page_size = mutable_block.page_size();
        let inner_dim = mutable_block.inner_dim();

        // TODO: Confirm this is correct?
        let shape_i64: Vec<i64> = vec![num_blocks as i64, num_layers as i64, page_size as i64, inner_dim as i64];
        ShapeAndStrides::new_contiguous(&shape_i64)
    }
}

impl Drop for DlPackTensor {
    fn drop(&mut self) {
        println!("Dropping DlPackTensor");
    }
}

#[pyclass]
pub struct Block {
    inner: Arc<Mutex<BlockType>>,
}

impl Block {
    pub fn from_rust(block: Arc<Mutex<BlockType>>) -> Self {
        Self { inner: block }
    }
}

#[pymethods]
impl Block {
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(&self, stream: Option<PyObject>, max_version: Option<PyObject>, dl_device: Option<PyObject>, copy: Option<bool>) -> PyResult<PyObject> {
        // Panic if any arguments are provided
        if stream.is_some() {
            panic!("stream argument is not supported");
        }
        if max_version.is_some() {
            panic!("max_version argument is not supported");
        }
        if dl_device.is_some() {
            panic!("dl_device argument is not supported");
        }
        if copy.is_some() {
            panic!("copy argument is not supported");
        }

        // Create DLPack PyCapsule
        let manager_ctx = ManagerCtx::new(DlPackTensor {
            block: self.inner.clone(),
        });
        let py_capsule = Python::with_gil(|py| {
            manager_ctx.into_py(py)
        });
        Ok(py_capsule)
    }

    fn __dlpack_device__(&self) -> PyResult<Py<PyTuple>> {
        let dlpack_device = Python::with_gil(|py| {
            let device_type_list = py.eval(c_str!("[('CPU', 1), ('CUDA', 2), ('CPU_PINNED', 3), ('OPENCL', 4), ('VULKAN', 7), ('METAL', 8), ('VPI', 9), ('ROCM', 10)]"), None, None).unwrap();
            let device_type_enum = py.import("enum").unwrap().getattr("Enum").unwrap().call1(("DLDeviceType", device_type_list)).unwrap();
            let device_type = device_type_enum.getattr("CPU_PINNED").unwrap();
            let device_id = (0_i32).into_py(py).into_bound(py);
            let device = vec![device_type, device_id];
            PyTuple::new(py, device).unwrap().unbind()
        });
        Ok(dlpack_device)
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        println!("Dropping Block");
    }
}
