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

#![cfg(feature = "block-manager")]
// Silence warnings about deprecated features (like pyo3::IntoPy::into_py)
#![allow(deprecated)]

use super::*;

use dlpark::prelude::{DataType, Device, ManagerCtx, ShapeAndStrides, ToTensor};
use pyo3::{ffi::c_str, prelude::IntoPy, types::PyTuple, PyObject, PyResult, Python};
use std::sync::{Arc, Mutex};

use dynamo_llm::block_manager::block::BlockDataExt;

pub enum BlockType {
    Pinned(
        dynamo_llm::block_manager::block::MutableBlock<
            dynamo_llm::block_manager::storage::PinnedStorage,
            dynamo_llm::block_manager::block::BasicMetadata,
        >,
    ),
    Device(
        dynamo_llm::block_manager::block::MutableBlock<
            dynamo_llm::block_manager::storage::DeviceStorage,
            dynamo_llm::block_manager::block::BasicMetadata,
        >,
    ),
}

struct DlPackTensor {
    block: Arc<Mutex<BlockType>>,
    // TODO: Metadata should be stored in the block manager?
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

impl ToTensor for DlPackTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        let mut mutable_block = self.block.lock().unwrap();
        let ptr = match &mut *mutable_block {
            BlockType::Pinned(block) => {
                let mut block_view_mut = block
                    .block_view_mut()
                    .expect("Failed to get mutable Pinned block view");
                unsafe { block_view_mut.as_mut_ptr() }
            }
            BlockType::Device(block) => {
                let mut block_view_mut = block
                    .block_view_mut()
                    .expect("Failed to get mutable Device block view");
                unsafe { block_view_mut.as_mut_ptr() }
            }
        };
        ptr as *mut std::ffi::c_void
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        let mutable_block = self.block.lock().unwrap();
        match &*mutable_block {
            BlockType::Pinned(_) => {
                // TODO: Why torch does not support CPU_PINNED here?
                /*Device {
                    device_type: DeviceType::CudaHost,
                    device_id: 0,
                }*/
                Device::CPU
            }
            BlockType::Device(_) => Device::cuda(self.device_id),
        }
    }

    fn dtype(&self) -> DataType {
        // Map from dynamo_llm::common::dtype::DType to dlpark::prelude::DataType
        match self.dtype {
            dynamo_llm::common::dtype::DType::FP8 => {
                // No direct FP8 equivalent, use U8 as closest alternative
                DataType::U8
            }
            dynamo_llm::common::dtype::DType::FP16 => DataType::F16,
            dynamo_llm::common::dtype::DType::BF16 => DataType::BF16,
            dynamo_llm::common::dtype::DType::FP32 => DataType::F32,
            dynamo_llm::common::dtype::DType::U8 => DataType::U8,
            dynamo_llm::common::dtype::DType::U16 => DataType::U16,
            dynamo_llm::common::dtype::DType::U32 => DataType::U32,
            dynamo_llm::common::dtype::DType::U64 => DataType::U64,
            dynamo_llm::common::dtype::DType::I8 => DataType::I8,
            dynamo_llm::common::dtype::DType::I16 => DataType::I16,
            dynamo_llm::common::dtype::DType::I32 => DataType::I32,
            dynamo_llm::common::dtype::DType::I64 => DataType::I64,
        }
    }

    fn shape_and_strides(&self) -> ShapeAndStrides {
        let mutable_block = self.block.lock().unwrap();
        let (num_blocks, num_layers, page_size, inner_dim) = match &*mutable_block {
            BlockType::Pinned(block) => (
                block.num_blocks(),
                block.num_layers(),
                block.page_size(),
                block.inner_dim(),
            ),
            BlockType::Device(block) => (
                block.num_blocks(),
                block.num_layers(),
                block.page_size(),
                block.inner_dim(),
            ),
        };
        let shape_i64: Vec<i64> = vec![
            num_blocks as i64,
            num_layers as i64,
            page_size as i64,
            inner_dim as i64,
        ];
        ShapeAndStrides::new_contiguous(&shape_i64)
    }
}

/*impl Drop for DlPackTensor {
    fn drop(&mut self) {
        println!("Dropping DlPackTensor");
    }
}*/

#[pyclass]
pub struct Block {
    inner: Arc<Mutex<BlockType>>,
    // TODO: Metadata should be stored in the block manager?
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

impl Block {
    pub fn from_rust(
        block: Arc<Mutex<BlockType>>,
        dtype: dynamo_llm::common::dtype::DType,
        device_id: usize,
    ) -> Self {
        Self {
            inner: block,
            dtype: dtype,
            device_id: device_id,
        }
    }
}

#[pymethods]
impl Block {
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
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
            dtype: self.dtype.clone(),
            device_id: self.device_id,
        });
        let py_capsule = Python::with_gil(|py| manager_ctx.into_py(py));
        Ok(py_capsule)
    }

    fn __dlpack_device__(&self) -> PyResult<Py<PyTuple>> {
        let dlpack_device = Python::with_gil(|py| {
            let device_type_list = py.eval(c_str!("[('CPU', 1), ('CUDA', 2), ('CPU_PINNED', 3), ('OPENCL', 4), ('VULKAN', 7), ('METAL', 8), ('VPI', 9), ('ROCM', 10)]"), None, None).unwrap();
            let device_type_enum = py
                .import("enum")
                .unwrap()
                .getattr("Enum")
                .unwrap()
                .call1(("DLDeviceType", device_type_list))
                .unwrap();
            let block = self.inner.lock().unwrap();
            let device_type = match &*block {
                BlockType::Pinned(_) => device_type_enum.getattr("CPU_PINNED").unwrap(),
                BlockType::Device(_) => device_type_enum.getattr("CUDA").unwrap(),
            };
            let device_id = self.device_id.into_py(py).into_bound(py);
            let device = vec![device_type, device_id];
            PyTuple::new(py, device).unwrap().unbind()
        });
        Ok(dlpack_device)
    }
}

/*impl Drop for Block {
    fn drop(&mut self) {
        println!("Dropping Block");
    }
}*/
