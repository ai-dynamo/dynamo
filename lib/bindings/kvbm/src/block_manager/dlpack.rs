// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "block-manager")]
// Silence warnings about deprecated features (like pyo3::IntoPy::into_py)
#![allow(deprecated)]

use super::*;
use dlpark::prelude::{DataType, Device, ManagerCtx, ShapeAndStrides, ToTensor};
use pyo3::{ffi::c_str, prelude::IntoPy, types::PyTuple, PyObject, PyResult, Python};
use std::sync::{Arc, Mutex};

enum DlPackOwner {
    Single(Arc<Mutex<block::BlockType>>),
    Many(Vec<Arc<Mutex<block::BlockType>>>),
}

impl DlPackOwner {
    fn first_block(&self) -> &Arc<Mutex<block::BlockType>> {
        match self {
            Self::Single(block) => block,
            Self::Many(blocks) => &blocks[0],
        }
    }
}

struct DlPackTensor {
    owner: DlPackOwner,
    ptr: *mut std::ffi::c_void,
    shape_and_strides: ShapeAndStrides,
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
}

impl ToTensor for DlPackTensor {
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Device {
        let mutable_block = self.owner.first_block().lock().unwrap();
        match &*mutable_block {
            block::BlockType::Pinned(_) => {
                // TODO: Why torch does not support CPU_PINNED here?
                /*Device {
                    device_type: DeviceType::CudaHost,
                    device_id: 0,
                }*/
                Device::CPU
            }
            block::BlockType::Device(_) => Device::cuda(self.device_id),
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
        match &self.shape_and_strides {
            ShapeAndStrides::Contiguous(shape) => ShapeAndStrides::new_contiguous(shape.iter()),
            ShapeAndStrides::WithStrides(values) => {
                let ndim = values.len() / 2;
                ShapeAndStrides::new_with_strides(values[..ndim].iter(), values[ndim..].iter())
            }
            ShapeAndStrides::Borrowed { .. } => {
                ShapeAndStrides::new_borrowed(
                    self.shape_and_strides.shape(),
                    self.shape_and_strides.strides(),
                )
            }
        }
    }
}

/*impl Drop for DlPackTensor {
    fn drop(&mut self) {
        println!("Dropping DlPackTensor");
    }
}*/

pub fn dlpack<'py>(
    py: Python<'py>,
    block: Arc<Mutex<block::BlockType>>,
    ptr: *mut std::ffi::c_void,
    shape: Vec<i64>,
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
) -> PyResult<PyObject> {
    let manager_ctx = ManagerCtx::new(DlPackTensor {
        owner: DlPackOwner::Single(block),
        ptr,
        shape_and_strides: ShapeAndStrides::new_contiguous(&shape),
        dtype,
        device_id,
    });
    let py_capsule = manager_ctx.into_py(py);
    Ok(py_capsule)
}

pub fn dlpack_many<'py>(
    py: Python<'py>,
    blocks: Vec<Arc<Mutex<block::BlockType>>>,
    ptr: *mut std::ffi::c_void,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    dtype: dynamo_llm::common::dtype::DType,
    device_id: usize,
) -> PyResult<PyObject> {
    let shape_and_strides = match strides {
        Some(strides) => ShapeAndStrides::new_with_strides(&shape, &strides),
        None => ShapeAndStrides::new_contiguous(&shape),
    };
    let manager_ctx = ManagerCtx::new(DlPackTensor {
        owner: DlPackOwner::Many(blocks),
        ptr,
        shape_and_strides,
        dtype,
        device_id,
    });
    let py_capsule = manager_ctx.into_py(py);
    Ok(py_capsule)
}

pub fn dlpack_device<'py>(
    py: Python<'py>,
    block: Arc<Mutex<block::BlockType>>,
    device_id: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let dev_type_list = py.eval(c_str!("[('CPU', 1), ('CUDA', 2), ('CPU_PINNED', 3), ('OPENCL', 4), ('VULKAN', 7), ('METAL', 8), ('VPI', 9), ('ROCM', 10)]"), None, None).unwrap();
    let dev_type_enum = py
        .import("enum")
        .unwrap()
        .getattr("Enum")
        .unwrap()
        .call1(("DLDeviceType", dev_type_list))
        .unwrap();
    let dev_type = match &*block.lock().unwrap() {
        block::BlockType::Pinned(_) => dev_type_enum.getattr("CPU_PINNED").unwrap(),
        block::BlockType::Device(_) => dev_type_enum.getattr("CUDA").unwrap(),
    };
    let dev_id = device_id.into_py(py).into_bound(py);
    let dev = vec![dev_type, dev_id];
    PyTuple::new(py, dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_strides_support_non_contiguous_views() {
        let shape = vec![4, 2, 8];
        let strides = vec![64, 32, 1];

        let tensor = DlPackTensor {
            owner: DlPackOwner::Many(Vec::new()),
            ptr: std::ptr::null_mut(),
            shape_and_strides: ShapeAndStrides::new_with_strides(&shape, &strides),
            dtype: dynamo_llm::common::dtype::DType::FP16,
            device_id: 0,
        };

        let exported = tensor.shape_and_strides();
        assert_eq!(exported.shape(), shape.as_slice());
        assert_eq!(exported.strides(), Some(strides.as_slice()));
        assert!(!exported.is_contiguous());
    }
}
