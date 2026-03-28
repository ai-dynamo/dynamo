// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, wrap_pymodule};
#[cfg(not(test))]
use pyo3::wrap_pyfunction;

use super::vllm::{
    BlockState, BlockStates, KvbmBlockList, KvbmRequest, PyTrtllmKvConnectorLeader,
    PyTrtllmKvConnectorWorker, SchedulerOutput, SlotUpdate,
};
use super::{block, block_list};
use crate::to_pyerr;
use dynamo_llm::block_manager::BasicMetadata;
use dynamo_llm::block_manager::block::Blocks;
use dynamo_llm::block_manager::layout::{FullyContiguous, LayoutConfig};
use dynamo_llm::block_manager::storage::{DeviceAllocator, DeviceStorage};

fn dtype_from_name(dtype: &str) -> PyResult<dynamo_llm::common::dtype::DType> {
    match dtype {
        "float16" | "fp16" => Ok(dynamo_llm::common::dtype::DType::FP16),
        "bfloat16" | "bf16" => Ok(dynamo_llm::common::dtype::DType::BF16),
        "float32" | "fp32" => Ok(dynamo_llm::common::dtype::DType::FP32),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported dtype for TRTLLM primary pool export: {other}"
        ))),
    }
}

fn create_primary_pool_inner(
    num_blocks: usize,
    num_layers: usize,
    kv_factor: usize,
    page_size: usize,
    inner_dim: usize,
    dtype: &str,
    device_id: usize,
) -> PyResult<block_list::BlockList> {
    if num_blocks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_blocks must be greater than 0",
        ));
    }
    if num_layers == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_layers must be greater than 0",
        ));
    }
    if kv_factor == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "kv_factor must be greater than 0",
        ));
    }
    if page_size == 0 || inner_dim == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "page_size and inner_dim must be greater than 0",
        ));
    }

    let dtype = dtype_from_name(dtype)?;
    let config = LayoutConfig::builder()
        .num_blocks(num_blocks)
        .num_layers(num_layers)
        .outer_dim(kv_factor)
        .page_size(page_size)
        .inner_dim(inner_dim)
        .dtype_width_bytes(dtype.size_in_bytes())
        .build()
        .map_err(to_pyerr)?;
    let allocator = DeviceAllocator::new(device_id).map_err(to_pyerr)?;
    let layout = FullyContiguous::<DeviceStorage>::allocate(config, &allocator).map_err(to_pyerr)?;
    let blocks = Blocks::<_, BasicMetadata>::new(layout, 0, 0)
        .map_err(to_pyerr)?
        .into_blocks()
        .map_err(to_pyerr)?;
    let block_list = blocks
        .into_iter()
        .map(block::BlockType::DeviceOwned)
        .collect();

    Ok(block_list::BlockList::from_rust(block_list, dtype, device_id))
}

#[cfg(not(test))]
#[pyfunction(name = "create_primary_pool")]
#[pyo3(signature = (num_blocks, num_layers, kv_factor, page_size, inner_dim, dtype="float16", device_id=0))]
fn create_primary_pool(
    num_blocks: usize,
    num_layers: usize,
    kv_factor: usize,
    page_size: usize,
    inner_dim: usize,
    dtype: &str,
    device_id: usize,
) -> PyResult<block_list::BlockList> {
    create_primary_pool_inner(
        num_blocks,
        num_layers,
        kv_factor,
        page_size,
        inner_dim,
        dtype,
        device_id,
    )
}

#[pymodule]
fn _trtllm_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvbmRequest>()?;
    m.add_class::<KvbmBlockList>()?;
    m.add_class::<BlockState>()?;
    m.add_class::<BlockStates>()?;
    m.add_class::<SlotUpdate>()?;
    m.add_class::<PyTrtllmKvConnectorWorker>()?;
    m.add_class::<PyTrtllmKvConnectorLeader>()?;
    m.add_class::<SchedulerOutput>()?;
    #[cfg(not(test))]
    m.add_function(wrap_pyfunction!(create_primary_pool, m)?)?;
    Ok(())
}

/// Add bindings from this crate to the provided module.
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(_trtllm_integration))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_parser_accepts_supported_trtllm_pool_types() {
        assert!(matches!(
            dtype_from_name("float16").unwrap(),
            dynamo_llm::common::dtype::DType::FP16
        ));
        assert!(matches!(
            dtype_from_name("bf16").unwrap(),
            dynamo_llm::common::dtype::DType::BF16
        ));
        assert!(dtype_from_name("int8").is_err());
    }
}
