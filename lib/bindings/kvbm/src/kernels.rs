// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaContext, DevicePtr};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySequence;

use kvbm_kernels::{
    BlockLayout, OperationalCopyBackend, OperationalCopyDirection, TensorDataType,
    block_from_universal, operational_copy, universal_from_block,
};

use cudarc::runtime::sys as cuda_runtime;

static CUDA_CONTEXT: OnceCell<Arc<CudaContext>> = OnceCell::new();

// TODO: determine the right way to get the CUDA context for the python bindings
// this is currently disabled, but we'll migrate this to the bindings crate
fn get_context() -> PyResult<Arc<CudaContext>> {
    let ctx = CUDA_CONTEXT.get_or_try_init(|| {
        CudaContext::new(0).map_err(|err| {
            PyRuntimeError::new_err(format!("Failed to create CUDA context: {:?}", err))
        })
    })?;
    Ok(ctx.clone())
}

fn map_dtype(dtype_str: &str) -> PyResult<(TensorDataType, usize)> {
    match dtype_str {
        "torch.float16" => Ok((TensorDataType::F16, 2)),
        "torch.bfloat16" => Ok((TensorDataType::BF16, 2)),
        "torch.float32" => Ok((TensorDataType::F32, 4)),
        "torch.float64" => Ok((TensorDataType::F64, 8)),
        other => Err(PyTypeError::new_err(format!(
            "Unsupported tensor dtype: {other}"
        ))),
    }
}

fn parse_layout(layout: &str) -> PyResult<BlockLayout> {
    match layout {
        "NHD" | "nhd" | "Nhd" => Ok(BlockLayout::NHD),
        "HND" | "hnd" | "Hnd" => Ok(BlockLayout::HND),
        other => Err(PyValueError::new_err(format!(
            "Unsupported layout '{other}', expected 'NHD' or 'HND'"
        ))),
    }
}

struct TensorInfo {
    ptr: usize,
    shape: Vec<usize>,
    dtype: TensorDataType,
    elem_size: usize,
    device_index: i64,
}

fn tensor_info(_py: Python<'_>, tensor: &Bound<'_, PyAny>) -> PyResult<TensorInfo> {
    if !tensor.hasattr("is_cuda")? {
        return Err(PyTypeError::new_err(
            "Expected a torch.Tensor with CUDA storage",
        ));
    }

    let is_cuda: bool = tensor.getattr("is_cuda")?.extract()?;
    if !is_cuda {
        return Err(PyValueError::new_err(
            "Tensor must reside on CUDA device (device.type == 'cuda')",
        ));
    }

    let contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !contiguous {
        return Err(PyValueError::new_err(
            "Tensor must be contiguous. Call tensor.contiguous() before passing it in.",
        ));
    }

    let device_obj = tensor.getattr("device")?;
    let device_type: String = device_obj.getattr("type")?.extract()?;
    if device_type != "cuda" {
        return Err(PyValueError::new_err(format!(
            "Tensor must reside on CUDA device, found device type '{device_type}'"
        )));
    }
    let device_index: Option<i64> = device_obj.getattr("index")?.extract()?;

    let dtype_obj = tensor.getattr("dtype")?;
    let dtype_py_str = dtype_obj.str()?;
    let dtype_str = dtype_py_str.to_str()?;
    let (dtype, elem_size) = map_dtype(dtype_str)?;

    let shape_py = tensor.getattr("shape")?;
    let shape: Vec<usize> = shape_py.extract()?;

    let ptr: usize = tensor.call_method0("data_ptr")?.extract()?;

    // Drop references before returning to avoid keeping Python objects alive unnecessarily
    Ok(TensorInfo {
        ptr,
        shape,
        dtype,
        elem_size,
        device_index: device_index.unwrap_or(0),
    })
}

/// Validate and flatten the `[nl * no]` block pointer table for a single logical block.
fn collect_block_pointers(
    py: Python<'_>,
    blocks: &Bound<'_, PyAny>,
    expected_len: usize,
    expected_dtype: TensorDataType,
    expected_device: i64,
    expected_shape: Option<&[usize]>,
    expected_numel: Option<usize>,
) -> PyResult<Vec<usize>> {
    if expected_shape.is_none() && expected_numel.is_none() {
        return Err(PyRuntimeError::new_err(
            "Internal error: either expected_shape or expected_numel must be provided",
        ));
    }
    let seq = blocks.downcast::<PySequence>()?;
    let seq_len = seq.len()?;
    if seq_len != expected_len {
        return Err(PyValueError::new_err(format!(
            "Expected {expected_len} block tensors (nl * no), received {seq_len}"
        )));
    }

    let mut ptrs = Vec::with_capacity(expected_len);
    for idx in 0..seq_len {
        let item = seq.get_item(idx)?;
        let info = tensor_info(py, &item)?;
        if info.dtype != expected_dtype {
            return Err(PyTypeError::new_err(format!(
                "Block tensor at index {idx} has mismatched dtype. Expected {:?}, got {:?}",
                expected_dtype, info.dtype
            )));
        }
        if info.device_index != expected_device {
            return Err(PyValueError::new_err(format!(
                "Block tensor at index {idx} is on CUDA device {}, but expected device {}",
                info.device_index, expected_device
            )));
        }
        if let Some(shape) = expected_shape
            && info.shape != shape
        {
            return Err(PyValueError::new_err(format!(
                "Block tensor at index {idx} has unexpected shape {:?}; expected {:?}",
                info.shape, shape
            )));
        }
        if let Some(numel) = expected_numel {
            let actual_numel: usize = info.shape.iter().product();
            if actual_numel != numel {
                return Err(PyValueError::new_err(format!(
                    "Block tensor at index {idx} has {actual_numel} elements; expected {numel}"
                )));
            }
        }
        ptrs.push(info.ptr);
    }
    Ok(ptrs)
}

/// Treat any Python sequence (list, tuple, generator) as a Vec of Bound<PyAny>.
fn sequence_items<'py>(
    _py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let seq = obj.downcast::<PySequence>()?;
    let len = seq.len()?;
    let mut items = Vec::with_capacity(len);
    for idx in 0..len {
        let item = seq.get_item(idx)?;
        items.push(item);
    }
    Ok(items)
}

fn parse_backend(label: Option<&str>) -> PyResult<OperationalCopyBackend> {
    let Some(raw) = label else {
        return Ok(OperationalCopyBackend::Auto);
    };
    let normalized = raw.trim().to_ascii_lowercase();
    let backend = match normalized.as_str() {
        "auto" => OperationalCopyBackend::Auto,
        "kernel" | "kernel_only" => OperationalCopyBackend::KernelOnly,
        "async" | "memcpy_async" | "cuda_memcpy_async" => OperationalCopyBackend::MemcpyAsync,
        "batch" | "memcpy_batch" | "cuda_memcpy_batch" => OperationalCopyBackend::MemcpyBatch,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown backend '{other}'. Expected 'auto', 'kernel', 'async', or 'batch'"
            )));
        }
    };
    Ok(backend)
}

fn to_cuda_error(err: cuda_runtime::cudaError_t) -> PyErr {
    PyRuntimeError::new_err(format!("CUDA error: {:?}", err))
}

/// Copy one or more NHD/HND block stacks into universal tensors.
///
/// Parameters
/// ----------
/// blocks : Sequence[Sequence[torch.Tensor]]
///     `nb` groups, each containing `nl * no` CUDA tensors shaped like
///     `[nt, nh, hd]` (layout `NHD`) or `[nh, nt, hd]` (layout `HND`).
/// universals : Sequence[torch.Tensor] or torch.Tensor
///     `nb` CUDA tensors shaped `[nh, nl, no, nt, hd]`. A single tensor is
///     treated as `nb = 1`.
/// layout : {"NHD", "HND"}
///     Describes how each block chunk is laid out in memory.
#[pyfunction]
unsafe fn block_to_universal(
    py: Python<'_>,
    blocks: &Bound<'_, PyAny>,
    universals: &Bound<'_, PyAny>,
    layout: &str,
) -> PyResult<()> {
    let ctx = get_context()?;
    ctx.bind_to_thread()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind context: {:?}", e)))?;
    let stream = ctx.default_stream();
    let layout_enum = parse_layout(layout)?;

    let universal_items = if universals.hasattr("data_ptr")? {
        vec![universals.clone()]
    } else {
        sequence_items(py, universals)?
    };

    if universal_items.is_empty() {
        return Ok(());
    }

    let mut universal_infos = Vec::with_capacity(universal_items.len());
    for item in &universal_items {
        universal_infos.push(tensor_info(py, item)?);
    }

    let base_info = &universal_infos[0];
    if base_info.shape.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "Universal tensor must have 5 dimensions [nh, nl, no, nt, hd], found {:?}",
            base_info.shape
        )));
    }

    for (idx, info) in universal_infos.iter().enumerate() {
        if info.shape != base_info.shape {
            return Err(PyValueError::new_err(format!(
                "Universal tensor {} has mismatched shape {:?}; expected {:?}",
                idx, info.shape, base_info.shape
            )));
        }
        if info.dtype != base_info.dtype {
            return Err(PyTypeError::new_err(format!(
                "Universal tensor {} has mismatched dtype {:?}; expected {:?}",
                idx, info.dtype, base_info.dtype
            )));
        }
        if info.device_index != base_info.device_index {
            return Err(PyValueError::new_err(format!(
                "Universal tensor {} is on CUDA device {}; expected device {}",
                idx, info.device_index, base_info.device_index
            )));
        }
    }

    let nh = base_info.shape[0];
    let nl = base_info.shape[1];
    let no = base_info.shape[2];
    let nt = base_info.shape[3];
    let hd = base_info.shape[4];
    let chunk_count = nl * no;

    let expected_block_shape = match layout_enum {
        BlockLayout::NHD => vec![nt, nh, hd],
        BlockLayout::HND => vec![nh, nt, hd],
    };

    let block_groups = sequence_items(py, blocks)?;
    if block_groups.len() != universal_infos.len() {
        return Err(PyValueError::new_err(format!(
            "Expected {} block pointer groups, received {}",
            universal_infos.len(),
            block_groups.len()
        )));
    }

    let mut block_ptr_values: Vec<usize> = Vec::with_capacity(universal_infos.len() * chunk_count);
    for (block_idx, group) in block_groups.iter().enumerate() {
        let ptrs = collect_block_pointers(
            py,
            group,
            chunk_count,
            universal_infos[block_idx].dtype,
            universal_infos[block_idx].device_index,
            Some(&expected_block_shape),
            None,
        )?;
        block_ptr_values.extend(ptrs);
    }

    let block_ptrs_device = stream
        .memcpy_stod(block_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to upload pointer buffer: {:?}", e))
        })?;

    let universal_ptr_values: Vec<usize> = universal_infos.iter().map(|info| info.ptr).collect();
    let universal_ptrs_device = stream
        .memcpy_stod(universal_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to upload universal pointer buffer: {:?}",
                e
            ))
        })?;

    let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
    let block_ptrs_device_ptr = block_ptrs_device_raw as usize as *const *const c_void;
    let (universal_ptrs_device_raw, _univ_guard) = universal_ptrs_device.device_ptr(&stream);
    let universal_ptrs_device_ptr = universal_ptrs_device_raw as usize as *const *mut c_void;

    let status = unsafe {
        universal_from_block(
            universal_ptrs_device_ptr,
            block_ptrs_device_ptr,
            universal_infos.len(),
            nh,
            nl,
            no,
            nt,
            hd,
            base_info.dtype,
            layout_enum,
            stream.cu_stream() as cuda_runtime::cudaStream_t,
        )
    };

    if status != cuda_runtime::cudaError::cudaSuccess {
        return Err(to_cuda_error(status));
    }

    stream
        .synchronize()
        .map_err(|e| PyRuntimeError::new_err(format!("Stream sync failed: {:?}", e)))?;
    Ok(())
}

/// Scatter universal tensors back into their per-layer block stacks.
///
/// Parameters mirror `block_to_universal`. The `blocks` argument should be a
/// list of CUDA tensors that will be populated in-place.
#[pyfunction]
unsafe fn universal_to_block(
    py: Python<'_>,
    universals: &Bound<'_, PyAny>,
    blocks: &Bound<'_, PyAny>,
    layout: &str,
) -> PyResult<()> {
    let ctx = get_context()?;
    ctx.bind_to_thread()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind context: {:?}", e)))?;
    let stream = ctx.default_stream();
    let layout_enum = parse_layout(layout)?;

    let universal_items = if universals.hasattr("data_ptr")? {
        vec![universals.clone()]
    } else {
        sequence_items(py, universals)?
    };

    if universal_items.is_empty() {
        return Ok(());
    }

    let mut universal_infos = Vec::with_capacity(universal_items.len());
    for item in &universal_items {
        universal_infos.push(tensor_info(py, item)?);
    }

    let base_info = &universal_infos[0];
    if base_info.shape.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "Universal tensor must have 5 dimensions [nh, nl, no, nt, hd], found {:?}",
            base_info.shape
        )));
    }

    for (idx, info) in universal_infos.iter().enumerate() {
        if info.shape != base_info.shape {
            return Err(PyValueError::new_err(format!(
                "Universal tensor {} has mismatched shape {:?}; expected {:?}",
                idx, info.shape, base_info.shape
            )));
        }
        if info.dtype != base_info.dtype {
            return Err(PyTypeError::new_err(format!(
                "Universal tensor {} has mismatched dtype {:?}; expected {:?}",
                idx, info.dtype, base_info.dtype
            )));
        }
        if info.device_index != base_info.device_index {
            return Err(PyValueError::new_err(format!(
                "Universal tensor {} is on CUDA device {}; expected device {}",
                idx, info.device_index, base_info.device_index
            )));
        }
    }

    let nh = base_info.shape[0];
    let nl = base_info.shape[1];
    let no = base_info.shape[2];
    let nt = base_info.shape[3];
    let hd = base_info.shape[4];
    let chunk_count = nl * no;

    let expected_block_shape = match layout_enum {
        BlockLayout::NHD => vec![nt, nh, hd],
        BlockLayout::HND => vec![nh, nt, hd],
    };

    let block_groups = sequence_items(py, blocks)?;
    if block_groups.len() != universal_infos.len() {
        return Err(PyValueError::new_err(format!(
            "Expected {} block pointer groups, received {}",
            universal_infos.len(),
            block_groups.len()
        )));
    }

    let mut block_ptr_values: Vec<usize> = Vec::with_capacity(universal_infos.len() * chunk_count);
    for (block_idx, group) in block_groups.iter().enumerate() {
        let ptrs = collect_block_pointers(
            py,
            group,
            chunk_count,
            universal_infos[block_idx].dtype,
            universal_infos[block_idx].device_index,
            Some(&expected_block_shape),
            None,
        )?;
        block_ptr_values.extend(ptrs);
    }

    let block_ptrs_device = stream
        .memcpy_stod(block_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to upload pointer buffer: {:?}", e))
        })?;

    let universal_ptr_values: Vec<usize> = universal_infos.iter().map(|info| info.ptr).collect();
    let universal_ptrs_device = stream
        .memcpy_stod(universal_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to upload universal pointer buffer: {:?}",
                e
            ))
        })?;

    let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
    let block_ptrs_device_ptr = block_ptrs_device_raw as usize as *const *mut c_void;
    let (universal_ptrs_device_raw, _univ_guard) = universal_ptrs_device.device_ptr(&stream);
    let universal_ptrs_device_ptr = universal_ptrs_device_raw as usize as *const *const c_void;

    let status = unsafe {
        block_from_universal(
            universal_ptrs_device_ptr,
            block_ptrs_device_ptr,
            universal_infos.len(),
            nh,
            nl,
            no,
            nt,
            hd,
            base_info.dtype,
            layout_enum,
            stream.cu_stream() as cuda_runtime::cudaStream_t,
        )
    };

    if status != cuda_runtime::cudaError::cudaSuccess {
        return Err(to_cuda_error(status));
    }

    stream
        .synchronize()
        .map_err(|e| PyRuntimeError::new_err(format!("Stream sync failed: {:?}", e)))?;
    Ok(())
}

/// Flatten block stacks into operational buffers (`[nl, no, inner]`).
///
/// Parameters
/// ----------
/// backend: Optional[str]
///     "auto" (default) tries the fused kernel → cudaMemcpyBatchAsync → cudaMemcpyAsync.
///     "kernel", "async", "batch" force the respective backend.
#[pyfunction]
#[pyo3(signature = (blocks, operationals, backend=None))]
unsafe fn block_to_operational(
    py: Python<'_>,
    blocks: &Bound<'_, PyAny>,
    operationals: &Bound<'_, PyAny>,
    backend: Option<&str>,
) -> PyResult<()> {
    let ctx = get_context()?;
    ctx.bind_to_thread()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind context: {:?}", e)))?;
    let stream = ctx.default_stream();

    let backend = parse_backend(backend)?;

    let operational_items = if operationals.hasattr("data_ptr")? {
        vec![operationals.clone()]
    } else {
        sequence_items(py, operationals)?
    };

    if operational_items.is_empty() {
        return Ok(());
    }

    let mut operational_infos = Vec::with_capacity(operational_items.len());
    for item in &operational_items {
        operational_infos.push(tensor_info(py, item)?);
    }

    let base_info = &operational_infos[0];
    if base_info.shape.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "Operational tensor must have 3 dimensions [nl, no, inner], found {:?}",
            base_info.shape
        )));
    }

    for (idx, info) in operational_infos.iter().enumerate() {
        if info.shape != base_info.shape {
            return Err(PyValueError::new_err(format!(
                "Operational tensor {} has mismatched shape {:?}; expected {:?}",
                idx, info.shape, base_info.shape
            )));
        }
        if info.dtype != base_info.dtype {
            return Err(PyTypeError::new_err(format!(
                "Operational tensor {} has mismatched dtype {:?}; expected {:?}",
                idx, info.dtype, base_info.dtype
            )));
        }
        if info.device_index != base_info.device_index {
            return Err(PyValueError::new_err(format!(
                "Operational tensor {} is on CUDA device {}; expected device {}",
                idx, info.device_index, base_info.device_index
            )));
        }
    }

    let nl = base_info.shape[0];
    let no = base_info.shape[1];
    let inner = base_info.shape[2];
    let chunk_count = nl * no;

    let block_groups = sequence_items(py, blocks)?;
    if block_groups.len() != operational_infos.len() {
        return Err(PyValueError::new_err(format!(
            "Expected {} block pointer groups, received {}",
            operational_infos.len(),
            block_groups.len()
        )));
    }

    let mut block_ptr_values: Vec<usize> =
        Vec::with_capacity(operational_infos.len() * chunk_count);
    for (block_idx, group) in block_groups.iter().enumerate() {
        let ptrs = collect_block_pointers(
            py,
            group,
            chunk_count,
            operational_infos[block_idx].dtype,
            operational_infos[block_idx].device_index,
            None,
            Some(inner),
        )?;
        block_ptr_values.extend(ptrs);
    }

    let block_ptrs_device = stream
        .memcpy_stod(block_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to upload pointer buffer: {:?}", e))
        })?;

    let block_ptrs_host: Vec<*const c_void> = block_ptr_values
        .iter()
        .map(|&ptr| ptr as *const c_void)
        .collect();

    let operational_ptrs_host: Vec<*mut c_void> = operational_infos
        .iter()
        .map(|info| info.ptr as *mut c_void)
        .collect();

    let operational_ptr_values: Vec<usize> =
        operational_infos.iter().map(|info| info.ptr).collect();
    let operational_ptrs_device = stream
        .memcpy_stod(operational_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to upload operational pointer buffer: {:?}",
                e
            ))
        })?;

    let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
    let block_ptrs_device_ptr = block_ptrs_device_raw as usize as *const *const c_void;
    let (operational_ptrs_device_raw, _op_guard) = operational_ptrs_device.device_ptr(&stream);
    let operational_ptrs_device_ptr = operational_ptrs_device_raw as usize as *const *const c_void;

    let status = unsafe {
        operational_copy(
            block_ptrs_host.as_ptr(),
            block_ptrs_device_ptr,
            operational_ptrs_host.as_ptr(),
            operational_ptrs_device_ptr,
            operational_infos.len(),
            nl,
            no,
            inner,
            base_info.elem_size,
            base_info.dtype,
            OperationalCopyDirection::BlockToOperational,
            backend,
            stream.cu_stream() as cuda_runtime::cudaStream_t,
        )
    };

    if status != cuda_runtime::cudaError::cudaSuccess {
        return Err(to_cuda_error(status));
    }

    stream
        .synchronize()
        .map_err(|e| PyRuntimeError::new_err(format!("Stream sync failed: {:?}", e)))?;
    Ok(())
}

/// Restore block stacks from operational buffers.
///
/// Parameters
/// ----------
/// backend: Optional[str]
///     Same semantics as `block_to_operational`.
#[pyfunction]
#[pyo3(signature = (operationals, blocks, backend=None))]
unsafe fn operational_to_block(
    py: Python<'_>,
    operationals: &Bound<'_, PyAny>,
    blocks: &Bound<'_, PyAny>,
    backend: Option<&str>,
) -> PyResult<()> {
    let ctx = get_context()?;
    ctx.bind_to_thread()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind context: {:?}", e)))?;
    let stream = ctx.default_stream();

    let backend = parse_backend(backend)?;

    let operational_items = if operationals.hasattr("data_ptr")? {
        vec![operationals.clone()]
    } else {
        sequence_items(py, operationals)?
    };

    if operational_items.is_empty() {
        return Ok(());
    }

    let mut operational_infos = Vec::with_capacity(operational_items.len());
    for item in &operational_items {
        operational_infos.push(tensor_info(py, item)?);
    }

    let base_info = &operational_infos[0];
    if base_info.shape.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "Operational tensor must have 3 dimensions [nl, no, inner], found {:?}",
            base_info.shape
        )));
    }

    for (idx, info) in operational_infos.iter().enumerate() {
        if info.shape != base_info.shape {
            return Err(PyValueError::new_err(format!(
                "Operational tensor {} has mismatched shape {:?}; expected {:?}",
                idx, info.shape, base_info.shape
            )));
        }
        if info.dtype != base_info.dtype {
            return Err(PyTypeError::new_err(format!(
                "Operational tensor {} has mismatched dtype {:?}; expected {:?}",
                idx, info.dtype, base_info.dtype
            )));
        }
        if info.device_index != base_info.device_index {
            return Err(PyValueError::new_err(format!(
                "Operational tensor {} is on CUDA device {}; expected device {}",
                idx, info.device_index, base_info.device_index
            )));
        }
    }

    let nl = base_info.shape[0];
    let no = base_info.shape[1];
    let inner = base_info.shape[2];
    let chunk_count = nl * no;

    let block_groups = sequence_items(py, blocks)?;
    if block_groups.len() != operational_infos.len() {
        return Err(PyValueError::new_err(format!(
            "Expected {} block pointer groups, received {}",
            operational_infos.len(),
            block_groups.len()
        )));
    }

    let mut block_ptr_values: Vec<usize> =
        Vec::with_capacity(operational_infos.len() * chunk_count);
    for (block_idx, group) in block_groups.iter().enumerate() {
        let ptrs = collect_block_pointers(
            py,
            group,
            chunk_count,
            operational_infos[block_idx].dtype,
            operational_infos[block_idx].device_index,
            None,
            Some(inner),
        )?;
        block_ptr_values.extend(ptrs);
    }

    let block_ptrs_device = stream
        .memcpy_stod(block_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to upload pointer buffer: {:?}", e))
        })?;

    let block_ptrs_host: Vec<*const c_void> = block_ptr_values
        .iter()
        .map(|&ptr| ptr as *const c_void)
        .collect();

    let operational_ptrs_host: Vec<*mut c_void> = operational_infos
        .iter()
        .map(|info| info.ptr as *mut c_void)
        .collect();

    let operational_ptr_values: Vec<usize> =
        operational_infos.iter().map(|info| info.ptr).collect();
    let operational_ptrs_device = stream
        .memcpy_stod(operational_ptr_values.as_slice())
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to upload operational pointer buffer: {:?}",
                e
            ))
        })?;

    let (block_ptrs_device_raw, _block_guard) = block_ptrs_device.device_ptr(&stream);
    let block_ptrs_device_ptr = block_ptrs_device_raw as usize as *const *const c_void;
    let (operational_ptrs_device_raw, _op_guard) = operational_ptrs_device.device_ptr(&stream);
    let operational_ptrs_device_ptr = operational_ptrs_device_raw as usize as *const *const c_void;

    let status = unsafe {
        operational_copy(
            block_ptrs_host.as_ptr(),
            block_ptrs_device_ptr,
            operational_ptrs_host.as_ptr(),
            operational_ptrs_device_ptr,
            operational_infos.len(),
            nl,
            no,
            inner,
            base_info.elem_size,
            base_info.dtype,
            OperationalCopyDirection::OperationalToBlock,
            backend,
            stream.cu_stream() as cuda_runtime::cudaStream_t,
        )
    };

    if status != cuda_runtime::cudaError::cudaSuccess {
        return Err(to_cuda_error(status));
    }

    stream
        .synchronize()
        .map_err(|e| PyRuntimeError::new_err(format!("Stream sync failed: {:?}", e)))?;
    Ok(())
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(block_to_universal, m)?)?;
    m.add_function(wrap_pyfunction!(universal_to_block, m)?)?;
    m.add_function(wrap_pyfunction!(block_to_operational, m)?)?;
    m.add_function(wrap_pyfunction!(operational_to_block, m)?)?;
    Ok(())
}
