// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::{prelude::*, wrap_pymodule};
#[cfg(not(test))]
use pyo3::wrap_pyfunction;

use std::collections::HashMap;

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

#[derive(Default)]
struct RequestState {
    block_ids: Vec<usize>,
    capacity: usize,
    history_length: usize,
    active: bool,
    committed_tokens: usize,
}

#[pyclass]
pub struct TrtllmStateManager {
    tokens_per_block: usize,
    max_seq_len: usize,
    num_blocks: usize,
    max_blocks_per_seq: usize,
    max_num_sequences: usize,
    device_id: usize,
    world_size: usize,
    tp_size: usize,
    tp_rank: usize,
    pp_size: usize,
    pp_rank: usize,
    kv_factor: usize,
    cache_mode: String,
    _max_beam_width: usize,
    request_state: HashMap<u64, RequestState>,
    free_block_ids: Vec<usize>,
    request_slots: HashMap<u64, usize>,
    free_slots: Vec<usize>,
}

impl TrtllmStateManager {
    fn request_state_for(&self, request_id: u64) -> PyResult<&RequestState> {
        self.request_state.get(&request_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("request {request_id} is not active"))
        })
    }

    fn request_state_for_mut(&mut self, request_id: u64) -> PyResult<&mut RequestState> {
        self.request_state.get_mut(&request_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("request {request_id} is not active"))
        })
    }

    fn slot_for(&mut self, request_id: u64) -> PyResult<usize> {
        if let Some(slot) = self.request_slots.get(&request_id) {
            return Ok(*slot);
        }
        let slot = self.free_slots.first().copied().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no free TRTLLM cache slots available")
        })?;
        self.free_slots.remove(0);
        self.request_slots.insert(request_id, slot);
        Ok(slot)
    }

    fn required_blocks(&self, token_capacity: usize) -> usize {
        if token_capacity == 0 {
            return 0;
        }
        usize::min(
            token_capacity.div_ceil(self.tokens_per_block),
            self.max_blocks_per_seq,
        )
    }

    fn resize_state(&mut self, request_id: u64, target_capacity: usize) -> PyResult<bool> {
        let target_capacity = usize::min(target_capacity, self.max_seq_len);
        let required_blocks = self.required_blocks(target_capacity);
        let current_blocks = self.request_state_for(request_id)?.block_ids.len();
        let tokens_per_block = self.tokens_per_block;

        if required_blocks > current_blocks {
            let needed = required_blocks - current_blocks;
            if needed > self.free_block_ids.len() {
                return Ok(false);
            }
            let newly_allocated = self.free_block_ids.drain(..needed).collect::<Vec<_>>();
            self.request_state_for_mut(request_id)?
                .block_ids
                .extend(newly_allocated);
        } else if required_blocks < current_blocks {
            let released = self
                .request_state_for_mut(request_id)?
                .block_ids
                .split_off(required_blocks);
            self.free_block_ids.extend(released);
        }

        self.request_state_for_mut(request_id)?.capacity =
            usize::min(target_capacity, required_blocks * tokens_per_block);
        Ok(true)
    }
}

#[pymethods]
impl TrtllmStateManager {
    #[new]
    #[pyo3(signature = (tokens_per_block, max_seq_len, num_blocks, max_blocks_per_seq, max_num_sequences=32, max_beam_width=1, device_id=0, world_size=1, tp_size=1, tp_rank=0, pp_size=1, pp_rank=0, kv_factor=2, cache_mode="standard"))]
    fn new(
        tokens_per_block: usize,
        max_seq_len: usize,
        num_blocks: usize,
        max_blocks_per_seq: usize,
        max_num_sequences: usize,
        max_beam_width: usize,
        device_id: usize,
        world_size: usize,
        tp_size: usize,
        tp_rank: usize,
        pp_size: usize,
        pp_rank: usize,
        kv_factor: usize,
        cache_mode: &str,
    ) -> PyResult<Self> {
        if tokens_per_block == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tokens_per_block must be greater than 0",
            ));
        }
        if world_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "world_size must be greater than 0",
            ));
        }
        if tp_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tp_size must be greater than 0",
            ));
        }
        if pp_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pp_size must be greater than 0",
            ));
        }
        if tp_rank >= tp_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tp_rank must be within [0, tp_size)",
            ));
        }
        if pp_rank >= pp_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pp_rank must be within [0, pp_size)",
            ));
        }
        if world_size != tp_size * pp_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "world_size must equal tp_size * pp_size",
            ));
        }
        if !(kv_factor == 1 || kv_factor == 2) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "kv_factor must be 1 or 2",
            ));
        }
        if !(cache_mode == "standard" || cache_mode == "mla") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported cache_mode: {cache_mode}"
            )));
        }
        Ok(Self {
            tokens_per_block,
            max_seq_len,
            num_blocks,
            max_blocks_per_seq,
            max_num_sequences,
            device_id,
            world_size,
            tp_size,
            tp_rank,
            pp_size,
            pp_rank,
            kv_factor,
            cache_mode: cache_mode.to_string(),
            _max_beam_width: max_beam_width,
            request_state: HashMap::new(),
            free_block_ids: (0..num_blocks).collect(),
            request_slots: HashMap::new(),
            free_slots: (0..(max_num_sequences + 1)).collect(),
        })
    }

    fn is_request_active(&self, request_id: u64) -> bool {
        self.request_state
            .get(&request_id)
            .map(|state| state.active)
            .unwrap_or(false)
    }

    fn prepare_context(&mut self, request_id: u64, is_first_context_chunk: bool) -> PyResult<bool> {
        if is_first_context_chunk {
            self.request_state.entry(request_id).or_insert(RequestState {
                active: true,
                ..Default::default()
            });
        } else {
            self.request_state_for(request_id)?;
        }
        self.slot_for(request_id)?;
        self.request_state_for_mut(request_id)?.active = true;
        Ok(true)
    }

    fn resize_context(
        &mut self,
        request_id: u64,
        context_current_position: usize,
        num_tokens: usize,
        num_extra_kv_tokens: usize,
        is_first_context_chunk: bool,
    ) -> PyResult<bool> {
        let current_capacity = self.request_state_for(request_id)?.capacity;
        let target = usize::max(
            current_capacity,
            context_current_position + num_tokens + num_extra_kv_tokens,
        );
        let resized = self.resize_state(request_id, target)?;
        if !resized && is_first_context_chunk {
            self.request_state_for_mut(request_id)?.active = false;
        }
        Ok(resized)
    }

    fn try_allocate_generation(&mut self, request_id: u64, draft_token_len: usize) -> PyResult<bool> {
        let Some(state) = self.request_state.get_mut(&request_id) else {
            return Ok(false);
        };
        state.active = true;
        let target = state.capacity + 1 + draft_token_len;
        self.resize_state(request_id, target)
    }

    fn suspend_request(&mut self, request_id: u64) -> PyResult<()> {
        self.request_state_for_mut(request_id)?.active = false;
        Ok(())
    }

    fn update_context(
        &mut self,
        request_id: u64,
        context_current_position: usize,
        num_extra_kv_tokens: usize,
    ) -> PyResult<bool> {
        let Some(state) = self.request_state.get(&request_id) else {
            return Ok(true);
        };
        if !state.active {
            return Ok(true);
        }
        if state.capacity < context_current_position {
            let resized =
                self.resize_state(request_id, context_current_position + num_extra_kv_tokens)?;
            if !resized {
                return Ok(false);
            }
        }
        let state = self.request_state_for_mut(request_id)?;
        state.history_length = context_current_position;
        state.committed_tokens = state.history_length;
        Ok(true)
    }

    fn update_generation(
        &mut self,
        request_id: u64,
        max_beam_num_tokens: usize,
        rewind_len: usize,
    ) -> PyResult<bool> {
        let Some(state) = self.request_state.get(&request_id) else {
            return Ok(true);
        };
        if !state.active {
            return Ok(true);
        }
        let history_length = max_beam_num_tokens.saturating_sub(1);
        let target = usize::max(state.capacity.saturating_sub(rewind_len), history_length);
        let resized = self.resize_state(request_id, target)?;
        if !resized {
            return Ok(false);
        }
        let state = self.request_state_for_mut(request_id)?;
        state.history_length = history_length;
        state.committed_tokens = state.history_length;
        Ok(true)
    }

    fn free_resources(&mut self, request_id: u64) -> Option<usize> {
        let state = self.request_state.remove(&request_id)?;
        self.free_block_ids.extend(state.block_ids);
        let slot = self.request_slots.remove(&request_id);
        if let Some(slot_idx) = slot {
            self.free_slots.push(slot_idx);
            self.free_slots.sort_unstable();
        }
        slot
    }

    fn get_cache_indices(&self, request_id: u64) -> PyResult<Vec<usize>> {
        Ok(self.request_state_for(request_id)?.block_ids.clone())
    }

    fn get_batch_cache_indices(&self, request_ids: Vec<u64>) -> PyResult<Vec<Vec<usize>>> {
        request_ids
            .into_iter()
            .map(|request_id| self.get_cache_indices(request_id))
            .collect()
    }

    fn get_padded_block_row(&mut self, request_id: u64, bad_page_index: i64) -> PyResult<Vec<i64>> {
        self.slot_for(request_id)?;
        let mut padded = self
            .request_state_for(request_id)?
            .block_ids
            .iter()
            .take(self.max_blocks_per_seq)
            .map(|block_id| *block_id as i64)
            .collect::<Vec<_>>();
        padded.resize(self.max_blocks_per_seq, bad_page_index);
        Ok(padded)
    }

    fn get_slot(&mut self, request_id: u64) -> PyResult<usize> {
        self.slot_for(request_id)
    }

    fn get_num_free_blocks(&self) -> usize {
        self.free_block_ids.len()
    }

    fn get_num_available_tokens(
        &self,
        token_num_upper_bound: usize,
        max_num_draft_tokens: usize,
        num_extra_kv_tokens: usize,
    ) -> usize {
        usize::min(
            token_num_upper_bound,
            self.free_block_ids.len() * self.tokens_per_block
                - usize::min(
                    self.free_block_ids.len() * self.tokens_per_block,
                    num_extra_kv_tokens + max_num_draft_tokens,
                ),
        )
    }

    fn get_num_kv_blocks(&self) -> usize {
        self.num_blocks
    }

    fn get_device_id(&self) -> usize {
        self.device_id
    }

    fn get_world_size(&self) -> usize {
        self.world_size
    }

    fn get_tp_size(&self) -> usize {
        self.tp_size
    }

    fn get_tp_rank(&self) -> usize {
        self.tp_rank
    }

    fn get_pp_size(&self) -> usize {
        self.pp_size
    }

    fn get_pp_rank(&self) -> usize {
        self.pp_rank
    }

    fn get_kv_factor(&self) -> usize {
        self.kv_factor
    }

    fn get_cache_mode(&self) -> String {
        self.cache_mode.clone()
    }

    fn add_dummy_request(&mut self, request_id: u64, target_capacity: usize) -> PyResult<bool> {
        self.request_state.entry(request_id).or_insert(RequestState {
            active: true,
            ..Default::default()
        });
        self.slot_for(request_id)?;
        self.request_state_for_mut(request_id)?.active = true;
        self.resize_state(request_id, target_capacity)
    }

    fn shutdown(&mut self) {
        self.request_state.clear();
        self.free_block_ids = (0..self.num_blocks).collect();
        self.request_slots.clear();
        self.free_slots = (0..(self.max_num_sequences + 1)).collect();
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
    m.add_class::<TrtllmStateManager>()?;
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
