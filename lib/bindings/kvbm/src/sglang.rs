// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed local KVBM data plane used by SGLang's direct cache linker.

use crate::{get_current_tokio_handle, to_pyerr};
use anyhow::{Context, Result, anyhow, bail};
use dynamo_memory::nixl::NixlDescriptor;
use dynamo_memory::{ExternalPinnedStorage, MemoryDescriptor, StorageKind, TensorDescriptor};
use dynamo_tokens::{
    PositionalLineageHash, compute_block_hash_for_tokens, compute_salt_hash_from_bytes,
};
use kvbm_logical::blocks::BlockDuplicationPolicy;
use kvbm_logical::{BlockManager, BlockRegistry, ImmutableBlock, MutableBlock};
use kvbm_physical::TransferManager;
use kvbm_physical::layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder};
use kvbm_physical::transfer::LayoutHandle;
use kvbm_physical::transfer::TransferOptions;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

const CODEC_VERSION: u16 = 1;
const COMPONENT_FULL_KV: u8 = 0;
const KEY_LEN: usize = 2 + 32 + 16 + 1;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct KvbmPageKeyV1 {
    manager_namespace: [u8; 32],
    sequence_hash: PositionalLineageHash,
}

impl KvbmPageKeyV1 {
    fn encode(&self) -> [u8; KEY_LEN] {
        let mut encoded = [0u8; KEY_LEN];
        encoded[..2].copy_from_slice(&CODEC_VERSION.to_be_bytes());
        encoded[2..34].copy_from_slice(&self.manager_namespace);
        encoded[34..50].copy_from_slice(&self.sequence_hash.to_be_bytes());
        encoded[50] = COMPONENT_FULL_KV;
        encoded
    }

    fn decode(encoded: &[u8], expected_namespace: &[u8; 32]) -> Result<Self> {
        if encoded.len() != KEY_LEN {
            bail!(
                "dynamo-plh-v1 key has {} bytes; expected {KEY_LEN}",
                encoded.len()
            );
        }
        let version = u16::from_be_bytes(encoded[..2].try_into().unwrap());
        if version != CODEC_VERSION {
            bail!("unsupported KVBM page-key codec version {version}");
        }
        let namespace: [u8; 32] = encoded[2..34].try_into().unwrap();
        if &namespace != expected_namespace {
            bail!("KVBM page key belongs to a different manager namespace");
        }
        if encoded[50] != COMPONENT_FULL_KV {
            bail!("KVBM V1 only supports the FullKv component");
        }
        let plh_bytes: [u8; 16] = encoded[34..50].try_into().unwrap();
        let sequence_hash = PositionalLineageHash::from_be_bytes(plh_bytes);
        let mode = sequence_hash.mode();
        if mode > 2 {
            bail!("dynamo-plh-v1 contains an invalid positional lineage hash");
        }
        let position = sequence_hash.position();
        let canonical_mode = if position < (1 << 8) {
            0
        } else if position < (1 << 16) {
            1
        } else {
            2
        };
        if mode != canonical_mode || (position == 0 && sequence_hash.parent_hash_fragment() != 0) {
            bail!("dynamo-plh-v1 contains an invalid positional lineage hash");
        }
        Ok(Self {
            manager_namespace: namespace,
            sequence_hash,
        })
    }
}

fn domain_payload(cache_salt: Option<&str>, extra_key: Option<&str>) -> Vec<u8> {
    fn append_field(output: &mut Vec<u8>, value: Option<&str>) {
        match value {
            None => output.push(0),
            Some(value) => {
                output.push(1);
                output.extend_from_slice(&(value.len() as u64).to_be_bytes());
                output.extend_from_slice(value.as_bytes());
            }
        }
    }

    let mut output = b"sglang-dynamo-plh-v1\0".to_vec();
    append_field(&mut output, cache_salt);
    append_field(&mut output, extra_key);
    output
}

fn decode_key_sequence(encoded: &[Vec<u8>], namespace: &[u8; 32]) -> Result<Vec<KvbmPageKeyV1>> {
    let keys: Vec<KvbmPageKeyV1> = encoded
        .iter()
        .map(|key| KvbmPageKeyV1::decode(key, namespace))
        .collect::<Result<_>>()?;
    for edge in keys.windows(2) {
        let parent = &edge[0].sequence_hash;
        let child = &edge[1].sequence_hash;
        if child.position() != parent.position() + 1
            || child.parent_hash_fragment()
                != parent.parent_fragment_for_child_position(child.position())
        {
            bail!("dynamo-plh-v1 keys do not form one contiguous lineage");
        }
    }
    Ok(keys)
}

fn validate_device_blocks(blocks: &[usize], num_device_blocks: usize) -> Result<()> {
    if blocks.iter().any(|block| *block >= num_device_blocks) {
        bail!("device block falls outside the registered G1 layout");
    }
    if blocks.iter().copied().collect::<HashSet<_>>().len() != blocks.len() {
        bail!("a KVBM operation cannot reuse a G1 device block");
    }
    Ok(())
}

#[pyclass]
struct DynamoPlhKeyCodec {
    namespace: [u8; 32],
}

#[pymethods]
impl DynamoPlhKeyCodec {
    #[new]
    fn new(manager_namespace: &[u8]) -> PyResult<Self> {
        let namespace = manager_namespace.try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "manager_namespace must contain exactly 32 bytes",
            )
        })?;
        Ok(Self { namespace })
    }

    #[getter]
    fn codec_id(&self) -> &'static str {
        "dynamo-plh-v1"
    }

    #[pyo3(signature = (parent_key, page_tokens, page_size, cache_salt=None, extra_key=None))]
    fn extend_pages(
        &self,
        py: Python<'_>,
        parent_key: Option<&[u8]>,
        page_tokens: Vec<u32>,
        page_size: usize,
        cache_salt: Option<&str>,
        extra_key: Option<&str>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        if page_size == 0 || page_tokens.is_empty() || !page_tokens.len().is_multiple_of(page_size)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "page_tokens must contain a whole, non-empty page sequence",
            ));
        }
        let mut parent = parent_key
            .map(|encoded| KvbmPageKeyV1::decode(encoded, &self.namespace))
            .transpose()
            .map_err(to_pyerr)?;
        let page_count = page_tokens.len() / page_size;
        let first_position = parent
            .as_ref()
            .map_or(0, |key| key.sequence_hash.position() + 1);
        if first_position
            .checked_add(page_count as u64)
            .is_none_or(|end| end > (1 << 24))
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dynamo-plh-v1 page position exceeds its 24-bit limit",
            ));
        }
        let salt_hash = compute_salt_hash_from_bytes(&domain_payload(cache_salt, extra_key));
        let mut output = Vec::with_capacity(page_tokens.len() / page_size);
        for tokens in page_tokens.chunks_exact(page_size) {
            let block_hash = compute_block_hash_for_tokens(tokens, salt_hash);
            let sequence_hash = match &parent {
                Some(parent) => parent.sequence_hash.extend(block_hash),
                None => PositionalLineageHash::root(block_hash),
            };
            let key = KvbmPageKeyV1 {
                manager_namespace: self.namespace,
                sequence_hash,
            };
            output.push(PyBytes::new(py, &key.encode()).unbind());
            parent = Some(key);
        }
        Ok(output)
    }
}

struct SglangTensor {
    _owner: Py<PyAny>,
    addr: usize,
    size: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
    element_size: usize,
    device_id: u32,
}

impl SglangTensor {
    fn from_python(py: Python<'_>, tensor: Py<PyAny>, device_id: u32) -> Result<Self> {
        let bound = tensor.bind(py);
        let device = bound.getattr("device")?;
        let device_type: String = device.getattr("type")?.extract()?;
        let actual_device: u32 = device
            .getattr("index")?
            .extract::<Option<u32>>()?
            .unwrap_or(0);
        if device_type != "cuda" || actual_device != device_id {
            bail!(
                "all SGLang KV tensors must be on cuda:{device_id}, got {device_type}:{actual_device}"
            );
        }
        if !bound.call_method0("is_contiguous")?.extract::<bool>()? {
            bail!("SGLang KV tensors must be contiguous");
        }
        Ok(Self {
            addr: bound.call_method0("data_ptr")?.extract()?,
            size: bound.getattr("nbytes")?.extract()?,
            shape: bound.getattr("shape")?.extract()?,
            stride: bound.call_method0("stride")?.extract()?,
            element_size: bound.call_method0("element_size")?.extract()?,
            device_id,
            _owner: tensor,
        })
    }
}

impl fmt::Debug for SglangTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SglangTensor")
            .field("addr", &format_args!("{:#x}", self.addr))
            .field("size", &self.size)
            .field("shape", &self.shape)
            .field("stride", &self.stride)
            .field("element_size", &self.element_size)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

impl MemoryDescriptor for SglangTensor {
    fn addr(&self) -> usize {
        self.addr
    }
    fn size(&self) -> usize {
        self.size
    }
    fn storage_kind(&self) -> StorageKind {
        StorageKind::Device(self.device_id)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl TensorDescriptor for SglangTensor {
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

#[derive(Clone, Debug)]
struct SglangG2;

struct LookupTicketState {
    keys: Vec<KvbmPageKeyV1>,
    blocks: Vec<ImmutableBlock<SglangG2>>,
}

#[derive(Clone)]
struct CompletionState {
    operation_id: u128,
    generation: u64,
    kind: &'static str,
    success: bool,
    error: Option<String>,
}

struct PendingState {
    count: Mutex<usize>,
    changed: Condvar,
}

impl PendingState {
    fn new() -> Self {
        Self {
            count: Mutex::new(0),
            changed: Condvar::new(),
        }
    }

    fn begin(&self) {
        *self.count.lock().unwrap() += 1;
    }

    fn finish(&self) {
        let mut count = self.count.lock().unwrap();
        *count -= 1;
        self.changed.notify_all();
    }

    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut count = self.count.lock().unwrap();
        while *count != 0 {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (next, status) = self.changed.wait_timeout(count, remaining).unwrap();
            count = next;
            if status.timed_out() && *count != 0 {
                return false;
            }
        }
        true
    }

    fn get(&self) -> usize {
        *self.count.lock().unwrap()
    }
}

struct StoreInner {
    namespace: [u8; 32],
    page_size: usize,
    num_device_blocks: usize,
    logical: Arc<BlockManager<SglangG2>>,
    transfers: TransferManager,
    g1: LayoutHandle,
    g2: LayoutHandle,
    generation: AtomicU64,
    closing: AtomicBool,
    tickets: Mutex<HashMap<u128, LookupTicketState>>,
    completions: Mutex<VecDeque<CompletionState>>,
    pending: PendingState,
    // Keep the external mapping/attachment alive until the final layout or
    // transfer task releases this shared state, including timeout paths.
    _host_region_guard: Py<PyAny>,
}

impl StoreInner {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closing.load(Ordering::Acquire) {
            bail!("SGLang local KV store is closing");
        }
        Ok(())
    }

    fn decode_keys(&self, encoded: &[Vec<u8>]) -> Result<Vec<KvbmPageKeyV1>> {
        decode_key_sequence(encoded, &self.namespace)
    }

    fn push_completion(&self, completion: CompletionState) {
        self.completions.lock().unwrap().push_back(completion);
    }
}

#[pyclass]
#[derive(Clone)]
struct SglangLookupTicket {
    #[pyo3(get)]
    ticket_id: u128,
    #[pyo3(get)]
    generation: u64,
    #[pyo3(get)]
    hit_pages: usize,
}

#[pyclass]
#[derive(Clone)]
struct SglangOperation {
    #[pyo3(get)]
    operation_id: u128,
    #[pyo3(get)]
    generation: u64,
    #[pyo3(get)]
    kind: &'static str,
}

#[pyclass]
#[derive(Clone)]
struct SglangCompletion {
    #[pyo3(get)]
    operation_id: u128,
    #[pyo3(get)]
    generation: u64,
    #[pyo3(get)]
    kind: &'static str,
    #[pyo3(get)]
    success: bool,
    #[pyo3(get)]
    error: Option<String>,
}

#[pyclass]
struct SglangLocalKvStore {
    inner: Option<Arc<StoreInner>>,
}

impl SglangLocalKvStore {
    fn inner(&self) -> PyResult<Arc<StoreInner>> {
        self.inner
            .as_ref()
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("KV store is closed"))
    }

    fn wait_for_drain(inner: &StoreInner, timeout_ms: u64) -> PyResult<()> {
        if inner.pending.wait(Duration::from_millis(timeout_ms)) {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyTimeoutError::new_err(format!(
                "timed out with {} KVBM operations still pending",
                inner.pending.get()
            )))
        }
    }
}

#[pymethods]
impl SglangLocalKvStore {
    #[new]
    #[pyo3(signature = (page_size, num_device_blocks, tensors, host_ptr, host_nbytes, manager_namespace, host_region_guard, device_id=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        page_size: usize,
        num_device_blocks: usize,
        tensors: Vec<Py<PyAny>>,
        host_ptr: usize,
        host_nbytes: usize,
        manager_namespace: &[u8],
        host_region_guard: Py<PyAny>,
        device_id: u32,
    ) -> PyResult<Self> {
        let namespace: [u8; 32] = manager_namespace
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("namespace must be 32 bytes"))?;
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "page_size must be a non-zero power of two",
            ));
        }
        if num_device_blocks == 0 || tensors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "device layout requires blocks and tensors",
            ));
        }
        if !tensors.len().is_multiple_of(2) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "device layout requires one K/V tensor pair per layer",
            ));
        }

        let tensors = tensors
            .into_iter()
            .map(|tensor| SglangTensor::from_python(py, tensor, device_id))
            .collect::<Result<Vec<_>>>()
            .map_err(to_pyerr)?;
        let element_size = tensors[0].element_size;
        let tensor_bytes = tensors[0].size;
        let tensor_shape = tensors[0].shape.as_slice();
        let tensor_stride = tensors[0].stride.as_slice();
        if tensors.iter().any(|tensor| {
            tensor.size != tensor_bytes
                || tensor.element_size != element_size
                || tensor.shape.as_slice() != tensor_shape
                || tensor.stride.as_slice() != tensor_stride
        }) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "KVBM V1 requires homogeneous K/V tensor geometry",
            ));
        }
        let denominator = num_device_blocks
            .checked_mul(page_size)
            .and_then(|value| value.checked_mul(element_size))
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("layout size overflow"))?;
        if denominator == 0 || !tensor_bytes.is_multiple_of(denominator) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tensor bytes are not divisible by page geometry",
            ));
        }
        let inner_dim = tensor_bytes / denominator;
        let num_layers = tensors.len();
        let config = LayoutConfig::builder()
            .num_blocks(num_device_blocks)
            .num_layers(num_layers)
            .outer_dim(1usize)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(element_size)
            .build()
            .map_err(to_pyerr)?;
        let bytes_per_block = config.bytes_per_block();
        let host_blocks = host_nbytes / bytes_per_block;
        if host_blocks == 0 || !host_nbytes.is_multiple_of(bytes_per_block) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "host region has {host_nbytes} bytes; it must contain whole \
                 {bytes_per_block}-byte KV blocks"
            )));
        }

        let transfers = TransferManager::builder()
            .cuda_device_id(device_id as usize)
            .build()
            .map_err(to_pyerr)?;
        let tensor_regions: Vec<Arc<dyn TensorDescriptor>> = tensors
            .into_iter()
            .map(|tensor| Arc::new(tensor) as Arc<dyn TensorDescriptor>)
            .collect();
        let g1_layout = PhysicalLayoutBuilder::new(transfers.nixl_agent().clone())
            .with_config(config.clone())
            .layer_separate(BlockDimension::BlockIsFirstDim)
            .with_external_device_regions(tensor_regions)
            .and_then(|builder| builder.build())
            .map_err(to_pyerr)?;

        let host_config = LayoutConfig::builder()
            .num_blocks(host_blocks)
            .num_layers(num_layers)
            .outer_dim(1usize)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(element_size)
            .build()
            .map_err(to_pyerr)?;
        let host_storage =
            unsafe { ExternalPinnedStorage::new(host_ptr as *mut u8, host_nbytes, device_id) }
                .map_err(to_pyerr)?;
        let g2_layout = PhysicalLayoutBuilder::new(transfers.nixl_agent().clone())
            .with_config(host_config)
            .fully_contiguous()
            .with_memory_regions(vec![host_storage])
            .and_then(|builder| builder.build())
            .map_err(to_pyerr)?;
        let g1 = transfers.register_layout(g1_layout).map_err(to_pyerr)?;
        let g2 = transfers.register_layout(g2_layout).map_err(to_pyerr)?;

        let logical = BlockManager::<SglangG2>::builder()
            .block_count(host_blocks)
            .block_size(page_size)
            .registry(BlockRegistry::new())
            .duplication_policy(BlockDuplicationPolicy::Reject)
            .with_lru_backend()
            .build()
            .map_err(to_pyerr)?;
        let inner = Arc::new(StoreInner {
            namespace,
            page_size,
            num_device_blocks,
            logical: Arc::new(logical),
            transfers,
            g1,
            g2,
            generation: AtomicU64::new(0),
            closing: AtomicBool::new(false),
            tickets: Mutex::new(HashMap::new()),
            completions: Mutex::new(VecDeque::new()),
            pending: PendingState::new(),
            _host_region_guard: host_region_guard,
        });
        Ok(Self { inner: Some(inner) })
    }

    fn lookup_prefix(&self, encoded_keys: Vec<Vec<u8>>) -> PyResult<SglangLookupTicket> {
        let inner = self.inner()?;
        inner.ensure_open().map_err(to_pyerr)?;
        let keys = inner.decode_keys(&encoded_keys).map_err(to_pyerr)?;
        let hashes: Vec<_> = keys.iter().map(|key| key.sequence_hash).collect();
        let blocks = inner.logical.match_blocks(&hashes);
        let keys = keys.into_iter().take(blocks.len()).collect();
        let ticket_id = Uuid::new_v4().as_u128();
        let hit_pages = blocks.len();
        inner
            .tickets
            .lock()
            .unwrap()
            .insert(ticket_id, LookupTicketState { keys, blocks });
        Ok(SglangLookupTicket {
            ticket_id,
            generation: inner.generation(),
            hit_pages,
        })
    }

    fn cancel_lookup(&self, ticket: &SglangLookupTicket) -> PyResult<()> {
        let inner = self.inner()?;
        if ticket.generation != inner.generation() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lookup ticket generation is stale",
            ));
        }
        inner
            .tickets
            .lock()
            .unwrap()
            .remove(&ticket.ticket_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("lookup ticket already consumed")
            })?;
        Ok(())
    }

    fn enqueue_store(
        &self,
        encoded_keys: Vec<Vec<u8>>,
        source_g1_blocks: Vec<usize>,
    ) -> PyResult<SglangOperation> {
        let inner = self.inner()?;
        inner.ensure_open().map_err(to_pyerr)?;
        let keys = inner.decode_keys(&encoded_keys).map_err(to_pyerr)?;
        if keys.len() != source_g1_blocks.len() || keys.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "store keys and source blocks must have the same non-zero length",
            ));
        }
        validate_device_blocks(&source_g1_blocks, inner.num_device_blocks)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let generation = inner.generation();
        let operation_id = Uuid::new_v4().as_u128();
        let hashes: Vec<_> = keys.iter().map(|key| key.sequence_hash).collect();
        let existing = inner.logical.match_blocks_scattered(&hashes);
        let mut seen = HashSet::new();
        let mut missing = Vec::new();
        for ((key, source), hit) in keys.into_iter().zip(source_g1_blocks).zip(existing) {
            if hit.is_none() && seen.insert(key.sequence_hash) {
                missing.push((key, source));
            }
        }
        drop(seen);

        let operation = SglangOperation {
            operation_id,
            generation,
            kind: "store",
        };
        if missing.is_empty() {
            inner.push_completion(CompletionState {
                operation_id,
                generation,
                kind: "store",
                success: true,
                error: None,
            });
            return Ok(operation);
        }

        let Some(mutable) = inner.logical.allocate_blocks(missing.len()) else {
            // Store admission is rank-local. Publish a normal failed completion
            // instead of returning synchronously so SGLang can reduce success
            // across every TP/CP rank without diverging pending queues.
            inner.push_completion(CompletionState {
                operation_id,
                generation,
                kind: "store",
                success: false,
                error: Some("KVBM G2 is full or all candidate blocks are pinned".into()),
            });
            return Ok(operation);
        };
        let source: Vec<_> = missing.iter().map(|(_, source)| *source).collect();
        let destination: Vec<_> = mutable.iter().map(MutableBlock::block_id).collect();
        let notification = inner
            .transfers
            .execute_transfer(
                inner.g1,
                &source,
                inner.g2,
                &destination,
                TransferOptions::default(),
            )
            .map_err(to_pyerr)?;
        let store_keys: Vec<_> = missing.into_iter().map(|(key, _)| key).collect();
        inner.pending.begin();
        let task_inner = inner.clone();
        get_current_tokio_handle().spawn(async move {
            let transfer_result = notification.await;
            let result = match transfer_result {
                Ok(()) => stage_and_register(&task_inner, mutable, store_keys),
                Err(error) => Err(error.context("G1 to G2 transfer failed")),
            };
            task_inner.push_completion(CompletionState {
                operation_id,
                generation,
                kind: "store",
                success: result.is_ok(),
                error: result.err().map(|error| format!("{error:#}")),
            });
            task_inner.pending.finish();
        });
        Ok(operation)
    }

    fn enqueue_load(
        &self,
        ticket: &SglangLookupTicket,
        encoded_keys: Vec<Vec<u8>>,
        target_g1_blocks: Vec<usize>,
    ) -> PyResult<SglangOperation> {
        let inner = self.inner()?;
        inner.ensure_open().map_err(to_pyerr)?;
        if ticket.generation != inner.generation() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "lookup ticket generation is stale",
            ));
        }
        let requested = inner.decode_keys(&encoded_keys).map_err(to_pyerr)?;
        if requested.len() != target_g1_blocks.len() || requested.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "load keys and target blocks must have the same non-zero length",
            ));
        }
        validate_device_blocks(&target_g1_blocks, inner.num_device_blocks)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let mut state = inner
            .tickets
            .lock()
            .unwrap()
            .remove(&ticket.ticket_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("lookup ticket already consumed")
            })?;
        if requested.len() > state.keys.len()
            || !state
                .keys
                .iter()
                .zip(&requested)
                .all(|(ticket_key, requested_key)| ticket_key == requested_key)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "load keys must be a prefix of their lookup ticket",
            ));
        }
        state.keys.truncate(requested.len());
        let tail_blocks = state.blocks.split_off(requested.len());
        drop(tail_blocks);
        let source: Vec<_> = state.blocks.iter().map(ImmutableBlock::block_id).collect();
        let generation = inner.generation();
        let operation_id = Uuid::new_v4().as_u128();
        let notification = inner
            .transfers
            .execute_transfer(
                inner.g2,
                &source,
                inner.g1,
                &target_g1_blocks,
                TransferOptions::default(),
            )
            .map_err(to_pyerr)?;
        inner.pending.begin();
        let task_inner = inner.clone();
        get_current_tokio_handle().spawn(async move {
            let result = notification.await.context("G2 to G1 transfer failed");
            // Keep the immutable ticket blocks alive through the real completion.
            drop(state);
            task_inner.push_completion(CompletionState {
                operation_id,
                generation,
                kind: "load",
                success: result.is_ok(),
                error: result.err().map(|error| format!("{error:#}")),
            });
            task_inner.pending.finish();
        });
        Ok(SglangOperation {
            operation_id,
            generation,
            kind: "load",
        })
    }

    fn poll_completions(&self) -> PyResult<Vec<SglangCompletion>> {
        let inner = self.inner()?;
        let mut queue = inner.completions.lock().unwrap();
        Ok(queue
            .drain(..)
            .map(|completion| SglangCompletion {
                operation_id: completion.operation_id,
                generation: completion.generation,
                kind: completion.kind,
                success: completion.success,
                error: completion.error,
            })
            .collect())
    }

    #[pyo3(signature = (timeout_ms=30_000))]
    fn drain(&self, timeout_ms: u64) -> PyResult<()> {
        let inner = self.inner()?;
        Self::wait_for_drain(&inner, timeout_ms)
    }

    #[pyo3(signature = (timeout_ms=30_000))]
    fn reset(&self, timeout_ms: u64) -> PyResult<()> {
        let inner = self.inner()?;
        Self::wait_for_drain(&inner, timeout_ms)?;
        inner.tickets.lock().unwrap().clear();
        inner.completions.lock().unwrap().clear();
        inner.logical.reset_inactive_pool().map_err(to_pyerr)?;
        inner.generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    #[pyo3(signature = (timeout_ms=30_000))]
    fn close(&mut self, timeout_ms: u64) -> PyResult<()> {
        let Some(inner) = self.inner.as_ref().cloned() else {
            return Ok(());
        };
        inner.closing.store(true, Ordering::Release);
        Self::wait_for_drain(&inner, timeout_ms)?;
        inner.tickets.lock().unwrap().clear();
        self.inner.take();
        drop(inner);
        Ok(())
    }

    fn pending_counts(&self) -> PyResult<(usize, usize, usize)> {
        let inner = self.inner()?;
        Ok((
            inner.tickets.lock().unwrap().len(),
            inner.pending.get(),
            inner.completions.lock().unwrap().len(),
        ))
    }
}

impl Drop for SglangLocalKvStore {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            inner.closing.store(true, Ordering::Release);
            if !inner.pending.wait(Duration::from_secs(30)) {
                tracing::error!(
                    pending = inner.pending.get(),
                    "dropping SGLang KVBM store with operations still pending"
                );
            }
        }
    }
}

fn stage_and_register(
    inner: &StoreInner,
    mutable: Vec<MutableBlock<SglangG2>>,
    keys: Vec<KvbmPageKeyV1>,
) -> Result<()> {
    let complete = mutable
        .into_iter()
        .zip(keys)
        .map(|(block, key)| {
            block
                .stage(key.sequence_hash, inner.page_size)
                .map_err(|error| anyhow!(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    drop(inner.logical.register_blocks(complete));
    Ok(())
}

pub fn add_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DynamoPlhKeyCodec>()?;
    module.add_class::<SglangLookupTicket>()?;
    module.add_class::<SglangOperation>()?;
    module.add_class::<SglangCompletion>()?;
    module.add_class::<SglangLocalKvStore>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_wire_roundtrip_and_namespace_validation() {
        let key = KvbmPageKeyV1 {
            manager_namespace: [7; 32],
            sequence_hash: PositionalLineageHash::root(42),
        };
        let encoded = key.encode();
        assert_eq!(KvbmPageKeyV1::decode(&encoded, &[7; 32]).unwrap(), key);
        assert!(KvbmPageKeyV1::decode(&encoded, &[8; 32]).is_err());

        let mut noncanonical = encoded;
        noncanonical[34] |= 0x40;
        assert!(KvbmPageKeyV1::decode(&noncanonical, &[7; 32]).is_err());

        let mut reserved_mode = encoded;
        reserved_mode[34] |= 0xc0;
        assert!(KvbmPageKeyV1::decode(&reserved_mode, &[7; 32]).is_err());
    }

    #[test]
    fn domain_encoding_distinguishes_none_empty_and_fields() {
        assert_ne!(domain_payload(None, None), domain_payload(Some(""), None));
        assert_ne!(
            domain_payload(Some("a"), Some("b")),
            domain_payload(Some("b"), Some("a"))
        );
    }

    #[test]
    fn device_block_validation_rejects_aliases_and_out_of_bounds_ids() {
        assert!(validate_device_blocks(&[0, 2], 3).is_ok());
        assert!(validate_device_blocks(&[0, 0], 3).is_err());
        assert!(validate_device_blocks(&[0, 3], 3).is_err());
    }

    #[test]
    fn key_sequence_rejects_disconnected_pages() {
        let namespace = [9; 32];
        let first = KvbmPageKeyV1 {
            manager_namespace: namespace,
            sequence_hash: PositionalLineageHash::root(1),
        };
        let disconnected = KvbmPageKeyV1 {
            manager_namespace: namespace,
            sequence_hash: PositionalLineageHash::root(2),
        };
        assert!(
            decode_key_sequence(
                &[first.encode().to_vec(), disconnected.encode().to_vec()],
                &namespace,
            )
            .is_err()
        );
    }

    #[test]
    fn key_codec_vector_covers_domain_and_parent_lineage() {
        let namespace = [0x11; 32];
        let salt =
            compute_salt_hash_from_bytes(&domain_payload(Some("tenant-a"), Some("adapter-a")));
        let first = KvbmPageKeyV1 {
            manager_namespace: namespace,
            sequence_hash: PositionalLineageHash::root(compute_block_hash_for_tokens(
                &[1, 2],
                salt,
            )),
        };
        let second = KvbmPageKeyV1 {
            manager_namespace: namespace,
            sequence_hash: first
                .sequence_hash
                .extend(compute_block_hash_for_tokens(&[3, 4], salt)),
        };
        let encoded = [first, second].map(|key| {
            key.encode()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        });
        assert_eq!(
            encoded,
            [
                "00011111111111111111111111111111111111111111111111111111111111111111001488c9a84e2f52f40000000000000000",
                "00011111111111111111111111111111111111111111111111111111111111111111004e67c2eab285ba9d6326a138bd4bd000",
            ]
        );
    }
}
