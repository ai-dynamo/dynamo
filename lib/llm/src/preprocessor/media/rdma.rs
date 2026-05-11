// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use dynamo_memory::SystemStorage;
use dynamo_memory::nixl::{self, NixlAgent, NixlDescriptor, RegisteredView};
use flate2::{Compression, write::ZlibEncoder};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::Arc;

use super::decoders::DecodedMediaMetadata;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    UINT8,
}

// Common tensor metadata shared between decoded and RDMA descriptors
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MediaTensorInfo {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DataType,
    pub(crate) metadata: Option<DecodedMediaMetadata>,
}

// Decoded media data (image RGB, video frames pixels, ...)
#[derive(Debug)]
pub struct DecodedMediaData {
    pub(crate) data: SystemStorage,
    pub(crate) tensor_info: MediaTensorInfo,
}

// Decoded media data NIXL descriptor (sent to the next step in the pipeline / NATS)

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    // b64 agent metadata
    pub(crate) nixl_metadata: String,
    // tensor descriptor
    pub(crate) nixl_descriptor: NixlDescriptor,

    #[serde(flatten)]
    pub(crate) tensor_info: MediaTensorInfo,

    // reference to the actual data, kept alive while the rdma descriptor is alive
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<Arc<nixl::NixlRegistered<SystemStorage>>>,
}

impl RdmaMediaDataDescriptor {
    /// xxh3-64 of `(shape, dtype, decoded byte payload)`. Returns `None` if
    /// the descriptor was deserialized from the wire and no longer holds
    /// local storage. Used by MM-aware KV routing to produce a
    /// content-addressed `mm_hash` so the same image reached through
    /// different (signed) URLs collides on the same routing key.
    ///
    /// `shape` + `dtype` are included in the preimage so two images that
    /// happen to share a raw byte buffer but differ in dimensions or
    /// element layout (e.g. `3x224x224` vs `224x224x3`) don't collide on
    /// the same routing hash.
    #[cfg(feature = "lightseek-mm")]
    pub(crate) fn content_hash(&self) -> Option<u64> {
        use dynamo_memory::MemoryDescriptor;
        use xxhash_rust::xxh3::Xxh3;
        let registered = self.source_storage.as_ref()?;
        let storage = registered.storage();
        let len = storage.size();
        if len == 0 {
            return None;
        }
        let mut hasher = Xxh3::new();
        // shape (rank + per-dim u64s) — fixed width so distinct shapes
        // can't alias each other after concatenation.
        hasher.update(&(self.tensor_info.shape.len() as u64).to_le_bytes());
        for &dim in &self.tensor_info.shape {
            hasher.update(&(dim as u64).to_le_bytes());
        }
        // dtype as a single discriminant byte; widen if DataType ever grows.
        let dtype_byte: u8 = match self.tensor_info.dtype {
            DataType::UINT8 => 0,
        };
        hasher.update(&[dtype_byte]);
        // raw payload via SystemStorage's Slice impl.
        // SAFETY: `storage` is borrowed for the duration of this call;
        // SystemStorage owns the malloc'd buffer and won't free or
        // relocate it while the borrow is alive. The Slice trait's
        // safety contract about no concurrent mutable access is upheld
        // because `content_hash` is only invoked on descriptors during
        // request building, before they're sent to NIXL.
        use dynamo_memory::actions::Slice;
        let bytes = unsafe { storage.as_slice().ok()? };
        hasher.update(bytes);
        Some(hasher.digest())
    }
}

impl DecodedMediaData {
    pub fn into_rdma_descriptor(self, nixl_agent: &NixlAgent) -> Result<RdmaMediaDataDescriptor> {
        let source_storage = self.data;
        let registered = nixl::register_with_nixl(source_storage, nixl_agent, None)
            .map_err(|_| anyhow::anyhow!("Failed to register storage with NIXL"))?;

        let nixl_descriptor = registered.descriptor();
        let nixl_metadata = get_nixl_metadata(nixl_agent, registered.storage())?;

        Ok(RdmaMediaDataDescriptor {
            nixl_metadata,
            nixl_descriptor,
            tensor_info: self.tensor_info,
            // Keep registered storage alive
            source_storage: Some(Arc::new(registered)),
        })
    }
}

// convert Array{N}<u8> to DecodedMediaData
// TODO: Array1<f32> for audio

impl<D: Dimension> TryFrom<ArrayBase<OwnedRepr<u8>, D>> for DecodedMediaData {
    type Error = anyhow::Error;

    fn try_from(array: ArrayBase<OwnedRepr<u8>, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();

        let (data_vec, _) = array.into_raw_vec_and_offset();
        let mut storage = SystemStorage::new(data_vec.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(data_vec.as_ptr(), storage.as_mut_ptr(), data_vec.len());
        }

        Ok(Self {
            data: storage,
            tensor_info: MediaTensorInfo {
                shape,
                dtype: DataType::UINT8,
                metadata: None,
            },
        })
    }
}

// Get NIXL metadata for a descriptor
// Returns zlib-compressed, base64-encoded metadata in format: "b64:<compressed_base64>"
// This format matches what Python nixl_connect expects for RdmaMetadata.nixl_metadata
// TODO: pre-allocate a fixed NIXL-registered RAM pool so metadata can be cached on the target?
pub fn get_nixl_metadata(agent: &NixlAgent, _storage: &SystemStorage) -> Result<String> {
    // WAR: Until https://github.com/ai-dynamo/nixl/pull/970 is merged, can't use get_local_partial_md
    let nixl_md = agent.raw_agent().get_local_md()?;
    // let mut reg_desc_list = RegDescList::new(MemType::Dram)?;
    // reg_desc_list.add_storage_desc(storage)?;
    // let nixl_partial_md = agent.raw_agent().get_local_partial_md(&reg_desc_list, None)?;

    // Compress with zlib (level 6, matching Python's default)
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(&nixl_md)?;
    let compressed = encoder.finish()?;

    let b64_encoded = general_purpose::STANDARD.encode(&compressed);
    Ok(format!("b64:{}", b64_encoded))
}

pub fn get_nixl_agent() -> Result<NixlAgent> {
    let name = format!("media-loader-{}", uuid::Uuid::new_v4());
    let nixl_agent = NixlAgent::with_backends(&name, &["UCX"])?;
    Ok(nixl_agent)
}
