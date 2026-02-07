// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};

#[cfg(feature = "media-nixl")]
use {
    base64::{Engine as _, engine::general_purpose},
    dynamo_memory::SystemStorage,
    dynamo_memory::nixl::{self, NixlAgent, NixlDescriptor, RegisteredView},
    flate2::{Compression, write::ZlibEncoder},
    std::io::Write,
    std::sync::Arc,
};

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
    #[cfg(feature = "media-nixl")]
    pub(crate) data: SystemStorage,
    pub(crate) tensor_info: MediaTensorInfo,
}

// Decoded media data NIXL descriptor (sent to the next step in the pipeline / NATS)

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    // b64 agent metadata
    #[cfg(feature = "media-nixl")]
    pub(crate) nixl_metadata: String,
    // tensor descriptor
    #[cfg(feature = "media-nixl")]
    pub(crate) nixl_descriptor: NixlDescriptor,

    #[serde(flatten)]
    pub(crate) tensor_info: MediaTensorInfo,

    // reference to the actual data, kept alive while the rdma descriptor is alive
    #[serde(skip, default)]
    #[allow(dead_code)]
    #[cfg(feature = "media-nixl")]
    pub(crate) source_storage: Option<Arc<nixl::NixlRegistered<SystemStorage>>>,
}

impl DecodedMediaData {
    #[cfg(feature = "media-nixl")]
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

        #[cfg(feature = "media-nixl")]
        let (data_vec, _) = array.into_raw_vec_and_offset();
        #[cfg(feature = "media-nixl")]
        let mut storage = SystemStorage::new(data_vec.len())?;
        #[cfg(feature = "media-nixl")]
        unsafe {
            std::ptr::copy_nonoverlapping(data_vec.as_ptr(), storage.as_mut_ptr(), data_vec.len());
        }

        Ok(Self {
            #[cfg(feature = "media-nixl")]
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
#[cfg(feature = "media-nixl")]
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

#[cfg(feature = "media-nixl")]
pub fn get_nixl_agent() -> Result<NixlAgent> {
    let name = format!("media-loader-{}", uuid::Uuid::new_v4());
    let nixl_agent = NixlAgent::with_backends(&name, &["UCX"])?;
    Ok(nixl_agent)
}
