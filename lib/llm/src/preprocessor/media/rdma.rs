// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use flate2::Compression;
use flate2::write::ZlibEncoder;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::Arc;

use crate::block_manager::storage::{
    StorageError, SystemStorage, nixl::NixlRegisterableStorage, nixl::NixlStorage,
};
use nixl_sys::Agent as NixlAgent;

// Decoded media data (image RGB, video frames pixels, ...)
#[derive(Debug)]
pub struct DecodedMediaData {
    pub(crate) data: SystemStorage,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: String,
}

// Decoded media data NIXL descriptor (sent to the next step in the pipeline / NATS)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    // b64 agent metadata
    pub(crate) nixl_metadata: String,
    // tensor descriptor
    pub(crate) nixl_descriptor: NixlStorage,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: String,
    // reference to the actual data, kept alive while the rdma descriptor is alive
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<Arc<SystemStorage>>,
}

impl DecodedMediaData {
    pub fn into_rdma_descriptor(self, nixl_agent: &NixlAgent) -> Result<RdmaMediaDataDescriptor> {
        // get NIXL metadata and descriptor
        let mut source_storage = self.data;
        source_storage.nixl_register(nixl_agent, None)?;
        let nixl_descriptor = unsafe { source_storage.as_nixl_descriptor() }
            .ok_or_else(|| anyhow::anyhow!("Cannot convert storage to NIXL descriptor"))?;

        // TODO: cache this if this is constant across the worker lifetime?
        let nixl_local_md = nixl_agent.get_local_md()?;
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
        encoder.write_all(&nixl_local_md)?;
        let nixl_local_md = encoder.finish()?;
        let nixl_metadata = format!("b64:{}", general_purpose::STANDARD.encode(&nixl_local_md));

        Ok(RdmaMediaDataDescriptor {
            nixl_metadata,
            nixl_descriptor,
            shape: self.shape,
            dtype: self.dtype,
            // do not drop / free the storage yet
            source_storage: Some(Arc::new(source_storage)),
        })
    }
}

// convert Array{N}<u8> to DecodedMediaData
// TODO: Array1<f32> for audio
impl<D: Dimension> TryFrom<ArrayBase<OwnedRepr<u8>, D>> for DecodedMediaData {
    type Error = StorageError;

    fn try_from(array: ArrayBase<OwnedRepr<u8>, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();
        let (data, _) = array.into_raw_vec_and_offset();
        Ok(Self {
            data: SystemStorage::try_from(data)?,
            shape,
            dtype: "uint8".to_string(),
        })
    }
}
