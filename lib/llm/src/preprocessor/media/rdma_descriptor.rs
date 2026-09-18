// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "media-nixl")]
use std::sync::Arc;

#[cfg(feature = "media-nixl")]
use dynamo_memory::{SystemStorage, nixl};
use serde::{Deserialize, Serialize};

use super::decoded::MediaTensorInfo;

/// NIXL descriptor for decoded media sent to the next pipeline stage.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    pub(crate) nixl_metadata: String,
    pub(crate) nixl_descriptor: RdmaMemoryDescriptor,

    #[serde(flatten)]
    pub(crate) tensor_info: MediaTensorInfo,

    /// Canonical xxh3-64 key for decoded media. Image identity covers shape,
    /// dtype, and RGB bytes; video identity additionally covers decoded
    /// metadata that affects the model-visible token sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) content_hash: Option<String>,

    /// Keep the registered bytes alive while the descriptor is in use.
    #[cfg(feature = "media-nixl")]
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<Arc<nixl::NixlRegistered<SystemStorage>>>,
}

impl RdmaMediaDataDescriptor {
    /// Canonical cache/routing key serialized on the descriptor.
    pub(crate) fn content_hash_key(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }

    /// Numeric form used by MM-aware KV routing.
    #[cfg(feature = "mm-routing")]
    pub(crate) fn content_hash(&self) -> Option<u64> {
        self.content_hash_key()
            .and_then(|key| u64::from_str_radix(key, 16).ok())
    }
}

/// Wire representation of a NIXL memory region.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct RdmaMemoryDescriptor {
    pub(crate) addr: u64,
    pub(crate) size: usize,
    pub(crate) mem_type: RdmaMemoryType,
    pub(crate) device_id: u64,
}

/// Wire representation of NIXL's supported memory types.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub(crate) enum RdmaMemoryType {
    Dram,
    Vram,
    Block,
    Object,
    File,
    Unknown,
}

#[cfg(feature = "media-nixl")]
impl From<nixl::NixlDescriptor> for RdmaMemoryDescriptor {
    fn from(descriptor: nixl::NixlDescriptor) -> Self {
        let mem_type = match descriptor.mem_type {
            nixl::MemType::Dram => RdmaMemoryType::Dram,
            nixl::MemType::Vram => RdmaMemoryType::Vram,
            nixl::MemType::Block => RdmaMemoryType::Block,
            nixl::MemType::Object => RdmaMemoryType::Object,
            nixl::MemType::File => RdmaMemoryType::File,
            nixl::MemType::Unknown => RdmaMemoryType::Unknown,
        };

        Self {
            addr: descriptor.addr,
            size: descriptor.size,
            mem_type,
            device_id: descriptor.device_id,
        }
    }
}
