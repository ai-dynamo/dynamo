// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire form of media the frontend decoded and registered for RDMA transfer.
//!
//! [`MultimodalData::Decoded`](super::preprocessor::MultimodalData) carries this descriptor
//! to the worker, which reads the pixels through NIXL. Only the serialized fields are defined
//! here, as plain serde types: the frontend's media loader owns the decoder and NIXL types and
//! converts its own descriptor into this one. Field names, order and serde attributes are the
//! wire contract; the loader's tests compare the two forms byte for byte.

use std::any::Any;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Memory kind of a NIXL-registered region, by NIXL's names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixlMemType {
    Dram,
    Vram,
    Block,
    Object,
    File,
    Unknown,
}

/// The registered memory region a worker reads the decoded media from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixlRegionDescriptor {
    /// Base address of the memory region.
    pub addr: u64,
    /// Size of the memory region in bytes.
    pub size: usize,
    /// Type of memory (host, device, etc.).
    pub mem_type: NixlMemType,
    /// Device identifier (GPU index for device memory, 0 for host memory).
    pub device_id: u64,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum DecodedMediaDataType {
    UINT8,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum DecodedImageLayout {
    HWC,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DecodedImageMetadata {
    /// Container format of the encoded image, by the `image` crate's variant name.
    pub format: Option<String>,
    /// Pixel type of the decoded buffer, by the `image` crate's variant name.
    pub color_type: String,
    pub layout: DecodedImageLayout,
}

#[cfg(feature = "media-ffmpeg")]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DecodedVideoMetadata {
    pub source_fps: f64,
    pub source_duration: f64,
    pub sampled_timestamps: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DecodedMediaMetadata {
    Image(DecodedImageMetadata),
    #[cfg(feature = "media-ffmpeg")]
    Video(DecodedVideoMetadata),
}

/// Tensor metadata of the decoded media.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DecodedMediaTensorInfo {
    pub shape: Vec<usize>,
    pub dtype: DecodedMediaDataType,
    pub metadata: Option<DecodedMediaMetadata>,
}

/// Decoded media data NIXL descriptor (sent to the next step in the pipeline / NATS).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DecodedMediaDescriptor {
    /// b64 agent metadata
    pub nixl_metadata: String,
    /// tensor descriptor
    pub nixl_descriptor: NixlRegionDescriptor,

    #[serde(flatten)]
    pub tensor_info: DecodedMediaTensorInfo,

    /// Canonical xxh3-64 key for decoded media. Image identity covers shape,
    /// dtype, and RGB bytes; video identity additionally covers decoded
    /// metadata that affects the model-visible token sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,

    /// Keeps the registered buffer alive while the descriptor is alive. The frontend's
    /// media loader sets it; it is never serialized, so a descriptor read from the wire
    /// has none.
    #[serde(skip, default)]
    pub source_storage: Option<Arc<dyn Any + Send + Sync>>,
}

impl DecodedMediaDescriptor {
    /// Canonical cache/routing key serialized on the descriptor.
    pub fn content_hash_key(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }
}
