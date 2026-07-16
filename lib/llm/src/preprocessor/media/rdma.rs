// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
use anyhow::Context;
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
    #[cfg(feature = "mm-routing")]
    fn local_payload(&self) -> Option<&[u8]> {
        use dynamo_memory::actions::Slice;
        let registered = self.source_storage.as_ref()?;
        let storage = registered.storage();
        // SAFETY: the descriptor keeps the registered SystemStorage alive and
        // request construction does not mutate it while this borrow exists.
        unsafe { storage.as_slice().ok() }
    }

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
    #[cfg(feature = "mm-routing")]
    pub(crate) fn content_hash(&self) -> Option<u64> {
        use xxhash_rust::xxh3::Xxh3;
        let bytes = self.local_payload()?;
        if bytes.is_empty() {
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
        hasher.update(bytes);
        Some(hasher.digest())
    }

    /// Canonical identity for one frontend-decoded video:
    /// `(modality, shape, dtype, decoded metadata, sampled RGB bytes)`.
    /// Metadata is serialized from the typed object sent to the worker, so new
    /// model-visible fields automatically participate in the identity.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_content_hash(&self) -> Result<u64> {
        let bytes = self
            .local_payload()
            .ok_or_else(|| anyhow::anyhow!("decoded video has no local payload"))?;
        hash_video_content(&self.tensor_info, bytes)
    }

    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_metadata(&self) -> Result<&super::decoders::VideoMetadata> {
        match self.tensor_info.metadata.as_ref() {
            Some(DecodedMediaMetadata::Video(metadata)) => Ok(metadata),
            Some(_) => anyhow::bail!("decoded media metadata is not video metadata"),
            None => anyhow::bail!("decoded video metadata is missing"),
        }
    }

    /// Validate the contiguous `[T, H, W, 3]` payload and return its video
    /// dimensions without copying or preprocessing pixels.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_dimensions(&self) -> Result<(usize, u32, u32)> {
        let bytes = self
            .local_payload()
            .ok_or_else(|| anyhow::anyhow!("decoded video has no local payload"))?;
        video_dimensions_from_parts(&self.tensor_info, bytes)
    }
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn hash_video_content(tensor_info: &MediaTensorInfo, bytes: &[u8]) -> Result<u64> {
    use xxhash_rust::xxh3::Xxh3;

    anyhow::ensure!(!bytes.is_empty(), "decoded video payload is empty");
    let metadata = tensor_info
        .metadata
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("decoded video metadata is missing"))?;
    let DecodedMediaMetadata::Video(metadata) = metadata else {
        anyhow::bail!("decoded media metadata is not video metadata");
    };
    let metadata_bytes = serde_json::to_vec(metadata)
        .map_err(|error| anyhow::anyhow!("failed to serialize video metadata: {error}"))?;

    let mut hasher = Xxh3::new();
    update_len_prefixed(&mut hasher, b"video");
    hasher.update(&(tensor_info.shape.len() as u64).to_le_bytes());
    for &dim in &tensor_info.shape {
        hasher.update(&(dim as u64).to_le_bytes());
    }
    let dtype_byte = match tensor_info.dtype {
        DataType::UINT8 => 0,
    };
    hasher.update(&[dtype_byte]);
    update_len_prefixed(&mut hasher, &metadata_bytes);
    update_len_prefixed(&mut hasher, bytes);
    Ok(hasher.digest())
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn video_dimensions_from_parts(
    tensor_info: &MediaTensorInfo,
    bytes: &[u8],
) -> Result<(usize, u32, u32)> {
    let [frames, height, width, channels] = tensor_info.shape.as_slice() else {
        anyhow::bail!(
            "decoded video shape must be [T, H, W, C], got {:?}",
            tensor_info.shape
        );
    };
    anyhow::ensure!(*frames > 0, "decoded video has no frames");
    anyhow::ensure!(
        *height > 0 && *width > 0,
        "decoded video dimensions are zero"
    );
    anyhow::ensure!(*channels == 3, "decoded video must contain RGB frames");
    anyhow::ensure!(
        tensor_info.dtype == DataType::UINT8,
        "decoded video dtype must be uint8"
    );

    let frame_len = height
        .checked_mul(*width)
        .and_then(|value| value.checked_mul(*channels))
        .ok_or_else(|| anyhow::anyhow!("decoded video frame size overflow"))?;
    let expected_len = frames
        .checked_mul(frame_len)
        .ok_or_else(|| anyhow::anyhow!("decoded video payload size overflow"))?;
    anyhow::ensure!(
        bytes.len() == expected_len,
        "decoded video payload has {} bytes, expected {}",
        bytes.len(),
        expected_len
    );
    let width = u32::try_from(*width).context("decoded video width exceeds u32")?;
    let height = u32::try_from(*height).context("decoded video height exceeds u32")?;

    Ok((*frames, width, height))
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn update_len_prefixed(hasher: &mut xxhash_rust::xxh3::Xxh3, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
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

#[cfg(all(test, feature = "mm-routing", feature = "media-ffmpeg"))]
mod tests {
    use super::*;
    use crate::preprocessor::media::decoders::VideoMetadata;

    fn video_info(sampled_timestamps: Vec<f64>) -> MediaTensorInfo {
        MediaTensorInfo {
            shape: vec![2, 1, 2, 3],
            dtype: DataType::UINT8,
            metadata: Some(DecodedMediaMetadata::Video(VideoMetadata {
                source_fps: 24.0,
                source_duration: 10.0,
                sampled_timestamps,
            })),
        }
    }

    #[test]
    fn video_hash_covers_metadata_shape_and_rgb_bytes() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let expected = hash_video_content(&info, &bytes).unwrap();

        assert_eq!(hash_video_content(&info, &bytes).unwrap(), expected);

        let changed_metadata = video_info(vec![0.0, 5.1]);
        assert_ne!(
            hash_video_content(&changed_metadata, &bytes).unwrap(),
            expected
        );

        let mut changed_shape = info.clone();
        changed_shape.shape = vec![1, 2, 2, 3];
        assert_ne!(
            hash_video_content(&changed_shape, &bytes).unwrap(),
            expected
        );

        let mut changed_bytes = bytes;
        changed_bytes[0] = 42;
        assert_ne!(hash_video_content(&info, &changed_bytes).unwrap(), expected);
    }

    #[test]
    fn video_dimensions_validate_rgb_payload_layout() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let dimensions = video_dimensions_from_parts(&info, &bytes).unwrap();

        assert_eq!(dimensions, (2, 2, 1));

        let mut rgba = info.clone();
        rgba.shape[3] = 4;
        assert!(video_dimensions_from_parts(&rgba, &bytes).is_err());
        assert!(video_dimensions_from_parts(&info, &bytes[..11]).is_err());
    }
}
