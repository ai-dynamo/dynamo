// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};

use super::decoders::DecodedMediaMetadata;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    UINT8,
}

/// Tensor metadata shared by decoded media and its transport descriptor.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MediaTensorInfo {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DataType,
    pub(crate) metadata: Option<DecodedMediaMetadata>,
}

/// Decoded media bytes and their tensor metadata.
#[derive(Debug)]
pub struct DecodedMediaData {
    pub(crate) data: Vec<u8>,
    pub(crate) tensor_info: MediaTensorInfo,
    pub(crate) content_hash: Option<u64>,
}

impl DecodedMediaData {
    /// Precompute the canonical media hash while still running on the decode
    /// thread. Images are always hashed; videos are hashed only when the
    /// request is eligible for exact MM routing.
    pub(crate) fn compute_content_hash(&mut self, hash_video: bool) {
        self.content_hash = content_hash(&self.tensor_info, &self.data, hash_video);
    }
}

impl<D: Dimension> TryFrom<ArrayBase<OwnedRepr<u8>, D>> for DecodedMediaData {
    type Error = anyhow::Error;

    fn try_from(array: ArrayBase<OwnedRepr<u8>, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();
        let (data, _) = array.into_raw_vec_and_offset();

        Ok(Self {
            data,
            tensor_info: MediaTensorInfo {
                shape,
                dtype: DataType::UINT8,
                metadata: None,
            },
            content_hash: None,
        })
    }
}

/// Hash tensor shape, type, and decoded bytes into a stable media identity.
fn canonical_content_hash(shape: &[usize], dtype: DataType, bytes: &[u8]) -> u64 {
    use xxhash_rust::xxh3::Xxh3;

    let mut hasher = Xxh3::new();
    // Rank and dimensions are fixed-width so distinct shapes cannot alias
    // after concatenation.
    hasher.update(&(shape.len() as u64).to_le_bytes());
    for &dim in shape {
        hasher.update(&(dim as u64).to_le_bytes());
    }
    // Widen this discriminant if DataType gains another variant.
    let dtype_byte: u8 = match dtype {
        DataType::UINT8 => 0,
    };
    hasher.update(&[dtype_byte]);
    hasher.update(bytes);
    hasher.digest()
}

/// Select the canonical hash algorithm for the decoded media modality.
fn content_hash(tensor_info: &MediaTensorInfo, bytes: &[u8], _hash_video: bool) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }

    match tensor_info.metadata.as_ref() {
        Some(DecodedMediaMetadata::Image(_)) => Some(canonical_content_hash(
            &tensor_info.shape,
            tensor_info.dtype,
            bytes,
        )),
        #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
        Some(DecodedMediaMetadata::Video(_)) if _hash_video => {
            match hash_video_content(tensor_info, bytes) {
                Ok(hash) => Some(hash),
                Err(error) => {
                    tracing::debug!(%error, "Skipping exact routing hash for decoded video");
                    None
                }
            }
        }
        #[cfg(feature = "media-ffmpeg")]
        Some(DecodedMediaMetadata::Video(_)) => None,
        None => None,
    }
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
/// Hash decoded video bytes together with model-visible sampling metadata.
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
/// Add an unambiguous length-prefixed byte sequence to an XXH3 hash.
fn update_len_prefixed(hasher: &mut xxhash_rust::xxh3::Xxh3, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_content_hash_payload_is_stable() {
        let bytes = [0_u8, 1, 2, 3, 4, 5];
        let hash = canonical_content_hash(&[1, 2, 3], DataType::UINT8, &bytes);

        assert_eq!(hash, 0x7a9b_bcb1_1a89_8630);
        assert_eq!(format!("{hash:016x}"), "7a9bbcb11a898630");
    }

    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    fn video_info(sampled_timestamps: Vec<f64>) -> MediaTensorInfo {
        use crate::preprocessor::media::decoders::VideoMetadata;

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

    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
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

    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    #[test]
    fn video_hash_is_precomputed_from_decoded_bytes() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);

        assert_eq!(
            content_hash(&info, &bytes, true),
            Some(hash_video_content(&info, &bytes).unwrap())
        );
    }

    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    #[test]
    fn video_hash_is_skipped_when_exact_routing_is_ineligible() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);

        assert_eq!(content_hash(&info, &bytes, false), None);
    }
}
