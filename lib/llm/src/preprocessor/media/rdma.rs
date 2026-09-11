// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::Arc;

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use dynamo_memory::SystemStorage;
use dynamo_memory::nixl::{self, NixlAgent, NixlDescriptor, RegisteredView};
use dynamo_protocols::types::{
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestUserMessageContentPart,
};
use flate2::{Compression, write::ZlibEncoder};
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::common::EncodedMediaData;
use super::decoded::{DataType, DecodedMediaData, MediaTensorInfo};
#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
use super::decoders::DecodedMediaMetadata;
use super::decoders::{Decoder, MediaDecoder};
use super::loader::MediaFetcher;

/// NIXL descriptor for decoded media sent to the next pipeline stage.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    pub(crate) nixl_metadata: String,
    pub(crate) nixl_descriptor: NixlDescriptor,

    #[serde(flatten)]
    pub(crate) tensor_info: MediaTensorInfo,

    /// Canonical xxh3-64 key for decoded media. Image identity covers shape,
    /// dtype, and RGB bytes; video identity additionally covers decoded
    /// metadata that affects the model-visible token sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) content_hash: Option<String>,

    /// Keep the registered bytes alive while the descriptor is in use.
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

    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    fn local_payload(&self) -> Option<&[u8]> {
        use dynamo_memory::actions::Slice;

        let registered = self.source_storage.as_ref()?;
        let storage = registered.storage();
        // SAFETY: the descriptor keeps the registered SystemStorage alive and
        // request construction does not mutate it while this borrow exists.
        unsafe { storage.as_slice().ok() }
    }

    /// Return the canonical identity for one frontend-decoded video.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_content_hash(&self) -> Result<u64> {
        use anyhow::Context;

        self.video_metadata()?;
        self.content_hash()
            .context("decoded video content hash is missing")
    }

    /// Return the decoded video metadata.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_metadata(&self) -> Result<&super::decoders::VideoMetadata> {
        match self.tensor_info.metadata.as_ref() {
            Some(DecodedMediaMetadata::Video(metadata)) => Ok(metadata),
            Some(_) => anyhow::bail!("decoded media metadata is not video metadata"),
            None => anyhow::bail!("decoded video metadata is missing"),
        }
    }

    /// Validate the contiguous `[T, H, W, 3]` payload and return its dimensions.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_dimensions(&self) -> Result<(usize, u32, u32)> {
        let bytes = self
            .local_payload()
            .ok_or_else(|| anyhow::anyhow!("decoded video has no local payload"))?;
        video_dimensions_from_parts(&self.tensor_info, bytes)
    }
}

impl DecodedMediaData {
    /// Register decoded bytes with NIXL and produce their transport descriptor.
    pub fn into_rdma_descriptor(self, nixl_agent: &NixlAgent) -> Result<RdmaMediaDataDescriptor> {
        let mut source_storage = SystemStorage::new(self.data.len())?;
        // SAFETY: both buffers are valid for `self.data.len()` bytes, do not
        // overlap, and `source_storage` owns the destination allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data.as_ptr(),
                source_storage.as_mut_ptr(),
                self.data.len(),
            );
        }
        let content_hash = self.content_hash.map(|hash| format!("{hash:016x}"));
        let registered = nixl::register_with_nixl(source_storage, nixl_agent, None)
            .map_err(|_| anyhow::anyhow!("Failed to register storage with NIXL"))?;

        let nixl_descriptor = registered.descriptor();
        let nixl_metadata = get_nixl_metadata(nixl_agent, registered.storage())?;

        Ok(RdmaMediaDataDescriptor {
            nixl_metadata,
            nixl_descriptor,
            tensor_info: self.tensor_info,
            content_hash,
            source_storage: Some(Arc::new(registered)),
        })
    }
}

/// Return compressed, base64-encoded metadata for a NIXL agent.
pub fn get_nixl_metadata(agent: &NixlAgent, _storage: &SystemStorage) -> Result<String> {
    let nixl_md = agent.raw_agent().get_local_md()?;
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(&nixl_md)?;
    let compressed = encoder.finish()?;
    let b64_encoded = general_purpose::STANDARD.encode(&compressed);
    Ok(format!("b64:{b64_encoded}"))
}

/// Create the process-local NIXL agent used for frontend media registration.
pub fn get_nixl_agent() -> Result<NixlAgent> {
    let name = format!("media-loader-{}", uuid::Uuid::new_v4());
    NixlAgent::with_backends(&name, &["UCX"])
}

/// Frontend media decoder backed by NIXL-registered storage.
pub struct MediaLoader {
    media_decoder: MediaDecoder,
    http_client: reqwest::Client,
    media_fetcher: MediaFetcher,
    nixl_agent: NixlAgent,
    cache: Option<Arc<Mutex<LoaderCache>>>,
}

impl MediaLoader {
    /// Convert a cache budget expressed in GiB into bytes.
    pub(super) fn cache_budget_bytes(value: Option<&str>) -> u64 {
        let gb = value
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0);
        (gb * (1024.0 * 1024.0 * 1024.0)) as u64
    }

    /// Read the decoded-media cache budget from the process environment.
    fn cache_budget_bytes_from_env() -> u64 {
        let value = std::env::var("DYN_MULTIMODAL_LOADER_CACHE_GB").ok();
        Self::cache_budget_bytes(value.as_deref())
    }

    /// Hash a media URL into the cache key used by this process.
    pub(super) fn cache_key(url: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        url.hash(&mut hasher);
        hasher.finish()
    }

    /// Create a frontend media loader using the supplied decoder policy.
    pub fn new(media_decoder: MediaDecoder, media_fetcher: Option<MediaFetcher>) -> Result<Self> {
        media_decoder.warn_if_unavailable_backends();
        let media_fetcher = media_fetcher.unwrap_or_else(MediaFetcher::from_env);
        let http_client = media_fetcher.build_http_client()?;
        let nixl_agent = get_nixl_agent()?;
        let cache = match Self::cache_budget_bytes_from_env() {
            0 => {
                tracing::debug!(
                    "[mm-cache] frontend media cache disabled (DYN_MULTIMODAL_LOADER_CACHE_GB=0)"
                );
                None
            }
            budget => {
                tracing::info!(
                    budget_bytes = budget,
                    "[mm-cache] frontend media cache enabled (DYN_MULTIMODAL_LOADER_CACHE_GB)"
                );
                Some(Arc::new(Mutex::new(LoaderCache::new(budget))))
            }
        };

        Ok(Self {
            media_decoder,
            http_client,
            media_fetcher,
            nixl_agent,
            cache,
        })
    }

    /// Build a loader with an explicit cache budget for tests.
    #[cfg(test)]
    pub fn with_cache_budget_bytes(
        media_decoder: MediaDecoder,
        media_fetcher: Option<MediaFetcher>,
        budget_bytes: u64,
    ) -> Result<Self> {
        let mut loader = Self::new(media_decoder, media_fetcher)?;
        loader.cache =
            (budget_bytes > 0).then(|| Arc::new(Mutex::new(LoaderCache::new(budget_bytes))));
        Ok(loader)
    }

    /// Return the current number of decoded entries in the media cache.
    pub fn cache_len(&self) -> usize {
        self.cache.as_ref().map(|c| c.lock().len()).unwrap_or(0)
    }

    /// Fetch, decode, and register one media request part.
    pub async fn fetch_and_decode_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
        media_io_kwargs: Option<&MediaDecoder>,
    ) -> Result<RdmaMediaDataDescriptor> {
        self.fetch_and_decode_media_part_with_video_hash(oai_content_part, media_io_kwargs, false)
            .await
    }

    /// Fetch and decode one media part, optionally hashing decoded video bytes.
    pub(crate) async fn fetch_and_decode_media_part_with_video_hash(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
        media_io_kwargs: Option<&MediaDecoder>,
        _hash_video: bool,
    ) -> Result<RdmaMediaDataDescriptor> {
        if let (Some(cache), ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part)) =
            (self.cache.as_ref(), oai_content_part)
            && media_io_kwargs.is_none()
            && let Some(url) = image_part.image_url.as_ref().map(|media| &media.url)
        {
            let key = Self::cache_key(url.as_str());
            if let Some(hit) = cache.lock().get(&key) {
                tracing::debug!(url_hash = key, "[mm-cache] hit");
                return Ok(hit);
            }
        }

        let decoded = match oai_content_part {
            ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                let mdc_decoder = self
                    .media_decoder
                    .image
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Model does not support image inputs"))?;
                let url = require_image_url(image_part)?;
                self.media_fetcher
                    .check_if_url_allowed_with_dns(url)
                    .await?;
                let data = EncodedMediaData::from_url(url, &self.http_client)
                    .await
                    .map_err(MediaFetcher::map_fetch_error)?;
                let decoder =
                    mdc_decoder.with_runtime(media_io_kwargs.and_then(|k| k.image.as_ref()));
                decoder.decode_async(data).await?
            }
            #[allow(unused_variables)]
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                #[cfg(not(feature = "media-ffmpeg"))]
                anyhow::bail!("Video decoding requires the 'media-ffmpeg' feature to be enabled");

                #[cfg(feature = "media-ffmpeg")]
                {
                    let mdc_decoder =
                        self.media_decoder.video.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("Model does not support video inputs")
                        })?;
                    let url = video_part
                        .video_url
                        .as_ref()
                        .map(|media| &media.url)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Cannot decode a video content part without a URL")
                        })?;
                    self.media_fetcher
                        .check_if_url_allowed_with_dns(url)
                        .await?;
                    let data = EncodedMediaData::from_url(url, &self.http_client)
                        .await
                        .map_err(MediaFetcher::map_fetch_error)?;
                    let decoder =
                        mdc_decoder.with_runtime(media_io_kwargs.and_then(|k| k.video.as_ref()));
                    decoder
                        .decode_async_with_video_hash(data, _hash_video)
                        .await?
                }
            }
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => {
                anyhow::bail!("Audio decoding is not supported yet");
            }
            _ => anyhow::bail!("Unsupported media type"),
        };

        let descriptor = decoded.into_rdma_descriptor(&self.nixl_agent)?;
        if let (Some(cache), ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part)) =
            (self.cache.as_ref(), oai_content_part)
            && media_io_kwargs.is_none()
            && let Some(url) = image_part.image_url.as_ref().map(|media| &media.url)
        {
            let key = Self::cache_key(url.as_str());
            let bytes = descriptor_bytes(&descriptor);
            cache.lock().put(key, descriptor.clone());
            tracing::debug!(url_hash = key, bytes, "[mm-cache] insert");
        }

        Ok(descriptor)
    }
}

struct LoaderCache {
    lru: LruCache<u64, RdmaMediaDataDescriptor>,
    bytes_used: u64,
    budget_bytes: u64,
}

impl LoaderCache {
    /// Create an empty cache with a decoded-byte budget.
    fn new(budget_bytes: u64) -> Self {
        Self {
            lru: LruCache::unbounded(),
            bytes_used: 0,
            budget_bytes,
        }
    }

    /// Return and promote an entry when it is present.
    fn get(&mut self, key: &u64) -> Option<RdmaMediaDataDescriptor> {
        self.lru.get(key).cloned()
    }

    /// Insert an entry and evict least-recently-used entries over budget.
    fn put(&mut self, key: u64, value: RdmaMediaDataDescriptor) {
        let value_bytes = descriptor_bytes(&value);
        if let Some(old) = self.lru.pop(&key) {
            self.bytes_used = self.bytes_used.saturating_sub(descriptor_bytes(&old));
        }
        self.lru.put(key, value);
        self.bytes_used = self.bytes_used.saturating_add(value_bytes);
        while self.bytes_used > self.budget_bytes && !self.lru.is_empty() {
            if let Some((_, old)) = self.lru.pop_lru() {
                self.bytes_used = self.bytes_used.saturating_sub(descriptor_bytes(&old));
            }
        }
    }

    /// Return the number of cached descriptors.
    fn len(&self) -> usize {
        self.lru.len()
    }
}

/// Calculate the decoded payload size represented by a descriptor.
fn descriptor_bytes(descriptor: &RdmaMediaDataDescriptor) -> u64 {
    let element_bytes = match descriptor.tensor_info.dtype {
        DataType::UINT8 => 1_u64,
    };
    descriptor
        .tensor_info
        .shape
        .iter()
        .try_fold(1_u64, |size, &dimension| size.checked_mul(dimension as u64))
        .unwrap_or(u64::MAX)
        .saturating_mul(element_bytes)
}

/// Return the URL from an image part or reject a UUID-only part.
fn require_image_url(part: &ChatCompletionRequestMessageContentPartImage) -> Result<&url::Url> {
    use anyhow::Context;

    Ok(&part
        .image_url
        .as_ref()
        .context(
            "Cannot decode an image content part without a URL; UUID-only parts must be resolved by the backend cache",
        )?
        .url)
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
/// Validate decoded video storage and return frame count, width, and height.
fn video_dimensions_from_parts(
    tensor_info: &MediaTensorInfo,
    bytes: &[u8],
) -> Result<(usize, u32, u32)> {
    use anyhow::Context;

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

#[cfg(all(test, feature = "mm-routing", feature = "media-ffmpeg"))]
mod tests {
    use super::*;
    use crate::preprocessor::media::decoders::VideoMetadata;

    #[test]
    fn video_dimensions_validate_rgb_payload_layout() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = MediaTensorInfo {
            shape: vec![2, 1, 2, 3],
            dtype: DataType::UINT8,
            metadata: Some(DecodedMediaMetadata::Video(VideoMetadata {
                source_fps: 24.0,
                source_duration: 10.0,
                sampled_timestamps: vec![0.0, 5.0],
            })),
        };

        assert_eq!(
            video_dimensions_from_parts(&info, &bytes).unwrap(),
            (2, 2, 1)
        );
        let mut rgba = info.clone();
        rgba.shape[3] = 4;
        assert!(video_dimensions_from_parts(&rgba, &bytes).is_err());
        assert!(video_dimensions_from_parts(&info, &bytes[..11]).is_err());
    }
}
