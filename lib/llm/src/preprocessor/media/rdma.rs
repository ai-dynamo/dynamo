// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use dynamo_memory::SystemStorage;
use dynamo_memory::nixl::{self, AgentConfig, NixlAgent, NixlDescriptor, RegisteredView};
use dynamo_runtime::config::{
    env_config::parse_or_default, environment_names::llm::DYN_MM_NIXL_PROGRESS_DELAY_US,
};
use flate2::{Compression, write::ZlibEncoder};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

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
    pub(crate) content_hash: Option<u64>,
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

    /// Canonical xxh3-64 key for decoded media. Image identity covers shape,
    /// dtype, and RGB bytes; video identity additionally covers decoded
    /// metadata that affects the model-visible token sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) content_hash: Option<String>,

    // reference to the actual data, kept alive while the rdma descriptor is alive
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<Arc<nixl::NixlRegistered<SystemStorage>>>,
}

/// How long decoded media stays registered after the frontend gives up on an
/// attempt that may still read it. A vLLM `nixl_ref` reader starts no read
/// later than 30 s after its own request ends; the other 5 s is margin.
const READ_GRACE: Duration = Duration::from_secs(35);

/// Registrations of a request's decoded media that a remote reader may still
/// read.
///
/// A worker reads the decoded media before its first token. So the frontend
/// keeps the media registered for [`READ_GRACE`] after an attempt that ends
/// before it delivered a token, and after a failed encoder hop. The grace
/// starts when the frontend gives up on the attempt or hop. It covers a late
/// read only when the worker's vLLM request ends within 5 s of that time, so
/// it does not cover a worker that the frontend lost contact with and that
/// submits the request to vLLM later.
pub(crate) struct DecodedMediaHold {
    _registrations: Vec<Arc<nixl::NixlRegistered<SystemStorage>>>,
}

impl DecodedMediaHold {
    /// The registrations of the decoded media in `data`, if there are any.
    pub(crate) fn of<'a>(
        data: impl IntoIterator<Item = &'a RdmaMediaDataDescriptor>,
    ) -> Option<Self> {
        let held: Vec<_> = data
            .into_iter()
            .filter_map(|descriptor| descriptor.source_storage.clone())
            .collect();
        (!held.is_empty()).then_some(Self {
            _registrations: held,
        })
    }

    /// Keep the registrations for [`READ_GRACE`] from now.
    pub(crate) fn release_after_grace(self) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            tracing::warn!("no async runtime to keep decoded media registered; releasing it now");
            return;
        };
        runtime.spawn(async move {
            tokio::time::sleep(READ_GRACE).await;
            drop(self);
        });
    }
}

impl RdmaMediaDataDescriptor {
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    fn local_payload(&self) -> Option<&[u8]> {
        use dynamo_memory::actions::Slice;
        let registered = self.source_storage.as_ref()?;
        let storage = registered.storage();
        // SAFETY: the descriptor keeps the registered SystemStorage alive and
        // request construction does not mutate it while this borrow exists.
        unsafe { storage.as_slice().ok() }
    }

    /// Canonical cache/routing key serialized on the descriptor.
    pub(crate) fn content_hash_key(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }

    /// Describe a decoded still image for a NIXL reader in another process.
    ///
    /// Keeps the decoder's channel count: 1 (L), 2 (LA), 3 (RGB), or 4
    /// (RGBA), 8 bits each. Fails for memory other than host memory and for
    /// shapes other than one image, `[H, W, C]` or `[1, H, W, C]`.
    pub fn remote_image(&self) -> Result<RemoteImage> {
        let descriptor = &self.nixl_descriptor;
        anyhow::ensure!(
            descriptor.mem_type == nixl::MemType::Dram,
            "decoded media is in {:?} memory; only host memory can be read remotely",
            descriptor.mem_type
        );
        let (height, width, channels) = match self.tensor_info.shape.as_slice() {
            [height, width, channels] | [1, height, width, channels]
                if (1..=4).contains(channels) =>
            {
                (*height, *width, *channels)
            }
            shape => anyhow::bail!("decoded media shape {shape:?} is not one 8-bit image"),
        };
        anyhow::ensure!(
            height
                .checked_mul(width)
                .and_then(|pixels| pixels.checked_mul(channels))
                == Some(descriptor.size),
            "decoded image of {height}x{width}x{channels} does not fill its {}-byte region",
            descriptor.size
        );
        Ok(RemoteImage {
            agent_metadata: decode_nixl_metadata(&self.nixl_metadata)?,
            addr: descriptor.addr,
            height: u32::try_from(height)
                .map_err(|_| anyhow::anyhow!("decoded image height exceeds u32"))?,
            width: u32::try_from(width)
                .map_err(|_| anyhow::anyhow!("decoded image width exceeds u32"))?,
            channels: channels as u32,
        })
    }

    /// Numeric form used by MM-aware KV routing.
    #[cfg(feature = "mm-routing")]
    pub(crate) fn content_hash(&self) -> Option<u64> {
        self.content_hash_key()
            .and_then(|key| u64::from_str_radix(key, 16).ok())
    }

    /// Canonical identity for one frontend-decoded video:
    /// `(modality, shape, dtype, decoded metadata, sampled RGB bytes)`.
    /// Metadata is serialized from the typed object sent to the worker, so new
    /// model-visible fields automatically participate in the identity.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_content_hash(&self) -> Result<u64> {
        self.video_metadata()?;
        self.content_hash()
            .context("decoded video content hash is missing")
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

fn content_hash_for_storage(
    tensor_info: &MediaTensorInfo,
    storage: &SystemStorage,
    _hash_video: bool,
) -> Option<u64> {
    use dynamo_memory::{MemoryDescriptor, actions::Slice};

    if storage.size() == 0 {
        return None;
    }

    // SAFETY: storage owns this stable buffer and is only read while the
    // descriptor is being built, before NIXL can access it concurrently.
    let bytes = unsafe { storage.as_slice().ok()? };
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
        #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
        Some(DecodedMediaMetadata::Video(_)) => None,
        #[cfg(all(not(feature = "mm-routing"), feature = "media-ffmpeg"))]
        Some(DecodedMediaMetadata::Video(_)) => None,
        None => None,
    }
}

impl DecodedMediaData {
    /// Precompute the canonical media hash while still running on the decode
    /// thread. Images are always hashed; videos are hashed only when the
    /// request is eligible for exact MM routing.
    pub(crate) fn compute_content_hash(&mut self, hash_video: bool) {
        self.content_hash = content_hash_for_storage(&self.tensor_info, &self.data, hash_video);
    }

    pub fn into_rdma_descriptor(self, nixl_agent: &MediaAgent) -> Result<RdmaMediaDataDescriptor> {
        let source_storage = self.data;
        let content_hash = self.content_hash.map(|hash| format!("{hash:016x}"));
        let registered = nixl::register_with_nixl(source_storage, nixl_agent, None)
            .map_err(|_| anyhow::anyhow!("Failed to register storage with NIXL"))?;

        let nixl_descriptor = registered.descriptor();
        let nixl_metadata = get_nixl_metadata(nixl_agent, &nixl_descriptor)?;

        Ok(RdmaMediaDataDescriptor {
            nixl_metadata,
            nixl_descriptor,
            tensor_info: self.tensor_info,
            content_hash,
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
            content_hash: None,
        })
    }
}

/// The NIXL agent of a media loader, with the part of its agent metadata
/// before the memory sections: the agent name and the connection info. That
/// part stays the same for the life of the agent, so the agent reads it once.
pub struct MediaAgent {
    agent: NixlAgent,
    connections: Vec<u8>,
}

impl MediaAgent {
    pub fn new(agent: NixlAgent) -> Result<Self> {
        // With an empty list, `getLocalPartialMD` gives only the connection info.
        let connections = agent
            .raw_agent()
            .get_local_partial_md(&nixl::RegDescList::new(nixl::MemType::Dram)?, None)?;
        let connections = connections[..memory_section_offset(&connections)?].to_vec();
        Ok(Self { agent, connections })
    }
}

impl std::ops::Deref for MediaAgent {
    type Target = NixlAgent;

    fn deref(&self) -> &NixlAgent {
        &self.agent
    }
}

/// Agent metadata with the connection info and the section of `region`.
///
/// nixl-sys 1.3.2 cannot ask `getLocalPartialMD` for both: with a region it
/// leaves out the connection info, and with no region it gives only the
/// connection info. Both blobs use the `nixlSerDes` layout (agent name,
/// connections, memory sections), so the connections of the first join the
/// sections of the second.
fn region_metadata(agent: &MediaAgent, region: &NixlDescriptor) -> Result<Vec<u8>> {
    let mut descs = nixl::RegDescList::new(region.mem_type)?;
    descs.add_storage_desc(region)?;
    let section = agent.raw_agent().get_local_partial_md(&descs, None)?;
    Ok([
        agent.connections.as_slice(),
        &section[memory_section_offset(&section)?..],
    ]
    .concat())
}

/// Byte offset of the `MemSection` field: after the agent name and the
/// connection entries.
fn memory_section_offset(metadata: &[u8]) -> Result<usize> {
    let malformed = || anyhow::anyhow!("NIXL agent metadata has an unexpected layout");
    let mut rest = metadata
        .strip_prefix(b"nixlSerDes|")
        .ok_or_else(malformed)?;
    let mut field = |tag: &str| -> Result<&[u8]> {
        let after_tag = rest.strip_prefix(tag.as_bytes()).ok_or_else(malformed)?;
        let (len, after_len) = after_tag
            .split_first_chunk::<{ std::mem::size_of::<usize>() }>()
            .ok_or_else(malformed)?;
        let (value, after_value) = after_len
            .split_at_checked(usize::from_ne_bytes(*len))
            .ok_or_else(malformed)?;
        rest = after_value.strip_prefix(b"|").ok_or_else(malformed)?;
        Ok(value)
    };
    field("Agent")?;
    let connections = usize::from_ne_bytes(field("Conns")?.try_into().map_err(|_| malformed())?);
    for _ in 0..connections {
        field("t")?;
        field("c")?;
    }
    Ok(metadata.len() - rest.len())
}

/// A decoded 8-bit image in host memory that another process can read over
/// NIXL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteImage {
    /// Raw NIXL agent metadata of the process that owns the memory.
    pub agent_metadata: Vec<u8>,
    /// Address of the first pixel in that process.
    pub addr: u64,
    pub height: u32,
    pub width: u32,
    /// 1 (L), 2 (LA), 3 (RGB), or 4 (RGBA).
    pub channels: u32,
}

/// Inverse of the encoding in [`get_nixl_metadata`].
fn decode_nixl_metadata(encoded: &str) -> Result<Vec<u8>> {
    use std::io::Read as _;

    let compressed = general_purpose::STANDARD.decode(
        encoded
            .strip_prefix("b64:")
            .ok_or_else(|| anyhow::anyhow!("NIXL metadata must start with `b64:`"))?,
    )?;
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(compressed.as_slice()).read_to_end(&mut raw)?;
    Ok(raw)
}

// Get NIXL metadata for one registered region: the agent's connection info and
// that region only. `get_local_md` would list every live registration of the
// agent, so its size would grow with the loader cache and with concurrent requests.
// Returns zlib-compressed, base64-encoded metadata in format: "b64:<compressed_base64>"
// This format matches what Python nixl_connect expects for RdmaMetadata.nixl_metadata
pub fn get_nixl_metadata(agent: &MediaAgent, region: &NixlDescriptor) -> Result<String> {
    let nixl_md = region_metadata(agent, region)?;

    // Compress with zlib (level 6, matching Python's default)
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(&nixl_md)?;
    let compressed = encoder.finish()?;

    let b64_encoded = general_purpose::STANDARD.encode(&compressed);
    Ok(format!("b64:{}", b64_encoded))
}

// Readers send a notification with each read, and the agent keeps every notification until
// someone takes it. The frontend frees buffers by reference count and never reads them, so
// without this drain each read leaks about 100 bytes.
pub fn drain_nixl_notifications(agent: &NixlAgent) {
    let result = nixl::NotificationMap::new()
        .and_then(|mut notifs| agent.get_notifications(&mut notifs, None));
    if let Err(error) = result {
        tracing::warn!(%error, "failed to drain media-loader NIXL notifications");
    }
}

// NIXL's default progress-thread delay of 0 is the `poll()` timeout of the UCX progress
// thread, so each idle agent spins a core. With a delay, the thread sleeps in `poll()` for
// at most this long and wakes early on UCX events. Keep it short: with UCX over TCP, some
// events do not wake `poll()` and wait for the timeout.
const DEFAULT_PROGRESS_THREAD_DELAY_US: u64 = 1_000;
// NIXL 1.4.1 and earlier cast the delay to `int` before they clamp it. The cast wraps
// modulo 2^32, so values of 2^31 or more give a negative, zero, or unrelated positive
// `poll()` timeout. NIXL main clamps first. 1 s is far below that limit.
const MAX_PROGRESS_THREAD_DELAY_US: u64 = 1_000_000;

static PROGRESS_THREAD_DELAY_US: LazyLock<u64> = LazyLock::new(|| {
    progress_thread_delay_or_default(parse_or_default(
        DYN_MM_NIXL_PROGRESS_DELAY_US,
        DEFAULT_PROGRESS_THREAD_DELAY_US,
    ))
});

fn progress_thread_delay_or_default(delay_us: u64) -> u64 {
    if delay_us <= MAX_PROGRESS_THREAD_DELAY_US {
        return delay_us;
    }
    tracing::warn!(
        env_var = DYN_MM_NIXL_PROGRESS_DELAY_US,
        value = delay_us,
        max = MAX_PROGRESS_THREAD_DELAY_US,
        "invalid environment variable, using default"
    );
    DEFAULT_PROGRESS_THREAD_DELAY_US
}

pub fn get_nixl_agent() -> Result<NixlAgent> {
    let name = format!("media-loader-{}", uuid::Uuid::new_v4());
    let config = AgentConfig {
        pthr_delay_us: *PROGRESS_THREAD_DELAY_US,
        ..AgentConfig::default()
    };
    let mut nixl_agent = NixlAgent::new_with_config(&name, &config)?;
    nixl_agent
        .add_backend("UCX")
        .context("add UCX backend to media-loader NIXL agent")?;
    Ok(nixl_agent)
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_PROGRESS_THREAD_DELAY_US, DataType, MAX_PROGRESS_THREAD_DELAY_US,
        canonical_content_hash, progress_thread_delay_or_default,
    };

    #[test]
    fn progress_thread_delay_out_of_range_uses_default() {
        assert_eq!(progress_thread_delay_or_default(0), 0);
        assert_eq!(
            progress_thread_delay_or_default(MAX_PROGRESS_THREAD_DELAY_US),
            MAX_PROGRESS_THREAD_DELAY_US
        );
        assert_eq!(
            progress_thread_delay_or_default(MAX_PROGRESS_THREAD_DELAY_US + 1),
            DEFAULT_PROGRESS_THREAD_DELAY_US
        );
    }

    #[test]
    fn canonical_content_hash_payload_is_stable() {
        let bytes = [0_u8, 1, 2, 3, 4, 5];
        let hash = canonical_content_hash(&[1, 2, 3], DataType::UINT8, &bytes);

        assert_eq!(hash, 0x7a9b_bcb1_1a89_8630);
        assert_eq!(format!("{hash:016x}"), "7a9bbcb11a898630");
    }
}
#[cfg(all(test, feature = "mm-routing", feature = "media-ffmpeg"))]
mod video_tests {
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
    fn video_hash_is_precomputed_from_decoded_storage() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let mut storage = SystemStorage::new(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.as_mut_ptr(), bytes.len());
        }

        assert_eq!(
            content_hash_for_storage(&info, &storage, true),
            Some(hash_video_content(&info, &bytes).unwrap())
        );
    }

    #[test]
    fn video_hash_is_skipped_when_exact_routing_is_ineligible() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let mut storage = SystemStorage::new(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.as_mut_ptr(), bytes.len());
        }

        assert_eq!(content_hash_for_storage(&info, &storage, false), None);
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

#[cfg(test)]
mod remote_image_tests {
    use super::*;

    fn encode(raw: &[u8]) -> String {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
        encoder.write_all(raw).unwrap();
        format!(
            "b64:{}",
            general_purpose::STANDARD.encode(encoder.finish().unwrap())
        )
    }

    fn descriptor(
        shape: serde_json::Value,
        size: usize,
        mem_type: &str,
    ) -> RdmaMediaDataDescriptor {
        serde_json::from_value(serde_json::json!({
            "nixl_metadata": encode(b"agent metadata"),
            "nixl_descriptor": {"addr": 4096, "size": size, "mem_type": mem_type, "device_id": 0},
            "shape": shape,
            "dtype": "UINT8",
            "metadata": null,
        }))
        .unwrap()
    }

    #[test]
    fn still_images_in_host_memory_describe_a_remote_read() {
        for channels in 1..=4 {
            for shape in [
                serde_json::json!([2, 3, channels]),
                serde_json::json!([1, 2, 3, channels]),
            ] {
                let image = descriptor(shape, 6 * channels, "Dram")
                    .remote_image()
                    .unwrap();
                assert_eq!(
                    image,
                    RemoteImage {
                        agent_metadata: b"agent metadata".to_vec(),
                        addr: 4096,
                        height: 2,
                        width: 3,
                        channels: channels as u32,
                    }
                );
            }
        }
    }

    #[test]
    fn other_memory_shapes_and_sizes_are_rejected() {
        for (shape, size, mem_type, message) in [
            (serde_json::json!([2, 3, 3]), 18, "Vram", "only host memory"),
            (
                serde_json::json!([4, 2, 3, 3]),
                72,
                "Dram",
                "is not one 8-bit image",
            ),
            (
                serde_json::json!([2, 3, 5]),
                30,
                "Dram",
                "is not one 8-bit image",
            ),
            (
                serde_json::json!([2, 3, 3]),
                20,
                "Dram",
                "does not fill its 20-byte region",
            ),
        ] {
            let error = descriptor(shape, size, mem_type)
                .remote_image()
                .unwrap_err()
                .to_string();
            assert!(error.contains(message), "{error}");
        }
    }

    #[test]
    fn metadata_must_use_the_frontend_encoding() {
        assert_eq!(decode_nixl_metadata(&encode(b"md")).unwrap(), b"md");
        assert!(decode_nixl_metadata("md").is_err());
        assert!(decode_nixl_metadata("b64:not base64!").is_err());
        assert!(
            decode_nixl_metadata(&format!("b64:{}", general_purpose::STANDARD.encode(b"md")))
                .is_err()
        );
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod region_metadata_tests {
    use super::*;

    fn register(agent: &NixlAgent) -> nixl::NixlRegistered<SystemStorage> {
        match nixl::register_with_nixl(SystemStorage::new(4096).unwrap(), agent, None) {
            Ok(registered) => registered,
            Err(_) => panic!("failed to register test storage with NIXL"),
        }
    }

    /// The metadata of one image must not list the other live images: the
    /// loader cache and concurrent requests keep many of them registered.
    #[test]
    fn region_metadata_does_not_grow_with_other_registrations() {
        let agent = MediaAgent::new(get_nixl_agent().expect("NIXL agent")).unwrap();
        let image = register(&agent);
        let alone = get_nixl_metadata(&agent, &image.descriptor()).unwrap();
        let others: Vec<_> = (0..32).map(|_| register(&agent)).collect();
        let crowded = get_nixl_metadata(&agent, &image.descriptor()).unwrap();
        assert_eq!(alone, crowded);

        let region = decode_nixl_metadata(&crowded).unwrap();
        let full = agent.raw_agent().get_local_md().unwrap();
        assert!(
            full.len() >= region.len() + 32 * 24,
            "full {} bytes, region {} bytes",
            full.len(),
            region.len()
        );
        // Another agent connects and loads the region from it alone.
        let reader = get_nixl_agent().expect("NIXL agent");
        let name = reader.raw_agent().load_remote_md(&region).unwrap();
        assert_eq!(name, agent.raw_agent().name());
        drop(others);
    }

    /// A hold keeps the registration for the read grace period after the
    /// request drops its own reference, and then releases it.
    #[tokio::test(start_paused = true)]
    async fn hold_releases_after_the_read_grace() {
        let agent = MediaAgent::new(get_nixl_agent().expect("NIXL agent")).unwrap();
        let decoded = DecodedMediaData::try_from(ndarray::Array3::<u8>::zeros((2, 2, 3))).unwrap();
        let descriptor = decoded.into_rdma_descriptor(&agent).unwrap();
        let registration = std::sync::Arc::downgrade(descriptor.source_storage.as_ref().unwrap());

        DecodedMediaHold::of([&descriptor])
            .unwrap()
            .release_after_grace();
        drop(descriptor);
        tokio::time::sleep(READ_GRACE - Duration::from_secs(1)).await;
        assert!(
            registration.upgrade().is_some(),
            "released before the grace ended"
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(
            registration.upgrade().is_none(),
            "still held after the grace"
        );
    }
}
