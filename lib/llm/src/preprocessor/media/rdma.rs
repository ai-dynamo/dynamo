// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use dynamo_memory::nixl::{self, NixlAgent, NixlDescriptor, RegisteredView};
use dynamo_memory::{MemoryDescriptor, StorageKind, SystemStorage};
use flate2::{Compression, write::ZlibEncoder};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Arc;

use super::decoders::DecodedMediaMetadata;

#[cfg(feature = "media-ffmpeg")]
const DEFAULT_MAX_PROCESSED_MEDIA_BYTES: usize = 256 * 1024 * 1024;

type SharedNixlStorage = Arc<nixl::NixlRegistered<Arc<dyn nixl::NixlMemory + Send + Sync>>>;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    UINT8,
    FLOAT32,
    INT64,
    FLOAT64,
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

    // The outer Arc makes the registration descriptor-cloneable. The inner
    // Arc is the storage owner required by NixlRegistered and permits decoded
    // SystemStorage and processed Vec<f32> to share this descriptor type.
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<SharedNixlStorage>,
}

#[derive(Debug)]
struct OwnedF32Storage {
    data: Vec<f32>,
}

impl MemoryDescriptor for OwnedF32Storage {
    fn addr(&self) -> usize {
        self.data.as_ptr() as usize
    }

    fn size(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::System
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl nixl::NixlCompatible for OwnedF32Storage {
    fn nixl_params(&self) -> (*const u8, usize, nixl::MemType, u64) {
        (
            self.data.as_ptr().cast::<u8>(),
            self.size(),
            nixl::MemType::Dram,
            0,
        )
    }
}

/// Model-ready named fields. The wire contract is independent of model and backend.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProcessedMediaDataDescriptor {
    pub modality: dynamo_multimodal::types::Modality,
    pub fields: BTreeMap<String, ProcessedFieldDescriptor>,
    pub feature_token_counts: Vec<usize>,
    pub original_sizes: Vec<(u32, u32)>,
    pub content_hashes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProcessedFieldDescriptor {
    pub layout: dynamo_multimodal::types::FieldLayout,
    pub keep_on_host: bool,
    pub forward: bool,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    #[serde(flatten)]
    pub storage: ProcessedFieldStorageDescriptor,
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<SharedNixlStorage>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum ProcessedFieldStorageDescriptor {
    Rdma {
        nixl_metadata: String,
        nixl_descriptor: NixlDescriptor,
    },
    Inline {
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
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
    #[cfg(feature = "mm-routing")]
    pub(crate) fn content_hash(&self) -> Option<u64> {
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
            DataType::FLOAT32 => 1,
            DataType::INT64 => 2,
            DataType::FLOAT64 => 3,
        };
        hasher.update(&[dtype_byte]);
        // SAFETY: the registered storage is borrowed for this call, keeps its
        // allocation alive, and cannot relocate it. Request construction does
        // not mutate registered media concurrently.
        let bytes = unsafe { std::slice::from_raw_parts(storage.addr() as *const u8, len) };
        hasher.update(bytes);
        Some(hasher.digest())
    }
}

impl DecodedMediaData {
    pub fn into_rdma_descriptor(self, nixl_agent: &NixlAgent) -> Result<RdmaMediaDataDescriptor> {
        let source_storage: Arc<dyn nixl::NixlMemory + Send + Sync> = Arc::new(self.data);
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

    #[cfg(feature = "media-ffmpeg")]
    pub(crate) fn preprocess_video(
        self,
        processor: &dyn dynamo_multimodal::registry::VideoProcessor,
    ) -> Result<dynamo_multimodal::processed::ProcessedMedia> {
        anyhow::ensure!(
            self.tensor_info.dtype == DataType::UINT8,
            "decoded video must be UINT8"
        );
        anyhow::ensure!(
            self.tensor_info.shape.len() == 4 && self.tensor_info.shape[3] == 3,
            "decoded video must have shape [T,H,W,3], got {:?}",
            self.tensor_info.shape
        );
        let timing = match self.tensor_info.metadata {
            Some(DecodedMediaMetadata::Video(metadata)) => dynamo_multimodal::VideoTiming {
                source_fps: metadata.source_fps,
                source_duration: metadata.source_duration,
                sampled_timestamps: metadata.sampled_timestamps,
            },
            _ => anyhow::bail!("decoded video timing metadata is required for preprocessing"),
        };
        let [frame_count, height, width, _] = self.tensor_info.shape.as_slice() else {
            unreachable!("shape validated above")
        };
        let frame_bytes = (*height)
            .checked_mul(*width)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| anyhow::anyhow!("decoded video frame size overflow"))?;
        use dynamo_memory::actions::Slice;
        let bytes = unsafe {
            self.data
                .as_slice()
                .map_err(|_| anyhow::anyhow!("decoded video storage is not host-readable"))?
        };
        anyhow::ensure!(
            bytes.len() == *frame_count * frame_bytes,
            "decoded video storage size mismatch"
        );
        let frames = bytes
            .chunks_exact(frame_bytes)
            .map(|data| dynamo_multimodal::types::RgbFrameRef {
                width: *width as u32,
                height: *height as u32,
                data,
            })
            .collect::<Vec<_>>();
        processor.process(&frames, &timing).map_err(Into::into)
    }
}

impl ProcessedMediaDataDescriptor {
    #[cfg(feature = "media-ffmpeg")]
    pub(crate) fn from_processed(
        processed: dynamo_multimodal::processed::ProcessedMedia,
        nixl_agent: &NixlAgent,
    ) -> Result<Self> {
        use dynamo_multimodal::processed::ProcessedValue;

        processed.validate()?;
        anyhow::ensure!(
            processed.feature_token_counts.len() == 1,
            "a processed-media descriptor must represent exactly one media item"
        );
        let total_bytes = processed
            .fields
            .values()
            .try_fold(0usize, |total, field| {
                let (count, element_size) = match &field.value {
                    ProcessedValue::F32Tensor { data, .. } => {
                        (data.len(), std::mem::size_of::<f32>())
                    }
                    ProcessedValue::I64Tensor { data, .. } => {
                        (data.len(), std::mem::size_of::<i64>())
                    }
                    ProcessedValue::F64Tensor { data, .. } => {
                        (data.len(), std::mem::size_of::<f64>())
                    }
                };
                count
                    .checked_mul(element_size)
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| anyhow::anyhow!("processed media byte size overflow"))?;
        anyhow::ensure!(
            total_bytes <= DEFAULT_MAX_PROCESSED_MEDIA_BYTES,
            "processed media is {total_bytes} bytes, exceeding the 256 MiB limit"
        );
        let mut hasher = blake3::Hasher::new();
        let mut fields = BTreeMap::new();
        for (name, field) in processed.fields {
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            hash_layout(&mut hasher, &field.layout);
            hasher.update(&[u8::from(field.keep_on_host), u8::from(field.forward)]);
            let (dtype, shape, storage, source_storage) = match field.value {
                ProcessedValue::F32Tensor { data, shape } => {
                    hash_shape(&mut hasher, 0, &shape);
                    hasher.update(bytemuck::cast_slice(&data));
                    let (storage, source_storage) = f32_into_rdma_storage(data, nixl_agent)?;
                    (DataType::FLOAT32, shape, storage, Some(source_storage))
                }
                ProcessedValue::I64Tensor { data, shape } => {
                    hash_shape(&mut hasher, 1, &shape);
                    hasher.update(bytemuck::cast_slice(&data));
                    (
                        DataType::INT64,
                        shape,
                        ProcessedFieldStorageDescriptor::Inline {
                            data: encode_i64_le(data),
                        },
                        None,
                    )
                }
                ProcessedValue::F64Tensor { data, shape } => {
                    hash_shape(&mut hasher, 2, &shape);
                    hasher.update(bytemuck::cast_slice(&data));
                    (
                        DataType::FLOAT64,
                        shape,
                        ProcessedFieldStorageDescriptor::Inline {
                            data: encode_f64_le(data),
                        },
                        None,
                    )
                }
            };
            fields.insert(
                name,
                ProcessedFieldDescriptor {
                    layout: field.layout,
                    keep_on_host: field.keep_on_host,
                    forward: field.forward,
                    dtype,
                    shape,
                    storage,
                    source_storage,
                },
            );
        }
        Ok(Self {
            modality: processed.modality,
            fields,
            feature_token_counts: processed.feature_token_counts,
            original_sizes: processed.original_sizes,
            content_hashes: vec![hasher.finalize().to_hex().to_string()],
        })
    }
}

fn hash_shape(hasher: &mut blake3::Hasher, dtype: u8, shape: &[usize]) {
    hasher.update(&[dtype]);
    hasher.update(&(shape.len() as u64).to_le_bytes());
    for dim in shape {
        hasher.update(&(*dim as u64).to_le_bytes());
    }
}

fn hash_layout(hasher: &mut blake3::Hasher, layout: &dynamo_multimodal::types::FieldLayout) {
    use dynamo_multimodal::types::FieldLayout;
    match layout {
        FieldLayout::Batched => {
            hasher.update(&[0]);
        }
        FieldLayout::Flat { sizes_key } => {
            hasher.update(&[1]);
            hasher.update(&(sizes_key.len() as u64).to_le_bytes());
            hasher.update(sizes_key.as_bytes());
        }
        FieldLayout::Shared => {
            hasher.update(&[2]);
        }
    };
}

fn f32_into_rdma_storage(
    data: Vec<f32>,
    nixl_agent: &NixlAgent,
) -> Result<(ProcessedFieldStorageDescriptor, SharedNixlStorage)> {
    let storage: Arc<dyn nixl::NixlMemory + Send + Sync> = Arc::new(OwnedF32Storage { data });
    let registered = nixl::register_with_nixl(storage, nixl_agent, None)
        .map_err(|_| anyhow::anyhow!("Failed to register processed tensor with NIXL"))?;
    let nixl_descriptor = registered.descriptor();
    let nixl_metadata = get_nixl_metadata(nixl_agent, registered.storage())?;
    Ok((
        ProcessedFieldStorageDescriptor::Rdma {
            nixl_metadata,
            nixl_descriptor,
        },
        Arc::new(registered),
    ))
}

fn encode_i64_le(data: Vec<i64>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<i64>());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_f64_le(data: Vec<f64>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<f64>());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
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

impl<D: Dimension> TryFrom<ArrayBase<OwnedRepr<f32>, D>> for DecodedMediaData {
    type Error = anyhow::Error;

    fn try_from(array: ArrayBase<OwnedRepr<f32>, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();
        let (data_vec, _) = array.into_raw_vec_and_offset();
        let byte_len = data_vec
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("preprocessed tensor byte size overflow"))?;
        let mut storage = SystemStorage::new(byte_len)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data_vec.as_ptr().cast::<u8>(),
                storage.as_mut_ptr(),
                byte_len,
            );
        }
        Ok(Self {
            data: storage,
            tensor_info: MediaTensorInfo {
                shape,
                dtype: DataType::FLOAT32,
                metadata: None,
            },
        })
    }
}

// Get NIXL metadata for a descriptor
// Returns zlib-compressed, base64-encoded metadata in format: "b64:<compressed_base64>"
// This format matches what Python nixl_connect expects for RdmaMetadata.nixl_metadata
// TODO: pre-allocate a fixed NIXL-registered RAM pool so metadata can be cached on the target?
pub fn get_nixl_metadata<S: ?Sized>(agent: &NixlAgent, _storage: &S) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processed_inline_field_has_backend_neutral_wire_shape() {
        let field = ProcessedFieldDescriptor {
            layout: dynamo_multimodal::types::FieldLayout::Batched,
            keep_on_host: true,
            forward: true,
            dtype: DataType::INT64,
            shape: vec![1, 3],
            storage: ProcessedFieldStorageDescriptor::Inline {
                data: encode_i64_le(vec![1, 2, 3]),
            },
            source_storage: None,
        };
        let value = serde_json::to_value(&field).unwrap();
        assert_eq!(value["storage"], "inline");
        assert_eq!(value["dtype"], "INT64");
        assert_eq!(value["layout"]["kind"], "batched");
        assert_eq!(value["shape"], serde_json::json!([1, 3]));
        assert!(value.get("format").is_none());

        let decoded: ProcessedFieldDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.dtype, DataType::INT64);
        assert_eq!(decoded.shape, vec![1, 3]);
        let ProcessedFieldStorageDescriptor::Inline { data } = decoded.storage else {
            panic!("expected inline storage");
        };
        assert_eq!(data, encode_i64_le(vec![1, 2, 3]));
    }
}
