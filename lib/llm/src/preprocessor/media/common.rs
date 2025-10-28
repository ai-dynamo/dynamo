// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use dynamo_async_openai::types::ChatCompletionRequestUserMessageContentPart;

use crate::block_manager::storage::{
    StorageError, SystemStorage, nixl::NixlRegisterableStorage, nixl::NixlStorage,
};
use crate::preprocessor::media::{ImageDecoder, VideoDecoder};
use nixl_sys::Agent as NixlAgent;

// Raw encoded media data (.png, .mp4, ...), optionally b64-encoded
pub struct EncodedMediaData {
    bytes: Vec<u8>,
    b64_encoded: bool,
}

// Decoded media data (image RGB, video frames pixels, ...)
pub struct DecodedMediaData {
    data: SystemStorage,
    shape: Vec<usize>,
    dtype: String,
}

// Decoded media data NIXL descriptor (sent to the next step in the pipeline / NATS)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    // b64 agent metadata
    nixl_metadata: String,
    // tensor descriptor
    nixl_descriptor: NixlStorage,
    shape: Vec<usize>,
    dtype: String,
    // reference to the actual data, kept alive while the rdma descriptor is alive
    #[serde(skip, default)]
    #[allow(dead_code)]
    source_storage: Option<Arc<SystemStorage>>,
}

impl EncodedMediaData {
    // Handles both web URLs (will download the bytes) and data URLs (will keep b64-encoded)
    // This function is kept in tokio runtime so we do not want any expensive operations
    pub async fn from_url(url: &url::Url, client: &reqwest::Client) -> Result<Self> {
        let (bytes, b64_encoded) = match url.scheme() {
            "data" => {
                let base64_data = url
                    .as_str()
                    .split_once(',')
                    .ok_or_else(|| anyhow::anyhow!("Invalid media data URL format"))?
                    .1;
                anyhow::ensure!(!base64_data.is_empty(), "Media data URL is empty");
                (base64_data.as_bytes().to_vec(), true)
            }
            "http" | "https" => {
                let bytes = client
                    .get(url.to_string())
                    .send()
                    .await?
                    .error_for_status()?
                    .bytes()
                    .await?;
                anyhow::ensure!(!bytes.is_empty(), "Media URL is empty");
                (bytes.to_vec(), false)
            }
            scheme => anyhow::bail!("Unsupported media URL scheme: {scheme}"),
        };

        Ok(Self { bytes, b64_encoded })
    }

    // Potentially decodes b64 bytes
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        if self.b64_encoded {
            Ok(general_purpose::STANDARD.decode(self.bytes)?)
        } else {
            Ok(self.bytes)
        }
    }
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
        let nixl_metadata = general_purpose::STANDARD.encode(&nixl_local_md);

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

#[async_trait::async_trait]
pub trait Decoder: Clone + Send + 'static {
    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData>;

    async fn decode_async(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        // light clone (only config params)
        let decoder = self.clone();
        // compute heavy -> rayon
        let result = tokio_rayon::spawn(move || decoder.decode(data)).await?;
        Ok(result)
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MediaDecoder {
    #[serde(default)]
    pub image_decoder: ImageDecoder,
    #[serde(default)]
    pub video_decoder: VideoDecoder,
}

pub struct MediaLoader {
    media_decoder: MediaDecoder,
    http_client: reqwest::Client,
    nixl_agent: NixlAgent,
}

impl MediaLoader {
    pub fn new(media_decoder: MediaDecoder) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .user_agent(
                "dynamo-ai/dynamo", // TODO: use a proper user agent
            )
            .build()?;

        let uuid = uuid::Uuid::new_v4();
        let nixl_agent = NixlAgent::new(&format!("media-loader-{}", uuid))?;
        let (_, ucx_params) = nixl_agent.get_plugin_params("UCX")?;
        nixl_agent.create_backend("UCX", &ucx_params)?;

        Ok(Self {
            media_decoder,
            http_client,
            nixl_agent,
        })
    }

    pub async fn fetch_and_decode_media_part(
        &self,
        oai_content_part: &ChatCompletionRequestUserMessageContentPart,
    ) -> Result<RdmaMediaDataDescriptor> {
        // TODO: request-level options
        // fetch and decode the media
        let decoded = match oai_content_part {
            ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                let url = &image_part.image_url.url;
                let data = EncodedMediaData::from_url(url, &self.http_client).await?;
                self.media_decoder.image_decoder.decode_async(data).await
            }
            ChatCompletionRequestUserMessageContentPart::VideoUrl(video_part) => {
                let url = &video_part.video_url.url;
                let data = EncodedMediaData::from_url(url, &self.http_client).await?;
                self.media_decoder.video_decoder.decode_async(data).await
            }
            ChatCompletionRequestUserMessageContentPart::AudioUrl(_) => {
                anyhow::bail!("Audio decoding is not supported yet");
            }
            _ => anyhow::bail!("Unsupported media type"),
        }?;

        let rdma_descriptor = decoded.into_rdma_descriptor(&self.nixl_agent)?;
        Ok(rdma_descriptor)
    }
}
