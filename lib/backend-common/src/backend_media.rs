// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-only media decoding for unified token workers.
//!
//! Unlike frontend decoding, this loader is never published on the model
//! card. The frontend therefore forwards URL variants unchanged and the
//! worker converts them to NIXL-backed `Decoded` variants immediately before
//! invoking the engine.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_llm::preprocessor::media::{MediaLoader, RdmaMediaDataDescriptor};
use dynamo_llm::protocols::common::preprocessor::{MultimodalData, PreprocessedRequest};
use futures::future::join_all;

use crate::error::{BackendError, DynamoError, ErrorType};

/// Keeps worker-owned decoded storage registered until the engine response
/// stream ends. The production implementation stores a cloned RDMA descriptor;
/// tests may attach a drop marker to verify the lifetime contract.
pub(crate) type MediaLifetime = Arc<dyn Send + Sync>;

pub(crate) struct LoadedMedia {
    descriptor: RdmaMediaDataDescriptor,
    lifetime: MediaLifetime,
}

impl LoadedMedia {
    fn from_descriptor(descriptor: RdmaMediaDataDescriptor) -> Self {
        Self {
            lifetime: Arc::new(descriptor.clone()),
            descriptor,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(descriptor: RdmaMediaDataDescriptor, lifetime: MediaLifetime) -> Self {
        Self {
            descriptor,
            lifetime,
        }
    }
}

#[async_trait]
pub(crate) trait BackendMediaLoader: Send + Sync {
    async fn load(&self, modality: &str, url: &str) -> anyhow::Result<LoadedMedia>;
}

#[async_trait]
impl BackendMediaLoader for MediaLoader {
    async fn load(&self, modality: &str, url: &str) -> anyhow::Result<LoadedMedia> {
        let descriptor = self.fetch_and_decode_url(modality, url, None).await?;
        Ok(LoadedMedia::from_descriptor(descriptor))
    }
}

struct PendingMedia {
    modality: String,
    index: usize,
    url: String,
}

fn invalid_media(message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(message.into())
        .build()
}

/// Decode every URL in `request` and replace it with a `Decoded` descriptor.
///
/// The request is not mutated until every fetch/decode succeeds. This makes a
/// multi-media request atomic from the engine's perspective: one corrupt item
/// rejects the request before `LLMEngine::generate` is invoked. Within each
/// modality vector, replacements are written back to their original indices.
pub(crate) async fn decode_request_media(
    mut request: PreprocessedRequest,
    loader: Option<&dyn BackendMediaLoader>,
) -> Result<(PreprocessedRequest, Vec<MediaLifetime>), DynamoError> {
    let Some(loader) = loader else {
        return Ok((request, Vec::new()));
    };
    let Some(media_map) = request.multi_modal_data.as_ref() else {
        return Ok((request, Vec::new()));
    };

    let mut pending = Vec::new();
    for (modality, items) in media_map {
        for (index, item) in items.iter().enumerate() {
            let url = match item {
                MultimodalData::Url(url) => Some(url.to_string()),
                MultimodalData::RawUrl(url) => Some(url.clone()),
                MultimodalData::Decoded(_) => None,
            };
            if let Some(url) = url {
                pending.push(PendingMedia {
                    modality: modality.clone(),
                    index,
                    url,
                });
            }
        }
    }

    if pending.is_empty() {
        return Ok((request, Vec::new()));
    }

    // join_all lets every already-started fetch/decode finish and release its
    // resources deterministically. We still inspect all results before
    // mutating the request, preserving failure atomicity.
    let results = join_all(
        pending
            .iter()
            .map(|item| loader.load(&item.modality, &item.url)),
    )
    .await;
    if let Some((position, error)) = results
        .iter()
        .enumerate()
        .find_map(|(position, result)| result.as_ref().err().map(|e| (position, e)))
    {
        let item = &pending[position];
        return Err(invalid_media(format!(
            "backend media decode failed for {} item {}: {}",
            item.modality, item.index, error
        )));
    }

    let loaded = results.into_iter().map(Result::unwrap).collect::<Vec<_>>();
    let media_map = request
        .multi_modal_data
        .as_mut()
        .expect("media map was present while staging backend decodes");
    let mut lifetimes = Vec::with_capacity(loaded.len());
    for (item, loaded) in pending.into_iter().zip(loaded) {
        let slot = media_map
            .get_mut(&item.modality)
            .and_then(|items| items.get_mut(item.index))
            .expect("media map must remain stable while backend decodes are staged");
        *slot = MultimodalData::Decoded(loaded.descriptor);
        lifetimes.push(loaded.lifetime);
    }

    Ok((request, lifetimes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeLoader {
        fail_url: Option<String>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl BackendMediaLoader for FakeLoader {
        async fn load(&self, _modality: &str, url: &str) -> anyhow::Result<LoadedMedia> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_url.as_deref() == Some(url) {
                anyhow::bail!("synthetic decode failure");
            }
            let id = url
                .rsplit('/')
                .next()
                .and_then(|name| name.as_bytes().first().copied())
                .unwrap_or_default() as u64;
            Ok(LoadedMedia::for_test(fake_descriptor(id), Arc::new(())))
        }
    }

    fn fake_descriptor(id: u64) -> RdmaMediaDataDescriptor {
        serde_json::from_value(serde_json::json!({
            "nixl_metadata": "test",
            "nixl_descriptor": {
                "addr": id,
                "size": 3,
                "mem_type": "Dram",
                "device_id": 0
            },
            "shape": [1, 1, 3],
            "dtype": "UINT8",
            "metadata": null
        }))
        .unwrap()
    }

    fn request(media: Vec<(&str, Vec<MultimodalData>)>) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![1])
            .multi_modal_data(Some(
                media
                    .into_iter()
                    .map(|(key, value)| (key.to_string(), value))
                    .collect(),
            ))
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    fn decoded_addr(item: &MultimodalData) -> u64 {
        match item {
            MultimodalData::Decoded(descriptor) => {
                serde_json::to_value(descriptor).unwrap()["nixl_descriptor"]["addr"]
                    .as_u64()
                    .unwrap()
            }
            other => panic!("expected Decoded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn disabled_loader_preserves_url_variants() {
        let request = request(vec![(
            "image_url",
            vec![MultimodalData::RawUrl("https://example.com/a.png".into())],
        )]);

        let (request, lifetimes) = decode_request_media(request, None).await.unwrap();

        assert!(lifetimes.is_empty());
        assert!(matches!(
            &request.multi_modal_data.unwrap()["image_url"][0],
            MultimodalData::RawUrl(url) if url.ends_with("a.png")
        ));
    }

    #[tokio::test]
    async fn urls_become_decoded_without_reordering_modalities() {
        let request = request(vec![
            (
                "image_url",
                vec![
                    MultimodalData::RawUrl("https://example.com/a.png".into()),
                    MultimodalData::RawUrl("https://example.com/b.png".into()),
                ],
            ),
            (
                "video_url",
                vec![MultimodalData::RawUrl("https://example.com/v.mp4".into())],
            ),
        ]);
        let loader = FakeLoader {
            fail_url: None,
            calls: AtomicUsize::new(0),
        };

        let (request, lifetimes) = decode_request_media(request, Some(&loader)).await.unwrap();
        let media = request.multi_modal_data.unwrap();

        assert_eq!(loader.calls.load(Ordering::SeqCst), 3);
        assert_eq!(lifetimes.len(), 3);
        assert_eq!(decoded_addr(&media["image_url"][0]), b'a' as u64);
        assert_eq!(decoded_addr(&media["image_url"][1]), b'b' as u64);
        assert_eq!(decoded_addr(&media["video_url"][0]), b'v' as u64);
    }

    #[tokio::test]
    async fn one_failed_decode_rejects_the_whole_transform() {
        let request = request(vec![(
            "image_url",
            vec![
                MultimodalData::RawUrl("https://example.com/a.png".into()),
                MultimodalData::RawUrl("https://example.com/b.png".into()),
            ],
        )]);
        let loader = FakeLoader {
            fail_url: Some("https://example.com/b.png".into()),
            calls: AtomicUsize::new(0),
        };

        let error = match decode_request_media(request, Some(&loader)).await {
            Ok(_) => panic!("decode should fail"),
            Err(error) => error,
        };

        assert_eq!(loader.calls.load(Ordering::SeqCst), 2);
        assert!(error.to_string().contains("image_url item 1"));
    }
}
