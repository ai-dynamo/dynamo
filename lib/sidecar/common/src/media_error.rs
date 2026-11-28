// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provenance for engine-owned fetch failures whose native protocols flatten their types.

use dynamo_backend_common::{DynamoError, MultimodalData, PreprocessedRequest};

/// Select the fixed fetch-error contract of the engine that received the request.
#[derive(Clone, Copy)]
pub enum EngineFlavor {
    Vllm,
    Sglang,
    Trtllm,
}

/// Request-scoped facts retained after the engine's request has been consumed.
#[derive(Clone, Copy, Default)]
pub struct MediaErrorContext {
    remote_media: bool,
}

impl MediaErrorContext {
    /// Record only URL-backed image, video, or audio inputs that the engine may fetch.
    pub fn from_request(request: &PreprocessedRequest) -> Self {
        let mut context = Self::default();
        for (modality, items) in request.multi_modal_data.iter().flatten() {
            if !matches!(modality.as_str(), "image_url" | "video_url" | "audio_url") {
                continue;
            }
            for item in items {
                context = match item {
                    MultimodalData::Url(url) => {
                        context.remote_media |=
                            matches!(url.scheme(), "http" | "https") && url.has_host();
                        context
                    }
                    MultimodalData::RawUrl(url) => context.with_url(url),
                    _ => context,
                };
            }
        }
        context
    }

    /// Include one actual media source from an engine's native request schema.
    pub fn with_url(mut self, source: &str) -> Self {
        self.remote_media = self.remote_media
            || url::Url::parse(source)
                .is_ok_and(|url| matches!(url.scheme(), "http" | "https") && url.has_host());
        self
    }

    /// Add provenance only; existing identities, retry hints, and private families remain owned locally.
    pub fn tag(self, engine: EngineFlavor, mut error: DynamoError) -> DynamoError {
        if !self.remote_media {
            return error;
        }
        let Some(upstream) = error.upstream_error() else {
            return error;
        };
        if !upstream.metadata_valid()
            || upstream.family_name().is_some()
            || !is_media_fetch_error(engine, upstream.message())
        {
            return error;
        }
        if let Some(upstream) = error.upstream_error.take() {
            error.upstream_error = Some(Box::new((*upstream).family("media_fetch")));
        }
        error
    }
}

/// Match only prefixes emitted by the pinned native media-fetch implementations.
// Engine/protocol upgrades must recheck this wording and the fixtures before
// updating the reviewed pins in tests/dependencies/test_sidecar_media_error_pins.py.
fn is_media_fetch_error(engine: EngineFlavor, message: &str) -> bool {
    match engine {
        EngineFlavor::Vllm => {
            // vLLM 0.33/0.34 chat/error.rs wraps the same llm-multimodal connector reports.
            let Some(detail) = message.strip_prefix("multimodal preprocessing error: ") else {
                return false;
            };
            detail.starts_with("HTTP error while fetching media: ")
                || (detail.starts_with("media host '")
                    && (detail.contains("' could not be resolved: ")
                        || detail.contains("' resolves to ")))
                || detail.starts_with("media fetch of http://")
                || detail.starts_with("media fetch of https://")
                || detail.starts_with("the fetch of http://")
                || detail.starts_with("the fetch of https://")
        }
        // SGLang 0.5.23 sglang-mm/common/fetch.rs owns this URL-bearing prefix.
        EngineFlavor::Sglang => {
            message.starts_with("media fetch: GET http://")
                || message.starts_with("media fetch: GET https://")
        }
        EngineFlavor::Trtllm => {
            // TRT-LLM 1.4.0rc3/rc4 names the failing URL field, not inline media.
            let Some(rest) = message.strip_prefix("media[") else {
                return false;
            };
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            (1..=20).contains(&digits)
                && rest.get(digits..).is_some_and(|suffix| {
                    suffix.starts_with("].url is temporarily unavailable: ")
                        || suffix.starts_with("].url could not be loaded: ")
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::{ErrorType, UpstreamError};
    use serde_json::json;

    fn received(message: &str) -> DynamoError {
        DynamoError::builder()
            .error_type(ErrorType::Backend(
                dynamo_backend_common::BackendError::Unknown,
            ))
            .diagnostic("private sidecar wrapper")
            .upstream_error(
                UpstreamError::new(message)
                    .error_type("grpc.Status")
                    .code("Internal"),
            )
            .build()
    }

    #[test]
    fn proven_fetch_signatures_only_add_media_family() {
        for (engine, message) in [
            (
                EngineFlavor::Vllm,
                "multimodal preprocessing error: HTTP error while fetching media: connection refused",
            ),
            (
                EngineFlavor::Vllm,
                "multimodal preprocessing error: media host 'missing.invalid' could not be resolved: DNS failed",
            ),
            (
                EngineFlavor::Vllm,
                "multimodal preprocessing error: media fetch of https://media.invalid/a timed out after 30s",
            ),
            (
                EngineFlavor::Vllm,
                "multimodal preprocessing error: HTTP error while fetching media: HTTP status client error (404 Not Found)",
            ),
            (
                EngineFlavor::Vllm,
                "multimodal preprocessing error: HTTP error while fetching media: HTTP status server error (503 Service Unavailable)",
            ),
            (
                EngineFlavor::Vllm,
                "multimodal preprocessing error: media host 'private.invalid' resolves to 10.0.0.1, a private address",
            ),
            (
                EngineFlavor::Vllm,
                "multimodal preprocessing error: the fetch of https://media.invalid/a was redirected to http://10.0.0.1/a, which is refused: private",
            ),
            (
                EngineFlavor::Sglang,
                "media fetch: GET https://media.invalid/a: Connection Failed: Connection refused",
            ),
            (
                EngineFlavor::Sglang,
                "media fetch: GET https://media.invalid/a: Dns Failed: resolve failed",
            ),
            (
                EngineFlavor::Sglang,
                "media fetch: GET https://media.invalid/a: timed out",
            ),
            (
                EngineFlavor::Sglang,
                "media fetch: GET https://media.invalid/a: status code 404",
            ),
            (
                EngineFlavor::Sglang,
                "media fetch: GET https://media.invalid/a: status code 503",
            ),
            (
                EngineFlavor::Trtllm,
                "media[0].url could not be loaded: hostname unresolved",
            ),
            (
                EngineFlavor::Trtllm,
                "media[12].url is temporarily unavailable: connection refused",
            ),
        ] {
            let error = received(message);
            let mut expected = serde_json::to_value(&error).unwrap();
            expected["upstream_error"]["family"] = json!("media_fetch");
            let tagged = MediaErrorContext::default()
                .with_url("https://media.invalid/a")
                .tag(engine, error);
            assert_eq!(serde_json::to_value(tagged).unwrap(), expected);
        }
    }

    #[test]
    fn inline_text_validation_and_wrapped_signatures_stay_unchanged() {
        for engine in [
            EngineFlavor::Vllm,
            EngineFlavor::Sglang,
            EngineFlavor::Trtllm,
        ] {
            for message in [
                "max_tokens must be positive",
                "unsupported multimodal content: image_embeds",
                "validation rejected literal media fetch: GET https://media.invalid/a: refused",
                r#"{"message":"media fetch: GET https://media.invalid/a: refused"}"#,
                "media[0].data_uri could not be loaded: bad inline input",
                "media[-1].url could not be loaded: invalid index",
                "media[111111111111111111111].url could not be loaded: invalid index",
            ] {
                let error = received(message);
                let expected = serde_json::to_value(&error).unwrap();
                let tagged = MediaErrorContext::default()
                    .with_url("https://media.invalid/a")
                    .tag(engine, error);
                assert_eq!(serde_json::to_value(tagged).unwrap(), expected);
            }
        }
        for source in [
            "data:image/png;base64,aHR0cHM6Ly8=",
            "file:///private/a",
            "a user mentioned https://media.invalid/a",
        ] {
            let error = received("media fetch: GET https://media.invalid/a: refused");
            let tagged = MediaErrorContext::default()
                .with_url(source)
                .tag(EngineFlavor::Sglang, error);
            assert_eq!(tagged.upstream_error().unwrap().family_name(), None);
        }
    }

    #[test]
    fn existing_private_families_and_invalid_metadata_are_preserved() {
        for family in ["transport", "python_programming", "media_fetch", "http"] {
            let mut error = received("media fetch: GET https://media.invalid/a: refused");
            let upstream = error.upstream_error.take().unwrap();
            error.upstream_error = Some(Box::new((*upstream).family(family)));
            let expected = serde_json::to_value(&error).unwrap();
            let tagged = MediaErrorContext::default()
                .with_url("https://media.invalid/a")
                .tag(EngineFlavor::Sglang, error);
            assert_eq!(serde_json::to_value(tagged).unwrap(), expected);
        }
        let error = DynamoError::builder()
            .upstream_error(
                UpstreamError::new("media fetch: GET https://media.invalid/a: refused")
                    .mark_incomplete_metadata(),
            )
            .build();
        let tagged = MediaErrorContext::default()
            .with_url("https://media.invalid/a")
            .tag(EngineFlavor::Sglang, error);
        assert!(!tagged.upstream_error().unwrap().metadata_valid());
        assert_eq!(tagged.upstream_error().unwrap().family_name(), None);
    }

    #[test]
    fn request_context_uses_actual_media_sources() {
        for (item, remote) in [
            (
                MultimodalData::Url(url::Url::parse("https://media.invalid/a").unwrap()),
                true,
            ),
            (
                MultimodalData::RawUrl("HTTP://media.invalid/a".into()),
                true,
            ),
            (
                MultimodalData::RawUrl("data:image/png;base64,eA==".into()),
                false,
            ),
            (
                MultimodalData::Json(json!({"url":"https://media.invalid/a"})),
                false,
            ),
            (
                MultimodalData::UuidOnly("https://media.invalid/a".into()),
                false,
            ),
        ] {
            let mut request = PreprocessedRequest::builder()
                .model("test".into())
                .token_ids(vec![1])
                .sampling_options(Default::default())
                .stop_conditions(Default::default())
                .output_options(Default::default())
                .build()
                .unwrap();
            request.multi_modal_data = Some(std::collections::HashMap::from([(
                "image_url".into(),
                vec![item],
            )]));
            assert_eq!(
                MediaErrorContext::from_request(&request).remote_media,
                remote
            );
        }
    }
}
