// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-request, streaming-response audio speech adapter.

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_backend_common::{DynamoError, EngineConfig, GenerateContext, RawEngine};
use dynamo_llm::protocols::openai::{
    MediaDelivery,
    audios::{AudioData, NvAudioSpeechResponse, NvCreateAudioSpeechRequest},
};
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, invalid_argument, status_to_dynamo,
};
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;

use crate::proto::{
    AudioEncoding,
    tts::{SynthesizeSpeechRequest, riva_speech_synthesis_client::RivaSpeechSynthesisClient},
};
use crate::{config::TtsConfig, events};

pub struct TtsEngine {
    config: TtsConfig,
    channels: Arc<GrpcChannelPool>,
}

impl TtsEngine {
    pub fn new(config: TtsConfig, channels: Arc<GrpcChannelPool>) -> Self {
        Self { config, channels }
    }

    fn validate(request: &NvCreateAudioSpeechRequest, model_name: &str) -> Result<(), DynamoError> {
        let error = if request.input.trim().is_empty() {
            Some("input must contain text".into())
        } else if request.model.as_deref().is_some_and(|m| m != model_name) {
            Some(format!("model must be '{model_name}'"))
        } else if request
            .response_format
            .as_deref()
            .is_some_and(|format| format != "pcm")
        {
            Some("Speech NIM TTS requires response_format='pcm'".into())
        } else if request
            .data_source
            .is_some_and(|s| s != MediaDelivery::B64Json)
        {
            Some("Speech NIM TTS currently supports data_source='b64_json'".into())
        } else if request.speed != 1.0 {
            Some("Speech NIM TTS does not support the speed parameter".into())
        } else if request.instructions.as_ref().is_some_and(|s| !s.is_empty()) {
            Some("Speech NIM TTS does not support instructions".into())
        } else {
            [
                ("task_type", request.task_type.is_some()),
                ("ref_audio", request.ref_audio.is_some()),
                ("ref_text", request.ref_text.is_some()),
                ("max_new_tokens", request.max_new_tokens.is_some()),
                (
                    "nvext.cfg_scale",
                    request
                        .nvext
                        .as_ref()
                        .is_some_and(|n| n.cfg_scale.is_some()),
                ),
            ]
            .into_iter()
            .find(|(_, present)| *present)
            .map(|(name, _)| format!("Speech NIM TTS does not support {name}"))
        };
        match error {
            Some(message) => Err(invalid_argument(message)),
            None => Ok(()),
        }
    }
}

fn synthesis_error(status: tonic::Status) -> DynamoError {
    if status.code() == tonic::Code::InvalidArgument {
        invalid_argument("Speech NIM rejected the synthesis parameters")
    } else {
        status_to_dynamo("SynthesizeOnline", status)
    }
}

#[async_trait]
impl RawEngine for TtsEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        Ok(EngineConfig {
            model: self.config.model_name.clone(),
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: Value,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<Value, DynamoError>>, DynamoError> {
        let request: NvCreateAudioSpeechRequest = serde_json::from_value(request)
            .map_err(|e| invalid_argument(format!("invalid audio speech request: {e}")))?;
        Self::validate(&request, &self.config.model_name)?;
        let rpc_request = SynthesizeSpeechRequest {
            text: request.input,
            voice_name: request
                .voice
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| self.config.voice.clone()),
            language_code: request
                .language
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| self.config.language_code.clone()),
            encoding: AudioEncoding::LinearPcm as i32,
            sample_rate_hz: self.config.sample_rate_hz,
            ..Default::default()
        };
        let request = futures::stream::once(async { rpc_request });
        let mut client = RivaSpeechSynthesisClient::new(self.channels.next_channel())
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        let model = self.config.model_name.clone();
        let audio = async_stream::try_stream! {
            let mut responses = client.synthesize_online(request).await
                .map_err(synthesis_error)?.into_inner();
            let id = events::id("speech");
            let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
            loop {
                let response = responses.message().await.map_err(synthesis_error)?;
                let Some(response) = response else {
                    break;
                };
                let response = NvAudioSpeechResponse {
                    id: id.clone(), model: model.clone(), created,
                    data: vec![AudioData {
                        output_format: "pcm".into(), url: None,
                        b64_json: Some(STANDARD.encode(response.audio)),
                    }],
                    ..NvAudioSpeechResponse::empty()
                };
                yield serde_json::to_value(response).map_err(|e| DynamoError::msg(e.to_string()))?;
            }
        };
        Ok(async_stream::stream! {
            let cancelled = async {
                tokio::select! {
                    _ = ctx.stopped() => {},
                    _ = ctx.killed() => {},
                }
            };
            // Drop the RPC on cancellation even if the caller retains the finished stream.
            let audio = audio.take_until(cancelled);
            tokio::pin!(audio);
            while let Some(response) = audio.next().await {
                yield response;
            }
        }
        .boxed())
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::{BackendError, ErrorType};
    use serde_json::json;

    #[test]
    fn tts_validates_pcm_format_defaults() {
        for (format, valid) in [
            (None, true),
            (Some(json!(null)), true),
            (Some(json!("pcm")), true),
            (Some(json!("mp3")), false),
            (Some(json!("wav")), false),
        ] {
            let mut request = json!({"input": "Hello"});
            if let Some(format) = format {
                request["response_format"] = format;
            }
            let request = serde_json::from_value(request).unwrap();
            let result = TtsEngine::validate(&request, "test-model");
            assert_eq!(result.is_ok(), valid, "{request:?}");
            if let Err(error) = result {
                assert_eq!(
                    error.error_type(),
                    ErrorType::Backend(BackendError::InvalidArgument)
                );
                assert!(error.to_string().contains("response_format='pcm'"));
            }
        }
    }
}
