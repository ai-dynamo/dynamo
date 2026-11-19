// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context;
use dynamo_sidecar_common::{DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcChannelPool, startup_deadline};
use tokio::time::{timeout, timeout_at};
use tonic::{Code, Status, transport::Channel};

use crate::{
    config::Mode,
    proto::{
        asr::{
            RivaSpeechRecognitionConfigRequest,
            riva_speech_recognition_client::RivaSpeechRecognitionClient,
            riva_speech_recognition_config_response::Config as AsrModelConfig,
        },
        tts::{
            RivaSynthesisConfigRequest, riva_speech_synthesis_client::RivaSpeechSynthesisClient,
        },
    },
};

pub async fn connect(mode: &Mode) -> anyhow::Result<Arc<GrpcChannelPool>> {
    let config = &mode.worker().connection;
    let transport = config.grpc.config();
    // Connection retries and model readiness share one total startup budget.
    let deadline = startup_deadline(transport.startup_deadline)?;
    timeout_at(deadline, async {
        let channels = Arc::new(
            GrpcChannelPool::connect("Speech NIM", &config.grpc_endpoint, transport, false).await?,
        );
        loop {
            match timeout(
                transport.connect_attempt_timeout,
                model_ready(mode, channels.next_channel()),
            )
            .await
            {
                Ok(Ok(true)) => return Ok(channels),
                Ok(Ok(false)) => tracing::debug!("Waiting for Speech NIM model configuration"),
                Ok(Err(error))
                    if matches!(
                        error.code(),
                        Code::Unimplemented
                            | Code::InvalidArgument
                            | Code::Unauthenticated
                            | Code::PermissionDenied
                    ) =>
                {
                    return Err(error).context("Speech NIM model readiness check failed");
                }
                Ok(Err(error)) => tracing::debug!(%error, "Waiting for Speech NIM model service"),
                Err(error) => tracing::debug!(%error, "Speech NIM model readiness check timed out"),
            }
            tokio::time::sleep(transport.retry_interval).await;
        }
    })
    .await
    .context("Speech NIM did not become ready before the startup deadline")?
}

async fn model_ready(mode: &Mode, channel: Channel) -> Result<bool, Status> {
    match mode {
        Mode::Asr(config) => {
            let response = RivaSpeechRecognitionClient::new(channel)
                .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
                .get_riva_speech_recognition_config(RivaSpeechRecognitionConfigRequest {
                    model_name: config.nim_model.clone(),
                })
                .await?;
            asr_models_ready(&response.into_inner().model_config, &config.nim_model)
                .map_err(Status::invalid_argument)
        }
        Mode::Tts(_) => {
            let response = RivaSpeechSynthesisClient::new(channel)
                .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
                .get_riva_synthesis_config(RivaSynthesisConfigRequest::default())
                .await?;
            Ok(!response.into_inner().model_config.is_empty())
        }
    }
}

fn asr_models_ready(models: &[AsrModelConfig], model_name: &str) -> Result<bool, String> {
    let mut candidates = models
        .iter()
        .filter(|model| model_name.is_empty() || model.model_name == model_name)
        .peekable();
    if candidates.peek().is_none() {
        return Ok(false);
    }
    if candidates.any(|model| {
        ![("offline", "true"), ("type", "offline")]
            .into_iter()
            .any(|(key, offline)| {
                model
                    .parameters
                    .get(key)
                    .is_some_and(|value| value.eq_ignore_ascii_case(offline))
            })
    }) {
        return Ok(true);
    }
    let selection = if model_name.is_empty() {
        "catalog".into()
    } else {
        format!("model '{model_name}'")
    };
    Err(format!(
        "Speech NIM ASR {selection} is offline-only; realtime transcription requires a streaming model"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str, parameters: &[(&str, &str)]) -> AsrModelConfig {
        AsrModelConfig {
            model_name: name.into(),
            parameters: parameters
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
        }
    }

    #[test]
    fn asr_readiness_requires_an_eligible_profile() {
        let offline = model("offline", &[("streaming", "True"), ("offline", "tRuE")]);
        let offline_type = model("offline-type", &[("type", "OfFlInE")]);
        let streaming = model("streaming", &[("offline", "False")]);
        let unmarked = model("unmarked", &[]);
        for (selection, models, expected) in [
            ("", vec![], Ok(false)),
            ("", vec![offline.clone()], Err(())),
            ("", vec![offline_type.clone()], Err(())),
            ("", vec![offline.clone(), streaming.clone()], Ok(true)),
            ("", vec![unmarked.clone()], Ok(true)),
            ("offline", vec![offline.clone(), streaming.clone()], Err(())),
            (
                "offline-type",
                vec![offline_type, streaming.clone()],
                Err(()),
            ),
            ("streaming", vec![offline, streaming.clone()], Ok(true)),
            ("unmarked", vec![unmarked], Ok(true)),
            ("missing", vec![streaming], Ok(false)),
        ] {
            assert_eq!(
                asr_models_ready(&models, selection).map_err(|_| ()),
                expected,
                "selection={selection:?}, models={models:?}"
            );
        }
    }
}
