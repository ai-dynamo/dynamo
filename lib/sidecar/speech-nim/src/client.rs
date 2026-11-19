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
            let models = response.into_inner().model_config;
            if !config.nim_model.is_empty()
                && models.iter().any(|model| {
                    model.model_name == config.nim_model
                        && (model
                            .parameters
                            .get("offline")
                            .is_some_and(|value| value.eq_ignore_ascii_case("true"))
                            || model
                                .parameters
                                .get("type")
                                .is_some_and(|value| value.eq_ignore_ascii_case("offline")))
                })
            {
                return Err(Status::invalid_argument(format!(
                    "Speech NIM ASR model '{}' is offline-only; realtime transcription requires a streaming model",
                    config.nim_model
                )));
            }
            Ok(!models.is_empty())
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
