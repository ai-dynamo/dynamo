// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_sidecar_common::DEFAULT_MAX_GRPC_MESSAGE_SIZE;
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{AsrAdapter, validation::SAMPLE_RATE};
use crate::{
    events,
    proto::{
        AudioEncoding,
        asr::{
            RecognitionConfig, StreamingRecognitionConfig, StreamingRecognizeRequest,
            riva_speech_recognition_client::RivaSpeechRecognitionClient,
            streaming_recognize_request::StreamingRequest,
        },
    },
};

/// Owns the RPC and its permit; dropping this future cancels both directions.
pub(super) async fn run(
    adapter: &AsrAdapter,
    item_id: &str,
    audio: mpsc::UnboundedReceiver<Vec<u8>>,
    mut closed: oneshot::Receiver<()>,
    output: mpsc::Sender<Value>,
) {
    let Ok(_permit) = adapter.slots.try_acquire() else {
        let _ = output
            .send(events::failed(
                item_id,
                "ASR capacity exhausted; retry the turn later",
            ))
            .await;
        return;
    };
    let config = StreamingRecognitionConfig {
        config: Some(RecognitionConfig {
            encoding: AudioEncoding::LinearPcm as i32,
            sample_rate_hertz: SAMPLE_RATE as i32,
            language_code: adapter.config.language_code.clone(),
            model: adapter.config.nim_model.clone(),
            max_alternatives: 1,
            enable_automatic_punctuation: true,
            ..Default::default()
        }),
        interim_results: true,
    };
    let requests = futures::stream::once(async { StreamingRequest::StreamingConfig(config) })
        .chain(UnboundedReceiverStream::new(audio).map(StreamingRequest::AudioContent))
        .map(|request| StreamingRecognizeRequest {
            streaming_request: Some(request),
            ..Default::default()
        });
    let mut client = RivaSpeechRecognitionClient::new(adapter.channels.next_channel())
        .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
    let transcribe = async {
        let mut responses = client.streaming_recognize(requests).await?.into_inner();
        let mut segments = Vec::new();
        let mut emitted = String::new();
        let mut interim = String::new();
        while let Some(response) = responses.message().await? {
            for result in response.results {
                let Some(alternative) = result.alternatives.into_iter().next() else {
                    continue;
                };
                let hypothesis = alternative.transcript;
                if hypothesis.is_empty() {
                    continue;
                }
                // Realtime deltas append text. A revised hypothesis cannot retract
                // already emitted text; the completed event carries the final result.
                if let Some(suffix) = hypothesis.strip_prefix(&emitted) {
                    if !suffix.is_empty() {
                        let delta = if emitted.is_empty() && !segments.is_empty() {
                            format!(" {suffix}")
                        } else {
                            suffix.to_owned()
                        };
                        output
                            .send(events::delta(item_id, &delta))
                            .await
                            .map_err(|_| {
                                tonic::Status::cancelled("realtime response stream closed")
                            })?;
                    }
                    emitted = hypothesis.clone();
                }
                if result.is_final {
                    segments.push(hypothesis);
                    emitted.clear();
                    interim.clear();
                } else {
                    interim = hypothesis;
                }
            }
        }
        if !interim.is_empty() {
            segments.push(interim);
        }
        Ok::<_, tonic::Status>(segments.join(" "))
    };
    tokio::pin!(transcribe);
    let result = tokio::select! {
        // A successful early response waits for commit, but a failed RPC must
        // free the open turn immediately so the session can recover.
        result = &mut transcribe => {
            if result.is_ok() { let _ = closed.await; }
            result
        },
        _ = &mut closed => match tokio::time::timeout(adapter.config.timeout_s, &mut transcribe).await {
            Ok(result) => result,
            Err(_) => Err(tonic::Status::deadline_exceeded("transcription completion deadline exceeded")),
        },
    };
    let event = match result {
        Ok(transcript) => events::completed(item_id, &transcript),
        Err(error) => {
            tracing::warn!(code = ?error.code(), "Speech NIM realtime transcription failed");
            events::failed(item_id, "Transcription failed")
        }
    };
    let _ = output.send(event).await;
}
