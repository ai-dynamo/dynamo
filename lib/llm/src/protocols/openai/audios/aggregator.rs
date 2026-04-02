// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};

use crate::types::Annotated;

use super::NvAudioSpeechResponse;

/// Aggregator for combining audio response deltas into a final response.
#[derive(Debug)]
pub struct DeltaAggregator {
    response: Option<NvAudioSpeechResponse>,
    error: Option<String>,
}

impl Default for DeltaAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    pub fn new() -> Self {
        DeltaAggregator {
            response: None,
            error: None,
        }
    }

    /// Aggregates a stream of annotated audio responses into a final response.
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvAudioSpeechResponse>>,
    ) -> Result<NvAudioSpeechResponse, String> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none()
                    && let Some(response) = delta.data
                {
                    match &mut aggregator.response {
                        Some(existing) => {
                            existing.data.extend(response.data);
                        }
                        None => {
                            aggregator.response = Some(response);
                        }
                    }
                }
                aggregator
            })
            .await;

        if let Some(error) = aggregator.error {
            return Err(error);
        }

        Ok(aggregator
            .response
            .unwrap_or_else(NvAudioSpeechResponse::empty))
    }
}

impl NvAudioSpeechResponse {
    /// Aggregates an annotated stream of audio responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvAudioSpeechResponse>>,
    ) -> Result<NvAudioSpeechResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}
