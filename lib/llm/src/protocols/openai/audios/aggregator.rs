// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};

use crate::types::Annotated;

use super::NvAudiosResponse;

/// Aggregator for combining audio response deltas into a final response.
#[derive(Debug)]
pub struct DeltaAggregator {
    response: Option<NvAudiosResponse>,
    error: Option<String>,
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    /// Creates a new empty aggregator.
    pub fn new() -> Self {
        DeltaAggregator {
            response: None,
            error: None,
        }
    }

    /// Aggregates a stream of annotated audio responses into a final response.
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvAudiosResponse>>,
    ) -> Result<NvAudiosResponse, String> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                // Attempt to unwrap the delta, capturing any errors.
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
                    // For audio, we expect a single complete response
                    match &mut aggregator.response {
                        Some(existing) => {
                            // If we get multiple responses, keep the latest one
                            *existing = response;
                        }
                        None => {
                            aggregator.response = Some(response);
                        }
                    }
                }
                aggregator
            })
            .await;

        // Return early if an error was encountered.
        if let Some(error) = aggregator.error {
            return Err(error);
        }

        // Return the aggregated response or an empty response if none was found.
        Ok(aggregator.response.unwrap_or_else(NvAudiosResponse::empty))
    }
}

impl NvAudiosResponse {
    /// Aggregates an annotated stream of audio responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvAudiosResponse>>,
    ) -> Result<NvAudiosResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}
