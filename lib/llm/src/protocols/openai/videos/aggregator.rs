// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};

use crate::types::Annotated;

use super::NvVideosResponse;

/// Aggregator for combining video response deltas into a final response.
#[derive(Debug)]
pub struct DeltaAggregator {
    response: Option<NvVideosResponse>,
    error: Option<String>,
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
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

    /// Aggregates a stream of annotated video responses into a final response.
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvVideosResponse>>,
    ) -> Result<NvVideosResponse, String> {
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
                    // For videos, we typically expect a single complete response
                    // or we accumulate data from multiple responses
                    match &mut aggregator.response {
                        Some(existing) => {
                            // Merge video data if we have multiple responses
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

        // Return early if an error was encountered.
        if let Some(error) = aggregator.error {
            return Err(error);
        }

        // Return the aggregated response or an empty response if none was found.
        Ok(aggregator.response.unwrap_or_else(NvVideosResponse::empty))
    }
}

impl NvVideosResponse {
    /// Aggregates an annotated stream of video responses into a final response.
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated video responses.
    ///
    /// # Returns
    /// * `Ok(NvVideosResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvVideosResponse>>,
    ) -> Result<NvVideosResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}
