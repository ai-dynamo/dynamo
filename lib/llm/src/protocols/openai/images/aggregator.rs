// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::{Stream, StreamExt};

use crate::types::Annotated;

use super::NvImagesResponse;

/// Aggregator for combining image response deltas into a final response.
#[derive(Debug)]
pub struct DeltaAggregator {
    response: Option<NvImagesResponse>,
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

    /// Aggregates a stream of annotated image responses into a final response.
    pub async fn apply(
        stream: impl Stream<Item = Annotated<NvImagesResponse>>,
    ) -> Result<NvImagesResponse, String> {
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
                    // For images, we typically expect a single complete response
                    // or we accumulate data from multiple responses
                    match &mut aggregator.response {
                        Some(existing) => {
                            // Merge image data if we have multiple responses
                            existing.inner.data.extend(response.inner.data);
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
        Ok(aggregator.response.unwrap_or_else(NvImagesResponse::empty))
    }
}

impl NvImagesResponse {
    /// Aggregates an annotated stream of image responses into a final response.
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated image responses.
    ///
    /// # Returns
    /// * `Ok(NvImagesResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvImagesResponse>>,
    ) -> Result<NvImagesResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}
