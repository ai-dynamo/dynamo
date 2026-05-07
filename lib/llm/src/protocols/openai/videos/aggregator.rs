// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::Stream;

use crate::protocols::openai::stream_aggregator::{StreamAggregable, aggregate_stream};
use crate::types::Annotated;

use super::NvVideosResponse;

impl StreamAggregable for NvVideosResponse {
    fn empty() -> Self {
        Self::empty()
    }

    fn merge(&mut self, next: Self) {
        self.data.extend(next.data);
    }
}

impl NvVideosResponse {
    /// Aggregates an annotated stream of video responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvVideosResponse>>,
    ) -> Result<NvVideosResponse, String> {
        aggregate_stream(stream).await
    }
}
