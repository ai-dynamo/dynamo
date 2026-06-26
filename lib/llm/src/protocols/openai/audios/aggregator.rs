// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::Stream;

use crate::protocols::openai::stream_aggregator::{StreamAggregable, aggregate_stream};
use crate::types::Annotated;

use super::NvAudioSpeechResponse;

impl StreamAggregable for NvAudioSpeechResponse {
    fn empty() -> Self {
        Self::empty()
    }

    fn merge(&mut self, next: Self) {
        self.data.extend(next.data);
    }
}

impl NvAudioSpeechResponse {
    /// Aggregates an annotated stream of audio responses into a final response.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvAudioSpeechResponse>>,
    ) -> Result<NvAudioSpeechResponse, String> {
        aggregate_stream(stream).await
    }
}
