// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::DynamoError;
use futures::Stream;

use crate::protocols::openai::stream_aggregator::collect_unary_stream;
use crate::types::Annotated;

use super::NvAudioTranscriptionResponse;

impl NvAudioTranscriptionResponse {
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<NvAudioTranscriptionResponse>>,
    ) -> Result<NvAudioTranscriptionResponse, DynamoError> {
        collect_unary_stream(stream).await
    }
}
