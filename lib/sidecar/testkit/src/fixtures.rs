// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{
    DynamoError, GenerateContext, LLMEngine, LLMEngineOutput, PreprocessedRequest, StopConditions,
};
use futures::StreamExt;

pub type Outputs = Vec<Result<LLMEngineOutput, DynamoError>>;

pub fn request(model: &str, tokens: Vec<u32>, max_tokens: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model(model.to_owned())
        .token_ids(tokens)
        .sampling_options(Default::default())
        .output_options(Default::default())
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            ..Default::default()
        })
        .build()
        .unwrap()
}

pub async fn collect(
    engine: &(impl LLMEngine + ?Sized),
    request: PreprocessedRequest,
    ctx: GenerateContext,
) -> Outputs {
    crate::bounded("collect generation", async {
        match engine.generate(request, ctx).await {
            Ok(stream) => stream.collect().await,
            Err(error) => vec![Err(error)],
        }
    })
    .await
}
