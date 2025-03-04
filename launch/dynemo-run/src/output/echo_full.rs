// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{sync::Arc, time::Duration};

use async_stream::stream;
use async_trait::async_trait;

use triton_distributed_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use triton_distributed_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;
use triton_distributed_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use triton_distributed_runtime::pipeline::{Error, ManyOut, SingleIn};
use triton_distributed_runtime::protocols::annotated::Annotated;

/// How long to sleep between echoed tokens.
/// 50ms gives us 20 tok/s.
const TOKEN_ECHO_DELAY: Duration = Duration::from_millis(50);

/// Engine that accepts un-preprocessed requests and echos the prompt back as the response
/// Useful for testing ingress such as service-http.
struct EchoEngineFull {}
pub fn make_engine_full() -> OpenAIChatCompletionsStreamingEngine {
    Arc::new(EchoEngineFull {})
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for EchoEngineFull
{
    async fn generate(
        &self,
        incoming_request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = incoming_request.transfer(());
        let deltas = request.response_generator();
        let ctx = context.context();
        let req = request.inner.messages.into_iter().last().unwrap();

        let prompt = match req {
            async_openai::types::ChatCompletionRequestMessage::User(user_msg) => {
                match user_msg.content {
                    async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt) => {
                        prompt
                    }
                    _ => anyhow::bail!("Invalid request content field, expected Content::Text"),
                }
            }
            _ => anyhow::bail!("Invalid request type, expected User message"),
        };

        let output = stream! {
            let mut id = 1;
            for c in prompt.chars() {
                // we are returning characters not tokens, so speed up some
                tokio::time::sleep(TOKEN_ECHO_DELAY/2).await;
                let inner = deltas.create_choice(0, Some(c.to_string()), None, None);
                let response = NvCreateChatCompletionStreamResponse {
                    inner,
                };
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None };
                id += 1;
            }

            let inner = deltas.create_choice(0, None, Some(async_openai::types::FinishReason::Stop), None);
            let response = NvCreateChatCompletionStreamResponse {
                inner,
            };
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None };
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}
