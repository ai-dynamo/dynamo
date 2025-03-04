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

use triton_distributed_llm::backend::ExecutionContext;
use triton_distributed_llm::preprocessor::BackendInput;
use triton_distributed_llm::protocols::common::llm_backend::LLMEngineOutput;
use triton_distributed_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use triton_distributed_runtime::pipeline::{Error, ManyOut, SingleIn};
use triton_distributed_runtime::protocols::annotated::Annotated;

/// How long to sleep between echoed tokens.
/// 50ms gives us 20 tok/s.
const TOKEN_ECHO_DELAY: Duration = Duration::from_millis(50);

/// Engine that accepts pre-processed requests and echos the tokens back as the response
/// The response will include the full prompt template.
/// Useful for testing pre-processing.
struct EchoEngineCore {}
pub fn make_engine_core() -> ExecutionContext {
    Arc::new(EchoEngineCore {})
}

#[async_trait]
impl AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for EchoEngineCore
{
    async fn generate(
        &self,
        incoming_request: SingleIn<BackendInput>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, context) = incoming_request.into_parts();
        let ctx = context.context();

        let output = stream! {
            for tok in request.token_ids {
                tokio::time::sleep(TOKEN_ECHO_DELAY).await;
                yield delta_core(tok);
            }
            yield Annotated::from_data(LLMEngineOutput::stop());
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

fn delta_core(tok: u32) -> Annotated<LLMEngineOutput> {
    let delta = LLMEngineOutput {
        token_ids: vec![tok],
        tokens: None,
        text: None,
        cum_log_probs: None,
        log_probs: None,
        finish_reason: None,
    };
    Annotated::from_data(delta)
}
