// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use std::sync::Arc;

use anyhow::{Error, Result};
use futures::{stream, stream::StreamExt};

use async_nats::client::{
    RequestError as NatsRequestError, RequestErrorKind::NoResponders as NatsNoResponders,
};
use tokenizers::Tokenizer as HfTokenizer;

use crate::{
    model_card::model::{ModelDeploymentCard, TokenizerKind},
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
    tokenizers::{HuggingFaceTokenizer, Tokenizer},
};

use dynamo_runtime::{
    pipeline::{
        async_trait, AsyncEngineContext, AsyncEngineContextProvider, ManyOut, Operator,
        ResponseStream, ServerStreamingEngine, SingleIn,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

#[allow(dead_code)]
pub struct Migration {
    pub tokenizer: Option<Tokenizer>,
}

impl Migration {
    pub async fn from_tokenizer(tokenizer: HfTokenizer) -> Result<Arc<Self>> {
        let tokenizer = HuggingFaceTokenizer::from_tokenizer(tokenizer);
        let tokenizer = Tokenizer::from(Arc::new(tokenizer));

        Ok(Arc::new(Self {
            tokenizer: Some(tokenizer),
        }))
    }

    pub async fn from_mdc(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let tokenizer = match &mdc.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(file)) => {
                HfTokenizer::from_file(file).map_err(Error::msg)?
            }
            Some(TokenizerKind::GGUF(t)) => *t.clone(),
            None => {
                return Ok(Arc::new(Self { tokenizer: None }));
            }
        };
        Self::from_tokenizer(tokenizer).await
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for Migration
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (preprocessed_request, context) = request.transfer(());
        let engine_ctx = context.context();
        const MAX_RETRIES: u16 = 3;
        let retry_manager =
            RetryManager::build(preprocessed_request, engine_ctx.clone(), next, MAX_RETRIES)
                .await?;
        let response_stream = stream::unfold(retry_manager, |mut retry_manager| async move {
            retry_manager
                .next()
                .await
                .map(|response| (response, retry_manager))
        });
        Ok(ResponseStream::new(Box::pin(response_stream), engine_ctx))
    }
}

#[allow(dead_code)]
struct RetryManager {
    request: PreprocessedRequest,
    engine_ctx: Arc<dyn AsyncEngineContext>,
    next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    next_stream: Option<ManyOut<Annotated<LLMEngineOutput>>>,
    retries_left: u16,
}

impl RetryManager {
    pub async fn build(
        preprocessed_request: PreprocessedRequest,
        engine_ctx: Arc<dyn AsyncEngineContext>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        retries_left: u16,
    ) -> Result<Self> {
        let mut slf = Self {
            request: preprocessed_request,
            engine_ctx,
            next_generate: next,
            next_stream: None,
            retries_left: retries_left + 1, // +1 to account for the initial attempt
        };
        slf.new_stream().await?;
        Ok(slf)
    }

    pub async fn next(&mut self) -> Option<Annotated<LLMEngineOutput>> {
        loop {
            let response_stream = match self.next_stream.as_mut() {
                Some(stream) => stream,
                None => {
                    tracing::error!("next() called with next_stream is None - should not happen");
                    return Some(Annotated::from_err(
                        Error::msg("next_stream is None").into(),
                    ));
                }
            };
            if let Some(response) = response_stream.next().await {
                if let Some(err) = response.err() {
                    const STREAM_ERR_MSG: &str = "Stream ended before generation completed";
                    if format!("{:?}", err) == STREAM_ERR_MSG {
                        tracing::info!("Stream disconnected... recreating stream...");
                        // TODO: Why generate() does not implement Sync?
                        if let Err(err) = self.new_stream_spawn().await {
                            tracing::info!("Cannot recreate stream: {:?}", err);
                        } else {
                            continue;
                        }
                    }
                }
                self.track_response(&response);
                return Some(response);
            }
            return None;
        }
    }

    async fn new_stream(&mut self) -> Result<()> {
        let mut response_stream: Option<Result<ManyOut<Annotated<LLMEngineOutput>>>> = None;
        while self.retries_left > 0 {
            self.retries_left -= 1;
            // TODO: Is there anything needed to pass between context?
            let request = SingleIn::new(self.request.clone());
            response_stream = Some(self.next_generate.generate(request).await);
            if let Some(err) = response_stream.as_ref().unwrap().as_ref().err() {
                if let Some(req_err) = err.downcast_ref::<NatsRequestError>() {
                    if matches!(req_err.kind(), NatsNoResponders) {
                        tracing::info!("Creating new stream... retrying...");
                        continue;
                    }
                }
            }
            break;
        }
        match response_stream {
            Some(Ok(next_stream)) => {
                self.next_stream = Some(next_stream);
                Ok(())
            }
            Some(Err(err)) => Err(err), // should propagate streaming error if stream started
            None => Err(Error::msg(
                "Retries exhausted - should propagate streaming error",
            )),
        }
    }

    // Same as `new_stream`, but spawns a new task to create the stream.
    // This can be used in place of `new_stream`, but keeping `new_stream` for the initial stream
    // creation due to performance reasons.
    async fn new_stream_spawn(&mut self) -> Result<()> {
        let mut response_stream: Option<Result<ManyOut<Annotated<LLMEngineOutput>>>> = None;
        while self.retries_left > 0 {
            self.retries_left -= 1;
            // TODO: Is there anything needed to pass between context?
            let request = SingleIn::new(self.request.clone());
            let next = self.next_generate.clone();
            let handle = tokio::spawn(async move { next.generate(request).await });
            response_stream = Some(match handle.await {
                Ok(response_stream) => response_stream,
                Err(err) => {
                    tracing::error!("Failed to spawn generate stream: {:?}", err);
                    return Err(Error::msg("Failed to spawn generate stream"));
                }
            });
            if let Some(err) = response_stream.as_ref().unwrap().as_ref().err() {
                if let Some(req_err) = err.downcast_ref::<NatsRequestError>() {
                    if matches!(req_err.kind(), NatsNoResponders) {
                        tracing::info!("Creating new stream... retrying...");
                        continue;
                    }
                }
            }
            break;
        }
        match response_stream {
            Some(Ok(next_stream)) => {
                self.next_stream = Some(next_stream);
                Ok(())
            }
            Some(Err(err)) => Err(err), // should propagate streaming error if stream started
            None => Err(Error::msg(
                "Retries exhausted - should propagate streaming error",
            )),
        }
    }

    fn track_response(&mut self, response: &Annotated<LLMEngineOutput>) {
        let llm_engine_output = match response.data.as_ref() {
            Some(output) => output,
            None => return,
        };
        for token_id in llm_engine_output.token_ids.iter() {
            self.request.token_ids.push(*token_id);
        }
    }
}
