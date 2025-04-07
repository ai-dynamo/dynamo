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

use anyhow::{Error, Result};
use async_trait::async_trait;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::stream;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_llm::protocols::common::llm_backend::{BackendInput, LLMEngineOutput};

use super::Executor;

struct State {
    request_id: String,

    cancel_token: CancellationToken,

    response_rx: mpsc::Receiver<Result<super::protocols::Output>>,

    _link_to_cancel_task: tokio::sync::oneshot::Receiver<()>,

    // set to true if we send what we expect to be a final message
    // if the engine's response stream is closed before we send a final message, we can
    // detect that condition and report an unknown error engine stream termination event
    sentinel: bool,
}

// impl Drop for State {
//     fn drop(&mut self) {
//         tracing::trace!(request_id = self.stream.id(), "dropping state");
//     }
// }

#[async_trait]
impl AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<LLMEngineOutput>>, Error> for Executor {
    async fn generate(
        &self,
        request: SingleIn<BackendInput>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        // unpack the request and context
        let (request, context) = request.into_parts();

        // grab the core context
        let context = context.context();
        let context_cloned = context.clone();

        // create a cancellation token and request id
        let cancel_token = CancellationToken::new();
        let request_id = context.id().to_string();

        let mut engine_context = self.enqueue_request(request.into())?;
        let (mut tx, rx) = tokio::sync::oneshot::channel::<()>();

        let state = State {
            request_id,
            cancel_token: cancel_token.clone(),
            _link_to_cancel_task: rx,
            response_rx: engine_context
                .take_response_rx()
                .ok_or(Error::msg("no response rx"))?,
            sentinel: false,
        };

        // create a task to monitor the the requests cancellation state
        // todo: spawn on low priority async thread pool
        tokio::spawn(async move {
            tokio::select! {
                _ = context.stopped() => {
                    tracing::debug!(request_id = context.id(), "request cancelled");
                    engine_context.cancel();
                    cancel_token.cancel();
                }
                _ = tx.closed() => {
                    tracing::debug!(request_id = context.id(), "response stream closed");
                }
            }
        });

        // create the response stream
        let stream = stream::unfold(state, |mut state| async move {
            if state.sentinel {
                tracing::debug!(
                    request_id = state.request_id,
                    "sentinel set, closing stream"
                );
                return None;
            }

            // let output = tokio::select! {
            let output = tokio::select! {
                biased;

                // await a response from the trtllm engine's response processor
                output = state.response_rx.recv() => {
                    output
                }

                // if the stream is stopped, we need to:
                // - cancel the request on the trtll engine
                // - return an output with a finish reason of cancelled
                // - mark the state as completed by setting the sentinel to true
                _ = state.cancel_token.cancelled() => {
                    tracing::debug!(request_id = state.request_id, "request cancelled");
                    // state.engine.cancel();
                    state.sentinel = true;
                    let output = LLMEngineOutput::cancelled();
                    return Some((Annotated::from_data(output), state))
                }
            };

            match output {
                Some(Ok(output)) => {
                    if output.is_final {
                        tracing::debug!(request_id = state.request_id, "final response");
                        state.sentinel = true;
                    }
                    tracing::trace!(request_id = state.request_id, "issue response");
                    let output = LLMEngineOutput::from(output);
                    Some((Annotated::from_data(output), state))
                }
                Some(Err(err)) => {
                    tracing::debug!(request_id = state.request_id, "request failed: {:?}", err);
                    state.sentinel = true;
                    Some((Annotated::from_error(err.to_string()), state))
                }
                None => {
                    tracing::debug!(request_id = state.request_id, "request completed");
                    if !state.sentinel {
                        tracing::warn!(
                            request_id = state.request_id,
                            "engine stream terminated before final response or error"
                        );
                        state.sentinel = true;
                        Some((
                            Annotated::<LLMEngineOutput>::from_error(
                                "engine stream terminated before final response".to_string(),
                            ),
                            state,
                        ))
                    } else {
                        None
                    }
                }
            }
        });

        Ok(ResponseStream::new(Box::pin(stream), context_cloned))
    }
}
