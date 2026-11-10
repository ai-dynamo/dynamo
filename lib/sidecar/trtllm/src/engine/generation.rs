// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request dispatch, encoder-result release, and cancellation settlement.

use dynamo_backend_common::{
    DynamoError, GenerateContext, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, usage,
};
use dynamo_llm::protocols::common::preprocessor::RELEASE_ENCODER_RESULT;
use futures::stream::BoxStream;

use super::TrtllmSidecarEngine;
use crate::client;
use crate::convert::{
    ResponseState, build_generate_request_with_routing, validate_remote_media_urls,
};

/// Terminal output emitted when a request is cancelled, carrying the usage
/// accumulated so far.
fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(state.prompt_tokens(), state.completion_tokens()))
}

impl TrtllmSidecarEngine {
    pub(super) async fn generate_request(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let (client, info) = self
            .connection
            .get()
            .filter(|_| self.started.initialized())
            .ok_or_else(|| client::engine_shutdown("TensorRT-LLM sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        if request
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get(RELEASE_ENCODER_RESULT))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            if !self.mode.is_encode() {
                return Err(client::invalid_argument(
                    "encoder cache release requires an encode worker",
                ));
            }
            let result = request.encoder_result.as_ref().ok_or_else(|| {
                client::invalid_argument("encoder cache release requires a reference")
            })?;
            client
                .release_encoder_cache(crate::convert::encoder_cache_from_result(result)?)
                .await?;
            ctx.notify_first_token();
            return Ok(Box::pin(futures::stream::once(async {
                Ok(LLMEngineOutput {
                    finish_reason: Some(dynamo_backend_common::FinishReason::Stop),
                    ..Default::default()
                })
            })));
        }
        // A single-rank engine needs no hint, even when it advertises the
        // capability. Suppress it and the associated anonymous conversation ID
        // to avoid retaining one-shot per-conversation cache state.
        let use_dp_targeting = !self.mode.is_encode()
            && self
                .routing
                .get()
                .is_some_and(|routing| routing.dp_targeting && routing.dp_size > 1);
        let target_dp_rank = request.routing.as_ref().and_then(|routing| {
            if self.mode.is_prefill() {
                routing.prefill_dp_rank.or(routing.dp_rank)
            } else {
                routing.dp_rank
            }
        });
        if let Some(rank) = target_dp_rank {
            let dp_size = self.routing.get().map_or(1, |routing| routing.dp_size);
            if rank >= dp_size {
                return Err(client::invalid_argument(format!(
                    "data-parallel rank {rank} is outside the engine range 0..{dp_size}",
                )));
            }
        }
        let target_dp_rank = target_dp_rank.filter(|_| use_dp_targeting);
        let parent_affinity = if self.mode.is_prefill() && self.subagent_affinity_auth_key.is_some()
        {
            request.agent_context.as_ref().and_then(|agent| {
                agent
                    .parent_session_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|parent| !parent.is_empty() && *parent != agent.session_id.trim())
                    .map(str::to_owned)
            })
        } else {
            None
        };
        if parent_affinity.is_some()
            && request
                .agent_context
                .as_ref()
                .is_none_or(|agent| agent.session_id.trim().is_empty())
        {
            return Err(client::invalid_argument(
                "subagent affinity requires a stable child session ID",
            ));
        }
        let mut state = ResponseState::new(&request, self.mode);
        let needs_cancellation_deferral = self.mode.is_decode() && request.prefill_result.is_some();
        let stopped_ctx = ctx.inner_arc();
        // Hoisted: `stopped()` is an async-trait method, so re-creating it per
        // streamed chunk costs a boxed future and a waker registration on every
        // token.
        let mut request_cancellation = Box::pin(async move { stopped_ctx.stopped().await });
        let shutdown = self.cancel.child_token();
        let mut shutdown_cancellation = Box::pin(async move { shutdown.cancelled().await });

        // Dispatch and response headers need the same protection as streaming.
        let stream = tokio::select! {
            biased;
            _ = &mut request_cancellation, if !needs_cancellation_deferral => None,
            _ = &mut shutdown_cancellation => None,
            result = async {
                validate_remote_media_urls(&request, self.allow_remote_media_urls)?;
                let proto_request = build_generate_request_with_routing(
                    request,
                    &request_id,
                    &self.model.source,
                    Some(info.limits),
                    self.mode,
                    use_dp_targeting,
                )?;
                client.generate(
                    proto_request,
                    target_dp_rank,
                    parent_affinity.as_deref().zip(self.subagent_affinity_auth_key.as_deref()),
                ).await
            } => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = cancelled(&state);
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            let mut is_transfer_settled = false;
            loop {
                tokio::select! {
                    biased;
                    _ = &mut request_cancellation, if !needs_cancellation_deferral || is_transfer_settled => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    _ = &mut shutdown_cancellation => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    message = stream.message() => {
                        match message {
                            Ok(Some(response)) => match state.convert(response) {
                                Ok(Some(output)) => {
                                    is_transfer_settled |= !output.token_ids.is_empty();
                                    let terminal = output.finish_reason.is_some();
                                    yield Ok(output);
                                    if terminal {
                                        break;
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            },
                            Ok(None) => {
                                yield Err(client::stream_incomplete(
                                    "TensorRT-LLM Generate ended before a terminal response",
                                ));
                                break;
                            }
                            Err(status) => {
                                yield Err(client::stream_status_to_dynamo("Generate", status));
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }
}
