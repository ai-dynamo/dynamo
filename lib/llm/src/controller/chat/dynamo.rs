// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{
    discovery::{ModelManager, ModelManagerError},
    preprocessor::{
        OpenAIPreprocessor, PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY,
        REQUEST_PARSING_OPTIONS_CONTEXT_KEY, apply_request_tool_call_parsing_options,
    },
    protocols::openai::chat_completions::NvCreateChatCompletionRequest,
};
use async_trait::async_trait;
use dynamo_runtime::pipeline::Context;

use super::{ChatController, ChatError, ChatModel, ChatOutput};

/// Dispatches chat requests through Dynamo's live model catalog.
pub struct DynamoChatController {
    manager: Arc<ModelManager>,
}

impl DynamoChatController {
    pub fn new(manager: Arc<ModelManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl ChatController for DynamoChatController {
    fn resolve_model(&self, model: &str) -> ChatModel {
        let name = self.manager.resolve_canonical_name(model);
        let metric_name = self.manager.metric_model_for(&name).to_string();
        ChatModel { name, metric_name }
    }

    fn check_model_ready(&self, model: &str) -> Result<(), ChatError> {
        // Uncommitted/unknown models fall through to the engine lookup, as in
        // the existing HTTP admission path.
        if let Some(entry) = self.manager.get_committed_model(model)
            && !entry.has_ready_workers()
        {
            return Err(ChatError::ModelUnavailable(model.to_string()));
        }
        Ok(())
    }

    async fn generate(
        &self,
        mut request: Context<NvCreateChatCompletionRequest>,
    ) -> Result<ChatOutput, ChatError> {
        // Preserve omitted limits so backend generation defaults still apply.
        if request.inner.max_completion_tokens.is_none() {
            request.insert(PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY, true);
        }

        // Select the engine and its parser configuration from the same WorkerSet.
        let (engine, parsing_options) = self
            .manager
            .get_chat_completions_engine_with_parsing(&request.inner.model)
            .map_err(|error| match error {
                ModelManagerError::ModelUnavailable(model) => ChatError::ModelUnavailable(model),
                _ => ChatError::ModelNotFound,
            })?;

        let parsing_options = apply_request_tool_call_parsing_options(parsing_options, &request)
            .map_err(ChatError::InvalidRequest)?
            .with_parallel_tool_calls(request.inner.parallel_tool_calls)
            .with_move_reasoning_to_content_when_empty(
                OpenAIPreprocessor::wants_reasoning_as_content_when_empty(
                    request.chat_template_args.as_ref(),
                ),
            );
        let can_defer_all_output = OpenAIPreprocessor::stream_can_defer_all_output(
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
            request.chat_template_args.as_ref(),
        );
        request.insert(REQUEST_PARSING_OPTIONS_CONTEXT_KEY, parsing_options.clone());

        let stream = engine
            .generate(request)
            .await
            .map_err(ChatError::Generation)?;
        Ok(ChatOutput {
            stream,
            parsing_options,
            can_defer_all_output,
        })
    }
}
