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

use super::*;

use minijinja::{context, value::Value};

use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, completions::CompletionRequest,
};
use tracing;

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn messages(&self) -> Value {
        Value::from_serialize(&self.inner.messages)
    }

    fn tools(&self) -> Option<Value> {
        if self.inner.tools.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.inner.tools))
        }
    }

    fn tool_choice(&self) -> Option<Value> {
        if self.inner.tool_choice.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.inner.tool_choice))
        }
    }

    fn should_add_generation_prompt(&self) -> bool {
        if let Some(last) = self.inner.messages.last() {
            matches!(
                last,
                async_openai::types::ChatCompletionRequestMessage::User(_)
            )
        } else {
            true
        }
    }
}

impl OAIChatLikeRequest for CompletionRequest {
    fn messages(&self) -> minijinja::value::Value {
        let message = async_openai::types::ChatCompletionRequestMessage::User(
            async_openai::types::ChatCompletionRequestUserMessage {
                content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    crate::protocols::openai::completions::prompt_to_string(&self.inner.prompt),
                ),
                name: None,
            },
        );

        minijinja::value::Value::from_serialize(vec![message])
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }
}

impl OAIPromptFormatter for HfTokenizerConfigJsonFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        self.supports_add_generation_prompt
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        let mixins = Value::from_dyn_object(self.mixins.clone());

        let tools = req.tools();
        let has_tools = tools.is_some();
        let add_generation_prompt = req.should_add_generation_prompt();

        tracing::trace!(
            "Rendering prompt with tools: {:?}, add_generation_prompt: {}",
            has_tools,
            add_generation_prompt
        );

        let ctx = context! {
            messages => req.messages(),
            tools => tools,
            bos_token => self.config.bos_tok(),
            eos_token => self.config.eos_tok(),
            unk_token => self.config.unk_tok(),
            add_generation_prompt => add_generation_prompt,
            ..mixins
        };

        let ctx = context! { ..ctx, ..context! {

        }};

        let tmpl = if has_tools {
            self.env.get_template("tool_use")?
        } else {
            self.env.get_template("default")?
        };

        Ok(tmpl.render(&ctx)?)
    }
}
