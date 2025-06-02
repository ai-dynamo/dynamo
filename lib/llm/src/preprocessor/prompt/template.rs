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

use std::{collections::HashSet, sync::Arc};

use anyhow::{Context, Ok, Result};
use minijinja::Environment;

use crate::model_card::model::{ModelDeploymentCard, PromptContextMixin, PromptFormatterArtifact};

mod context;
mod formatters;
mod oai;
mod tokcfg;

use super::{OAIChatLikeRequest, OAIPromptFormatter, PromptFormatter};
use tokcfg::ChatTemplate;

impl PromptFormatter {
    pub async fn from_mdc(mdc: ModelDeploymentCard) -> Result<PromptFormatter> {
        match mdc
            .prompt_formatter
            .ok_or(anyhow::anyhow!("MDC does not contain a prompt formatter"))?
        {
            PromptFormatterArtifact::HfTokenizerConfigJson(file) => {
                let content = std::fs::read_to_string(&file)
                    .with_context(|| format!("fs:read_to_string '{file}'"))?;
                let config: ChatTemplate = serde_json::from_str(&content)?;
                Self::from_parts(
                    config,
                    mdc.prompt_context
                        .map_or(ContextMixins::default(), |x| ContextMixins::new(&x)),
                )
            }
            PromptFormatterArtifact::GGUF(gguf_path) => {
                let config = ChatTemplate::from_gguf(&gguf_path)?;
                Self::from_parts(config, ContextMixins::default())
            }
        }
    }

    pub fn from_parts(config: ChatTemplate, context: ContextMixins) -> Result<PromptFormatter> {
        let formatter = HfTokenizerConfigJsonFormatter::new(config, context)?;
        Ok(Self::OAI(Arc::new(formatter)))
    }
}

/// Chat Template Jinja Renderer
///
/// Manages a Jinja environment with registered templates for chat formatting.
/// Handles two types of ChatTemplateValue templates:
///
/// 1. String template: Registered as the 'default' template
/// 2. Map template: Contains 'tool_use' and/or 'default' templates
///    - tool_use: Template for tool-based interactions
///    - default: Template for standard chat interactions
///
///   If the map contains both keys, the `tool_use` template is registered as the `tool_use` template
///   and the `default` template is registered as the `default` template.
struct JinjaEnvironment {
    env: Environment<'static>,
}

/// Formatter for HuggingFace tokenizer config JSON templates
///
/// Implements chat template rendering based on HuggingFace's tokenizer_config.json format.
/// Supports:
/// - Tool usage templates
/// - Generation prompts
/// - Context mixins for template customization
#[derive(Debug)]
struct HfTokenizerConfigJsonFormatter {
    env: Environment<'static>,
    config: ChatTemplate,
    mixins: Arc<ContextMixins>,
    supports_add_generation_prompt: bool,
}

// /// OpenAI Standard Prompt Formatter
// pub trait StandardPromptFormatter {
//     fn render(&self, context: &impl StandardPromptContext) -> Result<String>;
// }

// pub trait StandardPromptContext {
//     fn messages(&self) -> Value;
//     fn tools(&self) -> Option<Value>;
// }

#[derive(Debug, Clone, Default)]
pub struct ContextMixins {
    context_mixins: HashSet<PromptContextMixin>,
}
