// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::tokcfg::{ChatTemplate, raise_exception, strftime_now, tojson};
use super::{ContextMixins, HfTokenizerConfigJsonFormatter, JinjaEnvironment};
use either::Either;
use minijinja::{Environment, Value};
use tracing;

/// Remove known non-standard Jinja2 tags from chat templates
///
/// Some models use custom Jinja2 extensions that minijinja doesn't recognize. These tags
/// are typically metadata markers that don't affect the rendered output. For example:
/// - {% generation %} / {% endgeneration %}: Used by vLLM's AssistantTracker to mark
///   assistant-generated content. The tags themselves don't produce output.
///
/// By removing these tags before validation, we allow templates with backend-specific
/// extensions to work with minijinja while maintaining correct output semantics.
///
/// Note: This follows the same approach as Mistral.rs, which also strips these tags
/// for compatibility: https://github.com/EricLBuehler/mistral.rs/blob/2bcf0e9/mistralrs-core/src/pipeline/chat_template.rs#L318-L322
fn remove_known_non_jinja2_tags(template: &str) -> String {
    template
        .replace("{% generation %}", "")
        .replace("{% endgeneration %}", "")
}

impl JinjaEnvironment {
    fn env(self) -> Environment<'static> {
        self.env
    }
}

impl Default for JinjaEnvironment {
    fn default() -> Self {
        let mut env = Environment::new();

        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);

        JinjaEnvironment { env }
    }
}

impl HfTokenizerConfigJsonFormatter {
    pub fn new(config: ChatTemplate, mixins: ContextMixins) -> anyhow::Result<Self> {
        let mut env = JinjaEnvironment::default().env();

        let chat_template = config.chat_template.as_ref().ok_or(anyhow::anyhow!(
            "chat_template field is required in the tokenizer_config.json file"
        ))?;

        // Safely handle chat templates that check the length of arguments like `tools` even
        // when `tools=None` when rendered through minijinja. For example:
        // https://github.com/vllm-project/vllm/blob/d95d0f4b985f28ea381e301490f9d479b34d8980/examples/tool_chat_template_hermes.jinja#L36
        env.add_filter("length", |value: Value| -> usize {
            use minijinja::value::ValueKind;
            match value.kind() {
                ValueKind::Undefined | ValueKind::None => 0,
                _ => value.len().unwrap_or(0),
            }
        });

        // add pycompat
        // todo: should we use this: minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

        env.add_filter("tojson", tojson);

        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);

        let mut supports_add_generation_prompt = None;

        match &chat_template.0 {
            Either::Left(x) => {
                if x.contains("add_generation_prompt") {
                    tracing::debug!(
                        "Chat template contains `add_generation_prompt` key. This model supports add_generation_prompt."
                    );
                    supports_add_generation_prompt = Some(true);
                }
                // Remove known non-standard tags before validation (they don't affect output)
                let template_cleaned = remove_known_non_jinja2_tags(x);
                env.add_template_owned("default", template_cleaned.clone())?;
                env.add_template_owned("tool_use", template_cleaned)?;
            }
            Either::Right(map) => {
                for t in map {
                    for (k, v) in t.iter() {
                        if v.contains("add_generation_prompt") {
                            match supports_add_generation_prompt {
                                Some(true) | None => {
                                    tracing::debug!(
                                        "Chat template contains `add_generation_prompt` key. This model supports add_generation_prompt."
                                    );
                                    supports_add_generation_prompt = Some(true);
                                }
                                Some(false) => {
                                    tracing::warn!(
                                        "Not all templates contain `add_generation_prompt` key. This model does not support add_generation_prompt."
                                    );
                                }
                            }
                        } else {
                            supports_add_generation_prompt = Some(false);
                        }
                        // Remove known non-standard tags before validation (they don't affect output)
                        let template_cleaned = remove_known_non_jinja2_tags(v);
                        env.add_template_owned(k.to_string(), template_cleaned)?;
                    }
                }
                if env.templates().count() == 0 {
                    anyhow::bail!(
                        "Chat template does not contain a `tool_use` or `default` key. Please ensure it contains at least a `default` key, although `tool_use` should be specified for using tools."
                    );
                }
            }
        }

        Ok(HfTokenizerConfigJsonFormatter {
            env,
            config,
            mixins: Arc::new(mixins),
            supports_add_generation_prompt: supports_add_generation_prompt.unwrap_or(false),
        })
    }
}

// impl JinjaEnvironment {
//     /// Renders the template with the provided messages.
//     /// This function reuses the pre-compiled template for efficiency.
//     pub fn render(&self, template_id: &str, ctx: &dyn erased_serde::Serialize) -> Result<String> {
//         let tmpl = self.env.get_template(template_id)?;
//         Ok(tmpl.render(ctx)?)
//     }

//     // fn apply_tool_template()
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_known_non_jinja2_tags() {
        let template =
            "USER: {{ message }} ASSISTANT: {% generation %}Reply here{% endgeneration %}";
        let result = remove_known_non_jinja2_tags(template);
        assert_eq!(result, "USER: {{ message }} ASSISTANT: Reply here");
    }

    #[test]
    fn test_remove_known_non_jinja2_tags_preserves_standard_tags() {
        let template = "{% for item in items %}{{ item }}{% endfor %}";
        let result = remove_known_non_jinja2_tags(template);
        assert_eq!(result, template);
    }

    #[test]
    fn test_remove_known_non_jinja2_tags_multiple() {
        let template = "Start {% generation %}Part 1{% endgeneration %} middle {% generation %}Part 2{% endgeneration %}";
        let result = remove_known_non_jinja2_tags(template);
        assert_eq!(result, "Start Part 1 middle Part 2");
    }
}
