// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::tokcfg::{ChatTemplate, raise_exception, strftime_now, tojson};
use super::{ContextMixins, HfTokenizerConfigJsonFormatter, JinjaEnvironment};
use either::Either;
use minijinja::{Environment, Value};
use tracing;

/// Replace non-standard Jinja2 block tags with placeholders
///
/// minijinja doesn't expose its tag list publicly - they're hardcoded in a private match statement
/// in the parser. This list is derived from minijinja v2.12.0's parser.rs implementation.
/// See: https://github.com/mitsuhiko/minijinja/blob/main/minijinja/src/compiler/parser.rs#L542
fn replace_non_standard_blocks(template: &str) -> String {
    use regex::Regex;

    // Standard Jinja2/minijinja tags (cannot be queried from minijinja API)
    let standard_keywords = [
        "for",
        "endfor",
        "if",
        "elif",
        "else",
        "endif",
        "block",
        "endblock",
        "extends",
        "include",
        "import",
        "from",
        "macro",
        "endmacro",
        "call",
        "endcall",
        "set",
        "endset",
        "with",
        "endwith",
        "filter",
        "endfilter",
        "autoescape",
        "endautoescape",
        "raw",
        "endraw",
        "do",
    ];

    let re = Regex::new(r"\{%\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*%\}").unwrap();
    let mut result = template.to_string();
    let mut replacements = Vec::new();

    for cap in re.captures_iter(template) {
        let full_match = cap.get(0).unwrap().as_str();
        let tag_name = cap.get(1).unwrap().as_str();

        if !standard_keywords.contains(&tag_name) {
            // Non-standard tag (e.g., vLLM's {% generation %}) - replace with placeholder
            let placeholder = format!("__JINJA_BLOCK_{}", tag_name.to_uppercase());
            replacements.push((full_match.to_string(), placeholder));
        }
    }

    for (original, placeholder) in replacements {
        result = result.replace(&original, &placeholder);
    }

    result
}

/// Detects whether a chat template requires message content as arrays (multimodal)
/// or accepts simple strings (standard text-only templates).
///
/// This function test-renders the template with both formats:
/// - Array format: `[{"type": "text", "text": "X"}]`
/// - String format: `"X"`
///
/// If the array format works but string format doesn't produce output,
/// the template requires arrays (e.g., llava, Qwen-VL multimodal templates).
fn detect_content_array_usage(env: &Environment) -> bool {
    use minijinja::context;
    use serde_json::json;

    // Test with array format
    let test_array = context! {
        messages => json!([{"role": "user", "content": [{"type": "text", "text": "X"}]}]),
        add_generation_prompt => false,
    };

    // Test with string format
    let test_string = context! {
        messages => json!([{"role": "user", "content": "X"}]),
        add_generation_prompt => false,
    };

    let out_array = env
        .get_template("default")
        .and_then(|t| t.render(&test_array))
        .unwrap_or_default();
    let out_string = env
        .get_template("default")
        .and_then(|t| t.render(&test_string))
        .unwrap_or_default();

    // If array works but string doesn't, template requires arrays
    out_array.contains("X") && !out_string.contains("X")
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
                // Replace non-standard Jinja2 block tags with placeholders for minijinja validation
                // Standard Jinja2/minijinja blocks: for, if, block, macro, call, filter, set, with, autoescape, trans
                // Any other {% tag %} blocks are likely backend-specific extensions (like vLLM's {% generation %})
                let template_for_validation = replace_non_standard_blocks(x);
                env.add_template_owned("default", template_for_validation.clone())?;
                env.add_template_owned("tool_use", template_for_validation)?;
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
                        // Replace non-standard Jinja2 block tags with placeholders for minijinja validation
                        let template_for_validation = replace_non_standard_blocks(v);
                        env.add_template_owned(k.to_string(), template_for_validation)?;
                    }
                }
                if env.templates().count() == 0 {
                    anyhow::bail!(
                        "Chat template does not contain a `tool_use` or `default` key. Please ensure it contains at least a `default` key, although `tool_use` should be specified for using tools."
                    );
                }
            }
        }

        // Detect at model load time whether this template requires content arrays
        let requires_content_arrays = detect_content_array_usage(&env);

        tracing::info!(
            "Template analysis: requires_content_arrays = {}",
            requires_content_arrays
        );

        Ok(HfTokenizerConfigJsonFormatter {
            env,
            config,
            mixins: Arc::new(mixins),
            supports_add_generation_prompt: supports_add_generation_prompt.unwrap_or(false),
            requires_content_arrays,
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
