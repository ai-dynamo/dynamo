// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::tokcfg::{ChatTemplate, raise_exception, strftime_now, tojson};
use super::{ContextMixins, HfTokenizerConfigJsonFormatter, JinjaEnvironment};
use either::Either;
use minijinja::{Environment, Value, context};
use serde_json::json;
use tracing;

/// Detects if a template requires content as arrays (multimodal) vs strings (text-only).
/// Returns true if the template only works with array format.
fn detect_content_array_usage(env: &Environment) -> bool {
    // Test with array format
    let array_msg = context! {
        messages => json!([{"role": "user", "content": [{"type": "text", "text": "template_test"}]}]),
        add_generation_prompt => false,
    };

    // Test with string format
    let string_msg = context! {
        messages => json!([{"role": "user", "content": "template_test"}]),
        add_generation_prompt => false,
    };

    let out_array = env
        .get_template("default")
        .and_then(|t| t.render(&array_msg))
        .unwrap_or_default();
    let out_string = env
        .get_template("default")
        .and_then(|t| t.render(&string_msg))
        .unwrap_or_default();

    // If array works but string doesn't, template requires arrays
    out_array.contains("template_test") && !out_string.contains("template_test")
}

/// Picks an image-placeholder template by sniffing the chat template source
/// for distinctive role/end markers.
///
/// Returned string is a format with `{n}` standing in for the 1-based image
/// index (numbered Phi-3 placeholders) or just a static placeholder (LLaVA).
/// Returns `None` when we don't know a flatten strategy for this template —
/// callers leave the mixed-content array untouched in that case.
///
/// The detection is intentionally narrow: only families whose chat templates
/// concatenate `message.content` with strings (and therefore can't render a
/// content array) need this. Qwen-VL / LLaVA-NeXT iterate `content` natively
/// and don't reach the flatten path at all.
fn detect_image_placeholder_template(env: &Environment) -> Option<&'static str> {
    let src = env
        .get_template("default")
        .ok()
        .map(|t| t.source().to_string())
        .unwrap_or_default();
    // Phi-3-vision template constructs `<|user|>` at runtime via
    // `'<|' + message['role'] + '|>'`, so the literal `<|user|>` never
    // appears in the source. The literals that ARE in the source are
    // `<|end|>` (end-of-turn) and `<|assistant|>` (generation prompt).
    if src.contains("<|end|>") && src.contains("<|assistant|>") {
        return Some("<|image_{n}|>");
    }
    // LLaVA-1.5: USER:/ASSISTANT: convention with `+ message['content']`.
    if src.contains("USER:") && src.contains("ASSISTANT:") {
        return Some("<image>");
    }
    None
}

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

/// Temporary compatibility shim for MiniJinja dotted integer lookup parsing.
///
/// MiniJinja supports terminal dotted integer lookup like `items.0`, but currently
/// parses mid-chain lookup like `m.content.0.type` as a float token (`0.`) and
/// fails before rendering. Jinja2 treats `a.0.b` like `a[0].b`, so rewrite only
/// `.digits.` inside Jinja expression/block regions.
///
/// TODO: remove after minijinja PR 900:
/// https://github.com/mitsuhiko/minijinja/pull/900
fn rewrite_midchain_dotted_integer_lookup(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut i = 0;

    while i < template.len() {
        let rest = &template[i..];

        if rest.starts_with("{#") {
            if let Some(end) = template[i + 2..].find("#}") {
                let end = i + 2 + end + 2;
                out.push_str(&template[i..end]);
                i = end;
            } else {
                out.push_str(rest);
                break;
            }
            continue;
        }

        if rest.starts_with("{%") {
            if let Some(tag_end) = find_jinja_tag_end(template, i + 2, "%}") {
                let tag_inner = &template[i + 2..tag_end - 2];
                if is_basic_block_tag(tag_inner, "raw") {
                    if let Some(raw_end) = find_raw_block_end(template, tag_end) {
                        out.push_str(&template[i..raw_end]);
                        i = raw_end;
                    } else {
                        out.push_str(rest);
                        break;
                    }
                    continue;
                }
            }
        }

        let close = if rest.starts_with("{{") {
            Some("}}")
        } else if rest.starts_with("{%") {
            Some("%}")
        } else {
            None
        };

        if let Some(close) = close {
            out.push_str(&template[i..i + 2]);
            i += 2;

            let mut quote = None;
            let mut escaped = false;
            while i < template.len() {
                if quote.is_none() && template[i..].starts_with(close) {
                    out.push_str(&template[i..i + close.len()]);
                    i += close.len();
                    break;
                }

                let ch = template[i..].chars().next().unwrap();

                if let Some(active_quote) = quote {
                    out.push(ch);
                    i += ch.len_utf8();
                    if escaped {
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == active_quote {
                        quote = None;
                    }
                    continue;
                }

                if ch == '\'' || ch == '"' {
                    quote = Some(ch);
                    out.push(ch);
                    i += ch.len_utf8();
                    continue;
                }

                if ch == '.' {
                    let previous_is_digit = i > 0 && template.as_bytes()[i - 1].is_ascii_digit();
                    if !previous_is_digit {
                        let digits_start = i + 1;
                        let mut digits_end = digits_start;
                        while digits_end < template.len() {
                            let next = template[digits_end..].chars().next().unwrap();
                            if next.is_ascii_digit() {
                                digits_end += next.len_utf8();
                            } else {
                                break;
                            }
                        }

                        if digits_end > digits_start
                            && digits_end < template.len()
                            && template[digits_end..].starts_with('.')
                        {
                            out.push('[');
                            out.push_str(&template[digits_start..digits_end]);
                            out.push(']');
                            i = digits_end;
                            continue;
                        }
                    }
                }

                out.push(ch);
                i += ch.len_utf8();
            }
            continue;
        }

        let ch = rest.chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn find_jinja_tag_end(template: &str, mut i: usize, close: &str) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;

    while i < template.len() {
        if quote.is_none() && template[i..].starts_with(close) {
            return Some(i + close.len());
        }

        let ch = template[i..].chars().next().unwrap();
        if let Some(active_quote) = quote {
            i += ch.len_utf8();
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active_quote {
                quote = None;
            }
            continue;
        }

        if ch == '\'' || ch == '"' {
            quote = Some(ch);
        }
        i += ch.len_utf8();
    }

    None
}

fn is_basic_block_tag(tag_inner: &str, tag_name: &str) -> bool {
    let mut trimmed = tag_inner.trim();
    if trimmed.starts_with('-') || trimmed.starts_with('+') {
        trimmed = trimmed[1..].trim_start();
    }

    let Some(rest) = trimmed.strip_prefix(tag_name) else {
        return false;
    };

    match rest.as_bytes().first() {
        Some(b) if b.is_ascii_alphanumeric() || *b == b'_' => return false,
        _ => {}
    }

    let rest = rest.trim();
    rest.is_empty() || rest == "-" || rest == "+"
}

fn find_raw_block_end(template: &str, mut i: usize) -> Option<usize> {
    while i < template.len() {
        let tag_start = i + template[i..].find("{%")?;
        let tag_end = find_jinja_tag_end(template, tag_start + 2, "%}")?;
        let tag_inner = &template[tag_start + 2..tag_end - 2];
        if is_basic_block_tag(tag_inner, "endraw") {
            return Some(tag_end);
        }
        i = tag_end;
    }

    None
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
    #[cfg(test)]
    pub fn new(config: ChatTemplate, mixins: ContextMixins) -> anyhow::Result<Self> {
        Self::with_options(config, mixins, true)
    }

    pub fn with_options(
        config: ChatTemplate,
        mixins: ContextMixins,
        exclude_tools_when_tool_choice_none: bool,
    ) -> anyhow::Result<Self> {
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
                let mut template_cleaned = remove_known_non_jinja2_tags(x);
                // Temporary: patch behavior of MiniJinja so that it recognizes
                // references like xxx.0.yyy.
                // TODO: remove after minijinja PR 900
                // (https://github.com/mitsuhiko/minijinja/pull/900)
                template_cleaned = rewrite_midchain_dotted_integer_lookup(&template_cleaned);
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
                        let mut template_cleaned = remove_known_non_jinja2_tags(v);
                        // Temporary: patch behavior of MiniJinja so that it recognizes
                        // references like xxx.0.yyy.
                        // TODO: remove after minijinja PR 900
                        // (https://github.com/mitsuhiko/minijinja/pull/900)
                        template_cleaned =
                            rewrite_midchain_dotted_integer_lookup(&template_cleaned);
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

        // Detect at model load time whether this template requires content arrays
        let requires_content_arrays = detect_content_array_usage(&env);

        // Pick a per-family placeholder for the mixed-content → string flatten
        // path. `None` is the safe default — the existing behavior in
        // `may_be_fix_msg_content` leaves mixed arrays untouched.
        let image_placeholder_template = if requires_content_arrays {
            None
        } else {
            detect_image_placeholder_template(&env)
        };

        // Detect if the template natively handles reasoning_content (e.g. Nemotron, Qwen3).
        // If so, we must NOT inject <think> blocks — the template does it itself.
        let template_handles_reasoning = env
            .templates()
            .any(|(_, tmpl)| tmpl.source().contains("reasoning_content"));

        // Detect if a given template branches on `tool_call.arguments is string` (Qwen3, Hermes).
        // Such templates render a JSON-string `arguments` field verbatim; if we pre-parse
        // it into an object, the `tojson` branch fires instead and emits compact JSON,
        // breaking byte-level append-only across multi-step tool-use turns. The check is
        // per-template (default vs tool_use) because in HF configs they can differ — and
        // because `arguments is string` only appears inside tool-call iteration, the flag
        // is naturally tied to the `tool_use` template in practice. It is also
        // tool_calls-specific: legacy `function_call.arguments` lives outside this branch
        // and must still be normalized.
        let template_handles_args_string = |name: &str| -> bool {
            env.templates()
                .find(|(n, _)| *n == name)
                .map(|(_, tmpl)| tmpl.source().contains("arguments is string"))
                .unwrap_or(false)
        };
        let default_template_handles_tool_calls_arguments_string =
            template_handles_args_string("default");
        let tool_use_template_handles_tool_calls_arguments_string =
            template_handles_args_string("tool_use");

        Ok(HfTokenizerConfigJsonFormatter {
            env,
            config,
            mixins: Arc::new(mixins),
            supports_add_generation_prompt: supports_add_generation_prompt.unwrap_or(false),
            requires_content_arrays,
            exclude_tools_when_tool_choice_none,
            template_handles_reasoning,
            image_placeholder_template,
            default_template_handles_tool_calls_arguments_string,
            tool_use_template_handles_tool_calls_arguments_string,
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

    #[test]
    fn test_rewrite_midchain_dotted_integer_lookup() {
        let template = "{{ m.content.0.type }}";
        let result = rewrite_midchain_dotted_integer_lookup(template);
        assert_eq!(result, "{{ m.content[0].type }}");
    }

    #[test]
    fn test_rewrite_midchain_dotted_integer_lookup_in_block() {
        let template = r#"{% if m.content.0.type == "tool_reference" %}x{% endif %}"#;
        let result = rewrite_midchain_dotted_integer_lookup(template);
        assert_eq!(
            result,
            r#"{% if m.content[0].type == "tool_reference" %}x{% endif %}"#
        );
    }

    #[test]
    fn test_rewrite_preserves_text_strings_comments_and_terminal_lookup() {
        let template = r#"text m.content.0.type {{ "m.content.0.type" }} {# m.content.0.type #} {{ items.0 }}"#;
        let result = rewrite_midchain_dotted_integer_lookup(template);
        assert_eq!(result, template);
    }

    #[test]
    fn test_rewrite_preserves_raw_blocks() {
        let template = r#"{% raw %}{{ m.content.0.type }}{% endraw %} {{ m.content.0.type }}"#;
        let result = rewrite_midchain_dotted_integer_lookup(template);
        assert_eq!(
            result,
            r#"{% raw %}{{ m.content.0.type }}{% endraw %} {{ m.content[0].type }}"#
        );
    }

    #[test]
    fn test_rewrite_preserves_raw_blocks_with_whitespace_control() {
        let template = r#"{%- raw -%}{{ m.content.0.type }}{%- endraw -%} {{ m.content.0.type }}"#;
        let result = rewrite_midchain_dotted_integer_lookup(template);
        assert_eq!(
            result,
            r#"{%- raw -%}{{ m.content.0.type }}{%- endraw -%} {{ m.content[0].type }}"#
        );
    }

    #[test]
    fn test_rewrite_preserves_numeric_literals() {
        let template = r#"{{ 1.5.10 }} {{ 3.14.something }} {{ m.content.0.type }}"#;
        let result = rewrite_midchain_dotted_integer_lookup(template);
        assert_eq!(
            result,
            r#"{{ 1.5.10 }} {{ 3.14.something }} {{ m.content[0].type }}"#
        );
    }
}
