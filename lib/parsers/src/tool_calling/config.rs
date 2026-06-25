// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::json::JsonParserType;
use super::structural_tag::format::JsonSchemaStyle;
use super::structural_tag::{
    DsmlToolCallsConfig, StructuralTagBuilder, TOOL_NAME_PLACEHOLDER, TriggeredTagsConfig,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JsonParserConfig {
    /// Start token for individual tool calls (e.g., `<TOOLCALL>`)
    pub tool_call_start_tokens: Vec<String>,
    /// End token for individual tool calls (e.g., `</TOOLCALL>`)
    pub tool_call_end_tokens: Vec<String>,
    /// Separator tokens between function name and arguments
    /// (e.g., "<｜tool▁sep｜>" for DeepSeek v3.1)
    /// Used by some models to separate function name from arguments
    pub tool_call_separator_tokens: Vec<String>,
    /// The key for the function name in the tool call
    /// i.e. `{"name": "function", "arguments": {...}}` it would be
    /// "name"
    pub function_name_keys: Vec<String>,
    /// The key for the arguments in the tool call
    /// i.e. `{"name": "function", "arguments": {...}}` it would be
    /// "arguments"
    pub arguments_keys: Vec<String>,

    /// The type of JSON parser to use
    #[serde(default)]
    pub parser_type: JsonParserType,

    /// Parse input as bare JSON (a `{...}` object or `[...]` array) with no
    /// wrapping markers. Intended for guided-decoding paths where the backend
    /// emits a raw JSON shape. When true, `tool_call_start_tokens` /
    /// `tool_call_end_tokens` are ignored.
    #[serde(default)]
    pub bare_json_mode: bool,

    /// Allow recovery when the outer end-token is missing (max_tokens / EOS
    /// truncation). Streaming jails MUST leave this `false` — otherwise the
    /// parser claims "complete tool call" before the end-token has actually
    /// arrived and `should_exit_jail_early` fires too aggressively. Finalize
    /// / aggregate paths set it to `true` so a real but unterminated call
    /// isn't silently dropped.
    #[serde(default)]
    pub allow_eof_recovery: bool,

    /// Strict recovery: never leak wrapper markers (e.g. `<TOOLCALL>` /
    /// `</TOOLCALL>`) into `normal_text`. When the normal parse path produces
    /// no tool call, strip all configured start/end tokens from the raw text
    /// and retry a strict parse of the remaining payload. If it yields one or
    /// more tool calls, surface them (the model just emitted malformed framing,
    /// e.g. an orphan close tag); otherwise drop the content entirely (empty
    /// `normal_text`). Either way the wrapper markers never reach the user, and
    /// a `tracing::warn!` records what was recovered or dropped so the loss is
    /// debuggable. Default `false` keeps every other JSON family's existing
    /// impl-defined recovery (which may leave raw text in `normal_text`)
    /// unchanged. Only `nemotron_deci` opts in today.
    ///
    /// Only takes effect when `allow_eof_recovery` is also true (finalize /
    /// non-streaming aggregate paths). On a mid-stream chunk it would otherwise
    /// claim a complete call before the end token arrives and strand the
    /// trailing marker as leaked text on the next chunk.
    #[serde(default)]
    pub strip_markup_on_recovery: bool,
}

impl Default for JsonParserConfig {
    fn default() -> Self {
        Self {
            tool_call_start_tokens: vec!["<TOOLCALL>".to_string(), "<|python_tag|>".to_string()],
            tool_call_end_tokens: vec!["</TOOLCALL>".to_string(), "".to_string()],
            tool_call_separator_tokens: vec![],
            function_name_keys: vec!["name".to_string()],
            arguments_keys: vec!["arguments".to_string(), "parameters".to_string()],
            parser_type: JsonParserType::Basic,
            bare_json_mode: false,
            allow_eof_recovery: false,
            strip_markup_on_recovery: false,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct XmlParserConfig {
    /// Start token for individual tool calls (e.g., "<tool_call>")
    pub tool_call_start_token: String,
    /// End token for individual tool calls (e.g., `</tool_call>`)
    pub tool_call_end_token: String,
    /// Start token for function name (e.g., `<function=`)
    pub function_start_token: String,
    /// End token for function (e.g., `</function>`)
    pub function_end_token: String,
    /// Start token for parameter (e.g., `<parameter=`)
    pub parameter_start_token: String,
    /// End token for parameter (e.g., `</parameter>`)
    pub parameter_end_token: String,

    /// See [`JsonParserConfig::allow_eof_recovery`]. Streaming jails MUST
    /// leave this `false`.
    #[serde(default)]
    pub allow_eof_recovery: bool,

    /// When true, the function- and parameter-regex omit the `|$` end-of-block
    /// fallback (so missing `</function>` / `</parameter>` causes the match to
    /// fail rather than being silently recovered). Finalize recovery may still
    /// accept a missing outer wrapper end marker if the inner blocks are
    /// complete. Used by families whose official reference parser is
    /// strict-match (e.g. MiniMax-M2 — see
    /// https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/tool_calling_guide.md).
    #[serde(default)]
    pub strict_match: bool,

    /// When true, if `function_start_token` is absent anywhere in the input,
    /// short-circuit `try_tool_call_parse_xml` and return
    /// `(calls=[], normal_text=<input>)`. Matches the early-return passthrough
    /// in Qwen3-Coder's official reference parser
    /// (https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/qwen3coder_tool_parser.py).
    #[serde(default)]
    pub passthrough_when_no_function: bool,

    /// When true, if `tool_call_start_token` is absent but `function_start_
    /// token` is present, parse the entire input as a single tool-call block
    /// (back-off strategy). Used by XML families that can recover a complete
    /// function/invoke body even when the outer wrapper opener is missing.
    /// Independent of `passthrough_when_no_function` (different trigger: this
    /// one fires when function tags exist but the outer wrapper does not).
    #[serde(default)]
    pub backoff_when_no_wrapper: bool,
}

impl Default for XmlParserConfig {
    fn default() -> Self {
        Self {
            tool_call_start_token: "<tool_call>".to_string(),
            tool_call_end_token: "</tool_call>".to_string(),
            function_start_token: "<function=".to_string(),
            function_end_token: "</function>".to_string(),
            parameter_start_token: "<parameter=".to_string(),
            parameter_end_token: "</parameter>".to_string(),
            allow_eof_recovery: false,
            strict_match: false,
            passthrough_when_no_function: false,
            backoff_when_no_wrapper: false,
        }
    }
}

impl XmlParserConfig {
    /// Returns true when the chunk lacks the outer `<tool_call>` wrapper but
    /// contains `<function=...>`, and the family opts into back-off parsing
    /// (qwen3_coder, nemotron_nano). In this mode the function-level tokens
    /// act as the tool-call boundary for both start detection and end-position
    /// search, mirroring the wrapped path's behavior so streaming and batch
    /// agree on what counts as a tool call.
    pub fn is_bare_function_mode(&self, chunk: &str) -> bool {
        self.backoff_when_no_wrapper
            && !chunk.contains(self.tool_call_start_token.as_str())
            && chunk.contains(self.function_start_token.as_str())
    }
}

/// Configuration for DSML-style tool call parser (DeepSeek V3.2+)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsmlParserConfig {
    /// Start token for the DSML block (e.g., "<｜DSML｜function_calls>" or "<｜DSML｜tool_calls>")
    #[serde(alias = "function_calls_start")]
    pub block_start: String,
    /// End token for the DSML block (e.g., "</｜DSML｜function_calls>" or "</｜DSML｜tool_calls>")
    #[serde(alias = "function_calls_end")]
    pub block_end: String,
    /// Start prefix for invoke (e.g., "<｜DSML｜invoke name=")
    pub invoke_start_prefix: String,
    /// End token for invoke (e.g., "</｜DSML｜invoke>")
    pub invoke_end: String,
    /// Start prefix for parameter (e.g., "<｜DSML｜parameter name=")
    pub parameter_prefix: String,
    /// End token for parameter (e.g., "</｜DSML｜parameter>")
    pub parameter_end: String,

    /// See [`JsonParserConfig::allow_eof_recovery`]. Streaming jails MUST
    /// leave this `false`.
    #[serde(default)]
    pub allow_eof_recovery: bool,
}

impl Default for DsmlParserConfig {
    fn default() -> Self {
        Self {
            block_start: "<｜DSML｜function_calls>".to_string(),
            block_end: "</｜DSML｜function_calls>".to_string(),
            invoke_start_prefix: "<｜DSML｜invoke name=".to_string(),
            invoke_end: "</｜DSML｜invoke>".to_string(),
            parameter_prefix: "<｜DSML｜parameter name=".to_string(),
            parameter_end: "</｜DSML｜parameter>".to_string(),
            allow_eof_recovery: false,
        }
    }
}

/// Configuration for GLM-4.7 style tool call parser
/// Format: <tool_call>function_name<arg_key>param</arg_key><arg_value>value</arg_value></tool_call>
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Glm47ParserConfig {
    /// Start token for tool call block (e.g., "<tool_call>")
    pub tool_call_start: String,
    /// End token for tool call block (e.g., "</tool_call>")
    pub tool_call_end: String,
    /// Start token for argument key (e.g., "<arg_key>")
    pub arg_key_start: String,
    /// End token for argument key (e.g., "</arg_key>")
    pub arg_key_end: String,
    /// Start token for argument value (e.g., "<arg_value>")
    pub arg_value_start: String,
    /// End token for argument value (e.g., "</arg_value>")
    pub arg_value_end: String,

    /// See [`JsonParserConfig::allow_eof_recovery`]. Streaming jails MUST
    /// leave this `false`.
    #[serde(default)]
    pub allow_eof_recovery: bool,
}

impl Default for Glm47ParserConfig {
    fn default() -> Self {
        Self {
            tool_call_start: "<tool_call>".to_string(),
            tool_call_end: "</tool_call>".to_string(),
            arg_key_start: "<arg_key>".to_string(),
            arg_key_end: "</arg_key>".to_string(),
            arg_value_start: "<arg_value>".to_string(),
            arg_value_end: "</arg_value>".to_string(),
            allow_eof_recovery: false,
        }
    }
}

/// Configuration for Kimi K2 tool call parser
///
/// Format:
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// The model may emit either plural or singular forms of section tokens
/// (e.g., `<|tool_calls_section_begin|>` or `<|tool_call_section_begin|>`).
/// Both forms are supported via the `section_start_variants` and `section_end_variants` fields.
/// See vllm `kimi_k2_tool_parser.py` for reference.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KimiK2ParserConfig {
    /// Primary start token for the tool calls section
    pub section_start: String,
    /// Primary end token for the tool calls section
    pub section_end: String,
    /// All recognized start tokens for the tool calls section (includes singular variants)
    pub section_start_variants: Vec<String>,
    /// All recognized end tokens for the tool calls section (includes singular variants)
    pub section_end_variants: Vec<String>,
    /// Start token for an individual tool call (e.g., "<|tool_call_begin|>")
    pub call_start: String,
    /// End token for an individual tool call (e.g., "<|tool_call_end|>")
    pub call_end: String,
    /// Token separating function ID from JSON arguments (e.g., "<|tool_call_argument_begin|>")
    pub argument_begin: String,
}

impl Default for KimiK2ParserConfig {
    fn default() -> Self {
        Self {
            section_start: "<|tool_calls_section_begin|>".to_string(),
            section_end: "<|tool_calls_section_end|>".to_string(),
            section_start_variants: vec![
                "<|tool_calls_section_begin|>".to_string(),
                "<|tool_call_section_begin|>".to_string(),
            ],
            section_end_variants: vec![
                "<|tool_calls_section_end|>".to_string(),
                "<|tool_call_section_end|>".to_string(),
            ],
            call_start: "<|tool_call_begin|>".to_string(),
            call_end: "<|tool_call_end|>".to_string(),
            argument_begin: "<|tool_call_argument_begin|>".to_string(),
        }
    }
}

/// Parser-specific configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParserConfig {
    Json(JsonParserConfig),
    Xml(XmlParserConfig),
    Pythonic,
    Harmony(JsonParserConfig),
    Typescript,
    Dsml(DsmlParserConfig),
    KimiK2(KimiK2ParserConfig),
    Glm47(Glm47ParserConfig),
    /// Gemma 4 uses a custom non-JSON grammar with bare keys, `<|"|>` string
    /// delimiters, and fixed `<|tool_call>...<tool_call|>` markers. No
    /// configuration is required at runtime — markers are not user-tunable.
    Gemma4,
}

impl ParserConfig {
    /// Get the tool call start tokens for this parser configuration
    /// Returns a vector of start tokens that indicate the beginning of a tool call
    pub fn tool_call_start_tokens(&self) -> Vec<String> {
        match self {
            ParserConfig::Json(config) => config.tool_call_start_tokens.clone(),
            ParserConfig::Harmony(config) => config.tool_call_start_tokens.clone(),
            ParserConfig::Xml(config) => {
                let mut tokens = vec![config.tool_call_start_token.clone()];
                if config.backoff_when_no_wrapper {
                    tokens.push(config.function_start_token.clone());
                }
                tokens
            }
            ParserConfig::Pythonic => vec![],
            ParserConfig::Typescript => vec![],
            ParserConfig::Dsml(config) => {
                vec![
                    config.block_start.clone(),
                    config.invoke_start_prefix.clone(),
                ]
            }
            ParserConfig::Glm47(config) => vec![config.tool_call_start.clone()],
            ParserConfig::KimiK2(config) => {
                let mut tokens = config.section_start_variants.clone();
                tokens.push(config.call_start.clone());
                tokens
            }
            ParserConfig::Gemma4 => {
                vec![crate::tool_calling::gemma4::TOOL_CALL_START.to_string()]
            }
        }
    }

    /// Get the tool call end tokens for this parser configuration
    /// Returns a vector of end tokens that indicate the end of a tool call
    pub fn tool_call_end_tokens(&self) -> Vec<String> {
        match self {
            ParserConfig::Json(config) => config.tool_call_end_tokens.clone(),
            ParserConfig::Harmony(config) => config.tool_call_end_tokens.clone(),
            ParserConfig::Xml(config) => vec![config.tool_call_end_token.clone()],
            ParserConfig::Pythonic => vec![],
            ParserConfig::Typescript => vec![],
            ParserConfig::Dsml(config) => vec![config.block_end.clone()],
            ParserConfig::Glm47(config) => vec![config.tool_call_end.clone()],
            ParserConfig::KimiK2(config) => config.section_end_variants.clone(),
            ParserConfig::Gemma4 => vec![crate::tool_calling::gemma4::TOOL_CALL_END.to_string()],
        }
    }
}

/// Configuration for parsing tool calls with different formats
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallConfig {
    /// Parser-specific configuration.
    pub parser_config: ParserConfig,

    /// Structural tag builder for xgrammar guided decoding.
    ///
    /// When `Some`, the Dynamo preprocessor can generate structural tags that
    /// constrain the backend into producing model-native tool call format.
    #[serde(skip)]
    pub structural_tag_builder: Option<StructuralTagBuilder>,
}

impl Default for ToolCallConfig {
    fn default() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig::default()),
            structural_tag_builder: None,
        }
    }
}

impl ToolCallConfig {
    /// Default configuration for hermes tool calls
    /// <tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n</tool_call>
    pub fn hermes() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<tool_call>".to_string()],
                tool_call_end_tokens: vec!["</tool_call>".to_string()],
                ..Default::default()
            }),
            structural_tag_builder: Some(StructuralTagBuilder::TriggeredTags(
                TriggeredTagsConfig {
                    begin_template: format!(
                        "<tool_call>\n{{\"name\": \"{}\", \"arguments\": ",
                        TOOL_NAME_PLACEHOLDER
                    ),
                    end_template: "}\n</tool_call>".to_string(),
                    triggers: vec!["<tool_call>".to_string()],
                    content_style: JsonSchemaStyle::Json,
                    tool_call_ban_tokens: vec!["<tool_call>".to_string()],
                    reasoning_end: Some("</think>".to_string()),
                },
            )),
        }
    }

    /// Default configuration for nemotron tool calls
    /// <TOOLCALL>[{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}]</TOOLCALL>
    pub fn nemotron_deci() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<TOOLCALL>".to_string()],
                tool_call_end_tokens: vec!["</TOOLCALL>".to_string()],
                // Nemotron Ultra/Deci is an agent target where leaking a bare
                // <TOOLCALL> envelope into message.content breaks OpenAI-shaped
                // clients (they read tool_calls, never the content body). Strip
                // the markers on recovery and either surface the call or drop it.
                strip_markup_on_recovery: true,
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    pub fn llama3_json() -> Self {
        // <|python_tag|>{ "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"} }
        // or { "name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"} }
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<|python_tag|>".to_string()],
                tool_call_end_tokens: vec!["".to_string()],
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    pub fn mistral() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["[TOOL_CALLS]".to_string()],
                tool_call_end_tokens: vec!["[/TOOL_CALLS]".to_string(), "".to_string()],
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    pub fn phi4() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["functools".to_string()],
                tool_call_end_tokens: vec!["".to_string()],
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    pub fn pythonic() -> Self {
        Self {
            parser_config: ParserConfig::Pythonic,
            structural_tag_builder: None,
        }
    }

    pub fn harmony() -> Self {
        Self {
            parser_config: ParserConfig::Harmony(JsonParserConfig {
                tool_call_start_tokens: vec![
                    "<|start|>assistant<|channel|>commentary".to_string(),
                    "<|channel|>commentary".to_string(),
                ],
                tool_call_end_tokens: vec!["<|call|>".to_string()],
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    pub fn deepseek_v3_1() -> Self {
        // The whole tool calls block is wrapped between
        // <｜tool▁calls▁begin｜> ... <｜tool▁calls▁end｜>
        // regardless of number of tool calls. For external use of this
        // config, we want them to only be operating on the whole block,
        // so the tool parser can properly consume all tool call tokens.
        // https://huggingface.co/deepseek-ai/DeepSeek-V3.1#toolcall
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec![
                    "<｜tool▁calls▁begin｜>".to_string(),
                    // "<｜tool▁call▁begin｜>".to_string(),
                ],
                tool_call_end_tokens: vec![
                    "<｜tool▁calls▁end｜>".to_string(),
                    // "<｜tool▁call▁end｜>".to_string(),
                ],
                tool_call_separator_tokens: vec!["<｜tool▁sep｜>".to_string()],
                parser_type: JsonParserType::DeepseekV31,
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    pub fn deepseek_v3() -> Self {
        // DeepSeek V3 format:
        // <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{type}<｜tool▁sep｜>{function_name}\n```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
        // There are some differences between DeepSeek V3 and DeepSeek V3.1
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<｜tool▁calls▁begin｜>".to_string()],
                tool_call_end_tokens: vec!["<｜tool▁calls▁end｜>".to_string()],
                tool_call_separator_tokens: vec!["<｜tool▁sep｜>".to_string()],
                parser_type: JsonParserType::DeepseekV3,
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    pub fn qwen3_coder() -> Self {
        // <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        // Behavior aligned with Qwen3-Coder's official reference parser
        // (huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/.../qwen3coder_tool_parser.py):
        //  - passthrough: when no `<function=` in input, return raw input as content
        //  - back-off: when `<function=` present but no `<tool_call>` wrapper,
        //              parse the entire input as a tool block
        Self {
            parser_config: ParserConfig::Xml(XmlParserConfig {
                passthrough_when_no_function: true,
                backoff_when_no_wrapper: true,
                ..XmlParserConfig::default()
            }),
            structural_tag_builder: Some(StructuralTagBuilder::TriggeredTags(
                TriggeredTagsConfig {
                    begin_template: format!("<tool_call>\n<function={}>\n", TOOL_NAME_PLACEHOLDER),
                    end_template: "\n</function>\n</tool_call>".to_string(),
                    triggers: vec!["<tool_call>\n<function=".to_string()],
                    content_style: JsonSchemaStyle::QwenXml,
                    tool_call_ban_tokens: vec!["<tool_call>".to_string()],
                    reasoning_end: Some("</think>".to_string()),
                },
            )),
        }
    }

    pub fn jamba() -> Self {
        Self {
            parser_config: ParserConfig::Json(JsonParserConfig {
                tool_call_start_tokens: vec!["<tool_calls>".to_string()],
                tool_call_end_tokens: vec!["</tool_calls>".to_string()],
                ..Default::default()
            }),
            structural_tag_builder: None,
        }
    }

    fn deepseek_dsml(block_name: &str) -> Self {
        let dsml_config = DsmlParserConfig {
            block_start: format!("<｜DSML｜{}>", block_name),
            block_end: format!("</｜DSML｜{}>", block_name),
            ..Default::default()
        };
        let structural_tag = StructuralTagBuilder::DsmlToolCalls(DsmlToolCallsConfig {
            trigger: dsml_config.block_start.clone(),
            block_begin: format!("{}\n", dsml_config.block_start),
            block_end: dsml_config.block_end.clone(),
            invoke_begin_template: format!(
                "{}\"{}\">\n",
                dsml_config.invoke_start_prefix, TOOL_NAME_PLACEHOLDER
            ),
            // Keep the newline in invoke_end and separator empty so the model
            // can either emit another invoke or close the DSML block.
            invoke_end: format!("{}\n", dsml_config.invoke_end),
            separator: "".to_string(),
            // The DSML opening is several tokens (`<`, `｜DSML｜`, ...), so `tool_choice=none`
            // cannot suppress the entire prefix, so we ban the `｜DSML｜` token.
            // The model may still begin a tool block and hurt plain-text quality; prefer omitting tools
            // from the prompt (default `--exclude-tools-when-tool-choice-none`) when that matters.
            tool_call_ban_tokens: vec!["｜DSML｜".to_string()],
            reasoning_end: Some("</think>".to_string()),
        });
        Self {
            parser_config: ParserConfig::Dsml(dsml_config),
            structural_tag_builder: Some(structural_tag),
        }
    }

    pub fn deepseek_v3_2() -> Self {
        // DeepSeek V3.2 format (DSML):
        // <｜DSML｜function_calls>
        // <｜DSML｜invoke name="function_name">
        // <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
        // </｜DSML｜invoke>
        // </｜DSML｜function_calls>
        Self::deepseek_dsml("function_calls")
    }

    pub fn deepseek_v4() -> Self {
        // DeepSeek V4 format (DSML):
        // <｜DSML｜tool_calls>
        // <｜DSML｜invoke name="function_name">
        // <｜DSML｜parameter name="param_name" string="true|false">value</｜DSML｜parameter>
        // </｜DSML｜invoke>
        // </｜DSML｜tool_calls>
        Self::deepseek_dsml("tool_calls")
    }

    pub fn minimax_m2() -> Self {
        // MiniMax-M2.1 format:
        // <minimax:tool_call>
        // <invoke name="function_name">
        // <parameter name="param_name">value</parameter>
        // </invoke>
        // </minimax:tool_call>
        // Reference: https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/main/docs/tool_calling_guide.md
        //
        // MiniMax uses strict inner invoke/parameter fences. Dynamo still
        // recovers complete inner invokes when only the outer wrapper is
        // missing or truncated so valid tool calls do not disappear.
        Self {
            parser_config: ParserConfig::Xml(XmlParserConfig {
                tool_call_start_token: "<minimax:tool_call>".to_string(),
                tool_call_end_token: "</minimax:tool_call>".to_string(),
                function_start_token: "<invoke name=".to_string(),
                function_end_token: "</invoke>".to_string(),
                parameter_start_token: "<parameter name=".to_string(),
                parameter_end_token: "</parameter>".to_string(),
                allow_eof_recovery: false,
                strict_match: true,
                passthrough_when_no_function: false,
                backoff_when_no_wrapper: true,
            }),
            structural_tag_builder: None,
        }
    }

    pub fn glm47() -> Self {
        // GLM-4.7 format:
        // <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
        // Reference: https://huggingface.co/zai-org/GLM-4.7/blob/main/chat_template.jinja
        Self {
            parser_config: ParserConfig::Glm47(Glm47ParserConfig::default()),
            structural_tag_builder: None,
        }
    }

    pub fn kimi_k2() -> Self {
        // Kimi K2 format:
        // <|tool_calls_section_begin|>
        // <|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
        // <|tool_calls_section_end|>
        // Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
        Self {
            parser_config: ParserConfig::KimiK2(KimiK2ParserConfig::default()),
            structural_tag_builder: None,
        }
    }

    /// Gemma 4 tool-call format (custom non-JSON grammar):
    ///
    /// ```text
    /// <|tool_call>call:func_name{location:<|"|>Tokyo<|"|>,count:42}<tool_call|>
    /// ```
    ///
    /// Bare unquoted keys, `<|"|>`-delimited strings, supports nested objects /
    /// arrays. Multiple tool calls are concatenated without separators.
    pub fn gemma4() -> Self {
        Self {
            parser_config: ParserConfig::Gemma4,
            structural_tag_builder: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsml_config_deserializes_legacy_function_calls_aliases() {
        let legacy = serde_json::json!({
            "function_calls_start": "<｜DSML｜function_calls>",
            "function_calls_end": "</｜DSML｜function_calls>",
            "invoke_start_prefix": "<｜DSML｜invoke name=",
            "invoke_end": "</｜DSML｜invoke>",
            "parameter_prefix": "<｜DSML｜parameter name=",
            "parameter_end": "</｜DSML｜parameter>",
        });
        let cfg: DsmlParserConfig = serde_json::from_value(legacy).unwrap();
        assert_eq!(cfg.block_start, "<｜DSML｜function_calls>");
        assert_eq!(cfg.block_end, "</｜DSML｜function_calls>");
        assert_eq!(cfg.invoke_start_prefix, "<｜DSML｜invoke name=");
    }

    #[test]
    fn deepseek_dsml_factory_produces_expected_block_tokens() {
        let v3_2 = ToolCallConfig::deepseek_v3_2();
        let v4 = ToolCallConfig::deepseek_v4();
        let v3_2_cfg = match v3_2.parser_config {
            ParserConfig::Dsml(c) => c,
            _ => panic!("expected Dsml variant for v3_2"),
        };
        let v4_cfg = match v4.parser_config {
            ParserConfig::Dsml(c) => c,
            _ => panic!("expected Dsml variant for v4"),
        };
        assert_eq!(v3_2_cfg.block_start, "<｜DSML｜function_calls>");
        assert_eq!(v3_2_cfg.block_end, "</｜DSML｜function_calls>");
        assert_eq!(v4_cfg.block_start, "<｜DSML｜tool_calls>");
        assert_eq!(v4_cfg.block_end, "</｜DSML｜tool_calls>");
    }
}
