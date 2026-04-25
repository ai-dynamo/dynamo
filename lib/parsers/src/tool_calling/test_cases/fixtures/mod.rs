// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! All concrete `ToolCallFixture` impls for the 18 registered parsers.
//!
//! Most parsers fall into one of three on-the-wire format families and
//! reuse a parametric family fixture; the rest have unique grammars and
//! get their own file. Adding a new parser is one of:
//!   - JSON-wrapped {start}{json}{end}: append a `JsonWrappedFixture` entry
//!   - Generic XML <outer><fn-form/>...</outer>: append a `GenericXmlFixture` entry
//!   - DSML: append a `DsmlFixture` entry
//!   - Anything else: a new file under `fixtures/` and a new entry below

use super::ToolCallFixture;

mod deepseek_v3;
mod deepseek_v3_1;
mod dsml;
mod generic_xml;
mod glm47;
mod harmony;
mod json_wrapped;
mod kimi_k2;
mod nemotron_deci;
mod pythonic;

use deepseek_v3::DeepseekV3Fixture;
use deepseek_v3_1::DeepseekV31Fixture;
use dsml::DsmlFixture;
use generic_xml::{GenericXmlFixture, XmlFunctionForm};
use glm47::Glm47Fixture;
use harmony::HarmonyFixture;
use json_wrapped::{JsonWrappedFixture, MissingEndBehavior};
use kimi_k2::KimiK2Fixture;
use nemotron_deci::NemotronDeciFixture;
use pythonic::PythonicFixture;

/// Every parser fixture, in stable alphabetical order by registry name.
/// The matrix report iterates this vec; new parsers go here.
pub fn all_fixtures() -> Vec<Box<dyn ToolCallFixture>> {
    vec![
        // --- default: bare JSON, no tokens ---
        Box::new(JsonWrappedFixture {
            name: "default",
            start: "",
            end: "",
            on_missing_end: MissingEndBehavior::NotApplicable {
                reason: "Default config has no start/end tokens; the bare JSON is the entire \
                         message. Truncation is malformed-args (a future case), not section-end recovery.",
            },
        }),
        // --- DeepSeek V3 with code-fence body ---
        Box::new(DeepseekV3Fixture),
        // --- DeepSeek V3.1 with sep-style body ---
        Box::new(DeepseekV31Fixture),
        // --- DSML family ---
        Box::new(DsmlFixture {
            name: "deepseek_v3_2",
            outer_tag: "function_calls",
        }),
        Box::new(DsmlFixture {
            name: "deepseek_v4",
            outer_tag: "tool_calls",
        }),
        // --- GLM 4.7 (own grammar) ---
        Box::new(Glm47Fixture),
        // --- Harmony (channel/recipient stream) ---
        Box::new(HarmonyFixture),
        // --- Hermes: JSON-wrapped <tool_call>...</tool_call> ---
        Box::new(JsonWrappedFixture {
            name: "hermes",
            start: "<tool_call>",
            end: "\n</tool_call>",
            on_missing_end: MissingEndBehavior::Drops {
                reason: "hermes has no missing-end recovery yet; follow-up to generalize PR #8208.",
            },
        }),
        // --- Jamba: JSON-wrapped <tool_calls>...</tool_calls> ---
        Box::new(JsonWrappedFixture {
            name: "jamba",
            start: "<tool_calls>",
            end: "</tool_calls>",
            on_missing_end: MissingEndBehavior::Drops {
                reason: "jamba has no missing-end recovery yet; follow-up to generalize PR #8208.",
            },
        }),
        // --- Kimi K2 (special-token format) ---
        Box::new(KimiK2Fixture),
        // --- Llama 3 JSON: <|python_tag|>{json} (no end token) ---
        Box::new(JsonWrappedFixture {
            name: "llama3_json",
            start: "<|python_tag|>",
            end: "",
            on_missing_end: MissingEndBehavior::NotApplicable {
                reason: "llama3_json has no end token (config end-token is empty string); \
                         the start sentinel opens a JSON object and EOF terminates it.",
            },
        }),
        // --- MiniMax M2: generic XML with <minimax:tool_call> wrapper ---
        Box::new(GenericXmlFixture {
            name: "minimax_m2",
            outer_start: "<minimax:tool_call>",
            outer_end: "</minimax:tool_call>",
            function_form: XmlFunctionForm::NameAttr,
            recovery_reason: "minimax_m2 has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }),
        // --- Mistral: JSON-wrapped [TOOL_CALLS]...[/TOOL_CALLS] (recovers) ---
        Box::new(JsonWrappedFixture {
            name: "mistral",
            start: "[TOOL_CALLS]",
            end: "[/TOOL_CALLS]",
            // mistral's JsonParserConfig declares both `[/TOOL_CALLS]` and ``
            // as end tokens, so dropping the explicit closer still parses.
            on_missing_end: MissingEndBehavior::Recovers,
        }),
        // --- Nemotron Deci: <TOOLCALL>[...]</TOOLCALL> (own format, JSON array) ---
        Box::new(NemotronDeciFixture),
        // --- Nemotron Nano: registry-aliased to qwen3_coder XML ---
        Box::new(GenericXmlFixture {
            name: "nemotron_nano",
            outer_start: "<tool_call>",
            outer_end: "</tool_call>",
            function_form: XmlFunctionForm::EqualsName,
            recovery_reason: "nemotron_nano (qwen3_coder XML) has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }),
        // --- Phi-4: functools{json} (no end token) ---
        Box::new(JsonWrappedFixture {
            name: "phi4",
            start: "functools",
            end: "",
            on_missing_end: MissingEndBehavior::NotApplicable {
                reason: "phi4 has no end token (config end-token is empty string); the \
                         `functools` sentinel opens a JSON object and EOF terminates it.",
            },
        }),
        // --- Pythonic: [name(k=v)] ---
        Box::new(PythonicFixture),
        // --- Qwen3-Coder: generic XML, default config ---
        Box::new(GenericXmlFixture {
            name: "qwen3_coder",
            outer_start: "<tool_call>",
            outer_end: "</tool_call>",
            function_form: XmlFunctionForm::EqualsName,
            recovery_reason: "qwen3_coder XML parser has no missing-end recovery yet; follow-up to generalize PR #8208.",
        }),
    ]
}
