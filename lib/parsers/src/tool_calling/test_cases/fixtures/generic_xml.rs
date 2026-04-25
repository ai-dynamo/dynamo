// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{FixtureCase, ToolCallFixture};
use serde_json::Value;

/// Two function-name encodings seen across the generic-XML family.
#[derive(Clone, Copy)]
pub enum XmlFunctionForm {
    /// `<function=NAME>...</function>` with `<parameter=K>V</parameter>`.
    /// (qwen3_coder, nemotron_nano)
    EqualsName,
    /// `<invoke name="NAME">...</invoke>` with `<parameter name="K">V</parameter>`.
    /// (minimax_m2)
    NameAttr,
}

/// Family fixture for parsers whose grammar is generic XML — `<outer>`
/// wraps a single `<function-form>` element, which wraps one or more
/// `<parameter-form/>` elements. The two `XmlFunctionForm` variants
/// cover all current incarnations.
pub struct GenericXmlFixture {
    pub name: &'static str,
    pub outer_start: &'static str,
    pub outer_end: &'static str,
    pub function_form: XmlFunctionForm,
    pub recovery_reason: &'static str,
}

impl GenericXmlFixture {
    fn render_body(&self, function_name: &str, arguments: &Value) -> String {
        let mut params = String::new();
        if let Some(map) = arguments.as_object() {
            for (k, v) in map {
                let v_str = match v {
                    Value::String(s) => s.clone(),
                    _ => v.to_string(),
                };
                match self.function_form {
                    XmlFunctionForm::EqualsName => {
                        params.push_str(&format!("<parameter={k}>\n{v_str}\n</parameter>\n"));
                    }
                    XmlFunctionForm::NameAttr => {
                        params.push_str(&format!("<parameter name=\"{k}\">{v_str}</parameter>"));
                    }
                }
            }
        }
        match self.function_form {
            XmlFunctionForm::EqualsName => {
                format!("<function={function_name}>\n{params}</function>")
            }
            XmlFunctionForm::NameAttr => {
                format!("<invoke name=\"{function_name}\">{params}</invoke>")
            }
        }
    }
}

impl ToolCallFixture for GenericXmlFixture {
    fn parser_name(&self) -> &'static str {
        self.name
    }

    fn case_1_single_call(&self, function_name: &str, arguments: &Value) -> FixtureCase<String> {
        let body = self.render_body(function_name, arguments);
        FixtureCase::Sample(format!("{}\n{body}\n{}", self.outer_start, self.outer_end))
    }

    fn case_5_missing_end_token_recovery(
        &self,
        function_name: &str,
        arguments: &Value,
    ) -> FixtureCase<String> {
        let body = self.render_body(function_name, arguments);
        // Drop the outer-end token; inner function/parameter tags stay closed
        // so the call is logically complete, only the wrapper is missing.
        FixtureCase::KnownBroken {
            input: format!("{}\n{body}", self.outer_start),
            reason: self.recovery_reason,
        }
    }
}
