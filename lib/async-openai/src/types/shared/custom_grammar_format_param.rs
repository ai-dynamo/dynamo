// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::OpenAIError;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum GrammarSyntax {
    Lark,
    #[default]
    Regex,
}

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq, Default, Builder)]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CustomGrammarFormatParam {
    /// The grammar definition.
    pub definition: String,
    /// The syntax of the grammar definition. One of `lark` or `regex`.
    pub syntax: GrammarSyntax,
}
