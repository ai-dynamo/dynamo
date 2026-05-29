// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// #[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
// #[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    Function,
}

// #[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
// #[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}

// #[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
// #[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: ToolCallType,
    pub function: CalledFunction,
}

/// Streaming (delta) variant of a parsed tool call.
///
/// Mirrors the field shape of a streaming tool-call chunk so consumers can map
/// it onto their own wire types without `dynamo-parsers` depending on those
/// types. Field semantics match the unary [`ToolCallResponse`]: `name` and
/// `arguments` live under `function`, and `index` orders parallel calls.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolCallResponseChunk {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tp: Option<ToolCallType>,
    pub function: Option<CalledFunctionStream>,
}

/// Streaming variant of [`CalledFunction`] where both fields are optional.
#[derive(Clone, Debug, serde::Serialize)]
pub struct CalledFunctionStream {
    pub name: Option<String>,
    pub arguments: Option<String>,
}
