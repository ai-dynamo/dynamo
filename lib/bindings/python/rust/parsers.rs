// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::reasoning::get_available_reasoning_parsers;
use dynamo_parsers::tool_calling::parsers::{detect_and_parse_tool_call, get_available_tool_parsers};
use dynamo_parsers::ToolCallResponse;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Get list of available  parser names
#[pyfunction]
pub fn get_tool_parser_names() -> Vec<&'static str> {
    get_available_tool_parsers()
}

/// Get list of available reasoning parser names
#[pyfunction]
pub fn get_reasoning_parser_names() -> Vec<&'static str> {
    get_available_reasoning_parsers()
}

/// Parse tool calls from message content using the specified parser
/// 
/// Args:
///     message: The message content to parse (string containing tool calls)
///     parser_name: Optional parser name (e.g., "hermes", "llama3_json", etc.)
///                  If None, uses default parser
/// 
/// Returns:
///     A tuple of (tool_calls, normal_text) where:
///     - tool_calls: List of dicts with keys: id, type, function (dict with name, arguments)
///     - normal_text: Optional string of non-tool-call content
#[pyfunction]
#[pyo3(signature = (message, parser_name=None))]
fn parse_tool_calls_py<'py>(
    py: Python<'py>,
    message: &str,
    parser_name: Option<&str>,
) -> PyResult<(Vec<Bound<'py, PyDict>>, Option<String>)> {
    // Call the async Rust parser in a blocking way
    let result = pyo3_async_runtimes::tokio::get_runtime()
        .block_on(async { detect_and_parse_tool_call(message, parser_name).await });

    match result {
        Ok((tool_calls, normal_text)) => {
            // Convert Vec<ToolCallResponse> to Vec<PyDict>
            let py_tool_calls: Vec<Bound<'py, PyDict>> = tool_calls
                .into_iter()
                .map(|tc| tool_call_to_dict(py, tc))
                .collect::<PyResult<Vec<_>>>()?;
            Ok((py_tool_calls, normal_text))
        }
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to parse tool calls: {}",
            e
        ))),
    }
}

/// Convert a ToolCallResponse to a Python dict
fn tool_call_to_dict<'py>(py: Python<'py>, tc: ToolCallResponse) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", tc.id)?;
    
    // Convert ToolCallType enum to string
    let type_str = match tc.tp {
        dynamo_parsers::ToolCallType::Function => "function",
    };
    dict.set_item("type", type_str)?;
    
    let function_dict = PyDict::new(py);
    function_dict.set_item("name", tc.function.name)?;
    function_dict.set_item("arguments", tc.function.arguments)?;
    dict.set_item("function", function_dict)?;
    
    Ok(dict)
}

/// Add parsers module functions to the Python module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_tool_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_reasoning_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tool_calls_py, m)?)?;
    Ok(())
}