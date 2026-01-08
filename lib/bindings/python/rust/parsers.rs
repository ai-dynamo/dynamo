// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::reasoning::get_available_reasoning_parsers;
use dynamo_parsers::tool_calling::parsers::get_available_tool_parsers;
use pyo3::prelude::*;

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

/// Add parsers module functions to the Python module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_tool_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_reasoning_parser_names, m)?)?;
    Ok(())
}
