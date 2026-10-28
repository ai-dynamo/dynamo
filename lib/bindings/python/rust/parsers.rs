// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::tool_parser_v2::unified_family_names;
use dynamo_parsers::reasoning::get_available_reasoning_parsers;
use dynamo_parsers::tool_calling::parsers::get_available_tool_parsers;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Expose unified families and aliases in both selector lists, including families
/// absent from the V1 registry. Model validation checks the selected generation.
fn with_unified_families(mut names: Vec<&'static str>) -> Vec<&'static str> {
    for &name in unified_family_names() {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// Get list of available tool parser names
#[pyfunction]
pub fn get_tool_parser_names() -> Vec<&'static str> {
    with_unified_families(get_available_tool_parsers())
}

/// Get list of available reasoning parser names
#[pyfunction]
pub fn get_reasoning_parser_names() -> Vec<&'static str> {
    with_unified_families(get_available_reasoning_parsers())
}

/// Whether this parser pair has a structural-tag builder under DYN_PARSER_VERSION.
/// Output-parser availability and backend XGrammar support are separate capabilities.
#[pyfunction]
#[pyo3(signature = (parser_name, reasoning_parser=None))]
pub fn supports_structural_tag_parser(
    parser_name: &str,
    reasoning_parser: Option<&str>,
) -> PyResult<bool> {
    dynamo_llm::preprocessor::supports_structural_tag_parser(parser_name, reasoning_parser)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))
}

/// Reject deployment preflight when an explicitly required grammar is unavailable.
#[pyfunction]
#[pyo3(signature = (parser_name, reasoning_parser=None))]
pub fn validate_structural_tag_parser(
    parser_name: &str,
    reasoning_parser: Option<&str>,
) -> PyResult<()> {
    dynamo_llm::preprocessor::validate_structural_tag_parser(parser_name, reasoning_parser)
        .map_err(|error| PyValueError::new_err(format!("{error:#}")))
}

/// Add parsers module functions to the Python module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_tool_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_reasoning_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(supports_structural_tag_parser, m)?)?;
    m.add_function(wrap_pyfunction!(validate_structural_tag_parser, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_families_are_selectable_in_both_lists() {
        for parser in ["muse_glimmer", "muse_spark", "muse_spark_open"] {
            assert!(
                get_tool_parser_names().contains(&parser),
                "{parser} must be a selectable tool-call parser name"
            );
            assert!(
                get_reasoning_parser_names().contains(&parser),
                "{parser} must be a selectable reasoning parser name"
            );
        }
    }
}
