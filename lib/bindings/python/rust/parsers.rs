// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::tool_parser_v2::unified_family_names;
use dynamo_parsers::reasoning::get_available_reasoning_parsers;
use dynamo_parsers::tool_calling::parsers::get_available_tool_parsers;
use pyo3::prelude::*;

/// Append default-on unified family names, skipping any already present. They live
/// outside the v1 registries, and either the tool-call or reasoning setting can select
/// the unified pass, so both option lists must expose them.
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

/// Add parsers module functions to the Python module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_tool_parser_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_reasoning_parser_names, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_families_are_selectable_in_both_lists() {
        for parser in ["muse_glimmer", "muse_spark"] {
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
