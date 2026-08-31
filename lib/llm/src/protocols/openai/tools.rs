// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, FunctionObject};
use serde_json::{Value, json};
use thiserror::Error;

/// Errors that can occur when deriving JSON schemas for tool_choice requests.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ToolChoiceError {
    #[error("tool_choice requires a matching `tools` array")]
    MissingTools,
    #[error("tool `{0}` was not provided in `tools`")]
    ToolNotFound(String),
    #[error("$defs for tool `{0}` must be an object")]
    InvalidDefinitionMap(String),
    #[error("duplicate $defs entry `{0}` has conflicting schemas")]
    ConflictingDefinition(String),
    #[error("tool_choice `required` needs at least one tool definition")]
    EmptyTools,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolChoiceValidation<'a> {
    Unforced,
    Required,
    Named(&'a str),
}

/// Validate the forced-choice contract shared by wire and parser-facing tool types.
pub(crate) fn validate_tool_choice_against_names<'a>(
    tool_choice: ToolChoiceValidation<'_>,
    tool_names: impl IntoIterator<Item = &'a str>,
) -> Result<(), ToolChoiceError> {
    let mut tool_names = tool_names.into_iter();
    match tool_choice {
        ToolChoiceValidation::Unforced => Ok(()),
        ToolChoiceValidation::Required if tool_names.next().is_none() => {
            Err(ToolChoiceError::EmptyTools)
        }
        ToolChoiceValidation::Named(name) if !tool_names.any(|tool_name| tool_name == name) => {
            Err(ToolChoiceError::ToolNotFound(name.to_string()))
        }
        ToolChoiceValidation::Required | ToolChoiceValidation::Named(_) => Ok(()),
    }
}

/// Validate an OpenAI `tool_choice` against the request's declared tools.
pub(crate) fn validate_openai_tool_choice(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
) -> Result<(), ToolChoiceError> {
    let Some(tool_choice) = tool_choice else {
        return Ok(());
    };

    match tool_choice {
        ChatCompletionToolChoiceOption::None | ChatCompletionToolChoiceOption::Auto => Ok(()),
        ChatCompletionToolChoiceOption::Required => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            validate_tool_choice_against_names(
                ToolChoiceValidation::Required,
                tools.iter().map(|tool| tool.function.name.as_str()),
            )
        }
        ChatCompletionToolChoiceOption::Named(named) => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            validate_tool_choice_against_names(
                ToolChoiceValidation::Named(&named.function.name),
                tools.iter().map(|tool| tool.function.name.as_str()),
            )
        }
    }
}

/// Builds the JSON schema enforced by Guided Decoding for the given tool_choice/tools pair.
pub fn get_json_schema_from_tools(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
    parallel_tool_calls: Option<bool>,
) -> Result<Option<Value>, ToolChoiceError> {
    let Some(choice) = tool_choice else {
        return Ok(None);
    };
    validate_openai_tool_choice(Some(choice), tools)?;

    match choice {
        ChatCompletionToolChoiceOption::None | ChatCompletionToolChoiceOption::Auto => Ok(None),
        ChatCompletionToolChoiceOption::Named(named) => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            let tool = find_tool(tools, &named.function.name)
                .ok_or_else(|| ToolChoiceError::ToolNotFound(named.function.name.clone()))?;
            Ok(Some(clone_parameters(&tool.function)))
        }
        ChatCompletionToolChoiceOption::Required => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            build_required_schema(tools, parallel_tool_calls).map(Some)
        }
    }
}

fn find_tool<'a>(tools: &'a [ChatCompletionTool], name: &str) -> Option<&'a ChatCompletionTool> {
    tools.iter().find(|tool| tool.function.name == name)
}

fn clone_parameters(function: &FunctionObject) -> Value {
    function
        .parameters
        .clone()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
}

/// Builds a JSON Schema for `tool_choice=required` that enforces an array of tool calls.
///
/// # Schema Structure
///
/// The generated schema looks like:
/// ```json
/// {
///   "type": "array",
///   "minItems": 1,
///   "items": {
///     "type": "object",
///     "anyOf": [
///       {
///         "properties": {
///           "name": {"type": "string", "enum": ["tool1"]},
///           "parameters": { /* tool1's parameter schema */ }
///         },
///         "required": ["name", "parameters"]
///       },
///       {
///         "properties": {
///           "name": {"type": "string", "enum": ["tool2"]},
///           "parameters": { /* tool2's parameter schema */ }
///         },
///         "required": ["name", "parameters"]
///       }
///     ]
///   },
///   "$defs": { /* shared type definitions from all tools */ }
/// }
/// ```
///
/// # $defs Handling
///
/// `$defs` contains shared JSON Schema definitions that can be referenced via `$ref`.
/// For example, if two tools reference a common type:
/// ```json
/// {
///   "$defs": {
///     "Location": {
///       "type": "object",
///       "properties": {
///         "city": {"type": "string"},
///         "country": {"type": "string"}
///       }
///     }
///   }
/// }
/// ```
///
/// We extract `$defs` from each tool's schema and merge them into a global `$defs` map
/// at the root level. If multiple tools define the same type, we verify they match to
/// avoid conflicts.
fn build_required_schema(
    tools: &[ChatCompletionTool],
    parallel_tool_calls: Option<bool>,
) -> Result<Value, ToolChoiceError> {
    // Accumulator for all shared type definitions ($defs) across tools
    let mut defs: BTreeMap<String, Value> = BTreeMap::new();
    let mut any_of = Vec::with_capacity(tools.len());

    for tool in tools {
        // Extract parameter schema and its $defs (if any)
        let ParamsAndDefs {
            schema,
            defs: new_defs,
        } = split_defs(&tool.function)?;
        merge_defs(&mut defs, new_defs)?;
        any_of.push(json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name],
                },
                "parameters": schema,
            },
            "required": ["name", "parameters"],
        }));
    }

    // Build the top-level array schema with anyOf constraints
    let mut result = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of,
        },
    });

    // `parallel_tool_calls: false` is otherwise enforced only downstream, by discarding
    // tool indices above zero in the HTTP stream. That leaves the extra calls GENERATED
    // and observable by any consumer upstream of that filter, and it wastes the tokens
    // spent producing them. Constraining generation is the earlier, cheaper fix; the
    // HTTP filter stays as defense in depth.
    if parallel_tool_calls == Some(false)
        && let Value::Object(map) = &mut result
    {
        map.insert("maxItems".to_string(), json!(1));
    }

    // Attach the merged $defs at the root level if any were collected
    if !defs.is_empty()
        && let Value::Object(map) = &mut result
    {
        map.insert(
            "$defs".to_string(),
            Value::Object(defs.into_iter().collect()),
        );
    }

    Ok(result)
}

/// Holds a tool's parameter schema and its extracted $defs (if any).
///
/// When a tool's parameters reference shared types via `$ref`, those types
/// are defined in a `$defs` section within the schema. We extract them separately
/// to merge into a global definitions map.
struct ParamsAndDefs {
    /// The parameter schema with `$defs` removed (if it had one)
    schema: Value,
    /// Extracted `$defs` map, or None if the schema had no definitions
    defs: Option<BTreeMap<String, Value>>,
}

/// Extracts `$defs` from a function's parameter schema, returning both the
/// cleaned schema and the definitions separately.
///
/// # Example
///
/// Input schema:
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "location": {"$ref": "#/$defs/Location"}
///   },
///   "$defs": {
///     "Location": {
///       "type": "object",
///       "properties": {"city": {"type": "string"}}
///     }
///   }
/// }
/// ```
///
/// Returns:
/// - schema: same as input but with `$defs` removed
/// - defs: `Some({"Location": {...}})`
fn split_defs(function: &FunctionObject) -> Result<ParamsAndDefs, ToolChoiceError> {
    let mut schema = clone_parameters(function);
    let defs = match &mut schema {
        Value::Object(obj) => {
            if let Some(value) = obj.remove("$defs") {
                Some(convert_defs(function, value)?)
            } else {
                None
            }
        }
        _ => None,
    };

    Ok(ParamsAndDefs { schema, defs })
}

fn convert_defs(
    function: &FunctionObject,
    defs_value: Value,
) -> Result<BTreeMap<String, Value>, ToolChoiceError> {
    match defs_value {
        Value::Object(map) => Ok(map.into_iter().collect()),
        _ => Err(ToolChoiceError::InvalidDefinitionMap(function.name.clone())),
    }
}

/// Merges definitions from one tool into the global `$defs` accumulator.
///
/// # Conflict Detection
///
/// If two tools define the same type name but with different schemas, we return
/// an error. This ensures consistency across tool definitions.
///
/// # Example
///
/// If `target` contains:
/// ```json
/// {"Location": {"type": "object", "properties": {"city": {"type": "string"}}}}
/// ```
///
/// And we try to merge:
/// ```json
/// {"Location": {"type": "object", "properties": {"city": {"type": "number"}}}}
/// ```
///
/// This will return `ToolChoiceError::ConflictingDefinition("Location")`.
fn merge_defs(
    target: &mut BTreeMap<String, Value>,
    defs: Option<BTreeMap<String, Value>>,
) -> Result<(), ToolChoiceError> {
    let Some(defs) = defs else {
        return Ok(());
    };

    for (name, schema) in defs {
        if let Some(existing) = target.get(&name) {
            if existing != &schema {
                return Err(ToolChoiceError::ConflictingDefinition(name));
            }
        } else {
            target.insert(name, schema);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{ChatCompletionToolChoiceOption, ChatCompletionToolType};

    fn sample_tools() -> Vec<ChatCompletionTool> {
        vec![
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: "add_numbers".to_string(),
                    description: Some("Add two integers".to_string()),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    })),
                    strict: None,
                },
            },
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location", "unit"],
                    })),
                    strict: None,
                },
            },
        ]
    }

    #[test]
    fn named_choice_returns_parameters() {
        let tools = sample_tools();
        let tool_choice = ChatCompletionToolChoiceOption::Named(
            dynamo_protocols::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_protocols::types::FunctionName {
                    name: "get_weather".to_string(),
                },
            },
        );
        let schema =
            get_json_schema_from_tools(Some(&tool_choice), Some(&tools), None).expect("schema");

        assert_eq!(
            schema.unwrap(),
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            })
        );
    }

    #[test]
    fn required_choice_builds_any_of_schema() {
        let tools = sample_tools();
        let schema = get_json_schema_from_tools(
            Some(&ChatCompletionToolChoiceOption::Required),
            Some(&tools),
            None,
        )
        .expect("schema");

        let schema = schema.expect("required schema");
        assert_eq!(schema["type"], "array");
        assert_eq!(schema["minItems"], 1);
        assert!(schema["items"]["anyOf"].is_array());

        let any_of = schema["items"]["anyOf"].as_array().unwrap();
        assert_eq!(any_of.len(), 2);
        assert_eq!(
            any_of[0]["properties"]["name"],
            json!({"type": "string", "enum": ["add_numbers"]})
        );
    }

    #[test]
    fn missing_tool_errors() {
        let tools = sample_tools();
        let tool_choice = ChatCompletionToolChoiceOption::Named(
            dynamo_protocols::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_protocols::types::FunctionName {
                    name: "unknown".to_string(),
                },
            },
        );
        let err = get_json_schema_from_tools(Some(&tool_choice), Some(&tools), None).unwrap_err();
        assert_eq!(err, ToolChoiceError::ToolNotFound("unknown".to_string()));
    }

    #[test]
    fn conflicting_defs_errors() {
        let tool = ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "foo".to_string(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "$defs": {
                        "shared": {"type": "string"}
                    }
                })),
                strict: None,
            },
        };

        let mut tool_with_conflict = tool.clone();
        tool_with_conflict.function.parameters = Some(json!({
            "type": "object",
            "$defs": {
                "shared": {"type": "number"}
            }
        }));

        let tools = vec![tool, tool_with_conflict];
        let err = build_required_schema(&tools, None).unwrap_err();
        assert_eq!(
            err,
            ToolChoiceError::ConflictingDefinition("shared".to_string())
        );
    }

    #[test]
    fn required_schema_is_unbounded_by_default() {
        let tools = sample_tools();
        let schema = build_required_schema(&tools, None).expect("schema");
        assert_eq!(schema["minItems"], json!(1));
        assert!(
            schema.get("maxItems").is_none(),
            "parallel calls stay unbounded unless the request disables them"
        );
        let schema = build_required_schema(&tools, Some(true)).expect("schema");
        assert!(schema.get("maxItems").is_none());
    }

    /// `parallel_tool_calls: false` must constrain GENERATION, not just be filtered
    /// downstream - otherwise the extra calls are still produced, still cost tokens,
    /// and are still observable upstream of the HTTP filter.
    #[test]
    fn required_schema_caps_at_one_when_parallel_calls_are_disabled() {
        let tools = sample_tools();
        let schema = build_required_schema(&tools, Some(false)).expect("schema");
        assert_eq!(schema["minItems"], json!(1));
        assert_eq!(
            schema["maxItems"],
            json!(1),
            "parallel_tool_calls=false must cap the array at one element"
        );
    }
}
