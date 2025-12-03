// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use dynamo_async_openai::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, FunctionObject,
};
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

/// Builds the JSON schema enforced by Guided Decoding for the given tool_choice/tools pair.
pub fn get_json_schema_from_tools(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
) -> Result<Option<Value>, ToolChoiceError> {
    let Some(choice) = tool_choice else {
        return Ok(None);
    };

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
            if tools.is_empty() {
                return Err(ToolChoiceError::EmptyTools);
            }
            build_required_schema(tools).map(Some)
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

fn build_required_schema(tools: &[ChatCompletionTool]) -> Result<Value, ToolChoiceError> {
    let mut defs: BTreeMap<String, Value> = BTreeMap::new();
    let mut any_of = Vec::with_capacity(tools.len());

    for tool in tools {
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

    let mut result = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of,
        },
    });

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

struct ParamsAndDefs {
    schema: Value,
    defs: Option<BTreeMap<String, Value>>,
}

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
    use dynamo_async_openai::types::{ChatCompletionToolChoiceOption, ChatCompletionToolType};

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
            dynamo_async_openai::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_async_openai::types::FunctionName {
                    name: "get_weather".to_string(),
                },
            },
        );
        let schema = get_json_schema_from_tools(Some(&tool_choice), Some(&tools)).expect("schema");

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
            dynamo_async_openai::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_async_openai::types::FunctionName {
                    name: "unknown".to_string(),
                },
            },
        );
        let err = get_json_schema_from_tools(Some(&tool_choice), Some(&tools)).unwrap_err();
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
        let err = build_required_schema(&tools).unwrap_err();
        assert_eq!(
            err,
            ToolChoiceError::ConflictingDefinition("shared".to_string())
        );
    }
}
