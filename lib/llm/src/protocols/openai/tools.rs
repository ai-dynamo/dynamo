// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    #[error("tool_choice `required` needs at least one tool definition")]
    EmptyTools,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolChoiceValidation<'a> {
    Unforced,
    Required,
    Named(&'a str),
}

/// The guided-decoding grammar selected for a forced tool choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoiceGuidance {
    Json(Value),
    Regex(String),
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

/// Builds the guided-decoding grammar enforced for the given tool_choice/tools pair.
pub fn get_tool_choice_guidance_from_tools(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
    parallel_tool_calls: Option<bool>,
) -> Result<Option<ToolChoiceGuidance>, ToolChoiceError> {
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
            let parameters = clone_parameters(&tool.function);
            if admits_only_empty_object(&parameters) {
                // JSON schemas allow whitespace between tokens. For the single-value `{}`
                // schema, greedy decoding can therefore emit whitespace until max_tokens.
                // Keep the named-tool constraint and make the only legal argument value exact.
                return Ok(Some(ToolChoiceGuidance::Regex(r"\{\}".to_string())));
            }
            Ok(Some(ToolChoiceGuidance::Json(parameters)))
        }
        ChatCompletionToolChoiceOption::Required => {
            let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
            build_required_schema(tools, parallel_tool_calls)
                .map(ToolChoiceGuidance::Json)
                .map(Some)
        }
    }
}

/// Builds the JSON-schema branch of the guided-decoding grammar.
///
/// Callers that install guided decoding should use `get_tool_choice_guidance_from_tools`
/// so named zero-argument tools retain their exact regex constraint.
pub fn get_json_schema_from_tools(
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    tools: Option<&[ChatCompletionTool]>,
    parallel_tool_calls: Option<bool>,
) -> Result<Option<Value>, ToolChoiceError> {
    Ok(
        get_tool_choice_guidance_from_tools(tool_choice, tools, parallel_tool_calls)?.and_then(
            |guidance| match guidance {
                ToolChoiceGuidance::Json(schema) => Some(schema),
                ToolChoiceGuidance::Regex(_) => None,
            },
        ),
    )
}

fn find_tool<'a>(tools: &'a [ChatCompletionTool], name: &str) -> Option<&'a ChatCompletionTool> {
    tools.iter().find(|tool| tool.function.name == name)
}

/// True when `schema` admits exactly one document, the empty object `{}`.
///
/// Only the fully closed, property-less object qualifies. Leaving `additionalProperties`
/// unset admits other documents, and a schema with any property gives the grammar a
/// required key to emit, so neither can stall. The keyword allowlist keeps an unfamiliar
/// constraint from being read as "empty".
fn admits_only_empty_object(schema: &Value) -> bool {
    let Value::Object(map) = schema else {
        return false;
    };
    if map.get("type").and_then(Value::as_str) != Some("object") {
        return false;
    }
    if map.get("additionalProperties") != Some(&Value::Bool(false)) {
        return false;
    }
    let properties_empty = match map.get("properties") {
        None => true,
        Some(Value::Object(properties)) => properties.is_empty(),
        Some(_) => false,
    };
    if !properties_empty {
        return false;
    }
    let required_empty = map
        .get("required")
        .is_none_or(|required| required.as_array().is_some_and(|list| list.is_empty()));
    if !required_empty {
        return false;
    }
    map.iter().all(|(key, value)| {
        matches!(
            key.as_str(),
            "type"
                | "properties"
                | "required"
                | "additionalProperties"
                | "title"
                | "description"
                | "$comment"
                | "default"
                | "deprecated"
                | "examples"
                | "readOnly"
                | "writeOnly"
        ) || key == "minProperties" && value.as_u64() == Some(0)
            || key == "maxProperties" && value.as_u64().is_some()
    })
}

fn clone_parameters(function: &FunctionObject) -> Value {
    function
        .parameters
        .clone()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
}

/// Build an array of tool calls, preserving each parameter document's local definitions.
/// Parameter schemas containing `$defs` live under separate root definitions so references
/// can be relocated without mixing types from independent tools.
fn build_required_schema(
    tools: &[ChatCompletionTool],
    parallel_tool_calls: Option<bool>,
) -> Result<Value, ToolChoiceError> {
    let mut defs = serde_json::Map::new();
    let mut any_of = Vec::with_capacity(tools.len());

    for (index, tool) in tools.iter().enumerate() {
        let mut schema = clone_parameters(&tool.function);
        if let Some(local_defs) = schema.get("$defs") {
            if !local_defs.is_object() {
                return Err(ToolChoiceError::InvalidDefinitionMap(
                    tool.function.name.clone(),
                ));
            }
            let name = format!("tool_{index}");
            let reference = format!("#/$defs/{name}");
            rebase_schema_refs(&mut schema, &reference);
            defs.insert(name, schema);
            schema = json!({"$ref": reference});
        }
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

    // Keep each parameter document in its own definition scope.
    if !defs.is_empty()
        && let Value::Object(map) = &mut result
    {
        map.insert("$defs".to_string(), Value::Object(defs));
    }

    Ok(result)
}

/// Relocate fragment-only JSON Pointer references with their parameter document.
/// An explicit `$id` starts a separate resource; its references retain that base.
fn rebase_schema_refs(schema: &mut Value, root: &str) {
    match schema {
        Value::Array(schemas) => {
            for schema in schemas {
                rebase_schema_refs(schema, root);
            }
        }
        Value::Object(schema) => {
            if schema.contains_key("$id") {
                return;
            }
            if let Some(Value::String(reference)) = schema.get_mut("$ref")
                && let Some(fragment) = reference.strip_prefix('#')
                && (fragment.is_empty()
                    || fragment.starts_with('/')
                    || fragment
                        .get(..3)
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("%2f")))
            {
                *reference = format!("{root}{fragment}");
            }
            for (keyword, value) in schema {
                match keyword.as_str() {
                    "$defs" | "definitions" | "properties" | "patternProperties"
                    | "dependentSchemas" | "dependencies" => {
                        if let Value::Object(schemas) = value {
                            for schema in schemas.values_mut() {
                                rebase_schema_refs(schema, root);
                            }
                        }
                    }
                    "items"
                    | "prefixItems"
                    | "additionalItems"
                    | "contains"
                    | "additionalProperties"
                    | "unevaluatedProperties"
                    | "unevaluatedItems"
                    | "propertyNames"
                    | "allOf"
                    | "anyOf"
                    | "oneOf"
                    | "not"
                    | "if"
                    | "then"
                    | "else"
                    | "contentSchema" => rebase_schema_refs(value, root),
                    _ => {}
                }
            }
        }
        _ => {}
    }
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

    fn zero_arg_tool(parameters: Value) -> Vec<ChatCompletionTool> {
        vec![ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "get_server_time".to_string(),
                description: Some("Get the current server time.".to_string()),
                parameters: Some(parameters),
                strict: None,
            },
        }]
    }

    fn named_choice(name: &str) -> ChatCompletionToolChoiceOption {
        ChatCompletionToolChoiceOption::Named(
            dynamo_protocols::types::ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: dynamo_protocols::types::FunctionName {
                    name: name.to_string(),
                },
            },
        )
    }

    /// GH-13789: a named choice on a tool whose schema admits only `{}` needs an exact regex.
    /// JSON-schema whitespace can otherwise run to `max_tokens` before the closing brace.
    #[test]
    fn named_choice_on_closed_zero_arg_tool_uses_exact_regex() {
        let tools = zero_arg_tool(json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false,
        }));
        let guidance = get_tool_choice_guidance_from_tools(
            Some(&named_choice("get_server_time")),
            Some(&tools),
            None,
        )
        .expect("guidance");
        assert_eq!(
            guidance,
            Some(ToolChoiceGuidance::Regex(r"\{\}".to_string())),
            "a schema admitting only the empty object needs an exact constraint"
        );
    }

    /// The escape hatch is deliberately narrow. Anything that admits more than `{}` keeps
    /// its constraint, because the grammar then always has a required token to emit.
    #[test]
    fn named_choice_keeps_constraint_for_schemas_admitting_more_than_empty() {
        let open = json!({"type": "object", "properties": {}});
        let bare = json!({"type": "object"});
        let with_property = json!({
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": [],
            "additionalProperties": false,
        });
        let unknown_keyword = json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false,
            "patternProperties": {},
        });

        for parameters in [open, bare, with_property, unknown_keyword] {
            let tools = zero_arg_tool(parameters.clone());
            let schema = get_json_schema_from_tools(
                Some(&named_choice("get_server_time")),
                Some(&tools),
                None,
            )
            .expect("schema");
            assert_eq!(
                schema,
                Some(parameters.clone()),
                "constraint should survive for {parameters}"
            );
        }
    }

    #[test]
    fn named_choice_uses_regex_for_zero_property_bounds_and_annotations() {
        for (key, value) in [
            ("minProperties", json!(0)),
            ("maxProperties", json!(1)),
            ("examples", json!([{}])),
        ] {
            let tools = zero_arg_tool(json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
                key: value,
            }));
            assert_eq!(
                get_tool_choice_guidance_from_tools(
                    Some(&named_choice("get_server_time")),
                    Some(&tools),
                    None,
                )
                .expect("guidance"),
                Some(ToolChoiceGuidance::Regex(r"\{\}".to_string()))
            );
        }
    }

    /// `tool_choice=required` wraps every tool with a required `name` key, so it can never
    /// stall on whitespace and must keep its constraint even for a zero-argument tool.
    #[test]
    fn required_choice_on_closed_zero_arg_tool_keeps_constraint() {
        let tools = zero_arg_tool(json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false,
        }));
        let schema = get_json_schema_from_tools(
            Some(&ChatCompletionToolChoiceOption::Required),
            Some(&tools),
            None,
        )
        .expect("schema")
        .expect("required always installs a constraint");

        let item = &schema["items"]["anyOf"][0];
        assert_eq!(item["properties"]["name"]["enum"][0], "get_server_time");
        assert_eq!(item["required"], json!(["name", "parameters"]));
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
    fn required_choice_isolates_conflicting_definitions() {
        let mut tools = sample_tools();
        for (tool, kind) in tools.iter_mut().zip(["string", "number"]) {
            tool.function.parameters = Some(json!({
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/shared"}},
                "$defs": {"shared": {"type": kind}}
            }));
        }
        let schema = build_required_schema(&tools, Some(false)).expect("independent schemas");
        assert_eq!(schema["maxItems"], 1);
        for (index, kind) in ["string", "number"].into_iter().enumerate() {
            let root = format!("#/$defs/tool_{index}");
            assert_eq!(
                schema["items"]["anyOf"][index]["properties"]["parameters"]["$ref"],
                root
            );
            let parameters = schema.pointer(&root[1..]).expect("tool schema");
            assert_eq!(
                parameters["properties"]["value"]["$ref"],
                format!("{root}/$defs/shared")
            );
            assert_eq!(parameters["$defs"]["shared"]["type"], kind);
        }
        assert_eq!(
            tools[0].function.parameters.as_ref().unwrap()["properties"]["value"]["$ref"],
            "#/$defs/shared"
        );
    }

    #[test]
    fn required_choice_rebases_only_document_local_schema_references() {
        let mut tools = sample_tools();
        tools.truncate(1);
        tools[0].function.parameters = Some(json!({
            "type": "object",
            "properties": {
                "value": {"$ref": "#/$defs/a~1b~0c"},
                "child": {"$ref": "#"},
                "encoded": {"$ref": "#%2F$defs%2Fa~1b~0c"},
                "remote": {"$ref": "urn:example:remote"},
                "relative": {"$ref": "a%2Fremote"},
                "scoped": {"$id": "urn:example:scoped", "$ref": "#/$defs/local", "$defs": {"local": {"type": "string"}}},
                "literal": {"const": {"$ref": "#/$defs/a~1b~0c"}}
            },
            "$defs": {"a/b~c": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/a~1b~0c"}]}},
            "default": {"$ref": "#"}
        }));
        let schema = build_required_schema(&tools, None).unwrap();
        let parameters = &schema["$defs"]["tool_0"];
        assert_eq!(
            parameters["properties"]["value"]["$ref"],
            "#/$defs/tool_0/$defs/a~1b~0c"
        );
        assert_eq!(parameters["properties"]["child"]["$ref"], "#/$defs/tool_0");
        assert_eq!(
            parameters["properties"]["encoded"]["$ref"],
            "#/$defs/tool_0%2F$defs%2Fa~1b~0c"
        );
        assert_eq!(
            parameters["$defs"]["a/b~c"]["anyOf"][1]["$ref"],
            "#/$defs/tool_0/$defs/a~1b~0c"
        );
        assert_eq!(
            parameters["properties"]["remote"]["$ref"],
            "urn:example:remote"
        );
        assert_eq!(parameters["properties"]["relative"]["$ref"], "a%2Fremote");
        assert_eq!(parameters["properties"]["scoped"]["$ref"], "#/$defs/local");
        assert_eq!(
            parameters["properties"]["literal"]["const"]["$ref"],
            "#/$defs/a~1b~0c"
        );
        assert_eq!(parameters["default"]["$ref"], "#");
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
