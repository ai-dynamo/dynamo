// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Number, Value};
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::MiniMaxM3ParserConfig;
use super::parsed_value::{coerce_integer_literal, raw_number_literal};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

// Used by the streaming jail to detect a complete or split MiniMax M3 tool-call opener.
pub fn detect_tool_call_start_minimax_m3(chunk: &str, config: &MiniMaxM3ParserConfig) -> bool {
    let tool_call_start = tool_call_start(config);

    if chunk.contains(tool_call_start.as_str()) {
        return true;
    }

    for i in 1..tool_call_start.len() {
        if chunk.ends_with(&tool_call_start[..i]) {
            return true;
        }
    }

    false
}

// Used by streaming jail to split the completed tool-call block from trailing content.
pub fn find_tool_call_end_position_minimax_m3(
    chunk: &str,
    config: &MiniMaxM3ParserConfig,
) -> usize {
    let tool_call_end = tool_call_end(config);
    chunk
        .find(tool_call_end.as_str())
        .map(|pos| pos + tool_call_end.len())
        .unwrap_or(chunk.len())
}

// Main entry point: strips normal prefix text and turns M3 tool markup into tool-call responses.
pub fn try_tool_call_parse_minimax_m3(
    message: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let Some(start_pos) = message.find(tool_call_start(config).as_str()) else {
        return Ok((vec![], Some(message.to_string())));
    };

    let prefix = message[..start_pos].to_string();
    let block_start = start_pos + tool_call_start(config).len();
    let tool_call_end = tool_call_end(config);
    let (block, complete) =
        if let Some(end_rel) = message[block_start..].find(tool_call_end.as_str()) {
            let block_end = block_start + end_rel;
            (&message[block_start..block_end], true)
        } else {
            (&message[block_start..], false)
        };

    if !complete && !config.allow_eof_recovery {
        return Ok((vec![], Some(prefix)));
    }

    let calls = parse_invokes(block, config, tools)?;
    Ok((calls, Some(prefix)))
}

// Builds the configured outer tool-call start marker.
fn tool_call_start(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}<{}>", config.namespace_token, config.tool_call_tag)
}

// Builds the configured outer tool-call end marker.
fn tool_call_end(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}</{}>", config.namespace_token, config.tool_call_tag)
}

// Builds the marker that introduces an individual function invocation before attributes.
fn invoke_start(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}<invoke", config.namespace_token)
}

// Builds the marker that closes an individual function invocation.
fn invoke_end(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}</invoke>", config.namespace_token)
}

// Builds the shared namespace prefix that starts any M3 XML-ish tag.
fn parameter_start(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}<", config.namespace_token)
}

// Extracts one or more `<invoke name="...">` blocks from the outer tool-call block.
fn parse_invokes(
    block: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let invoke_start = invoke_start(config);
    let invoke_end = invoke_end(config);
    let mut calls = Vec::new();
    let mut cursor = 0;

    while let Some(start_rel) = block[cursor..].find(invoke_start.as_str()) {
        let tag_attrs_start = cursor + start_rel + invoke_start.len();
        let Some(tag_end_rel) = block[tag_attrs_start..].find('>') else {
            break;
        };
        let tag_attrs = &block[tag_attrs_start..tag_attrs_start + tag_end_rel];
        let function_name = parse_invoke_name(tag_attrs);
        let body_start = tag_attrs_start + tag_end_rel + 1;
        let Some(body_end_rel) = block[body_start..].find(invoke_end.as_str()) else {
            break;
        };
        let body_end = body_start + body_end_rel;
        let function_body = &block[body_start..body_end];

        if let Some(function_name) = function_name
            && !function_name.is_empty()
        {
            let arguments = parse_parameters(&function_name, function_body, config, tools)?;
            calls.push(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: function_name,
                    arguments: serde_json::to_string(&Value::Object(arguments))?,
                },
            });
        }

        cursor = body_end + invoke_end.len();
    }

    Ok(calls)
}

// Reads the `name` attribute from `<invoke ...>` using vLLM-compatible quoting variants.
fn parse_invoke_name(tag_attrs: &str) -> Option<String> {
    let attrs = tag_attrs.trim_start();
    let after_name = attrs.strip_prefix("name")?.trim_start();
    let value = after_name.strip_prefix('=')?.trim_start();

    if let Some(value) = value.strip_prefix('"') {
        return value.find('"').map(|end| value[..end].trim().to_string());
    }
    if let Some(value) = value.strip_prefix('\'') {
        return value.find('\'').map(|end| value[..end].trim().to_string());
    }

    let end = value.find(char::is_whitespace).unwrap_or(value.len());
    if end == 0 {
        None
    } else {
        Some(value[..end].trim().to_string())
    }
}

// Extracts MiniMax M3 parameter tags, where each parameter name is the tag name itself.
fn parse_parameters(
    function_name: &str,
    body: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Map<String, Value>> {
    let parameter_start = parameter_start(config);
    let param_config = get_arguments_config(function_name, tools);
    let mut parameters = Map::new();
    let mut cursor = 0;

    while let Some(start_rel) = body[cursor..].find(parameter_start.as_str()) {
        let start = cursor + start_rel + parameter_start.len();
        if body[start..].starts_with('/') {
            cursor = start + 1;
            continue;
        }

        let Some(name_end_rel) = body[start..].find('>') else {
            break;
        };
        let parameter_name = &body[start..start + name_end_rel];
        if parameter_name.is_empty() || parameter_name.contains(char::is_whitespace) {
            cursor = start + name_end_rel + 1;
            continue;
        }

        let value_start = start + name_end_rel + 1;
        let parameter_end = format!("{}</{}>", config.namespace_token, parameter_name);
        let Some(value_end_rel) = body[value_start..].find(parameter_end.as_str()) else {
            break;
        };
        let value_end = value_start + value_end_rel;
        let raw_value = &body[value_start..value_end];
        let schema = param_config.get(parameter_name);
        let value = parse_parameter_value(raw_value, schema);
        insert_parameter(&mut parameters, parameter_name.to_string(), value);

        cursor = value_end + parameter_end.len();
    }

    Ok(parameters)
}

// Preserves duplicate XML tags by collecting repeated values into arrays.
fn insert_parameter(parameters: &mut Map<String, Value>, key: String, value: Value) {
    if let Some(existing) = parameters.remove(&key) {
        let merged = match existing {
            Value::Array(mut values) => {
                values.push(value);
                Value::Array(values)
            }
            existing => Value::Array(vec![existing, value]),
        };
        parameters.insert(key, merged);
    } else {
        parameters.insert(key, value);
    }
}

// Chooses scalar conversion or nested XML parsing based on whether the value contains M3 tags.
fn parse_parameter_value(raw: &str, schema: Option<&Value>) -> Value {
    if raw.contains("]<]minimax[>[<") {
        parse_nested_minimax_xml(raw, schema.cloned())
    } else {
        convert_scalar_value(raw, schema)
    }
}

// Parses nested parameter bodies such as arrays of `<item>` objects into JSON values.
fn parse_nested_minimax_xml(raw: &str, schema: Option<Value>) -> Value {
    let chunks: Vec<&str> = raw.split("]<]minimax[>[").collect();
    let leading_text = chunks.first().copied().unwrap_or_default();
    let root_value = if schema_has_type(schema.as_ref(), "array")
        && chunks
            .get(1)
            .is_some_and(|chunk| chunk.starts_with("<item>"))
    {
        Some(StackValue::Array(Vec::new()))
    } else {
        Some(StackValue::Object(Map::new()))
    };
    let mut stack = vec![StackItem {
        tag: None,
        value: root_value,
        texts: if leading_text.is_empty() {
            Vec::new()
        } else {
            vec![leading_text.to_string()]
        },
        schema,
    }];

    for (chunk_index, chunk) in chunks.iter().enumerate().skip(1) {
        if chunk.starts_with("</") {
            let (tag, trailing_text) = split_end_tag_chunk(chunk);
            while stack.len() > 1 {
                let item = stack.pop().expect("stack has child item");
                let matched = item.tag.as_deref() == Some(tag.as_str());
                stack
                    .last_mut()
                    .expect("stack has parent item")
                    .append(item);
                if matched {
                    break;
                }
            }
            if !trailing_text.is_empty() {
                stack
                    .last_mut()
                    .expect("stack has current item")
                    .append_text(trailing_text);
            }
        } else if chunk.starts_with('<') {
            let (tag, trailing_text) = split_start_tag_chunk(chunk);
            if tag.is_empty() {
                continue;
            }
            let child_schema = stack
                .last()
                .expect("stack has current item")
                .schema_for_child(tag.as_str());
            let child_value = if schema_has_type(child_schema.as_ref(), "array")
                && chunks
                    .get(chunk_index + 1)
                    .is_some_and(|next| next.starts_with("<item>"))
            {
                Some(StackValue::Array(Vec::new()))
            } else if schema_has_type(child_schema.as_ref(), "object") {
                Some(StackValue::Object(Map::new()))
            } else {
                None
            };
            stack.push(StackItem {
                tag: Some(tag),
                value: child_value,
                texts: if trailing_text.is_empty() {
                    Vec::new()
                } else {
                    vec![trailing_text.to_string()]
                },
                schema: child_schema,
            });
        } else if !chunk.is_empty() {
            stack
                .last_mut()
                .expect("stack has current item")
                .append_text(chunk);
        }
    }

    while stack.len() > 1 {
        let item = stack.pop().expect("stack has child item");
        stack
            .last_mut()
            .expect("stack has parent item")
            .append(item);
    }

    stack.pop().expect("root item exists").into_value()
}

// Splits a start-tag chunk into its tag name and any text after `>`.
fn split_start_tag_chunk(chunk: &str) -> (String, &str) {
    let Some(gt) = chunk.find('>') else {
        return (chunk.trim_start_matches('<').to_string(), "");
    };
    (chunk[1..gt].to_string(), &chunk[gt + 1..])
}

// Splits an end-tag chunk into its tag name and any text after `>`.
fn split_end_tag_chunk(chunk: &str) -> (String, &str) {
    let Some(gt) = chunk.find('>') else {
        return (chunk.trim_start_matches("</").to_string(), "");
    };
    (chunk[2..gt].to_string(), &chunk[gt + 1..])
}

#[derive(Debug)]
enum StackValue {
    Object(Map<String, Value>),
    Array(Vec<Value>),
}

#[derive(Debug)]
struct StackItem {
    tag: Option<String>,
    value: Option<StackValue>,
    texts: Vec<String>,
    schema: Option<Value>,
}

impl StackItem {
    // Converts a stack node into the JSON value it represents.
    fn into_value(self) -> Value {
        match self.value {
            None => convert_scalar_value(self.texts.join("").as_str(), self.schema.as_ref()),
            Some(StackValue::Object(mut map)) => {
                if !self.texts.is_empty() {
                    let mut text_key = "$text".to_string();
                    while map.contains_key(&text_key) {
                        text_key = format!("${text_key}");
                    }
                    map.insert(text_key, Value::String(self.texts.join("")));
                }
                Value::Object(map)
            }
            Some(StackValue::Array(values)) => Value::Array(values),
        }
    }

    // Attaches a completed child node to the current object, array, or implicit object.
    fn append(&mut self, item: StackItem) {
        let key = item.tag.clone().unwrap_or_default();
        let value = item.into_value();
        match self.value.as_mut() {
            None => {
                let mut map = Map::new();
                map.insert(key, value);
                self.value = Some(StackValue::Object(map));
            }
            Some(StackValue::Object(map)) => insert_parameter(map, key, value),
            Some(StackValue::Array(values)) => values.push(value),
        }
    }

    // Adds text to the current node, coercing array items through item schema when available.
    fn append_text(&mut self, text: &str) {
        if let Some(StackValue::Array(values)) = self.value.as_mut() {
            let item_schema = schema_array_item(self.schema.as_ref());
            values.push(convert_scalar_value(text, item_schema.as_ref()));
        } else {
            self.texts.push(text.to_string());
        }
    }

    // Finds the schema that should be used for a nested child tag.
    fn schema_for_child(&self, tag: &str) -> Option<Value> {
        if tag == "item" {
            if let Some(item_schema) = schema_array_item(self.schema.as_ref()) {
                return Some(item_schema);
            }
        }

        let schema = self.schema.as_ref()?;
        if let Some(child_schema) = schema
            .get("properties")
            .and_then(|properties| properties.get(tag))
        {
            return Some(child_schema.clone());
        }

        schema
            .get("additionalProperties")
            .filter(|additional| additional.is_object())
            .cloned()
    }
}

// Looks up the selected tool's parameter schema so parsed strings can be type-coerced.
fn get_arguments_config(func_name: &str, tools: Option<&[ToolDefinition]>) -> Map<String, Value> {
    let Some(tools) = tools else {
        return Map::new();
    };

    for tool in tools {
        if tool.name == func_name {
            let Some(params) = &tool.parameters else {
                return Map::new();
            };
            if let Some(properties) = params.get("properties").and_then(Value::as_object) {
                return properties.clone();
            }
            if let Some(params_obj) = params.as_object() {
                return params_obj.clone();
            }
            return Map::new();
        }
    }

    tracing::warn!("Tool '{}' is not defined in the tools list.", func_name);
    Map::new()
}

// Converts a scalar XML text value into the schema-expected JSON type when possible.
fn convert_scalar_value(raw: &str, schema: Option<&Value>) -> Value {
    let value = html_unescape(raw);
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("null") {
        return Value::Null;
    }

    let Some(schema) = schema else {
        return Value::String(value);
    };

    if schema_has_type(Some(schema), "string") || schema_has_type(Some(schema), "enum") {
        return Value::String(value);
    }
    if schema_has_type(Some(schema), "integer") {
        return coerce_integer_literal(trimmed)
            .and_then(|parsed| serde_json::to_value(parsed).ok())
            .unwrap_or(Value::String(value));
    }
    if schema_has_type(Some(schema), "number") {
        if let Some(parsed) = coerce_integer_literal(trimmed)
            && let Ok(json) = serde_json::to_value(parsed)
        {
            return json;
        }
        if let Ok(number) = trimmed.parse::<f64>()
            && let Some(number) = Number::from_f64(number)
        {
            return Value::Number(number);
        }
        if let Some(parsed) = raw_number_literal(trimmed)
            && let Ok(json) = serde_json::to_value(parsed)
        {
            return json;
        }
        return Value::String(value);
    }
    if schema_has_type(Some(schema), "boolean") {
        return match trimmed.to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "1" => Value::Bool(true),
            "false" => Value::Bool(false),
            "0" => Value::Bool(false),
            _ => Value::String(value),
        };
    }
    if schema_has_type(Some(schema), "object") {
        if trimmed.is_empty() {
            return Value::Object(Map::new());
        }
        if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
            return json;
        }
    }
    if schema_has_type(Some(schema), "array") {
        if trimmed.is_empty() {
            return Value::Array(Vec::new());
        }
        if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
            return json;
        }
    }

    Value::String(value)
}

// Checks JSON Schema `type`, `anyOf`, and `oneOf` for a target primitive/container type.
fn schema_has_type(schema: Option<&Value>, expected: &str) -> bool {
    let Some(schema) = schema else {
        return false;
    };
    if let Some(ty) = schema.get("type") {
        if ty.as_str() == Some(expected) {
            return true;
        }
        if let Some(types) = ty.as_array()
            && types.iter().any(|ty| ty.as_str() == Some(expected))
        {
            return true;
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(options) = schema.get(key).and_then(Value::as_array)
            && options
                .iter()
                .any(|option| schema_has_type(Some(option), expected))
        {
            return true;
        }
    }
    false
}

// Finds the schema for array elements, including schemas wrapped in `anyOf` or `oneOf`.
fn schema_array_item(schema: Option<&Value>) -> Option<Value> {
    schema
        .and_then(|schema| schema.get("items"))
        .cloned()
        .or_else(|| {
            schema.and_then(|schema| {
                for key in ["anyOf", "oneOf"] {
                    if let Some(options) = schema.get(key).and_then(Value::as_array) {
                        for option in options {
                            if let Some(items) = option.get("items") {
                                return Some(items.clone());
                            }
                        }
                    }
                }
                None
            })
        })
}

// Decodes common XML/HTML entities so tool arguments receive the intended literal text.
fn html_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
}
