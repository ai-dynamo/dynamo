// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, fmt::Display, sync::LazyLock};

use dynamo_runtime::config::{
    env_is_truthy, environment_names::llm::DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS,
};

use super::tools::{ToolChoiceError, validate_openai_tool_choice};

//
// Hyperparameter Contraints
//

/// Minimum allowed value for OpenAI's `temperature` sampling option
pub const MIN_TEMPERATURE: f32 = 0.0;
/// Maximum allowed value for OpenAI's `temperature` sampling option
pub const MAX_TEMPERATURE: f32 = 2.0;
/// Allowed range of values for OpenAI's `temperature`` sampling option
pub const TEMPERATURE_RANGE: (f32, f32) = (MIN_TEMPERATURE, MAX_TEMPERATURE);

/// Minimum allowed value for OpenAI's `top_p` sampling option
pub const MIN_TOP_P: f32 = 0.0;
/// Maximum allowed value for OpenAI's `top_p` sampling option
pub const MAX_TOP_P: f32 = 1.0;

/// Minimum allowed value for `min_p`
pub const MIN_MIN_P: f32 = 0.0;
/// Maximum allowed value for `min_p`
pub const MAX_MIN_P: f32 = 1.0;
/// Allowed range of values for `min_p`
pub const MIN_P_RANGE: (f32, f32) = (MIN_MIN_P, MAX_MIN_P);

/// Minimum allowed value for OpenAI's `frequency_penalty` sampling option
pub const MIN_FREQUENCY_PENALTY: f32 = -2.0;
/// Maximum allowed value for OpenAI's `frequency_penalty` sampling option
pub const MAX_FREQUENCY_PENALTY: f32 = 2.0;
/// Allowed range of values for OpenAI's `frequency_penalty` sampling option
pub const FREQUENCY_PENALTY_RANGE: (f32, f32) = (MIN_FREQUENCY_PENALTY, MAX_FREQUENCY_PENALTY);

/// Minimum allowed value for OpenAI's `presence_penalty` sampling option
pub const MIN_PRESENCE_PENALTY: f32 = -2.0;
/// Maximum allowed value for OpenAI's `presence_penalty` sampling option
pub const MAX_PRESENCE_PENALTY: f32 = 2.0;
/// Allowed range of values for OpenAI's `presence_penalty` sampling option
pub const PRESENCE_PENALTY_RANGE: (f32, f32) = (MIN_PRESENCE_PENALTY, MAX_PRESENCE_PENALTY);

/// Minimum allowed value for `length_penalty`
pub const MIN_LENGTH_PENALTY: f32 = -2.0;
/// Maximum allowed value for `length_penalty`
pub const MAX_LENGTH_PENALTY: f32 = 2.0;
/// Allowed range of values for `length_penalty`
pub const LENGTH_PENALTY_RANGE: (f32, f32) = (MIN_LENGTH_PENALTY, MAX_LENGTH_PENALTY);

/// Maximum allowed value for `top_logprobs`
pub const MIN_TOP_LOGPROBS: u8 = 0;
/// Maximum allowed value for `top_logprobs`
pub const MAX_TOP_LOGPROBS: u8 = 20;

/// Minimum allowed value for `logprobs` in completion requests
pub const MIN_LOGPROBS: u8 = 0;
/// Maximum allowed value for `logprobs` in completion requests
pub const MAX_LOGPROBS: u8 = 5;

/// Minimum allowed value for `n` (number of choices)
pub const MIN_N: u8 = 1;
/// Maximum allowed value for `n` (number of choices)
pub const MAX_N: u8 = 128;
/// Allowed range of values for `n` (number of choices)
pub const N_RANGE: (u8, u8) = (MIN_N, MAX_N);

/// Maximum allowed total number of choices (batch_size × n)
pub const MAX_TOTAL_CHOICES: usize = 128;

/// Minimum allowed value for OpenAI's `logit_bias` values
pub const MIN_LOGIT_BIAS: f32 = -100.0;
/// Maximum allowed value for OpenAI's `logit_bias` values
pub const MAX_LOGIT_BIAS: f32 = 100.0;

/// Minimum allowed value for `best_of`
pub const MIN_BEST_OF: u8 = 0;
/// Maximum allowed value for `best_of`
pub const MAX_BEST_OF: u8 = 20;
/// Allowed range of values for `best_of`
pub const BEST_OF_RANGE: (u8, u8) = (MIN_BEST_OF, MAX_BEST_OF);

/// Maximum allowed number of stop sequences.
pub const MAX_STOP_SEQUENCES: usize = 32;
/// Maximum allowed number of tools.
pub const MAX_TOOLS: usize = 1536;
// Metadata validation constants removed - we are no longer restricting the metadata field char limits
/// Both `/v1/messages` and `/v1/responses` define a 128-character tool-name limit.
pub const MAX_FUNCTION_NAME_LENGTH: usize = 128;
/// Minimum allowed value for `repetition_penalty`
pub const MIN_REPETITION_PENALTY: f32 = 0.0;
/// Maximum allowed value for `repetition_penalty`
pub const MAX_REPETITION_PENALTY: f32 = 2.0;

//
// Shared Fields
//

/// Extra-body fields accepted for backend-specific handling.
pub const PASSTHROUGH_EXTRA_FIELDS: &[&str] = &[
    "cache_salt",
    "stop_token_ids",
    "detokenize",
    "allowed_token_ids",
    "bad_words_token_ids",
    "logprob_token_ids",
];

static IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy(DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS));

/// Validates that no unsupported fields are present in the request.
///
/// Fields in `PASSTHROUGH_EXTRA_FIELDS` are validated by downstream handlers.
/// Other fields may be ignored and dropped when
/// `DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS` is truthy.
pub fn validate_no_unsupported_fields(
    unsupported_fields: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<(), anyhow::Error> {
    validate_no_unsupported_fields_with_ignore(
        unsupported_fields,
        *IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS,
    )
}

fn validate_no_unsupported_fields_with_ignore(
    unsupported_fields: &std::collections::HashMap<String, serde_json::Value>,
    ignore_unsupported_fields: bool,
) -> Result<(), anyhow::Error> {
    let unknown: Vec<_> = unsupported_fields
        .keys()
        .filter(|k| !PASSTHROUGH_EXTRA_FIELDS.contains(&k.as_str()))
        .map(|s| format!("`{}`", s))
        .collect();
    if !unknown.is_empty() && !ignore_unsupported_fields {
        anyhow::bail!("Unsupported parameter(s): {}", unknown.join(", "));
    }
    if let Some(value) = unsupported_fields.get("cache_salt")
        && !value.is_string()
    {
        anyhow::bail!("`cache_salt` must be a string");
    }
    if let Some(value) = unsupported_fields.get("stop_token_ids") {
        serde_json::from_value::<Vec<crate::types::TokenIdType>>(value.clone())
            .map_err(|_| anyhow::anyhow!("`stop_token_ids` must be an array of token IDs"))?;
    }
    if let Some(value) = unsupported_fields.get("detokenize")
        && !value.is_boolean()
    {
        anyhow::bail!("`detokenize` must be a boolean");
    }
    if let Some(value) = unsupported_fields.get("allowed_token_ids") {
        serde_json::from_value::<Vec<crate::types::TokenIdType>>(value.clone())
            .map_err(|_| anyhow::anyhow!("`allowed_token_ids` must be an array of token IDs"))?;
    }
    if let Some(value) = unsupported_fields.get("bad_words_token_ids") {
        serde_json::from_value::<Vec<Vec<crate::types::TokenIdType>>>(value.clone()).map_err(
            |_| anyhow::anyhow!("`bad_words_token_ids` must be an array of token ID arrays"),
        )?;
    }
    if let Some(value) = unsupported_fields.get("logprob_token_ids") {
        serde_json::from_value::<Vec<crate::types::TokenIdType>>(value.clone())
            .map_err(|_| anyhow::anyhow!("`logprob_token_ids` must be an array of token IDs"))?;
    }
    Ok(())
}

/// Validates response_format for chat completions.
///
/// Dynamo currently supports translating:
/// - `{"type":"json_object"}` -> guided decoding JSON object schema
/// - `{"type":"json_schema","json_schema":{"schema": ...}}` -> guided decoding JSON schema
///
/// `{"type":"text"}` is accepted and means no structured constraint.
pub fn validate_response_format(
    response_format: &Option<dynamo_protocols::types::ResponseFormat>,
) -> Result<(), anyhow::Error> {
    use dynamo_protocols::types::ResponseFormat;

    let Some(fmt) = response_format else {
        return Ok(());
    };

    match fmt {
        ResponseFormat::Text => Ok(()),
        ResponseFormat::JsonObject => Ok(()),
        ResponseFormat::JsonSchema { json_schema } => {
            // Validate name field format
            if json_schema.name.is_empty() {
                anyhow::bail!("`response_format.json_schema.name` cannot be empty");
            }

            // Validate schema presence. `schema` is a non-optional
            // `serde_json::Value`, so an explicit `null` is the only way it
            // can still arrive empty.
            if json_schema.schema.is_null() {
                anyhow::bail!(
                    "`response_format.json_schema.schema` is required when `response_format.type` is `json_schema`"
                );
            }

            // Schema must be a JSON object — numbers, strings, arrays, and
            // booleans are not valid JSON Schema documents.
            if !json_schema.schema.is_object() {
                anyhow::bail!(
                    "`response_format.json_schema.schema` must be a JSON object, got {}",
                    match &json_schema.schema {
                        serde_json::Value::Array(_) => "array",
                        serde_json::Value::String(_) => "string",
                        serde_json::Value::Number(_) => "number",
                        serde_json::Value::Bool(_) => "boolean",
                        _ => "non-object",
                    }
                );
            }
            Ok(())
        }
    }
}

/// Validates the temperature parameter
pub fn validate_temperature(temperature: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(temp) = temperature
        && !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temp)
    {
        anyhow::bail!(
            "Temperature must be between {} and {}, got {}",
            MIN_TEMPERATURE,
            MAX_TEMPERATURE,
            temp
        );
    }
    Ok(())
}

/// Validates the top_p parameter
pub fn validate_top_p(top_p: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(p) = top_p
        && !(p.is_finite() && p > MIN_TOP_P && p <= MAX_TOP_P)
    {
        anyhow::bail!(
            "Top_p must be between {} and {}, got {}",
            MIN_TOP_P,
            MAX_TOP_P,
            p
        );
    }
    Ok(())
}

// Validate top_k
pub fn validate_top_k(top_k: Option<i32>) -> Result<(), anyhow::Error> {
    match top_k {
        None => Ok(()),
        Some(k) if k >= -1 => Ok(()),
        _ => anyhow::bail!("Top_k must be null or greater than or equal to -1"),
    }
}

/// Validates mutual exclusion of temperature and top_p
pub fn validate_temperature_top_p_exclusion(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), anyhow::Error> {
    match (temperature, top_p) {
        (Some(t), Some(p)) if t != 1.0 && p != 1.0 => {
            anyhow::bail!("Only one of temperature or top_p should be set (not both)");
        }
        _ => Ok(()),
    }
}

/// Validates frequency penalty parameter
pub fn validate_frequency_penalty(frequency_penalty: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(penalty) = frequency_penalty
        && !(MIN_FREQUENCY_PENALTY..=MAX_FREQUENCY_PENALTY).contains(&penalty)
    {
        anyhow::bail!(
            "Frequency penalty must be between {} and {}, got {}",
            MIN_FREQUENCY_PENALTY,
            MAX_FREQUENCY_PENALTY,
            penalty
        );
    }
    Ok(())
}

/// Validates presence penalty parameter
pub fn validate_presence_penalty(presence_penalty: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(penalty) = presence_penalty
        && !(MIN_PRESENCE_PENALTY..=MAX_PRESENCE_PENALTY).contains(&penalty)
    {
        anyhow::bail!(
            "Presence penalty must be between {} and {}, got {}",
            MIN_PRESENCE_PENALTY,
            MAX_PRESENCE_PENALTY,
            penalty
        );
    }
    Ok(())
}

pub fn validate_repetition_penalty(repetition_penalty: Option<f32>) -> Result<(), anyhow::Error> {
    // It should be greater than 0.0 and less than equal to 2.0
    if let Some(penalty) = repetition_penalty
        && (penalty <= MIN_REPETITION_PENALTY || penalty > MAX_REPETITION_PENALTY)
    {
        anyhow::bail!(
            "Repetition penalty must be between {} and {}, got {}",
            MIN_REPETITION_PENALTY,
            MAX_REPETITION_PENALTY,
            penalty
        );
    }
    Ok(())
}

/// Validates min_p parameter
pub fn validate_min_p(min_p: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(p) = min_p
        && !(MIN_MIN_P..=MAX_MIN_P).contains(&p)
    {
        anyhow::bail!(
            "Min_p must be between {} and {}, got {}",
            MIN_MIN_P,
            MAX_MIN_P,
            p
        );
    }
    Ok(())
}

/// Validates logit bias map
pub fn validate_logit_bias(
    logit_bias: &Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Result<(), anyhow::Error> {
    let logit_bias = match logit_bias {
        Some(val) => val,
        None => return Ok(()),
    };

    for (token, bias_value) in logit_bias {
        let bias = bias_value.as_f64().ok_or_else(|| {
            anyhow::anyhow!(
                "Logit bias value for token '{}' must be a number, got {:?}",
                token,
                bias_value
            )
        })? as f32;

        if !(MIN_LOGIT_BIAS..=MAX_LOGIT_BIAS).contains(&bias) {
            anyhow::bail!(
                "Logit bias for token '{}' must be between {} and {}, got {}",
                token,
                MIN_LOGIT_BIAS,
                MAX_LOGIT_BIAS,
                bias
            );
        }
    }
    Ok(())
}

/// Validates n parameter (number of choices)
pub fn validate_n(n: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = n
        && !(MIN_N..=MAX_N).contains(&value)
    {
        anyhow::bail!("n must be between {} and {}, got {}", MIN_N, MAX_N, value);
    }
    Ok(())
}

/// Validates total choices (batch_size × n) doesn't exceed maximum
pub fn validate_total_choices(batch_size: usize, n: u8) -> Result<(), anyhow::Error> {
    let total_choices = batch_size * (n as usize);
    if total_choices > MAX_TOTAL_CHOICES {
        anyhow::bail!(
            "Total choices (batch_size × n = {} × {} = {}) exceeds maximum of {}",
            batch_size,
            n,
            total_choices,
            MAX_TOTAL_CHOICES
        );
    }
    Ok(())
}

/// Validates n and temperature interaction
/// When n > 1, temperature must be > 0 to ensure diverse outputs
pub fn validate_n_with_temperature(
    n: Option<u8>,
    temperature: Option<f32>,
) -> Result<(), anyhow::Error> {
    if let Some(n_value) = n
        && n_value > 1
    {
        let temp = temperature.unwrap_or(1.0);
        if temp == 0.0 {
            anyhow::bail!(
                "When n > 1, temperature must be greater than 0 to ensure diverse outputs. Got n={}, temperature={}",
                n_value,
                temp
            );
        }
    }
    Ok(())
}

/// Validates model parameter
pub fn validate_model(model: &str) -> Result<(), anyhow::Error> {
    if model.trim().is_empty() {
        anyhow::bail!("Model cannot be empty");
    }
    Ok(())
}

/// Validates user parameter
pub fn validate_user(user: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(user_id) = user
        && user_id.trim().is_empty()
    {
        anyhow::bail!("User ID cannot be empty");
    }
    Ok(())
}

/// Validates stop sequences
pub fn validate_stop(stop: &Option<dynamo_protocols::types::Stop>) -> Result<(), anyhow::Error> {
    if let Some(stop_value) = stop {
        match stop_value {
            dynamo_protocols::types::Stop::String(s) => {
                if s.is_empty() {
                    anyhow::bail!("Stop sequence cannot be empty");
                }
            }
            dynamo_protocols::types::Stop::StringArray(sequences) => {
                if sequences.is_empty() {
                    anyhow::bail!("Stop sequences array cannot be empty");
                }
                if sequences.len() > MAX_STOP_SEQUENCES {
                    anyhow::bail!(
                        "Maximum of {} stop sequences allowed, got {}",
                        MAX_STOP_SEQUENCES,
                        sequences.len()
                    );
                }
                for (i, sequence) in sequences.iter().enumerate() {
                    if sequence.is_empty() {
                        anyhow::bail!("Stop sequence at index {} cannot be empty", i);
                    }
                }
            }
            dynamo_protocols::types::Stop::TokenIdArray(token_ids) => {
                if token_ids.is_empty() {
                    anyhow::bail!("Stop token IDs array cannot be empty");
                }
                if token_ids.len() > MAX_STOP_SEQUENCES {
                    anyhow::bail!(
                        "Maximum of {} stop token IDs allowed, got {}",
                        MAX_STOP_SEQUENCES,
                        token_ids.len()
                    );
                }
            }
        }
    }
    Ok(())
}

//
// Chat Completion Specific
//

/// Validates messages array
pub fn validate_messages(
    messages: &[dynamo_protocols::types::ChatCompletionRequestMessage],
) -> Result<(), anyhow::Error> {
    if messages.is_empty() {
        anyhow::bail!("Messages array cannot be empty");
    }
    Ok(())
}

/// Validates prior assistant tool calls against matching tools in this request.
///
/// Arguments must always be JSON object strings. When the current request also
/// supplies a matching tool with a parameters schema, this additionally checks
/// the schema's basic JSON types, required properties, nested object/array
/// properties, and `additionalProperties: false`. Tool calls without a matching
/// current definition retain syntax-only validation, which allows clients to
/// replay history after removing or renaming a tool.
pub fn validate_tool_call_arguments(
    messages: &[dynamo_protocols::types::ChatCompletionRequestMessage],
    tools: Option<&[dynamo_protocols::types::ChatCompletionTool]>,
) -> Result<(), anyhow::Error> {
    let mut schemas_by_name = HashMap::new();
    if let Some(tools) = tools {
        for tool in tools {
            if let Some(parameters) = &tool.function.parameters {
                // Preserve the first definition, matching `iter().find()` semantics
                // without an O(messages * tools) scan.
                schemas_by_name
                    .entry(tool.function.name.as_str())
                    .or_insert(parameters);
            }
        }
    }

    for (message_index, message) in messages.iter().enumerate() {
        if let dynamo_protocols::types::ChatCompletionRequestMessage::Assistant(assistant) = message
            && let Some(tool_calls) = &assistant.tool_calls
        {
            for (tool_call_index, tool_call) in tool_calls.iter().enumerate() {
                let field = format!(
                    "messages[{message_index}].tool_calls[{tool_call_index}].function.arguments"
                );
                let Some(arguments) =
                    parse_json_object_string(&tool_call.function.arguments, &field)?
                else {
                    continue;
                };

                if let Some(schema) = schemas_by_name.get(tool_call.function.name.as_str()) {
                    validate_basic_json_schema(&arguments, schema, &field)?;
                }
            }
        }
    }
    Ok(())
}

fn parse_json_object_string(
    value: &str,
    field: &str,
) -> Result<Option<serde_json::Value>, anyhow::Error> {
    if value.trim().is_empty() {
        return Ok(None);
    }
    let parsed: serde_json::Value = serde_json::from_str(value).map_err(|error| {
        anyhow::anyhow!("`{field}` must be a valid JSON object string: {error}")
    })?;
    if !parsed.is_object() {
        anyhow::bail!("`{field}` must be a valid JSON object string");
    }
    Ok(Some(parsed))
}

fn validate_basic_json_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    field: &str,
) -> Result<(), anyhow::Error> {
    if schema == &serde_json::Value::Bool(true) {
        return Ok(());
    }
    if schema == &serde_json::Value::Bool(false) {
        anyhow::bail!("`{field}` is not allowed by the tool parameters schema");
    }

    let Some(schema) = schema.as_object() else {
        // Top-level parameter schemas are checked by `validate_tools`. Ignore
        // malformed nested schemas here rather than treating them as data types.
        return Ok(());
    };

    if let Some(expected) = schema.get("type")
        && !matches_json_schema_type(value, expected)
    {
        anyhow::bail!(
            "`{field}` must be of type {}, got {}",
            display_schema_type(expected),
            json_type_name(value),
        );
    }

    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(serde_json::Value::as_array) {
            for property in required.iter().filter_map(serde_json::Value::as_str) {
                if !object.contains_key(property) {
                    anyhow::bail!("`{field}.{property}` is required");
                }
            }
        }

        let properties = schema
            .get("properties")
            .and_then(serde_json::Value::as_object);
        let pattern_properties = schema
            .get("patternProperties")
            .and_then(serde_json::Value::as_object)
            .map(|patterns| {
                patterns
                    .iter()
                    .map(|(pattern, property_schema)| {
                        regex::Regex::new(pattern)
                            .map(|regex| (regex, property_schema))
                            .map_err(|error| {
                                anyhow::anyhow!(
                                    "`{field}` has invalid patternProperties pattern {pattern:?}: {error}"
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();

        for (property, property_value) in object {
            let mut matched_schema = false;
            if let Some(property_schema) = properties.and_then(|values| values.get(property)) {
                validate_basic_json_schema(
                    property_value,
                    property_schema,
                    &format!("{field}.{property}"),
                )?;
                matched_schema = true;
            }

            for (pattern, property_schema) in &pattern_properties {
                if pattern.is_match(property) {
                    validate_basic_json_schema(
                        property_value,
                        property_schema,
                        &format!("{field}.{property}"),
                    )?;
                    matched_schema = true;
                }
            }

            if matched_schema {
                continue;
            }

            match schema.get("additionalProperties") {
                Some(serde_json::Value::Bool(false)) => {
                    anyhow::bail!("`{field}.{property}` is not allowed")
                }
                Some(additional_schema @ serde_json::Value::Object(_)) => {
                    validate_basic_json_schema(
                        property_value,
                        additional_schema,
                        &format!("{field}.{property}"),
                    )?;
                }
                _ => {}
            }
        }
    }

    if let Some(array) = value.as_array() {
        let prefix_len = if let Some(prefix_items) = schema
            .get("prefixItems")
            .and_then(serde_json::Value::as_array)
        {
            for (index, (item, item_schema)) in array.iter().zip(prefix_items.iter()).enumerate() {
                validate_basic_json_schema(item, item_schema, &format!("{field}[{index}]"))?;
            }
            prefix_items.len()
        } else {
            0
        };

        if let Some(item_schema) = schema.get("items") {
            for (index, item) in array.iter().enumerate().skip(prefix_len) {
                validate_basic_json_schema(item, item_schema, &format!("{field}[{index}]"))?;
            }
        }
    }

    Ok(())
}

fn matches_json_schema_type(value: &serde_json::Value, expected: &serde_json::Value) -> bool {
    match expected {
        serde_json::Value::String(expected) => matches_json_type(value, expected),
        serde_json::Value::Array(expected) => expected.iter().any(|expected| {
            expected
                .as_str()
                .is_some_and(|expected| matches_json_type(value, expected))
        }),
        // `type` itself is malformed, so leave schema-shape validation to a
        // future full JSON Schema validator instead of rejecting arguments.
        _ => true,
    }
}

fn matches_json_type(value: &serde_json::Value, expected: &str) -> bool {
    match expected {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => {
            value.as_i64().is_some()
                || value.as_u64().is_some()
                || value
                    .as_f64()
                    .is_some_and(|number| number.is_finite() && number.fract() == 0.0)
        }
        "string" => value.is_string(),
        // Unknown type names belong to schema validation, not argument
        // validation. Ignoring them preserves compatibility with extensions.
        _ => true,
    }
}

fn display_schema_type(expected: &serde_json::Value) -> String {
    match expected {
        serde_json::Value::String(expected) => format!("\"{expected}\""),
        serde_json::Value::Array(expected) => {
            let values = expected
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|value| format!("\"{value}\""))
                .collect::<Vec<_>>()
                .join(" or ");
            if values.is_empty() {
                "the declared JSON type".to_string()
            } else {
                values
            }
        }
        _ => "the declared JSON type".to_string(),
    }
}

fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Validates top_logprobs parameter
pub fn validate_top_logprobs(top_logprobs: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = top_logprobs
        && !(0..=20).contains(&value)
    {
        anyhow::bail!(
            "Top_logprobs must be between 0 and {}, got {}",
            MAX_TOP_LOGPROBS,
            value
        );
    }
    Ok(())
}

/// Validates tools array
pub fn validate_tools(
    tools: &Option<&[dynamo_protocols::types::ChatCompletionTool]>,
) -> Result<(), anyhow::Error> {
    let tools = match tools {
        Some(val) => val,
        None => return Ok(()),
    };

    if tools.len() > MAX_TOOLS {
        anyhow::bail!(
            "Maximum of {} tools are supported, got {}",
            MAX_TOOLS,
            tools.len()
        );
    }

    for (i, tool) in tools.iter().enumerate() {
        if tool.function.name.len() > MAX_FUNCTION_NAME_LENGTH {
            anyhow::bail!(
                "Function name at index {} exceeds {} character limit, got {} characters",
                i,
                MAX_FUNCTION_NAME_LENGTH,
                tool.function.name.len()
            );
        }
        if tool.function.name.trim().is_empty() {
            anyhow::bail!("Function name at index {} cannot be empty", i);
        }
        if !tool
            .function
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            anyhow::bail!(
                "Function at index {} has an invalid name: \"{}\". \
                 Only a-z, A-Z, 0-9, underscores, and dashes are allowed.",
                i,
                tool.function.name,
            );
        }
        if let Some(parameters) = &tool.function.parameters
            && !parameters.is_object()
        {
            anyhow::bail!(
                "Function parameters at index {} for \"{}\" must be a JSON Schema object",
                i,
                tool.function.name,
            );
        }
    }
    Ok(())
}

/// Validates that forced tool_choice requests refer to available tools.
pub fn validate_tool_choice(
    tool_choice: &Option<dynamo_protocols::types::ChatCompletionToolChoiceOption>,
    tools: Option<&[dynamo_protocols::types::ChatCompletionTool]>,
) -> Result<(), anyhow::Error> {
    use dynamo_protocols::types::ChatCompletionToolChoiceOption;

    match validate_openai_tool_choice(tool_choice.as_ref(), tools) {
        Ok(()) => Ok(()),
        Err(ToolChoiceError::EmptyTools) => {
            anyhow::bail!("tool_choice is \"required\" but tools is empty")
        }
        Err(ToolChoiceError::MissingTools) => match tool_choice {
            Some(ChatCompletionToolChoiceOption::Required) => {
                anyhow::bail!("tool_choice is \"required\" but tools is empty")
            }
            Some(ChatCompletionToolChoiceOption::Named(named)) => anyhow::bail!(
                "tool named \"{}\" in tool_choice is not present in tools",
                named.function.name
            ),
            _ => Err(ToolChoiceError::MissingTools.into()),
        },
        Err(ToolChoiceError::ToolNotFound(name)) => {
            anyhow::bail!("tool named \"{name}\" in tool_choice is not present in tools")
        }
        Err(error) => Err(error.into()),
    }
}

/// Validates reasoning effort parameter
pub fn validate_reasoning_effort(
    _reasoning_effort: &Option<dynamo_protocols::types::ReasoningEffort>,
) -> Result<(), anyhow::Error> {
    // TODO ADD HERE
    // ReasoningEffort is an enum, so if it exists, it's valid by definition
    // This function is here for completeness and future validation needs
    Ok(())
}

/// Validates service tier parameter
pub fn validate_service_tier(
    _service_tier: &Option<dynamo_protocols::types::ServiceTier>,
) -> Result<(), anyhow::Error> {
    // TODO ADD HERE
    // ServiceTier is an enum, so if it exists, it's valid by definition
    // This function is here for completeness and future validation needs
    Ok(())
}

//
// Completion Specific
//

/// Validates prompt
pub fn validate_prompt(prompt: &dynamo_protocols::types::Prompt) -> Result<(), anyhow::Error> {
    match prompt {
        dynamo_protocols::types::Prompt::String(s) => {
            if s.is_empty() {
                anyhow::bail!("Prompt string cannot be empty");
            }
        }
        dynamo_protocols::types::Prompt::StringArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt string array cannot be empty");
            }
            for (i, s) in arr.iter().enumerate() {
                if s.is_empty() {
                    anyhow::bail!("Prompt string at index {} cannot be empty", i);
                }
            }
        }
        dynamo_protocols::types::Prompt::IntegerArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt integer array cannot be empty");
            }
        }
        dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt array of integer arrays cannot be empty");
            }
            for (i, inner_arr) in arr.iter().enumerate() {
                if inner_arr.is_empty() {
                    anyhow::bail!("Prompt integer array at index {} cannot be empty", i);
                }
            }
        }
    }
    Ok(())
}

/// Validates prompt and prompt_embeds fields together.
///
/// This function consolidates all prompt-related validation:
/// - Ensures at least one of prompt or prompt_embeds is provided
/// - If prompt_embeds is provided, validates its format (base64, size limits)
/// - If prompt_embeds is NOT provided, validates that prompt is non-empty
///
/// Format for prompt_embeds: PyTorch tensor serialized with torch.save() and base64-encoded
pub fn validate_prompt_or_embeds(
    prompt: Option<&dynamo_protocols::types::Prompt>,
    prompt_embeds: Option<&str>,
) -> Result<(), anyhow::Error> {
    // Check that at least one is provided
    if prompt.is_none() && prompt_embeds.is_none() {
        anyhow::bail!("At least one of 'prompt' or 'prompt_embeds' must be provided");
    }

    // If prompt_embeds is provided, validate it
    if let Some(embeds) = prompt_embeds {
        validate_prompt_embeds_format(embeds)?;
    } else if let Some(p) = prompt {
        // Only validate prompt content if prompt_embeds is NOT provided
        // When embeddings are present, prompt can be empty/placeholder
        validate_prompt(p)?;
    }

    Ok(())
}

/// Validates prompt_embeds format (internal helper)
/// Format: PyTorch tensor serialized with torch.save() and base64-encoded
fn validate_prompt_embeds_format(embeds: &str) -> Result<(), anyhow::Error> {
    use base64::{Engine as _, engine::general_purpose};

    // Validate base64 encoding first
    let decoded = general_purpose::STANDARD
        .decode(embeds)
        .map_err(|_| anyhow::anyhow!("prompt_embeds must be valid base64-encoded data"))?;

    // Check minimum size on decoded bytes (100 bytes)
    const MIN_SIZE: usize = 100;
    if decoded.len() < MIN_SIZE {
        anyhow::bail!(
            "prompt_embeds decoded data must be at least {MIN_SIZE} bytes, got {} bytes",
            decoded.len()
        );
    }

    // Check maximum size on decoded bytes (10MB)
    const MAX_SIZE: usize = 10 * 1024 * 1024;
    if decoded.len() > MAX_SIZE {
        anyhow::bail!(
            "prompt_embeds decoded data exceeds maximum size of 10MB, got {} bytes",
            decoded.len()
        );
    }

    Ok(())
}

/// Validates prompt_embeds field (public wrapper for standalone validation)
/// Format: PyTorch tensor serialized with torch.save() and base64-encoded
pub fn validate_prompt_embeds(prompt_embeds: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(embeds) = prompt_embeds {
        validate_prompt_embeds_format(embeds)?;
    }
    Ok(())
}

/// Validates logprobs parameter (for completion requests)
pub fn validate_logprobs(logprobs: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = logprobs
        && !(MIN_LOGPROBS..=MAX_LOGPROBS).contains(&value)
    {
        anyhow::bail!(
            "Logprobs must be between 0 and {}, got {}",
            MAX_LOGPROBS,
            value
        );
    }
    Ok(())
}

/// Validates best_of parameter
pub fn validate_best_of(best_of: Option<u8>, n: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(best_of_value) = best_of {
        if !(MIN_BEST_OF..=MAX_BEST_OF).contains(&best_of_value) {
            anyhow::bail!(
                "Best_of must be between 0 and {}, got {}",
                MAX_BEST_OF,
                best_of_value
            );
        }

        if let Some(n_value) = n
            && best_of_value < n_value
        {
            anyhow::bail!(
                "Best_of must be greater than or equal to n, got best_of={} and n={}",
                best_of_value,
                n_value
            );
        }
    }
    Ok(())
}

/// Validates suffix parameter
pub fn validate_suffix(suffix: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(suffix_str) = suffix {
        // Suffix can be empty, but if it's very long it might cause issues
        if suffix_str.len() > 10000 {
            anyhow::bail!("Suffix is too long, maximum 10000 characters");
        }
    }
    Ok(())
}

const MAX_OUTPUT_TOKENS: u32 = 1_048_576;

/// Validates max_tokens parameter
pub fn validate_max_tokens(max_tokens: Option<u32>) -> Result<(), anyhow::Error> {
    if let Some(tokens) = max_tokens
        && tokens == 0
    {
        anyhow::bail!("Max tokens must be greater than 0, got {}", tokens);
    }
    if let Some(tokens) = max_tokens
        && tokens > MAX_OUTPUT_TOKENS
    {
        anyhow::bail!(
            "Max tokens must not exceed {}, got {}",
            MAX_OUTPUT_TOKENS,
            tokens
        );
    }
    Ok(())
}

/// Validates max_completion_tokens parameter
pub fn validate_max_completion_tokens(
    max_completion_tokens: Option<u32>,
) -> Result<(), anyhow::Error> {
    if let Some(tokens) = max_completion_tokens
        && tokens == 0
    {
        anyhow::bail!(
            "Max completion tokens must be greater than 0, got {}",
            tokens
        );
    }
    if let Some(tokens) = max_completion_tokens
        && tokens > MAX_OUTPUT_TOKENS
    {
        anyhow::bail!(
            "Max completion tokens must not exceed {}, got {}",
            MAX_OUTPUT_TOKENS,
            tokens
        );
    }
    Ok(())
}

//
// Helpers
//

pub fn validate_range<T>(value: Option<T>, range: &(T, T)) -> anyhow::Result<Option<T>>
where
    T: PartialOrd + Display,
{
    if value.is_none() {
        return Ok(None);
    }
    let value = value.unwrap();
    if value < range.0 || value > range.1 {
        anyhow::bail!("Value {} is out of range [{}, {}]", value, range.0, range.1);
    }
    Ok(Some(value))
}

/// A nested `chat_template` bypasses Dynamo's top-level rejection and is
/// promoted into the rendered template, so block it for every chat processor.
pub fn validate_chat_template_args(
    chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<(), anyhow::Error> {
    if let Some(args) = chat_template_args
        && args.contains_key("chat_template")
    {
        anyhow::bail!("`chat_template` is not supported inside `chat_template_args`");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;

    fn unknown_fields() -> HashMap<String, serde_json::Value> {
        HashMap::from([("experimental_field".to_string(), json!("value"))])
    }

    #[test]
    fn validate_chat_template_args_rejects_nested_chat_template() {
        let args = HashMap::from([(
            "chat_template".to_string(),
            json!("{% for _ in range(10**9) %}x{% endfor %}"),
        )]);
        let err = validate_chat_template_args(Some(&args)).unwrap_err();
        assert!(err.to_string().contains("chat_template"));
    }

    #[test]
    fn validate_chat_template_args_accepts_other_keys() {
        let args = HashMap::from([("enable_thinking".to_string(), json!(false))]);
        validate_chat_template_args(Some(&args)).unwrap();
        validate_chat_template_args(None).unwrap();
    }

    #[test]
    fn validate_response_format_rejects_null_json_schema() {
        let response_format = serde_json::from_value(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": null
            }
        }))
        .unwrap();

        let err = validate_response_format(&Some(response_format)).unwrap_err();
        assert!(err.to_string().contains("schema` is required"));
    }

    #[test]
    fn validate_no_unsupported_fields_accepts_logprob_token_ids() {
        let fields = HashMap::from([("logprob_token_ids".to_string(), json!([14, 15]))]);
        validate_no_unsupported_fields_with_ignore(&fields, false).unwrap();
    }

    #[test]
    fn validate_no_unsupported_fields_rejects_malformed_logprob_token_ids() {
        for bad in [json!(["notanint"]), json!(7), json!([[1, 2]]), json!([-1])] {
            let fields = HashMap::from([("logprob_token_ids".to_string(), bad)]);
            let err = validate_no_unsupported_fields_with_ignore(&fields, false).unwrap_err();
            assert!(err.to_string().contains("must be an array of token IDs"));
        }
    }

    #[test]
    fn validate_no_unsupported_fields_rejects_unknown_fields_by_default() {
        let err = validate_no_unsupported_fields_with_ignore(&unknown_fields(), false).unwrap_err();
        assert!(err.to_string().contains("Unsupported parameter(s)"));
    }

    #[test]
    fn validate_no_unsupported_fields_ignores_unknown_fields_when_configured() {
        validate_no_unsupported_fields_with_ignore(&unknown_fields(), true).unwrap();
    }

    #[test]
    fn validate_no_unsupported_fields_still_validates_passthrough_fields_when_ignoring_unknowns() {
        let unsupported_fields = HashMap::from([
            ("experimental_field".to_string(), json!("value")),
            ("stop_token_ids".to_string(), json!("bad")),
        ]);

        let err =
            validate_no_unsupported_fields_with_ignore(&unsupported_fields, true).unwrap_err();
        assert!(err.to_string().contains("stop_token_ids"));
    }

    #[test]
    fn validate_top_p_rejects_zero() {
        let err = validate_top_p(Some(0.0)).unwrap_err();
        assert!(err.to_string().contains("Top_p"));
    }

    #[test]
    fn validate_top_p_accepts_valid_values() {
        validate_top_p(Some(0.1)).unwrap();
        validate_top_p(Some(1.0)).unwrap();
        validate_top_p(None).unwrap();
    }

    #[test]
    fn validate_response_format_rejects_non_object_schema() {
        let fmt = serde_json::from_value(json!({
            "type": "json_schema",
            "json_schema": { "name": "test", "schema": 42 }
        }))
        .unwrap();
        let err = validate_response_format(&Some(fmt)).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"));
    }

    #[test]
    fn validate_response_format_accepts_valid_object_schema() {
        let fmt = serde_json::from_value(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": { "type": "object", "properties": {} }
            }
        }))
        .unwrap();
        validate_response_format(&Some(fmt)).unwrap();
    }
}
