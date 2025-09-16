// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;

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
/// Allowed range of values for OpenAI's `top_p` sampling option
pub const TOP_P_RANGE: (f32, f32) = (MIN_TOP_P, MAX_TOP_P);

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

/// Minimum allowed value for OpenAI's `logit_bias` values
pub const MIN_LOGIT_BIAS: f32 = -100.0;
/// Maximum allowed value for OpenAI's `logit_bias` values
pub const MAX_LOGIT_BIAS: f32 = 100.0;

/// Minimum allowed value for `best_of`
pub const MIN_BEST_OF: u8 = 0;
/// Maximum allowed value for `best_of`
pub const MAX_BEST_OF: u8 = 20;

/// Maximum allowed number of stop sequences
pub const MAX_STOP_SEQUENCES: usize = 4;
/// Maximum allowed number of tools
pub const MAX_TOOLS: usize = 128;
/// Maximum allowed number of metadata key-value pairs
pub const MAX_METADATA_PAIRS: usize = 16;
/// Maximum allowed length for metadata keys
pub const MAX_METADATA_KEY_LENGTH: usize = 64;
/// Maximum allowed length for metadata values
pub const MAX_METADATA_VALUE_LENGTH: usize = 512;
/// Maximum allowed length for function names
pub const MAX_FUNCTION_NAME_LENGTH: usize = 64;
/// Maximum allowed value for Prompt IntegerArray elements
pub const MAX_PROMPT_TOKEN_ID: u32 = 50256;
/// Minimum allowed value for `repetition_penalty`
pub const MIN_REPETITION_PENALTY: f32 = 0.0;
/// Maximum allowed value for `repetition_penalty`
pub const MAX_REPETITION_PENALTY: f32 = 2.0;

//
// Shared Fields
//

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
        && !(MIN_TOP_P..=MAX_TOP_P).contains(&p)
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
    if let Some(penalty) = repetition_penalty
        && !(MIN_REPETITION_PENALTY..=MAX_REPETITION_PENALTY).contains(&penalty)
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
pub fn validate_stop(stop: &Option<dynamo_async_openai::types::Stop>) -> Result<(), anyhow::Error> {
    if let Some(stop_value) = stop {
        match stop_value {
            dynamo_async_openai::types::Stop::String(s) => {
                if s.is_empty() {
                    anyhow::bail!("Stop sequence cannot be empty");
                }
            }
            dynamo_async_openai::types::Stop::StringArray(sequences) => {
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
        }
    }
    Ok(())
}

//
// Chat Completion Specific
//

/// Validates messages array
pub fn validate_messages(
    messages: &[dynamo_async_openai::types::ChatCompletionRequestMessage],
) -> Result<(), anyhow::Error> {
    if messages.is_empty() {
        anyhow::bail!("Messages array cannot be empty");
    }
    Ok(())
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
    tools: &Option<&[dynamo_async_openai::types::ChatCompletionTool]>,
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
    }
    Ok(())
}

/// Validates metadata
pub fn validate_metadata(metadata: &Option<serde_json::Value>) -> Result<(), anyhow::Error> {
    let metadata = match metadata {
        Some(val) => val,
        None => return Ok(()),
    };

    if let Some(obj) = metadata.as_object() {
        if obj.len() > MAX_METADATA_PAIRS {
            anyhow::bail!(
                "Metadata cannot have more than {} key-value pairs, got {}",
                MAX_METADATA_PAIRS,
                obj.len()
            );
        }

        for (key, value) in obj {
            if key.len() > MAX_METADATA_KEY_LENGTH {
                anyhow::bail!(
                    "Metadata key '{}' exceeds {} character limit",
                    key,
                    MAX_METADATA_KEY_LENGTH
                );
            }

            if let Some(value_str) = value.as_str()
                && value_str.len() > MAX_METADATA_VALUE_LENGTH
            {
                anyhow::bail!(
                    "Metadata value for key '{}' exceeds {} character limit",
                    key,
                    MAX_METADATA_VALUE_LENGTH
                );
            }
        }
    }
    Ok(())
}

/// Validates reasoning effort parameter
pub fn validate_reasoning_effort(
    _reasoning_effort: &Option<dynamo_async_openai::types::ReasoningEffort>,
) -> Result<(), anyhow::Error> {
    // TODO ADD HERE
    // ReasoningEffort is an enum, so if it exists, it's valid by definition
    // This function is here for completeness and future validation needs
    Ok(())
}

/// Validates service tier parameter
pub fn validate_service_tier(
    _service_tier: &Option<dynamo_async_openai::types::ServiceTier>,
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
pub fn validate_prompt(prompt: &dynamo_async_openai::types::Prompt) -> Result<(), anyhow::Error> {
    match prompt {
        dynamo_async_openai::types::Prompt::String(s) => {
            if s.is_empty() {
                anyhow::bail!("Prompt string cannot be empty");
            }
        }
        dynamo_async_openai::types::Prompt::StringArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt string array cannot be empty");
            }
            for (i, s) in arr.iter().enumerate() {
                if s.is_empty() {
                    anyhow::bail!("Prompt string at index {} cannot be empty", i);
                }
            }
        }
        dynamo_async_openai::types::Prompt::IntegerArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt integer array cannot be empty");
            }
            for (i, &token_id) in arr.iter().enumerate() {
                if token_id > MAX_PROMPT_TOKEN_ID {
                    anyhow::bail!(
                        "Token ID at index {} must be between 0 and {}, got {}",
                        i,
                        MAX_PROMPT_TOKEN_ID,
                        token_id
                    );
                }
            }
        }
        dynamo_async_openai::types::Prompt::ArrayOfIntegerArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt array of integer arrays cannot be empty");
            }
            for (i, inner_arr) in arr.iter().enumerate() {
                if inner_arr.is_empty() {
                    anyhow::bail!("Prompt integer array at index {} cannot be empty", i);
                }
                for (j, &token_id) in inner_arr.iter().enumerate() {
                    if token_id > MAX_PROMPT_TOKEN_ID {
                        anyhow::bail!(
                            "Token ID at index [{}][{}] must be between 0 and {}, got {}",
                            i,
                            j,
                            MAX_PROMPT_TOKEN_ID,
                            token_id
                        );
                    }
                }
            }
        }
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

/// Validates max_tokens parameter
pub fn validate_max_tokens(max_tokens: Option<u32>) -> Result<(), anyhow::Error> {
    if let Some(tokens) = max_tokens
        && tokens == 0
    {
        anyhow::bail!("Max tokens must be greater than 0, got {}", tokens);
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
