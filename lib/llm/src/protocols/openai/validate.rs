// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
/// Maximum length of model name suffix
pub const MAX_SUFFIX_LEN: usize = 64;
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

//
// Shared Fields
//

/// Validates the temperature parameter
pub fn validate_temperature(temperature: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(temp) = temperature {
        if !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temp) {
            anyhow::bail!(
                "Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, got {temp}"
            );
        }
    }
    Ok(())
}

/// Validates the top_p parameter
pub fn validate_top_p(top_p: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(p) = top_p {
        if !(MIN_TOP_P..=MAX_TOP_P).contains(&p) {
            anyhow::bail!("Top_p must be between {MIN_TOP_P} and {MAX_TOP_P}, got {p}",);
        }
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
    if let Some(penalty) = frequency_penalty {
        if !(MIN_FREQUENCY_PENALTY..=MAX_FREQUENCY_PENALTY).contains(&penalty) {
            anyhow::bail!(
                "Frequency penalty must be between {MIN_FREQUENCY_PENALTY} and {MAX_FREQUENCY_PENALTY}, got {penalty}",
            );
        }
    }
    Ok(())
}

/// Validates presence penalty parameter
pub fn validate_presence_penalty(presence_penalty: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(penalty) = presence_penalty {
        if !(MIN_PRESENCE_PENALTY..=MAX_PRESENCE_PENALTY).contains(&penalty) {
            anyhow::bail!(
                "Presence penalty must be between {MIN_PRESENCE_PENALTY} and {MAX_PRESENCE_PENALTY}, got {penalty}",
            );
        }
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
                "Logit bias value for token '{token}' must be a number, got {bias_value}",
            )
        })? as f32;

        if !(MIN_LOGIT_BIAS..=MAX_LOGIT_BIAS).contains(&bias) {
            anyhow::bail!(
                "Logit bias for token '{token}' must be between {MIN_LOGIT_BIAS} and {MAX_LOGIT_BIAS}, got {bias}",
            );
        }
    }
    Ok(())
}

/// Validates n parameter (number of choices)
pub fn validate_n(n: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = n {
        if !(MIN_N..=MAX_N).contains(&value) {
            anyhow::bail!("n must be between {MIN_N} and {MAX_N}, got {value}");
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
    if let Some(user_id) = user {
        if user_id.trim().is_empty() {
            anyhow::bail!("User ID cannot be empty");
        }
    }
    Ok(())
}

/// Validates stop sequences
pub fn validate_stop(stop: &Option<async_openai::types::Stop>) -> Result<(), anyhow::Error> {
    if let Some(stop_value) = stop {
        match stop_value {
            async_openai::types::Stop::String(s) => {
                if s.is_empty() {
                    anyhow::bail!("Stop sequence cannot be empty");
                }
            }
            async_openai::types::Stop::StringArray(sequences) => {
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
                        anyhow::bail!("Stop sequence at index {i} cannot be empty");
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
    messages: &[async_openai::types::ChatCompletionRequestMessage],
) -> Result<(), anyhow::Error> {
    if messages.is_empty() {
        anyhow::bail!("Messages array cannot be empty");
    }
    Ok(())
}

/// Validates top_logprobs parameter
pub fn validate_top_logprobs(top_logprobs: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = top_logprobs {
        if !(0..=20).contains(&value) {
            anyhow::bail!("Top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}, got {value}");
        }
    }
    Ok(())
}

/// Validates tools array
pub fn validate_tools(
    tools: &Option<&[async_openai::types::ChatCompletionTool]>,
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
            anyhow::bail!("Function name at index {i} cannot be empty");
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

    let Some(obj) = metadata.as_object() else {
        return Ok(());
    };
    if obj.len() > MAX_METADATA_PAIRS {
        anyhow::bail!(
            "Metadata cannot have more than {} key-value pairs, got {}",
            MAX_METADATA_PAIRS,
            obj.len()
        );
    }

    for (key, value) in obj {
        if key.len() > MAX_METADATA_KEY_LENGTH {
            anyhow::bail!("Metadata key '{key}' exceeds {MAX_METADATA_KEY_LENGTH} character limit",);
        }

        if let Some(value_str) = value.as_str() {
            if value_str.len() > MAX_METADATA_VALUE_LENGTH {
                anyhow::bail!(
                    "Metadata value for key '{key}' exceeds {MAX_METADATA_VALUE_LENGTH} character limit",
                );
            }
        }
    }
    Ok(())
}

/// Validates reasoning effort parameter
pub fn validate_reasoning_effort(
    _reasoning_effort: &Option<async_openai::types::ReasoningEffort>,
) -> Result<(), anyhow::Error> {
    // ReasoningEffort is an enum, so if it exists, it's valid by definition
    // This function is here for completeness and future validation needs
    Ok(())
}

/// Validates service tier parameter
pub fn validate_service_tier(
    _service_tier: &Option<async_openai::types::ServiceTier>,
) -> Result<(), anyhow::Error> {
    // ServiceTier is an enum, so if it exists, it's valid by definition
    // This function is here for completeness and future validation needs
    Ok(())
}

//
// Completion Specific
//

/// Validates prompt
pub fn validate_prompt(prompt: &async_openai::types::Prompt) -> Result<(), anyhow::Error> {
    match prompt {
        async_openai::types::Prompt::String(s) => {
            if s.is_empty() {
                anyhow::bail!("Prompt string cannot be empty");
            }
        }
        async_openai::types::Prompt::StringArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt string array cannot be empty");
            }
            for (i, s) in arr.iter().enumerate() {
                if s.is_empty() {
                    anyhow::bail!("Prompt string at index {i} cannot be empty");
                }
            }
        }
        async_openai::types::Prompt::IntegerArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt integer array cannot be empty");
            }
            for (i, &token_id) in arr.iter().enumerate() {
                if token_id > MAX_PROMPT_TOKEN_ID {
                    anyhow::bail!(
                        "Token ID at index {i} must be between 0 and {MAX_PROMPT_TOKEN_ID}, got {token_id}",
                    );
                }
            }
        }
        async_openai::types::Prompt::ArrayOfIntegerArray(arr) => {
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
                            "Token ID at index [{i}][{j}] must be between 0 and {MAX_PROMPT_TOKEN_ID}, got {token_id}",
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
    if let Some(value) = logprobs {
        if !(MIN_LOGPROBS..=MAX_LOGPROBS).contains(&value) {
            anyhow::bail!("Logprobs must be between 0 and {MAX_LOGPROBS}, got {value}",);
        }
    }
    Ok(())
}

/// Validates best_of parameter
pub fn validate_best_of(best_of: Option<u8>, n: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(best_of_value) = best_of {
        if !(MIN_BEST_OF..=MAX_BEST_OF).contains(&best_of_value) {
            anyhow::bail!("Best_of must be between 0 and {MAX_BEST_OF}, got {best_of_value}",);
        }

        if let Some(n_value) = n {
            if best_of_value < n_value {
                anyhow::bail!(
                    "Best_of must be greater than or equal to n, got best_of={best_of_value} and n={n_value}",
                );
            }
        }
    }
    Ok(())
}

/// Validates suffix parameter
pub fn validate_suffix(suffix: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(suffix_str) = suffix {
        if suffix_str.len() > MAX_SUFFIX_LEN {
            anyhow::bail!("Suffix is too long, maximum of {MAX_SUFFIX_LEN} characters");
        }
    }
    Ok(())
}

/// Validates max_tokens parameter
pub fn validate_max_tokens(max_tokens: Option<u32>) -> Result<(), anyhow::Error> {
    if let Some(tokens) = max_tokens {
        if tokens == 0 {
            anyhow::bail!("Max tokens must be greater than 0, got {tokens}");
        }
    }
    Ok(())
}

/// Validates max_completion_tokens parameter
pub fn validate_max_completion_tokens(
    max_completion_tokens: Option<u32>,
) -> Result<(), anyhow::Error> {
    if let Some(tokens) = max_completion_tokens {
        if tokens == 0 {
            anyhow::bail!("Max completion tokens must be greater than 0, got {tokens}");
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{
        ChatCompletionRequestMessage, Prompt, ReasoningEffort, ServiceTier, Stop,
    };
    use std::collections::HashMap;

    #[test]
    fn test_validate_temperature_valid() {
        assert!(validate_temperature(None).is_ok());
        assert!(validate_temperature(Some(0.0)).is_ok());
        assert!(validate_temperature(Some(1.0)).is_ok());
        assert!(validate_temperature(Some(2.0)).is_ok());
        assert!(validate_temperature(Some(0.7)).is_ok());
    }

    #[test]
    fn test_validate_temperature_invalid() {
        assert!(validate_temperature(Some(-0.1)).is_err());
        assert!(validate_temperature(Some(2.1)).is_err());
        assert!(validate_temperature(Some(-1.0)).is_err());
        assert!(validate_temperature(Some(3.0)).is_err());
    }

    #[test]
    fn test_validate_top_p_valid() {
        assert!(validate_top_p(None).is_ok());
        assert!(validate_top_p(Some(0.0)).is_ok());
        assert!(validate_top_p(Some(0.5)).is_ok());
        assert!(validate_top_p(Some(1.0)).is_ok());
    }

    #[test]
    fn test_validate_top_p_invalid() {
        assert!(validate_top_p(Some(-0.1)).is_err());
        assert!(validate_top_p(Some(1.1)).is_err());
        assert!(validate_top_p(Some(-1.0)).is_err());
        assert!(validate_top_p(Some(2.0)).is_err());
    }

    #[test]
    fn test_validate_temperature_top_p_exclusion_valid() {
        assert!(validate_temperature_top_p_exclusion(None, None).is_ok());
        assert!(validate_temperature_top_p_exclusion(Some(0.7), None).is_ok());
        assert!(validate_temperature_top_p_exclusion(None, Some(0.9)).is_ok());
        assert!(validate_temperature_top_p_exclusion(Some(1.0), Some(0.9)).is_ok());
        assert!(validate_temperature_top_p_exclusion(Some(0.7), Some(1.0)).is_ok());
        assert!(validate_temperature_top_p_exclusion(Some(1.0), Some(1.0)).is_ok());
    }

    #[test]
    fn test_validate_temperature_top_p_exclusion_invalid() {
        assert!(validate_temperature_top_p_exclusion(Some(0.7), Some(0.9)).is_err());
        assert!(validate_temperature_top_p_exclusion(Some(0.5), Some(0.5)).is_err());
    }

    #[test]
    fn test_validate_frequency_penalty_valid() {
        assert!(validate_frequency_penalty(None).is_ok());
        assert!(validate_frequency_penalty(Some(-2.0)).is_ok());
        assert!(validate_frequency_penalty(Some(0.0)).is_ok());
        assert!(validate_frequency_penalty(Some(2.0)).is_ok());
        assert!(validate_frequency_penalty(Some(0.5)).is_ok());
    }

    #[test]
    fn test_validate_frequency_penalty_invalid() {
        assert!(validate_frequency_penalty(Some(-2.1)).is_err());
        assert!(validate_frequency_penalty(Some(2.1)).is_err());
        assert!(validate_frequency_penalty(Some(-5.0)).is_err());
        assert!(validate_frequency_penalty(Some(5.0)).is_err());
    }

    #[test]
    fn test_validate_presence_penalty_valid() {
        assert!(validate_presence_penalty(None).is_ok());
        assert!(validate_presence_penalty(Some(-2.0)).is_ok());
        assert!(validate_presence_penalty(Some(0.0)).is_ok());
        assert!(validate_presence_penalty(Some(2.0)).is_ok());
        assert!(validate_presence_penalty(Some(1.5)).is_ok());
    }

    #[test]
    fn test_validate_presence_penalty_invalid() {
        assert!(validate_presence_penalty(Some(-2.1)).is_err());
        assert!(validate_presence_penalty(Some(2.1)).is_err());
        assert!(validate_presence_penalty(Some(-3.0)).is_err());
        assert!(validate_presence_penalty(Some(3.0)).is_err());
    }

    #[test]
    fn test_validate_logit_bias_valid() {
        assert!(validate_logit_bias(&None).is_ok());

        let mut valid_bias = HashMap::new();
        valid_bias.insert(
            "50256".to_string(),
            serde_json::Value::Number(serde_json::Number::from(-100)),
        );
        valid_bias.insert(
            "198".to_string(),
            serde_json::Value::Number(serde_json::Number::from(100)),
        );
        valid_bias.insert(
            "42".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0)),
        );
        assert!(validate_logit_bias(&Some(valid_bias)).is_ok());
    }

    #[test]
    fn test_validate_logit_bias_invalid() {
        let mut invalid_bias = HashMap::new();
        invalid_bias.insert(
            "50256".to_string(),
            serde_json::Value::Number(serde_json::Number::from(-101)),
        );
        assert!(validate_logit_bias(&Some(invalid_bias)).is_err());

        let mut invalid_bias2 = HashMap::new();
        invalid_bias2.insert(
            "50256".to_string(),
            serde_json::Value::Number(serde_json::Number::from(101)),
        );
        assert!(validate_logit_bias(&Some(invalid_bias2)).is_err());

        let mut invalid_bias3 = HashMap::new();
        invalid_bias3.insert(
            "50256".to_string(),
            serde_json::Value::String("invalid".to_string()),
        );
        assert!(validate_logit_bias(&Some(invalid_bias3)).is_err());
    }

    #[test]
    fn test_validate_n_valid() {
        assert!(validate_n(None).is_ok());
        assert!(validate_n(Some(1)).is_ok());
        assert!(validate_n(Some(64)).is_ok());
        assert!(validate_n(Some(128)).is_ok());
    }

    #[test]
    fn test_validate_n_invalid() {
        assert!(validate_n(Some(0)).is_err());
        assert!(validate_n(Some(129)).is_err());
        assert!(validate_n(Some(255)).is_err());
    }

    #[test]
    fn test_validate_model_valid() {
        assert!(validate_model("gpt-3.5-turbo").is_ok());
        assert!(validate_model("gpt-4").is_ok());
        assert!(validate_model("claude-3").is_ok());
        assert!(validate_model("  model-with-spaces  ").is_ok()); // trimmed
    }

    #[test]
    fn test_validate_model_invalid() {
        assert!(validate_model("").is_err());
        assert!(validate_model("   ").is_err()); // only whitespace
        assert!(validate_model("\t\n").is_err()); // only whitespace
    }

    #[test]
    fn test_validate_user_valid() {
        assert!(validate_user(None).is_ok());
        assert!(validate_user(Some("user123")).is_ok());
        assert!(validate_user(Some("user@example.com")).is_ok());
    }

    #[test]
    fn test_validate_user_invalid() {
        assert!(validate_user(Some("")).is_err());
        assert!(validate_user(Some("   ")).is_err());
        assert!(validate_user(Some("\t")).is_err());
    }

    #[test]
    fn test_validate_stop_valid() {
        assert!(validate_stop(&None).is_ok());
        assert!(validate_stop(&Some(Stop::String("STOP".to_string()))).is_ok());
        assert!(validate_stop(&Some(Stop::StringArray(vec![
            "STOP".to_string(),
            "END".to_string()
        ])))
        .is_ok());
        assert!(validate_stop(&Some(Stop::StringArray(vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string()
        ])))
        .is_ok());
    }

    #[test]
    fn test_validate_stop_invalid() {
        assert!(validate_stop(&Some(Stop::String("".to_string()))).is_err());
        assert!(validate_stop(&Some(Stop::StringArray(vec![]))).is_err());
        assert!(validate_stop(&Some(Stop::StringArray(vec![
            "A".to_string(),
            "".to_string()
        ])))
        .is_err());
        assert!(validate_stop(&Some(Stop::StringArray(vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string()
        ])))
        .is_err()); // Too many
    }

    #[test]
    fn test_validate_messages_valid() {
        let messages = vec![ChatCompletionRequestMessage::User(
            async_openai::types::ChatCompletionRequestUserMessage {
                content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    "Hello".to_string(),
                ),
                name: None,
            },
        )];
        assert!(validate_messages(&messages).is_ok());
    }

    #[test]
    fn test_validate_messages_invalid() {
        assert!(validate_messages(&[]).is_err());
    }

    #[test]
    fn test_validate_top_logprobs_valid() {
        assert!(validate_top_logprobs(None).is_ok());
        assert!(validate_top_logprobs(Some(0)).is_ok());
        assert!(validate_top_logprobs(Some(10)).is_ok());
        assert!(validate_top_logprobs(Some(20)).is_ok());
    }

    #[test]
    fn test_validate_top_logprobs_invalid() {
        assert!(validate_top_logprobs(Some(21)).is_err());
        assert!(validate_top_logprobs(Some(255)).is_err());
    }

    #[test]
    fn test_validate_metadata_valid() {
        assert!(validate_metadata(&None).is_ok());

        let metadata = serde_json::json!({
            "user_id": "123",
            "session": "abc"
        });
        assert!(validate_metadata(&Some(metadata)).is_ok());
    }

    #[test]
    fn test_validate_metadata_invalid() {
        let too_many_pairs: serde_json::Value = serde_json::json!({
            "key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4",
            "key5": "value5", "key6": "value6", "key7": "value7", "key8": "value8",
            "key9": "value9", "key10": "value10", "key11": "value11", "key12": "value12",
            "key13": "value13", "key14": "value14", "key15": "value15", "key16": "value16",
            "key17": "value17" // 17 pairs, max is 16
        });
        assert!(validate_metadata(&Some(too_many_pairs)).is_err());

        let long_key = serde_json::json!({
            "a".repeat(65): "value" // Key too long
        });
        assert!(validate_metadata(&Some(long_key)).is_err());

        let long_value = serde_json::json!({
            "key": "a".repeat(513) // Value too long
        });
        assert!(validate_metadata(&Some(long_value)).is_err());
    }

    #[test]
    fn test_validate_reasoning_effort() {
        assert!(validate_reasoning_effort(&None).is_ok());
        assert!(validate_reasoning_effort(&Some(ReasoningEffort::Low)).is_ok());
        assert!(validate_reasoning_effort(&Some(ReasoningEffort::Medium)).is_ok());
        assert!(validate_reasoning_effort(&Some(ReasoningEffort::High)).is_ok());
    }

    #[test]
    fn test_validate_service_tier() {
        assert!(validate_service_tier(&None).is_ok());
        assert!(validate_service_tier(&Some(ServiceTier::Auto)).is_ok());
        assert!(validate_service_tier(&Some(ServiceTier::Default)).is_ok());
    }

    #[test]
    fn test_validate_prompt_valid() {
        assert!(validate_prompt(&Prompt::String("Hello world".to_string())).is_ok());
        assert!(validate_prompt(&Prompt::StringArray(vec![
            "Hello".to_string(),
            "world".to_string()
        ]))
        .is_ok());
        assert!(validate_prompt(&Prompt::IntegerArray(vec![1, 2, 3, 4, 50256])).is_ok());
        assert!(
            validate_prompt(&Prompt::ArrayOfIntegerArray(vec![vec![1, 2], vec![3, 4]])).is_ok()
        );
    }

    #[test]
    fn test_validate_prompt_invalid() {
        assert!(validate_prompt(&Prompt::String("".to_string())).is_err());
        assert!(validate_prompt(&Prompt::StringArray(vec![])).is_err());
        assert!(validate_prompt(&Prompt::StringArray(vec![
            "Hello".to_string(),
            "".to_string()
        ]))
        .is_err());
        assert!(validate_prompt(&Prompt::IntegerArray(vec![])).is_err());
        assert!(validate_prompt(&Prompt::IntegerArray(vec![1, 2, 50257])).is_err()); // Token ID too high
        assert!(validate_prompt(&Prompt::ArrayOfIntegerArray(vec![])).is_err());
        assert!(validate_prompt(&Prompt::ArrayOfIntegerArray(vec![vec![]])).is_err());
        assert!(validate_prompt(&Prompt::ArrayOfIntegerArray(vec![vec![1, 50257]])).is_err());
        // Token ID too high
    }

    #[test]
    fn test_validate_logprobs_valid() {
        assert!(validate_logprobs(None).is_ok());
        assert!(validate_logprobs(Some(0)).is_ok());
        assert!(validate_logprobs(Some(3)).is_ok());
        assert!(validate_logprobs(Some(5)).is_ok());
    }

    #[test]
    fn test_validate_logprobs_invalid() {
        assert!(validate_logprobs(Some(6)).is_err());
        assert!(validate_logprobs(Some(10)).is_err());
        assert!(validate_logprobs(Some(255)).is_err());
    }

    #[test]
    fn test_validate_best_of_valid() {
        assert!(validate_best_of(None, None).is_ok());
        assert!(validate_best_of(Some(1), Some(1)).is_ok());
        assert!(validate_best_of(Some(5), Some(3)).is_ok());
        assert!(validate_best_of(Some(20), None).is_ok());
        assert!(validate_best_of(Some(0), None).is_ok()); // Edge case: 0 is allowed
    }

    #[test]
    fn test_validate_best_of_invalid() {
        assert!(validate_best_of(Some(21), None).is_err()); // Too high
        assert!(validate_best_of(Some(3), Some(5)).is_err()); // best_of < n
        assert!(validate_best_of(Some(255), None).is_err());
    }

    #[test]
    fn test_validate_suffix_valid() {
        assert!(validate_suffix(None).is_ok());
        assert!(validate_suffix(Some("")).is_ok()); // Empty is allowed
        assert!(validate_suffix(Some("test suffix")).is_ok());
        assert!(validate_suffix(Some(&"a".repeat(64))).is_ok()); // At limit
    }

    #[test]
    fn test_validate_suffix_invalid() {
        assert!(validate_suffix(Some(&"a".repeat(65))).is_err()); // Too long
    }

    #[test]
    fn test_validate_max_tokens_valid() {
        assert!(validate_max_tokens(None).is_ok());
        assert!(validate_max_tokens(Some(1)).is_ok());
        assert!(validate_max_tokens(Some(100)).is_ok());
        assert!(validate_max_tokens(Some(4096)).is_ok());
    }

    #[test]
    fn test_validate_max_tokens_invalid() {
        assert!(validate_max_tokens(Some(0)).is_err());
    }

    #[test]
    fn test_validate_max_completion_tokens_valid() {
        assert!(validate_max_completion_tokens(None).is_ok());
        assert!(validate_max_completion_tokens(Some(1)).is_ok());
        assert!(validate_max_completion_tokens(Some(100)).is_ok());
        assert!(validate_max_completion_tokens(Some(4096)).is_ok());
    }

    #[test]
    fn test_validate_max_completion_tokens_invalid() {
        assert!(validate_max_completion_tokens(Some(0)).is_err());
    }

    #[test]
    fn test_validate_range_valid() {
        assert_eq!(validate_range(None::<f32>, &(0.0, 1.0)).unwrap(), None);
        assert_eq!(validate_range(Some(0.5), &(0.0, 1.0)).unwrap(), Some(0.5));
        assert_eq!(validate_range(Some(0.0), &(0.0, 1.0)).unwrap(), Some(0.0));
        assert_eq!(validate_range(Some(1.0), &(0.0, 1.0)).unwrap(), Some(1.0));
    }

    #[test]
    fn test_validate_range_invalid() {
        assert!(validate_range(Some(-0.1), &(0.0, 1.0)).is_err());
        assert!(validate_range(Some(1.1), &(0.0, 1.0)).is_err());
        assert!(validate_range(Some(-1.0), &(0.0, 1.0)).is_err());
        assert!(validate_range(Some(2.0), &(0.0, 1.0)).is_err());
    }

    #[test]
    fn test_boundary_values() {
        // Test exact boundary values
        assert!(validate_temperature(Some(MIN_TEMPERATURE)).is_ok());
        assert!(validate_temperature(Some(MAX_TEMPERATURE)).is_ok());
        assert!(validate_top_p(Some(MIN_TOP_P)).is_ok());
        assert!(validate_top_p(Some(MAX_TOP_P)).is_ok());
        assert!(validate_frequency_penalty(Some(MIN_FREQUENCY_PENALTY)).is_ok());
        assert!(validate_frequency_penalty(Some(MAX_FREQUENCY_PENALTY)).is_ok());
        assert!(validate_presence_penalty(Some(MIN_PRESENCE_PENALTY)).is_ok());
        assert!(validate_presence_penalty(Some(MAX_PRESENCE_PENALTY)).is_ok());
        assert!(validate_n(Some(MIN_N)).is_ok());
        assert!(validate_n(Some(MAX_N)).is_ok());
        assert!(validate_logprobs(Some(MIN_LOGPROBS)).is_ok());
        assert!(validate_logprobs(Some(MAX_LOGPROBS)).is_ok());
        assert!(validate_top_logprobs(Some(MIN_TOP_LOGPROBS)).is_ok());
        assert!(validate_top_logprobs(Some(MAX_TOP_LOGPROBS)).is_ok());
        assert!(validate_best_of(Some(MIN_BEST_OF), None).is_ok());
        assert!(validate_best_of(Some(MAX_BEST_OF), None).is_ok());
    }
}
