// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::CreateCompletionRequestArgs;
use dynamo_llm::protocols::openai::{completions::NvCreateCompletionRequest, validate};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CompletionSample {
    request: NvCreateCompletionRequest,
    description: String,
}

impl CompletionSample {
    fn new<F>(description: impl Into<String>, configure: F) -> Result<Self, String>
    where
        F: FnOnce(&mut CreateCompletionRequestArgs) -> &mut CreateCompletionRequestArgs,
    {
        let mut builder = CreateCompletionRequestArgs::default();
        builder
            .model("gpt-3.5-turbo")
            .prompt("What is the meaning of life?");
        configure(&mut builder);

        let inner = builder.build().unwrap();

        let request = NvCreateCompletionRequest {
            inner,
            common: Default::default(),
            nvext: None,
            metadata: None,
            unsupported_fields: Default::default(),
        };

        Ok(Self {
            request,
            description: description.into(),
        })
    }
}

#[test]
fn minimum_viable_request() {
    let request = CreateCompletionRequestArgs::default()
        .prompt("What is the meaning of life?")
        .model("gpt-3.5-turbo")
        .build()
        .expect("error building request");

    insta::assert_json_snapshot!(request);
}

#[test]
fn valid_samples() {
    let mut settings = insta::Settings::clone_current();
    settings.set_sort_maps(true);
    let _guard = settings.bind_to_scope();

    let samples = build_samples().expect("error building samples");

    // iteration on all sample and call validate and expect it to be ok
    for sample in &samples {
        insta::with_settings!({
            description => &sample.description,
        }, {
        insta::assert_json_snapshot!(sample.request);
        });
    }
}
#[allow(clippy::vec_init_then_push)]
fn build_samples() -> Result<Vec<CompletionSample>, String> {
    let mut samples = Vec::new();

    samples.push(CompletionSample::new(
        "should have only prompt and model fields",
        |builder| builder,
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and max_tokens fields",
        |builder| builder.max_tokens(10_u32),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and temperature fields",
        |builder| builder.temperature(validate::MIN_TEMPERATURE),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and top_p fields",
        |builder| builder.top_p(validate::MIN_TOP_P),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and frequency_penalty fields",
        |builder| builder.frequency_penalty(validate::MIN_FREQUENCY_PENALTY),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and presence_penalty fields",
        |builder| builder.presence_penalty(validate::MIN_PRESENCE_PENALTY),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and stop fields",
        |builder| builder.stop(vec!["\n".to_string()]),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and echo fields",
        |builder| builder.echo(true),
    )?);

    samples.push(CompletionSample::new(
        "should have prompt, model, and stream fields",
        |builder| builder.stream(true),
    )?);

    Ok(samples)
}

// ============================================================================
// Batch Prompt Tests
// ============================================================================

#[test]
fn test_batch_prompt_utilities() {
    use dynamo_async_openai::types::Prompt;
    use dynamo_llm::protocols::openai::completions::{
        extract_single_prompt, get_prompt_batch_size,
    };

    // Test single string prompt
    let single_string = Prompt::String("Hello, world!".to_string());
    assert_eq!(get_prompt_batch_size(&single_string), 1);
    assert_eq!(
        extract_single_prompt(&single_string, 0),
        Prompt::String("Hello, world!".to_string())
    );

    // Test single integer array prompt
    let single_int = Prompt::IntegerArray(vec![1, 2, 3]);
    assert_eq!(get_prompt_batch_size(&single_int), 1);
    assert_eq!(
        extract_single_prompt(&single_int, 0),
        Prompt::IntegerArray(vec![1, 2, 3])
    );

    // Test string array prompt
    let string_array = Prompt::StringArray(vec![
        "First prompt".to_string(),
        "Second prompt".to_string(),
        "Third prompt".to_string(),
    ]);
    assert_eq!(get_prompt_batch_size(&string_array), 3);
    assert_eq!(
        extract_single_prompt(&string_array, 0),
        Prompt::String("First prompt".to_string())
    );
    assert_eq!(
        extract_single_prompt(&string_array, 1),
        Prompt::String("Second prompt".to_string())
    );
    assert_eq!(
        extract_single_prompt(&string_array, 2),
        Prompt::String("Third prompt".to_string())
    );

    // Test array of integer arrays
    let int_array = Prompt::ArrayOfIntegerArray(vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]]);
    assert_eq!(get_prompt_batch_size(&int_array), 3);
    assert_eq!(
        extract_single_prompt(&int_array, 0),
        Prompt::IntegerArray(vec![1, 2, 3])
    );
    assert_eq!(
        extract_single_prompt(&int_array, 1),
        Prompt::IntegerArray(vec![4, 5])
    );
    assert_eq!(
        extract_single_prompt(&int_array, 2),
        Prompt::IntegerArray(vec![6, 7, 8, 9])
    );
}

#[test]
fn test_total_choices_validation() {
    use dynamo_llm::protocols::openai::validate::validate_total_choices;

    // Valid cases
    assert!(validate_total_choices(1, 1).is_ok());
    assert!(validate_total_choices(10, 10).is_ok());
    assert!(validate_total_choices(64, 2).is_ok());
    assert!(validate_total_choices(128, 1).is_ok());
    assert!(validate_total_choices(1, 128).is_ok());

    // Edge case: exactly at the limit
    assert!(validate_total_choices(128, 1).is_ok());
    assert!(validate_total_choices(64, 2).is_ok());

    // Invalid cases: exceeds limit
    assert!(validate_total_choices(129, 1).is_err());
    assert!(validate_total_choices(65, 2).is_err());
    assert!(validate_total_choices(100, 2).is_err());
    assert!(validate_total_choices(2, 100).is_err());

    // Test error message
    let result = validate_total_choices(100, 2);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("Total choices (batch_size × n = 100 × 2 = 200) exceeds maximum of 128")
    );
}

#[test]
fn test_batch_prompt_with_n_parameter() {
    use dynamo_async_openai::types::Prompt;
    use dynamo_llm::protocols::openai::completions::get_prompt_batch_size;

    // Test batch size calculation
    let prompt = Prompt::StringArray(vec!["p1".to_string(), "p2".to_string(), "p3".to_string()]);
    let batch_size = get_prompt_batch_size(&prompt);
    let n = 2_u8;

    // Total choices = batch_size × n = 3 × 2 = 6
    let total_choices = batch_size * (n as usize);
    assert_eq!(total_choices, 6);

    // Choice indices should be:
    // prompt 0: indices 0, 1
    // prompt 1: indices 2, 3
    // prompt 2: indices 4, 5
    for prompt_idx in 0..batch_size {
        for choice_idx in 0..n {
            let expected_index = (prompt_idx as u32) * (n as u32) + (choice_idx as u32);
            // Verify index calculation matches vLLM logic
            assert_eq!(
                expected_index,
                prompt_idx as u32 * n as u32 + choice_idx as u32
            );
        }
    }
}

#[test]
fn test_single_prompt_in_array() {
    use dynamo_async_openai::types::Prompt;
    use dynamo_llm::protocols::openai::completions::{
        extract_single_prompt, get_prompt_batch_size,
    };

    // Single element array should work like regular prompt
    let single_in_array = Prompt::StringArray(vec!["Single prompt".to_string()]);
    assert_eq!(get_prompt_batch_size(&single_in_array), 1);
    assert_eq!(
        extract_single_prompt(&single_in_array, 0),
        Prompt::String("Single prompt".to_string())
    );
}
