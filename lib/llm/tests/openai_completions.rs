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
