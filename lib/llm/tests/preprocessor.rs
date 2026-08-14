// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Ok, Result};
use dynamo_runtime::config::environment_names::model::huggingface as env_hf;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_renderer::PromptContextMixin;
use dynamo_renderer::PromptFormatter;
use serde::{Deserialize, Serialize};

use hf_hub::{Cache, Repo, RepoType, api::tokio::ApiBuilder};
use rstest::rstest;

use std::path::PathBuf;

/// Gets the HF_TOKEN environment variable if it exists and is not empty.
///
/// These tests require a Hugging Face token to be set in the environment variable `HF_TOKEN`.
/// The model is downloaded and cached in `tests/data/sample-models` directory.
/// Make sure the token has access to `meta-llama/Llama-3.1-70B-Instruct` model.
///
/// This function checks for the presence of the `HF_TOKEN` environment variable
/// and validates that it's not empty or whitespace-only. The token is used for
/// downloading models from Hugging Face to a local cache directory in
/// `tests/data/sample-models`. These tests require a Hugging Face token to be
/// set in the environment variable `HF_TOKEN`. The model is downloaded and
/// cached in `tests/data/sample-models` directory.
///
/// # Returns
///
/// - `Ok(String)` - The token value if it exists and is not empty
/// - `Err(anyhow::Error)` - An error if the token is missing or empty
///
/// # Errors
///
/// - Returns an error if `HF_TOKEN` environment variable is not set
/// - Returns an error if `HF_TOKEN` environment variable is empty or whitespace-only
fn get_hf_token() -> Result<String> {
    let token = std::env::var(env_hf::HF_TOKEN)
        .map_err(|_| anyhow::anyhow!("HF_TOKEN environment variable is not set"))?;

    if token.trim().is_empty() {
        anyhow::bail!("HF_TOKEN environment variable is empty");
    }

    Ok(token)
}

async fn make_mdc_from_repo(
    local_path: &str,
    hf_repo: &str,
    hf_revision: &str,
    mixins: Option<Vec<PromptContextMixin>>,
) -> ModelDeploymentCard {
    let downloaded_path = maybe_download_model(local_path, hf_repo, hf_revision).await;
    let display_name = format!("{}--{}", hf_repo, hf_revision);
    let mut mdc = ModelDeploymentCard::load_from_disk(downloaded_path, None).unwrap();
    mdc.set_name(&display_name);
    mdc.prompt_context = mixins;
    mdc
}

async fn maybe_download_model(local_path: &str, model: &str, revision: &str) -> String {
    let cache = Cache::new(PathBuf::from(local_path));

    // Use check_hf_token for consistency with the rest of the codebase
    let token = get_hf_token().expect("HF_TOKEN is required to download models from Hugging Face");

    let api = ApiBuilder::from_cache(cache)
        .with_progress(false)
        .with_token(Some(token))
        .build()
        .unwrap();
    let repo = Repo::with_revision(String::from(model), RepoType::Model, String::from(revision));

    let files_to_download = vec!["config.json", "tokenizer.json", "tokenizer_config.json"];
    let optional_files = vec![
        "generation_config.json",
        "chat_template.jinja",
        "chat_template.json",
    ];
    let repo_builder = api.repo(repo);

    let mut downloaded_path = PathBuf::new();
    for file in &files_to_download {
        downloaded_path = repo_builder.get(file).await.unwrap();
    }
    for file in &optional_files {
        if let Err(e) = repo_builder.get(file).await {
            println!(
                "Failed to download optional file {} for model {}: {}",
                file, model, e
            );
        }
    }
    downloaded_path.parent().unwrap().display().to_string()
}

async fn make_mdcs() -> Vec<ModelDeploymentCard> {
    vec![
        make_mdc_from_repo(
            "tests/data/sample-models",
            "meta-llama/Llama-3.1-70B-Instruct",
            "1605565",
            Some(vec![PromptContextMixin::Llama3DateTime]),
        )
        .await,
    ]
}

const SINGLE_CHAT_MESSAGE: &str = r#"
[
    {
      "role": "user",
      "content": "What is deep learning?"
    }
]
"#;

/// Sample Message with `user` and `assistant`, no `system`
const THREE_TURN_CHAT_MESSAGE: &str = r#"
[
    {
      "role": "user",
      "content": "How do I reverse a string in Python?"
    },
    {
      "role": "assistant",
      "content": "You can reverse a string in Python using slicing:\n\n```python\nreversed_string = your_string[::-1]\n```\n\nAlternatively, you can use `reversed()` with `join()`:\n\n```python\nreversed_string = ''.join(reversed(your_string))\n```\n"
    },
    {
      "role": "user",
      "content": "What if I want to reverse each word in a sentence but keep their order?"
    }
]"#;

/// Sample Message with `user` and `assistant`, no `system`
const THREE_TURN_CHAT_MESSAGE_WITH_SYSTEM: &str = r#"
[
    {
      "role": "system",
      "content": "You are a very helpful assistant!"
    },
    {
      "role": "user",
      "content": "How do I reverse a string in Python?"
    },
    {
      "role": "assistant",
      "content": "You can reverse a string in Python using slicing:\n\n```python\nreversed_string = your_string[::-1]\n```\n\nAlternatively, you can use `reversed()` with `join()`:\n\n```python\nreversed_string = ''.join(reversed(your_string))\n```\n"
    },
    {
      "role": "user",
      "content": "What if I want to reverse each word in a sentence but keep their order?"
    }
]"#;

/// Sample Message with `user` and `assistant`, no `system`
const MULTI_TURN_WITH_CONTINUATION: &str = r#"
[
    {
      "role": "system",
      "content": "You are a very helpful assistant!"
    },
    {
      "role": "user",
      "content": "How do I reverse a string in Python?"
    },
    {
      "role": "assistant",
      "content": "You can reverse a "
    }
]"#;

const TOOLS: &str = r#"
[
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature for a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g., San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["Celsius", "Fahrenheit"],
              "description": "The temperature unit to use. Infer this from the user's location."
            }
          },
          "required": ["location", "unit"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_rain_probability",
        "description": "Get the probability of rain for a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g., San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
"#;

// Notes:
// protocols::openai::chat_completions::ChatCompletionMessage -> dynamo_protocols::types::ChatCompletionRequestMessage
// protocols::openai::chat_completions::Tool -> dynamo_protocols::types::ChatCompletionTool
// protocols::openai::chat_completions::ToolChoiceType -> dynamo_protocols::types::ChatCompletionToolChoiceOption
#[derive(Serialize, Deserialize)]
struct Request {
    messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage>,
    tools: Option<Vec<dynamo_protocols::types::ChatCompletionTool>>,
    tool_choice: Option<dynamo_protocols::types::ChatCompletionToolChoiceOption>,
}

impl Request {
    fn from(
        messages: &str,
        tools: Option<&str>,
        tool_choice: Option<dynamo_protocols::types::ChatCompletionToolChoiceOption>,
        model: String,
    ) -> NvCreateChatCompletionRequest {
        let messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage> =
            serde_json::from_str(messages).unwrap();
        let tools: Option<Vec<dynamo_protocols::types::ChatCompletionTool>> =
            tools.map(|x| serde_json::from_str(x).unwrap());
        //let tools = tools.unwrap();
        //let tool_choice = tool_choice.unwrap();

        let mut inner = dynamo_protocols::types::CreateChatCompletionRequestArgs::default();
        inner.model(model);
        inner.messages(messages);
        if let Some(tools) = tools {
            inner.tools(tools);
        }
        if let Some(tool_choice) = tool_choice {
            inner.tool_choice(tool_choice);
        }
        let inner = inner.build().unwrap();

        NvCreateChatCompletionRequest {
            inner,
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }
}

#[tokio::test]
async fn test_single_turn() {
    if let Err(e) = get_hf_token() {
        println!("HF_TOKEN is not set, skipping test: {}", e);
        return;
    }
    let mdcs = make_mdcs().await;

    for mdc in mdcs.iter() {
        let formatter = prompt_formatter_from_mdc(mdc).unwrap();

        // assert its an OAI formatter
        let formatter = match formatter {
            PromptFormatter::OAI(formatter) => Ok(formatter),
        }
        .unwrap();

        let request = Request::from(SINGLE_CHAT_MESSAGE, None, None, mdc.slug().to_string());
        let formatted_prompt = formatter.render(&request).unwrap();

        insta::with_settings!({
          info => &request,
          snapshot_suffix => mdc.slug().to_string(),
          filters => vec![
            (r"Today Date: .*", "Today Date: <redacted>"),
          ]
        }, {
          insta::assert_snapshot!(formatted_prompt);
        });
    }
}

#[tokio::test]
async fn test_single_turn_with_tools() {
    if let Err(e) = get_hf_token() {
        println!("HF_TOKEN is not set, skipping test: {}", e);
        return;
    }
    let mdcs = make_mdcs().await;

    for mdc in mdcs.iter() {
        let formatter = prompt_formatter_from_mdc(mdc).unwrap();

        // assert its an OAI formatter
        let formatter = match formatter {
            PromptFormatter::OAI(formatter) => Ok(formatter),
        }
        .unwrap();

        let request = Request::from(
            SINGLE_CHAT_MESSAGE,
            Some(TOOLS),
            Some(dynamo_protocols::types::ChatCompletionToolChoiceOption::Auto),
            mdc.slug().to_string(),
        );
        let formatted_prompt = formatter.render(&request).unwrap();

        insta::with_settings!({
          info => &request,
          snapshot_suffix => mdc.slug().to_string(),
          filters => vec![
            (r"Today Date: .*", "Today Date: <redacted>"),
          ]
        }, {
          insta::assert_snapshot!(formatted_prompt);
        });
    }
}

#[tokio::test]
async fn test_mulit_turn_without_system() {
    if let Err(e) = get_hf_token() {
        println!("HF_TOKEN is not set, skipping test: {}", e);
        return;
    }
    let mdcs = make_mdcs().await;

    for mdc in mdcs.iter() {
        let formatter = prompt_formatter_from_mdc(mdc).unwrap();

        // assert its an OAI formatter
        let formatter = match formatter {
            PromptFormatter::OAI(formatter) => Ok(formatter),
        }
        .unwrap();

        let request = Request::from(THREE_TURN_CHAT_MESSAGE, None, None, mdc.slug().to_string());
        let formatted_prompt = formatter.render(&request).unwrap();

        insta::with_settings!({
          info => &request,
          snapshot_suffix => mdc.slug().to_string(),
          filters => vec![
            (r"Today Date: .*", "Today Date: <redacted>"),
          ]
        }, {
          insta::assert_snapshot!(formatted_prompt);
        });
    }
}

#[tokio::test]
async fn test_mulit_turn_with_system() {
    if let Err(e) = get_hf_token() {
        println!("HF_TOKEN is not set, skipping test: {}", e);
        return;
    }
    let mdcs = make_mdcs().await;

    for mdc in mdcs.iter() {
        let formatter = prompt_formatter_from_mdc(mdc).unwrap();

        // assert its an OAI formatter
        let formatter = match formatter {
            PromptFormatter::OAI(formatter) => Ok(formatter),
        }
        .unwrap();

        let request = Request::from(
            THREE_TURN_CHAT_MESSAGE_WITH_SYSTEM,
            None,
            None,
            mdc.slug().to_string(),
        );
        let formatted_prompt = formatter.render(&request).unwrap();

        insta::with_settings!({
          info => &request,
          snapshot_suffix => mdc.slug().to_string(),
          filters => vec![
            (r"Today Date: .*", "Today Date: <redacted>"),
          ]
        }, {
          insta::assert_snapshot!(formatted_prompt);
        });
    }
}

/// Test the prompt formatter with a multi-turn conversation that includes system message and tools
#[tokio::test]
async fn test_multi_turn_with_system_with_tools() {
    if let Err(e) = get_hf_token() {
        println!("HF_TOKEN is not set, skipping test: {}", e);
        return;
    }
    let mdcs = make_mdcs().await;

    for mdc in mdcs.iter() {
        let formatter = prompt_formatter_from_mdc(mdc).unwrap();

        // assert its an OAI formatter
        let formatter = match formatter {
            PromptFormatter::OAI(formatter) => Ok(formatter),
        }
        .unwrap();

        let request = Request::from(
            THREE_TURN_CHAT_MESSAGE_WITH_SYSTEM,
            Some(TOOLS),
            Some(dynamo_protocols::types::ChatCompletionToolChoiceOption::Auto),
            mdc.slug().to_string(),
        );
        let formatted_prompt = formatter.render(&request).unwrap();

        insta::with_settings!({
          info => &request,
          snapshot_suffix => mdc.slug().to_string(),
          filters => vec![
            (r"Today Date: .*", "Today Date: <redacted>"),
          ]
        }, {
          insta::assert_snapshot!(formatted_prompt);
        });
    }
}

/// Test the prompt formatter with a multi-turn conversation that includes a continuation
#[tokio::test]
async fn test_multi_turn_with_continuation() {
    if let Err(e) = get_hf_token() {
        println!("HF_TOKEN is not set, skipping test: {}", e);
        return;
    }
    let mdc = make_mdc_from_repo(
        "tests/data/sample-models",
        "meta-llama/Llama-3.1-70B-Instruct",
        "1605565",
        Some(vec![PromptContextMixin::Llama3DateTime]),
    )
    .await;

    let formatter = prompt_formatter_from_mdc(&mdc).unwrap();

    // assert its an OAI formatter
    let formatter = match formatter {
        PromptFormatter::OAI(formatter) => Ok(formatter),
    }
    .unwrap();

    let request = Request::from(
        MULTI_TURN_WITH_CONTINUATION,
        None,
        None,
        mdc.slug().to_string(),
    );
    let formatted_prompt = formatter.render(&request).unwrap();

    insta::with_settings!({
      info => &request,
      snapshot_suffix => mdc.slug().to_string(),
      filters => vec![
        (r"Today Date: .*", "Today Date: <redacted>"),
      ]
    }, {
      insta::assert_snapshot!(formatted_prompt);
    });
}

pub mod openai_preprocessor_tests {
    // re-export all the tests from the parent module
    pub use super::*;
    use std::collections::HashSet;

    #[tokio::test]
    async fn test_stop_condition() {
        if let Err(e) = get_hf_token() {
            println!("HF_TOKEN is not set, skipping test: {}", e);
            return;
        }
        let mdc = make_mdc_from_repo(
            "tests/data/sample-models",
            "openai/gpt-oss-120b",
            "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
            Some(vec![PromptContextMixin::OaiChat]),
        )
        .await;

        let oai_preprocessor = OpenAIPreprocessor::new(mdc.clone()).unwrap();
        let request = Request::from(SINGLE_CHAT_MESSAGE, None, None, mdc.slug().to_string());
        let preprocessed_request = oai_preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap()
            .0;
        assert!(
            preprocessed_request
                .stop_conditions
                .stop_token_ids_hidden
                .is_some()
        );
        // eos_token_ids can be in any order as long as the set is correct
        let eos_token_id_set: HashSet<_> = preprocessed_request
            .stop_conditions
            .stop_token_ids_hidden
            .unwrap()
            .iter()
            .cloned()
            .collect();
        assert_eq!(
            eos_token_id_set,
            vec![200002, 199999, 200012].into_iter().collect()
        );
    }
}

// Helper to build message with media chunks (single or mixed types)
fn build_message(text: &str, chunks: &[(&str, usize)]) -> String {
    let mut content_parts = vec![format!(r#"{{"type": "text", "text": "{}"}}"#, text)];

    for (chunk_type, count) in chunks {
        for i in 1..=*count {
            let chunk = match *chunk_type {
                "image_url" => format!(
                    r#"{{"type": "image_url", "image_url": {{"url": "https://example.com/img{}.jpg"}}}}"#,
                    i
                ),
                "video_url" => format!(
                    r#"{{"type": "video_url", "video_url": {{"url": "https://example.com/vid{}.mp4"}}}}"#,
                    i
                ),
                "audio_url" => format!(
                    r#"{{"type": "audio_url", "audio_url": {{"url": "https://example.com/audio{}.mp3"}}}}"#,
                    i
                ),
                _ => panic!("Unknown chunk type: {}", chunk_type),
            };
            content_parts.push(chunk);
        }
    }

    format!(
        r#"[{{"role": "user", "content": [{}]}}]"#,
        content_parts.join(", ")
    )
}

/// Test the preprocessor with multimodal data (single and mixed types) to verify gather_multi_modal_data code path
#[rstest]
// No media case
#[case::no_media(&[])]
// Single media item cases
#[case::single_video(&[("video_url", 1)])]
// Multiple media items of the same type
#[case::three_images(&[("image_url", 3)])]
// Mixed media types
#[case::mixed_multiple(&[("image_url", 2), ("video_url", 1), ("audio_url", 2)])]
#[tokio::test]
async fn test_media_url_passthrough(#[case] media_chunks: &[(&str, usize)]) {
    if let Err(e) = get_hf_token() {
        println!("HF_TOKEN is not set, skipping test: {}", e);
        return;
    }

    let mdcs = make_mdcs().await;

    for mdc in mdcs.iter() {
        let preprocessor = dynamo_llm::preprocessor::OpenAIPreprocessor::new(mdc.clone()).unwrap();

        // Build the message with the specified media chunks
        let message = build_message("Test multimodal content", media_chunks);
        let request = Request::from(&message, None, None, mdc.slug().to_string());

        let (preprocessed, _annotations, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap();

        // Verify multimodal data handling
        if media_chunks.is_empty() {
            // No media case - should be None or empty
            assert!(
                preprocessed.multi_modal_data.is_none()
                    || preprocessed.multi_modal_data.as_ref().unwrap().is_empty(),
                "Multimodal data should be None or empty when no media is present"
            );
        } else {
            // Media present - should be captured
            assert!(
                preprocessed.multi_modal_data.is_some(),
                "Multimodal data should be present"
            );
            let media_map = preprocessed.multi_modal_data.as_ref().unwrap();

            // Check each media type and count
            for (media_type, expected_count) in media_chunks {
                assert!(
                    media_map.contains_key(*media_type),
                    "Should contain {} key",
                    media_type
                );
                assert_eq!(
                    media_map.get(*media_type).unwrap().len(),
                    *expected_count,
                    "Should have {} {} item(s)",
                    expected_count,
                    media_type
                );
            }
        }
    }
}

mod cached_multimodal_uuid {
    use std::sync::Arc;

    use super::Request;
    use dynamo_llm::model_card::ModelDeploymentCard;
    use dynamo_llm::preprocessor::OpenAIPreprocessor;
    use dynamo_llm::protocols::common::preprocessor::MultimodalData;
    use dynamo_runtime::error::{DynamoError, ErrorType};

    const MODEL_PATH: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

    fn make_preprocessor() -> Arc<OpenAIPreprocessor> {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        mdc.set_name("test-model");
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn assert_invalid_argument(error: &anyhow::Error) {
        let dynamo_error = error
            .chain()
            .find_map(|source| source.downcast_ref::<DynamoError>())
            .expect("error chain should contain DynamoError");
        assert_eq!(dynamo_error.error_type(), ErrorType::InvalidArgument);
    }

    #[tokio::test]
    async fn preserves_url_and_uuid_only_slot_alignment() {
        let messages = r#"[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/first.png"},
                        "uuid": "image-a"
                    },
                    {
                        "type": "image_url",
                        "image_url": null,
                        "uuid": "image-b"
                    }
                ]
            }
        ]"#;
        let request = Request::from(messages, None, None, "test-model".to_string());

        let preprocessor = make_preprocessor();

        #[cfg(feature = "mm-routing")]
        {
            let mut builder = dynamo_llm::preprocessor::PreprocessedRequest::builder();
            let mm_image_entries = preprocessor
                .gather_multi_modal_data(&request, &mut builder, None, &[])
                .await
                .unwrap();
            assert!(mm_image_entries.is_empty());
        }

        let (preprocessed, _, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap();

        let images = &preprocessed.multi_modal_data.as_ref().unwrap()["image_url"];
        assert_eq!(images.len(), 2);
        assert!(matches!(images[0], MultimodalData::Url(_)));
        assert!(matches!(
            &images[1],
            MultimodalData::UuidOnly(value) if value == "image-b"
        ));
        assert_eq!(
            preprocessed.multi_modal_uuids.as_ref().unwrap()["image_url"],
            vec![Some("image-a".to_string()), Some("image-b".to_string())]
        );
        assert!(
            preprocessed
                .extra_args
                .as_ref()
                .and_then(|args| args.get("mm_hashes"))
                .is_none()
        );
        assert!(preprocessed.mm_routing_info.is_none());
    }

    #[tokio::test]
    async fn rejects_audio_and_video_cache_uuids() {
        for messages in [
            r#"[{
                "role": "user",
                "content": [{
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/video.mp4"},
                    "uuid": "cached-video"
                }]
            }]"#,
            r#"[{
                "role": "user",
                "content": [{
                    "type": "audio_url",
                    "audio_url": {"url": "https://example.com/audio.wav"},
                    "uuid": "cached-audio"
                }]
            }]"#,
        ] {
            let request = Request::from(messages, None, None, "test-model".to_string());
            let error = make_preprocessor()
                .preprocess_request(&request, None)
                .await
                .expect_err("audio and video cache UUIDs must be rejected");

            assert!(
                format!("{error:#}").contains("supported only for image_url parts with vLLM"),
                "unexpected error: {error:#}"
            );
            assert_invalid_argument(&error);
        }
    }

    #[tokio::test]
    async fn rejects_media_part_without_url_or_uuid() {
        let messages = r#"[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}}
                ]
            }
        ]"#;
        let request = Request::from(messages, None, None, "test-model".to_string());

        let error = make_preprocessor()
            .preprocess_request(&request, None)
            .await
            .expect_err("media without a URL or UUID must be rejected");

        let error_chain = format!("{error:#}");
        assert!(
            error_chain.contains("neither `url` nor `uuid`"),
            "unexpected error: {error_chain}"
        );
        assert_invalid_argument(&error);
    }

    #[tokio::test]
    async fn rejects_empty_media_uuid() {
        let messages = r#"[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                        "uuid": ""
                    }
                ]
            }
        ]"#;
        let request = Request::from(messages, None, None, "test-model".to_string());

        let error = make_preprocessor()
            .preprocess_request(&request, None)
            .await
            .expect_err("an empty media UUID must be rejected");

        let error_chain = format!("{error:#}");
        assert!(
            error_chain.contains("uuid must be a non-empty string"),
            "unexpected error: {error_chain}"
        );
        assert_invalid_argument(&error);
    }
}

mod context_length_validation {
    use dynamo_llm::local_model::runtime_config::{TOKEN_BUDGET_RUNTIME_KEY, TokenBudget};
    use dynamo_llm::model_card::ModelDeploymentCard;
    use dynamo_llm::preprocessor::OpenAIPreprocessor;
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use dynamo_runtime::error::{DynamoError, ErrorType};

    // mock-llama has a chat_template in tokenizer_config.json (required for preprocessing)
    const MODEL_PATH: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

    fn make_chat_request(message: &str, model: &str) -> NvCreateChatCompletionRequest {
        make_chat_request_with_token_limits(message, model, None, None)
    }

    fn make_chat_request_with_max_tokens(
        message: &str,
        model: &str,
        max_tokens: Option<u32>,
    ) -> NvCreateChatCompletionRequest {
        make_chat_request_with_token_limits(message, model, max_tokens, None)
    }

    fn make_chat_request_with_token_limits(
        message: &str,
        model: &str,
        max_tokens: Option<u32>,
        max_completion_tokens: Option<u32>,
    ) -> NvCreateChatCompletionRequest {
        let messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage> =
            serde_json::from_str(message).unwrap();
        let mut builder = dynamo_protocols::types::CreateChatCompletionRequestArgs::default();
        builder.model(model).messages(messages);
        if let Some(max_tokens) = max_tokens {
            builder.max_tokens(max_tokens);
        }
        if let Some(max_completion_tokens) = max_completion_tokens {
            builder.max_completion_tokens(max_completion_tokens);
        }
        let inner = builder.build().unwrap();
        NvCreateChatCompletionRequest {
            inner,
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }

    async fn model_card_and_prompt_len(message: &str) -> (ModelDeploymentCard, u32) {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        mdc.runtime_config.context_length = Some(131072);
        let preprocessor = OpenAIPreprocessor::new(mdc.clone()).unwrap();
        let request = make_chat_request(message, "test-model");
        let (preprocessed, _, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap();

        (mdc, preprocessed.token_ids.len() as u32)
    }

    fn set_token_budget(
        mdc: &mut ModelDeploymentCard,
        combined_limit: u32,
        reject_prompt_overflow: bool,
        reject_total_overflow: bool,
    ) {
        mdc.runtime_config.context_length = Some(combined_limit);
        mdc.runtime_config
            .set_engine_specific(
                TOKEN_BUDGET_RUNTIME_KEY,
                TokenBudget {
                    combined_limit,
                    reject_prompt_overflow,
                    reject_total_overflow,
                },
            )
            .unwrap();
    }

    fn set_rejecting_token_budget(mdc: &mut ModelDeploymentCard, combined_limit: u32) {
        set_token_budget(mdc, combined_limit, true, true);
    }

    #[tokio::test]
    async fn test_prompt_exceeding_advertised_limit_returns_400() {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        set_rejecting_token_budget(&mut mdc, 5);

        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request = make_chat_request(
            r#"[{"role": "user", "content": "What is deep learning?"}]"#,
            "test-model",
        );

        let result = preprocessor.preprocess_request(&request, None).await;

        // Should fail with a DynamoError with InvalidArgument type
        let err = result.expect_err("should reject prompt exceeding the advertised limit");
        let dynamo_err = err
            .downcast_ref::<DynamoError>()
            .expect("error should be DynamoError");
        assert_eq!(dynamo_err.error_type(), ErrorType::InvalidArgument);
        assert!(
            dynamo_err
                .message()
                .contains("maximum context length is 5 tokens"),
            "error message should state the context limit, got: {}",
            dynamo_err.message()
        );
        assert!(
            dynamo_err.message().contains("Please reduce the length"),
            "error message should tell user what to do, got: {}",
            dynamo_err.message()
        );
    }

    #[tokio::test]
    async fn test_prompt_exactly_at_advertised_limit_returns_400() {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        // First, preprocess with a large context_length to discover the token count
        mdc.runtime_config.context_length = Some(131072);
        let preprocessor = OpenAIPreprocessor::new(mdc.clone()).unwrap();
        let request = make_chat_request(
            r#"[{"role": "user", "content": "What is deep learning?"}]"#,
            "test-model",
        );
        let (preprocessed, _, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap();
        let token_count = preprocessed.token_ids.len() as u32;

        // The prompt fills the combined budget, leaving no room for output.
        set_rejecting_token_budget(&mut mdc, token_count);
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request = make_chat_request(
            r#"[{"role": "user", "content": "What is deep learning?"}]"#,
            "test-model",
        );

        let result = preprocessor.preprocess_request(&request, None).await;

        // Should reject: prompt fills entire context, no room for output
        let err = result.expect_err("should reject prompt that fills the combined limit");
        let dynamo_err = err
            .downcast_ref::<DynamoError>()
            .expect("error should be DynamoError");
        assert_eq!(dynamo_err.error_type(), ErrorType::InvalidArgument);
    }

    #[tokio::test]
    async fn test_missing_token_budget_defers_prompt_validation() {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        // Missing capability metadata always delegates validation to the backend.
        mdc.runtime_config.context_length = Some(1);

        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request = make_chat_request(
            r#"[{"role": "user", "content": "What is deep learning?"}]"#,
            "test-model",
        );

        let (preprocessed, _, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .expect("missing token budget should delegate validation");
        assert!(
            preprocessed.stop_conditions.max_tokens.is_none(),
            "delegation should preserve an omitted output limit"
        );
    }

    #[tokio::test]
    async fn test_requested_tokens_exceeding_context_length_returns_400() {
        let message = r#"[{"role": "user", "content": "What is deep learning?"}]"#;
        let (mut mdc, prompt_len) = model_card_and_prompt_len(message).await;

        // Exceed the engine-published total context budget by one token.
        set_rejecting_token_budget(&mut mdc, prompt_len + 10);
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request = make_chat_request_with_token_limits(message, "test-model", Some(1), Some(11));

        let err = preprocessor
            .preprocess_request(&request, None)
            .await
            .expect_err("should reject input plus output exceeding context_length");
        let dynamo_err = err
            .downcast_ref::<DynamoError>()
            .expect("error should be DynamoError");
        assert_eq!(dynamo_err.error_type(), ErrorType::InvalidArgument);
        assert!(
            dynamo_err.message().contains(&format!(
                "{} input tokens and asks for 11 output tokens",
                prompt_len
            )),
            "error message should state the requested token budget, got: {}",
            dynamo_err.message()
        );
    }

    #[tokio::test]
    async fn test_requested_tokens_exactly_at_context_length_succeeds() {
        let message = r#"[{"role": "user", "content": "What is deep learning?"}]"#;
        let (mut mdc, prompt_len) = model_card_and_prompt_len(message).await;

        set_rejecting_token_budget(&mut mdc, prompt_len + 10);
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request =
            make_chat_request_with_token_limits(message, "test-model", Some(11), Some(10));

        assert!(
            preprocessor
                .preprocess_request(&request, None)
                .await
                .is_ok(),
            "input plus output exactly equal to context_length should be accepted"
        );
    }

    #[tokio::test]
    async fn test_omitted_output_budget_uses_engine_limit() {
        let message = r#"[{"role": "user", "content": "What is deep learning?"}]"#;
        let (mut mdc, prompt_len) = model_card_and_prompt_len(message).await;

        set_rejecting_token_budget(&mut mdc, prompt_len + 10);
        // The raw model context is larger because the engine reserves tokens.
        mdc.runtime_config.context_length = Some(prompt_len + 100);
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request = make_chat_request(message, "test-model");

        let (preprocessed, _, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .unwrap();

        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(10));
    }

    #[tokio::test]
    async fn test_missing_token_budget_defers_total_validation() {
        let message = r#"[{"role": "user", "content": "What is deep learning?"}]"#;
        let (mut mdc, prompt_len) = model_card_and_prompt_len(message).await;

        mdc.runtime_config.context_length = Some(prompt_len + 10);
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request = make_chat_request_with_max_tokens(message, "test-model", Some(11));

        assert!(
            preprocessor
                .preprocess_request(&request, None)
                .await
                .is_ok(),
            "without a token-budget policy, total-token validation should defer to the backend"
        );
    }

    #[tokio::test]
    async fn test_malformed_token_budget_defers_validation() {
        let message = r#"[{"role": "user", "content": "What is deep learning?"}]"#;
        let (mut mdc, prompt_len) = model_card_and_prompt_len(message).await;
        mdc.runtime_config.context_length = Some(prompt_len + 10);
        mdc.runtime_config.runtime_data.insert(
            TOKEN_BUDGET_RUNTIME_KEY.to_string(),
            serde_json::json!("not-a-token-budget"),
        );

        let preprocessor = OpenAIPreprocessor::new(mdc)
            .expect("malformed optional metadata must not fail startup");
        let request = make_chat_request_with_max_tokens(message, "test-model", Some(11));
        assert!(
            preprocessor
                .preprocess_request(&request, None)
                .await
                .is_ok(),
            "malformed token budget should delegate validation to the backend"
        );
    }

    #[tokio::test]
    async fn test_backend_owned_overflow_defers_without_mutating_request() {
        let message = r#"[{"role": "user", "content": "What is deep learning?"}]"#;
        let (mut mdc, prompt_len) = model_card_and_prompt_len(message).await;

        set_token_budget(&mut mdc, prompt_len + 10, true, false);
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let request = make_chat_request_with_max_tokens(message, "test-model", Some(11));
        let (preprocessed, _, _) = preprocessor
            .preprocess_request(&request, None)
            .await
            .expect("output clamping belongs to the backend");
        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(11));

        let (mut mdc, prompt_len) = model_card_and_prompt_len(message).await;
        set_token_budget(&mut mdc, prompt_len, false, false);
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let (preprocessed, _, _) = preprocessor
            .preprocess_request(&make_chat_request(message, "test-model"), None)
            .await
            .expect("prompt truncation belongs to the backend");
        assert_eq!(preprocessed.stop_conditions.max_tokens, None);
    }
}

mod embedding_without_chat_template {
    use dynamo_llm::model_card::ModelDeploymentCard;
    use dynamo_llm::model_type::ModelType;
    use dynamo_llm::preprocessor::OpenAIPreprocessor;
    use dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc;
    use dynamo_llm::protocols::common::preprocessor::PreprocessedEmbeddingRequest;
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use dynamo_llm::protocols::openai::embeddings::NvCreateEmbeddingRequest;
    use dynamo_renderer::PromptFormatter;
    use serde::Deserialize;
    use serde_json::json;
    use serial_test::serial;
    use std::sync::Arc;
    use tempfile::TempDir;

    const MODEL_PATH: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";
    const PARITY_MODEL_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";
    const ADD_SPECIAL_TOKENS_ENV: &str = "DYN_EMBEDDING_TOKENIZATION_ADD_SPECIAL_TOKENS";

    #[derive(Debug, Deserialize)]
    struct TokenizationParityFixture {
        text: String,
        bos_token_id: u32,
        cases: Vec<TokenizationParityCase>,
    }

    #[derive(Debug, Deserialize)]
    struct TokenizationParityCase {
        name: String,
        add_special_tokens: bool,
        truncate_prompt_tokens: Option<i64>,
        max_model_len: u32,
        expected_token_ids: Vec<u32>,
    }

    fn tokenization_parity_fixture() -> TokenizationParityFixture {
        serde_json::from_str(include_str!(
            "data/sample-models/TinyLlama_v1.1/embedding_right_truncation_parity.json"
        ))
        .unwrap()
    }

    fn model_bos_token_id(model_path: &str) -> u32 {
        let model_config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::path::Path::new(model_path).join("config.json")).unwrap(),
        )
        .unwrap();
        model_config["bos_token_id"].as_u64().unwrap() as u32
    }

    fn mock_tokenizer_bos_token_id() -> u32 {
        model_bos_token_id(MODEL_PATH)
    }

    fn model_copy_with_chat_template(chat_template: Option<&str>) -> TempDir {
        let temp_dir = tempfile::tempdir().unwrap();
        for filename in [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ] {
            std::fs::copy(
                std::path::Path::new(MODEL_PATH).join(filename),
                temp_dir.path().join(filename),
            )
            .unwrap();
        }

        let config_path = temp_dir.path().join("tokenizer_config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        match chat_template {
            Some(template) => config["chat_template"] = json!(template),
            None => {
                config.as_object_mut().unwrap().remove("chat_template");
            }
        }
        std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        temp_dir
    }

    fn embedding_mdc(path: impl AsRef<std::path::Path>) -> ModelDeploymentCard {
        let mut mdc = ModelDeploymentCard::load_from_disk(path, None).unwrap();
        mdc.model_type = ModelType::Embedding;
        mdc
    }

    fn embedding_preprocessor_without_template() -> Arc<OpenAIPreprocessor> {
        let mut mdc = embedding_mdc(MODEL_PATH);
        mdc.prompt_formatter = None;
        mdc.chat_template_file = None;
        OpenAIPreprocessor::new_for_embeddings(mdc).unwrap()
    }

    fn chat_request() -> NvCreateChatCompletionRequest {
        let messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage> =
            serde_json::from_str(r#"[{"role":"user","content":"hello"}]"#).unwrap();
        let inner = dynamo_protocols::types::CreateChatCompletionRequestArgs::default()
            .model("test-model")
            .messages(messages)
            .build()
            .unwrap();
        NvCreateChatCompletionRequest {
            inner,
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }

    #[test]
    fn embedding_preprocessor_does_not_require_chat_template() {
        let model_dir = model_copy_with_chat_template(None);
        let mdc = embedding_mdc(model_dir.path());

        assert!(
            OpenAIPreprocessor::new(mdc.clone()).is_err(),
            "the general chat preprocessor should still require a prompt formatter"
        );
        OpenAIPreprocessor::new_for_embeddings(mdc)
            .expect("embedding-only preprocessing must not require a chat template");
    }

    #[test]
    fn embedding_preprocessor_uses_chat_template_when_present() {
        let preprocessor =
            OpenAIPreprocessor::new_for_embeddings(embedding_mdc(MODEL_PATH)).unwrap();
        let rendered = preprocessor
            .apply_template(&chat_request())
            .unwrap()
            .unwrap();
        assert!(
            rendered
                .as_str()
                .contains("<|start_header_id|>user<|end_header_id|>")
        );
    }

    #[test]
    fn hybrid_general_constructor_does_not_initialize_embedding_tokenizers() {
        let model_dir = model_copy_with_chat_template(Some(
            "{% for message in messages %}{{ message['content'] }}{% endfor %}",
        ));
        let mut mdc = embedding_mdc(model_dir.path());
        mdc.model_type = ModelType::Chat | ModelType::Embedding;
        let tokenizer = mdc.tokenizer().unwrap();
        let PromptFormatter::OAI(formatter) = prompt_formatter_from_mdc(&mdc).unwrap();

        std::fs::remove_file(model_dir.path().join("tokenizer.json")).unwrap();
        let preprocessor = OpenAIPreprocessor::new_with_parts(mdc, formatter, tokenizer)
            .expect("general construction must not load embedding-only tokenizers");
        let rendered = preprocessor
            .apply_template(&chat_request())
            .unwrap()
            .unwrap();
        assert_eq!(rendered.as_str(), "hello");
    }

    #[test]
    fn embedding_preprocessor_rejects_malformed_chat_template() {
        let model_dir = model_copy_with_chat_template(Some("{% invalid template %}"));
        assert!(OpenAIPreprocessor::new_for_embeddings(embedding_mdc(model_dir.path())).is_err());
    }

    #[test]
    fn embedding_constructor_rejects_non_embedding_model() {
        let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, None).unwrap();
        mdc.model_type = ModelType::Chat;

        let err = match OpenAIPreprocessor::new_for_embeddings(mdc) {
            Ok(_) => panic!("non-embedding models must use the general constructor"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("requires an embedding-capable model")
        );
    }

    fn embedding_request_with_options(
        input: serde_json::Value,
        add_special_tokens: Option<bool>,
        truncate_prompt_tokens: Option<i64>,
    ) -> NvCreateEmbeddingRequest {
        let mut value = json!({
            "model": "test-model",
            "input": input
        });
        if let Some(value_) = add_special_tokens {
            value["add_special_tokens"] = json!(value_);
        }
        if let Some(value_) = truncate_prompt_tokens {
            value["truncate_prompt_tokens"] = json!(value_);
        }
        serde_json::from_value(value).unwrap()
    }

    fn embedding_request(add_special_tokens: Option<bool>) -> NvCreateEmbeddingRequest {
        embedding_request_with_options(json!("hello"), add_special_tokens, None)
    }

    async fn preprocess_embedding(
        preprocessor: &OpenAIPreprocessor,
        input: serde_json::Value,
        truncate_prompt_tokens: Option<i64>,
    ) -> anyhow::Result<PreprocessedEmbeddingRequest> {
        Ok(preprocessor
            .preprocess_embedding_request(&embedding_request_with_options(
                input,
                None,
                truncate_prompt_tokens,
            ))
            .await?
            .0)
    }

    async fn token_ids(
        preprocessor: &OpenAIPreprocessor,
        add_special_tokens: Option<bool>,
    ) -> Vec<Vec<u32>> {
        preprocessor
            .preprocess_embedding_request(&embedding_request(add_special_tokens))
            .await
            .unwrap()
            .0
            .token_ids
    }

    #[tokio::test]
    #[serial]
    async fn request_add_special_tokens_defaults_to_true() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, None::<&str>)], async {
            let preprocessor = embedding_preprocessor_without_template();
            let omitted = token_ids(&preprocessor, None).await;
            let explicit_false = token_ids(&preprocessor, Some(false)).await;
            let explicit_true = token_ids(&preprocessor, Some(true)).await;

            assert_eq!(omitted, explicit_true);
            assert_ne!(omitted, explicit_false);
            let bos_token_id = mock_tokenizer_bos_token_id();
            assert_eq!(explicit_true[0][0], bos_token_id, "explicit true adds BOS");
            assert_ne!(explicit_false[0].first(), Some(&bos_token_id));
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn embedding_tokenizer_variants_initialize_on_demand() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, None::<&str>)], async {
            let model_dir = model_copy_with_chat_template(None);
            let preprocessor =
                OpenAIPreprocessor::new_for_embeddings(embedding_mdc(model_dir.path())).unwrap();

            let expected = token_ids(&preprocessor, None).await;
            std::fs::remove_file(model_dir.path().join("tokenizer.json")).unwrap();

            assert_eq!(token_ids(&preprocessor, None).await, expected);
            let error = preprocessor
                .preprocess_embedding_request(&embedding_request(Some(false)))
                .await
                .unwrap_err();
            assert!(
                error.to_string().contains("tokenizer.json"),
                "the unselected tokenizer variant should initialize only when requested: {error}"
            );
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn env_add_special_tokens_is_default_and_request_wins() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, Some("true"))], async {
            let preprocessor = embedding_preprocessor_without_template();
            let bos_token_id = mock_tokenizer_bos_token_id();
            assert_eq!(token_ids(&preprocessor, None).await[0][0], bos_token_id);
            assert_ne!(
                token_ids(&preprocessor, Some(false)).await[0].first(),
                Some(&bos_token_id)
            );
        })
        .await;

        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, Some("false"))], async {
            let preprocessor = embedding_preprocessor_without_template();
            let env_false = token_ids(&preprocessor, None).await;
            let request_false = token_ids(&preprocessor, Some(false)).await;
            assert_eq!(env_false, request_false);
            assert_eq!(
                token_ids(&preprocessor, Some(true)).await[0][0],
                mock_tokenizer_bos_token_id()
            );
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn invalid_env_add_special_tokens_fails_construction() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, Some("yes-please"))], async {
            let mut mdc = embedding_mdc(MODEL_PATH);
            mdc.prompt_formatter = None;
            mdc.chat_template_file = None;
            let err = match OpenAIPreprocessor::new_for_embeddings(mdc) {
                Ok(_) => panic!("invalid environment value must fail construction"),
                Err(err) => err,
            };
            assert!(err.to_string().contains("expected true/false/yes/no/1/0"));
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn frontend_tokenization_matches_vllm_right_truncation_fixture() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, None::<&str>)], async {
            let fixture = tokenization_parity_fixture();
            assert_eq!(fixture.bos_token_id, model_bos_token_id(PARITY_MODEL_PATH));

            for case in fixture.cases {
                let mut mdc = embedding_mdc(PARITY_MODEL_PATH);
                mdc.prompt_formatter = None;
                mdc.chat_template_file = None;
                mdc.runtime_config.context_length = Some(case.max_model_len);
                let preprocessor = OpenAIPreprocessor::new_for_embeddings(mdc).unwrap();
                let request = embedding_request_with_options(
                    json!(fixture.text),
                    Some(case.add_special_tokens),
                    case.truncate_prompt_tokens,
                );
                let actual = preprocessor
                    .preprocess_embedding_request(&request)
                    .await
                    .unwrap()
                    .0
                    .token_ids;

                assert_eq!(
                    actual,
                    vec![case.expected_token_ids],
                    "vLLM right-truncation parity case {:?}",
                    case.name
                );
            }
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn frontend_right_truncation_keeps_the_token_prefix() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, None::<&str>)], async {
            let preprocessor = embedding_preprocessor_without_template();
            let input = json!("<|eot_id|><|eot_id|><|eot_id|>");
            let full = preprocess_embedding(&preprocessor, input.clone(), None)
                .await
                .unwrap()
                .token_ids;

            for limit in [0, 2] {
                let truncated = preprocess_embedding(&preprocessor, input.clone(), Some(limit))
                    .await
                    .unwrap()
                    .token_ids;
                assert_eq!(truncated[0], full[0][..limit as usize]);
            }
            assert_eq!(
                full[0][0],
                mock_tokenizer_bos_token_id(),
                "BOS is added before truncation"
            );
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn frontend_truncation_handles_batches_and_preserves_token_inputs() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, None::<&str>)], async {
            let preprocessor = embedding_preprocessor_without_template();
            let input = json!([
                "<|eot_id|><|eot_id|><|eot_id|>",
                "<|eot_id|><|eot_id|><|eot_id|><|eot_id|>"
            ]);
            let full = preprocess_embedding(&preprocessor, input.clone(), None)
                .await
                .unwrap()
                .token_ids;
            let truncated = preprocess_embedding(&preprocessor, input, Some(3))
                .await
                .unwrap()
                .token_ids;
            for (actual, expected) in truncated.iter().zip(&full) {
                assert_eq!(actual, &expected[..3]);
            }

            let caller_tokens = preprocess_embedding(&preprocessor, json!([11, 12, 13]), Some(2))
                .await
                .unwrap();
            assert_eq!(caller_tokens.token_ids, vec![vec![11, 12, 13]]);
            assert_eq!(caller_tokens.truncate_prompt_tokens, Some(2));
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn frontend_truncation_validates_model_limits() {
        temp_env::async_with_vars([(ADD_SPECIAL_TOKENS_ENV, None::<&str>)], async {
            let mut mdc = embedding_mdc(MODEL_PATH);
            mdc.prompt_formatter = None;
            mdc.chat_template_file = None;
            mdc.runtime_config.context_length = Some(3);
            let preprocessor = OpenAIPreprocessor::new_for_embeddings(mdc).unwrap();
            let input = json!("<|eot_id|><|eot_id|><|eot_id|><|eot_id|>");

            let truncated = preprocess_embedding(&preprocessor, input.clone(), Some(-1))
                .await
                .unwrap()
                .token_ids;
            assert_eq!(truncated[0].len(), 3);

            for (limit, message) in [
                (-2, "must be >= -1"),
                (4, "cannot be greater than max_model_len=3"),
            ] {
                let error = preprocess_embedding(&preprocessor, input.clone(), Some(limit))
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains(message), "{error}");
            }
        })
        .await;
    }
}
