// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Ok, Result};

use dynamo_llm::model_card::{ModelDeploymentCard, PromptContextMixin};
use dynamo_llm::preprocessor::prompt::PromptFormatter;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use serde::{Deserialize, Serialize};

use hf_hub::{Cache, Repo, RepoType, api::tokio::ApiBuilder};

use std::path::PathBuf;

/// ----------------- NOTE ---------------
/// Currently ModelDeploymentCard does support downloading models using nim-hub.
/// As a temporary workaround, we will download the models from Hugging Face to a local cache
/// directory in `tests/data/sample-models`. These tests require a Hugging Face token to be
/// set in the environment variable `HF_TOKEN`.
/// The model is downloaded and cached in `tests/data/sample-models` directory.
/// make sure the token has access to `meta-llama/Llama-3.1-70B-Instruct` model
/// Gets the HF_TOKEN environment variable if it exists and is not empty.
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
    let token = std::env::var("HF_TOKEN")
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
    //TODO: remove this once we have nim-hub support. See the NOTE above.
    let downloaded_path = maybe_download_model(local_path, hf_repo, hf_revision).await;
    let display_name = format!("{}--{}", hf_repo, hf_revision);
    let mut mdc = ModelDeploymentCard::load(downloaded_path, None).unwrap();
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
    let repo_builder = api.repo(repo);

    let mut downloaded_path = PathBuf::new();
    for file in &files_to_download {
        downloaded_path = repo_builder.get(file).await.unwrap();
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

// fn load_nim_mdcs() -> Vec<ModelDeploymentCard> {
//     // get all .json files from test/data/model_deployment_cards/nim
//     std::fs::read_dir("tests/data/model_deployment_cards/nim")
//         .unwrap()
//         .map(|res| res.map(|e| e.path()).unwrap().clone())
//         .filter(|path| path.extension().unwrap() == "json")
//         .map(|path| ModelDeploymentCard::load_from_json_file(path).unwrap())
//         .collect::<Vec<_>>()
// }

// #[ignore]
// #[tokio::test]
// async fn create_mdc_from_repo() {
//     for repo in NGC_MODEL_REPOS.iter() {
//         println!("Creating MDC for {}", repo);
//         let mdc = make_mdc_from_repo(repo).await;
//         mdc.save_to_json_file(&format!(
//             "tests/data/model_deployment_cards/nim/{}.json",
//             Slug::slugify(repo)
//         ))
//         .unwrap();
//     }
// }

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
// protocols::openai::chat_completions::ChatCompletionMessage -> dynamo_async_openai::types::ChatCompletionRequestMessage
// protocols::openai::chat_completions::Tool -> dynamo_async_openai::types::ChatCompletionTool
// protocols::openai::chat_completions::ToolChoiceType -> dynamo_async_openai::types::ChatCompletionToolChoiceOption
#[derive(Serialize, Deserialize)]
struct Request {
    messages: Vec<dynamo_async_openai::types::ChatCompletionRequestMessage>,
    tools: Option<Vec<dynamo_async_openai::types::ChatCompletionTool>>,
    tool_choice: Option<dynamo_async_openai::types::ChatCompletionToolChoiceOption>,
}

impl Request {
    fn from(
        messages: &str,
        tools: Option<&str>,
        tool_choice: Option<dynamo_async_openai::types::ChatCompletionToolChoiceOption>,
        model: String,
    ) -> NvCreateChatCompletionRequest {
        let messages: Vec<dynamo_async_openai::types::ChatCompletionRequestMessage> =
            serde_json::from_str(messages).unwrap();
        let tools: Option<Vec<dynamo_async_openai::types::ChatCompletionTool>> =
            tools.map(|x| serde_json::from_str(x).unwrap());
        //let tools = tools.unwrap();
        //let tool_choice = tool_choice.unwrap();

        let mut inner = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default();
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
        let formatter = PromptFormatter::from_mdc(mdc).unwrap();

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
        let formatter = PromptFormatter::from_mdc(mdc).unwrap();

        // assert its an OAI formatter
        let formatter = match formatter {
            PromptFormatter::OAI(formatter) => Ok(formatter),
        }
        .unwrap();

        let request = Request::from(
            SINGLE_CHAT_MESSAGE,
            Some(TOOLS),
            Some(dynamo_async_openai::types::ChatCompletionToolChoiceOption::Auto),
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
        let formatter = PromptFormatter::from_mdc(mdc).unwrap();

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
        let formatter = PromptFormatter::from_mdc(mdc).unwrap();

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
        let formatter = PromptFormatter::from_mdc(mdc).unwrap();

        // assert its an OAI formatter
        let formatter = match formatter {
            PromptFormatter::OAI(formatter) => Ok(formatter),
        }
        .unwrap();

        let request = Request::from(
            THREE_TURN_CHAT_MESSAGE_WITH_SYSTEM,
            Some(TOOLS),
            Some(dynamo_async_openai::types::ChatCompletionToolChoiceOption::Auto),
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

    let formatter = PromptFormatter::from_mdc(&mdc).unwrap();

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
