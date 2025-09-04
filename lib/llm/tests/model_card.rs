// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::model_card::{ModelDeploymentCard, PromptFormatterArtifact, TokenizerKind};
use tempfile::tempdir;

const HF_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";

#[tokio::test]
async fn test_model_info_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load(HF_PATH, None).await.unwrap();
    let info = mdc.model_info.unwrap().get_model_info().await.unwrap();
    assert_eq!(info.model_type(), "llama");
    assert_eq!(info.bos_token_id(), 1);
    assert_eq!(info.eos_token_ids(), vec![2]);
    assert_eq!(info.max_position_embeddings(), Some(2048));
    assert_eq!(info.vocab_size(), Some(32000));
}

#[tokio::test]
async fn test_model_info_from_non_existent_local_repo() {
    let path = "tests/data/sample-models/this-model-does-not-exist";
    let result = ModelDeploymentCard::load(path, None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_tokenizer_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load(HF_PATH, None).await.unwrap();
    // Verify tokenizer file was found
    match mdc.tokenizer.unwrap() {
        TokenizerKind::HfTokenizerJson(_) => (),
        TokenizerKind::GGUF(_) => (),
    }
}

#[tokio::test]
async fn test_prompt_formatter_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load(HF_PATH, None).await.unwrap();
    // Verify prompt formatter was found
    match mdc.prompt_formatter {
        Some(PromptFormatterArtifact::HfTokenizerConfigJson(_)) => (),
        _ => panic!("Expected HfTokenizerConfigJson prompt formatter"),
    }
}

#[tokio::test]
async fn test_missing_required_files() {
    // Create empty temp directory
    let temp_dir = tempdir().unwrap();
    let result = ModelDeploymentCard::load(temp_dir.path(), None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Should fail because config.json is missing
    assert!(err.contains("unable to extract"));
}
