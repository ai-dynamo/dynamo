// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::model_card::{ModelDeploymentCard, PromptFormatterArtifact, TokenizerKind};
use tempfile::tempdir;

const HF_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";

#[tokio::test]
async fn test_model_info_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    let info = mdc.model_info.unwrap().get_model_info().unwrap();
    assert_eq!(info.model_type(), "llama");
    assert_eq!(info.bos_token_id(), Some(1));
    assert_eq!(info.eos_token_ids(), vec![2]);
    assert_eq!(info.max_position_embeddings(), Some(2048));
    assert_eq!(info.vocab_size(), Some(32000));
}

#[tokio::test]
async fn test_model_info_from_non_existent_local_repo() {
    let path = "tests/data/sample-models/this-model-does-not-exist";
    let result = ModelDeploymentCard::load_from_disk(path, None);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_tokenizer_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    // Verify tokenizer file was found
    match mdc.tokenizer.unwrap() {
        TokenizerKind::HfTokenizerJson(_) => (),
        TokenizerKind::TikTokenModel(_) => panic!("Expected HfTokenizerJson, got TikTokenModel"),
    }
}

#[tokio::test]
async fn test_prompt_formatter_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
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
    let result = ModelDeploymentCard::load_from_disk(temp_dir.path(), None);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Should fail because config.json is missing
    assert!(err.contains("unable to extract"));
}

/// Models without tokenizer.json (e.g. Qwen3-Omni which ships vocab.json + merges.txt)
/// should load successfully with tokenizer set to None. The frontend must use a
/// non-Rust chat processor for these models (e.g. --dyn-chat-processor vllm).
#[tokio::test]
async fn test_model_loads_without_tokenizer_json() {
    let path = "tests/data/sample-models/mock-no-tokenizer-json";
    let mdc = ModelDeploymentCard::load_from_disk(path, None).unwrap();
    assert!(
        mdc.tokenizer.is_none(),
        "Expected tokenizer to be None for model without tokenizer.json"
    );
    assert!(!mdc.has_tokenizer(), "has_tokenizer() should be false");
    // Model info should still be loaded
    assert!(mdc.model_info.is_some());
}

/// chat_template.json should be picked up as a fallback when chat_template.jinja
/// does not exist (e.g. Qwen3-Omni). The fixture's tokenizer_config.json has no
/// inline chat_template, so this is the only template source.
#[tokio::test]
async fn test_chat_template_json_fallback() {
    let path = "tests/data/sample-models/mock-no-tokenizer-json";
    let mdc = ModelDeploymentCard::load_from_disk(path, None).unwrap();
    match &mdc.chat_template_file {
        Some(PromptFormatterArtifact::HfChatTemplateJson { file, is_custom }) => {
            assert!(!is_custom, "Should not be marked as custom template");
            let p = file.path().expect("Should be a local path");
            assert!(
                p.ends_with("chat_template.json"),
                "Expected chat_template.json, got {:?}",
                p
            );
        }
        other => panic!("Expected HfChatTemplateJson, got {:?}", other),
    }
}

const HF_PATH_CHAT: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

fn tokenizer_path(mdc: &ModelDeploymentCard) -> Option<std::path::PathBuf> {
    mdc.tokenizer
        .as_ref()
        .and_then(|t| match t {
            TokenizerKind::HfTokenizerJson(cf) | TokenizerKind::TikTokenModel(cf) => cf.path(),
        })
        .map(|p| p.to_path_buf())
}

#[tokio::test]
async fn test_override_files_with_replaces_slots() {
    let mut worker = ModelDeploymentCard::load_from_disk(HF_PATH_CHAT, None).unwrap();
    let local = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    let before = tokenizer_path(&worker);

    worker.override_files_with(&local);

    let after = tokenizer_path(&worker).expect("tokenizer slot populated");
    assert_ne!(before, Some(after.clone()));
    assert!(after.starts_with(HF_PATH));
}

#[tokio::test]
async fn test_override_files_with_preserves_runtime_fields() {
    let mut worker = ModelDeploymentCard::load_from_disk(HF_PATH_CHAT, None).unwrap();
    worker.set_name("served-name");
    worker.kv_cache_block_size = 32;
    worker.context_length = 4096;

    worker.override_files_with(&ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap());

    assert_eq!(worker.display_name, "served-name");
    assert_eq!(worker.kv_cache_block_size, 32);
    assert_eq!(worker.context_length, 4096);
}

#[tokio::test]
async fn test_override_files_with_clears_slots_from_empty_source() {
    let mut worker = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    assert!(worker.tokenizer.is_some());
    assert!(worker.model_info.is_some());

    worker.override_files_with(&ModelDeploymentCard::with_name_only("x"));

    assert!(worker.tokenizer.is_none());
    assert!(worker.model_info.is_none());
}

#[tokio::test]
async fn test_override_files_with_invalidates_mdcsum_cache() {
    let mut worker = ModelDeploymentCard::load_from_disk(HF_PATH_CHAT, None).unwrap();
    let before = worker.mdcsum().to_string();

    let local = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    worker.override_files_with(&local);

    let after = worker.mdcsum();
    assert_ne!(
        before, after,
        "mdcsum cache must be invalidated so a fresh checksum reflects the new files"
    );
}
