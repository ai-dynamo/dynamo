// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::local_model::runtime_config::TokenizerBackend;
use dynamo_llm::model_card::{ModelDeploymentCard, PromptFormatterArtifact, TokenizerKind};
use dynamo_llm::tokenizers::{BasetenTokenizer, EncodeSegment, TikTokenTokenizer, traits::Encoder};
use tempfile::tempdir;

const HF_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";
const TIKTOKEN_PATH: &str = "tests/data/sample-models/mock-tiktoken";

#[tokio::test]
async fn test_model_info_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    let info = mdc.model_info.as_ref().unwrap().get_model_info().unwrap();
    assert_eq!(info.model_type(), "llama");
    assert_eq!(info.bos_token_id(), Some(1));
    assert_eq!(info.eos_token_ids(), vec![2]);
    assert_eq!(info.max_position_embeddings(), Some(2048));
    assert_eq!(info.vocab_size(), Some(32000));
    assert_eq!(mdc.architectural_max_context_length, Some(2048));
    assert_eq!(mdc.runtime_config.context_length, None);
    assert_eq!(mdc.effective_context_length(), 2048);
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

#[test]
fn test_tiktoken_model_card_cache_matches_direct_tokenizer_and_records_tokens() {
    let model = "model-card-tiktoken-cache-integration";
    let mut mdc = ModelDeploymentCard::load_from_disk(TIKTOKEN_PATH, None).unwrap();
    mdc.set_name(model);

    let production = mdc.tokenizer().unwrap();
    let direct =
        TikTokenTokenizer::from_file_auto(&format!("{TIKTOKEN_PATH}/tiktoken.model")).unwrap();

    let cached_tokens = dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_CACHED_TOKENS_TOTAL
        .with_label_values(&[model]);
    let uncached_tokens =
        dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL
            .with_label_values(&[model]);
    let cached_before = cached_tokens.get();
    let uncached_before = uncached_tokens.get();

    let prompts = [
        "<|im_start|>system\nYou are concise.<|im_end|><|im_start|>user\nExplain prefix caching.<|im_end|>",
        "<|im_start|>system\nYou are concise.<|im_end|><|im_start|>user\nNow include Unicode: 北京 😀.<|im_end|>",
    ];
    let mut returned_tokens = 0_u64;
    for prompt in prompts {
        let actual = production.encode(prompt).unwrap().token_ids().to_vec();
        let expected = direct.encode(prompt).unwrap().token_ids().to_vec();
        assert_eq!(
            actual, expected,
            "cached production path must remain token-exact"
        );
        returned_tokens += actual.len() as u64;
    }

    let cached_delta = cached_tokens.get() - cached_before;
    let uncached_delta = uncached_tokens.get() - uncached_before;
    assert!(
        cached_delta > 0,
        "the second request should reuse the shared chat prefix"
    );
    assert_eq!(
        cached_delta + uncached_delta,
        returned_tokens,
        "cache token accounting must cover every returned token"
    );
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

/// A minimal BPE `tokenizer.json` with no normalizer, pre-tokenizer or decoder.
/// `BasetenTokenizer` rejects the Llama-style tokenizers used by the other
/// fixtures (see `test_basetenkenizer_falls_back_when_unsupported`), so this is
/// the fixture that exercises the backend rather than its fallback.
const MINIMAL_BPE_PATH: &str = "tests/data/sample-models/mock-minimal-bpe";

fn tokenizer_for(
    path: &str,
    name: &str,
    backend: TokenizerBackend,
) -> dynamo_llm::tokenizers::Tokenizer {
    let mut mdc = ModelDeploymentCard::load_from_disk(path, None).unwrap();
    mdc.set_name(name);
    mdc.runtime_config.tokenizer_backend = Some(backend);
    mdc.tokenizer().unwrap()
}

/// Selecting `basetenkenizer` must actually route `tokenizer()` to
/// `BasetenTokenizer`.
///
/// Token IDs cannot prove this: on this fixture Baseten and HuggingFace agree
/// exactly (see `test_basetenkenizer_token_parity_with_huggingface`), so a
/// regression that silently kept HuggingFace would still compare equal.
/// `encode_segments` is the discriminator. `BasetenTokenizer` implements it,
/// HuggingFace and fastokens inherit the trait default that returns an error,
/// and the L1 prefix cache forwards it to the inner tokenizer. So it succeeding
/// through the production path means Baseten really is underneath.
#[test]
fn test_basetenkenizer_backend_is_selected() {
    let production = tokenizer_for(
        MINIMAL_BPE_PATH,
        "model-card-basetenkenizer-selection",
        TokenizerBackend::Basetenkenizer,
    );
    let direct =
        BasetenTokenizer::from_file(&format!("{MINIMAL_BPE_PATH}/tokenizer.json")).unwrap();

    for prompt in ["Hello, world!", "Hello", " world", "He llo", ""] {
        assert_eq!(
            production.encode(prompt).unwrap().token_ids().to_vec(),
            direct.encode(prompt).unwrap().token_ids().to_vec(),
            "production path must match a directly constructed BasetenTokenizer for {prompt:?}"
        );
    }

    let segments = [
        EncodeSegment::new("Hello", true),
        EncodeSegment::new(", world!", false),
    ];
    assert!(
        production.encode_segments(&segments).is_ok(),
        "segmented encoding must reach BasetenTokenizer, so the backend was selected"
    );

    let default_backend = tokenizer_for(
        MINIMAL_BPE_PATH,
        "model-card-basetenkenizer-selection-default",
        TokenizerBackend::Default,
    );
    assert!(
        default_backend.encode_segments(&segments).is_err(),
        "the default backend does not support segmented encoding, which is what \
         makes the assertion above a real discriminator"
    );
}

/// Token parity with the default HuggingFace backend on a tokenizer both can
/// load. A new backend that silently changed token IDs would corrupt KV-cache
/// reuse and prompt accounting, so this is the assertion that matters most.
#[test]
fn test_basetenkenizer_token_parity_with_huggingface() {
    let baseten = tokenizer_for(
        MINIMAL_BPE_PATH,
        "model-card-basetenkenizer-parity",
        TokenizerBackend::Basetenkenizer,
    );
    let hf = tokenizer_for(
        MINIMAL_BPE_PATH,
        "model-card-basetenkenizer-parity-hf",
        TokenizerBackend::Default,
    );

    for prompt in ["Hello, world!", "Hello", " world", "He llo", "the sailor"] {
        assert_eq!(
            baseten.encode(prompt).unwrap().token_ids().to_vec(),
            hf.encode(prompt).unwrap().token_ids().to_vec(),
            "basetenkenizer and HuggingFace must agree on token IDs for {prompt:?}"
        );
    }
}

/// `BasetenTokenizer` does not support every `tokenizer.json`; TinyLlama uses a
/// `Prepend` normalizer it rejects. Selecting the backend must then degrade to
/// HuggingFace rather than failing the model, matching fastokens' behavior.
/// Asserting equality with the default backend proves it fell back rather than
/// producing something else.
#[test]
fn test_basetenkenizer_falls_back_when_unsupported() {
    assert!(
        BasetenTokenizer::from_file(&format!("{HF_PATH}/tokenizer.json")).is_err(),
        "fixture is only meaningful while BasetenTokenizer rejects it"
    );

    let requested = tokenizer_for(
        HF_PATH,
        "model-card-basetenkenizer-fallback",
        TokenizerBackend::Basetenkenizer,
    );
    let hf = tokenizer_for(
        HF_PATH,
        "model-card-basetenkenizer-fallback-hf",
        TokenizerBackend::Default,
    );

    for prompt in ["Explain prefix caching.", "Now include Unicode: 北京 😀."] {
        let ids = requested.encode(prompt).unwrap().token_ids().to_vec();
        assert!(!ids.is_empty(), "fallback must still tokenize {prompt:?}");
        assert_eq!(
            ids,
            hf.encode(prompt).unwrap().token_ids().to_vec(),
            "unsupported backend must fall back to HuggingFace for {prompt:?}"
        );
    }
}

/// `tokenizer()` wraps whichever backend it selected in the L1 prefix cache, so
/// the new backend has to survive that wrapping and stay token-exact across
/// repeated and shared-prefix requests. Cache *accounting* is already covered by
/// `test_tiktoken_model_card_cache_matches_direct_tokenizer_and_records_tokens`;
/// this fixture declares no special tokens, so there are no prefix boundaries to
/// drive those counters and asserting on them here would test the cache's
/// heuristics rather than this backend.
#[test]
fn test_basetenkenizer_cached_path_is_token_exact() {
    let cached = tokenizer_for(
        MINIMAL_BPE_PATH,
        "model-card-basetenkenizer-cache",
        TokenizerBackend::Basetenkenizer,
    );
    let direct =
        BasetenTokenizer::from_file(&format!("{MINIMAL_BPE_PATH}/tokenizer.json")).unwrap();

    // Repeat the first prompt and share its prefix with the second, so any
    // caching in play is actually exercised.
    for prompt in [
        "the sailor the sailor",
        "the sailor the sailor",
        "the sailor the world",
    ] {
        assert_eq!(
            cached.encode(prompt).unwrap().token_ids().to_vec(),
            direct.encode(prompt).unwrap().token_ids().to_vec(),
            "cached path must remain token-exact for {prompt:?}"
        );
    }
}
