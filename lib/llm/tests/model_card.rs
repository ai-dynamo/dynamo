// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::common::checked_file::CheckedFile;
use dynamo_llm::local_model::runtime_config::TokenizerBackend;
use dynamo_llm::model_card::{ModelDeploymentCard, PromptFormatterArtifact, TokenizerKind};
use dynamo_llm::tokenizers::{
    BasetenTokenizer, EncodeSegment, HuggingFaceTokenizer, TikTokenTokenizer, traits::Encoder,
};
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};

const HF_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";
const TIKTOKEN_PATH: &str = "tests/data/sample-models/mock-tiktoken";

fn unsupported_alternate_tokenizer() -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let tokenizer_path = dir.path().join("tokenizer.json");
    std::fs::write(
        &tokenizer_path,
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1},
                "unk_token": "[UNK]"
            }
        }"#,
    )
    .unwrap();
    (dir, tokenizer_path)
}

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
#[serial_test::serial]
fn test_basetenkenizer_model_card_matches_hf_and_uses_prefix_cache() {
    temp_env::with_vars(
        [
            ("DYN_TOKENIZER_CACHE", Some("1")),
            ("DYN_TOKENIZER_CACHE_EXTEND", Some("1")),
        ],
        || {
            let model = "model-card-basetenkenizer-cache-integration";
            let dir = tempdir().unwrap();
            let tokenizer_path = dir.path().join("tokenizer.json");
            let vocab: serde_json::Map<String, serde_json::Value> = [
                "<unk>", " ", "!", ",", "H", "T", "d", "e", "h", "l", "o", "r", "w", "He", "ll",
                "llo", "or", "ld",
            ]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_string(), id.into()))
            .collect();
            let tokenizer = serde_json::json!({
                "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>",
                    "continuing_subword_prefix": null, "end_of_word_suffix": null,
                    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                    "vocab": vocab,
                    "merges": [["H", "e"], ["l", "l"], ["ll", "o"], ["o", "r"], ["l", "d"]]}
            });
            std::fs::write(&tokenizer_path, tokenizer.to_string()).unwrap();
            std::fs::write(
                dir.path().join("tokenizer_config.json"),
                r#"{
                    "added_tokens_decoder": {
                        "18": {
                            "content": "<special>",
                            "single_word": false,
                            "lstrip": false,
                            "rstrip": false,
                            "normalized": false,
                            "special": true
                        }
                    }
                }"#,
            )
            .unwrap();

            let mut mdc = ModelDeploymentCard::with_name_only(model);
            mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                CheckedFile::from_disk(&tokenizer_path).unwrap(),
            ));
            mdc.runtime_config.tokenizer_backend = Some(TokenizerBackend::Basetenkenizer);

            let production = mdc.tokenizer().unwrap();
            let direct = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();

            let cached_tokens =
                dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_CACHED_TOKENS_TOTAL
                    .with_label_values(&[model]);
            let uncached_tokens =
                dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL
                    .with_label_values(&[model]);
            let cached_before = cached_tokens.get();
            let uncached_before = uncached_tokens.get();

            let prompts = ["<special>Hello, world!", "<special>The world"];
            let mut returned_tokens = 0_u64;
            for prompt in prompts {
                let actual = production.encode(prompt).unwrap().token_ids().to_vec();
                let expected = direct.encode(prompt).unwrap().token_ids().to_vec();
                assert_eq!(
                    actual.first(),
                    Some(&18),
                    "the sibling-only special token must retain its configured token ID"
                );
                assert_eq!(
                    actual, expected,
                    "Baseten and HuggingFace paths must remain token-exact"
                );
                returned_tokens += actual.len() as u64;
            }

            let cached_delta = cached_tokens.get() - cached_before;
            let uncached_delta = uncached_tokens.get() - uncached_before;
            assert!(
                cached_delta > 0,
                "the second request should reuse the shared special-token prefix"
            );
            assert_eq!(
                cached_delta + uncached_delta,
                returned_tokens,
                "cache token accounting must cover every returned token"
            );
            assert_eq!(
                production.decode(&[18], false).unwrap().as_str(),
                "<special>",
                "the sibling-only special token must remain decodable"
            );

            production
                .encode_segments(&[EncodeSegment::new("Hello, world!", false)])
                .expect("selected Baseten backend must preserve segmented encoding support");
        },
    );
}

#[test]
#[serial_test::serial]
fn test_basetenkenizer_load_failure_falls_back_to_hf() {
    temp_env::with_vars(
        [
            ("DYN_TOKENIZER_CACHE", Some("0")),
            ("DYN_TOKENIZER_FALLBACK", None),
        ],
        || {
            let (_dir, tokenizer_path) = unsupported_alternate_tokenizer();

            assert!(
                BasetenTokenizer::from_file(tokenizer_path.to_str().unwrap()).is_err(),
                "fixture must remain unsupported by the Baseten BPE backend"
            );

            let mut mdc = ModelDeploymentCard::with_name_only("baseten-fallback");
            mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                CheckedFile::from_disk(&tokenizer_path).unwrap(),
            ));
            mdc.runtime_config.tokenizer_backend = Some(TokenizerBackend::Basetenkenizer);

            let tokenizer = mdc
                .tokenizer()
                .expect("unsupported Baseten tokenizer must fall back to HuggingFace");
            assert_eq!(
                tokenizer.encode("hello").unwrap().token_ids(),
                &[1],
                "fallback must remain usable"
            );
        },
    );
}

#[test]
#[serial_test::serial]
fn test_alternate_tokenizer_load_failure_is_rejected_when_fallback_is_disabled() {
    temp_env::with_vars(
        [
            ("DYN_TOKENIZER_CACHE", Some("0")),
            ("DYN_TOKENIZER_FALLBACK", Some("1")),
        ],
        || {
            for backend in [
                TokenizerBackend::Fastokens,
                TokenizerBackend::Basetenkenizer,
            ] {
                let (_dir, tokenizer_path) = unsupported_alternate_tokenizer();
                let mut mdc = ModelDeploymentCard::with_name_only("strict-tokenizer");
                mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                    CheckedFile::from_disk(&tokenizer_path).unwrap(),
                ));
                mdc.runtime_config.tokenizer_backend = Some(backend);
                mdc.runtime_config.tokenizer_fallback_enabled = Some(false);

                let error = mdc
                    .tokenizer()
                    .err()
                    .expect("unsupported alternate tokenizer must fail without fallback")
                    .to_string();
                assert!(error.contains(backend.as_str()));
                assert!(error.contains("fallback is disabled"));
            }
        },
    );
}

/// With the fastokens backend, text comes back from the same Hugging Face decoder as the default
/// backend. The ids, text and stream chunks are what the `tokenizers` 0.23.2 decoder returned;
/// they come from `tokenizers/tests/data/hf-expectations/TinyLlama_v1.1` in frontend-crates.
#[test]
#[serial_test::serial]
fn test_fastokens_model_card_decodes_like_the_default_backend() {
    struct Decoded {
        ids: &'static [u32],
        text: &'static str,
        text_without_specials: &'static str,
        chunks: &'static [Option<&'static str>],
        last_chunk: Option<&'static str>,
    }
    // TinyLlama: `</s>` is 2 and `<0xNN>` is 3 + NN.
    let cases = [
        Decoded {
            ids: &[22172, 29871, 243, 162, 155, 131, 29871, 30662, 30675, 29991],
            text: "hello 😀 北京!",
            text_without_specials: "hello 😀 北京!",
            chunks: &[
                Some("hello"),
                Some(" "),
                None,
                None,
                None,
                None,
                Some("😀 "),
                Some("北"),
                Some("京"),
                Some("!"),
            ],
            last_chunk: None,
        },
        // The four bytes of the emoji end the output, so only the final flush emits it.
        Decoded {
            ids: &[274, 28059, 29871, 30662, 30675, 29871, 243, 162, 155, 131],
            text: "café 北京 😀",
            text_without_specials: "café 北京 😀",
            chunks: &[
                Some("c"),
                Some("afé"),
                Some(" "),
                Some("北"),
                Some("京"),
                Some(" "),
                None,
                None,
                None,
                None,
            ],
            last_chunk: Some("😀"),
        },
        // `a`, `</s>`, then an invalid byte: skipping the special joins the two bytes into one
        // run, and the invalid byte replaces the whole run.
        Decoded {
            ids: &[100, 2, 248],
            text: "a</s>\u{fffd}",
            text_without_specials: "\u{fffd}\u{fffd}",
            chunks: &[None, Some("a</s>"), None],
            last_chunk: Some("\u{fffd}"),
        },
        Decoded {
            ids: &[198, 172, 2],
            text: "é</s>",
            text_without_specials: "é",
            chunks: &[None, None, Some("é</s>")],
            last_chunk: None,
        },
    ];

    temp_env::with_vars([("DYN_TOKENIZER_CACHE", Some("0"))], || {
        for backend in [TokenizerBackend::Default, TokenizerBackend::Fastokens] {
            let mut mdc = ModelDeploymentCard::with_name_only("fastokens-decode");
            mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                CheckedFile::from_disk(format!("{HF_PATH}/tokenizer.json")).unwrap(),
            ));
            mdc.runtime_config.tokenizer_backend = Some(backend);
            // A fastokens load failure must fail here, not fall back to the default backend.
            mdc.runtime_config.tokenizer_fallback_enabled = Some(false);
            let tokenizer = mdc.tokenizer().unwrap();
            let name = backend.as_str();

            assert_eq!(
                tokenizer.encode(cases[0].text).unwrap().token_ids(),
                cases[0].ids,
                "{name}"
            );
            for case in &cases {
                let ids = case.ids;
                assert_eq!(
                    tokenizer.decode(ids, false).unwrap().as_str(),
                    case.text,
                    "{name}: {ids:?}"
                );
                assert_eq!(
                    tokenizer.decode(ids, true).unwrap().as_str(),
                    case.text_without_specials,
                    "{name}: {ids:?}, skipping special tokens"
                );
                let mut stream = tokenizer.decode_stream(&[], false);
                let chunks: Vec<Option<String>> =
                    ids.iter().map(|id| stream.step(*id).unwrap()).collect();
                let chunks: Vec<Option<&str>> = chunks.iter().map(Option::as_deref).collect();
                assert_eq!(chunks, case.chunks, "{name}: {ids:?}");
                assert_eq!(
                    stream.finish().unwrap().as_deref(),
                    case.last_chunk,
                    "{name}: {ids:?}"
                );
            }
        }
    });
}

/// A configuration the Hugging Face backend cannot serve with the same ids and text fails at
/// load, naming the model and the file. Nothing else can serve it: an alternate backend that
/// falls back reaches the same error.
#[test]
#[serial_test::serial]
fn test_unsupported_hf_configuration_is_rejected_with_model_and_path() {
    temp_env::with_vars([("DYN_TOKENIZER_CACHE", Some("0"))], || {
        let dir = tempdir().unwrap();
        let tokenizer_path = dir.path().join("tokenizer.json");
        // Under the lowercase normalizer the added token is keyed as `control`, so `CONTROL`
        // would have no id.
        std::fs::write(
            &tokenizer_path,
            r#"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
                    {"id": 2, "content": "CONTROL", "special": true, "single_word": false,
                     "lstrip": false, "rstrip": false, "normalized": true}
                ],
                "normalizer": {"type": "Lowercase"},
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "hello": 1},
                    "unk_token": "[UNK]"
                }
            }"#,
        )
        .unwrap();

        for backend in [TokenizerBackend::Default, TokenizerBackend::Fastokens] {
            let mut mdc = ModelDeploymentCard::with_name_only("normalized-added-token");
            mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                CheckedFile::from_disk(&tokenizer_path).unwrap(),
            ));
            mdc.runtime_config.tokenizer_backend = Some(backend);
            mdc.runtime_config.tokenizer_fallback_enabled = Some(true);
            let error = format!("{:#}", mdc.tokenizer().err().expect("must not load"));
            assert!(
                error.contains("unsupported added token \"CONTROL\"")
                    && error.contains("normalized-added-token")
                    && error.contains(tokenizer_path.to_str().unwrap()),
                "{}: {error}",
                backend.as_str()
            );
        }
    });
}

#[test]
#[serial_test::serial]
fn test_hf_model_card_disables_serialized_padding_and_truncation() {
    temp_env::with_vars([("DYN_TOKENIZER_CACHE", Some("0"))], || {
        let dir = tempdir().unwrap();
        let tokenizer_path = dir.path().join("tokenizer.json");
        std::fs::write(
            &tokenizer_path,
            r#"{
                "version": "1.0",
                "truncation": {
                    "direction": "Right",
                    "max_length": 2,
                    "strategy": "LongestFirst",
                    "stride": 0
                },
                "padding": {
                    "strategy": {"Fixed": 8},
                    "direction": "Right",
                    "pad_to_multiple_of": null,
                    "pad_id": 0,
                    "pad_type_id": 0,
                    "pad_token": "[UNK]"
                },
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "again": 3},
                    "unk_token": "[UNK]"
                }
            }"#,
        )
        .unwrap();

        let mut mdc = ModelDeploymentCard::with_name_only("hf-online-normalization");
        mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
            CheckedFile::from_disk(&tokenizer_path).unwrap(),
        ));

        let tokenizer = mdc.tokenizer().unwrap();
        assert_eq!(
            tokenizer.encode("hello world again").unwrap().token_ids(),
            &[1, 2, 3],
            "online tokenization must not apply serialized padding or truncation, \
             which would give [1, 2, 0, 0, 0, 0, 0, 0]"
        );
    });
}

#[test]
#[serial_test::serial]
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
