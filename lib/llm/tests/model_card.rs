// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::common::checked_file::CheckedFile;
use dynamo_llm::local_model::runtime_config::TokenizerBackend;
use dynamo_llm::model_card::{ModelDeploymentCard, PromptFormatterArtifact, TokenizerKind};
use dynamo_llm::tokenizers::{
    BasetenTokenizer, DecodeStream, EncodeSegment, Encoding, HuggingFaceTokenizer,
    TikTokenTokenizer,
    traits::{Decoder, Encoder},
};
use serde_json::json;
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};
use tokenizers::models::bpe::{BPE, Vocab};

const HF_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";
const TIKTOKEN_PATH: &str = "tests/data/sample-models/mock-tiktoken";

fn byte_level_tokenizer() -> tokenizers::Tokenizer {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let mut vocab = Vocab::from_iter(
        alphabet
            .into_iter()
            .enumerate()
            .map(|(id, c)| (c.to_string(), id as u32)),
    );
    let merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o")];
    for (left, right) in merges {
        vocab.insert(format!("{left}{right}"), vocab.len() as u32);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(
            vocab,
            merges.map(|(a, b)| (a.to_owned(), b.to_owned())).to_vec(),
        )
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, true),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
}

fn card_for_tokenizer(
    tokenizer: &tokenizers::Tokenizer,
    name: &str,
) -> (TempDir, ModelDeploymentCard) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    tokenizer.save(&path, false).unwrap();
    let mut card = ModelDeploymentCard::with_name_only(name);
    card.tokenizer = Some(TokenizerKind::HfTokenizerJson(
        CheckedFile::from_disk(path).unwrap(),
    ));
    (dir, card)
}

#[test]
#[serial_test::serial]
fn test_hf_rc_mock_byte_alphabets_preserve_declared_special_ids() {
    temp_env::with_vars([("DYN_TOKENIZER_CACHE", Some("0"))], || {
        for model in ["mock-llama-3.1-8b-instruct", "mock-deepseek-r1"] {
            let path = format!("tests/data/sample-models/{model}");
            let mut legacy =
                tokenizers::Tokenizer::from_file(format!("{path}/tokenizer.json")).unwrap();
            let mut card = ModelDeploymentCard::with_name_only(model);
            card.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                CheckedFile::from_disk(format!("{path}/tokenizer.json")).unwrap(),
            ));
            let rc = card.tokenizer().unwrap();
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(format!("{path}/tokenizer.json")).unwrap(),
            )
            .unwrap();
            for token in config["added_tokens"].as_array().unwrap() {
                let id = token["id"].as_u64().unwrap() as u32;
                let content = token["content"].as_str().unwrap();
                assert_eq!(
                    rc.encode(content).unwrap().token_ids(),
                    &[id],
                    "{model}: {content}"
                );
                for skip in [false, true] {
                    assert_eq!(
                        rc.decode(&[id], skip).unwrap().as_str(),
                        legacy.decode(&[id], skip).unwrap(),
                        "{model}: token {id}"
                    );
                }
            }
            let ids = legacy.encode("hello café 北京 😀", false).unwrap();
            assert_eq!(
                rc.encode("hello café 北京 😀").unwrap().token_ids(),
                ids.get_ids()
            );
            assert_eq!(
                rc.decode(ids.get_ids(), false).unwrap().as_str(),
                legacy.decode(ids.get_ids(), false).unwrap()
            );

            // rc.2 reassigns a new added token above these sparse model IDs.
            legacy.add_special_tokens(&[tokenizers::AddedToken::from("<new>", true)]);
            let (_dir, card) = card_for_tokenizer(&legacy, "sparse-added-id");
            let error = format!(
                "{:#}",
                card.tokenizer().err().expect("changed token ID must fail")
            );
            assert!(
                error.contains("unsupported added token") && error.contains("sparse-added-id"),
                "{error}"
            );
        }
    });
}

#[test]
#[serial_test::serial]
fn test_hf_rc_encode_decode_parity() {
    temp_env::with_vars([("DYN_TOKENIZER_CACHE", Some("0"))], || {
        let mut cases = vec![
            ("byte-level BPE", byte_level_tokenizer()),
            (
                "byte-fallback BPE",
                tokenizers::Tokenizer::from_file(format!("{HF_PATH}/tokenizer.json")).unwrap(),
            ),
        ];
        for (name, model, pre_tokenizer, decoder) in [
            (
                "WordPiece",
                json!({"type":"WordPiece", "unk_token":"[UNK]", "continuing_subword_prefix":"##", "max_input_chars_per_word":100,
                "vocab":{"[UNK]":0,"hello":1,"world":2,"##s":3,"café":4,"北":5,"京":6,"😀":7}}),
                json!({"type":"Whitespace"}),
                json!({"type":"WordPiece","prefix":"##","cleanup":true}),
            ),
            (
                "WordLevel",
                json!({"type":"WordLevel", "unk_token":"[UNK]", "vocab":{"[UNK]":0,"hello":1,"world":2,"café":3,"北京":4,"😀":5}}),
                json!({"type":"Whitespace"}),
                json!(null),
            ),
            (
                "Unigram",
                json!({"type":"Unigram", "unk_id":0, "byte_fallback":false,
                "vocab":[["<unk>",0.0],["hello",-1.0],["world",-1.0],[" ",-1.0],["café",-1.0],["北",-1.0],["京",-1.0],["😀",-1.0]]}),
                json!(null),
                json!({"type":"Fuse"}),
            ),
        ] {
            cases.push((name, tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json!({
                "version":"1.0", "model":model, "pre_tokenizer":pre_tokenizer, "decoder":decoder,
                "added_tokens":[], "normalizer":null, "post_processor":null, "padding":null, "truncation":null,
            })).unwrap()).unwrap()));
        }

        for (name, mut legacy) in cases {
            let first_added_id = legacy.get_vocab_size(true) as u32;
            legacy.add_special_tokens(&[tokenizers::AddedToken::from("<special>", true)]);
            assert_eq!(legacy.token_to_id("<special>"), Some(first_added_id));
            let (_dir, card) = card_for_tokenizer(&legacy, name);
            let rc = card.tokenizer().unwrap();
            let inputs = [
                "",
                "hello world",
                "hello worlds",
                "café 北京 😀",
                "e\u{301}\n\t",
                "<special>hello<special>",
            ];
            let batch = rc.encode_batch(&inputs).unwrap();
            assert_eq!(batch.len(), inputs.len(), "{name}");
            assert!(rc.encode_batch(&[]).unwrap().is_empty(), "{name}");
            for (input, batch_encoding) in inputs.iter().zip(batch) {
                let expected = legacy.encode(*input, false).unwrap();
                let actual = rc.encode(input).unwrap();
                assert!(
                    matches!(actual, Encoding::Sp(_)),
                    "{name}: must use RC adapter"
                );
                assert_eq!(actual.token_ids(), expected.get_ids(), "{name}: {input:?}");
                assert_eq!(
                    batch_encoding.token_ids(),
                    expected.get_ids(),
                    "{name}: batch {input:?}"
                );
                for skip in [false, true] {
                    assert_eq!(
                        rc.decode(actual.token_ids(), skip).unwrap().as_str(),
                        legacy.decode(expected.get_ids(), skip).unwrap(),
                        "{name}: decode {input:?}, skip={skip}"
                    );
                }
            }
            assert_eq!(
                rc.decode(&[first_added_id], false).unwrap().as_str(),
                "<special>",
                "{name}"
            );
            assert_eq!(
                rc.decode(&[first_added_id], true).unwrap().as_str(),
                "",
                "{name}"
            );
            for (id, token) in legacy.get_added_tokens_decoder() {
                assert_eq!(
                    rc.encode(&token.content).unwrap().token_ids(),
                    legacy.encode(token.content, false).unwrap().get_ids(),
                    "{name}: added token {id}"
                );
                for skip in [false, true] {
                    assert_eq!(
                        rc.decode(&[id], skip).unwrap().as_str(),
                        legacy.decode(&[id], skip).unwrap(),
                        "{name}: added token {id}, skip={skip}"
                    );
                }
            }
        }
    });
}

#[test]
#[serial_test::serial]
fn test_hf_rc_streaming_utf8_and_prompt_context() {
    temp_env::with_vars([("DYN_TOKENIZER_CACHE", Some("0"))], || {
        for legacy in [
            byte_level_tokenizer(),
            tokenizers::Tokenizer::from_file(format!("{HF_PATH}/tokenizer.json")).unwrap(),
        ] {
            let (_dir, card) = card_for_tokenizer(&legacy, "streaming-parity");
            let rc = card.tokenizer().unwrap();
            let oracle = std::sync::Arc::new(HuggingFaceTokenizer::from_tokenizer(legacy.clone()));
            let ids = legacy
                .encode("hello 😀 北京!", false)
                .unwrap()
                .get_ids()
                .to_vec();
            let mut saw_partial = false;
            for end in 1..=ids.len() {
                let expected = legacy.decode(&ids[..end], false).unwrap();
                let decoded = rc.decode(&ids[..end], false).unwrap();
                assert_eq!(decoded.as_str(), expected);
                assert_eq!(decoded.is_partial(), expected.ends_with('\u{fffd}'));
                saw_partial |= decoded.is_partial();
            }
            assert!(
                saw_partial,
                "fixture must split at least one UTF-8 character"
            );
            // Test empty, complete, and incomplete UTF-8 prompt contexts.
            for prompt_len in 0..ids.len() {
                let mut actual = rc.decode_stream(&ids[..prompt_len], false);
                let mut expected = DecodeStream::new(oracle.clone(), &ids[..prompt_len], false);
                for &id in &ids[prompt_len..] {
                    assert_eq!(
                        actual.step(id).unwrap(),
                        expected.step(id).unwrap(),
                        "prompt length {prompt_len}"
                    );
                }
            }
        }
    });
}

#[test]
#[serial_test::serial]
fn test_hf_rc_rejects_unsupported_configurations_without_legacy_fallback() {
    temp_env::with_vars(
        [
            ("DYN_TOKENIZER_CACHE", Some("0")),
            ("DYN_TOKENIZER_FALLBACK", Some("1")),
        ],
        || {
            let base: serde_json::Value =
                serde_json::from_str(&byte_level_tokenizer().to_string(false).unwrap()).unwrap();
            let byte_level = base["decoder"].clone();
            for (decoder, supported) in [
                (byte_level.clone(), true),
                (
                    json!({"type":"Sequence","decoders":[byte_level.clone()]}),
                    true,
                ),
                (json!(null), false),
                (
                    json!({"type":"Sequence","decoders":[byte_level, {"type":"Replace","pattern":{"String":"h"},"content":"x"}]}),
                    false,
                ),
            ] {
                let mut config = base.clone();
                config["decoder"] = decoder;
                let legacy =
                    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&config).unwrap())
                        .unwrap();
                let (dir, card) = card_for_tokenizer(&legacy, "decoder-compatibility");
                match card.tokenizer() {
                    Ok(rc) => {
                        assert!(supported);
                        let ids = legacy.encode("hello", false).unwrap().get_ids().to_vec();
                        assert_eq!(
                            rc.decode(&ids, false).unwrap().as_str(),
                            legacy.decode(&ids, false).unwrap()
                        );
                    }
                    Err(error) => {
                        assert!(!supported);
                        let message = format!("{error:#}");
                        assert!(
                            message.contains("unsupported byte-level BPE decoder"),
                            "{message}"
                        );
                        assert!(
                            message.contains("decoder-compatibility")
                                && message.contains(dir.path().to_str().unwrap()),
                            "{message}"
                        );
                    }
                }
            }
            let mut config = base;
            config["pre_tokenizer"]["add_prefix_space"] = json!(true);
            let legacy =
                tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&config).unwrap()).unwrap();
            assert!(legacy.encode("hello", false).is_ok());
            let (dir, card) = card_for_tokenizer(&legacy, "prefix-space-compatibility");
            let error = format!(
                "{:#}",
                card.tokenizer()
                    .err()
                    .expect("RC conversion must fail, without legacy fallback")
            );
            assert!(error.contains("add_prefix_space"), "{error}");
            assert!(
                error.contains("prefix-space-compatibility")
                    && error.contains(dir.path().to_str().unwrap()),
                "{error}"
            );
        },
    );
}

#[test]
#[serial_test::serial]
fn test_hf_rc_preserves_config_only_special_token_matching_flags() {
    temp_env::with_vars([("DYN_TOKENIZER_CACHE", Some("0"))], || {
        let mut legacy = byte_level_tokenizer();
        legacy.with_normalizer(Some(tokenizers::normalizers::Lowercase));
        let special_id = legacy.get_vocab_size(true) as u32;
        let (dir, card) = card_for_tokenizer(&legacy, "config-only-special");
        let path = dir.path().join("tokenizer.json");
        let original = std::fs::read(&path).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"added_tokens_decoder": {special_id.to_string(): {
                "content":"control", "special":true, "single_word":true,
                "lstrip":true, "rstrip":true, "normalized":true,
            }}})
            .to_string(),
        )
        .unwrap();
        let rc = card.tokenizer().unwrap();
        let oracle = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        for input in ["CONTROL", "control", "hello   control   world", "xcontrolx"] {
            assert_eq!(
                rc.encode(input).unwrap().token_ids(),
                oracle.encode(input).unwrap().token_ids(),
                "{input:?}"
            );
        }
        assert_eq!(
            rc.encode("   control   ").unwrap().token_ids(),
            &[special_id]
        );
        assert!(
            !rc.encode("xcontrolx")
                .unwrap()
                .token_ids()
                .contains(&special_id)
        );
        for skip in [false, true] {
            assert_eq!(
                rc.decode(&[special_id], skip).unwrap(),
                oracle.decode(&[special_id], skip).unwrap(),
                "special-token decode with skip={skip}"
            );
        }
        assert_eq!(
            std::fs::read(path).unwrap(),
            original,
            "conversion must not rewrite model files"
        );

        // rc.2 accepts this configuration, but decodes CONTROL as control.
        // Dynamo must reject it at startup instead of returning changed text.
        legacy
            .add_special_tokens(&[tokenizers::AddedToken::from("CONTROL", true).normalized(true)]);
        assert_eq!(legacy.decode(&[special_id], false).unwrap(), "CONTROL");
        let (_dir, card) = card_for_tokenizer(&legacy, "changed-added-token");
        let error = format!(
            "{:#}",
            card.tokenizer()
                .err()
                .expect("changed decoded form must fail")
        );
        assert!(
            error.contains("unsupported added token") && error.contains("changed-added-token"),
            "{error}"
        );
    });
}

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
fn test_hf_and_baseten_model_card_match_legacy_and_use_prefix_cache() {
    temp_env::with_vars(
        [
            ("DYN_TOKENIZER_CACHE", Some("1")),
            ("DYN_TOKENIZER_CACHE_EXTEND", Some("1")),
        ],
        || {
            for backend in [TokenizerBackend::Default, TokenizerBackend::Basetenkenizer] {
                let model = match backend {
                    TokenizerBackend::Default => "model-card-hf-rc-cache-integration",
                    _ => "model-card-basetenkenizer-cache-integration",
                };
                let dir = tempdir().unwrap();
                let tokenizer_path = dir.path().join("tokenizer.json");
                let vocab = Vocab::from_iter(
                    [
                        "<unk>", " ", "!", ",", "H", "T", "d", "e", "h", "l", "o", "r", "w", "He",
                        "ll", "llo", "or", "ld",
                    ]
                    .into_iter()
                    .enumerate()
                    .map(|(id, token)| (token.to_string(), id as u32)),
                );
                let merges = [("H", "e"), ("l", "l"), ("ll", "o"), ("o", "r"), ("l", "d")]
                    .into_iter()
                    .map(|(left, right)| (left.to_string(), right.to_string()))
                    .collect();
                let bpe = BPE::builder()
                    .vocab_and_merges(vocab, merges)
                    .unk_token("<unk>".to_string())
                    .build()
                    .unwrap();
                tokenizers::Tokenizer::new(bpe)
                    .save(&tokenizer_path, false)
                    .unwrap();
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
                mdc.runtime_config.tokenizer_backend = Some(backend);

                let production = mdc.tokenizer().unwrap();
                let direct =
                    HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();

                let cached_tokens =
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_CACHED_TOKENS_TOTAL
                        .with_label_values(&[model]);
                let uncached_tokens =
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL
                        .with_label_values(&[model]);
                let cached_before = cached_tokens.get();
                let uncached_before = uncached_tokens.get();

                let prompts = [
                    "<special>Hello, world!",
                    "<special>The world",
                    "<special>The world<special>Hello",
                    "<special>The world<special>Hello, world!",
                ];
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
                        "both production backends must remain token-exact"
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

                if matches!(backend, TokenizerBackend::Basetenkenizer) {
                    production
                        .encode_segments(&[EncodeSegment::new("Hello, world!", false)])
                        .expect(
                            "selected Baseten backend must preserve segmented encoding support",
                        );
                }
            }
        },
    );
}

#[test]
#[serial_test::serial]
fn test_alternate_tokenizer_load_failure_falls_back_to_hf_rc() {
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

            for backend in [
                TokenizerBackend::Basetenkenizer,
                TokenizerBackend::Fastokens,
            ] {
                let mut mdc = ModelDeploymentCard::with_name_only("alternate-fallback");
                mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
                    CheckedFile::from_disk(&tokenizer_path).unwrap(),
                ));
                mdc.runtime_config.tokenizer_backend = Some(backend);

                let tokenizer = mdc
                    .tokenizer()
                    .expect("unsupported Baseten tokenizer must fall back to HuggingFace");
                assert_eq!(
                    tokenizer.encode("hello").unwrap().token_ids(),
                    &[1],
                    "fallback must remain usable"
                );
                assert!(matches!(
                    tokenizer.encode("hello").unwrap(),
                    Encoding::Sp(_)
                ));
                assert_eq!(tokenizer.decode(&[1], false).unwrap().as_str(), "hello");
            }
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

        let raw = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
        assert_eq!(
            raw.encode("hello world again").unwrap().token_ids(),
            &[1, 2, 0, 0, 0, 0, 0, 0],
            "fixture must exercise serialized truncation and padding"
        );

        let mut mdc = ModelDeploymentCard::with_name_only("hf-online-normalization");
        mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
            CheckedFile::from_disk(&tokenizer_path).unwrap(),
        ));

        let tokenizer = mdc.tokenizer().unwrap();
        assert_eq!(
            tokenizer.encode("hello world again").unwrap().token_ids(),
            &[1, 2, 3],
            "online tokenization must not apply serialized padding or truncation"
        );
        let batch = tokenizer
            .encode_batch(&["hello world again", "hello"])
            .unwrap();
        assert_eq!(batch[0].token_ids(), &[1, 2, 3]);
        assert_eq!(batch[1].token_ids(), &[1]);
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
