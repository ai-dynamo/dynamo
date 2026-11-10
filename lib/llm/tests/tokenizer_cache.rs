// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use dynamo_llm::{
    local_model::runtime_config::TokenizerBackend,
    model_card::ModelDeploymentCard,
    tokenizers::{EncodeSegment, HuggingFaceTokenizer, traits::Encoder},
};
use dynamo_runtime::metrics::frontend_perf::{
    TOKENIZER_CACHE_CACHED_TOKENS_TOTAL, TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL,
};

// Keep these tests in their own binary, and give them all the same environment: the
// production cache reads its budget once per process.
const CACHE_ENV: [(&str, Option<&str>); 3] = [
    ("DYN_TOKENIZER_CACHE", Some("1")),
    ("DYN_TOKENIZER_CACHE_BYTES", Some("1048576")),
    ("DYN_TOKENIZER_CACHE_EXTEND", Some("1")),
];

fn load_card(dir: &Path, backend: TokenizerBackend) -> anyhow::Result<ModelDeploymentCard> {
    let mut card = ModelDeploymentCard::load_from_disk(dir, None)?;
    card.runtime_config.tokenizer_backend = Some(backend);
    card.runtime_config.tokenizer_fallback_enabled = Some(false);
    Ok(card)
}

#[test]
fn model_card_tokenizer_cache_reuses_only_matching_identities() -> anyhow::Result<()> {
    temp_env::with_vars(CACHE_ENV, || -> anyhow::Result<()> {
        let model_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-models/TinyLlama_v1.1");
        let card = load_card(&model_dir, TokenizerBackend::Default)?;
        let cached = TOKENIZER_CACHE_CACHED_TOKENS_TOTAL.with_label_values(&[&card.display_name]);
        let uncached =
            TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL.with_label_values(&[&card.display_name]);
        let input = "<s>system\nHello 世界</s><s>user\nOne</s>";
        let expected_encoding =
            HuggingFaceTokenizer::from_file(model_dir.join("tokenizer.json").to_str().unwrap())?
                .encode(input)?;
        let expected = expected_encoding.token_ids();
        let check = |card: &ModelDeploymentCard, hit: bool| -> anyhow::Result<()> {
            let before = (cached.get(), uncached.get());
            // The wrapper is dropped after each encode; storage must survive it.
            assert_eq!(card.tokenizer()?.encode(input)?.token_ids(), expected);
            let cached_tokens = cached.get() - before.0;
            let uncached_tokens = uncached.get() - before.1;
            assert_eq!(cached_tokens > 0, hit);
            assert_eq!(cached_tokens + uncached_tokens, expected.len() as u64);
            Ok(())
        };

        check(&card, false)?;
        let reload = load_card(&model_dir, TokenizerBackend::Default)?;
        assert_eq!(card.mdcsum(), reload.mdcsum());
        check(&reload, true)?;

        let mut changed = load_card(&model_dir, TokenizerBackend::Default)?;
        changed.kv_cache_block_size = 16;
        assert_eq!(card.display_name, changed.display_name);
        assert_ne!(card.mdcsum(), changed.mdcsum());
        check(&changed, false)?;

        #[cfg(unix)]
        {
            use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

            use dynamo_llm::{common::checked_file::CheckedFile, model_card::TokenizerKind};

            // HF accepts a Path; the alternate backends require UTF-8 paths.
            let temp = tempfile::tempdir()?;
            let path = temp.path().join(OsStr::from_bytes(b"tokenizer-\xff.json"));
            std::fs::copy(model_dir.join("tokenizer.json"), &path)?;
            std::fs::copy(
                model_dir.join("tokenizer_config.json"),
                temp.path().join("tokenizer_config.json"),
            )?;
            let mut fallback = load_card(&model_dir, TokenizerBackend::Fastokens)?;
            fallback.tokenizer = Some(TokenizerKind::HfTokenizerJson(CheckedFile::from_disk(
                path,
            )?));
            assert!(fallback.tokenizer().is_err(), "fallback is still disabled");
            fallback.runtime_config.tokenizer_fallback_enabled = Some(true);
            assert_eq!(card.mdcsum(), fallback.mdcsum());
            // Only HF is warm: a fallback tagged as Fastokens must miss.
            check(&fallback, true)?;
        }

        // Fallback is disabled, so this must successfully load Fastokens.
        let fast = load_card(&model_dir, TokenizerBackend::Fastokens)?;
        assert_eq!(card.mdcsum(), fast.mdcsum());
        check(&fast, false)?;
        check(&fast, true)?;
        Ok(())
    })
}

// The cache encodes long segments and the specials around them separately, so any
// truncation or padding left in `tokenizer.json` would apply to each piece.
#[test]
fn model_card_tokenizer_ignores_truncation_and_padding() -> anyhow::Result<()> {
    temp_env::with_vars(CACHE_ENV, || -> anyhow::Result<()> {
        let model_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-models/TinyLlama_v1.1");
        let temp = tempfile::tempdir()?;
        for entry in std::fs::read_dir(&model_dir)? {
            let entry = entry?;
            std::fs::copy(entry.path(), temp.path().join(entry.file_name()))?;
        }
        let path = temp.path().join("tokenizer.json");
        let mut json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        json["truncation"] = serde_json::json!({
            "direction": "Right", "max_length": 12, "strategy": "LongestFirst", "stride": 0
        });
        json["padding"] = serde_json::json!({
            "strategy": {"Fixed": 512}, "direction": "Right", "pad_to_multiple_of": null,
            "pad_id": 0, "pad_type_id": 0, "pad_token": "<unk>"
        });
        std::fs::write(&path, serde_json::to_vec(&json)?)?;

        let user = "hello world ".repeat(16);
        let segments = [
            EncodeSegment::control("<s>"),
            EncodeSegment::ordinary(&user),
            EncodeSegment::control("</s>"),
        ];
        let expected =
            HuggingFaceTokenizer::from_file(model_dir.join("tokenizer.json").to_str().unwrap())?
                .encode_segments(&segments)?;
        for backend in [TokenizerBackend::Default, TokenizerBackend::Fastokens] {
            let card = load_card(temp.path(), backend)?;
            let cached =
                TOKENIZER_CACHE_CACHED_TOKENS_TOTAL.with_label_values(&[&card.display_name]);
            let tokenizer = card.tokenizer()?;
            for warm in [false, true] {
                let before = cached.get();
                assert_eq!(
                    tokenizer.encode_segments(&segments)?.token_ids(),
                    expected.token_ids(),
                    "{backend:?}"
                );
                assert_eq!(cached.get() > before, warm, "{backend:?}");
            }
        }
        Ok(())
    })
}
