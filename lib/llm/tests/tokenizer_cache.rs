// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use dynamo_llm::{
    local_model::runtime_config::TokenizerBackend,
    model_card::ModelDeploymentCard,
    tokenizers::{HuggingFaceTokenizer, traits::Encoder},
};
use dynamo_runtime::metrics::frontend_perf::{
    TOKENIZER_CACHE_CACHED_TOKENS_TOTAL, TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL,
};

// Keep this in its own test binary: the production cache reads its budget once.
#[test]
fn model_card_tokenizer_cache_reuses_only_matching_identities() -> anyhow::Result<()> {
    temp_env::with_vars(
        [
            ("DYN_TOKENIZER_CACHE", Some("1")),
            ("DYN_TOKENIZER_CACHE_BYTES", Some("1048576")),
            ("DYN_TOKENIZER_CACHE_EXTEND", Some("1")),
        ],
        || -> anyhow::Result<()> {
            let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/TinyLlama_v1.1");
            let load_card = |backend| -> anyhow::Result<ModelDeploymentCard> {
                let mut card = ModelDeploymentCard::load_from_disk(&model_dir, None)?;
                card.runtime_config.tokenizer_backend = Some(backend);
                card.runtime_config.tokenizer_fallback_enabled = Some(false);
                Ok(card)
            };
            let card = load_card(TokenizerBackend::Default)?;
            let cached =
                TOKENIZER_CACHE_CACHED_TOKENS_TOTAL.with_label_values(&[&card.display_name]);
            let uncached =
                TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL.with_label_values(&[&card.display_name]);
            let input = "<s>system\nHello 世界</s><s>user\nOne</s>";
            let expected_encoding = HuggingFaceTokenizer::from_file(
                model_dir.join("tokenizer.json").to_str().unwrap(),
            )?
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
            let reload = load_card(TokenizerBackend::Default)?;
            assert_eq!(card.mdcsum(), reload.mdcsum());
            check(&reload, true)?;

            let mut changed = load_card(TokenizerBackend::Default)?;
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
                let mut fallback = load_card(TokenizerBackend::Fastokens)?;
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
            let fast = load_card(TokenizerBackend::Fastokens)?;
            assert_eq!(card.mdcsum(), fast.mdcsum());
            check(&fast, false)?;
            check(&fast, true)?;
            Ok(())
        },
    )
}
