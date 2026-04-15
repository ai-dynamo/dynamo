// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub struct HfTokenizerFactory {
    tokenizer_path: PathBuf,
}

impl HfTokenizerFactory {
    pub fn resolve(model_or_path: &str) -> Result<Self> {
        Ok(Self {
            tokenizer_path: resolve_tokenizer_path(model_or_path)?,
        })
    }
}

pub trait TokenizerWorker: Send + 'static {
    fn encode(&mut self, text: &str) -> Result<Vec<u32>>;

    fn encode_with_word_overlap(
        &mut self,
        text: &str,
        previous_text: Option<&str>,
        previous_tokens: Option<&[u32]>,
        overlap_words: usize,
    ) -> Result<Vec<u32>> {
        if overlap_words == 0 {
            return self.encode(text);
        }
        let Some(previous_text) = previous_text else {
            return self.encode(text);
        };
        let Some(previous_tokens) = previous_tokens else {
            return self.encode(text);
        };
        if !text.starts_with(previous_text) {
            return self.encode(text);
        }

        let overlap_start = last_word_overlap_start(previous_text, overlap_words);
        let previous_suffix_tokens = self.encode(&previous_text[overlap_start..])?;
        let prefix_token_count = previous_tokens
            .len()
            .saturating_sub(previous_suffix_tokens.len());
        let suffix_tokens = self.encode(&text[overlap_start..])?;

        let mut merged = Vec::with_capacity(prefix_token_count + suffix_tokens.len());
        merged.extend_from_slice(&previous_tokens[..prefix_token_count]);
        merged.extend(suffix_tokens);
        Ok(merged)
    }
}

pub trait TokenizerFactory: Clone + Send + Sync + 'static {
    type Worker: TokenizerWorker;

    fn create_worker(&self) -> Result<Self::Worker>;
}

impl TokenizerFactory for HfTokenizerFactory {
    type Worker = HfTokenizerWorker;

    fn create_worker(&self) -> Result<Self::Worker> {
        HfTokenizerWorker::from_file(&self.tokenizer_path)
    }
}

pub struct HfTokenizerWorker {
    tokenizer: Tokenizer,
}

impl HfTokenizerWorker {
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|error| {
            anyhow!("failed to load tokenizer from {}: {error}", path.display())
        })?;
        Ok(Self { tokenizer })
    }
}

impl TokenizerWorker for HfTokenizerWorker {
    fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| anyhow!("failed to tokenize input: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }
}

fn resolve_tokenizer_path(model_or_path: &str) -> Result<PathBuf> {
    let path = Path::new(model_or_path);
    if path.is_file() {
        return Ok(path.to_path_buf());
    }

    if path.is_dir() {
        let tokenizer_path = path.join("tokenizer.json");
        if tokenizer_path.exists() {
            return Ok(tokenizer_path);
        }
        bail!(
            "directory '{}' does not contain tokenizer.json",
            path.display()
        );
    }

    let cache = hf_hub::Cache::default();
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .context("failed to create HuggingFace API client")?;
    let repo = api.model(model_or_path.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .with_context(|| format!("failed to download tokenizer.json from '{model_or_path}'"))?;
    Ok(tokenizer_path)
}

pub(crate) fn last_word_overlap_start(text: &str, overlap_words: usize) -> usize {
    if overlap_words == 0 || text.is_empty() {
        return text.len();
    }

    let mut starts = Vec::new();
    let mut in_word = false;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            in_word = false;
            continue;
        }
        if !in_word {
            starts.push(index);
            in_word = true;
        }
    }

    if starts.len() <= overlap_words {
        return 0;
    }
    starts[starts.len() - overlap_words]
}

#[cfg(test)]
mod tests {
    use super::{TokenizerWorker, last_word_overlap_start};
    use anyhow::Result;
    use std::thread;
    use std::time::Duration;

    struct StubWorker;

    impl TokenizerWorker for StubWorker {
        fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
            if text.contains("slow") {
                thread::sleep(Duration::from_millis(5));
            }
            Ok(text
                .split_whitespace()
                .map(|word| word.len() as u32)
                .collect())
        }
    }

    #[test]
    fn overlap_start_returns_zero_for_short_text() {
        assert_eq!(last_word_overlap_start("one two", 10), 0);
    }

    #[test]
    fn overlap_encoding_falls_back_on_prefix_break() {
        let mut worker = StubWorker;
        let tokens = worker
            .encode_with_word_overlap("a b c", Some("z y"), Some(&[1, 2]), 2)
            .unwrap();
        assert_eq!(tokens, vec![1, 1, 1]);
    }
}
