// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Check token-ID parity between HuggingFace tokenizers, Fastokens, and Gigatoken.
//!
//! This is intentionally a standalone nightly-only experiment: it keeps the
//! Gigatoken dependency out of Dynamo's stable production dependency graph.

use std::{env, fs, path::PathBuf};

use gigatoken_rs::{
    WorkerPool, encode_docs_ragged,
    load_tokenizer::hf::{HfTokenizer, load_hf_slice},
    sp_encode_docs_ragged,
};

// Keep this identical to `lib/llm/benches/tokenizer_simple.rs`.
const SIMPLE_PROMPT: &str = "The cat sat by the window, watching raindrops race down the glass. Far thunder rumbled. She purred softly, feeling safe at home.";
const SIMPLE_INPUT_SENTINEL: &str = "__dynamo_tokenizer_simple_input__";

struct Args {
    tokenizer: PathBuf,
    input: PathBuf,
    documents: usize,
}

fn required_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut tokenizer = None;
        let mut input = None;
        let mut documents = 1;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokenizer" => {
                    tokenizer = Some(PathBuf::from(required_value(&mut args, "--tokenizer")?))
                }
                "--input" => input = Some(PathBuf::from(required_value(&mut args, "--input")?)),
                "--simple" => input = Some(PathBuf::from(SIMPLE_INPUT_SENTINEL)),
                "--documents" => {
                    documents = required_value(&mut args, "--documents")?
                        .parse()
                        .map_err(|_| "--documents must be a positive integer".to_string())?
                }
                "-h" | "--help" => return Err(Self::usage()),
                _ => return Err(format!("unknown argument {arg}\n\n{}", Self::usage())),
            }
        }

        let args = Self {
            tokenizer: tokenizer.ok_or_else(Self::usage)?,
            input: input.ok_or_else(Self::usage)?,
            documents,
        };
        if args.documents == 0 {
            return Err("--documents must be positive".to_string());
        }
        Ok(args)
    }

    fn usage() -> String {
        "Usage: cargo +nightly -Zprofile-rustflags run -- --tokenizer PATH (--simple | --input PATH) [--documents N]".to_string()
    }
}

fn split_documents(input: &str, count: usize) -> Vec<&str> {
    if count == 1 {
        return vec![input];
    }

    let target = input.len().div_ceil(count);
    let mut documents = Vec::with_capacity(count);
    let mut remaining = input;
    while !remaining.is_empty() && documents.len() + 1 < count {
        let mut split_at = target.min(remaining.len());
        while split_at > 0 && !remaining.is_char_boundary(split_at) {
            split_at -= 1;
        }
        if let Some(newline) = remaining[..split_at].rfind('\n') {
            split_at = newline + 1;
        }
        if split_at == 0 {
            split_at = remaining
                .char_indices()
                .nth(1)
                .map_or(remaining.len(), |(index, _)| index);
        }
        let (document, rest) = remaining.split_at(split_at);
        documents.push(document);
        remaining = rest;
    }
    if !remaining.is_empty() {
        documents.push(remaining);
    }
    documents
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let input = if args.input == PathBuf::from(SIMPLE_INPUT_SENTINEL) {
        SIMPLE_PROMPT.repeat(8_000 / SIMPLE_PROMPT.len())
    } else {
        fs::read_to_string(&args.input)
            .map_err(|error| format!("failed to read {}: {error}", args.input.display()))?
    };
    let documents = split_documents(&input, args.documents);
    let bytes = documents
        .iter()
        .map(|document| document.len())
        .sum::<usize>();
    println!("input={bytes} bytes, documents={}", documents.len());

    let hf = tokenizers::Tokenizer::from_file(&args.tokenizer)
        .map_err(|error| format!("failed to load HF tokenizer: {error}"))?;
    let fast = fastokens::Tokenizer::from_file(&args.tokenizer)
        .map_err(|error| format!("failed to load Fastokens tokenizer: {error}"))?;
    let hf_documents = documents.to_vec();
    let fast_documents = documents.to_vec();
    let gigatoken_documents = documents.clone();

    let hf_ids = hf
        .encode_batch(hf_documents, false)
        .map(|encodings| {
            encodings
                .into_iter()
                .flat_map(|encoding| encoding.get_ids().to_vec())
                .collect::<Vec<_>>()
        })
        .map_err(|error| error.to_string())?;
    let fast_ids = fast
        .encode_batch(&fast_documents, false)
        .map(|encodings| encodings.into_iter().flatten().collect::<Vec<_>>())
        .map_err(|error| error.to_string())?;

    let tokenizer_json = fs::read(&args.tokenizer)
        .map_err(|error| format!("failed to read {}: {error}", args.tokenizer.display()))?;
    let gigatoken = load_hf_slice(&tokenizer_json)
        .map_err(|error| format!("failed to load Gigatoken tokenizer: {error}"))?;
    let gigatoken_ids = match gigatoken {
        HfTokenizer::Bpe(tokenizer) => {
            let worker_pool = WorkerPool::new();
            let documents: Vec<&[u8]> = gigatoken_documents
                .iter()
                .map(|document| document.as_bytes())
                .collect();
            encode_docs_ragged(&worker_pool, &tokenizer, &documents).0
        }
        HfTokenizer::SentencePiece(tokenizer) => {
            sp_encode_docs_ragged(&tokenizer, &gigatoken_documents).0
        }
    };

    if hf_ids != fast_ids || hf_ids != gigatoken_ids {
        return Err(format!(
            "token parity failed: hf={}, fastokens={}, gigatoken={}",
            hf_ids.len(),
            fast_ids.len(),
            gigatoken_ids.len(),
        ));
    }

    println!(
        "token parity: OK ({} token IDs for all backends)",
        hf_ids.len()
    );
    Ok(())
}
