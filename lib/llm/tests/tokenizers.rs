// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Tokenizer Tests
//!
//! This module contains tests for the Tokenizer.
//!
//! For each tokenizer we use in production, we should have either a url to or a local copy
//! of either the tokenizer.json or the .model file.
//!
//! For a small set of common prompts, we need to have a hashable representation of the the encoding
//! object. We will precompute the hashes for each of these prompts for each tokenizer and store them
//! in a hashmap. We will then use these hashes to test that the tokenizer is working correctly. This
//! will detect if upstream dependency changes result in different/new behavior.

use dynamo_llm::tokenizers::traits::{Decoder, Encoder};
use dynamo_llm::tokenizers::*;
use std::collections::HashMap;
use std::sync::Arc;

const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

const LONG_TEST_PROMPTS: [&str; 4] = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.",
    "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem."
];

const TINYLLAMA_TOKENIZER_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1/tokenizer.json";

const HF_TOKENIZERS_LOCAL: [&str; 1] = [TINYLLAMA_TOKENIZER_PATH];

const HASHES: [(&str, [u64; 4]); 1] = [(
    TINYLLAMA_TOKENIZER_PATH,
    [
        1209591529327510910,
        4181375434596349981,
        6245658446118930933,
        5097285695902185237,
    ],
)];

fn compute_hashes_for_tokenizer<E: Encoder>(tokenizer: &E, prompts: &[&str]) -> Vec<u64> {
    prompts
        .iter()
        .map(|&prompt| {
            tokenizer
                .encode(prompt)
                .expect("Failed to encode prompt")
                .get_hash()
            // Assuming `get_hash` returns a `u64`
        })
        .collect()
}

#[test]
fn compute_hashes_hf() {
    let hash_map: HashMap<&str, [u64; 4]> = HASHES.iter().cloned().collect();

    for &tokenizer_name in HF_TOKENIZERS_LOCAL.iter() {
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_name)
            .expect("Failed to load HuggingFace tokenizer");

        let prompt_hashes = compute_hashes_for_tokenizer(&tokenizer, &TEST_PROMPTS);

        println!(
            "HF Tokenizer: {:?} Hashes: {:?}",
            tokenizer_name, prompt_hashes
        );

        assert_eq!(prompt_hashes, hash_map[tokenizer_name]);
    }
}

#[test]
fn test_hf_lifecycle() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    let encoding = tokenizer
        .encode(TEST_PROMPTS[0])
        .expect("Failed to encode prompt");

    let decoded = tokenizer
        .decode(encoding.token_ids(), false)
        .expect("Failed to decode token_ids");

    assert_eq!(decoded, TEST_PROMPTS[0]);
}

#[test]
fn test_sequence() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    // let tokenizer = shared_tokenizer.read().unwrap();

    let encoding = shared_tokenizer
        .encode(TEST_PROMPTS[0])
        .expect("Failed to encode prompt");

    let mut sequence = Sequence::new(shared_tokenizer.clone().into());
    sequence
        .append_text(TEST_PROMPTS[0])
        .expect("Failed to append prompt");

    assert_eq!(sequence.len(), encoding.token_ids().len());

    let mut decoder = Sequence::new(shared_tokenizer.clone().into());

    let mut output = String::new();
    for token_id in encoding.token_ids() {
        let text = decoder
            .append_token_id(*token_id)
            .expect("Failed to decode token_id");
        output.push_str(text.as_str());
    }

    assert_eq!(decoder.len(), sequence.len());
    assert_eq!(decoder.token_ids(), sequence.token_ids());
    assert_eq!(output, TEST_PROMPTS[0]);

    let mut decoder = DecodeStream::new(shared_tokenizer.clone(), false);
    let mut output = String::new();
    for token_id in encoding.token_ids() {
        let text = decoder.step(*token_id).expect("Failed to decode token_id");
        if let Some(text) = text {
            output.push_str(text.as_str());
        }
    }
    assert_eq!(output, TEST_PROMPTS[0]);
}

#[test]
fn test_long_sequence_sliding_window() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    for sequence in LONG_TEST_PROMPTS.iter() {
        let encoding = shared_tokenizer
            .encode(sequence)
            .expect("Failed to encode prompt");

        let mut decoder = DecodeStream::new(shared_tokenizer.clone(), false);

        let mut output = String::new();
        for token_id in encoding.token_ids() {
            let text = decoder.step(*token_id).expect("Failed to decode token_id");
            if let Some(text) = text {
                output.push_str(text.as_str());
            }
        }

        assert_eq!(output, sequence.to_string());
    }
}
