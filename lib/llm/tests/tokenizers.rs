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

use std::collections::HashMap;
use std::sync::Arc;
use dynemo.llm::tokenizers::traits::{Decoder, Encoder};
use dynemo.llm::tokenizers::*;

const TEST_PROMPTS: [&str; 4] = [
    "deep learning is",
    "Deep learning is",
    "has anyone seen nemo lately",
    "another prompt",
];

const TINYLLAMA_TOKENIZER_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1/tokenizer.json";

const HF_TOKENIZERS_LOCAL: [&str; 1] = [TINYLLAMA_TOKENIZER_PATH];

const HASHES: [(&str, [u64; 4]); 1] = [(
    TINYLLAMA_TOKENIZER_PATH,
    [
        771185775798505393,
        8538328482215529710,
        17087868772360018644,
        1660219240238826577,
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
        .decode(&encoding.token_ids, false)
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

    assert_eq!(sequence.len(), encoding.token_ids.len());

    let mut decoder = Sequence::new(shared_tokenizer.clone().into());

    let mut output = String::new();
    for token_id in encoding.token_ids.clone() {
        let text = decoder
            .append_token_id(token_id)
            .expect("Failed to decode token_id");
        output.push_str(text.as_str());
    }

    assert_eq!(decoder.len(), sequence.len());
    assert_eq!(decoder.token_ids(), sequence.token_ids());
    assert_eq!(output, TEST_PROMPTS[0]);

    let mut decoder = DecodeStream::new(shared_tokenizer.clone(), false);
    let mut output = String::new();
    for token_id in encoding.token_ids {
        let text = decoder.step(token_id).expect("Failed to decode token_id");
        if let Some(text) = text {
            output.push_str(text.as_str());
        }
    }
    assert_eq!(output, TEST_PROMPTS[0]);
}
