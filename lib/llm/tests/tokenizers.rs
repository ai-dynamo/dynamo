// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

const LONG_TEST_PROMPTS: [(&str, &str); 6] = [
    ("Tell me about the following text.", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."),
    ("Tell me about the following text.", "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."),
    ("Tell me about the following text.", "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt."),
    ("Tell me about the following text.", "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem."),
    // Note(jthomson04): Ishan asked me to add this one.
    ("Tell me about the following text.", "In the ancient realm of Tennisia, the very magic of the land is drawn from the sport itself. Forehands light the skies, backhands carve the earth, and serves rumble like thunder across kingdoms. At the center of this balance lie four sacred Grand Slam relics: the Sapphire Trophy of Melbourne, the Emerald Chalice of Paris, the Ruby Crown of London, and the Diamond Orb of New York. Together, they keep the game's spirit alive.
    But the relics are scattered, guarded by champions of legendary skill. The first is the Fire King of Clay, ruler of the crimson courts, whose topspin arcs blaze high and heavy, scorching all who dare stand across from him. The second is the Tempest Trickster, master of the baseline fortress, whose footwork and precision can turn back any storm, and whose returns arrive as if pulled by invisible strings. The third is the Shadow-Dancer of the Highlands, a tactician who thrives in the long rallies of twilight, changing pace and spin until opponents lose their rhythm. The fourth and final guardian is a towering Diamond Titan, a net-charging colossus whose volleys shatter the air itself.
    Into this arena of gods steps the Silver-Wristed Knight — a player of impossible grace, whose game is an art form. His quest: to claim each relic not for glory, but to restore harmony to the rankings of the realm.
    He travels across the Kingdom of Clay, where the points stretch like marathons and the air tastes of iron; through the Grasslands of London, where the ball skids low and the margins are razor-thin; over the Hard Courts of the East, where rallies turn into duels of endurance; and finally to the Cathedral of Lights in New York, where night matches burn with fevered energy.
    Each battle is played under enchanted floodlights, the lines patrolled by spectral line judges whose calls are final. The crowd's roar swells with every break point, and the Silver-Wristed Knight's racket glows brightest when the match teeters at deuce. There are moments when doubt grips him — when his serve falters or his touch deserts him — but each challenge teaches a new stroke, culminating in the legendary Forehand of Dawn.
    When the last relic is claimed, he stands not as a conqueror but as a custodian of the game, knowing that rivalries forge the very magic he protects. The balance is restored — until the next season begins."),
    // Emoji stress test
    ("Tell me about the following text.", "😀😃😄😁😆🥹😅😂🤣🥲☺️😊😇🙂🙃😉🤩😎 🤪🥳🤓🙄🤪😵👻")
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

    let mut decoder = DecodeStream::new(shared_tokenizer.clone(), &[], false);
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
fn test_long_sequence_incremental_decode_with_prefill() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    let shared_tokenizer = Arc::new(tokenizer);

    for (input_text, output_text) in LONG_TEST_PROMPTS.iter() {
        let input_encoding = shared_tokenizer
            .encode(input_text)
            .expect("Failed to encode prompt");

        let output_encoding = shared_tokenizer
            .encode(output_text)
            .expect("Failed to encode prompt");

        let mut decoder =
            DecodeStream::new(shared_tokenizer.clone(), input_encoding.token_ids(), false);

        let mut output = String::new();
        for token_id in output_encoding.token_ids() {
            let text = decoder.step(*token_id).expect("Failed to decode token_id");
            if let Some(text) = text {
                output.push_str(text.as_str());
            }
        }

        assert_eq!(output.trim(), output_text.to_string());
    }
}

#[test]
fn test_decode_with_skip_special_tokens() {
    let tokenizer = HuggingFaceTokenizer::from_file(TINYLLAMA_TOKENIZER_PATH)
        .expect("Failed to load remote HuggingFace tokenizer");

    // Create a sequence with special tokens:
    // <s> (token_id: 1) + "Hello world" + </s> (token_id: 2)
    let text = "Hello world";
    let encoding = tokenizer.encode(text).expect("Failed to encode text");
    let mut token_ids = vec![1]; // <s>
    token_ids.extend(encoding.token_ids());
    token_ids.push(2); // </s>

    // Decode with skip_special_tokens = false (should keep special tokens)
    let decoded_with_special = tokenizer
        .decode(&token_ids, false)
        .expect("Failed to decode with skip_special_tokens=false");

    // Decode with skip_special_tokens = true (should remove special tokens)
    let decoded_without_special = tokenizer
        .decode(&token_ids, true)
        .expect("Failed to decode with skip_special_tokens=true");

    // Validate exact matches on the entire decoded strings
    assert_eq!(decoded_with_special, "<s> Hello world</s>");
    assert_eq!(decoded_without_special, "Hello world");
}
