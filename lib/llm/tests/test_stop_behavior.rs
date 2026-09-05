// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_llm::backend::{Decoder, StopTrigger};
use dynamo_llm::protocols::common::StopConditions;
use dynamo_llm::tokenizers::{self, Encoding, traits as tokenizer_traits};

const HI: u32 = 1;
const STOP: u32 = 2;
const THERE: u32 = 3;
const EOS: u32 = 99;

// Tokens whose decoded fragments only form a complete stop string once several of them
// are concatenated -- used to exercise a stop sequence split across multiple decode steps.
const DELTA: u32 = 10;
const EPSILON: u32 = 11;
const ZETA: u32 = 12;
const ETA: u32 = 13;
const THETA: u32 = 14;
const OH: u32 = 15;
const OTHER: u32 = 16;

struct TestTokenizer;

impl tokenizer_traits::Encoder for TestTokenizer {
    fn encode(&self, _: &str) -> Result<Encoding> {
        Ok(Encoding::Sp(vec![]))
    }
    fn encode_batch(&self, _: &[&str]) -> Result<Vec<Encoding>> {
        Ok(vec![])
    }
}

impl tokenizer_traits::Decoder for TestTokenizer {
    fn decode(&self, ids: &[u32], skip_special: bool) -> Result<tokenizer_traits::DecodeResult> {
        let text: String = ids
            .iter()
            .filter_map(|&id| match id {
                EOS if skip_special => None,
                HI => Some("hi"),
                STOP => Some("STOP"),
                THERE => Some("there"),
                EOS => Some("</s>"),
                DELTA => Some(" delta"),
                EPSILON => Some(" epsilon"),
                ZETA => Some(" zeta"),
                ETA => Some(" eta"),
                THETA => Some(" theta"),
                OH => Some("o"),
                OTHER => Some("there"),
                _ => Some("?"),
            })
            .collect();
        Ok(text.into())
    }
}

impl tokenizer_traits::Tokenizer for TestTokenizer {}

fn make_decoder(
    max_tokens: Option<u32>,
    min_tokens: Option<u32>,
    hidden_stop_ids: Option<Vec<u32>>,
    stop_sequences: Option<Vec<&str>>,
    include_stop_str: bool,
) -> Decoder {
    let tokenizer: Arc<dyn tokenizer_traits::Tokenizer> = Arc::new(TestTokenizer);
    let decode_stream = tokenizers::DecodeStream::new(tokenizer, &[], false);
    let stop_conditions = StopConditions {
        max_tokens,
        min_tokens,
        stop_token_ids_hidden: hidden_stop_ids,
        stop: stop_sequences.map(|v| v.into_iter().map(String::from).collect()),
        ..Default::default()
    };
    Decoder::new(decode_stream, stop_conditions, include_stop_str, None)
}

#[test]
fn normal_completion_no_stop() {
    let mut decoder = make_decoder(None, None, None, None, false);
    let result = decoder.process_token_ids(&[HI, THERE]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hithere"));
    assert!(result.stop_trigger.is_none());
}

#[test]
fn hidden_stop_token_excluded() {
    let mut decoder = make_decoder(None, None, Some(vec![EOS]), None, false);
    let result = decoder.process_token_ids(&[HI, EOS]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopTokenDetected(id)) if id == EOS
    ));
}

#[test]
fn include_stop_str_false_excludes() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["STOP"]), false);
    let result = decoder.process_token_ids(&[HI, STOP, THERE]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "STOP"
    ));
}

#[test]
fn include_stop_str_true_includes() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["STOP"]), true);
    let result = decoder.process_token_ids(&[HI, STOP, THERE]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hiSTOP"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::VisibleStopSequenceDetected(ref s)) if s == "STOP"
    ));
}

#[test]
fn trailing_tokens_ignored_after_stop() {
    let mut decoder = make_decoder(None, None, Some(vec![EOS]), None, false);
    let result = decoder.process_token_ids(&[HI, EOS, THERE]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert_eq!(result.tokens.len(), 2);
}

#[test]
fn min_tokens_delays_stop() {
    let mut decoder = make_decoder(None, Some(3), Some(vec![EOS]), None, false);
    let result = decoder.process_token_ids(&[HI, EOS]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi</s>"));
    assert!(result.stop_trigger.is_none());
}

#[test]
fn stop_token_priority_over_sequence() {
    let mut decoder = make_decoder(None, None, Some(vec![STOP]), Some(vec!["STOP"]), false);
    let result = decoder.process_token_ids(&[HI, STOP]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopTokenDetected(id)) if id == STOP
    ));
}

#[test]
fn user_stop_token_reports_distinct_trigger() {
    let tokenizer: Arc<dyn tokenizer_traits::Tokenizer> = Arc::new(TestTokenizer);
    let decode_stream = tokenizers::DecodeStream::new(tokenizer, &[], false);
    let stop_conditions = StopConditions {
        stop_token_ids: Some(vec![STOP]),
        stop_token_ids_hidden: Some(vec![EOS]),
        ..Default::default()
    };
    let mut decoder = Decoder::new(decode_stream, stop_conditions, false, None);
    let result = decoder.process_token_ids(&[HI, STOP]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::UserStopTokenDetected(id)) if id == STOP
    ));
}

/// Regression test for a hidden stop sequence split across several decode steps
/// (https://github.com/ai-dynamo/dynamo/issues/14375). The stop sequence " zeta eta theta"
/// arrives as three separate decoded fragments (" zeta", " eta", " theta"); none of the
/// earlier fragments should reach the caller before the full sequence is recognized.
#[test]
fn hidden_stop_sequence_split_across_tokens_is_not_leaked() {
    let mut decoder = make_decoder(None, None, None, Some(vec![" zeta eta theta"]), false);
    let result = decoder
        .process_token_ids(&[DELTA, EPSILON, ZETA, ETA, THETA])
        .unwrap();

    assert_eq!(result.text.as_deref(), Some(" delta epsilon"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == " zeta eta theta"
    ));
}

/// A withheld candidate prefix that turns out not to be part of the stop sequence must be
/// released once it can no longer complete, rather than being lost forever.
#[test]
fn withheld_prefix_is_released_once_it_cannot_complete() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["ozzy"]), false);
    let result = decoder.process_token_ids(&[OH, OTHER]).unwrap();

    // "o" is a prefix of "ozzy" and is withheld after the first token; once "there" arrives
    // the buffered text can no longer become "ozzy", so all of it is released together.
    assert_eq!(result.text.as_deref(), Some("othere"));
    assert!(result.stop_trigger.is_none());
}

/// A stop sequence that never completes must still be flushed when generation ends for a
/// reason our decoder did not itself detect (e.g. the engine's own `max_tokens` limit) --
/// otherwise a partial match silently swallows real output forever.
#[test]
fn flush_jailed_releases_incomplete_partial_match() {
    let mut decoder = make_decoder(None, None, None, Some(vec![" zeta eta theta"]), false);
    let result = decoder.process_token_ids(&[DELTA, ZETA, ETA]).unwrap();

    // " zeta eta" is a genuine (incomplete) prefix of the stop sequence, so it must not be
    // in the normal result text yet.
    assert_eq!(result.text.as_deref(), Some(" delta"));
    assert!(result.stop_trigger.is_none());

    let flushed = decoder.flush_jailed();
    assert_eq!(flushed.as_deref(), Some(" zeta eta"));
    // A second flush has nothing left to give.
    assert_eq!(decoder.flush_jailed(), None);
}
