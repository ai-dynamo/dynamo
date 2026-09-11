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

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(result.stop_trigger.is_none());
}

#[test]
fn min_tokens_stops_at_eos_once_floor_is_reached() {
    let mut decoder = make_decoder(None, Some(2), Some(vec![EOS]), None, false);
    let result = decoder.process_token_ids(&[HI, THERE, EOS]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hithere"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopTokenDetected(id)) if id == EOS
    ));
}

/// The floor is a floor on the tokens the caller receives. An end-of-sequence
/// token arriving on the boundary is not one of them, so with a floor of two it
/// neither stops the sequence nor counts toward the two.
#[test]
fn min_tokens_does_not_count_a_suppressed_eos() {
    let mut decoder = make_decoder(None, Some(2), Some(vec![EOS]), None, false);
    let result = decoder.process_token_ids(&[HI, EOS]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(result.stop_trigger.is_none());

    let result = decoder.process_token_ids(&[THERE, EOS]).unwrap();
    assert_eq!(result.text.as_deref(), Some("there"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopTokenDetected(id)) if id == EOS
    ));
}

/// A suppressed token still occupies its slot in `tokens`, so the arrays the
/// backend indexes alongside it stay aligned with the token ids it was given.
#[test]
fn min_tokens_keeps_suppressed_eos_aligned() {
    let mut decoder = make_decoder(None, Some(3), Some(vec![EOS]), None, false);
    let result = decoder.process_token_ids(&[HI, EOS, THERE]).unwrap();

    assert_eq!(result.tokens.len(), 3);
    assert_eq!(result.tokens[1], None);
    assert_eq!(result.text.as_deref(), Some("hithere"));
}

/// A visible stop token is part of the output, so below the floor it is emitted
/// and counted like any other token — only the stop itself waits.
#[test]
fn min_tokens_keeps_visible_stop_tokens_below_the_floor() {
    let tokenizer: Arc<dyn tokenizer_traits::Tokenizer> = Arc::new(TestTokenizer);
    let decode_stream = tokenizers::DecodeStream::new(tokenizer, &[], false);
    let stop_conditions = StopConditions {
        min_tokens: Some(3),
        stop_token_ids_visible: Some(vec![STOP]),
        ..Default::default()
    };
    let mut decoder = Decoder::new(decode_stream, stop_conditions, false, None);

    let result = decoder.process_token_ids(&[HI, STOP]).unwrap();
    assert_eq!(result.text.as_deref(), Some("hiSTOP"));
    assert!(result.stop_trigger.is_none());

    let result = decoder.process_token_ids(&[THERE, STOP]).unwrap();
    assert_eq!(result.text.as_deref(), Some("thereSTOP"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::VisibleStopTokenDetected(id)) if id == STOP
    ));
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
