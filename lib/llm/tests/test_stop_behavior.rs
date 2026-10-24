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

// Single-character tokens used to build a long, self-similar (periodic) run for the
// prefix-matching regression test below.
const A: u32 = 20;
const B: u32 = 21;

// Fragments of a `STOP_HERE`-style marker, used to test stop strings that span tokens at
// several granularities (word pieces, single characters, multi-byte characters).
const ALPHA: u32 = 30; // "alpha"
const SP_STOP: u32 = 31; // " STOP"
const UND_HERE: u32 = 32; // "_HERE"
const BETA: u32 = 33; // " beta"
const ALPHA_SP: u32 = 34; // "alpha "
const ALPHA_STOP: u32 = 35; // "alpha STOP"
const UND_OTHER: u32 = 36; // "_OTHER beta"
const CH_S: u32 = 40;
const CH_T: u32 = 41;
const CH_O: u32 = 42;
const CH_P: u32 = 43;
const CH_UND: u32 = 44;
const CH_H: u32 = 45;
const CH_E: u32 = 46;
const CH_R: u32 = 47;
const HELLO: u32 = 50; // "hello "
const E_ACUTE: u32 = 51; // "é"
const ZHONG: u32 = 52; // "终"
const ZHI_TAIL: u32 = 53; // "止 tail"
// One token whose text contains several complete stop strings at once.
const BIG_STOP_TOKEN: u32 = 60; // "alpha STOP_TOKEN beta"
const BIG_ABCDE: u32 = 61; // "ABCDE tail"

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
                A => Some("a"),
                B => Some("b"),
                ALPHA => Some("alpha"),
                SP_STOP => Some(" STOP"),
                UND_HERE => Some("_HERE"),
                BETA => Some(" beta"),
                ALPHA_SP => Some("alpha "),
                ALPHA_STOP => Some("alpha STOP"),
                UND_OTHER => Some("_OTHER beta"),
                CH_S => Some("S"),
                CH_T => Some("T"),
                CH_O => Some("O"),
                CH_P => Some("P"),
                CH_UND => Some("_"),
                CH_H => Some("H"),
                CH_E => Some("E"),
                CH_R => Some("R"),
                HELLO => Some("hello "),
                E_ACUTE => Some("é"),
                ZHONG => Some("终"),
                ZHI_TAIL => Some("止 tail"),
                BIG_STOP_TOKEN => Some("alpha STOP_TOKEN beta"),
                BIG_ABCDE => Some("ABCDE tail"),
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
    Decoder::new(
        decode_stream,
        stop_conditions,
        include_stop_str,
        false,
        None,
        None,
    )
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
    let mut decoder = make_decoder(None, Some(2), Some(vec![EOS]), None, false);
    let result = decoder.process_token_ids(&[HI, EOS]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi</s>"));
    assert!(result.stop_trigger.is_none());

    let result = decoder.process_token_ids(&[EOS]).unwrap();
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopTokenDetected(id)) if id == EOS
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
    let mut decoder = Decoder::new(decode_stream, stop_conditions, false, false, None, None);
    let result = decoder.process_token_ids(&[HI, STOP]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::UserStopTokenDetected(id)) if id == STOP
    ));
}

/// A hidden stop sequence split across several decode fragments must not leak any
/// fragment before the full sequence is recognized.
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
    // Withholding for `text` must not disturb `tokens[i]`: it always reports token_ids[i]'s
    // own decoded text, matched-hidden-sequence tokens included, so logprob consumers that
    // zip `tokens` with `token_ids` stay aligned.
    assert_eq!(
        result.tokens,
        vec![
            Some(" delta".to_string()),
            Some(" epsilon".to_string()),
            Some(" zeta".to_string()),
            Some(" eta".to_string()),
            Some(" theta".to_string()),
        ]
    );
}

/// A withheld candidate prefix that turns out not to be part of the stop sequence must be
/// released once it can no longer complete, rather than being lost forever.
#[test]
fn withheld_prefix_is_released_once_it_cannot_complete() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["ozzy"]), false);
    let result = decoder.process_token_ids(&[OH, OTHER]).unwrap();

    assert_eq!(result.text.as_deref(), Some("othere"));
    assert!(result.stop_trigger.is_none());
    // Regression check: withholding used to bunch both tokens' text onto the releasing
    // step, pairing (OH, OTHER) with ("", "othere") instead of their own ("o", "there").
    assert_eq!(
        result.tokens,
        vec![Some("o".to_string()), Some("there".to_string())]
    );
}

/// A stop sequence that never completes must still be flushed when generation ends for a
/// reason our decoder did not itself detect (e.g. the engine's own `max_tokens` limit) --
/// otherwise a partial match silently swallows real output forever.
#[test]
fn flush_jailed_releases_incomplete_partial_match() {
    let mut decoder = make_decoder(None, None, None, Some(vec![" zeta eta theta"]), false);
    let result = decoder.process_token_ids(&[DELTA, ZETA, ETA]).unwrap();

    assert_eq!(result.text.as_deref(), Some(" delta"));
    assert!(result.stop_trigger.is_none());

    let flushed = decoder.flush_jailed();
    assert_eq!(flushed.as_deref(), Some(" zeta eta"));
    assert_eq!(decoder.flush_jailed(), None);
}

/// A hidden stop *token* ending the stream is a different stop from any hidden stop
/// *sequence* prefix still being withheld -- that withheld text never completed, so it must
/// still reach the caller even though this particular stop hides its own token's text.
#[test]
fn hidden_stop_token_flushes_prior_jailed_prefix() {
    // "hi" (from HI) is a genuine, never-completed prefix of the hidden sequence "hiya".
    let mut decoder = make_decoder(None, None, Some(vec![EOS]), Some(vec!["hiya"]), false);
    let result = decoder.process_token_ids(&[HI, EOS]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hi"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopTokenDetected(id)) if id == EOS
    ));
    assert_eq!(result.tokens, vec![Some("hi".to_string()), None]);
}

/// Same as above but the terminating stop is a *visible* token: the flushed backlog must
/// come out before this token's own (included) text, not after or in place of it.
#[test]
fn visible_stop_token_flushes_and_orders_prior_jailed_prefix() {
    let tokenizer: Arc<dyn tokenizer_traits::Tokenizer> = Arc::new(TestTokenizer);
    let decode_stream = tokenizers::DecodeStream::new(tokenizer, &[], false);
    let stop_conditions = StopConditions {
        stop_token_ids_visible: Some(vec![STOP]),
        stop: Some(vec!["hiya".to_string()]),
        ..Default::default()
    };
    let mut decoder = Decoder::new(decode_stream, stop_conditions, false, false, None, None);
    let result = decoder.process_token_ids(&[HI, STOP]).unwrap();

    assert_eq!(result.text.as_deref(), Some("hiSTOP"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::VisibleStopTokenDetected(id)) if id == STOP
    ));
    assert_eq!(
        result.tokens,
        vec![Some("hi".to_string()), Some("STOP".to_string())]
    );
}

/// Pins down exactly *when* a withheld candidate prefix is released, using
/// `Decoder::step`'s `released_text` directly rather than the aggregate `text` from
/// `process_token_ids`: the first token's text must stay withheld on its own step (not
/// released early, in case it still completes into the stop sequence) and only come out
/// once the following step proves it cannot.
#[test]
fn released_text_is_withheld_exactly_until_prefix_is_ruled_out() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["ozzy"]), false);

    let first = decoder.step(OH).unwrap();
    assert_eq!(first.token.as_deref(), Some("o"));
    assert_eq!(
        first.released_text, None,
        "\"o\" is still a viable prefix of \"ozzy\" and must not be released yet"
    );
    assert!(first.stop_trigger.is_none());

    let second = decoder.step(OTHER).unwrap();
    assert_eq!(second.token.as_deref(), Some("there"));
    assert_eq!(
        second.released_text.as_deref(),
        Some("othere"),
        "once \"o\" can no longer complete, it must be released together with this step's own text"
    );
    assert!(second.stop_trigger.is_none());
}

/// Regression test for a self-similar (periodic) run of withheld candidate bytes -- the
/// shape of input that made the previous byte-by-byte prefix scan quadratic (each
/// candidate length re-scans from scratch instead of reusing prior work). This asserts
/// correctness, which a unit test can pin down, not the algorithm's complexity, which it
/// cannot: a sliding match window must still land on the right byte offset when an extra
/// repeated character precedes the real match, or either a real stop is missed or content
/// is dropped/leaked.
#[test]
fn hidden_stop_sequence_survives_self_similar_prefix_run() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["aaaab"]), false);
    // "aaaaa" (five 'a's) grows the withheld tail beyond the stop sequence's own prefix
    // length one byte at a time, forcing the matcher to keep re-deriving the longest
    // still-viable prefix length as the window slides, before the final 'b' completes it.
    let result = decoder
        .process_token_ids(&[A, A, A, A, A, B])
        .expect("decode succeeds");

    assert_eq!(
        result.text.as_deref(),
        Some("a"),
        "only the one 'a' that fell out of the sliding window may be released; \
         the rest belongs to the matched stop sequence and must stay hidden"
    );
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "aaaab"
    ));
    assert_eq!(
        result.tokens.len(),
        6,
        "one token report per input token id"
    );
}

#[test]
fn no_stop_trim_preserves_seed_order_and_special_token_policy() {
    for (no_stop_trim, skip_special) in [(false, false), (true, false), (true, true)] {
        let tokenizer: Arc<dyn tokenizer_traits::Tokenizer> = Arc::new(TestTokenizer);
        let mut decoder = Decoder::new(
            tokenizers::DecodeStream::new(tokenizer, &[], skip_special),
            StopConditions {
                stop_token_ids_hidden: Some(vec![EOS]),
                stop: Some(vec!["other".into()]),
                ..Default::default()
            },
            false,
            no_stop_trim,
            None,
            Some("o".into()),
        );
        let result = decoder.process_token_ids(&[EOS, THERE]).unwrap();
        assert_eq!(
            result.text.as_deref(),
            Some(if no_stop_trim && !skip_special {
                "o</s>"
            } else {
                "o"
            })
        );
        assert_eq!(result.tokens.len(), 1);
        assert!(matches!(
            result.stop_trigger,
            Some(StopTrigger::HiddenStopTokenDetected(EOS))
        ));
    }
}

/// A marker split into word pieces (`" STOP"` + `"_HERE"`) must not leak `" STOP"`.
#[test]
fn stop_here_split_into_word_pieces_is_not_leaked() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["STOP_HERE"]), false);
    let result = decoder
        .process_token_ids(&[ALPHA, SP_STOP, UND_HERE, BETA])
        .unwrap();

    assert_eq!(result.text.as_deref(), Some("alpha "));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "STOP_HERE"
    ));
}

/// The worst case for withholding: one character per token.
#[test]
fn stop_here_split_one_character_per_token_is_not_leaked() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["STOP_HERE"]), false);
    let result = decoder
        .process_token_ids(&[
            ALPHA_SP, CH_S, CH_T, CH_O, CH_P, CH_UND, CH_H, CH_E, CH_R, CH_E,
        ])
        .unwrap();

    assert_eq!(result.text.as_deref(), Some("alpha "));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "STOP_HERE"
    ));
}

/// With `include_stop_str_in_output`, a split marker is returned whole, in order, and
/// nothing after it is emitted.
#[test]
fn visible_stop_here_split_across_tokens_is_returned_whole() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["STOP_HERE"]), true);
    let result = decoder
        .process_token_ids(&[ALPHA, SP_STOP, UND_HERE, BETA])
        .unwrap();

    assert_eq!(result.text.as_deref(), Some("alpha STOP_HERE"));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::VisibleStopSequenceDetected(ref s)) if s == "STOP_HERE"
    ));
}

/// A candidate prefix that is followed by non-matching text is released unchanged.
#[test]
fn split_stop_prefix_followed_by_other_text_is_released() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["STOP_HERE"]), false);
    let result = decoder.process_token_ids(&[ALPHA_STOP, UND_OTHER]).unwrap();

    assert_eq!(result.text.as_deref(), Some("alpha STOP_OTHER beta"));
    assert!(result.stop_trigger.is_none());
}

/// Withholding must cut on character boundaries when a multi-byte stop string is split
/// across tokens and a shorter overlapping stop string is also configured.
#[test]
fn multibyte_stop_split_across_tokens_with_overlapping_stop() {
    let mut decoder = make_decoder(None, None, None, Some(vec!["é终止", "终止"]), false);
    let result = decoder
        .process_token_ids(&[HELLO, E_ACUTE, ZHONG, ZHI_TAIL])
        .unwrap();

    assert_eq!(result.text.as_deref(), Some("hello "));
    assert!(matches!(
        result.stop_trigger,
        Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "é终止"
    ));
}

/// When one decoded token contains several complete stop strings, the first one in the
/// request's `stop` list is reported. This matches SGLang; vLLM instead selects the stop
/// that completes earliest in the text, so the two engines differ here. Pinning the
/// current order keeps a change in either direction deliberate.
#[test]
fn several_stops_in_one_token_follow_list_order() {
    for (stops, expected_text, expected_stop) in [
        (vec!["STOP_TOKEN", "STOP"], "alpha ", "STOP_TOKEN"),
        (vec!["STOP", "STOP_TOKEN"], "alpha ", "STOP"),
    ] {
        let mut decoder = make_decoder(None, None, None, Some(stops.clone()), false);
        let result = decoder.process_token_ids(&[BIG_STOP_TOKEN]).unwrap();
        assert_eq!(result.text.as_deref(), Some(expected_text), "{stops:?}");
        assert!(
            matches!(
                result.stop_trigger,
                Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == expected_stop
            ),
            "{stops:?}"
        );
    }

    // The stops complete at different offsets: list order still wins.
    for (stops, expected_text, expected_stop) in [
        (vec!["ABCDE", "BC"], "", "ABCDE"),
        (vec!["BC", "ABCDE"], "A", "BC"),
    ] {
        let mut decoder = make_decoder(None, None, None, Some(stops.clone()), false);
        let result = decoder.process_token_ids(&[BIG_ABCDE]).unwrap();
        assert_eq!(
            result.text.as_deref().unwrap_or(""),
            expected_text,
            "{stops:?}"
        );
        assert!(
            matches!(
                result.stop_trigger,
                Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == expected_stop
            ),
            "{stops:?}"
        );
    }
}
