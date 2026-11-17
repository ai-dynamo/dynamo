// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend-local interpretation of an engine's final-answer grammar.

use std::ops::Range;

use serde::{Deserialize, Serialize};

pub(crate) const GUIDED_OUTPUT_REASONING_RUNTIME_KEY: &str = "guided_output_reasoning";
pub(crate) const GUIDED_OUTPUT_COHORT_CONFLICT_RUNTIME_KEY: &str = "guided_output_cohort_conflict";

/// Published only after the engine and its tokenizer have initialized.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct GuidedOutputCapability {
    pub version: u32,
    pub engine: String,
    pub mode: GuidedReasoningMode,
    #[serde(default)]
    pub request_controls: bool,
    #[serde(default)]
    pub reasoner: Option<String>,
    #[serde(default)]
    pub start_token_ids: Vec<u32>,
    #[serde(default)]
    pub end_token_ids: Vec<u32>,
    #[serde(default)]
    pub start_text: String,
    #[serde(default)]
    pub end_text: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GuidedReasoningMode {
    Immediate,
    ExplicitGate,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GuidedOutputPlan {
    Direct,
    EngineGated(GuidedOutputCapability),
    /// Includes combined tool/answer tags. Their parser still owns the envelope.
    NativeJson,
}

impl GuidedOutputPlan {
    pub(crate) fn projects_answer(&self) -> bool {
        !matches!(self, Self::NativeJson)
    }
}

/// Never crosses the worker or client wire.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuidedOutputDelta {
    pub content: String,
    pub reasoning: String,
    pub reasoning_tokens: u32,
}

/// One choice's boundary matcher and released-text accounting.
///
/// The stop jail releases a prefix of the raw decoded text. Absolute offsets retain
/// its original channel even when a release occurs after the engine boundary.
#[derive(Clone, Debug)]
pub(crate) struct GuidedOutputState {
    plan: GuidedOutputPlan,
    raw_offset: usize,
    released_offset: usize,
    pending: String,
    tail: String,
    end_candidates: Vec<(u32, usize)>,
    start_candidates: Vec<u32>,
    inspect_start: bool,
    start_range: Option<Range<usize>>,
    end_range: Option<Range<usize>>,
    in_answer: bool,
    delta: GuidedOutputDelta,
}

impl GuidedOutputState {
    pub(crate) fn new(plan: GuidedOutputPlan) -> Self {
        let direct = matches!(plan, GuidedOutputPlan::Direct);
        Self {
            plan,
            raw_offset: 0,
            released_offset: 0,
            pending: String::new(),
            tail: String::new(),
            end_candidates: Vec::new(),
            start_candidates: Vec::new(),
            inspect_start: !direct,
            start_range: None,
            end_range: None,
            in_answer: direct,
            delta: GuidedOutputDelta::default(),
        }
    }

    /// A control token must not release a byte-fallback suffix from the prior phase.
    pub(crate) fn flush_before_token(&self, id: u32) -> bool {
        let GuidedOutputPlan::EngineGated(capability) = &self.plan else {
            return false;
        };
        !self.in_answer && capability.end_token_ids.first() == Some(&id)
    }

    pub(crate) fn count_hidden_token(&mut self) {
        if !self.in_answer {
            self.delta.reasoning_tokens += 1;
        }
    }

    pub(crate) fn observe_text(&mut self, text: &str) {
        self.raw_offset += text.len();
        let GuidedOutputPlan::EngineGated(capability) = &self.plan else {
            return;
        };
        let keep = capability.end_text.len().max(capability.start_text.len());
        self.tail.push_str(text);
        let mut trim = self.tail.len().saturating_sub(keep);
        while !self.tail.is_char_boundary(trim) {
            trim += 1;
        }
        self.tail.drain(..trim);
    }

    pub(crate) fn observe_token(&mut self, id: u32, text: &str) -> anyhow::Result<()> {
        self.count_hidden_token();
        let before = self.raw_offset;
        self.observe_text(text);
        if self.in_answer {
            return Ok(());
        }
        let GuidedOutputPlan::EngineGated(capability) = &self.plan else {
            return Ok(());
        };

        if self.inspect_start {
            self.start_candidates.push(id);
            if capability.start_token_ids.is_empty()
                || !capability
                    .start_token_ids
                    .starts_with(&self.start_candidates)
            {
                self.inspect_start = false;
            } else if self.start_candidates == capability.start_token_ids {
                anyhow::ensure!(
                    self.tail.ends_with(&capability.start_text),
                    "guided reasoning opener did not decode as advertised"
                );
                self.start_range = Some(
                    self.raw_offset.saturating_sub(capability.start_text.len())..self.raw_offset,
                );
                self.inspect_start = false;
            }
        }

        self.end_candidates.push((id, before));
        while !self.end_candidates.is_empty()
            && !capability
                .end_token_ids
                .iter()
                .copied()
                .zip(self.end_candidates.iter().map(|(id, _)| *id))
                .all(|(expected, actual)| expected == actual)
        {
            self.end_candidates.remove(0);
        }
        if self.end_candidates.len() == capability.end_token_ids.len() {
            anyhow::ensure!(
                self.tail.ends_with(&capability.end_text),
                "guided reasoning boundary did not decode as advertised"
            );
            self.end_range =
                Some(self.raw_offset.saturating_sub(capability.end_text.len())..self.raw_offset);
            self.end_candidates.clear();
            self.in_answer = true;
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, text: &str) {
        self.pending.push_str(text);
        self.drain(false);
    }

    pub(crate) fn finish(&mut self) {
        self.inspect_start = false;
        self.end_candidates.clear();
        self.drain(true);
    }

    fn drain(&mut self, finishing: bool) {
        let safe = if finishing || self.in_answer {
            self.raw_offset
        } else if self.inspect_start {
            0
        } else {
            self.end_candidates
                .first()
                .map_or(self.raw_offset, |(_, offset)| *offset)
        };
        let available = safe
            .saturating_sub(self.released_offset)
            .min(self.pending.len());
        debug_assert!(self.pending.is_char_boundary(available));
        let end = self.released_offset + available;
        let mut offset = self.released_offset;
        while offset < end {
            let control = [&self.start_range, &self.end_range]
                .into_iter()
                .flatten()
                .find(|range| range.contains(&offset));
            if let Some(range) = control {
                offset = range.end.min(end);
                continue;
            }
            let next = [&self.start_range, &self.end_range]
                .into_iter()
                .flatten()
                .filter(|range| range.start > offset)
                .map(|range| range.start)
                .min()
                .unwrap_or(end)
                .min(end);
            let text = &self.pending[offset - self.released_offset..next - self.released_offset];
            if matches!(self.plan, GuidedOutputPlan::Direct)
                || self
                    .end_range
                    .as_ref()
                    .is_some_and(|range| offset >= range.end)
            {
                self.delta.content.push_str(text);
            } else {
                self.delta.reasoning.push_str(text);
            }
            offset = next;
        }
        self.pending.drain(..available);
        self.released_offset = end;
    }

    pub(crate) fn take_delta(&mut self) -> GuidedOutputDelta {
        std::mem::take(&mut self.delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability() -> GuidedOutputCapability {
        GuidedOutputCapability {
            version: 1,
            engine: "test".into(),
            mode: GuidedReasoningMode::ExplicitGate,
            request_controls: true,
            reasoner: Some("step3".into()),
            start_token_ids: vec![],
            end_token_ids: vec![7, 8],
            start_text: String::new(),
            end_text: "</think>".into(),
        }
    }

    #[test]
    fn token_boundary_and_delayed_stop_jail_release_preserve_channels() {
        let mut state = GuidedOutputState::new(GuidedOutputPlan::EngineGated(capability()));
        state.observe_token(1, "Thought </think> literal.").unwrap();
        state.release("Thought ");
        assert_eq!(state.take_delta().reasoning, "Thought ");
        state.observe_token(7, "</thi").unwrap();
        state.release("</think> literal.</thi");
        assert_eq!(state.take_delta().reasoning, "</think> literal.");
        state.observe_token(8, "nk>").unwrap();
        state.observe_token(9, "<think>yes</think>").unwrap();
        state.release("nk><think>yes</think>");
        let delta = state.take_delta();
        assert_eq!(delta.content, "<think>yes</think>");
        assert_eq!(delta.reasoning, "");
        assert_eq!(delta.reasoning_tokens, 1);
    }

    #[test]
    fn incomplete_terminator_stays_reasoning_at_eof() {
        let mut state = GuidedOutputState::new(GuidedOutputPlan::EngineGated(capability()));
        state.observe_token(7, "</thi").unwrap();
        state.release("</thi");
        assert_eq!(state.take_delta().reasoning, "");
        state.finish();
        let delta = state.take_delta();
        assert_eq!(delta.reasoning, "</thi");
        assert_eq!(delta.content, "");
    }

    #[test]
    fn direct_answer_is_opaque_and_has_no_reasoning_tokens() {
        let mut state = GuidedOutputState::new(GuidedOutputPlan::Direct);
        state.observe_token(7, "<think>30</think>").unwrap();
        state.release("<think>30</think>");
        assert_eq!(
            state.take_delta(),
            GuidedOutputDelta {
                content: "<think>30</think>".into(),
                ..Default::default()
            }
        );
    }
}
