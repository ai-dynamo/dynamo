// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-output delivery contract shared by routing and generate formats.

/// Request annotation asking the prefill router to send prefill's final
/// `engine_data` into the decode stream, in an item with this event name. It
/// arrives before decode's final output.
pub const PREFILL_OUTPUT_ANNOTATION: &str = "prefill_output";
/// With [`PREFILL_OUTPUT_ANNOTATION`]: hold decode output until it arrives.
pub const PREFILL_OUTPUT_FIRST_ANNOTATION: &str = "prefill_output_first";

/// When a response needs prefill's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillNeed {
    #[default]
    None,
    /// Only the final chunk carries prefill's values.
    BeforeLast,
    /// Every chunk carries prefill's values.
    BeforeFirst,
}

/// The annotations asking the prefill router for prefill's output. Decode
/// output waits for it when every chunk needs it, unary too: a decode failure
/// before a late prefill output would lose it, and a retry's prefill computes
/// prompt-side values over a prompt that now holds output.
pub fn prefill_annotations(need: PrefillNeed) -> Vec<String> {
    match need {
        PrefillNeed::None => vec![],
        PrefillNeed::BeforeLast => vec![PREFILL_OUTPUT_ANNOTATION.to_string()],
        PrefillNeed::BeforeFirst => vec![
            PREFILL_OUTPUT_ANNOTATION.to_string(),
            PREFILL_OUTPUT_FIRST_ANNOTATION.to_string(),
        ],
    }
}
