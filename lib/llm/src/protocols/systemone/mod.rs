// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TypeSafe-compatible System One request and response primitives.

mod native;
mod prompt;
mod score;
mod types;

pub use native::{build_native_score_request, parse_candidate_scores};
pub use prompt::{QuestionPrompt, render_question_prompt};
pub use score::answer_from_logprobs;
pub use types::{
    ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneAnswer, SystemOneError, SystemOneQuestion,
    SystemOneRequest, SystemOneResponse, SystemOneUsage,
};
